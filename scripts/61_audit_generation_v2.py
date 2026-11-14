#!/usr/bin/env python3
"""
Read-only comprehensive audit for the expanded IoT generation-v2 dataset.

Reads:
  data/frozen/chunks_v1_frozen.jsonl
  data/interim/dataset_v0_raw.jsonl
  data/interim/improved_generation/dataset_v0_additional.jsonl
  data/interim/production_generation_v2/dataset_v0_additional_v2_resumed.jsonl

Writes reports only to:
  data/interim/production_generation_v2/audit_v1/

No source dataset, checkpoint, or frozen corpus is modified.
Near-duplicate pairs are flagged for review; they are never automatically removed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FROZEN_FILE = ROOT / "data" / "frozen" / "chunks_v1_frozen.jsonl"
ORIGINAL_RAW = ROOT / "data" / "interim" / "dataset_v0_raw.jsonl"
PRIOR_ADDITIONAL = ROOT / "data" / "interim" / "improved_generation" / "dataset_v0_additional.jsonl"
V2_FILE = ROOT / "data" / "interim" / "production_generation_v2" / "dataset_v0_additional_v2_resumed.jsonl"
AUDIT_DIR = ROOT / "data" / "interim" / "production_generation_v2" / "audit_v1"

TASK_TYPES = {"qa", "extraction", "classification"}
NEAR_DUPLICATE_THRESHOLD = 0.88


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON: {path} line {line_no}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object: {path} line {line_no}")
            rows.append(value)
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def first_value(row: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        value = row.get(name)
        if value is not None:
            return value
    return None


def chunk_id(row: dict[str, Any]) -> str | None:
    value = first_value(row, ("chunk_id", "id", "chunkId"))
    return str(value) if value is not None else None


def chunk_content(row: dict[str, Any]) -> str | None:
    value = first_value(row, ("content", "text", "chunk_text", "page_content"))
    return value if isinstance(value, str) else None


def evidence_blocks(source: str) -> list[dict[str, Any]]:
    blocks = []
    pattern = re.compile(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", re.DOTALL)
    for match in pattern.finditer(source):
        block = match.group(0)
        if len(block.strip()) >= 30:
            blocks.append({"id": len(blocks) + 1, "text": block})
    return blocks


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def similarity(a: str, b: str) -> float:
    na, nb = normalize(a), normalize(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    seq = SequenceMatcher(None, na, nb).ratio()
    ta, tb = set(na.split()), set(nb.split())
    union = ta | tb
    jac = len(ta & tb) / len(union) if union else 0.0
    return max(seq, jac)


def main() -> int:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    frozen = read_jsonl(FROZEN_FILE)
    raw = read_jsonl(ORIGINAL_RAW)
    prior = read_jsonl(PRIOR_ADDITIONAL)
    v2 = read_jsonl(V2_FILE)

    frozen_by_id: dict[str, str] = {}
    duplicate_chunk_ids: list[str] = []
    for row in frozen:
        cid = chunk_id(row)
        content = chunk_content(row)
        if not cid or not content:
            continue
        if cid in frozen_by_id:
            duplicate_chunk_ids.append(cid)
        frozen_by_id[cid] = content

    # The full pool is used only for duplicate comparison; no rows are changed.
    all_rows: list[tuple[str, dict[str, Any]]] = (
        [("original_raw", row) for row in raw]
        + [("prior_additional", row) for row in prior]
        + [("v2", row) for row in v2]
    )

    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    v2_ids = [str(row.get("example_id", "")).strip() for row in v2]
    id_counts = Counter(x for x in v2_ids if x)
    for example_id, count in id_counts.items():
        if count > 1:
            errors.append({
                "type": "duplicate_v2_example_id",
                "example_id": example_id,
                "count": count,
            })

    task_counts = Counter()
    doc_counts = Counter()
    chunk_counts = Counter()
    provenance_ok = 0
    task_missing_or_invalid = 0
    missing_fields = 0
    empty_evidence_ids = 0
    evidence_reconstruction_failures = 0
    evidence_id_failures = 0

    for index, row in enumerate(v2, 1):
        eid = str(row.get("example_id", "")).strip() or f"<missing-id-row-{index}>"
        required = ("example_id", "chunk_id", "instruction", "response", "evidence_block_ids", "evidence_span")
        missing = [field for field in required if row.get(field) is None or row.get(field) == ""]
        if missing:
            missing_fields += 1
            errors.append({"type": "missing_required_fields", "example_id": eid, "fields": missing})

        task = str(row.get("task_type", "")).strip().lower()
        if task not in TASK_TYPES:
            task_missing_or_invalid += 1
            errors.append({"type": "invalid_task_type", "example_id": eid, "task_type": task})
        else:
            task_counts[task] += 1

        cid = str(row.get("chunk_id", "")).strip()
        chunk_counts[cid or "<missing>"] += 1
        doc_counts[str(row.get("source_doc", row.get("document_id", "<not-stored>")))] += 1

        source = frozen_by_id.get(cid)
        if source is None:
            errors.append({"type": "chunk_not_found_in_frozen_corpus", "example_id": eid, "chunk_id": cid})
            continue

        ids = row.get("evidence_block_ids")
        if not isinstance(ids, list) or not ids:
            empty_evidence_ids += 1
            errors.append({"type": "missing_or_invalid_evidence_block_ids", "example_id": eid})
            continue

        blocks = evidence_blocks(source)
        lookup = {b["id"]: b["text"] for b in blocks}
        try:
            selected_ids = list(dict.fromkeys(int(value) for value in ids))
        except (TypeError, ValueError):
            selected_ids = []
        if not selected_ids or any(value not in lookup for value in selected_ids):
            evidence_id_failures += 1
            errors.append({
                "type": "invalid_evidence_block_ids",
                "example_id": eid,
                "provided_ids": ids,
                "available_block_ids": sorted(lookup),
            })
            continue

        reconstructed = "\n\n".join(lookup[value] for value in selected_ids)
        if reconstructed != row.get("evidence_span"):
            evidence_reconstruction_failures += 1
            errors.append({
                "type": "evidence_span_mismatch",
                "example_id": eid,
                "chunk_id": cid,
            })
            continue

        if reconstructed not in source:
            evidence_reconstruction_failures += 1
            errors.append({
                "type": "evidence_not_in_source",
                "example_id": eid,
                "chunk_id": cid,
            })
            continue
        provenance_ok += 1

        if not str(row.get("instruction", "")).strip() or not str(row.get("response", "")).strip():
            warnings.append({"type": "blank_instruction_or_response", "example_id": eid})

    # Pairwise duplicate scan across the 366-example candidate pool.
    # Report only pairs where at least one member belongs to v2.
    duplicate_pairs: list[dict[str, Any]] = []
    exact_seen: dict[str, tuple[str, str]] = {}
    normalized_rows = []
    for source_name, row in all_rows:
        instruction = str(row.get("instruction", "")).strip()
        key = normalize(instruction)
        if not key:
            continue
        eid = str(row.get("example_id", "<missing-id>"))
        if key in exact_seen:
            prior_source, prior_id = exact_seen[key]
            if source_name == "v2" or prior_source == "v2":
                duplicate_pairs.append({
                    "kind": "exact_normalized",
                    "similarity": 1.0,
                    "source_a": prior_source,
                    "example_id_a": prior_id,
                    "source_b": source_name,
                    "example_id_b": eid,
                    "instruction_a": next((str(r.get("instruction", "")) for s, r in all_rows if s == prior_source and str(r.get("example_id", "")) == prior_id), ""),
                    "instruction_b": instruction,
                })
        else:
            exact_seen[key] = (source_name, eid)
        normalized_rows.append((source_name, eid, instruction))

    # O(n^2) is practical for this project-sized pool.
    for i in range(len(normalized_rows)):
        src_a, id_a, text_a = normalized_rows[i]
        for j in range(i + 1, len(normalized_rows)):
            src_b, id_b, text_b = normalized_rows[j]
            if src_a != "v2" and src_b != "v2":
                continue
            score = similarity(text_a, text_b)
            if score >= NEAR_DUPLICATE_THRESHOLD and normalize(text_a) != normalize(text_b):
                duplicate_pairs.append({
                    "kind": "near_duplicate",
                    "similarity": round(score, 4),
                    "source_a": src_a,
                    "example_id_a": id_a,
                    "source_b": src_b,
                    "example_id_b": id_b,
                    "instruction_a": text_a,
                    "instruction_b": text_b,
                })

    # Ensure exact duplicate pairs are not repeated as near duplicates.
    duplicate_pairs.sort(key=lambda x: (-x["similarity"], x["kind"], x["example_id_a"], x["example_id_b"]))

    summary = {
        "audit_timestamp_local": __import__("datetime").datetime.now().astimezone().isoformat(),
        "inputs": {
            "frozen_corpus": str(FROZEN_FILE.relative_to(ROOT)),
            "frozen_corpus_sha256": sha256(FROZEN_FILE),
            "original_raw_count": len(raw),
            "prior_additional_count": len(prior),
            "v2_count": len(v2),
            "combined_candidate_count": len(raw) + len(prior) + len(v2),
        },
        "frozen_corpus": {
            "chunk_count": len(frozen),
            "unique_chunk_ids": len(frozen_by_id),
            "duplicate_chunk_ids": duplicate_chunk_ids,
        },
        "v2_dataset": {
            "unique_example_ids": len(id_counts),
            "duplicate_example_id_groups": sum(1 for count in id_counts.values() if count > 1),
            "task_type_distribution": dict(sorted(task_counts.items())),
            "chunk_coverage": len({str(r.get("chunk_id", "")) for r in v2}),
            "examples_per_chunk": dict(sorted(Counter(str(r.get("chunk_id", "")) for r in v2).items())),
            "source_document_distribution": dict(sorted(doc_counts.items())),
        },
        "evidence_audit": {
            "v2_examples_with_verified_exact_evidence": provenance_ok,
            "missing_or_invalid_evidence_id_rows": empty_evidence_ids,
            "invalid_evidence_id_rows": evidence_id_failures,
            "evidence_reconstruction_or_source_failures": evidence_reconstruction_failures,
        },
        "duplicate_scan": {
            "threshold": NEAR_DUPLICATE_THRESHOLD,
            "pairs_flagged": len(duplicate_pairs),
            "exact_normalized_pairs": sum(1 for p in duplicate_pairs if p["kind"] == "exact_normalized"),
            "near_duplicate_pairs": sum(1 for p in duplicate_pairs if p["kind"] == "near_duplicate"),
            "note": "Flags only; no examples were automatically removed.",
        },
        "quality_checks": {
            "missing_required_field_rows": missing_fields,
            "invalid_or_missing_task_type_rows": task_missing_or_invalid,
            "error_count": len(errors),
            "warning_count": len(warnings),
        },
        "status": "PASS_STRUCTURAL_AUDIT" if not errors else "REVIEW_REQUIRED",
    }

    (AUDIT_DIR / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (AUDIT_DIR / "audit_errors.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in errors), encoding="utf-8"
    )
    (AUDIT_DIR / "audit_warnings.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in warnings), encoding="utf-8"
    )

    duplicate_path = AUDIT_DIR / "duplicate_candidates.csv"
    columns = [
        "kind", "similarity", "source_a", "example_id_a", "source_b",
        "example_id_b", "instruction_a", "instruction_b",
    ]
    with duplicate_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(duplicate_pairs)

    print("\nCOMPREHENSIVE V2 AUDIT")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nREPORTS")
    print(f"  Summary:   {AUDIT_DIR / 'audit_summary.json'}")
    print(f"  Errors:    {AUDIT_DIR / 'audit_errors.jsonl'}")
    print(f"  Warnings:  {AUDIT_DIR / 'audit_warnings.jsonl'}")
    print(f"  Duplicates:{duplicate_path}")
    print("\nNo input dataset or checkpoint was modified.")
    return 0 if not errors else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"AUDIT FAILED: {exc}", file=sys.stderr)
        raise SystemExit(2)