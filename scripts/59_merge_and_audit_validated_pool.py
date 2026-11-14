#!/usr/bin/env python3
"""
Merge the previously reconciled 105 candidates with the 65 accepted v2
candidates, then run a read-only global provenance and duplicate audit.

Inputs:
  data/interim/production_filter/final_reconciliation/final_accepted_candidates.jsonl
  data/interim/production_generation_v2/manual_review/dataset_v2_reviewed.jsonl
  data/interim/production_generation_v2/groq_judge/v2_judge_results.jsonl
  data/frozen/chunks_v1_frozen.jsonl

Outputs:
  data/interim/validated_candidate_pool/
    combined_validated_candidates.jsonl
    global_duplicate_candidates.json
    global_pool_audit_errors.json
    global_pool_summary.json

No source dataset, frozen corpus, judge result, or prior reconciliation file
is modified. Near-duplicates are flagged for human review, never auto-removed.

Run from project root:
  python .\scripts\59_merge_and_audit_validated_pool.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

OLD_ACCEPTED = (
    ROOT / "data" / "interim" / "production_filter"
    / "final_reconciliation" / "final_accepted_candidates.jsonl"
)
V2_REVIEWED = (
    ROOT / "data" / "interim" / "production_generation_v2"
    / "manual_review" / "dataset_v2_reviewed.jsonl"
)
V2_JUDGE = (
    ROOT / "data" / "interim" / "production_generation_v2"
    / "groq_judge" / "v2_judge_results.jsonl"
)
FROZEN = ROOT / "data" / "frozen" / "chunks_v1_frozen.jsonl"

OUT_DIR = ROOT / "data" / "interim" / "validated_candidate_pool"
COMBINED_OUT = OUT_DIR / "combined_validated_candidates.jsonl"
DUPLICATES_OUT = OUT_DIR / "global_duplicate_candidates.json"
ERRORS_OUT = OUT_DIR / "global_pool_audit_errors.json"
SUMMARY_OUT = OUT_DIR / "global_pool_summary.json"

BLOCK_PATTERN = re.compile(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", re.DOTALL)
NEAR_DUPLICATE_THRESHOLD = 0.88


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Required input missing: {path}")
    result: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path.name}, line {line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object in {path.name}, line {line_no}")
            result.append(row)
    return result


def normalize(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9_+#./'-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def whitespace_normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def get_chunk_content(row: dict[str, Any]) -> str:
    value = row.get("content")
    if value is None:
        value = row.get("text")
    return str(value or "")


def load_frozen_chunks() -> dict[str, str]:
    chunks: dict[str, str] = {}
    for row in read_jsonl(FROZEN):
        chunk_id = str(row.get("chunk_id") or "")
        if chunk_id:
            chunks[chunk_id] = get_chunk_content(row)
    if not chunks:
        raise ValueError("No frozen chunks loaded.")
    return chunks


def get_blocks(source: str) -> list[str]:
    return [
        match.group(0)
        for match in BLOCK_PATTERN.finditer(source)
        if len(match.group(0).strip()) >= 30
    ]


def get_evidence_ids(row: dict[str, Any]) -> Any:
    # Existing reconciliation output and v2 generator use this field.
    return row.get("evidence_block_ids", row.get("evidence_ids"))


def verify_evidence(row: dict[str, Any], chunks: dict[str, str]) -> tuple[bool, str]:
    chunk_id = str(row.get("chunk_id") or "")
    if chunk_id not in chunks:
        return False, f"chunk_not_found:{chunk_id}"

    ids = get_evidence_ids(row)
    if not isinstance(ids, list) or not ids:
        return False, "missing_evidence_block_ids"

    blocks = get_blocks(chunks[chunk_id])
    selected: list[str] = []
    for raw_id in ids:
        try:
            block_id = int(raw_id)
        except (TypeError, ValueError):
            return False, f"invalid_evidence_block_id:{raw_id!r}"
        if block_id < 1 or block_id > len(blocks):
            return False, f"evidence_block_id_out_of_range:{block_id}/{len(blocks)}"
        selected.append(blocks[block_id - 1])

    reconstructed = "\n\n".join(selected)
    stored = row.get("evidence_span")
    if stored is None:
        return False, "missing_evidence_span"

    expected_norm = whitespace_normalize(reconstructed)
    actual_norm = whitespace_normalize(stored)
    if actual_norm == expected_norm or (expected_norm and expected_norm in actual_norm):
        return True, "ok"
    return False, "evidence_span_mismatch"


def main() -> int:
    old_rows = read_jsonl(OLD_ACCEPTED)
    v2_rows = read_jsonl(V2_REVIEWED)
    judge_rows = read_jsonl(V2_JUDGE)
    chunks = load_frozen_chunks()

    if len(old_rows) != 105:
        raise ValueError(
            f"Expected 105 previously reconciled accepted candidates, found {len(old_rows)}"
        )
    if len(v2_rows) != 66:
        raise ValueError(f"Expected 66 manually reviewed v2 rows, found {len(v2_rows)}")

    judge_by_id: dict[str, dict[str, Any]] = {}
    for result in judge_rows:
        example_id = str(result.get("example_id") or "")
        if example_id:
            judge_by_id[example_id] = result

    accepted_v2: list[dict[str, Any]] = []
    rejected_v2: list[str] = []
    missing_judge: list[str] = []

    for row in v2_rows:
        example_id = str(row.get("example_id") or "")
        judge = judge_by_id.get(example_id)
        if judge is None:
            missing_judge.append(example_id)
            continue
        if judge.get("verdict") == "ACCEPT":
            enriched = dict(row)
            enriched["_pool_source"] = "generation_v2"
            enriched["_judge_model"] = judge.get("judge_model")
            enriched["_judge_faithfulness"] = judge.get("faithfulness")
            enriched["_judge_verdict"] = judge.get("verdict")
            accepted_v2.append(enriched)
        else:
            rejected_v2.append(example_id)

    if missing_judge:
        raise ValueError(
            "Some v2 examples have no judge result; refusing partial merge: "
            + ", ".join(missing_judge)
        )
    if len(judge_by_id) != 66:
        raise ValueError(
            f"Expected 66 v2 judge results, found {len(judge_by_id)}"
        )
    if len(accepted_v2) != 65 or len(rejected_v2) != 1:
        raise ValueError(
            f"Expected 65 accepted and 1 rejected v2 rows; found "
            f"{len(accepted_v2)} accepted and {len(rejected_v2)} rejected"
        )

    combined: list[dict[str, Any]] = []
    for row in old_rows:
        enriched = dict(row)
        enriched["_pool_source"] = "previous_reconciliation"
        combined.append(enriched)
    combined.extend(accepted_v2)

    errors: list[dict[str, Any]] = []
    ids: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    evidence_valid = 0
    evidence_invalid = 0

    for row in combined:
        example_id = str(row.get("example_id") or "")
        ids[example_id] += 1
        task_counts[str(row.get("task_type") or "<unspecified>")] += 1
        source_counts[str(row.get("_pool_source") or "<unknown>")] += 1

        if not example_id:
            errors.append({"issue": "missing_example_id"})
        if not str(row.get("instruction") or "").strip():
            errors.append({"example_id": example_id, "issue": "empty_instruction"})
        if not str(row.get("response") or "").strip():
            errors.append({"example_id": example_id, "issue": "empty_response"})

        ok, reason = verify_evidence(row, chunks)
        if ok:
            evidence_valid += 1
        else:
            evidence_invalid += 1
            errors.append({
                "example_id": example_id,
                "issue": "evidence_provenance_failure",
                "detail": reason,
            })

    duplicate_ids = [
        {"example_id": key, "count": count}
        for key, count in ids.items()
        if key and count > 1
    ]
    for item in duplicate_ids:
        errors.append({"issue": "duplicate_example_id", **item})

    # Exact duplicate instruction groups and near-duplicate pairs across the
    # entire combined pool. No candidate is removed automatically.
    normalized_index: list[dict[str, Any]] = []
    exact_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in combined:
        normalized_instruction = normalize(row.get("instruction"))
        if not normalized_instruction:
            continue
        item = {
            "example_id": row.get("example_id"),
            "chunk_id": row.get("chunk_id"),
            "pool_source": row.get("_pool_source"),
            "task_type": row.get("task_type"),
            "instruction": row.get("instruction"),
            "normalized": normalized_instruction,
        }
        exact_groups[normalized_instruction].append(item)
        normalized_index.append(item)

    exact_duplicates = [
        {
            "instruction_normalized": normalized,
            "records": [
                {k: v for k, v in item.items() if k != "normalized"}
                for item in items
            ],
        }
        for normalized, items in exact_groups.items()
        if len(items) > 1
    ]

    near_duplicates: list[dict[str, Any]] = []
    for i, left in enumerate(normalized_index):
        for right in normalized_index[i + 1:]:
            if left["normalized"] == right["normalized"]:
                continue
            # SequenceMatcher is used as a review flag, not a semantic verdict.
            similarity = SequenceMatcher(
                None, left["normalized"], right["normalized"]
            ).ratio()
            if similarity >= NEAR_DUPLICATE_THRESHOLD:
                near_duplicates.append({
                    "similarity": round(similarity, 4),
                    "left": {k: v for k, v in left.items() if k != "normalized"},
                    "right": {k: v for k, v in right.items() if k != "normalized"},
                })

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with COMBINED_OUT.open("w", encoding="utf-8", newline="\n") as handle:
        for row in combined:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    duplicate_report = {
        "exact_duplicate_instruction_groups": exact_duplicates,
        "near_duplicate_candidate_pairs_threshold_0_88": near_duplicates,
        "note": (
            "Similarity matches are review candidates only. Do not automatically "
            "remove semantically distinct examples."
        ),
    }
    DUPLICATES_OUT.write_text(
        json.dumps(duplicate_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    ERRORS_OUT.write_text(
        json.dumps(errors, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = {
        "previously_reconciled_accepted": len(old_rows),
        "v2_manually_reviewed": len(v2_rows),
        "v2_judge_accepted": len(accepted_v2),
        "v2_judge_rejected": len(rejected_v2),
        "v2_rejected_ids": rejected_v2,
        "combined_candidate_count": len(combined),
        "frozen_chunk_count": len(chunks),
        "task_type_counts": dict(sorted(task_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "evidence_provenance": {
            "valid": evidence_valid,
            "invalid": evidence_invalid,
        },
        "duplicate_example_id_groups": len(duplicate_ids),
        "exact_duplicate_instruction_groups": len(exact_duplicates),
        "near_duplicate_candidate_pairs_threshold_0_88": len(near_duplicates),
        "structural_or_provenance_error_count": len(errors),
        "output": str(COMBINED_OUT.relative_to(ROOT)),
        "source_datasets_modified": False,
        "frozen_corpus_modified": False,
        "automatic_duplicate_removal": False,
    }
    SUMMARY_OUT.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("VALIDATED CANDIDATE POOL MERGE + GLOBAL AUDIT")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nDuplicate report: {DUPLICATES_OUT.relative_to(ROOT)}")
    print(f"Audit errors: {ERRORS_OUT.relative_to(ROOT)}")
    print("No source datasets, judge results, or frozen chunks were modified.")

    if errors:
        print("\nSTATUS: AUDIT FAILED — resolve provenance/structure errors first.")
        return 1
    if exact_duplicates or near_duplicates:
        print("\nSTATUS: PROVENANCE PASSED — duplicate candidates need review.")
        return 0
    print("\nSTATUS: PROVENANCE PASSED — no duplicate candidates detected.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nMERGE/AUDIT FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise