#!/usr/bin/env python3
"""
Read-only audit for Production Generation v2.

Checks:
- JSONL structure and unique IDs
- chunk references against the frozen corpus
- evidence block IDs and reconstructed evidence spans
- task-type / chunk distributions
- exact and near-duplicate instructions against the existing 115 examples
- a few source-specific review flags

This script does NOT modify any dataset or checkpoint.
Run from the project root:
    python .\scripts\56_audit_generation_v2.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
V2_PATH = ROOT / "data" / "interim" / "production_generation_v2" / "dataset_v0_additional_v2.jsonl"
RAW_PATH = ROOT / "data" / "interim" / "dataset_v0_raw.jsonl"
PRIOR_ADDITIONAL_PATH = ROOT / "data" / "interim" / "improved_generation" / "dataset_v0_additional.jsonl"
FROZEN_PATH = ROOT / "data" / "frozen" / "chunks_v1_frozen.jsonl"
OUT_DIR = ROOT / "data" / "interim" / "production_generation_v2" / "audit"

REQUIRED_FIELDS = (
    "example_id", "chunk_id", "instruction", "response",
    "evidence_block_ids", "evidence_span", "task_type",
)
VALID_TASKS = {"qa", "extraction", "classification"}
BLOCK_PATTERN = re.compile(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", re.DOTALL)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path.name}, line {line_no}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object in {path.name}, line {line_no}")
            rows.append(value)
    return rows


def normalize_text(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9_+#./'-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def whitespace_normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def evidence_blocks(source: str) -> list[str]:
    blocks: list[str] = []
    for match in BLOCK_PATTERN.finditer(source or ""):
        block = match.group(0)
        if len(block.strip()) >= 30:
            blocks.append(block)
    return blocks


def load_chunks() -> dict[str, str]:
    rows = read_jsonl(FROZEN_PATH)
    mapping: dict[str, str] = {}
    for row in rows:
        chunk_id = row.get("chunk_id")
        content = row.get("content")
        if content is None:
            content = row.get("text")
        if chunk_id and content is not None:
            mapping[str(chunk_id)] = str(content)
    if not mapping:
        raise ValueError(
            "No frozen chunks loaded. Expected each row to have chunk_id and content."
        )
    return mapping


def load_prior_candidates() -> list[dict[str, Any]]:
    prior: list[dict[str, Any]] = []
    for path in (RAW_PATH, PRIOR_ADDITIONAL_PATH):
        if path.exists():
            prior.extend(read_jsonl(path))
    return prior


def make_record_label(row: dict[str, Any], source: str) -> str:
    return f"{source}:{row.get('example_id', '<missing-id>')}"


def main() -> int:
    for required in (V2_PATH, FROZEN_PATH):
        if not required.exists():
            print(f"ERROR: Missing required file: {required}")
            return 2

    v2_rows = read_jsonl(V2_PATH)
    prior_rows = load_prior_candidates()
    chunks = load_chunks()

    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    v2_ids: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    chunk_counts: Counter[str] = Counter()
    evidence_valid = 0
    evidence_invalid = 0

    # Build comparable question/instruction index across previous 115 + current v2.
    all_records: list[tuple[dict[str, Any], str]] = [
        (row, "existing") for row in prior_rows
    ] + [(row, "v2") for row in v2_rows]

    duplicate_id_rows: list[dict[str, Any]] = []
    seen_v2_ids: dict[str, int] = {}

    for index, row in enumerate(v2_rows, 1):
        for field in REQUIRED_FIELDS:
            if field not in row or row.get(field) is None:
                errors.append({
                    "example_id": row.get("example_id"),
                    "issue": "missing_required_field",
                    "field": field,
                })

        example_id = str(row.get("example_id") or "")
        if example_id:
            v2_ids[example_id] += 1
            if example_id in seen_v2_ids:
                duplicate_id_rows.append({
                    "example_id": example_id,
                    "first_record": seen_v2_ids[example_id],
                    "duplicate_record": index,
                })
            else:
                seen_v2_ids[example_id] = index

        task = str(row.get("task_type") or "").lower()
        task_counts[task or "<missing>"] += 1
        if task not in VALID_TASKS:
            errors.append({
                "example_id": example_id,
                "issue": "invalid_task_type",
                "task_type": task,
            })

        chunk_id = str(row.get("chunk_id") or "")
        chunk_counts[chunk_id or "<missing>"] += 1
        source = chunks.get(chunk_id)
        if source is None:
            errors.append({
                "example_id": example_id,
                "issue": "chunk_not_found_in_frozen_corpus",
                "chunk_id": chunk_id,
            })
            continue

        raw_ids = row.get("evidence_block_ids")
        if not isinstance(raw_ids, list) or not raw_ids:
            errors.append({
                "example_id": example_id,
                "issue": "missing_or_invalid_evidence_block_ids",
                "evidence_block_ids": raw_ids,
            })
            evidence_invalid += 1
            continue

        blocks = evidence_blocks(source)
        selected: list[str] = []
        invalid_ids: list[Any] = []
        for block_id in raw_ids:
            try:
                numeric_id = int(block_id)
            except (TypeError, ValueError):
                invalid_ids.append(block_id)
                continue
            if numeric_id < 1 or numeric_id > len(blocks):
                invalid_ids.append(block_id)
            else:
                selected.append(blocks[numeric_id - 1])

        if invalid_ids or len(selected) != len(raw_ids):
            errors.append({
                "example_id": example_id,
                "issue": "evidence_block_id_out_of_range_or_invalid",
                "invalid_ids": invalid_ids,
                "available_block_count": len(blocks),
            })
            evidence_invalid += 1
            continue

        reconstructed = "\n\n".join(selected)
        expected_norm = whitespace_normalize(reconstructed)
        actual_norm = whitespace_normalize(row.get("evidence_span"))
        # Allow a small source/page label wrapper, but require the full selected
        # evidence text to be present unchanged after whitespace normalization.
        evidence_matches = (
            actual_norm == expected_norm
            or (expected_norm and expected_norm in actual_norm)
        )
        if evidence_matches:
            evidence_valid += 1
        else:
            evidence_invalid += 1
            errors.append({
                "example_id": example_id,
                "issue": "evidence_span_reconstruction_mismatch",
                "expected_preview": reconstructed[:500],
                "actual_preview": str(row.get("evidence_span") or "")[:500],
            })

        instruction = normalize_text(row.get("instruction"))
        response = normalize_text(row.get("response"))
        if not instruction or not response:
            errors.append({
                "example_id": example_id,
                "issue": "empty_instruction_or_response",
            })

        # Review flags; these are not automatic rejections.
        combined = f"{row.get('instruction', '')}\n{row.get('response', '')}\n{row.get('evidence_span', '')}".lower()
        flags: list[str] = []
        if "dht11" in combined and ("raspberry pi" in combined or "gpio" in combined):
            flags.append("HARDWARE_WIRING_SAFETY_REVIEW")
        if "syncdb" in combined:
            flags.append("LEGACY_DJANGO_COMMAND_CONTEXT_REVIEW")
        if "_init_.py" in combined or "__init__.py" in combined:
            flags.append("SOURCE_TRANSCRIPTION_OR_OCR_REVIEW")
        if "fortune 500" in combined or "fortune 500" in str(row.get("response", "")).lower():
            flags.append("SOURCE_STATISTIC_CONTEXT_REVIEW")
        if flags:
            warnings.append({
                "example_id": example_id,
                "chunk_id": chunk_id,
                "flags": flags,
            })

    if duplicate_id_rows:
        for item in duplicate_id_rows:
            errors.append({"issue": "duplicate_example_id", **item})

    # Exact and near duplicate instruction scan across all 183 candidates.
    exact_groups: dict[str, list[str]] = defaultdict(list)
    indexed: list[dict[str, Any]] = []
    for row, source_name in all_records:
        q = normalize_text(row.get("instruction"))
        if not q:
            continue
        label = make_record_label(row, source_name)
        exact_groups[q].append(label)
        indexed.append({
            "label": label,
            "source": source_name,
            "example_id": row.get("example_id"),
            "chunk_id": row.get("chunk_id"),
            "task_type": row.get("task_type"),
            "instruction": str(row.get("instruction") or ""),
            "normalized": q,
        })

    exact_duplicates = [
        {"instruction_normalized": key, "records": labels}
        for key, labels in exact_groups.items()
        if len(labels) > 1
    ]

    near_duplicates: list[dict[str, Any]] = []
    # Only compare pairs where at least one side is v2; old-old pairs were
    # already reviewed in the earlier production filter.
    for i, left in enumerate(indexed):
        for right in indexed[i + 1:]:
            if left["source"] != "v2" and right["source"] != "v2":
                continue
            if left["normalized"] == right["normalized"]:
                continue
            similarity = SequenceMatcher(
                None, left["normalized"], right["normalized"]
            ).ratio()
            if similarity >= 0.88:
                near_duplicates.append({
                    "similarity": round(similarity, 4),
                    "left": {k: left[k] for k in (
                        "label", "chunk_id", "task_type", "instruction"
                    )},
                    "right": {k: right[k] for k in (
                        "label", "chunk_id", "task_type", "instruction"
                    )},
                })

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "v2_example_count": len(v2_rows),
        "existing_candidate_count": len(prior_rows),
        "combined_candidate_count": len(v2_rows) + len(prior_rows),
        "frozen_chunk_count": len(chunks),
        "task_type_counts_v2": dict(sorted(task_counts.items())),
        "chunk_counts_v2": dict(sorted(chunk_counts.items())),
        "evidence_reconstruction": {
            "valid": evidence_valid,
            "invalid": evidence_invalid,
        },
        "duplicate_example_id_count_v2": len(duplicate_id_rows),
        "exact_duplicate_instruction_groups_combined": len(exact_duplicates),
        "near_duplicate_candidate_pairs_combined": len(near_duplicates),
        "manual_review_flagged_examples": len(warnings),
        "structural_or_provenance_error_count": len(errors),
        "read_only": True,
        "source_datasets_modified": False,
    }

    (OUT_DIR / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUT_DIR / "audit_errors.json").write_text(
        json.dumps(errors, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUT_DIR / "exact_duplicate_groups.json").write_text(
        json.dumps(exact_duplicates, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUT_DIR / "near_duplicate_candidates.json").write_text(
        json.dumps(near_duplicates, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUT_DIR / "manual_review_flags.json").write_text(
        json.dumps(warnings, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\nPRODUCTION GENERATION V2 AUDIT")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nReports written to: {OUT_DIR.relative_to(ROOT)}")
    print("No datasets, checkpoints, or frozen source files were modified.")

    if errors:
        print("\nAUDIT STATUS: REVIEW REQUIRED")
        return 1
    print("\nAUDIT STATUS: STRUCTURE AND EVIDENCE PROVENANCE PASSED")
    print("Exact/near duplicate groups and manual review flags still require review.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nAUDIT FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise