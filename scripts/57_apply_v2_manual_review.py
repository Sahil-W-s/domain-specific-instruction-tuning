#!/usr/bin/env python3
"""
Apply the seven manually reviewed decisions to Generation v2.

Reads the original v2 JSONL and writes a separate reviewed candidate file.
The original v2 file, frozen corpus, and all prior datasets remain unchanged.

Run from project root:
    python .\scripts\57_apply_v2_manual_review.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = (
    ROOT / "data" / "interim" / "production_generation_v2"
    / "dataset_v0_additional_v2.jsonl"
)
OUTPUT_DIR = (
    ROOT / "data" / "interim" / "production_generation_v2" / "manual_review"
)
OUTPUT_PATH = OUTPUT_DIR / "dataset_v2_reviewed.jsonl"
DECISIONS_PATH = OUTPUT_DIR / "manual_review_decisions.json"

# Exclusions are limited to the two reviewed examples below.
EXCLUDE_REASONS = {
    "genv2_iot_cloud_case_studies_chunk_0007_p01_extraction":
        "Contains Raspberry Pi 5V/GPIO wiring instructions; electrical compatibility and safe wiring cannot be validated from the text alone.",
    "genv2_iot_cloud_case_studies_chunk_0011_p01_extraction":
        "Near-identical duplicate of the Django project-structure extraction example from chunk 0009.",
}

# Corrections change response wording only. Evidence spans and evidence block IDs
# remain byte-for-byte as they appeared in the input records.
RESPONSE_CORRECTIONS = {
    "genv2_iot_cloud_case_studies_chunk_0009_p01_extraction": (
        "The source lists `manage.py` and a `Kfirst` subdirectory containing "
        "`_init_.py`, `settings.py`, `urls.py`, and `wsgi.py` "
        "(filename spelling preserved as transcribed in the source)."
    ),
    "genv2_iot_cloud_case_studies_chunk_0010_p01_qa": (
        "According to the legacy Django instructions shown in the source, "
        "run `python manage.py syncdb` from the `ks1/Kfirst` directory."
    ),
    "genv2_iot_cloud_case_studies_chunk_0013_p01_classification": (
        "AWS is described as a Global Cloud Platform offering Infrastructure "
        "as a Service, Software as a Service, Platform as a Service, and "
        "Cloud Storage Platform."
    ),
}

DECISION_NOTES = {
    "genv2_iot_cloud_case_studies_chunk_0008_p01_qa":
        "Retain: GPIO pin 23 is directly supported by the script's read_retry call.",
    "genv2_iot_cloud_case_studies_chunk_0008_p01_extraction":
        "Retain: imported modules/libraries are directly listed in the source.",
    "genv2_iot_cloud_case_studies_chunk_0009_p01_extraction":
        "Retain with source-transcription caveat; response preserves the source's literal _init_.py spelling.",
    "genv2_iot_cloud_case_studies_chunk_0010_p01_qa":
        "Retain with explicit historical/source attribution; do not present syncdb as current Django guidance.",
    "genv2_iot_cloud_case_studies_chunk_0013_p01_classification":
        "Retain after removing the unrelated Fortune 500 statistic.",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at line {line_number}")
            rows.append(row)
    return rows


def main() -> None:
    rows = read_jsonl(INPUT_PATH)
    ids = [str(row.get("example_id", "")) for row in rows]

    if len(rows) != 68:
        raise ValueError(f"Expected 68 v2 examples, found {len(rows)}")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate example IDs found in v2 input")

    known_ids = set(ids)
    expected_ids = set(EXCLUDE_REASONS) | set(RESPONSE_CORRECTIONS) | set(DECISION_NOTES)
    missing = expected_ids - known_ids
    if missing:
        raise ValueError(f"Reviewed IDs not found in input: {sorted(missing)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    reviewed: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    corrected: list[dict[str, str]] = []

    for row in rows:
        example_id = str(row.get("example_id", ""))

        if example_id in EXCLUDE_REASONS:
            excluded.append({
                "example_id": example_id,
                "reason": EXCLUDE_REASONS[example_id],
            })
            continue

        updated = dict(row)
        if example_id in RESPONSE_CORRECTIONS:
            old_response = str(updated.get("response", ""))
            new_response = RESPONSE_CORRECTIONS[example_id]
            updated["response"] = new_response
            corrected.append({
                "example_id": example_id,
                "old_response": old_response,
                "new_response": new_response,
                "evidence_unchanged": True,
            })

        reviewed.append(updated)

    if len(reviewed) != 66:
        raise ValueError(f"Expected 66 retained examples, got {len(reviewed)}")

    # Confirm no evidence fields changed during review.
    original_by_id = {str(row.get("example_id")): row for row in rows}
    for row in reviewed:
        original = original_by_id[str(row.get("example_id"))]
        if row.get("evidence_span") != original.get("evidence_span"):
            raise AssertionError(f"Evidence span changed: {row.get('example_id')}")
        if row.get("evidence_block_ids") != original.get("evidence_block_ids"):
            raise AssertionError(f"Evidence block IDs changed: {row.get('example_id')}")

    with OUTPUT_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        for row in reviewed:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    decisions = {
        "input_count": len(rows),
        "retained_count": len(reviewed),
        "excluded_count": len(excluded),
        "corrected_count": len(corrected),
        "excluded": excluded,
        "corrected": corrected,
        "additional_retained_decisions": DECISION_NOTES,
        "evidence_spans_modified": 0,
        "evidence_block_ids_modified": 0,
        "original_v2_file_modified": False,
        "frozen_corpus_modified": False,
        "output_file": str(OUTPUT_PATH.relative_to(ROOT)),
    }
    DECISIONS_PATH.write_text(
        json.dumps(decisions, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("V2 MANUAL REVIEW APPLIED")
    print(json.dumps({
        "input_count": len(rows),
        "retained_count": len(reviewed),
        "excluded_count": len(excluded),
        "corrected_count": len(corrected),
        "evidence_spans_modified": 0,
        "evidence_block_ids_modified": 0,
        "original_v2_file_modified": False,
        "frozen_corpus_modified": False,
        "reviewed_dataset": str(OUTPUT_PATH.relative_to(ROOT)),
        "decision_report": str(DECISIONS_PATH.relative_to(ROOT)),
    }, indent=2))
    print("\nExcluded:")
    for item in excluded:
        print(f"- {item['example_id']}: {item['reason']}")
    print("\nCorrected:")
    for item in corrected:
        print(f"- {item['example_id']}")
    print("\nNo source dataset or frozen corpus was modified.")


if __name__ == "__main__":
    main()