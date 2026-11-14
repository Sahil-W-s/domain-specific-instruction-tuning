"""
Apply the approved manual near-duplicate decisions to the 115-example
heuristic-accepted pool. Source files are read-only.

Input:
  data/interim/production_filter/heuristic_accepted.jsonl

Outputs:
  data/interim/production_filter/deduplicated_candidates.jsonl
  data/interim/production_filter/duplicate_decisions.json
  data/interim/production_filter/deduplication_summary.json

This script removes only the five explicitly approved records.
It does not run fuzzy matching or make new duplicate decisions.
"""

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "interim" / "production_filter" / "heuristic_accepted.jsonl"
OUTPUT_DIR = ROOT / "data" / "interim" / "production_filter"

OUTPUT_FILE = OUTPUT_DIR / "deduplicated_candidates.jsonl"
DECISIONS_FILE = OUTPUT_DIR / "duplicate_decisions.json"
SUMMARY_FILE = OUTPUT_DIR / "deduplication_summary.json"

# Exact decisions approved by the user after reviewing all 8 candidate pairs.
DUPLICATE_GROUPS = [
    {
        "pair": 2,
        "reason": "Same core question about the primary functions/purpose of sensors and actuators; retain the more detailed answer.",
        "keep": "production_batch_009_example_01",
        "exclude": "production_batch_033_example_03",
    },
    {
        "pair": 3,
        "reason": "Same automated irrigation system management requirement and identical answer; human audit also rejected the excluded record.",
        "keep": "production_batch_024_example_01",
        "exclude": "production_batch_011_example_03",
    },
    {
        "pair": 4,
        "reason": "Same three physical entities in the smart irrigation example; retain the more detailed answer.",
        "keep": "production_batch_011_example_04",
        "exclude": "production_batch_024_example_02",
    },
    {
        "pair": 5,
        "reason": "Same five M2M standardization activities with wording differences only.",
        "keep": "production_batch_016_example_03",
        "exclude": "production_batch_024_example_03",
    },
    {
        "pair": 6,
        "reason": "Same IEEE 1451 standardization goal; retain the more detailed answer.",
        "keep": "production_batch_016_example_04",
        "exclude": "production_batch_024_example_04",
    },
]

DISTINCT_PAIRS = [
    {
        "pair": 1,
        "example_ids": [
            "production_batch_002_example_01",
            "production_batch_002_example_02",
        ],
        "decision": "KEEP_BOTH",
        "reason": "WAMP publish-subscribe Router role (Broker) and RPC Router role (Dealer) are distinct concepts.",
    },
    {
        "pair": 7,
        "example_ids": [
            "improved_iot_cloud_case_studies_chunk_0013_classification",
            "improved_iot_cloud_case_studies_chunk_0014_classification",
        ],
        "decision": "KEEP_BOTH",
        "reason": "AWS and Skynet are different platforms with different source-grounded classifications.",
    },
    {
        "pair": 8,
        "example_ids": [
            "improved_iot_cloud_case_studies_chunk_0014_classification",
            "improved_iot_unit1_chunk_0012_classification",
        ],
        "decision": "KEEP_BOTH",
        "reason": "Skynet and Google Compute Engine are different platforms and their classifications are not duplicates.",
    },
]


def read_jsonl(path):
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at line {line_number}")
            rows.append(row)
    return rows


def write_jsonl(path, rows):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(path)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(INPUT_FILE)

    ids = [str(row.get("example_id") or "") for row in rows]
    if len(ids) != len(set(ids)):
        duplicates = [key for key, count in Counter(ids).items() if key and count > 1]
        raise ValueError(f"Input contains duplicate example IDs; resolve before deduplication: {duplicates}")

    id_set = set(ids)
    keep_ids = {item["keep"] for item in DUPLICATE_GROUPS}
    exclude_ids = {item["exclude"] for item in DUPLICATE_GROUPS}

    missing = sorted((keep_ids | exclude_ids) - id_set)
    if missing:
        raise ValueError(
            "Approved IDs missing from heuristic-accepted input; no output written: "
            + ", ".join(missing)
        )

    if keep_ids & exclude_ids:
        raise ValueError("An example is simultaneously marked keep and exclude.")

    output_rows = []
    excluded_rows = []
    for row in rows:
        example_id = str(row.get("example_id") or "")
        if example_id in exclude_ids:
            excluded_rows.append(row)
        else:
            output_rows.append(row)

    expected_count = len(rows) - len(exclude_ids)
    if len(output_rows) != expected_count:
        raise AssertionError(
            f"Unexpected output size: expected {expected_count}, got {len(output_rows)}"
        )

    output_ids = {str(row.get("example_id") or "") for row in output_rows}
    if not keep_ids.issubset(output_ids):
        raise AssertionError("At least one approved retained example is missing.")
    if output_ids & exclude_ids:
        raise AssertionError("At least one excluded example remains in output.")

    write_jsonl(OUTPUT_FILE, output_rows)

    decisions = {
        "reviewed_near_duplicate_pairs": 8,
        "duplicate_pairs": DUPLICATE_GROUPS,
        "distinct_pairs_kept_both": DISTINCT_PAIRS,
        "excluded_example_ids": sorted(exclude_ids),
        "retained_duplicate_representative_ids": sorted(keep_ids),
        "decision_source": "Explicit user approval in conversation",
    }
    DECISIONS_FILE.write_text(
        json.dumps(decisions, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = {
        "input_file": str(INPUT_FILE.relative_to(ROOT)),
        "input_examples": len(rows),
        "duplicate_pairs_removed": len(DUPLICATE_GROUPS),
        "examples_excluded": len(excluded_rows),
        "examples_retained": len(output_rows),
        "distinct_pairs_kept_both": len(DISTINCT_PAIRS),
        "near_duplicate_pairs_reviewed": 8,
        "excluded_example_ids": sorted(exclude_ids),
        "retained_duplicate_representative_ids": sorted(keep_ids),
        "source_files_modified": False,
        "outputs": {
            "deduplicated_candidates": str(OUTPUT_FILE.relative_to(ROOT)),
            "decisions": str(DECISIONS_FILE.relative_to(ROOT)),
            "summary": str(SUMMARY_FILE.relative_to(ROOT)),
        },
    }
    temp_summary = SUMMARY_FILE.with_suffix(SUMMARY_FILE.suffix + ".tmp")
    temp_summary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temp_summary.replace(SUMMARY_FILE)

    print("\nAPPROVED NEAR-DUPLICATE DECISIONS APPLIED")
    print("=" * 48)
    print("Input examples:", len(rows))
    print("Duplicate pairs removed:", len(DUPLICATE_GROUPS))
    print("Examples excluded:", len(excluded_rows))
    print("Examples retained:", len(output_rows))
    print("Distinct pairs kept intact:", len(DISTINCT_PAIRS))
    print("\nExcluded IDs:")
    for item in DUPLICATE_GROUPS:
        print(f"  {item['exclude']}  (kept {item['keep']})")
    print("\nOutputs:")
    print(" ", OUTPUT_FILE)
    print(" ", DECISIONS_FILE)
    print(" ", SUMMARY_FILE)
    print("\nOriginal datasets, heuristic outputs, judge results, and frozen chunks were not modified.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise