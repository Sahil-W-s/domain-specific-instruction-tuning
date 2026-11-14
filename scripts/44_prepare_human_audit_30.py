#!/usr/bin/env python3
"""
Create exactly 30 accepted examples for the Phase 2 human audit.

Sampling design:
  - 15 accepted examples from the original raw dataset
  - 15 accepted examples from the improved pilot
  - deterministic, balanced round-robin across source document/task groups
  - includes instruction, response, evidence span, full source chunk, and judge result

Outputs:
  data/interim/human_audit_30.csv
  data/interim/human_audit_30_summary.json

This script does not modify any dataset or evaluation result.
"""

from __future__ import annotations

import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

OLD_DATA = ROOT / "data" / "interim" / "dataset_v0_raw.jsonl"
OLD_JUDGE = ROOT / "data" / "interim" / "groq_evaluation_results.jsonl"

NEW_DATA = ROOT / "data" / "interim" / "improved_generation" / "dataset_v0_additional.jsonl"
NEW_JUDGE = ROOT / "data" / "interim" / "improved_generation" / "groq_pilot_judge_results.jsonl"

CHUNKS = ROOT / "data" / "frozen" / "chunks_v1_frozen.jsonl"

OUT_CSV = ROOT / "data" / "interim" / "human_audit_30.csv"
OUT_SUMMARY = ROOT / "data" / "interim" / "human_audit_30_summary.json"

SEED = 20261112
PER_DATASET = 15


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}, line {line_no}: {exc}")
    return rows


def first(row: dict, names: tuple[str, ...], default=""):
    for name in names:
        if row.get(name) is not None:
            return row[name]
    return default


def get_chunk_id(row: dict) -> str:
    return str(first(row, ("chunk_id", "chunkId", "source_chunk_id"), ""))


def get_source_doc(chunk_id: str, row: dict) -> str:
    explicit = first(row, ("source_doc", "document_id", "doc_id", "source"), "")
    if explicit:
        return str(explicit)
    if "_chunk_" in chunk_id:
        return chunk_id.split("_chunk_", 1)[0]
    return "unknown_source"


def get_evidence_blocks(source: str) -> list[dict]:
    blocks = []
    pattern = re.compile(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", re.DOTALL)
    for match in pattern.finditer(source):
        text = match.group(0)
        if len(text.strip()) >= 30:
            blocks.append({"id": len(blocks) + 1, "text": text})
    return blocks


def make_candidates(dataset_name: str, data_rows: list[dict],
                    judge_rows: list[dict], chunk_map: dict[str, dict]) -> list[dict]:
    data_by_id = {
        str(row.get("example_id")): row
        for row in data_rows if row.get("example_id") is not None
    }

    accepted_judgments = {}
    for judgment in judge_rows:
        eid = str(judgment.get("example_id", ""))
        if eid and judgment.get("accepted") is True:
            accepted_judgments[eid] = judgment

    candidates = []
    for eid, judgment in accepted_judgments.items():
        row = data_by_id.get(eid)
        if row is None:
            continue

        chunk_id = get_chunk_id(row)
        chunk_row = chunk_map.get(chunk_id)
        source_text = ""
        if chunk_row:
            source_text = str(first(
                chunk_row, ("text", "content", "chunk_text", "page_content"), ""
            ))

        evidence_span = str(row.get("evidence_span", ""))
        if not evidence_span and source_text:
            ids = row.get("evidence_block_ids", [])
            try:
                block_map = {
                    item["id"]: item["text"]
                    for item in get_evidence_blocks(source_text)
                }
                evidence_span = "\n\n".join(
                    block_map[int(block_id)] for block_id in ids
                )
            except (KeyError, TypeError, ValueError):
                evidence_span = ""

        task_type = str(row.get("task_type") or "unspecified").lower()
        source_doc = get_source_doc(chunk_id, row)

        candidates.append({
            "dataset_version": dataset_name,
            "example_id": eid,
            "source_doc": source_doc,
            "chunk_id": chunk_id,
            "task_type": task_type,
            "instruction": str(row.get("instruction", "")),
            "response": str(row.get("response", "")),
            "evidence_span": evidence_span,
            "full_source_chunk": source_text,
            "faithfulness_score": judgment.get("faithfulness_score", ""),
            "judge_reason": judgment.get("reason", ""),
        })
    return candidates


def stratified_round_robin(candidates: list[dict], count: int,
                           rng: random.Random) -> list[dict]:
    if len(candidates) < count:
        raise RuntimeError(
            f"Need {count} accepted candidates, found {len(candidates)}."
        )

    groups = defaultdict(list)
    for candidate in candidates:
        key = (candidate["source_doc"], candidate["task_type"])
        groups[key].append(candidate)

    keys = sorted(groups)
    for key in keys:
        rng.shuffle(groups[key])

    selected = []
    while len(selected) < count:
        progressed = False
        for key in keys:
            if groups[key] and len(selected) < count:
                selected.append(groups[key].pop())
                progressed = True
        if not progressed:
            break
    return selected


def main():
    old_data = read_jsonl(OLD_DATA)
    old_judge = read_jsonl(OLD_JUDGE)
    new_data = read_jsonl(NEW_DATA)
    new_judge = read_jsonl(NEW_JUDGE)
    chunk_rows = read_jsonl(CHUNKS)

    chunk_map = {}
    for row in chunk_rows:
        chunk_id = str(first(row, ("chunk_id", "id", "chunkId"), ""))
        if chunk_id:
            chunk_map[chunk_id] = row

    old_candidates = make_candidates(
        "original_dataset_v0", old_data, old_judge, chunk_map
    )
    new_candidates = make_candidates(
        "improved_pilot", new_data, new_judge, chunk_map
    )

    rng = random.Random(SEED)
    old_selected = stratified_round_robin(old_candidates, PER_DATASET, rng)
    new_selected = stratified_round_robin(new_candidates, PER_DATASET, rng)

    selected = old_selected + new_selected
    for index, row in enumerate(selected, 1):
        row["audit_sample_number"] = index
        row["human_verdict"] = ""
        row["issue_category"] = ""
        row["reviewer_notes"] = ""

    columns = [
        "audit_sample_number",
        "dataset_version",
        "example_id",
        "source_doc",
        "chunk_id",
        "task_type",
        "instruction",
        "response",
        "evidence_span",
        "full_source_chunk",
        "faithfulness_score",
        "judge_reason",
        "human_verdict",
        "issue_category",
        "reviewer_notes",
    ]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(selected)

    summary = {
        "human_audit_sample_size": len(selected),
        "original_dataset_selected": len(old_selected),
        "improved_pilot_selected": len(new_selected),
        "original_accepted_candidates_available": len(old_candidates),
        "improved_pilot_accepted_candidates_available": len(new_candidates),
        "seed": SEED,
        "selection_method": "deterministic round-robin stratified by source document and task type within each dataset",
        "task_type_counts": {},
        "source_document_counts": {},
        "output_csv": str(OUT_CSV.relative_to(ROOT)),
        "source_datasets_modified": False,
        "reviewer_instruction": (
            "For each row, inspect instruction, response, evidence_span, and "
            "full_source_chunk. Enter human_verdict as ACCEPT or REJECT. "
            "If rejecting, enter an issue_category and explain in reviewer_notes."
        ),
    }

    for row in selected:
        task = row["task_type"]
        doc = row["source_doc"]
        summary["task_type_counts"][task] = summary["task_type_counts"].get(task, 0) + 1
        summary["source_document_counts"][doc] = summary["source_document_counts"].get(doc, 0) + 1

    OUT_SUMMARY.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nHuman review columns are blank in the CSV. No automatic human verdicts were assigned.")


if __name__ == "__main__":
    main()