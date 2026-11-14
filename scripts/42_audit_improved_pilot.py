#!/usr/bin/env python3
"""Read-only audit of the improved-generation pilot dataset."""

from __future__ import annotations

import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEW_FILE = ROOT / "data" / "interim" / "improved_generation" / "dataset_v0_additional.jsonl"
RAW_FILE = ROOT / "data" / "interim" / "dataset_v0_raw.jsonl"
CHUNKS_FILE = ROOT / "data" / "frozen" / "chunks_v1_frozen.jsonl"
OUT_FILE = ROOT / "data" / "interim" / "improved_generation" / "pilot_audit_report.json"
SIMILARITY_THRESHOLD = 0.88


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"Invalid JSON: {path}, line {n}: {exc}")
    return rows


def first(row: dict, keys: tuple[str, ...], default=None):
    for key in keys:
        if row.get(key) is not None:
            return row[key]
    return default


def cid(row: dict) -> str:
    value = first(row, ("chunk_id", "id", "chunkId"))
    return str(value) if value is not None else ""


def ctext(row: dict) -> str:
    return str(first(row, ("text", "content", "chunk_text", "page_content"), ""))


def blocks(source: str) -> list[dict]:
    out = []
    pattern = re.compile(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", re.DOTALL)
    for match in pattern.finditer(source):
        value = match.group(0)
        if len(value.strip()) >= 30:
            out.append({"id": len(out) + 1, "text": value})
    return out


def norm(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def sim(a: str, b: str) -> float:
    a, b = norm(a), norm(b)
    if not a or not b:
        return 0.0
    return 1.0 if a == b else SequenceMatcher(None, a, b).ratio()


def main():
    new_rows = read_jsonl(NEW_FILE)
    old_rows = read_jsonl(RAW_FILE)
    chunk_rows = read_jsonl(CHUNKS_FILE)

    chunk_map = {cid(row): row for row in chunk_rows}
    errors = []
    exact_dupes = []
    near_dupes = []
    seen_ids = set()
    task_counts = Counter()
    chunk_counts = Counter()
    evidence_lengths = []
    instruction_lengths = []
    response_lengths = []

    old_instructions = [
        (str(row.get("example_id", "raw_existing")), str(row.get("instruction", "")))
        for row in old_rows if row.get("instruction")
    ]

    for index, row in enumerate(new_rows, 1):
        eid = str(row.get("example_id", f"row_{index}"))
        instruction = str(row.get("instruction", "")).strip()
        response = str(row.get("response", "")).strip()
        chunk_id = cid(row)
        task = str(row.get("task_type", "missing")).lower()
        task_counts[task] += 1
        chunk_counts[chunk_id or "missing"] += 1
        instruction_lengths.append(len(instruction))
        response_lengths.append(len(response))

        if eid in seen_ids:
            errors.append({"example_id": eid, "issue": "duplicate_example_id"})
        seen_ids.add(eid)

        if not instruction:
            errors.append({"example_id": eid, "issue": "empty_instruction"})
        if not response:
            errors.append({"example_id": eid, "issue": "empty_response"})
        if task not in {"qa", "extraction", "classification"}:
            errors.append({"example_id": eid, "issue": f"unexpected_task_type:{task}"})
        if chunk_id not in chunk_map:
            errors.append({"example_id": eid, "issue": "chunk_id_not_in_frozen_corpus"})
            continue

        source = ctext(chunk_map[chunk_id])
        evidence_blocks = blocks(source)
        by_id = {b["id"]: b["text"] for b in evidence_blocks}
        raw_ids = row.get("evidence_block_ids", [])
        if not isinstance(raw_ids, list) or not raw_ids:
            errors.append({"example_id": eid, "issue": "missing_evidence_block_ids"})
            continue
        try:
            selected_ids = list(dict.fromkeys(int(x) for x in raw_ids))
        except (ValueError, TypeError):
            errors.append({"example_id": eid, "issue": "invalid_evidence_block_ids"})
            continue
        if any(x not in by_id for x in selected_ids):
            errors.append({"example_id": eid, "issue": "evidence_block_id_out_of_range"})
            continue

        reconstructed = "\n\n".join(by_id[x] for x in selected_ids)
        saved_span = str(row.get("evidence_span", ""))
        evidence_lengths.append(len(saved_span))
        if reconstructed != saved_span:
            errors.append({"example_id": eid, "issue": "evidence_span_mismatch"})
        if reconstructed not in source:
            errors.append({"example_id": eid, "issue": "evidence_not_exact_source_substring"})

        for prior_id, prior_instruction in old_instructions:
            score = sim(instruction, prior_instruction)
            if score >= 0.999:
                exact_dupes.append({
                    "example_id": eid, "duplicate_of": prior_id, "similarity": round(score, 4)
                })
                break
            if score >= SIMILARITY_THRESHOLD:
                near_dupes.append({
                    "example_id": eid, "similar_to": prior_id, "similarity": round(score, 4)
                })
                break

    # Check pairwise duplicate/near-duplicate instructions within the new pilot.
    for i in range(len(new_rows)):
        for j in range(i + 1, len(new_rows)):
            a = str(new_rows[i].get("instruction", ""))
            b = str(new_rows[j].get("instruction", ""))
            score = sim(a, b)
            if score >= 0.999:
                exact_dupes.append({
                    "example_id": new_rows[j].get("example_id"),
                    "duplicate_of": new_rows[i].get("example_id"),
                    "similarity": round(score, 4),
                })
            elif score >= SIMILARITY_THRESHOLD:
                near_dupes.append({
                    "example_id": new_rows[j].get("example_id"),
                    "similar_to": new_rows[i].get("example_id"),
                    "similarity": round(score, 4),
                })

    def stats(values):
        return {
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "average": round(sum(values) / len(values), 2) if values else None,
        }

    report = {
        "new_examples": len(new_rows),
        "existing_raw_examples_read_only": len(old_rows),
        "frozen_chunks": len(chunk_rows),
        "task_type_counts": dict(task_counts),
        "examples_per_chunk": dict(chunk_counts),
        "instruction_character_lengths": stats(instruction_lengths),
        "response_character_lengths": stats(response_lengths),
        "evidence_character_lengths": stats(evidence_lengths),
        "structural_evidence_errors": errors,
        "exact_duplicate_instruction_matches": exact_dupes,
        "near_duplicate_instruction_matches_threshold_0_88": near_dupes,
        "structural_audit_passed": len(errors) == 0,
        "raw_dataset_modified": False,
        "note": "Structural checks do not establish semantic faithfulness; an independent judge and human audit are still required.",
    }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nSaved report: {OUT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()