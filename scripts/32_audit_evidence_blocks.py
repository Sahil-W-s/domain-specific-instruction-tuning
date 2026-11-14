"""
Audit evidence-block provenance for generated examples.

This corrects the false-positive check in script 31: evidence_span may contain
multiple source blocks joined with blank lines, so the whole span need not be
one contiguous substring of the source chunk.

Run from project root:
    python scripts/32_audit_evidence_blocks.py
"""

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "interim" / "dataset_v0_raw.jsonl"
CHUNKS_PATH = ROOT / "data" / "frozen" / "chunks_v1_frozen.jsonl"
REPORT_PATH = ROOT / "data" / "interim" / "evidence_provenance_audit.json"


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def extract_evidence_blocks(source):
    """Keep this identical to the production generator's block extraction."""
    blocks = []
    pattern = re.compile(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", re.DOTALL)
    for match in pattern.finditer(source):
        block = match.group(0)
        if len(block.strip()) >= 30:
            blocks.append({"id": len(blocks) + 1, "text": block})
    return blocks


def main():
    examples = read_jsonl(RAW_PATH)
    chunks = read_jsonl(CHUNKS_PATH)
    chunk_lookup = {row["chunk_id"]: row for row in chunks}

    missing_chunk = []
    missing_ids = []
    invalid_ids = []
    reconstructed_mismatch = []
    empty_evidence = []
    valid = 0
    selected_block_count = 0

    for ex in examples:
        example_id = ex.get("example_id", "<missing-example-id>")
        chunk_id = ex.get("chunk_id")
        if chunk_id not in chunk_lookup:
            missing_chunk.append(example_id)
            continue

        source = chunk_lookup[chunk_id].get("content", "")
        blocks = extract_evidence_blocks(source)
        block_lookup = {b["id"]: b["text"] for b in blocks}

        raw_ids = ex.get("evidence_block_ids", [])
        if not isinstance(raw_ids, list) or not raw_ids:
            missing_ids.append(example_id)
            continue

        try:
            ids = [int(x) for x in raw_ids]
        except (TypeError, ValueError):
            invalid_ids.append({"example_id": example_id, "reason": "non-integer block ID"})
            continue

        unknown = [block_id for block_id in ids if block_id not in block_lookup]
        if unknown:
            invalid_ids.append({
                "example_id": example_id,
                "unknown_block_ids": unknown,
                "available_block_count": len(blocks),
            })
            continue

        selected_block_count += len(ids)
        reconstructed = "\n\n".join(block_lookup[block_id] for block_id in ids)
        stored = ex.get("evidence_span", "")

        if not stored:
            empty_evidence.append(example_id)
            continue

        if reconstructed != stored:
            reconstructed_mismatch.append({
                "example_id": example_id,
                "chunk_id": chunk_id,
                "selected_block_ids": ids,
                "stored_evidence_chars": len(stored),
                "reconstructed_evidence_chars": len(reconstructed),
            })
            continue

        # Every selected block is directly taken from this exact frozen chunk.
        # The reconstructed span is allowed to be non-contiguous in source.
        valid += 1

    report = {
        "raw_example_count": len(examples),
        "frozen_chunk_count": len(chunks),
        "valid_provenance_count": valid,
        "selected_source_block_count": selected_block_count,
        "missing_chunk_count": len(missing_chunk),
        "missing_block_ids_count": len(missing_ids),
        "invalid_block_ids_count": len(invalid_ids),
        "reconstructed_span_mismatch_count": len(reconstructed_mismatch),
        "empty_evidence_count": len(empty_evidence),
        "missing_chunk_examples": missing_chunk,
        "missing_block_id_examples": missing_ids,
        "invalid_block_id_examples": invalid_ids,
        "reconstructed_span_mismatches": reconstructed_mismatch,
        "empty_evidence_examples": empty_evidence,
        "interpretation": (
            "This audit verifies exact evidence provenance against the frozen "
            "chunk and generator's block extraction logic. It does NOT verify "
            "that the evidence semantically supports the answer."
        ),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 68)
    print("EVIDENCE-BLOCK PROVENANCE AUDIT")
    print("=" * 68)
    print(f"Raw examples:                    {len(examples)}")
    print(f"Frozen chunks:                   {len(chunks)}")
    print(f"Valid exact provenance:          {valid}")
    print(f"Selected source blocks:          {selected_block_count}")
    print(f"Missing chunk references:        {len(missing_chunk)}")
    print(f"Missing evidence block IDs:      {len(missing_ids)}")
    print(f"Invalid evidence block IDs:      {len(invalid_ids)}")
    print(f"Reconstructed span mismatches:   {len(reconstructed_mismatch)}")
    print(f"Empty evidence spans:            {len(empty_evidence)}")
    print("-" * 68)

    if valid == len(examples) and not any([
        missing_chunk, missing_ids, invalid_ids,
        reconstructed_mismatch, empty_evidence
    ]):
        print("PROVENANCE AUDIT PASSED")
    else:
        print("PROVENANCE AUDIT FAILED — inspect the JSON report.")

    print(f"\nReport saved: {REPORT_PATH.relative_to(ROOT)}")
    print("\nImportant: provenance passing does not establish evidence relevance or answer faithfulness.")


if __name__ == "__main__":
    main()