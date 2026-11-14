#!/usr/bin/env python3
"""Prepare and validate the B2/B3/B4/B5 training datasets for Phase 3.

This script is read-only with respect to all source datasets. It writes new
normalized JSONL exports under data/phase3/ and refuses to overwrite outputs.

Domain evidence is validated by reproducing the exact evidence-block extraction
used during generation, selecting the stored evidence_block_ids, and comparing
the reconstructed text byte-for-byte (Unicode code points) with evidence_span.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from datasets import load_dataset
except ImportError as exc:  # pragma: no cover - environment-specific
    raise SystemExit(
        "Missing dependency 'datasets'. Install it in the active venv with:\n"
        "  python -m pip install datasets"
    ) from exc

try:
    from pydantic import BaseModel, ValidationError
except ImportError as exc:  # pragma: no cover - environment-specific
    raise SystemExit(
        "Missing dependency 'pydantic'. Install it in the active venv with:\n"
        "  python -m pip install pydantic"
    ) from exc


ROOT = Path.cwd()
OUT = ROOT / "data" / "phase3"
FILES = {
    "raw": ROOT / "data/interim/dataset_v0_raw.jsonl",
    "prior": ROOT / "data/interim/improved_generation/dataset_v0_additional.jsonl",
    "v2": ROOT / "data/interim/production_generation_v2/dataset_v0_additional_v2_resumed.jsonl",
    "b4": ROOT / "data/frozen/dataset_v1_filtered.jsonl",
    "chunks": ROOT / "data/frozen/chunks_v1_frozen.jsonl",
}

# Fingerprints captured from the source inventory. These checks prevent an
# accidental change in the source pools or either frozen artifact.
EXPECTED_HASHES = {
    "raw": "6fa08c9a3caa70bb3c81ad4b06d10e8e8afd31d5abd1d921755130f9cf3a8265",
    "prior": "6399a0d0f8c5a15b043238e79dc8fbfaaea32130d3aa483fcfba12d9be526dba",
    "v2": "6316ec13a17ec7d06877efc8c054225df63276597d0f14151f6f67a44e81e077",
    "b4": "9e541db71c7d5325e3817cd8c29469cfac7137aad0a399cb8b6a88a2e72007a3",
    "chunks": "67e90d3428ff0a421111e19916ceb2dfc007460ae047f18131131be2f80bb827",
}
EXPECTED_COUNTS = {"raw": 88, "prior": 27, "v2": 485, "b4": 577}
DOLLY_DATASET = "databricks/databricks-dolly-15k"
SEED = 20261114


class TrainingExample(BaseModel):
    instruction: str
    input: str
    output: str
    source_id: str
    metadata: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}, line {line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object in {path}, line {line_number}")
            rows.append(value)
    return rows


def write_jsonl_exclusive(path: Path, rows: list[dict[str, Any]]) -> None:
    # 'x' mode ensures an existing output can never be silently overwritten.
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_rows(rows: list[dict[str, Any]], label: str) -> None:
    for index, row in enumerate(rows, start=1):
        try:
            TrainingExample(**row)
        except ValidationError as exc:
            raise ValueError(f"{label} row {index} failed Pydantic validation:\n{exc}") from exc
        for field in ("instruction", "output", "source_id"):
            if not row[field].strip():
                raise ValueError(f"{label} row {index} has an empty {field}")
        if not isinstance(row["input"], str):
            raise ValueError(f"{label} row {index}: input must be a string")


def extract_evidence_blocks(source: str) -> list[str]:
    """Use the exact block-splitting pattern used by the generation pipeline."""
    blocks: list[str] = []
    pattern = re.compile(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", re.DOTALL)
    for match in pattern.finditer(source):
        block = match.group(0)
        if len(block.strip()) >= 30:
            blocks.append(block)
    return blocks


def reconstruct_evidence(
    row: dict[str, Any],
    chunk_blocks: dict[str, list[str]],
) -> str:
    example_id = row.get("example_id", "<missing-example-id>")
    chunk_id = row.get("chunk_id")
    evidence = row.get("evidence_span")
    block_ids = row.get("evidence_block_ids")

    if not isinstance(chunk_id, str) or not chunk_id:
        raise ValueError(f"{example_id}: missing/invalid chunk_id")
    if chunk_id not in chunk_blocks:
        raise ValueError(f"{example_id}: unknown frozen chunk_id {chunk_id!r}")
    if not isinstance(evidence, str) or not evidence:
        raise ValueError(f"{example_id}: missing/empty evidence_span")
    if not isinstance(block_ids, list) or not block_ids:
        raise ValueError(f"{example_id}: missing/empty evidence_block_ids")

    available = chunk_blocks[chunk_id]
    selected: list[str] = []
    for block_id in block_ids:
        # Reject booleans, strings, zero, negative IDs and out-of-range IDs.
        if isinstance(block_id, bool) or not isinstance(block_id, int):
            raise ValueError(f"{example_id}: evidence block ID must be an integer: {block_id!r}")
        if block_id < 1 or block_id > len(available):
            raise ValueError(
                f"{example_id}: evidence block ID {block_id} outside 1..{len(available)}"
            )
        selected.append(available[block_id - 1])

    reconstructed = "\n\n".join(selected)
    if reconstructed != evidence:
        raise ValueError(
            f"{example_id}: evidence reconstruction mismatch; "
            f"chunk={chunk_id}, block_ids={block_ids}, "
            f"reconstructed_chars={len(reconstructed)}, stored_chars={len(evidence)}"
        )
    return reconstructed


def convert_domain(
    row: dict[str, Any],
    dataset_label: str,
    chunks_by_id: dict[str, dict[str, Any]],
    chunk_blocks: dict[str, list[str]],
) -> dict[str, Any]:
    example_id = row.get("example_id", "<missing-example-id>")
    for field in ("example_id", "instruction", "response", "chunk_id", "evidence_span"):
        if not row.get(field):
            raise ValueError(f"{dataset_label} {example_id}: missing/empty {field}")

    evidence = reconstruct_evidence(row, chunk_blocks)
    chunk = chunks_by_id[row["chunk_id"]]

    # Preserve the source record's review/judge/provenance fields in metadata.
    source_metadata = {
        key: value
        for key, value in row.items()
        if key not in {"instruction", "response"}
    }
    return {
        "instruction": row["instruction"],
        "input": evidence,
        "output": row["response"],
        "source_id": row["chunk_id"],
        "metadata": {
            "example_id": example_id,
            "task_type": row.get("task_type", "unspecified"),
            "dataset_version": dataset_label,
            "source_doc_id": chunk.get("doc_id"),
            "source_file": chunk.get("source_file"),
            "source_pages": chunk.get("source_pages"),
            "evidence_block_ids": row["evidence_block_ids"],
            "evidence_span": evidence,
            "source_record_metadata": source_metadata,
        },
    }


def convert_dolly(row: dict[str, Any], assignment: str) -> dict[str, Any]:
    instruction = (row.get("instruction") or "").strip()
    context = (row.get("context") or "").strip()
    response = (row.get("response") or "").strip()
    if not instruction or not response:
        raise ValueError("Dolly contains a row with an empty instruction or response")
    return {
        "instruction": instruction,
        "input": context,
        "output": response,
        "source_id": "dolly15k:" + str(row.get("category") or "uncategorized"),
        "metadata": {
            "dataset": DOLLY_DATASET,
            "category": row.get("category"),
            "assignment": assignment,
        },
    }


def assert_unique_example_ids(rows: list[dict[str, Any]], label: str) -> None:
    ids = [row.get("example_id") for row in rows]
    if any(not isinstance(value, str) or not value for value in ids):
        raise ValueError(f"{label}: one or more records have a missing/invalid example_id")
    if len(ids) != len(set(ids)):
        seen: set[str] = set()
        duplicates: list[str] = []
        for value in ids:
            if value in seen and value not in duplicates:
                duplicates.append(value)
            seen.add(value)
        raise ValueError(f"{label}: duplicate example_id values: {duplicates[:10]}")


def main() -> int:
    print("Phase 3 dataset preparation and validation")
    print("Project root:", ROOT)
    print("Output directory:", OUT)

    # Do not delete or overwrite prior work.
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(
            f"{OUT} already contains files. Nothing was overwritten. "
            "Inspect or move those outputs before rerunning."
        )
    OUT.mkdir(parents=True, exist_ok=True)

    for key, path in FILES.items():
        if not path.is_file():
            raise FileNotFoundError(f"Required source file not found: {path}")

    source_hashes = {key: sha256_file(path) for key, path in FILES.items()}
    for key, expected in EXPECTED_HASHES.items():
        if source_hashes[key] != expected:
            raise ValueError(
                f"Source fingerprint mismatch for {key}:\n"
                f"  expected: {expected}\n  actual:   {source_hashes[key]}\n"
                "No source files were modified. Stop and inspect the source inventory."
            )

    print("Source fingerprints: PASS")

    raw = read_jsonl(FILES["raw"])
    prior = read_jsonl(FILES["prior"])
    v2 = read_jsonl(FILES["v2"])
    b4_source = read_jsonl(FILES["b4"])
    chunk_rows = read_jsonl(FILES["chunks"])

    for key, actual_count in (
        ("raw", len(raw)),
        ("prior", len(prior)),
        ("v2", len(v2)),
        ("b4", len(b4_source)),
    ):
        if actual_count != EXPECTED_COUNTS[key]:
            raise ValueError(
                f"Unexpected {key} count: expected {EXPECTED_COUNTS[key]}, got {actual_count}"
            )

    chunks_by_id = {row.get("chunk_id"): row for row in chunk_rows}
    if None in chunks_by_id or len(chunks_by_id) != len(chunk_rows):
        raise ValueError("Frozen chunk corpus has missing or duplicate chunk IDs")
    chunk_blocks = {
        chunk_id: extract_evidence_blocks(row.get("content", ""))
        for chunk_id, row in chunks_by_id.items()
    }

    b3_source = raw + prior + v2
    assert_unique_example_ids(b3_source, "B3 source pool")
    if len(b3_source) != 600:
        raise ValueError(f"B3 source pool expected 600 candidates, got {len(b3_source)}")
    assert_unique_example_ids(b4_source, "B4 frozen filtered dataset")

    print("Reconstructing evidence blocks for B3 (600 candidates) and B4 (577 rows)...")
    b3 = [convert_domain(row, "B3_unfiltered_reconstructed", chunks_by_id, chunk_blocks) for row in b3_source]
    b4 = [convert_domain(row, "B4_frozen_filtered", chunks_by_id, chunk_blocks) for row in b4_source]
    validate_rows(b3, "B3")
    validate_rows(b4, "B4")
    print("Domain evidence reconstruction: PASS (1,177 records)")

    print(f"Loading {DOLLY_DATASET} from Hugging Face...")
    try:
        dataset = load_dataset(DOLLY_DATASET, split="train")
    except Exception as exc:
        raise RuntimeError(
            "Could not load Databricks Dolly 15k. Check internet access and the "
            "Hugging Face datasets installation, then rerun. Source files remain unchanged."
        ) from exc

    dolly_rows = [dict(row) for row in dataset]
    required_dolly = 1577
    if len(dolly_rows) < required_dolly:
        raise ValueError(f"Dolly dataset has {len(dolly_rows)} rows; at least {required_dolly} required")

    rng = random.Random(SEED)
    indices = list(range(len(dolly_rows)))
    rng.shuffle(indices)
    b2_indices = indices[:1000]
    b5_general_indices = indices[1000:1577]
    if set(b2_indices).intersection(b5_general_indices):
        raise AssertionError("B2 and B5 Dolly samples overlap")

    b2 = [convert_dolly(dolly_rows[index], "B2_generic_only") for index in b2_indices]
    b5_general = [convert_dolly(dolly_rows[index], "B5_general_mix") for index in b5_general_indices]
    b5_domain = [dict(row) for row in b4]
    b5 = b5_domain + b5_general
    rng.shuffle(b5)

    for label, rows in (
        ("B2", b2),
        ("B3", b3),
        ("B4", b4),
        ("B5 general", b5_general),
        ("B5 combined", b5),
    ):
        validate_rows(rows, label)

    outputs = {
        "b2_dolly_1000.jsonl": b2,
        "b3_iot_unfiltered_reconstructed_600.jsonl": b3,
        "b4_iot_filtered_577.jsonl": b4,
        "b5_domain_577.jsonl": b5_domain,
        "b5_general_577.jsonl": b5_general,
        "b5_iot_plus_dolly_1154.jsonl": b5,
    }

    for filename, rows in outputs.items():
        write_jsonl_exclusive(OUT / filename, rows)

    # Read back and validate the files actually written.
    for filename, expected_rows in outputs.items():
        saved_rows = read_jsonl(OUT / filename)
        if len(saved_rows) != len(expected_rows):
            raise ValueError(f"Output count mismatch after writing {filename}")
        validate_rows(saved_rows, filename)

    # Confirm every source, especially the two frozen sources, remained intact.
    final_source_hashes = {key: sha256_file(path) for key, path in FILES.items()}
    if final_source_hashes != source_hashes:
        raise RuntimeError("A source file changed during preparation; investigate immediately")

    output_hashes = {filename: sha256_file(OUT / filename) for filename in outputs}
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_DATASET_PREPARATION",
        "schema": ["instruction", "input", "output", "source_id", "metadata"],
        "pydantic_validation": "PASS",
        "evidence_validation": {
            "method": "Exact reconstruction from frozen chunk paragraph blocks using evidence_block_ids",
            "B3_records_validated": len(b3),
            "B4_records_validated": len(b4),
            "mismatches": 0,
        },
        "counts": {
            "B2_generic_Dolly": len(b2),
            "B3_unfiltered_reconstructed": len(b3),
            "B4_filtered_frozen": len(b4),
            "B5_domain": len(b5_domain),
            "B5_general_Dolly": len(b5_general),
            "B5_combined": len(b5),
        },
        "B3_label": (
            "Reconstructed candidate pool: 88 original raw + 27 prior additional + "
            "485 V2 candidates. This is not a pre-filter frozen snapshot."
        ),
        "Dolly": {
            "dataset": DOLLY_DATASET,
            "rows_available": len(dolly_rows),
            "seed": SEED,
            "B2_count": len(b2_indices),
            "B5_count": len(b5_general_indices),
            "B2_B5_disjoint": True,
        },
        "source_hashes_before": source_hashes,
        "source_hashes_after": final_source_hashes,
        "output_hashes": output_hashes,
        "frozen_sources_modified": False,
    }
    report_path = OUT / "phase3_validation_report.json"
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print("\nPHASE 3 DATASET PREPARATION: PASS")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"\nPREPARATION FAILED: {error}", file=sys.stderr)
        raise