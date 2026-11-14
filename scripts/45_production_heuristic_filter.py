"""
Production heuristic filter for the IoT instruction dataset.

Reads (read-only):
  data/interim/dataset_v0_raw.jsonl                         (88 examples)
  data/interim/improved_generation/dataset_v0_additional.jsonl
  data/frozen/chunks_v1_frozen.jsonl

Writes only to:
  data/interim/production_filter/

Near-duplicate candidates are flagged for review, NOT automatically rejected.
Exact duplicate instruction+response pairs are rejected after the first occurrence.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILES = [
    ROOT / "data" / "interim" / "dataset_v0_raw.jsonl",
    ROOT / "data" / "interim" / "improved_generation" / "dataset_v0_additional.jsonl",
]
CHUNKS_FILE = ROOT / "data" / "frozen" / "chunks_v1_frozen.jsonl"

OUTPUT_DIR = ROOT / "data" / "interim" / "production_filter"
ACCEPTED_FILE = OUTPUT_DIR / "heuristic_accepted.jsonl"
REJECTED_FILE = OUTPUT_DIR / "heuristic_rejected.jsonl"
REVIEW_FILE = OUTPUT_DIR / "near_duplicate_candidates.csv"
REPORT_FILE = OUTPUT_DIR / "heuristic_filter_report.json"

# Conservative lexical candidate thresholds. These flag pairs for review only.
SEQUENCE_THRESHOLD = 0.82
TOKEN_JACCARD_THRESHOLD = 0.72

REFUSAL_TERMS = (
    "i cannot answer",
    "i can't answer",
    "i cannot help",
    "i can't help",
    "i am unable to answer",
    "i'm unable to answer",
    "i'm unable to provide",
    "as an ai language model",
    "i cannot provide",
    "i can't provide",
)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Required input does not exist: {path}")
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_number}: {exc}"
                ) from exc
            if not isinstance(item, dict):
                raise ValueError(
                    f"Expected JSON object in {path}, line {line_number}"
                )
            item = dict(item)
            item["_input_file"] = str(path.relative_to(ROOT))
            records.append(item)
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in records:
            # Internal metadata is useful for traceability and retained in outputs.
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    temp_path.replace(path)


def normalize_text(value: object) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", normalize_text(value)))


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def extract_evidence_blocks(source: str) -> list[dict]:
    """Keep identical to the generator and Groq evaluator's block extraction."""
    blocks = []
    pattern = re.compile(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", re.DOTALL)
    for match in pattern.finditer(source):
        block = match.group(0)
        if len(block.strip()) >= 30:
            blocks.append({"id": len(blocks) + 1, "text": block})
    return blocks


def build_chunk_lookup() -> dict:
    chunks = load_jsonl(CHUNKS_FILE)
    lookup = {}
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        if not chunk_id or chunk_id in lookup:
            raise ValueError(f"Missing or duplicate chunk_id in frozen corpus: {chunk_id!r}")
        lookup[chunk_id] = chunk
    return lookup


def evaluate_record(record: dict, chunk_lookup: dict) -> tuple[list[str], bool]:
    reasons = []

    example_id = str(record.get("example_id") or "").strip()
    instruction = str(record.get("instruction") or "").strip()
    response = str(record.get("response") or "").strip()
    chunk_id = record.get("chunk_id")
    chunk = chunk_lookup.get(chunk_id)

    if not example_id:
        reasons.append("missing_example_id")
    if not instruction:
        reasons.append("empty_instruction")
    if not response:
        reasons.append("empty_response")
    if chunk is None:
        reasons.append("unknown_chunk_id")

    if instruction and len(instruction) < 10:
        reasons.append("instruction_too_short")
    if response and len(response) < 3:
        reasons.append("response_too_short")

    response_normalized = normalize_text(response)
    if any(term in response_normalized for term in REFUSAL_TERMS):
        reasons.append("refusal_keyword")

    evidence_ids = record.get("evidence_block_ids")
    evidence_span = record.get("evidence_span")
    evidence_valid = False

    if not isinstance(evidence_ids, list) or not evidence_ids:
        reasons.append("missing_evidence_block_ids")
    elif chunk is not None:
        source = str(chunk.get("content") or "")
        block_lookup = {
            block["id"]: block["text"]
            for block in extract_evidence_blocks(source)
        }
        try:
            ids = [int(value) for value in evidence_ids]
        except (TypeError, ValueError):
            ids = []
            reasons.append("invalid_evidence_block_id_type")

        if not ids:
            if "invalid_evidence_block_id_type" not in reasons:
                reasons.append("empty_evidence_block_ids")
        elif any(block_id not in block_lookup for block_id in ids):
            reasons.append("unknown_evidence_block_id")
        elif len(ids) != len(set(ids)):
            reasons.append("duplicate_evidence_block_id")
        else:
            reconstructed = "\n\n".join(block_lookup[block_id] for block_id in ids)
            if not isinstance(evidence_span, str) or not evidence_span:
                reasons.append("empty_evidence_span")
            elif reconstructed != evidence_span:
                reasons.append("evidence_span_reconstruction_mismatch")
            else:
                evidence_valid = True

    return reasons, evidence_valid


def build_near_duplicate_candidates(records: list[dict]) -> list[dict]:
    candidates = []
    normalized_questions = [
        normalize_text(record.get("instruction", "")) for record in records
    ]
    token_sets = [tokenize(question) for question in normalized_questions]

    for i in range(len(records)):
        if not normalized_questions[i]:
            continue
        for j in range(i + 1, len(records)):
            if not normalized_questions[j]:
                continue

            sequence_score = SequenceMatcher(
                None, normalized_questions[i], normalized_questions[j]
            ).ratio()
            token_score = jaccard(token_sets[i], token_sets[j])

            if (
                sequence_score >= SEQUENCE_THRESHOLD
                or token_score >= TOKEN_JACCARD_THRESHOLD
            ):
                candidates.append({
                    "example_id_1": records[i].get("example_id"),
                    "input_file_1": records[i].get("_input_file"),
                    "chunk_id_1": records[i].get("chunk_id"),
                    "instruction_1": records[i].get("instruction"),
                    "response_1": records[i].get("response"),
                    "example_id_2": records[j].get("example_id"),
                    "input_file_2": records[j].get("_input_file"),
                    "chunk_id_2": records[j].get("chunk_id"),
                    "instruction_2": records[j].get("instruction"),
                    "response_2": records[j].get("response"),
                    "sequence_similarity": round(sequence_score, 4),
                    "token_jaccard": round(token_score, 4),
                    "same_normalized_instruction": (
                        normalized_questions[i] == normalized_questions[j]
                    ),
                    "review_status": "PENDING_MANUAL_REVIEW",
                })
    return candidates


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_records = []
    input_counts = {}
    for path in INPUT_FILES:
        rows = load_jsonl(path)
        input_counts[str(path.relative_to(ROOT))] = len(rows)
        raw_records.extend(rows)

    chunk_lookup = build_chunk_lookup()

    # IDs must be globally unique. We report collisions and reject later occurrences.
    seen_ids = set()
    seen_exact_pairs = {}
    accepted = []
    rejected = []
    duplicate_id_collisions = []
    exact_duplicate_groups = defaultdict(list)

    for record in raw_records:
        reasons, evidence_valid = evaluate_record(record, chunk_lookup)
        example_id = str(record.get("example_id") or "").strip()

        if example_id:
            if example_id in seen_ids:
                reasons.append("duplicate_example_id")
                duplicate_id_collisions.append(example_id)
            else:
                seen_ids.add(example_id)

        instruction_key = normalize_text(record.get("instruction", ""))
        response_key = normalize_text(record.get("response", ""))
        if instruction_key and response_key:
            pair_key = (instruction_key, response_key)
            if pair_key in seen_exact_pairs:
                first_id = seen_exact_pairs[pair_key]
                reasons.append("exact_duplicate_instruction_response")
                exact_duplicate_groups[first_id].append(example_id)
            else:
                seen_exact_pairs[pair_key] = example_id

        output_record = dict(record)
        output_record["heuristic_reasons"] = list(dict.fromkeys(reasons))
        output_record["evidence_provenance_valid"] = evidence_valid
        output_record["heuristic_pass"] = not output_record["heuristic_reasons"]

        if output_record["heuristic_pass"]:
            accepted.append(output_record)
        else:
            rejected.append(output_record)

    near_candidates = build_near_duplicate_candidates(raw_records)

    # CSV is Excel-friendly and contains all pair details for human review.
    if near_candidates:
        fieldnames = list(near_candidates[0].keys())
    else:
        fieldnames = [
            "example_id_1", "input_file_1", "chunk_id_1", "instruction_1",
            "response_1", "example_id_2", "input_file_2", "chunk_id_2",
            "instruction_2", "response_2", "sequence_similarity",
            "token_jaccard", "same_normalized_instruction", "review_status",
        ]
    with REVIEW_FILE.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(near_candidates)

    write_jsonl(ACCEPTED_FILE, accepted)
    write_jsonl(REJECTED_FILE, rejected)

    reason_counts = Counter(
        reason
        for record in rejected
        for reason in record["heuristic_reasons"]
    )
    report = {
        "stage": "production_heuristic_filter",
        "input_files": input_counts,
        "total_input_examples": len(raw_records),
        "frozen_chunks": len(chunk_lookup),
        "heuristic_accepted": len(accepted),
        "heuristic_rejected": len(rejected),
        "near_duplicate_candidate_pairs": len(near_candidates),
        "duplicate_example_id_collisions": duplicate_id_collisions,
        "exact_duplicate_groups": {
            first_id: duplicate_ids
            for first_id, duplicate_ids in exact_duplicate_groups.items()
        },
        "rejection_reason_counts": dict(reason_counts),
        "near_duplicate_policy": (
            "Candidate pairs are flagged for manual review and are not automatically rejected."
        ),
        "exact_duplicate_policy": (
            "A later exact normalized instruction+response pair is rejected; the first occurrence is retained."
        ),
        "source_files_modified": False,
        "outputs": {
            "accepted": str(ACCEPTED_FILE.relative_to(ROOT)),
            "rejected": str(REJECTED_FILE.relative_to(ROOT)),
            "near_duplicate_candidates": str(REVIEW_FILE.relative_to(ROOT)),
            "report": str(REPORT_FILE.relative_to(ROOT)),
        },
    }

    temp_report = REPORT_FILE.with_suffix(REPORT_FILE.suffix + ".tmp")
    temp_report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temp_report.replace(REPORT_FILE)

    print("\nPRODUCTION HEURISTIC FILTER")
    print("=" * 42)
    print("Input examples:", len(raw_records))
    for name, count in input_counts.items():
        print(f"  {name}: {count}")
    print("Frozen chunks:", len(chunk_lookup))
    print("Heuristic accepted:", len(accepted))
    print("Heuristic rejected:", len(rejected))
    print("Near-duplicate candidate pairs:", len(near_candidates))
    print("\nRejection reasons:")
    for reason, count in sorted(reason_counts.items()):
        print(f"  {reason}: {count}")
    print("\nOutputs:")
    print(" ", ACCEPTED_FILE)
    print(" ", REJECTED_FILE)
    print(" ", REVIEW_FILE)
    print(" ", REPORT_FILE)
    print("\nOriginal raw datasets and frozen chunks were not modified.")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise