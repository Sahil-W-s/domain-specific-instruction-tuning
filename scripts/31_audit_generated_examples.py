import csv
import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "interim" / "dataset_v0_raw.jsonl"
CHUNKS_FILE = ROOT / "data" / "frozen" / "chunks_v1_frozen.jsonl"
REPORT_DIR = ROOT / "data" / "interim"
REPORT_JSON = REPORT_DIR / "dataset_v0_audit_report.json"
DUPLICATES_CSV = REPORT_DIR / "dataset_v0_duplicate_candidates.csv"

NEAR_DUPLICATE_THRESHOLD = 0.78


def read_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}")
    return records


def normalize(text):
    text = text.casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def token_jaccard(a, b):
    left = set(normalize(a).split())
    right = set(normalize(b).split())
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def main():
    examples = read_jsonl(RAW_FILE)
    chunks = read_jsonl(CHUNKS_FILE)
    chunk_lookup = {chunk["chunk_id"]: chunk for chunk in chunks}

    print("=" * 72)
    print("RAW DATASET QUALITY AUDIT")
    print("=" * 72)
    print("Raw examples:", len(examples))
    print("Frozen chunks:", len(chunks))

    ids = [e.get("example_id") for e in examples]
    duplicate_ids = [item for item, count in Counter(ids).items() if count > 1]

    exact_groups = defaultdict(list)
    for example in examples:
        key = normalize(example.get("instruction", ""))
        exact_groups[key].append(example)

    exact_duplicates = [
        group for group in exact_groups.values() if len(group) > 1
    ]

    candidates = []
    for i in range(len(examples)):
        left = examples[i]
        q_left = left.get("instruction", "")
        for j in range(i + 1, len(examples)):
            right = examples[j]
            q_right = right.get("instruction", "")

            if normalize(q_left) == normalize(q_right):
                continue

            sequence_score = SequenceMatcher(
                None, normalize(q_left), normalize(q_right)
            ).ratio()
            jaccard_score = token_jaccard(q_left, q_right)

            # Candidate flag only. Human/model review is needed to establish
            # semantic duplication; lexical similarity is not proof.
            if sequence_score >= NEAR_DUPLICATE_THRESHOLD or jaccard_score >= 0.72:
                candidates.append({
                    "example_id_1": left.get("example_id"),
                    "chunk_id_1": left.get("chunk_id"),
                    "question_1": q_left,
                    "example_id_2": right.get("example_id"),
                    "chunk_id_2": right.get("chunk_id"),
                    "question_2": q_right,
                    "sequence_similarity": round(sequence_score, 3),
                    "token_jaccard": round(jaccard_score, 3),
                })

    evidence_missing = []
    unknown_chunks = []
    per_document = Counter()
    per_chunk = Counter()
    per_batch = Counter()

    for example in examples:
        chunk_id = example.get("chunk_id")
        if chunk_id not in chunk_lookup:
            unknown_chunks.append(example.get("example_id"))
            continue

        source = chunk_lookup[chunk_id].get("content", "")
        evidence = example.get("evidence_span", "")

        if not evidence or evidence not in source:
            evidence_missing.append(example.get("example_id"))

        per_document[chunk_lookup[chunk_id].get("doc_id", "unknown")] += 1
        per_chunk[chunk_id] += 1
        per_batch[example.get("batch_id", "unknown")] += 1

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with DUPLICATES_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        fields = [
            "example_id_1", "chunk_id_1", "question_1",
            "example_id_2", "chunk_id_2", "question_2",
            "sequence_similarity", "token_jaccard"
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(candidates)

    report = {
        "total_examples": len(examples),
        "unique_example_ids": len(set(ids)),
        "duplicate_example_ids": duplicate_ids,
        "exact_duplicate_groups": [
            [e.get("example_id") for e in group]
            for group in exact_duplicates
        ],
        "near_duplicate_candidate_pairs": candidates,
        "evidence_not_found_in_source_chunk": evidence_missing,
        "unknown_chunk_examples": unknown_chunks,
        "examples_by_document": dict(per_document),
        "examples_by_chunk": dict(per_chunk),
        "examples_by_batch": dict(per_batch),
        "note": (
            "Near-duplicate candidates use lexical similarity only. "
            "They require semantic review and are not automatic rejections."
        )
    }

    with REPORT_JSON.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\nSTRUCTURAL RESULTS")
    print("Unique example IDs:", len(set(ids)))
    print("Duplicate example IDs:", len(duplicate_ids))
    print("Exact duplicate question groups:", len(exact_duplicates))
    print("Near-duplicate candidate pairs:", len(candidates))
    print("Evidence not found in source chunk:", len(evidence_missing))
    print("Examples with unknown chunk IDs:", len(unknown_chunks))

    print("\nEXAMPLES BY DOCUMENT")
    for doc_id, count in sorted(per_document.items()):
        print(f"  {doc_id}: {count}")

    print("\nEXAMPLES BY BATCH")
    for batch_id, count in sorted(per_batch.items()):
        print(f"  {batch_id}: {count}")

    print("\nNear-duplicate candidates (first 20):")
    for item in candidates[:20]:
        print(
            f"- {item['example_id_1']} vs {item['example_id_2']} "
            f"(sequence={item['sequence_similarity']}, "
            f"jaccard={item['token_jaccard']})"
        )
        print("  Q1:", item["question_1"])
        print("  Q2:", item["question_2"])

    print("\nReports saved:")
    print(" ", REPORT_JSON)
    print(" ", DUPLICATES_CSV)
    print("\nNo examples were modified or deleted.")


if __name__ == "__main__":
    main()