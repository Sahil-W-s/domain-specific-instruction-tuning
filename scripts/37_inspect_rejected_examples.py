"""
Inspect rejected Groq judge evaluations.

Run from the project root:
    python scripts/37_inspect_rejected_examples.py

Reads:
    data/interim/groq_evaluation_results.jsonl
    data/interim/dataset_v0_raw.jsonl
    data/frozen/chunks_v1_frozen.jsonl

Does not modify any dataset or evaluation file.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVALUATION_FILE = ROOT / "data" / "interim" / "groq_evaluation_results.jsonl"
RAW_FILE = ROOT / "data" / "interim" / "dataset_v0_raw.jsonl"
CHUNKS_FILE = ROOT / "data" / "frozen" / "chunks_v1_frozen.jsonl"
OUTPUT_FILE = ROOT / "data" / "interim" / "rejected_examples_inspection.txt"


def load_jsonl(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path.name}, line {line_no}: {exc}") from exc
    return rows


def first_value(record, *keys, default=None):
    for key in keys:
        value = record.get(key)
        if value is not None:
            return value
    return default


def pretty(value):
    if value is None:
        return "(not provided)"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value)


def main():
    evaluations = load_jsonl(EVALUATION_FILE)
    raw_rows = load_jsonl(RAW_FILE)
    chunks = load_jsonl(CHUNKS_FILE)

    raw_by_id = {
        first_value(row, "example_id", "id"): row
        for row in raw_rows
        if first_value(row, "example_id", "id") is not None
    }
    chunk_by_id = {
        first_value(row, "chunk_id", "id"): row
        for row in chunks
        if first_value(row, "chunk_id", "id") is not None
    }

    rejected = []
    for evaluation in evaluations:
        accepted = first_value(evaluation, "accepted", "is_accepted")
        score = first_value(evaluation, "faithfulness_score", "score")
        # The evaluator's accepted flag is authoritative. Fall back to score
        # only if the flag is absent.
        if accepted is False or (accepted is None and isinstance(score, (int, float)) and score < 4):
            rejected.append(evaluation)

    output = []
    output.append("GROQ REJECTED EXAMPLE INSPECTION")
    output.append("=" * 80)
    output.append(f"Evaluation records: {len(evaluations)}")
    output.append(f"Raw records: {len(raw_rows)}")
    output.append(f"Rejected records found: {len(rejected)}")
    output.append("Read-only inspection; no source files modified.")
    output.append("")

    for index, evaluation in enumerate(rejected, 1):
        example_id = first_value(evaluation, "example_id", "id", default="UNKNOWN_ID")
        raw = raw_by_id.get(example_id, {})
        chunk_id = first_value(raw, "chunk_id", default=first_value(evaluation, "chunk_id"))
        chunk = chunk_by_id.get(chunk_id, {})

        output.extend([
            "#" * 80,
            f"REJECTED EXAMPLE {index}/{len(rejected)}",
            "#" * 80,
            f"Example ID: {example_id}",
            f"Chunk ID: {chunk_id or '(not found)'}",
            f"Faithfulness score: {pretty(first_value(evaluation, 'faithfulness_score', 'score'))}",
            f"Accepted: {pretty(first_value(evaluation, 'accepted', 'is_accepted'))}",
            f"Evidence relevant: {pretty(first_value(evaluation, 'evidence_relevant'))}",
            f"Unsupported claims: {pretty(first_value(evaluation, 'unsupported_claims'))}",
            f"Answer correct from source: {pretty(first_value(evaluation, 'answer_correct_from_source'))}",
            "",
            "JUDGE REASON / EXPLANATION:",
            pretty(first_value(evaluation, "reason", "explanation", "judge_reason", "rationale")),
            "",
            "JUDGE ISSUES:",
            pretty(first_value(evaluation, "issues", "problems", "error_types")),
            "",
            "INSTRUCTION / QUESTION:",
            pretty(first_value(raw, "instruction", "question", "prompt")),
            "",
            "RESPONSE / ANSWER:",
            pretty(first_value(raw, "response", "answer", "output")),
            "",
            "EVIDENCE SPAN:",
            pretty(first_value(raw, "evidence_span", "evidence")),
            "",
            "SOURCE CHUNK:",
            pretty(first_value(chunk, "content", "text")),
            "",
            "FULL EVALUATION RECORD:",
            json.dumps(evaluation, ensure_ascii=False, indent=2),
            "",
        ])

    rendered = "\n".join(output)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(rendered, encoding="utf-8")
    print(rendered)
    print(f"\nSaved inspection report to: {OUTPUT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()