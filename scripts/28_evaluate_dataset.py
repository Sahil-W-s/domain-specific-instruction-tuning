
import json
import os
import re
import time
from pathlib import Path
from collections import defaultdict

from dotenv import load_dotenv
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = ROOT / "data" / "interim" / "dataset_v0_raw.jsonl"
CHUNKS_FILE = ROOT / "data" / "frozen" / "chunks_v1_frozen.jsonl"

OUTPUT_DIR = ROOT / "data" / "interim"
HEURISTIC_FILE = OUTPUT_DIR / "heuristic_results.jsonl"
EVALUATION_FILE = OUTPUT_DIR / "evaluation_results.jsonl"
SUMMARY_FILE = OUTPUT_DIR / "evaluation_summary.json"

JUDGE_MODEL = "google/gemma-4-26b-a4b-it:free"

JUDGE_BATCH_SIZE = 5
MIN_FAITHFULNESS_SCORE = 4

REFUSAL_TERMS = [
    "i cannot answer",
    "i can't answer",
    "i cannot help",
    "i can't help",
    "i am unable to answer",
    "as an ai language model",
    "i'm unable to provide"
]


def load_jsonl(path):
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at {path}, line {line_number}: {error}"
                )

    return records


def save_jsonl(path, records):
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


def load_chunks():
    chunks = load_jsonl(CHUNKS_FILE)

    return {
        chunk["chunk_id"]: chunk["content"]
        for chunk in chunks
    }


def normalize_text(value):
    return re.sub(r"\s+", " ", value.strip().lower())


def run_heuristics(records, chunks):
    seen = set()
    results = []

    for record in records:
        reasons = []

        instruction = record.get("instruction", "").strip()
        response = record.get("response", "").strip()
        evidence = record.get("evidence_span", "")
        chunk_id = record.get("chunk_id")

        if not instruction:
            reasons.append("empty_instruction")

        if not response:
            reasons.append("empty_response")

        if not chunk_id or chunk_id not in chunks:
            reasons.append("unknown_chunk_id")

        duplicate_key = (
            normalize_text(instruction),
            normalize_text(response)
        )

        if instruction and response:
            if duplicate_key in seen:
                reasons.append("exact_duplicate")
            else:
                seen.add(duplicate_key)

        combined_text = f"{instruction} {response}".lower()

        if any(term in combined_text for term in REFUSAL_TERMS):
            reasons.append("refusal_keyword")

        if not evidence:
            reasons.append("empty_evidence")
        elif chunk_id in chunks and evidence not in chunks[chunk_id]:
            reasons.append("evidence_not_exact_substring")

        results.append({
            "example_id": record.get("example_id"),
            "chunk_id": chunk_id,
            "instruction": instruction,
            "response": response,
            "evidence_span": evidence,
            "heuristic_pass": len(reasons) == 0,
            "heuristic_reasons": reasons
        })

    return results


def parse_judge_response(content):
    if not content:
        raise ValueError("Judge returned empty content")

    content = content.strip()

    content = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        content,
        flags=re.IGNORECASE
    ).strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")

        if start < 0 or end < 0:
            raise ValueError("Judge response contains no JSON object")

        return json.loads(content[start:end + 1])


def build_judge_prompt(batch):
    examples = []

    for item in batch:
        examples.append({
            "example_id": item["example_id"],
            "question": item["instruction"],
            "answer": item["response"],
            "selected_evidence": item["evidence_span"]
        })

    return f"""
You are an independent evaluator of source-grounded technical
question-answer examples.

Evaluate ONLY the selected evidence supplied with each example.
Do not use outside knowledge or assume facts not present in evidence.

For each example assess:

1. Does the evidence directly support the answer?
2. Is the answer fully supported, without unsupported claims?
3. Is the question answerable from the selected evidence?
4. Is the answer faithful to the evidence?

Give a faithfulness score from 1 to 5:

5 = Fully supported; evidence directly supports every material claim.
4 = Supported; minor wording differences, no material unsupported claim.
3 = Partially supported; some claims lack sufficient evidence.
2 = Mostly unsupported or evidence is poorly matched.
1 = Contradicted, fabricated, or unrelated to evidence.

Be strict. Do not reward an answer merely because it sounds plausible.

Return ONLY valid JSON:
{{
  "evaluations": [
    {{
      "example_id": "ID",
      "faithfulness_score": 1,
      "evidence_relevant": true,
      "unsupported_claims": [],
      "reason": "Short explanation"
    }}
  ]
}}

EXAMPLES:
{json.dumps(examples, ensure_ascii=False, indent=2)}
"""


def judge_batch(client, batch):
    prompt = build_judge_prompt(batch)

    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=1800
    )

    parsed = parse_judge_response(
        response.choices[0].message.content
    )

    evaluations = parsed.get("evaluations")

    if not isinstance(evaluations, list):
        raise ValueError("Judge did not return evaluations list")

    expected_ids = {item["example_id"] for item in batch}
    received_ids = {
        item.get("example_id")
        for item in evaluations
        if isinstance(item, dict)
    }

    if expected_ids != received_ids:
        raise ValueError(
            f"Judge ID mismatch. Expected {expected_ids}, "
            f"received {received_ids}"
        )

    return evaluations, response.model


def main():
    load_dotenv(ROOT / ".env", override=True)

    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {RAW_FILE}\n"
            "Run production generation before evaluation."
        )

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY missing from .env")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        timeout=120
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_records = load_jsonl(RAW_FILE)
    chunks = load_chunks()

    print("Raw records:", len(raw_records))
    print("Frozen chunks:", len(chunks))

    heuristic_results = run_heuristics(raw_records, chunks)
    save_jsonl(HEURISTIC_FILE, heuristic_results)

    candidates = [
        item for item in heuristic_results
        if item["heuristic_pass"]
    ]

    rejected_heuristics = len(raw_records) - len(candidates)

    print("Heuristic passes:", len(candidates))
    print("Heuristic rejections:", rejected_heuristics)

    # Resume completed judge decisions.
    completed = {}

    if EVALUATION_FILE.exists():
        for item in load_jsonl(EVALUATION_FILE):
            if item.get("judge_status") == "success":
                completed[item["example_id"]] = item

    pending = [
        item for item in candidates
        if item["example_id"] not in completed
    ]

    print("Already judged:", len(completed))
    print("Pending judge evaluations:", len(pending))

    with EVALUATION_FILE.open("a", encoding="utf-8") as output:
        for start in range(0, len(pending), JUDGE_BATCH_SIZE):
            batch = pending[start:start + JUDGE_BATCH_SIZE]

            print(
                f"\nJudging batch "
                f"{start // JUDGE_BATCH_SIZE + 1}"
            )

            try:
                evaluations, actual_model = judge_batch(
                    client, batch
                )

                for evaluation in evaluations:
                    score = evaluation.get("faithfulness_score")

                    if (
                        not isinstance(score, int)
                        or isinstance(score, bool)
                        or score < 1
                        or score > 5
                    ):
                        raise ValueError(
                            "Invalid faithfulness score"
                        )

                    result = {
                        **evaluation,
                        "judge_model": actual_model,
                        "judge_status": "success",
                        "accepted": (
                            score >= MIN_FAITHFULNESS_SCORE
                            and evaluation.get("evidence_relevant") is True
                            and not evaluation.get("unsupported_claims")
                        )
                    }

                    output.write(
                        json.dumps(result, ensure_ascii=False) + "\n"
                    )
                    output.flush()

                    completed[result["example_id"]] = result

                print("Completed:", len(evaluations))

            except Exception as error:
                print(
                    "Judge batch failed:",
                    type(error).__name__,
                    str(error)[:500]
                )

                # Failed batches remain pending for the next run.
                continue

            time.sleep(1)

    accepted = sum(
        result.get("accepted") is True
        for result in completed.values()
    )

    rejected_judge = sum(
        result.get("accepted") is False
        for result in completed.values()
    )

    summary = {
        "raw_examples": len(raw_records),
        "heuristic_passed": len(candidates),
        "heuristic_rejected": rejected_heuristics,
        "judge_completed": len(completed),
        "judge_pending": len(candidates) - len(completed),
        "judge_accepted": accepted,
        "judge_rejected": rejected_judge,
        "judge_model": JUDGE_MODEL,
        "minimum_faithfulness_score": MIN_FAITHFULNESS_SCORE
    }

    with SUMMARY_FILE.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nEVALUATION SUMMARY")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()