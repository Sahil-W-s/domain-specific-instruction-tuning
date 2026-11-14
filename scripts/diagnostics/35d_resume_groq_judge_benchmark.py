"""
Robustly resume the 8-example Groq benchmark.

Run from project root:
    python scripts/35d_resume_groq_judge_benchmark.py

- Reuses successful rows in groq_judge_benchmark_v2.jsonl.
- Evaluates only missing benchmark examples.
- Waits 20 seconds between requests.
- On TPM 429, waits 65 seconds and retries (up to 3 attempts).
- Writes each successful evaluation immediately.
- Never modifies dataset_v0_raw.jsonl or frozen chunks.
"""
import json
import os
import re
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "interim" / "dataset_v0_raw.jsonl"
CHUNKS_PATH = ROOT / "data" / "frozen" / "chunks_v1_frozen.jsonl"
OUTPUT_PATH = ROOT / "data" / "interim" / "groq_judge_benchmark_v2.jsonl"
SUMMARY_PATH = ROOT / "data" / "interim" / "groq_judge_benchmark_summary_v2.json"

API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-120b"
INTER_REQUEST_DELAY = 20
TPM_RETRY_DELAY = 65
MAX_ATTEMPTS = 3


def read_jsonl(path):
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path} line {line_no}: {exc}")
    return rows


def extract_blocks(source):
    pattern = re.compile(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", re.DOTALL)
    blocks = []
    for match in pattern.finditer(source):
        block = match.group(0)
        if len(block.strip()) >= 30:
            blocks.append({"id": len(blocks) + 1, "text": block})
    return blocks


def select_benchmark_examples(raw_examples, chunk_map):
    grouped = {}
    for ex in raw_examples:
        chunk = chunk_map.get(ex.get("chunk_id"))
        if chunk is None:
            continue
        grouped.setdefault(chunk["doc_id"], []).append(ex)

    if len(grouped) != 4:
        raise RuntimeError(f"Expected 4 source documents; found {sorted(grouped)}")

    selected = []
    for doc_id in sorted(grouped):
        rows = sorted(grouped[doc_id], key=lambda x: x.get("example_id", ""))
        if len(rows) < 2:
            raise RuntimeError(f"Not enough examples for {doc_id}")
        selected.extend([rows[0], rows[-1]])

    if len(selected) != 8:
        raise RuntimeError(f"Expected 8 benchmark examples, selected {len(selected)}")
    return selected, {doc: len(rows) for doc, rows in grouped.items()}


def build_request(example, chunk):
    source = chunk["content"]
    block_map = {b["id"]: b["text"] for b in extract_blocks(source)}
    ids = [int(x) for x in example.get("evidence_block_ids", [])]
    if not ids or any(i not in block_map for i in ids):
        raise ValueError(f"Invalid evidence block IDs for {example['example_id']}")

    payload = {
        "example_id": example["example_id"],
        "doc_id": chunk["doc_id"],
        "chunk_id": example["chunk_id"],
        "instruction": example.get("instruction"),
        "response": example.get("response"),
        "selected_evidence": "\n\n".join(block_map[i] for i in ids),
        "full_source_chunk": source,
    }

    system_prompt = """You are an independent evaluator of source-grounded technical QA data.
Evaluate only against the supplied full source chunk. Determine whether the selected
evidence directly supports the answer. Do not use outside knowledge.

Return one JSON object with exactly:
{
  "faithfulness_score": 1,
  "evidence_relevant": true,
  "unsupported_claims": false,
  "answer_correct_from_source": true,
  "answer_complete_from_source": true,
  "issues": [],
  "reason": "brief source-grounded explanation"
}
Score 5 = fully supported and accurate; 4 = supported with minor omissions;
3 = partly supported or materially incomplete; 2 = substantially unsupported;
1 = mostly unsupported or contradicted.
If selected evidence is irrelevant, set evidence_relevant=false.
Return JSON only."""

    body = {
        "model": MODEL,
        "temperature": 0,
        "max_completion_tokens": 1400,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": "Evaluate this example against the complete source chunk:\n"
                + json.dumps(payload, ensure_ascii=False),
            },
        ],
    }
    return body


def validate_evaluation(evaluation, example_id):
    score = evaluation.get("faithfulness_score")
    if not isinstance(score, int) or score not in range(1, 6):
        raise ValueError(f"Invalid faithfulness score for {example_id}: {score!r}")
    for key in (
        "evidence_relevant",
        "unsupported_claims",
        "answer_correct_from_source",
        "answer_complete_from_source",
    ):
        if not isinstance(evaluation.get(key), bool):
            raise ValueError(f"Invalid {key} for {example_id}")
    if not isinstance(evaluation.get("issues"), list):
        raise ValueError(f"Invalid issues for {example_id}")
    if not isinstance(evaluation.get("reason"), str):
        raise ValueError(f"Invalid reason for {example_id}")


def call_judge(session, headers, body, example_id):
    for attempt in range(1, MAX_ATTEMPTS + 1):
        response = session.post(API_URL, headers=headers, json=body, timeout=180)

        if response.status_code == 200:
            result = response.json()
            choice = result["choices"][0]
            content = choice.get("message", {}).get("content")
            if not content:
                raise RuntimeError(
                    f"Empty response for {example_id}; finish_reason={choice.get('finish_reason')}"
                )
            evaluation = json.loads(content.strip())
            validate_evaluation(evaluation, example_id)
            return result, evaluation

        if response.status_code == 429:
            body_text = response.text
            wait = TPM_RETRY_DELAY
            match = re.search(r"try again in ([0-9.]+)s", body_text, re.IGNORECASE)
            if match:
                wait = max(TPM_RETRY_DELAY, float(match.group(1)) + 5)
            if attempt < MAX_ATTEMPTS:
                print(f"\n429 rate limit. Waiting {wait:.1f}s before retry {attempt + 1}/{MAX_ATTEMPTS}...")
                time.sleep(wait)
                continue
            raise RuntimeError(f"HTTP 429 after {MAX_ATTEMPTS} attempts: {body_text[:1500]}")

        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:1500]}")

    raise RuntimeError("Unexpected retry exit")


def main():
    load_dotenv(ROOT / ".env", override=True)
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY missing from project-root .env")

    raw_examples = read_jsonl(RAW_PATH)
    chunks = read_jsonl(CHUNKS_PATH)
    chunk_map = {c["chunk_id"]: c for c in chunks}
    selected, raw_distribution = select_benchmark_examples(raw_examples, chunk_map)

    existing_rows = read_jsonl(OUTPUT_PATH)
    # Keep only one stored result per ID, preserving first occurrence.
    existing_by_id = {}
    for row in existing_rows:
        if row.get("example_id") and row["example_id"] not in existing_by_id:
            existing_by_id[row["example_id"]] = row

    selected_ids = {ex["example_id"] for ex in selected}
    unexpected = set(existing_by_id) - selected_ids
    if unexpected:
        raise RuntimeError(
            "Existing benchmark file contains unexpected example IDs: "
            + ", ".join(sorted(unexpected))
        )

    pending = [ex for ex in selected if ex["example_id"] not in existing_by_id]

    print(f"Model: {MODEL}")
    print(f"Raw examples: {len(raw_examples)}")
    print(f"Source distribution: {json.dumps(raw_distribution)}")
    print(f"Benchmark target: {len(selected)}")
    print(f"Already completed and reused: {len(existing_by_id)}")
    print(f"Remaining: {len(pending)}")

    if not pending:
        print("All benchmark examples already have saved evaluations.")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    with requests.Session() as session:
        for index, example in enumerate(pending, start=1):
            example_id = example["example_id"]
            chunk = chunk_map[example["chunk_id"]]
            body = build_request(example, chunk)
            print(f"\n[{index}/{len(pending)}] {chunk['doc_id']} | {example_id}")
            started = time.time()
            result, evaluation = call_judge(session, headers, body, example_id)

            row = {
                "example_id": example_id,
                "doc_id": chunk["doc_id"],
                "chunk_id": example["chunk_id"],
                "requested_model": MODEL,
                "actual_model": result.get("model"),
                "finish_reason": result["choices"][0].get("finish_reason"),
                "usage": result.get("usage"),
                "evaluation": evaluation,
            }
            existing_by_id[example_id] = row

            # Rewrite atomically through a temporary file to avoid partial JSONL.
            ordered_rows = [
                existing_by_id[ex["example_id"]]
                for ex in selected
                if ex["example_id"] in existing_by_id
            ]
            temp_path = OUTPUT_PATH.with_suffix(".tmp")
            with temp_path.open("w", encoding="utf-8") as f:
                for saved in ordered_rows:
                    f.write(json.dumps(saved, ensure_ascii=False) + "\n")
            temp_path.replace(OUTPUT_PATH)

            print(
                f"score={evaluation['faithfulness_score']}, "
                f"evidence={evaluation['evidence_relevant']}, "
                f"unsupported={evaluation['unsupported_claims']}, "
                f"elapsed={time.time() - started:.1f}s"
            )
            if index < len(pending):
                print(f"Cooling down for {INTER_REQUEST_DELAY}s to respect TPM...")
                time.sleep(INTER_REQUEST_DELAY)

    final_rows = [
        existing_by_id[ex["example_id"]]
        for ex in selected
        if ex["example_id"] in existing_by_id
    ]
    scores = [r["evaluation"]["faithfulness_score"] for r in final_rows]
    summary = {
        "requested_model": MODEL,
        "selected_example_count": len(selected),
        "successful_evaluations": len(final_rows),
        "missing_evaluations": len(selected) - len(final_rows),
        "score_counts": {str(i): scores.count(i) for i in range(1, 6)},
        "mean_faithfulness_score": round(sum(scores) / len(scores), 3) if scores else None,
        "evidence_relevant_true": sum(r["evaluation"]["evidence_relevant"] for r in final_rows),
        "unsupported_claims_true": sum(r["evaluation"]["unsupported_claims"] for r in final_rows),
        "document_counts": {
            doc: sum(1 for r in final_rows if r["doc_id"] == doc)
            for doc in sorted(raw_distribution)
        },
        "output_file": str(OUTPUT_PATH.relative_to(ROOT)),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nFINAL BENCHMARK SUMMARY")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()