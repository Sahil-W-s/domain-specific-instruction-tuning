"""
Small-batch Groq judge benchmark: 8 examples, 2 per source document.

Run from project root:
    python scripts/35_benchmark_groq_judge.py

Reads GROQ_API_KEY from .env. Does not alter raw dataset or frozen chunks.
"""
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "interim" / "dataset_v0_raw.jsonl"
CHUNKS_PATH = ROOT / "data" / "frozen" / "chunks_v1_frozen.jsonl"
OUTPUT_PATH = ROOT / "data" / "interim" / "groq_judge_benchmark.jsonl"
SUMMARY_PATH = ROOT / "data" / "interim" / "groq_judge_benchmark_summary.json"

API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-120b"
EXAMPLES_PER_DOC = 2
REQUEST_TIMEOUT = 180


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_evidence_blocks(source):
    blocks = []
    pattern = re.compile(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", re.DOTALL)
    for match in pattern.finditer(source):
        block = match.group(0)
        if len(block.strip()) >= 30:
            blocks.append({"id": len(blocks) + 1, "text": block})
    return blocks


def select_examples(examples):
    grouped = defaultdict(list)
    for ex in examples:
        grouped[ex.get("doc_id", "unknown")].append(ex)

    selected = []
    for doc_id in sorted(grouped):
        rows = sorted(grouped[doc_id], key=lambda x: x.get("example_id", ""))
        # Take first and last examples for deterministic coverage across batches.
        if len(rows) <= EXAMPLES_PER_DOC:
            selected.extend(rows)
        else:
            selected.extend([rows[0], rows[-1]])
    return selected


def evaluate_one(session, headers, example, chunk_lookup):
    chunk_id = example.get("chunk_id")
    if chunk_id not in chunk_lookup:
        raise ValueError(f"Unknown chunk_id for {example.get('example_id')}: {chunk_id}")

    source = chunk_lookup[chunk_id]["content"]
    blocks = extract_evidence_blocks(source)
    block_lookup = {b["id"]: b["text"] for b in blocks}
    ids = [int(x) for x in example.get("evidence_block_ids", [])]
    if not ids or any(i not in block_lookup for i in ids):
        raise ValueError(f"Invalid evidence block IDs for {example.get('example_id')}")

    selected_evidence = "\n\n".join(block_lookup[i] for i in ids)
    payload = {
        "example_id": example.get("example_id"),
        "doc_id": example.get("doc_id"),
        "chunk_id": chunk_id,
        "instruction": example.get("instruction"),
        "response": example.get("response"),
        "selected_evidence": selected_evidence,
        "full_source_chunk": source,
    }

    system_prompt = """You are an independent evaluator of source-grounded technical QA data.
Evaluate only against the supplied full source chunk. The selected evidence is the
specific evidence cited by the generator; determine whether it directly supports
the answer. Do not use outside knowledge to rescue unsupported claims.

Return one JSON object with exactly these keys:
{
  "faithfulness_score": 1,
  "evidence_relevant": true,
  "unsupported_claims": false,
  "answer_correct_from_source": true,
  "answer_complete_from_source": true,
  "issues": [],
  "reason": "brief explanation grounded in the source"
}
Scoring:
5 = fully supported, accurate, and appropriately complete.
4 = supported with only minor omissions or imprecision.
3 = partly supported or contains a material omission/ambiguity.
2 = substantial unsupported content or incorrect interpretation.
1 = mostly unsupported, contradicted, or irrelevant.
If selected evidence is irrelevant but another part of the full chunk supports
the answer, set evidence_relevant=false. Do not treat topic similarity as support.
Return JSON only."""

    user_prompt = "Evaluate this example against the complete source chunk:\n\n" + json.dumps(payload, ensure_ascii=False)
    body = {
        "model": MODEL,
        "temperature": 0,
        "max_completion_tokens": 1400,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    response = session.post(
        API_URL, headers=headers, json=body, timeout=REQUEST_TIMEOUT
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"HTTP {response.status_code} for {example.get('example_id')}: "
            f"{response.text[:1500]}"
        )

    result = response.json()
    choice = result["choices"][0]
    content = choice.get("message", {}).get("content")
    if not content:
        raise RuntimeError(
            f"Empty response for {example.get('example_id')}; "
            f"finish_reason={choice.get('finish_reason')}"
        )
    evaluation = json.loads(content.strip())

    score = evaluation.get("faithfulness_score")
    if not isinstance(score, int) or score not in range(1, 6):
        raise ValueError(f"Invalid score for {example.get('example_id')}: {score!r}")
    for key in (
        "evidence_relevant", "unsupported_claims",
        "answer_correct_from_source", "answer_complete_from_source"
    ):
        if not isinstance(evaluation.get(key), bool):
            raise ValueError(f"Invalid {key} for {example.get('example_id')}")
    if not isinstance(evaluation.get("issues"), list):
        raise ValueError(f"Invalid issues field for {example.get('example_id')}")
    if not isinstance(evaluation.get("reason"), str):
        raise ValueError(f"Invalid reason field for {example.get('example_id')}")

    return {
        "example_id": example.get("example_id"),
        "doc_id": example.get("doc_id"),
        "chunk_id": chunk_id,
        "requested_model": MODEL,
        "actual_model": result.get("model"),
        "finish_reason": choice.get("finish_reason"),
        "usage": result.get("usage"),
        "evaluation": evaluation,
    }


def main():
    load_dotenv(ROOT / ".env", override=True)
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY missing from project-root .env")

    examples = read_jsonl(RAW_PATH)
    chunks = read_jsonl(CHUNKS_PATH)
    chunk_lookup = {row["chunk_id"]: row for row in chunks}
    selected = select_examples(examples)

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    results = []
    failures = []

    print(f"Model: {MODEL}")
    print(f"Selected: {len(selected)} examples ({EXAMPLES_PER_DOC} per document)")
    print("This benchmark does not modify the raw dataset.\n")

    with requests.Session() as session:
        for index, example in enumerate(selected, start=1):
            example_id = example.get("example_id")
            print(f"[{index}/{len(selected)}] Evaluating {example_id} ...", end=" ")
            started = time.time()
            try:
                result = evaluate_one(session, headers, example, chunk_lookup)
                results.append(result)
                evaluation = result["evaluation"]
                print(
                    f"score={evaluation['faithfulness_score']}, "
                    f"evidence={evaluation['evidence_relevant']}, "
                    f"unsupported={evaluation['unsupported_claims']}, "
                    f"{time.time() - started:.1f}s"
                )
            except Exception as exc:
                failures.append({"example_id": example_id, "error": str(exc)})
                print(f"FAILED: {exc}")
                # Stop on quota/auth errors rather than repeatedly hitting the API.
                if "HTTP 429" in str(exc) or "HTTP 401" in str(exc) or "HTTP 403" in str(exc):
                    print("Stopping benchmark because the API rejected the request.")
                    break

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    scores = [r["evaluation"]["faithfulness_score"] for r in results]
    summary = {
        "requested_model": MODEL,
        "successful_evaluations": len(results),
        "failed_evaluations": len(failures),
        "score_counts": {str(i): scores.count(i) for i in range(1, 6)},
        "mean_faithfulness_score": round(sum(scores) / len(scores), 3) if scores else None,
        "evidence_relevant_true": sum(r["evaluation"]["evidence_relevant"] for r in results),
        "unsupported_claims_true": sum(r["evaluation"]["unsupported_claims"] for r in results),
        "failures": failures,
        "output_file": str(OUTPUT_PATH.relative_to(ROOT)),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nBENCHMARK SUMMARY")
    print(json.dumps(summary, indent=2))
    print(f"\nDetailed results: {OUTPUT_PATH.relative_to(ROOT)}")
    print(f"Summary: {SUMMARY_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()