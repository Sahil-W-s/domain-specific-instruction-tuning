"""
One-example Groq semantic-judge smoke test.
Run from project root:
    python scripts/34_test_groq_semantic_judge.py
Reads GROQ_API_KEY from .env. Does not modify dataset_v0_raw.jsonl.
"""
import json
import os
import re
from pathlib import Path
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "interim" / "dataset_v0_raw.jsonl"
CHUNKS_PATH = ROOT / "data" / "frozen" / "chunks_v1_frozen.jsonl"
OUTPUT_PATH = ROOT / "data" / "interim" / "groq_semantic_judge_smoke_test.json"
API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-120b"

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

def main():
    load_dotenv(ROOT / ".env", override=True)
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing from project-root .env")

    examples = read_jsonl(RAW_PATH)
    chunks = read_jsonl(CHUNKS_PATH)
    chunk_lookup = {row["chunk_id"]: row for row in chunks}
    candidates = [ex for ex in examples if ex.get("chunk_id") in chunk_lookup and ex.get("evidence_block_ids")]
    if not candidates:
        raise RuntimeError("No examples with evidence block IDs and valid chunk references.")

    example = candidates[0]
    source_chunk = chunk_lookup[example["chunk_id"]]["content"]
    block_lookup = {b["id"]: b["text"] for b in extract_evidence_blocks(source_chunk)}
    selected_ids = [int(x) for x in example["evidence_block_ids"]]
    missing = [i for i in selected_ids if i not in block_lookup]
    if missing:
        raise RuntimeError(f"Evidence block IDs not found in frozen chunk: {missing}")
    selected_evidence = "\n\n".join(block_lookup[i] for i in selected_ids)

    payload = {
        "example_id": example.get("example_id"),
        "instruction": example.get("instruction"),
        "response": example.get("response"),
        "selected_evidence": selected_evidence,
        "full_source_chunk": source_chunk,
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
If selected evidence is irrelevant but another part of the full chunk supports the
answer, set evidence_relevant=false. Do not treat topic similarity as support.
Return JSON only, without markdown fences."""

    user_prompt = "Evaluate this example against the complete source chunk:\n\n" + json.dumps(payload, ensure_ascii=False, indent=2)
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
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    print(f"Testing Groq model: {MODEL}")
    print(f"Example: {payload['example_id']}")
    print(f"Chunk: {example['chunk_id']}")
    print("Sending one evaluation request...")

    response = requests.post(API_URL, headers=headers, json=body, timeout=180)
    if response.status_code != 200:
        print(f"HTTP status: {response.status_code}")
        print(response.text[:3000])
        response.raise_for_status()

    result = response.json()
    choice = result["choices"][0]
    content = choice.get("message", {}).get("content")
    finish_reason = choice.get("finish_reason")
    if not content:
        raise RuntimeError(f"Empty judge response. finish_reason={finish_reason}; usage={result.get('usage')}")

    try:
        evaluation = json.loads(content.strip())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Judge response was not valid JSON:\n{content}") from exc

    score = evaluation.get("faithfulness_score")
    if not isinstance(score, int) or score not in range(1, 6):
        raise RuntimeError(f"Invalid faithfulness_score: {score!r}")
    for key in ("evidence_relevant", "unsupported_claims", "answer_correct_from_source", "answer_complete_from_source"):
        if not isinstance(evaluation.get(key), bool):
            raise RuntimeError(f"Invalid or missing boolean field: {key}")
    if not isinstance(evaluation.get("issues"), list) or not isinstance(evaluation.get("reason"), str):
        raise RuntimeError("Invalid issues/reason fields.")

    output = {
        "requested_model": MODEL,
        "actual_model": result.get("model"),
        "finish_reason": finish_reason,
        "usage": result.get("usage"),
        "example": payload,
        "evaluation": evaluation,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nJUDGE EVALUATION")
    print(json.dumps(evaluation, ensure_ascii=False, indent=2))
    print("\nActual model:", result.get("model"))
    print("Usage:", json.dumps(result.get("usage", {})))
    print("Saved:", OUTPUT_PATH.relative_to(ROOT))

if __name__ == "__main__":
    main()