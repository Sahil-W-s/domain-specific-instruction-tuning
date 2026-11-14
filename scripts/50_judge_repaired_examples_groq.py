import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "interim" / "production_filter" / "repair_v1" / "repair_candidates.jsonl"
CHUNKS_FILE = ROOT / "data" / "frozen" / "chunks_v1_frozen.jsonl"
OUT_DIR = ROOT / "data" / "interim" / "production_filter" / "repair_v1" / "judge"
RESULTS_FILE = OUT_DIR / "repair_judge_results.jsonl"
SUMMARY_FILE = OUT_DIR / "repair_judge_summary.json"
FAILURES_FILE = OUT_DIR / "repair_judge_failures.jsonl"

MODEL = "openai/gpt-oss-120b"
BASE_URL = "https://api.groq.com/openai/v1"
DELAY_SECONDS = 20
MAX_RETRIES = 4
TIMEOUT_SECONDS = 180

TARGET_IDS = {
    "production_batch_011_example_04",
    "production_batch_012_example_04",
    "production_batch_013_example_03",
    "production_batch_015_example_04",
    "improved_iot_cloud_case_studies_chunk_0014_classification",
}

SYSTEM_PROMPT = """You are an independent evaluator of instruction-response examples for a technical-document dataset.
Evaluate only against the supplied full source chunk and selected evidence span. Do not use outside knowledge.
Be strict: a response must be factually supported by the selected evidence, not merely somewhere in the full chunk.
Check whether the evidence span directly supports every material claim in the response.
A response is complete only if it includes the relevant answer details requested by the instruction and supported by the source.
Return one valid JSON object only, with these keys:
faithfulness_score (integer 1-5),
evidence_relevant (boolean),
unsupported_claims (boolean),
answer_correct_from_source (boolean),
answer_complete_from_source (boolean),
issues (array of short strings),
reason (string).
A score of 4 or 5 requires no material unsupported claim and directly relevant evidence.
Do not reward an answer for facts that appear in the full chunk but are absent from the selected evidence span.
Do not penalize harmless wording differences. Preserve source-specific terminology.
"""

def read_jsonl(path):
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}")
    return rows

def append_jsonl(path, obj):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def parse_json_content(content):
    if not content:
        raise ValueError("Judge returned empty content.")
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S)
    data = json.loads(text)
    required = {
        "faithfulness_score", "evidence_relevant", "unsupported_claims",
        "answer_correct_from_source", "answer_complete_from_source",
        "issues", "reason"
    }
    missing = required - set(data)
    if missing:
        raise ValueError(f"Judge JSON missing keys: {sorted(missing)}")
    if not isinstance(data["faithfulness_score"], int) or not 1 <= data["faithfulness_score"] <= 5:
        raise ValueError("faithfulness_score must be an integer from 1 to 5")
    for key in (
        "evidence_relevant", "unsupported_claims",
        "answer_correct_from_source", "answer_complete_from_source"
    ):
        if not isinstance(data[key], bool):
            raise ValueError(f"{key} must be boolean")
    if not isinstance(data["issues"], list) or not isinstance(data["reason"], str):
        raise ValueError("issues must be a list and reason must be a string")
    return data

def accepted(result):
    return (
        result["faithfulness_score"] >= 4
        and result["evidence_relevant"] is True
        and result["unsupported_claims"] is False
        and result["answer_correct_from_source"] is True
        and result["answer_complete_from_source"] is True
    )

def main():
    load_dotenv(ROOT / ".env", override=True)
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing from the project .env file.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = read_jsonl(INPUT_FILE)
    chunks = read_jsonl(CHUNKS_FILE)
    chunk_map = {x["chunk_id"]: x for x in chunks}

    selected = [x for x in candidates if x.get("example_id") in TARGET_IDS]
    found_ids = {x["example_id"] for x in selected}
    missing_ids = TARGET_IDS - found_ids
    if missing_ids:
        raise ValueError(f"Target IDs missing from repaired candidates: {sorted(missing_ids)}")
    if len(selected) != len(TARGET_IDS):
        raise ValueError("Unexpected duplicate target IDs.")

    prior = {}
    if RESULTS_FILE.exists():
        prior = {
            x["example_id"]: x
            for x in read_jsonl(RESULTS_FILE)
            if x.get("status") == "completed"
        }

    client = OpenAI(api_key=api_key, base_url=BASE_URL, timeout=TIMEOUT_SECONDS)
    completed_this_run = 0
    failed_this_run = 0

    for index, item in enumerate(selected, 1):
        example_id = item["example_id"]
        if example_id in prior:
            print(f"[{index}/{len(selected)}] SKIP already judged: {example_id}")
            continue

        chunk = chunk_map.get(item["chunk_id"])
        if chunk is None:
            raise ValueError(f"Frozen source chunk not found: {item['chunk_id']}")

        user_prompt = f"""Evaluate this example.

EXAMPLE ID: {example_id}

INSTRUCTION:
{item['instruction']}

RESPONSE:
{item['response']}

SELECTED EVIDENCE SPAN:
{item['evidence_span']}

FULL SOURCE CHUNK:
{chunk['content']}

Return the required JSON evaluation only.
"""

        success = False
        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0,
                    max_tokens=1800,
                    response_format={"type": "json_object"},
                )
                choice = response.choices[0]
                evaluation = parse_json_content(choice.message.content)
                result = {
                    "example_id": example_id,
                    "status": "completed",
                    **evaluation,
                    "accepted": accepted(evaluation),
                    "judge_model": MODEL,
                    "finish_reason": choice.finish_reason,
                    "usage": (
                        response.usage.model_dump()
                        if response.usage is not None else None
                    ),
                    "evaluated_at": datetime.now(timezone.utc).isoformat(),
                    "evaluation_policy": "score>=4; evidence relevant; no unsupported claims; correct; complete",
                }
                append_jsonl(RESULTS_FILE, result)
                completed_this_run += 1
                print(
                    f"[{index}/{len(selected)}] COMPLETED {example_id} | "
                    f"score={result['faithfulness_score']} | accepted={result['accepted']}"
                )
                success = True
                break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                message = str(exc).lower()
                retryable = any(x in message for x in ("429", "rate limit", "timeout", "503", "502", "connection"))
                if attempt < MAX_RETRIES and retryable:
                    wait = max(DELAY_SECONDS, 30 * attempt)
                    print(f"  Retryable error on attempt {attempt}: {last_error}; waiting {wait}s")
                    time.sleep(wait)
                else:
                    break

        if not success:
            failed_this_run += 1
            failure = {
                "example_id": example_id,
                "status": "failed",
                "error": last_error,
                "attempted_at": datetime.now(timezone.utc).isoformat(),
            }
            append_jsonl(FAILURES_FILE, failure)
            print(f"[{index}/{len(selected)}] FAILED {example_id}: {last_error}")

        if index < len(selected):
            time.sleep(DELAY_SECONDS)

    all_results = read_jsonl(RESULTS_FILE) if RESULTS_FILE.exists() else []
    latest = {}
    for result in all_results:
        latest[result["example_id"]] = result
    target_results = [latest[x] for x in TARGET_IDS if x in latest]

    summary = {
        "target_count": len(TARGET_IDS),
        "completed_count": len(target_results),
        "accepted_count": sum(x.get("accepted") is True for x in target_results),
        "rejected_count": sum(x.get("accepted") is False for x in target_results),
        "pending_count": len(TARGET_IDS) - len(target_results),
        "completed_this_run": completed_this_run,
        "failed_this_run": failed_this_run,
        "results": [
            {
                "example_id": x["example_id"],
                "faithfulness_score": x["faithfulness_score"],
                "evidence_relevant": x["evidence_relevant"],
                "unsupported_claims": x["unsupported_claims"],
                "answer_correct_from_source": x["answer_correct_from_source"],
                "answer_complete_from_source": x["answer_complete_from_source"],
                "accepted": x["accepted"],
            }
            for x in target_results
        ],
        "raw_datasets_modified": False,
        "frozen_chunks_modified": False,
        "outputs": {
            "results": str(RESULTS_FILE.relative_to(ROOT)),
            "summary": str(SUMMARY_FILE.relative_to(ROOT)),
            "failures": str(FAILURES_FILE.relative_to(ROOT)),
        },
    }
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nSUMMARY")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()