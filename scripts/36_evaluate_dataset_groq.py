"""
Production Groq evaluator for the IoT instruction dataset.

Run from project root:
    python scripts/36_evaluate_dataset_groq.py

Inputs (read-only):
    data/interim/dataset_v0_raw.jsonl
    data/frozen/chunks_v1_frozen.jsonl

Outputs (new files; does not overwrite the old OpenRouter/Gemma evaluator):
    data/interim/groq_heuristic_results.jsonl
    data/interim/groq_evaluation_results.jsonl
    data/interim/groq_evaluation_summary.json

Design:
- Exact evidence provenance is checked by reconstructing selected evidence blocks.
- One example per Groq request to control the 8K TPM limit.
- Sequential requests, 20-second cooldown, retry handling for HTTP 429.
- Every successful judge result is persisted immediately.
- Resume skips examples with a saved successful result.
"""

import json
import os
import re
import time
from pathlib import Path

import requests
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "interim" / "dataset_v0_raw.jsonl"
CHUNKS_FILE = ROOT / "data" / "frozen" / "chunks_v1_frozen.jsonl"

OUTPUT_DIR = ROOT / "data" / "interim"
HEURISTIC_FILE = OUTPUT_DIR / "groq_heuristic_results.jsonl"
EVALUATION_FILE = OUTPUT_DIR / "groq_evaluation_results.jsonl"
SUMMARY_FILE = OUTPUT_DIR / "groq_evaluation_summary.json"

API_URL = "https://api.groq.com/openai/v1/chat/completions"
JUDGE_MODEL = "openai/gpt-oss-120b"

MIN_FAITHFULNESS_SCORE = 4
INTER_REQUEST_DELAY_SECONDS = 20
TPM_RETRY_FALLBACK_SECONDS = 65
MAX_RETRIES = 3
REQUEST_TIMEOUT_SECONDS = 180

REFUSAL_TERMS = [
    "i cannot answer",
    "i can't answer",
    "i cannot help",
    "i can't help",
    "i am unable to answer",
    "as an ai language model",
    "i'm unable to provide",
]


def load_jsonl(path):
    """Load JSONL; a missing output/checkpoint file is treated as empty."""
    if not path.exists():
        return []

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
                ) from error
    return records


def write_jsonl_atomic(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    temp_path.replace(path)


def append_jsonl(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def normalize_text(value):
    return re.sub(r"\s+", " ", str(value).strip().lower())


def extract_evidence_blocks(source):
    """Must stay identical to the production generator and provenance audit."""
    blocks = []
    pattern = re.compile(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", re.DOTALL)
    for match in pattern.finditer(source):
        block = match.group(0)
        if len(block.strip()) >= 30:
            blocks.append({"id": len(blocks) + 1, "text": block})
    return blocks


def build_chunk_lookup():
    chunk_records = load_jsonl(CHUNKS_FILE)
    lookup = {}
    for chunk in chunk_records:
        chunk_id = chunk.get("chunk_id")
        if not chunk_id or chunk_id in lookup:
            raise ValueError(f"Missing or duplicate chunk_id: {chunk_id!r}")
        lookup[chunk_id] = chunk
    return lookup


def run_heuristics(raw_records, chunk_lookup):
    seen = set()
    results = []

    for record in raw_records:
        reasons = []
        instruction = str(record.get("instruction") or "").strip()
        response = str(record.get("response") or "").strip()
        chunk_id = record.get("chunk_id")
        chunk = chunk_lookup.get(chunk_id)

        if not record.get("example_id"):
            reasons.append("missing_example_id")
        if not instruction:
            reasons.append("empty_instruction")
        if not response:
            reasons.append("empty_response")
        if chunk is None:
            reasons.append("unknown_chunk_id")

        duplicate_key = (normalize_text(instruction), normalize_text(response))
        if instruction and response:
            if duplicate_key in seen:
                reasons.append("exact_duplicate_instruction_response")
            else:
                seen.add(duplicate_key)

        combined_text = f"{instruction} {response}".lower()
        if any(term in combined_text for term in REFUSAL_TERMS):
            reasons.append("refusal_keyword")

        evidence_ids = record.get("evidence_block_ids")
        evidence_span = record.get("evidence_span")
        evidence_valid = False
        selected_evidence = ""
        selected_blocks = []

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
                selected_blocks = [
                    {"block_id": block_id, "text": block_lookup[block_id]}
                    for block_id in ids
                ]
                selected_evidence = "\n\n".join(
                    block_lookup[block_id] for block_id in ids
                )
                if not isinstance(evidence_span, str) or not evidence_span:
                    reasons.append("empty_evidence_span")
                elif selected_evidence != evidence_span:
                    reasons.append("evidence_span_reconstruction_mismatch")
                else:
                    evidence_valid = True

        results.append({
            "example_id": record.get("example_id"),
            "doc_id": chunk.get("doc_id") if chunk else record.get("doc_id"),
            "chunk_id": chunk_id,
            "instruction": instruction,
            "response": response,
            "evidence_block_ids": evidence_ids,
            "evidence_span": evidence_span,
            "selected_evidence_blocks": selected_blocks,
            "evidence_provenance_valid": evidence_valid,
            "heuristic_pass": len(reasons) == 0,
            "heuristic_reasons": reasons,
        })

    return results


def build_judge_messages(item, raw_record, source_chunk):
    evidence_blocks = item["selected_evidence_blocks"]
    payload = {
        "example_id": item["example_id"],
        "question": item["instruction"],
        "answer": item["response"],
        "selected_evidence_blocks": evidence_blocks,
        "full_source_chunk": source_chunk,
    }

    system_prompt = """You are an independent evaluator of source-grounded technical
question-answer examples. The generator used a different model family.

Use ONLY the full source chunk supplied. Assess whether the selected evidence blocks
directly support the answer. Do not use outside knowledge to rescue unsupported claims.
Do not infer that a claim is supported merely because it is technically plausible.

Return exactly one JSON object with these keys and types:
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
5 = Every material claim is directly supported; accurate and appropriately complete.
4 = Supported; only minor omissions or imprecision, no material unsupported claim.
3 = Partially supported, materially incomplete, or contains an ambiguous claim.
2 = Substantially unsupported or materially incorrect.
1 = Contradicted, fabricated, or unrelated to the source.

Set evidence_relevant=false if the selected evidence does not directly support the answer,
even if another part of the full chunk happens to discuss the same topic.
Set unsupported_claims=true if any material answer claim is not supported by the source.
If the source does not contain enough information to answer the question, reflect that
in the score and boolean fields. Be strict. Return JSON only."""

    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": "Evaluate this example:\n\n"
            + json.dumps(payload, ensure_ascii=False),
        },
    ]


def validate_judge_result(result, expected_id):
    if not isinstance(result, dict):
        raise ValueError("Judge response must be a JSON object")

    # Each API request evaluates exactly one known example. Correlate the
    # response with that request locally instead of requiring the model to
    # reproduce an ID (which is not part of the judge's scoring task).
    result["example_id"] = expected_id

    score = result.get("faithfulness_score")
    if (
        not isinstance(score, int)
        or isinstance(score, bool)
        or score < 1
        or score > 5
    ):
        raise ValueError(f"Invalid faithfulness_score: {score!r}")

    for key in (
        "evidence_relevant",
        "unsupported_claims",
        "answer_correct_from_source",
        "answer_complete_from_source",
    ):
        if not isinstance(result.get(key), bool):
            raise ValueError(f"Missing or invalid boolean field: {key}")

    if not isinstance(result.get("issues"), list):
        raise ValueError("Judge 'issues' must be a JSON array")
    if not isinstance(result.get("reason"), str) or not result["reason"].strip():
        raise ValueError("Judge 'reason' must be a non-empty string")

    return result


def parse_retry_wait(response_text):
    match = re.search(
        r"try again in ([0-9.]+)s",
        response_text,
        re.IGNORECASE,
    )
    if match:
        return max(
            TPM_RETRY_FALLBACK_SECONDS,
            float(match.group(1)) + 5,
        )
    return TPM_RETRY_FALLBACK_SECONDS


def call_groq(session, headers, messages, example_id):
    body = {
        "model": JUDGE_MODEL,
        "temperature": 0,
        "max_completion_tokens": 1400,
        "response_format": {"type": "json_object"},
        "messages": messages,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        response = session.post(
            API_URL,
            headers=headers,
            json=body,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        if response.status_code == 200:
            data = response.json()
            choice = data["choices"][0]
            content = choice.get("message", {}).get("content")
            if not content:
                raise ValueError(
                    f"Empty judge response for {example_id}; "
                    f"finish_reason={choice.get('finish_reason')}"
                )
            parsed = json.loads(content.strip())
            validated = validate_judge_result(parsed, example_id)
            return validated, data.get("model"), choice.get("finish_reason"), data.get("usage")

        if response.status_code == 429 and attempt < MAX_RETRIES:
            wait_seconds = parse_retry_wait(response.text)
            print(
                f"Rate limit for {example_id}. "
                f"Waiting {wait_seconds:.1f}s; retry {attempt + 1}/{MAX_RETRIES}."
            )
            time.sleep(wait_seconds)
            continue

        raise RuntimeError(
            f"Groq HTTP {response.status_code} for {example_id}: "
            f"{response.text[:1800]}"
        )

    raise RuntimeError(f"Groq retries exhausted for {example_id}")


def main():
    load_dotenv(ROOT / ".env", override=True)
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY missing from project-root .env")

    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Raw dataset not found: {RAW_FILE}")
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"Frozen chunks not found: {CHUNKS_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_records = load_jsonl(RAW_FILE)
    chunk_lookup = build_chunk_lookup()

    print("Raw records:", len(raw_records))
    print("Frozen chunks:", len(chunk_lookup))

    heuristic_results = run_heuristics(raw_records, chunk_lookup)
    write_jsonl_atomic(HEURISTIC_FILE, heuristic_results)

    candidates = [item for item in heuristic_results if item["heuristic_pass"]]
    heuristic_rejected = len(heuristic_results) - len(candidates)

    print("Heuristic passes:", len(candidates))
    print("Heuristic rejections:", heuristic_rejected)

    # Existing result file is append-only; resume only valid successful IDs.
    existing_rows = load_jsonl(EVALUATION_FILE)
    completed = {}
    for row in existing_rows:
        if row.get("judge_status") == "success" and row.get("example_id"):
            completed[row["example_id"]] = row

    candidate_ids = {item["example_id"] for item in candidates}
    unexpected_ids = set(completed) - candidate_ids
    if unexpected_ids:
        print(
            "Warning: ignoring saved results that are no longer heuristic candidates:",
            sorted(unexpected_ids),
        )

    pending = [
        item for item in candidates
        if item["example_id"] not in completed
    ]

    print("Already judged and reusable:", len(completed) - len(unexpected_ids))
    print("Pending judge evaluations:", len(pending))
    print("Judge model:", JUDGE_MODEL)
    print("Pacing:", INTER_REQUEST_DELAY_SECONDS, "seconds between requests")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    with requests.Session() as session:
        for index, item in enumerate(pending, start=1):
            example_id = item["example_id"]
            chunk = chunk_lookup[item["chunk_id"]]
            print(
                f"\n[{index}/{len(pending)}] "
                f"{chunk.get('doc_id')} | {example_id}"
            )

            messages = build_judge_messages(
                item,
                next(
                    record for record in raw_records
                    if record.get("example_id") == example_id
                ),
                chunk["content"],
            )

            started = time.time()
            try:
                evaluation, actual_model, finish_reason, usage = call_groq(
                    session,
                    headers,
                    messages,
                    example_id,
                )

                accepted = (
                    evaluation["faithfulness_score"] >= MIN_FAITHFULNESS_SCORE
                    and evaluation["evidence_relevant"] is True
                    and evaluation["unsupported_claims"] is False
                    and evaluation["answer_correct_from_source"] is True
                )

                saved = {
                    "example_id": example_id,
                    "doc_id": chunk.get("doc_id"),
                    "chunk_id": item["chunk_id"],
                    "instruction": item["instruction"],
                    "response": item["response"],
                    "evidence_block_ids": item["evidence_block_ids"],
                    "evidence_span": item["evidence_span"],
                    "evidence_provenance_valid": item["evidence_provenance_valid"],
                    **evaluation,
                    "judge_model": actual_model or JUDGE_MODEL,
                    "finish_reason": finish_reason,
                    "usage": usage,
                    "judge_status": "success",
                    "accepted": accepted,
                }

                append_jsonl(EVALUATION_FILE, saved)
                completed[example_id] = saved

                print(
                    f"score={evaluation['faithfulness_score']}, "
                    f"evidence={evaluation['evidence_relevant']}, "
                    f"unsupported={evaluation['unsupported_claims']}, "
                    f"accepted={accepted}, "
                    f"elapsed={time.time() - started:.1f}s"
                )

            except Exception as error:
                # Persist failure separately; it is not treated as completed.
                failure = {
                    "example_id": example_id,
                    "chunk_id": item["chunk_id"],
                    "judge_status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
                append_jsonl(
                    OUTPUT_DIR / "groq_evaluation_failures.jsonl",
                    failure,
                )
                print("FAILED:", failure["error"][:700])
                if "HTTP 401" in str(error) or "HTTP 403" in str(error):
                    print("Stopping: authentication/permission failure.")
                    break
                if "HTTP 429" in str(error):
                    print("Stopping: rate limit persisted after retries. Resume later.")
                    break

            # Deliberately pace all calls, not just successful calls.
            if index < len(pending):
                time.sleep(INTER_REQUEST_DELAY_SECONDS)

    final_rows = [
        row for row in completed.values()
        if row.get("example_id") in candidate_ids
    ]
    final_rows.sort(key=lambda row: row["example_id"])

    accepted_count = sum(row.get("accepted") is True for row in final_rows)
    rejected_count = sum(row.get("accepted") is False for row in final_rows)

    summary = {
        "raw_examples": len(raw_records),
        "heuristic_passed": len(candidates),
        "heuristic_rejected": heuristic_rejected,
        "judge_completed": len(final_rows),
        "judge_pending": len(candidates) - len(final_rows),
        "judge_accepted": accepted_count,
        "judge_rejected": rejected_count,
        "judge_model": JUDGE_MODEL,
        "minimum_faithfulness_score": MIN_FAITHFULNESS_SCORE,
        "inter_request_delay_seconds": INTER_REQUEST_DELAY_SECONDS,
        "evaluation_file": str(EVALUATION_FILE.relative_to(ROOT)),
        "raw_dataset_modified": False,
    }

    SUMMARY_FILE.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nGROQ EVALUATION SUMMARY")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()