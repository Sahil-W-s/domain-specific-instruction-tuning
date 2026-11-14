#!/usr/bin/env python3
"""
Cross-model judge for the 66 manually reviewed Generation v2 candidates.

Generator: NVIDIA Nemotron 3.5 Super VL (OpenRouter)
Judge: OpenAI GPT-OSS 120B (Groq)

Reads the reviewed v2 dataset and frozen source chunks.
Writes resumable judge results to a separate directory.
Never modifies source datasets, manual-review outputs, or frozen chunks.

Run from project root:
    python .\scripts\58_judge_v2_candidates_groq.py

Optional:
    python .\scripts\58_judge_v2_candidates_groq.py --delay 20
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = Path(os.getenv(
    "IOT_PIPELINE_JUDGE_INPUT",
    str(ROOT / "data" / "interim" / "production_generation_v2"
        / "manual_review" / "dataset_v2_reviewed.jsonl")
))
FROZEN_PATH = ROOT / "data" / "frozen" / "chunks_v1_frozen.jsonl"
OUT_DIR = Path(os.getenv(
    "IOT_PIPELINE_JUDGE_OUT_DIR",
    str(ROOT / "data" / "interim" / "production_generation_v2" / "groq_judge")
))
RESULTS_PATH = OUT_DIR / "v2_judge_results.jsonl"
SUMMARY_PATH = OUT_DIR / "v2_judge_summary.json"
ERRORS_PATH = OUT_DIR / "v2_judge_errors.jsonl"

API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-120b"
GENERATOR_MODEL = "nvidia/nemotron-3.5-super-vl:free"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path.name}, line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at line {line_number}")
            rows.append(row)
    return rows


def load_chunks() -> dict[str, str]:
    chunks: dict[str, str] = {}
    for row in read_jsonl(FROZEN_PATH):
        chunk_id = str(row.get("chunk_id") or "")
        content = row.get("content")
        if content is None:
            content = row.get("text")
        if chunk_id and content is not None:
            chunks[chunk_id] = str(content)
    if not chunks:
        raise ValueError("No chunks loaded from frozen corpus.")
    return chunks


def load_existing_results() -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    if not RESULTS_PATH.exists():
        return results
    for row in read_jsonl(RESULTS_PATH):
        example_id = str(row.get("example_id") or "")
        if example_id:
            results[example_id] = row
    return results


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def parse_json_object(content: str) -> dict[str, Any]:
    text = (content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Judge did not return a JSON object: {text[:500]}")
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("Judge response JSON was not an object.")
    return value


def build_messages(example: dict[str, Any], full_chunk: str) -> list[dict[str, str]]:
    system = """You are a strict evaluator of instruction-tuning examples for a technical-document QA dataset.

Evaluate only against the supplied source chunk. Do not use outside knowledge to repair, supplement, or reinterpret the source. Preserve the source's terminology and level of detail.

Check:
1. relevance: does the response answer the instruction?
2. faithfulness: is every factual claim supported by the source chunk?
3. correctness: is the response accurate relative to the source?
4. completeness: does it cover the requested information without material omissions?
5. evidence: does the supplied evidence span support the response?

A response may be concise. Do not penalize harmless paraphrasing. Penalize unsupported additions, contradictions, misleading current-practice implications, and answers that fail to answer the question.

Return ONLY one JSON object with these exact keys:
{
  "faithfulness": integer 1-5,
  "relevant": boolean,
  "unsupported_claims": boolean,
  "correct": boolean,
  "complete": boolean,
  "evidence_supports_answer": boolean,
  "verdict": "ACCEPT" or "REJECT",
  "reason": "brief explanation"
}

Use score 4 or 5 only when the answer is well supported. Verdict must be REJECT if faithfulness is below 4, relevant is false, unsupported_claims is true, correct is false, complete is false, or evidence_supports_answer is false. Otherwise ACCEPT."""

    user = {
        "instruction": example.get("instruction", ""),
        "response": example.get("response", ""),
        "evidence_span": example.get("evidence_span", ""),
        "full_source_chunk": full_chunk,
        "task_type": example.get("task_type", ""),
        "chunk_id": example.get("chunk_id", ""),
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def call_judge(
    session: requests.Session,
    api_key: str,
    messages: list[dict[str, str]],
    max_retries: int = 5,
) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0,
        "max_completion_tokens": 1200,
        "reasoning_effort": "low",
        "response_format": {"type": "json_object"},
    }

    for attempt in range(max_retries + 1):
        try:
            response = session.post(
                API_URL,
                headers=headers,
                json=payload,
                timeout=180,
            )
        except requests.RequestException as exc:
            if attempt >= max_retries:
                raise RuntimeError(f"Network request failed after retries: {exc}") from exc
            time.sleep(min(10 * (attempt + 1), 60))
            continue

        if response.status_code == 429:
            if attempt >= max_retries:
                raise RuntimeError(f"Groq rate limit persisted: {response.text[:1000]}")
            retry_after = response.headers.get("Retry-After")
            try:
                wait = max(float(retry_after), 20.0) if retry_after else 30.0 * (attempt + 1)
            except ValueError:
                wait = 30.0 * (attempt + 1)
            print(f"  Rate limited; waiting {wait:.0f}s before retry.")
            time.sleep(min(wait, 180))
            continue

        if response.status_code >= 500:
            if attempt >= max_retries:
                raise RuntimeError(
                    f"Groq server error {response.status_code}: {response.text[:1000]}"
                )
            time.sleep(min(15 * (attempt + 1), 90))
            continue

        if response.status_code >= 400:
            raise RuntimeError(
                f"Groq API error {response.status_code}: {response.text[:1500]}"
            )

        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"Groq returned no choices: {json.dumps(data)[:1000]}")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not content:
            raise RuntimeError(
                "Groq returned empty content. "
                f"finish_reason={choices[0].get('finish_reason')}; "
                f"response={json.dumps(data)[:1200]}"
            )
        return parse_json_object(content)

    raise RuntimeError("Judge request exhausted retry loop.")


def validate_judgment(value: dict[str, Any]) -> dict[str, Any]:
    score = value.get("faithfulness")
    if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 5:
        raise ValueError(f"Invalid faithfulness score: {score!r}")

    bool_fields = (
        "relevant", "unsupported_claims", "correct",
        "complete", "evidence_supports_answer",
    )
    for field in bool_fields:
        if not isinstance(value.get(field), bool):
            raise ValueError(f"Invalid/missing boolean field {field}: {value.get(field)!r}")

    verdict = str(value.get("verdict", "")).upper()
    if verdict not in {"ACCEPT", "REJECT"}:
        raise ValueError(f"Invalid verdict: {verdict!r}")

    # Enforce the project's strict acceptance rule locally.
    expected = (
        score >= 4
        and value["relevant"]
        and not value["unsupported_claims"]
        and value["correct"]
        and value["complete"]
        and value["evidence_supports_answer"]
    )
    enforced_verdict = "ACCEPT" if expected else "REJECT"
    return {
        "faithfulness": score,
        "relevant": value["relevant"],
        "unsupported_claims": value["unsupported_claims"],
        "correct": value["correct"],
        "complete": value["complete"],
        "evidence_supports_answer": value["evidence_supports_answer"],
        "model_verdict": verdict,
        "verdict": enforced_verdict,
        "reason": str(value.get("reason", ""))[:2000],
        "verdict_overridden_by_rule": verdict != enforced_verdict,
    }


def write_summary(input_count: int, results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    completed = list(results.values())
    accepted = [r for r in completed if r.get("verdict") == "ACCEPT"]
    rejected = [r for r in completed if r.get("verdict") == "REJECT"]
    scores = [r["faithfulness"] for r in completed if isinstance(r.get("faithfulness"), int)]
    summary = {
        "input_examples": input_count,
        "judged_examples": len(completed),
        "pending_examples": max(input_count - len(completed), 0),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "mean_faithfulness": round(sum(scores) / len(scores), 3) if scores else None,
        "minimum_faithfulness": min(scores) if scores else None,
        "judge_model": MODEL,
        "generator_model": GENERATOR_MODEL,
        "different_model_families": True,
        "acceptance_rule": (
            "faithfulness >= 4 AND relevant AND correct AND complete AND "
            "evidence_supports_answer AND NOT unsupported_claims"
        ),
        "resumable": True,
        "source_datasets_modified": False,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delay", type=float, default=20.0)
    parser.add_argument("--limit", type=int, default=0,
                        help="Optional maximum new examples to judge; 0 means all pending.")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env", override=True)
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY is missing from the project .env file.")
        return 2

    examples = read_jsonl(INPUT_PATH)
    if not examples:
        raise ValueError(f"Judge input is empty: {INPUT_PATH}")
    ids = [str(row.get("example_id") or "") for row in examples]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("Judge input contains missing or duplicate example_id values.")

    chunks = load_chunks()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = load_existing_results()

    pending = [
        row for row in examples
        if str(row.get("example_id") or "") not in existing
    ]
    if args.limit > 0:
        pending = pending[:args.limit]

    print(f"Input examples: {len(examples)}")
    print(f"Already judged: {len(existing)}")
    print(f"Pending this run: {len(pending)}")
    print(f"Judge: {MODEL}")
    print(f"Delay between requests: {args.delay:g}s")
    print("Results are checkpointed after each example.")

    session = requests.Session()
    errors_this_run = 0

    for index, example in enumerate(pending, 1):
        example_id = str(example.get("example_id") or "")
        chunk_id = str(example.get("chunk_id") or "")
        full_chunk = chunks.get(chunk_id)
        if full_chunk is None:
            error = {
                "example_id": example_id,
                "error": f"Chunk not found in frozen corpus: {chunk_id}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            append_jsonl(ERRORS_PATH, error)
            errors_this_run += 1
            print(f"[{index}/{len(pending)}] {example_id}: MISSING CHUNK")
            continue

        try:
            raw_judgment = call_judge(
                session,
                api_key,
                build_messages(example, full_chunk),
            )
            judgment = validate_judgment(raw_judgment)
            result = {
                "example_id": example_id,
                "chunk_id": chunk_id,
                "task_type": example.get("task_type"),
                "faithfulness": judgment["faithfulness"],
                "relevant": judgment["relevant"],
                "unsupported_claims": judgment["unsupported_claims"],
                "correct": judgment["correct"],
                "complete": judgment["complete"],
                "evidence_supports_answer": judgment["evidence_supports_answer"],
                "model_verdict": judgment["model_verdict"],
                "verdict": judgment["verdict"],
                "reason": judgment["reason"],
                "verdict_overridden_by_rule": judgment["verdict_overridden_by_rule"],
                "judge_model": MODEL,
                "generator_model": example.get("generator_model", GENERATOR_MODEL),
                "judged_at": datetime.now(timezone.utc).isoformat(),
            }
            append_jsonl(RESULTS_PATH, result)
            existing[example_id] = result
            print(
                f"[{index}/{len(pending)}] {example_id}: "
                f"{result['verdict']} ({result['faithfulness']}/5) — "
                f"{result['reason'][:180]}"
            )
        except Exception as exc:
            error = {
                "example_id": example_id,
                "error": f"{type(exc).__name__}: {exc}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            append_jsonl(ERRORS_PATH, error)
            errors_this_run += 1
            print(f"[{index}/{len(pending)}] {example_id}: ERROR — {exc}")
            # Stop on API/quota errors so the user can resume later without
            # burning through repeated failures. Other records remain pending.
            message = str(exc).lower()
            if "429" in message or "rate limit" in message or "quota" in message:
                print("Stopping on rate-limit/quota error. Rerun after reset to resume.")
                break

        if index < len(pending):
            time.sleep(max(args.delay, 0))

    summary = write_summary(len(examples), existing)
    summary["errors_this_run"] = errors_this_run
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\nV2 GROQ JUDGE SUMMARY")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nResults: {RESULTS_PATH.relative_to(ROOT)}")
    print(f"Summary: {SUMMARY_PATH.relative_to(ROOT)}")
    print("Source datasets and frozen corpus were not modified.")

    return 0 if summary["pending_examples"] == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nJUDGE FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
