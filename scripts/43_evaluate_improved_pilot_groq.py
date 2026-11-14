#!/usr/bin/env python3
"""
Independent Groq judge for the improved-generation pilot.

Reads:
  data/interim/improved_generation/dataset_v0_additional.jsonl
  data/frozen/chunks_v1_frozen.jsonl

Writes only:
  data/interim/improved_generation/groq_pilot_heuristic_results.jsonl
  data/interim/improved_generation/groq_pilot_judge_results.jsonl
  data/interim/improved_generation/groq_pilot_evaluation_summary.json
  data/interim/improved_generation/groq_pilot_evaluation_failures.jsonl

Does not modify dataset_v0_raw.jsonl, the additional dataset, old evaluation
outputs, or any generation checkpoints.
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "interim" / "improved_generation" / "dataset_v0_additional.jsonl"
CHUNKS_FILE = ROOT / "data" / "frozen" / "chunks_v1_frozen.jsonl"
OUT_DIR = ROOT / "data" / "interim" / "improved_generation"

HEURISTIC_FILE = OUT_DIR / "groq_pilot_heuristic_results.jsonl"
JUDGE_FILE = OUT_DIR / "groq_pilot_judge_results.jsonl"
SUMMARY_FILE = OUT_DIR / "groq_pilot_evaluation_summary.json"
FAILURE_FILE = OUT_DIR / "groq_pilot_evaluation_failures.jsonl"

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
JUDGE_MODEL = "openai/gpt-oss-120b"
MIN_FAITHFULNESS = 4
DELAY_SECONDS = 20
MAX_RETRIES = 4

JUDGE_SYSTEM = """You are an independent, strict evaluator of technical IoT
instruction-response training examples.

Evaluate ONLY against the supplied source chunk. Do not use outside knowledge.
The evidence span is the evidence selected by the generator. The full source
chunk is supplied only to help you determine whether the answer is factually
supported by the source and whether the selected evidence is sufficient.

Return exactly one JSON object with these keys:
{
  "faithfulness_score": integer from 1 to 5,
  "evidence_relevant": boolean,
  "unsupported_claims": boolean,
  "answer_correct_from_source": boolean,
  "answer_complete_from_source": boolean,
  "issues": [short strings],
  "reason": "brief explanation"
}

Scoring:
5 = fully correct, precise, complete enough, and supported by selected evidence.
4 = correct and supported, with only minor non-material omissions or wording issues.
3 = partially correct or has a material evidence gap / unsupported detail.
2 = substantially incorrect or materially unsupported.
1 = unrelated, fabricated, or unusable.

Rules:
- Mark unsupported_claims=true for ANY material claim not supported by the full source.
- evidence_relevant=true only if the selected evidence directly supports the complete
  answer, not merely the general topic.
- answer_correct_from_source=true only if all answer claims are correct per source.
- answer_complete_from_source=false if the answer omits important information
  explicitly requested by the instruction.
- Do not penalize a concise answer for omitting irrelevant source details.
- Be strict about numbers, units, standards, directions, protocol behavior, and
  causal explanations.
"""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"Invalid JSON in {path}, line {line_no}: {exc}")
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def first(row: dict[str, Any], keys: tuple[str, ...], default=None):
    for key in keys:
        if row.get(key) is not None:
            return row[key]
    return default


def chunk_id(row: dict[str, Any]) -> str:
    value = first(row, ("chunk_id", "id", "chunkId"))
    return str(value) if value is not None else ""


def chunk_text(row: dict[str, Any]) -> str:
    return str(first(row, ("text", "content", "chunk_text", "page_content"), ""))


def evidence_blocks(source: str) -> list[dict[str, Any]]:
    found = []
    pattern = re.compile(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", re.DOTALL)
    for match in pattern.finditer(source):
        value = match.group(0)
        if len(value.strip()) >= 30:
            found.append({"id": len(found) + 1, "text": value})
    return found


def parse_json_response(content: str) -> dict[str, Any]:
    content = content.strip()
    content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
    content = re.sub(r"\s*```$", "", content)
    start, end = content.find("{"), content.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"Judge returned no JSON object: {content[:1000]}")
    return json.loads(content[start:end + 1])


def call_judge(session: requests.Session, api_key: str, payload: dict) -> dict:
    response = session.post(
        GROQ_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=180,
    )
    if response.status_code == 429:
        detail = response.text[:2000]
        retry_after = response.headers.get("retry-after")
        raise RuntimeError(f"HTTP_429:{retry_after or ''}:{detail}")
    response.raise_for_status()
    data = response.json()
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content")
    if not content:
        raise ValueError(
            f"Judge returned empty content. finish_reason={choice.get('finish_reason')}; "
            f"response={json.dumps(data, ensure_ascii=False)[:1500]}"
        )
    return parse_json_response(str(content))


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delay", type=float, default=DELAY_SECONDS)
    parser.add_argument("--limit", type=int, default=0,
                        help="Optional maximum number of examples; 0 means all.")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env", override=True)
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("GROQ_API_KEY is missing from .env")

    examples = read_jsonl(INPUT_FILE)
    chunks = read_jsonl(CHUNKS_FILE)
    if not examples:
        raise SystemExit(f"No examples found: {INPUT_FILE}")
    chunk_map = {chunk_id(row): row for row in chunks}
    if args.limit > 0:
        examples = examples[:args.limit]

    # Structural heuristic results are separate from semantic judge decisions.
    heuristic_rows = read_jsonl(HEURISTIC_FILE)
    heuristic_ids = {str(x.get("example_id")) for x in heuristic_rows}
    judge_rows = read_jsonl(JUDGE_FILE)
    completed = {str(x.get("example_id")) for x in judge_rows if x.get("status") == "completed"}

    session = requests.Session()
    completed_this_run = 0
    failed_this_run = 0

    for example in tqdm(examples, desc="Groq pilot evaluation", unit="example"):
        eid = str(example.get("example_id", ""))
        if not eid:
            append_jsonl(FAILURE_FILE, {
                "example_id": None, "error": "missing_example_id",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            failed_this_run += 1
            continue

        if eid not in heuristic_ids:
            cid = str(example.get("chunk_id", ""))
            source_row = chunk_map.get(cid)
            heuristic_issues = []
            evidence_span = str(example.get("evidence_span", ""))
            if source_row is None:
                heuristic_issues.append("chunk_id_not_found")
                source = ""
            else:
                source = chunk_text(source_row)
                if not evidence_span or evidence_span not in source:
                    heuristic_issues.append("evidence_span_not_exact_source_substring")
            if not example.get("instruction"):
                heuristic_issues.append("empty_instruction")
            if not example.get("response"):
                heuristic_issues.append("empty_response")
            heuristic = {
                "example_id": eid,
                "passed": len(heuristic_issues) == 0,
                "issues": heuristic_issues,
                "checked_at": datetime.now(timezone.utc).isoformat(),
            }
            append_jsonl(HEURISTIC_FILE, heuristic)
            heuristic_rows.append(heuristic)
            heuristic_ids.add(eid)
        else:
            heuristic = next(x for x in heuristic_rows if str(x.get("example_id")) == eid)
            source_row = chunk_map.get(str(example.get("chunk_id", "")))
            source = chunk_text(source_row) if source_row else ""

        if not heuristic.get("passed"):
            if eid not in completed:
                result = {
                    "example_id": eid,
                    "status": "rejected_heuristic",
                    "accepted": False,
                    "heuristic_issues": heuristic.get("issues", []),
                    "judge_model": JUDGE_MODEL,
                    "evaluated_at": datetime.now(timezone.utc).isoformat(),
                }
                append_jsonl(JUDGE_FILE, result)
                judge_rows.append(result)
                completed.add(eid)
            continue

        if eid in completed:
            continue

        prompt = {
            "example_id": eid,
            "task_type": example.get("task_type"),
            "instruction": example.get("instruction"),
            "response": example.get("response"),
            "selected_evidence_span": example.get("evidence_span"),
            "full_source_chunk": source,
        }
        payload = {
            "model": JUDGE_MODEL,
            "temperature": 0,
            "max_completion_tokens": 1200,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            "response_format": {"type": "json_object"},
        }

        judge = None
        last_error = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                judge = call_judge(session, api_key, payload)
                break
            except Exception as exc:
                last_error = str(exc)
                if "HTTP_429:" in last_error:
                    # Respect server hint if present; otherwise back off.
                    match = re.match(r"HTTP_429:([^:]*):", last_error)
                    wait = float(match.group(1)) if match and match.group(1).isdigit() else min(120, 30 * (attempt + 1))
                else:
                    wait = min(30, 5 * (attempt + 1))
                if attempt < MAX_RETRIES:
                    time.sleep(wait)

        if judge is None:
            append_jsonl(FAILURE_FILE, {
                "example_id": eid,
                "error": last_error,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            failed_this_run += 1
            # Avoid burning quota on a systemic provider failure.
            if last_error and "HTTP_429:" in last_error:
                print("\nGroq returned HTTP 429. Stopping to preserve quota.")
                break
            continue

        try:
            score = int(judge.get("faithfulness_score"))
            evidence_relevant = judge.get("evidence_relevant") is True
            unsupported = judge.get("unsupported_claims") is True
            correct = judge.get("answer_correct_from_source") is True
            complete = judge.get("answer_complete_from_source") is True
            accepted = (
                score >= MIN_FAITHFULNESS
                and evidence_relevant
                and not unsupported
                and correct
                and complete
            )
            result = {
                "example_id": eid,
                "status": "completed",
                "faithfulness_score": score,
                "evidence_relevant": evidence_relevant,
                "unsupported_claims": unsupported,
                "answer_correct_from_source": correct,
                "answer_complete_from_source": complete,
                "issues": judge.get("issues", []),
                "reason": judge.get("reason", ""),
                "accepted": accepted,
                "judge_model": JUDGE_MODEL,
                "evaluated_at": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as exc:
            append_jsonl(FAILURE_FILE, {
                "example_id": eid,
                "error": f"invalid_judge_schema: {exc}",
                "judge_output": judge,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            failed_this_run += 1
            continue

        append_jsonl(JUDGE_FILE, result)
        judge_rows.append(result)
        completed.add(eid)
        completed_this_run += 1
        time.sleep(args.delay)

    # Recompute summary from persisted files so reruns are resumable.
    all_judged = read_jsonl(JUDGE_FILE)
    completed_rows = [x for x in all_judged if x.get("status") == "completed"]
    accepted_rows = [x for x in completed_rows if x.get("accepted") is True]
    rejected_rows = [
        x for x in all_judged
        if x.get("accepted") is False or x.get("status") == "rejected_heuristic"
    ]
    scores = [int(x["faithfulness_score"]) for x in completed_rows if "faithfulness_score" in x]
    summary = {
        "input_examples": len(examples),
        "heuristic_records": len(read_jsonl(HEURISTIC_FILE)),
        "judge_records": len(all_judged),
        "judge_completed": len(completed_rows),
        "judge_accepted": len(accepted_rows),
        "judge_rejected": len(rejected_rows),
        "judge_pending": max(0, len(examples) - len(all_judged)),
        "mean_faithfulness_score": round(sum(scores) / len(scores), 3) if scores else None,
        "minimum_faithfulness_score": MIN_FAITHFULNESS,
        "judge_model": JUDGE_MODEL,
        "completed_this_run": completed_this_run,
        "failed_this_run": failed_this_run,
        "raw_dataset_modified": False,
        "additional_dataset_modified": False,
        "evaluation_file": str(JUDGE_FILE.relative_to(ROOT)),
    }
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nEVALUATION SUMMARY")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["judge_pending"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())