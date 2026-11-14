from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "interim" / "production_filter" / "repair_v2" / "repair_candidates.jsonl"
FROZEN_CHUNKS_FILE = ROOT / "data" / "frozen" / "chunks_v1_frozen.jsonl"
OUTPUT_DIR = ROOT / "data" / "interim" / "production_filter" / "repair_v2" / "judge"
RESULTS_FILE = OUTPUT_DIR / "hma_repair_judge_results.jsonl"
SUMMARY_FILE = OUTPUT_DIR / "hma_repair_judge_summary.json"

TARGET_ID = "production_batch_013_example_03"
MODEL = "openai/gpt-oss-120b"
ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return records


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def extract_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Judge did not return a JSON object: {text[:1000]}")
        return json.loads(match.group(0))


def main() -> int:
    load_dotenv(ROOT / ".env", override=True)
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY was not found in the project's .env file.")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing repaired candidate file: {INPUT_FILE}")

    candidates = read_jsonl(INPUT_FILE)
    matches = [x for x in candidates if x.get("example_id") == TARGET_ID]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {TARGET_ID}; found {len(matches)}.")

    item = matches[0]
    evidence = item.get("evidence_span", "")
    source = item.get("full_source_chunk") or item.get("source_chunk") or ""

    if not evidence.strip():
        raise RuntimeError("Selected evidence span is empty.")

    # The candidate file intentionally stores the selected evidence, not the
    # complete source chunk. Retrieve the latter from the immutable frozen corpus.
    if not source.strip():
        if not FROZEN_CHUNKS_FILE.exists():
            raise FileNotFoundError(f"Missing frozen corpus: {FROZEN_CHUNKS_FILE}")
        chunk_id = item.get("chunk_id")
        if not chunk_id:
            raise RuntimeError("Candidate has no chunk_id; cannot retrieve its frozen source chunk.")
        frozen_chunks = read_jsonl(FROZEN_CHUNKS_FILE)
        source_matches = [
            chunk for chunk in frozen_chunks
            if chunk.get("chunk_id") == chunk_id or chunk.get("id") == chunk_id
        ]
        if len(source_matches) != 1:
            raise RuntimeError(
                f"Expected exactly one frozen chunk for {chunk_id}; found {len(source_matches)}."
            )
        source = source_matches[0].get("content", "")
        if not source.strip():
            raise RuntimeError(f"Frozen chunk {chunk_id} has no non-empty 'content' field.")

    prompt = f"""
You are an independent evaluator of a source-grounded IoT technical-document instruction/response example.

Evaluate ONLY whether the response is fully supported by the SELECTED EVIDENCE. The full source chunk is supplied for context, but claims that appear only in the full source and not in the selected evidence must be treated as unsupported.

Return exactly one JSON object with these fields:
{{
  "faithfulness_score": integer from 1 to 5,
  "evidence_relevant": boolean,
  "unsupported_claims": boolean,
  "answer_correct_from_source": boolean,
  "answer_complete_from_source": boolean,
  "issues": [string, ...],
  "reason": "brief explanation"
}}

Scoring:
5 = fully correct, complete, and directly supported by selected evidence.
4 = correct and supported, with only a minor omission or wording issue.
3 = partly supported or materially incomplete.
2 = substantial unsupported or incorrect content.
1 = mostly unsupported or incorrect.

Be strict. Do not reward facts merely because they appear somewhere in the full source chunk. They must be present in the selected evidence.

EXAMPLE ID:
{TARGET_ID}

INSTRUCTION:
{item.get("instruction", "")}

RESPONSE:
{item.get("response", "")}

SELECTED EVIDENCE:
{evidence}

FULL SOURCE CHUNK (context only; not a substitute for selected evidence):
{source}
""".strip()

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a strict evidence-grounding evaluator. Return valid JSON only."
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": 1200,
        "response_format": {"type": "json_object"},
    }

    response = requests.post(
        ENDPOINT,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=180,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Groq API returned HTTP {response.status_code}: {response.text[:2000]}"
        )

    body = response.json()
    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError(f"Groq returned no choices: {json.dumps(body)[:2000]}")

    message = choices[0].get("message") or {}
    content = message.get("content")
    if not content:
        raise RuntimeError(
            f"Groq returned empty content. finish_reason={choices[0].get('finish_reason')}; "
            f"usage={body.get('usage')}"
        )

    judged = extract_json_object(content)
    score = int(judged.get("faithfulness_score", 0))
    evidence_relevant = judged.get("evidence_relevant") is True
    unsupported = judged.get("unsupported_claims") is True
    correct = judged.get("answer_correct_from_source") is True
    complete = judged.get("answer_complete_from_source") is True

    accepted = (
        1 <= score <= 5
        and score >= 4
        and evidence_relevant
        and not unsupported
        and correct
        and complete
    )

    result = {
        "example_id": TARGET_ID,
        "status": "completed",
        "faithfulness_score": score,
        "evidence_relevant": evidence_relevant,
        "unsupported_claims": unsupported,
        "answer_correct_from_source": correct,
        "answer_complete_from_source": complete,
        "issues": judged.get("issues", []),
        "reason": judged.get("reason", ""),
        "accepted": accepted,
        "judge_model": MODEL,
        "finish_reason": choices[0].get("finish_reason"),
        "usage": body.get("usage", {}),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "evidence_block_ids": item.get("evidence_block_ids", []),
        "chunk_id": item.get("chunk_id"),
        "source_retrieved_from_frozen_corpus": True,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(
        json.dumps(result, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    summary = {
        "input_examples": 1,
        "judge_completed": 1,
        "judge_accepted": int(accepted),
        "judge_rejected": int(not accepted),
        "mean_faithfulness_score": score,
        "judge_model": MODEL,
        "target_example_id": TARGET_ID,
        "evidence_block_ids": item.get("evidence_block_ids", []),
        "chunk_id": item.get("chunk_id"),
        "source_file": str(FROZEN_CHUNKS_FILE.relative_to(ROOT)),
        "acceptance_rule": (
            "score >= 4 AND evidence_relevant AND NOT unsupported_claims "
            "AND answer_correct_from_source AND answer_complete_from_source"
        ),
        "result_file": str(RESULTS_FILE.relative_to(ROOT)),
        "raw_dataset_modified": False,
        "frozen_source_modified": False,
    }
    write_json(SUMMARY_FILE, summary)

    print(json.dumps({"result": result, "summary": summary}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise