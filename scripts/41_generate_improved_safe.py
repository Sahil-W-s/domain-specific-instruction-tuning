#!/usr/bin/env python3
"""
Improved, independently checkpointed Phase 2 instruction-data generator.

Default behavior is a controlled pilot:
  - first 12 eligible frozen chunks
  - 3 examples per chunk: QA, extraction, classification
  - writes ONLY to data/interim/improved_generation/
  - never edits dataset_v0_raw.jsonl or production_checkpoints/

After reviewing the pilot, scale with --all-chunks. The generator uses
OpenRouter's chat-completions API and the existing Nemotron model family.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
FROZEN_CHUNKS = ROOT / "data" / "frozen" / "chunks_v1_frozen.jsonl"
EXISTING_RAW = ROOT / "data" / "interim" / "dataset_v0_raw.jsonl"

OUTPUT_DIR = ROOT / "data" / "interim" / "improved_generation"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
OUTPUT_JSONL = OUTPUT_DIR / "dataset_v0_additional.jsonl"
FAILURES_JSONL = OUTPUT_DIR / "generation_failures.jsonl"
MANIFEST_JSON = OUTPUT_DIR / "run_manifest.json"

MODEL = "nvidia/nemotron-3.5-super-vl:free"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

TASKS = ("qa", "extraction", "classification")
TASK_INSTRUCTIONS = {
    "qa": (
        "Create a precise question-answering item. Ask one answerable question "
        "about a specific technical fact, mechanism, component, comparison, or "
        "relationship explicitly stated in the source."
    ),
    "extraction": (
        "Create an information-extraction item. Ask the model to extract a "
        "well-defined list, set of fields, sequence, parameter, protocol detail, "
        "or named entities from the source. The response must contain only "
        "information explicitly present in the cited evidence."
    ),
    "classification": (
        "Create a classification item. Ask the model to classify a named "
        "technology, protocol, component, architecture, or scenario using "
        "categories or criteria explicitly defined in the source. The evidence "
        "must state the classification rule or the facts needed to apply it."
    ),
}

SYSTEM_PROMPT = """You create high-quality instruction-response training examples
from technical IoT documentation.

Return exactly one valid JSON object and no markdown or surrounding prose.
Required keys:
- instruction: a self-contained user instruction/question
- response: a concise, technically accurate answer
- evidence_block_ids: a JSON array of one or more integer paragraph-block IDs

Strict rules:
1. Use ONLY facts explicitly stated in the supplied source chunk.
2. The answer must be fully supported by the selected evidence blocks, not merely
   by other parts of the chunk. Select every block needed to support every claim.
3. Do not add explanations, implications, examples, definitions, units, standards,
   ranges, or causal claims that the selected evidence does not state.
4. Do not use outside knowledge to fill gaps or correct the source.
5. Do not ask broad questions that require combining unrelated sections.
6. Avoid trivial questions, vague prompts, yes/no questions, and questions whose
   answer is simply copied from the instruction.
7. Keep the response direct and proportionate to the question.
8. Evidence block IDs must be copied exactly from the supplied numbered blocks.
9. Do not invent missing details. If the source does not support a strong example
   of the requested task type, return {"skip": true, "reason": "..."}.
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
                    raise ValueError(f"Invalid JSON in {path} line {line_no}: {exc}")
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_first(record: dict[str, Any], names: tuple[str, ...], default=None):
    for name in names:
        value = record.get(name)
        if value is not None:
            return value
    return default


def chunk_id_of(row: dict[str, Any]) -> str:
    value = get_first(row, ("chunk_id", "id", "chunkId"))
    if value is None:
        raise ValueError(f"Chunk record has no chunk_id/id field: {list(row)[:15]}")
    return str(value)


def chunk_text_of(row: dict[str, Any]) -> str:
    value = get_first(row, ("text", "content", "chunk_text", "page_content"))
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Chunk {chunk_id_of(row)} has no non-empty text/content field. "
            f"Available keys: {list(row)}"
        )
    return value.strip()


def extract_evidence_blocks(source: str) -> list[dict[str, Any]]:
    blocks = []
    pattern = re.compile(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", re.DOTALL)
    for match in pattern.finditer(source):
        block = match.group(0)
        if len(block.strip()) >= 30:
            blocks.append({"id": len(blocks) + 1, "text": block})
    return blocks


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def similarity(a: str, b: str) -> float:
    na, nb = normalize(a), normalize(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    return SequenceMatcher(None, na, nb).ratio()


def existing_instruction_index(rows: list[dict[str, Any]]) -> list[tuple[str, str]]:
    result = []
    for row in rows:
        instruction = str(row.get("instruction", "")).strip()
        if instruction:
            result.append((str(row.get("example_id", "existing")), instruction))
    return result


def call_model(
    session: requests.Session,
    api_key: str,
    model: str,
    chunk_id: str,
    source_doc: str,
    task: str,
    blocks: list[dict[str, Any]],
    prior_instructions: list[str],
    timeout: int,
    max_tokens: int,
) -> dict[str, Any]:
    numbered = "\n\n".join(
        f"[BLOCK {b['id']}]\n{b['text']}" for b in blocks
    )
    prior = "\n".join(f"- {q}" for q in prior_instructions[-100:])
    user_prompt = f"""SOURCE DOCUMENT: {source_doc}
CHUNK ID: {chunk_id}
REQUIRED TASK TYPE: {task.upper()}

TASK:
{TASK_INSTRUCTIONS[task]}

PREVIOUSLY GENERATED INSTRUCTIONS (avoid the same fact, topic, and wording):
{prior if prior else "(none)"}

NUMBERED SOURCE EVIDENCE BLOCKS:
{numbered}

Create one strong {task.upper()} example following the system rules.
Return evidence_block_ids that together directly support the complete response.
If this chunk cannot support a good item of this task type, return skip=true.
"""
    payload = {
        "model": model,
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }
    response = session.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://openrouter.ai/",
            "X-Title": "IoT Technical Dataset Generation",
        },
        json=payload,
        timeout=timeout,
    )
    if response.status_code == 429:
        try:
            detail = response.json()
        except ValueError:
            detail = response.text[:2000]
        raise RuntimeError(
            "HTTP_429_RATE_LIMIT: " + json.dumps(detail, ensure_ascii=False)[:3000]
        )
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    if isinstance(content, list):
        content = "".join(
            item.get("text", "") for item in content if isinstance(item, dict)
        )
    content = str(content).strip()
    content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
    content = re.sub(r"\s*```$", "", content)
    start, end = content.find("{"), content.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"Model did not return a JSON object: {content[:500]}")
    return json.loads(content[start : end + 1])


def validate_candidate(
    candidate: dict[str, Any],
    blocks: list[dict[str, Any]],
    chunk_text: str,
) -> tuple[bool, str, str]:
    if candidate.get("skip") is True:
        return False, "model_skip", str(candidate.get("reason", "unsupported task"))
    instruction = candidate.get("instruction")
    answer = candidate.get("response")
    ids = candidate.get("evidence_block_ids")
    if not isinstance(instruction, str) or len(instruction.strip()) < 15:
        return False, "invalid_instruction", ""
    if not isinstance(answer, str) or len(answer.strip()) < 5:
        return False, "invalid_response", ""
    if not isinstance(ids, list) or not ids:
        return False, "missing_evidence_ids", ""
    valid = {b["id"]: b["text"] for b in blocks}
    try:
        ids = list(dict.fromkeys(int(x) for x in ids))
    except (ValueError, TypeError):
        return False, "invalid_evidence_ids", ""
    if any(x not in valid for x in ids):
        return False, "unknown_evidence_id", ""
    evidence_span = "\n\n".join(valid[x] for x in ids)
    # Ensure selected evidence is genuinely drawn from the frozen source.
    if not evidence_span or evidence_span not in chunk_text:
        return False, "evidence_reconstruction_failed", ""
    if len(instruction) > 1200 or len(answer) > 3000:
        return False, "length_limit", ""
    return True, "", evidence_span


def load_completed_ids() -> set[str]:
    completed = set()
    for path in CHECKPOINT_DIR.glob("*.json"):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            if obj.get("status") == "complete":
                completed.add(str(obj["slot_id"]))
        except Exception:
            continue
    return completed


def append_failure(record: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with FAILURES_JSONL.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-chunks", type=int, default=12,
                        help="Pilot size; default 12. Use --all-chunks to scale.")
    parser.add_argument("--all-chunks", action="store_true",
                        help="Process all eligible chunks (after pilot approval).")
    parser.add_argument("--include-used-chunks", action="store_true",
                        help="Also use chunks already represented in the existing raw/additional datasets.")
    parser.add_argument("--start-index", type=int, default=0,
                        help="Zero-based starting index in the frozen chunk list.")
    parser.add_argument("--delay", type=float, default=12.0,
                        help="Seconds between successful model requests.")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-tokens", type=int, default=2200)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--similarity-threshold", type=float, default=0.88)
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()

    load_dotenv(ROOT / ".env", override=True)
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY is missing from the project .env file.")
        return 2
    if not FROZEN_CHUNKS.exists():
        print(f"ERROR: Frozen chunk file not found: {FROZEN_CHUNKS}")
        return 2

    chunks = read_jsonl(FROZEN_CHUNKS)
    if not chunks:
        print("ERROR: Frozen chunk file is empty.")
        return 2

    old_rows = read_jsonl(EXISTING_RAW)
    generated_rows = read_jsonl(OUTPUT_JSONL)
    already_used_chunks = {
        str(row.get("chunk_id")) for row in old_rows + generated_rows
        if row.get("chunk_id") is not None
    }
    eligible_chunks = chunks if args.include_used_chunks else [
        row for row in chunks if chunk_id_of(row) not in already_used_chunks
    ]

    end = len(eligible_chunks) if args.all_chunks else min(
        len(eligible_chunks), args.start_index + max(1, args.limit_chunks)
    )
    selected = eligible_chunks[max(0, args.start_index):end]
    if not selected:
        print("ERROR: No chunks selected. Check --start-index.")
        return 2

    seen = existing_instruction_index(old_rows + generated_rows)
    completed = load_completed_ids()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Capture the frozen input hash in the run manifest.
    frozen_hash = hashlib.sha256(FROZEN_CHUNKS.read_bytes()).hexdigest()
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "frozen_chunks": str(FROZEN_CHUNKS.relative_to(ROOT)),
        "frozen_chunks_sha256": frozen_hash,
        "existing_raw_examples_read_only": len(old_rows),
        "previous_additional_examples": len(generated_rows),
        "eligible_chunk_count_after_excluding_used_chunks": len(eligible_chunks),
        "selected_chunk_count": len(selected),
        "selected_chunk_ids": [chunk_id_of(x) for x in selected],
        "include_used_chunks": args.include_used_chunks,
        "tasks_per_chunk": list(TASKS),
        "output": str(OUTPUT_JSONL.relative_to(ROOT)),
        "checkpoint_dir": str(CHECKPOINT_DIR.relative_to(ROOT)),
    }
    MANIFEST_JSON.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    session = requests.Session()
    added = 0
    skipped = 0
    rejected = 0
    errors = 0

    total = len(selected) * len(TASKS)
    progress = tqdm(total=total, desc="Improved generation", unit="item")

    for chunk_position, chunk in enumerate(selected, start=max(0, args.start_index)):
        cid = chunk_id_of(chunk)
        text = chunk_text_of(chunk)
        source_doc = str(get_first(
            chunk, ("source_doc", "document_id", "doc_id", "source"), "unknown"
        ))
        blocks = extract_evidence_blocks(text)
        if not blocks:
            for task in TASKS:
                slot_id = f"{cid}__{task}"
                checkpoint = CHECKPOINT_DIR / f"{slot_id}.json"
                checkpoint.write_text(json.dumps({
                    "slot_id": slot_id, "status": "skipped",
                    "reason": "no_evidence_blocks",
                }, indent=2), encoding="utf-8")
                skipped += 1
                progress.update(1)
            continue

        for task in TASKS:
            slot_id = f"{cid}__{task}"
            checkpoint = CHECKPOINT_DIR / f"{slot_id}.json"
            if slot_id in completed or checkpoint.exists():
                progress.update(1)
                continue

            candidate = None
            last_error = None
            for attempt in range(args.max_retries + 1):
                try:
                    prior = [q for _, q in seen]
                    candidate = call_model(
                        session, api_key, args.model, cid, source_doc, task,
                        blocks, prior, args.timeout, args.max_tokens
                    )
                    break
                except Exception as exc:
                    last_error = str(exc)
                    # A 429 may indicate exhausted account quota. Do not retry it.
                    if "HTTP_429_RATE_LIMIT:" in last_error:
                        break
                    wait = min(30, 5 * (attempt + 1))
                    if attempt < args.max_retries:
                        time.sleep(wait)

            if candidate is None:
                errors += 1
                append_failure({
                    "slot_id": slot_id, "chunk_id": cid, "task_type": task,
                    "error": last_error, "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                progress.update(1)
                if last_error and "HTTP_429_RATE_LIMIT:" in last_error:
                    progress.close()
                    print("\\nHTTP 429 received. Stopping immediately to protect quota.")
                    print("Provider detail:", last_error)
                    summary = {
                        "selected_chunks": len(selected),
                        "slots_attempted": total,
                        "new_examples_added_this_run": added,
                        "additional_dataset_total": len(generated_rows),
                        "model_skips_or_validation_rejections": skipped,
                        "global_duplicate_rejections": rejected,
                        "request_errors": errors,
                        "stopped_early": True,
                        "stop_reason": "HTTP 429 rate limit",
                        "existing_raw_examples_read_only": len(old_rows),
                        "raw_dataset_modified": False,
                        "output_file": str(OUTPUT_JSONL.relative_to(ROOT)),
                        "checkpoint_dir": str(CHECKPOINT_DIR.relative_to(ROOT)),
                    }
                    (OUTPUT_DIR / "run_summary.json").write_text(
                        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
                    )
                    print(json.dumps(summary, indent=2))
                    return 1
                continue

            valid, reason, evidence_span = validate_candidate(candidate, blocks, text)
            if not valid:
                skipped += 1
                checkpoint.write_text(json.dumps({
                    "slot_id": slot_id, "status": "skipped",
                    "reason": reason,
                    "model_reason": candidate.get("reason"),
                }, indent=2, ensure_ascii=False), encoding="utf-8")
                progress.update(1)
                time.sleep(args.delay)
                continue

            instruction = candidate["instruction"].strip()
            response_text = candidate["response"].strip()
            duplicate_of = None
            duplicate_score = 0.0
            for prior_id, prior_instruction in seen:
                score = similarity(instruction, prior_instruction)
                if score >= args.similarity_threshold:
                    duplicate_of, duplicate_score = prior_id, score
                    break

            if duplicate_of:
                rejected += 1
                checkpoint.write_text(json.dumps({
                    "slot_id": slot_id, "status": "rejected_duplicate",
                    "duplicate_of": duplicate_of,
                    "similarity": round(duplicate_score, 4),
                    "instruction": instruction,
                }, indent=2, ensure_ascii=False), encoding="utf-8")
                progress.update(1)
                time.sleep(args.delay)
                continue

            example_id = f"improved_{cid}_{task}"
            record = {
                "example_id": example_id,
                "batch_id": "improved_pilot",
                "chunk_id": cid,
                "instruction": instruction,
                "response": response_text,
                "evidence_block_ids": candidate["evidence_block_ids"],
                "evidence_span": evidence_span,
                "generator_model": args.model,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "task_type": task,
            }
            generated_rows.append(record)
            write_jsonl(OUTPUT_JSONL, generated_rows)
            seen.append((example_id, instruction))
            checkpoint.write_text(json.dumps({
                "slot_id": slot_id, "status": "complete",
                "example_id": example_id,
            }, indent=2), encoding="utf-8")
            added += 1
            progress.update(1)
            time.sleep(args.delay)

    progress.close()

    summary = {
        "selected_chunks": len(selected),
        "slots_attempted": total,
        "new_examples_added_this_run": added,
        "additional_dataset_total": len(generated_rows),
        "model_skips_or_validation_rejections": skipped,
        "global_duplicate_rejections": rejected,
        "request_errors": errors,
        "existing_raw_examples_read_only": len(old_rows),
        "raw_dataset_modified": False,
        "output_file": str(OUTPUT_JSONL.relative_to(ROOT)),
        "checkpoint_dir": str(CHECKPOINT_DIR.relative_to(ROOT)),
    }
    (OUTPUT_DIR / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nRUN SUMMARY")
    print(json.dumps(summary, indent=2))
    print("\nThe existing dataset_v0_raw.jsonl and production checkpoints were not modified.")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())