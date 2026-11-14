#!/usr/bin/env python3
"""
Resumable production generation v2 for the IoT technical instruction dataset.

This recovery-safe version seeds from the verified 73-record recovery copy,
then appends records directly. It does not replace the existing v2 dataset.

Safety:
- Reads only the frozen training chunks and existing datasets.
- Writes only to data/interim/production_generation_v2/.
- Does not modify dataset_v0_raw.jsonl, dataset_v0_additional.jsonl,
  the frozen corpus, or any earlier checkpoint directory.
- One request per chunk/pass; asks for one QA, one extraction, and one
  classification example in each batch.
- Four passes provide a theoretical ceiling of 91 * 4 * 3 = 1,092
  candidate slots. Actual yield will be lower because unsupported tasks,
  invalid outputs, and duplicates are skipped.

Start with the default two-chunk pilot. Scale only after inspecting it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
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
FROZEN_FILE = ROOT / "data" / "frozen" / "chunks_v1_frozen.jsonl"
ORIGINAL_RAW = ROOT / "data" / "interim" / "dataset_v0_raw.jsonl"
PRIOR_ADDITIONAL = ROOT / "data" / "interim" / "improved_generation" / "dataset_v0_additional.jsonl"

OUTPUT_DIR = ROOT / "data" / "interim" / "production_generation_v2"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
RECOVERY_FILE = OUTPUT_DIR / "dataset_v2_recovered_20261113.jsonl"
OUTPUT_FILE = OUTPUT_DIR / "dataset_v0_additional_v2_resumed.jsonl"
FAILURES_FILE = OUTPUT_DIR / "generation_failures.jsonl"
MANIFEST_FILE = OUTPUT_DIR / "run_manifest.json"
SUMMARY_FILE = OUTPUT_DIR / "run_summary.json"

MODEL_DEFAULT = "nvidia/nemotron-3.5-super-vl:free"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

TASKS = ("qa", "extraction", "classification")
TASK_GUIDANCE = {
    "qa": "Ask a precise question about one technical fact, mechanism, component, relationship, or procedure stated in the source.",
    "extraction": "Ask for a well-defined list, sequence, parameter, field, protocol detail, or named entities explicitly stated in the source.",
    "classification": "Ask to classify a named technology, protocol, component, architecture, or scenario using criteria explicitly stated in the source.",
}

PASS_FOCUS = {
    1: "CORE COVERAGE: identify the most central, clearly stated facts and concepts in this chunk.",
    2: "TECHNICAL DEPTH: find different technical details such as mechanisms, parameters, message fields, protocol behavior, component roles, or architecture details. Do not repeat core facts already asked.",
    3: "RELATIONSHIPS AND PROCEDURES: focus on relationships between components, workflows, sequences, configuration steps, or cause/effect only when the source explicitly states them.",
    4: "ADDITIONAL DISTINCT FACTS: find remaining independently useful details not covered by prior questions. Prefer precise, non-overlapping details over broad summaries.",
}

SYSTEM_PROMPT = """You create high-quality source-grounded instruction/response training examples from technical IoT documentation.

Return exactly one JSON object:
{
  "examples": [
    {
      "task_type": "qa|extraction|classification",
      "instruction": "...",
      "response": "...",
      "evidence_block_ids": [1, 2]
    }
  ]
}

Generate exactly three examples when the source supports them: one QA, one extraction, and one classification. If a task type cannot be supported by the source, omit that item and add a short entry to "skipped" with its task_type and reason. Never invent facts to fill the quota.

Rules:
1. Use only facts explicitly stated in the source.
2. Every claim in each response must be supported by that example's selected evidence blocks.
3. Select all and only the evidence blocks needed to support the response.
4. Do not use outside knowledge or silently correct the source.
5. Do not create broad questions that combine unrelated sections.
6. Avoid trivial, vague, yes/no, or duplicate questions.
7. Keep answers concise and technically precise.
8. Evidence block IDs must match the numbered source blocks exactly.
9. Make each example test a different fact, detail, or relationship.
10. Do not repeat or paraphrase any previously generated instruction listed in the prompt.
11. If a strong example is not supported, skip it rather than guessing.
12. Output valid JSON only; no markdown fences or commentary.
"""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_no}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a one-time seed file directly; generation itself uses append_jsonl."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Append one committed record; never replace the existing JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Write small metadata files directly to avoid OneDrive rename conflicts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def ensure_resumed_output() -> None:
    """
    Seed the new working output from the verified 73-row recovery copy.
    Never overwrite an existing resumed output.
    """
    if OUTPUT_FILE.exists():
        return
    if not RECOVERY_FILE.exists():
        raise FileNotFoundError(
            "Verified recovery copy not found: "
            f"{RECOVERY_FILE}. Refusing to seed from the older 72-row main file."
        )
    shutil.copy2(RECOVERY_FILE, OUTPUT_FILE)
    # Validate the copied file before generation proceeds.
    seeded = read_jsonl(OUTPUT_FILE)
    if len(seeded) != 73:
        raise RuntimeError(
            f"Expected 73 records in recovery copy, found {len(seeded)}."
        )
    ids = [str(row.get("example_id") or "") for row in seeded]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise RuntimeError("Recovery copy contains missing or duplicate example IDs.")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def get_first(row: dict[str, Any], names: tuple[str, ...], default=None):
    for name in names:
        value = row.get(name)
        if value is not None:
            return value
    return default


def chunk_id_of(row: dict[str, Any]) -> str:
    value = get_first(row, ("chunk_id", "id", "chunkId"))
    if value is None:
        raise ValueError(f"Chunk has no ID field. Keys: {list(row)}")
    return str(value)


def chunk_text_of(row: dict[str, Any]) -> str:
    value = get_first(row, ("content", "text", "chunk_text", "page_content"))
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Chunk {chunk_id_of(row)} has no non-empty content.")
    return value.strip()


def evidence_blocks(source: str) -> list[dict[str, Any]]:
    result = []
    pattern = re.compile(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", re.DOTALL)
    for match in pattern.finditer(source):
        block = match.group(0)
        if len(block.strip()) >= 30:
            result.append({"id": len(result) + 1, "text": block})
    return result


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
    seq = SequenceMatcher(None, na, nb).ratio()
    ta, tb = set(na.split()), set(nb.split())
    jaccard = len(ta & tb) / len(ta | tb) if ta | tb else 0.0
    return max(seq, jaccard)


def parse_json_response(content: Any) -> dict[str, Any]:
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    content = str(content or "").strip()
    content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
    content = re.sub(r"\s*```$", "", content)
    start, end = content.find("{"), content.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"Model did not return a JSON object: {content[:600]}")
    return json.loads(content[start:end + 1])


def call_model(
    session: requests.Session,
    api_key: str,
    model: str,
    chunk_id: str,
    source_doc: str,
    pass_number: int,
    blocks: list[dict[str, Any]],
    prior_questions: list[str],
    timeout: int,
    max_tokens: int,
) -> dict[str, Any]:
    numbered = "\n\n".join(f"[BLOCK {b['id']}]\n{b['text']}" for b in blocks)
    prior = "\n".join(f"- {q}" for q in prior_questions[-60:]) or "(none)"
    focus = PASS_FOCUS[pass_number]
    prompt = f"""SOURCE DOCUMENT: {source_doc}
CHUNK ID: {chunk_id}
GENERATION PASS: {pass_number}
PASS OBJECTIVE: {focus}

Create a batch containing up to three high-quality examples:
- exactly one QA example, if supported
- exactly one extraction example, if supported
- exactly one classification example, if supported

Task guidance:
- QA: {TASK_GUIDANCE["qa"]}
- Extraction: {TASK_GUIDANCE["extraction"]}
- Classification: {TASK_GUIDANCE["classification"]}

PREVIOUSLY GENERATED QUESTIONS FROM THIS CHUNK — DO NOT REPEAT THEIR FACT, TOPIC, OR WORDING:
{prior}

NUMBERED SOURCE EVIDENCE BLOCKS:
{numbered}

Return the required JSON object. Each example must have its own task_type, instruction, response, and evidence_block_ids. Use different facts from the previous questions. If a task type has no genuinely distinct supported fact for this pass, omit it and explain in "skipped".
"""
    payload = {
        "model": model,
        "temperature": 0.35,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }
    response = session.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://openrouter.ai/",
            "X-Title": "IoT Technical Dataset Generation v2",
        },
        json=payload,
        timeout=timeout,
    )
    if response.status_code == 429:
        try:
            detail = response.json()
        except ValueError:
            detail = response.text[:2000]
        raise RuntimeError("HTTP_429_RATE_LIMIT: " + json.dumps(detail, ensure_ascii=False)[:2500])
    response.raise_for_status()
    body = response.json()
    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError(f"Model returned no choices: {json.dumps(body)[:1000]}")
    content = (choices[0].get("message") or {}).get("content")
    if not content:
        raise RuntimeError(
            f"Model returned empty content; finish_reason={choices[0].get('finish_reason')}"
        )
    return parse_json_response(content)


def validate_candidate(
    candidate: dict[str, Any],
    blocks: list[dict[str, Any]],
    source_text: str,
    expected_task: str,
) -> tuple[bool, str, str, list[int]]:
    task = str(candidate.get("task_type", "")).strip().lower()
    if task != expected_task:
        return False, "wrong_or_missing_task_type", "", []
    instruction = candidate.get("instruction")
    response = candidate.get("response")
    ids = candidate.get("evidence_block_ids")
    if not isinstance(instruction, str) or len(instruction.strip()) < 15:
        return False, "invalid_instruction", "", []
    if not isinstance(response, str) or len(response.strip()) < 5:
        return False, "invalid_response", "", []
    if len(instruction) > 1200 or len(response) > 3000:
        return False, "length_limit", "", []
    if not isinstance(ids, list) or not ids:
        return False, "missing_evidence_ids", "", []
    lookup = {b["id"]: b["text"] for b in blocks}
    try:
        selected = list(dict.fromkeys(int(x) for x in ids))
    except (TypeError, ValueError):
        return False, "invalid_evidence_ids", "", []
    if any(x not in lookup for x in selected):
        return False, "unknown_evidence_id", "", []
    span = "\n\n".join(lookup[x] for x in selected)
    if not span or span not in source_text:
        return False, "evidence_reconstruction_failed", "", []
    return True, "", span, selected


def checkpoint_path(chunk_id: str, pass_number: int) -> Path:
    safe_chunk = re.sub(r"[^A-Za-z0-9_.-]+", "_", chunk_id)
    return CHECKPOINT_DIR / f"{safe_chunk}__pass_{pass_number:02d}.json"


def append_failure(record: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with FAILURES_FILE.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-chunks", type=int, default=2,
                        help="Pilot chunk count. Default 2.")
    parser.add_argument("--all-chunks", action="store_true",
                        help="Process all frozen chunks.")
    parser.add_argument("--passes", type=int, default=4,
                        choices=(1, 2, 3, 4), help="Number of distinct generation passes.")
    parser.add_argument("--start-index", type=int, default=0,
                        help="Zero-based start index in frozen chunk list.")
    parser.add_argument("--delay", type=float, default=15.0,
                        help="Seconds between successful requests.")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-tokens", type=int, default=5000)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--similarity-threshold", type=float, default=0.88)
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--retry-skipped", action="store_true",
                        help="Retry prior pass checkpoints that produced zero accepted examples.")
    args = parser.parse_args()

    if args.limit_chunks < 1 or args.start_index < 0:
        parser.error("--limit-chunks must be >=1 and --start-index must be >=0")

    load_dotenv(ROOT / ".env", override=True)
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY is missing from the project .env file.", file=sys.stderr)
        return 2
    if not FROZEN_FILE.exists():
        print(f"ERROR: Frozen corpus not found: {FROZEN_FILE}", file=sys.stderr)
        return 2

    chunks = read_jsonl(FROZEN_FILE)
    if not chunks:
        print("ERROR: Frozen corpus is empty.", file=sys.stderr)
        return 2

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_resumed_output()

    old_rows = read_jsonl(ORIGINAL_RAW)
    prior_rows = read_jsonl(PRIOR_ADDITIONAL)
    generated_rows = read_jsonl(OUTPUT_FILE)

    # Detect duplicate IDs before continuing.
    all_ids = [str(r.get("example_id")) for r in generated_rows if r.get("example_id")]
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("Duplicate example_id values found in v2 output; refusing to continue.")

    # Existing records may have been appended immediately before an interrupted
    # pass failed to write its checkpoint. Track their task keys so they are
    # retained and never regenerated under the same deterministic example ID.
    existing_task_keys = {
        (
            str(row.get("chunk_id", "")),
            int(row.get("generation_pass", 0) or 0),
            str(row.get("task_type", "")).lower(),
        )
        for row in generated_rows
    }
    existing_ids = set(all_ids)

    selected = chunks[args.start_index:] if args.all_chunks else chunks[
        args.start_index:args.start_index + args.limit_chunks
    ]
    if not selected:
        print("ERROR: No chunks selected. Check --start-index.", file=sys.stderr)
        return 2

    # Existing instructions seed the global duplicate detector.
    seen: list[tuple[str, str]] = []
    for row in old_rows + prior_rows + generated_rows:
        instruction = str(row.get("instruction", "")).strip()
        if instruction:
            seen.append((str(row.get("example_id", "existing")), instruction))

    # Previous v2 questions are also used in prompts, grouped by chunk.
    questions_by_chunk: dict[str, list[str]] = {}
    for row in generated_rows:
        cid = str(row.get("chunk_id", ""))
        if cid:
            questions_by_chunk.setdefault(cid, []).append(str(row.get("instruction", "")))

    frozen_hash = sha256(FROZEN_FILE)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "frozen_corpus": str(FROZEN_FILE.relative_to(ROOT)),
        "frozen_corpus_sha256": frozen_hash,
        "original_raw_examples_read_only": len(old_rows),
        "prior_additional_examples_read_only": len(prior_rows),
        "previous_v2_examples": len(generated_rows),
        "frozen_chunk_count": len(chunks),
        "selected_chunk_count": len(selected),
        "selected_chunk_ids": [chunk_id_of(c) for c in selected],
        "passes": args.passes,
        "examples_requested_per_pass": 3,
        "theoretical_slots_in_selected_run": len(selected) * args.passes * 3,
        "task_types": list(TASKS),
        "similarity_threshold": args.similarity_threshold,
        "recovered_seed_file": str(RECOVERY_FILE.relative_to(ROOT)),
        "output_file": str(OUTPUT_FILE.relative_to(ROOT)),
        "checkpoint_dir": str(CHECKPOINT_DIR.relative_to(ROOT)),
    }
    write_json(MANIFEST_FILE, manifest)

    session = requests.Session()
    added = skipped = duplicates = errors = completed_passes = 0
    total_passes = len(selected) * args.passes
    progress = tqdm(total=total_passes, desc="Production generation v2", unit="pass")

    for chunk in selected:
        cid = chunk_id_of(chunk)
        source_text = chunk_text_of(chunk)
        source_doc = str(get_first(chunk, ("source_doc", "document_id", "doc_id", "source"), "unknown"))
        blocks = evidence_blocks(source_text)
        if not blocks:
            for pass_number in range(1, args.passes + 1):
                cp = checkpoint_path(cid, pass_number)
                if not cp.exists():
                    write_json(cp, {
                        "chunk_id": cid, "pass_number": pass_number,
                        "status": "no_evidence_blocks", "accepted_count": 0,
                    })
                progress.update(1)
            continue

        for pass_number in range(1, args.passes + 1):
            cp = checkpoint_path(cid, pass_number)
            if cp.exists():
                try:
                    previous_checkpoint = json.loads(cp.read_text(encoding="utf-8"))
                except Exception:
                    previous_checkpoint = {}
                status = previous_checkpoint.get("status")
                if status == "complete":
                    progress.update(1)
                    continue
                if status in ("no_examples_accepted", "no_evidence_blocks") and not args.retry_skipped:
                    progress.update(1)
                    continue
                if status not in ("no_examples_accepted", "no_evidence_blocks") and not args.retry_skipped:
                    progress.update(1)
                    continue

            candidate_batch = None
            last_error = None
            for attempt in range(args.max_retries + 1):
                try:
                    candidate_batch = call_model(
                        session, api_key, args.model, cid, source_doc,
                        pass_number, blocks, questions_by_chunk.get(cid, []),
                        args.timeout, args.max_tokens,
                    )
                    break
                except Exception as exc:
                    last_error = str(exc)
                    if "HTTP_429_RATE_LIMIT:" in last_error:
                        break
                    if attempt < args.max_retries:
                        time.sleep(min(45, 8 * (attempt + 1)))

            if candidate_batch is None:
                errors += 1
                append_failure({
                    "chunk_id": cid, "pass_number": pass_number,
                    "error": last_error,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                })
                progress.update(1)
                if last_error and "HTTP_429_RATE_LIMIT:" in last_error:
                    progress.close()
                    summary = {
                        "selected_chunks": len(selected),
                        "passes_per_chunk": args.passes,
                        "pass_slots": total_passes,
                        "new_examples_added_this_run": added,
                        "v2_dataset_total": len(generated_rows),
                        "model_skips_or_validation_rejections": skipped,
                        "global_duplicate_rejections": duplicates,
                        "request_errors": errors,
                        "completed_passes_this_run": completed_passes,
                        "stopped_early": True,
                        "stop_reason": "HTTP 429 rate limit",
                        "original_raw_modified": False,
                        "prior_additional_modified": False,
                        "frozen_corpus_modified": False,
                        "output_file": str(OUTPUT_FILE.relative_to(ROOT)),
                    }
                    write_json(SUMMARY_FILE, summary)
                    print(json.dumps(summary, indent=2, ensure_ascii=False))
                    return 1
                continue

            raw_examples = candidate_batch.get("examples", [])
            if not isinstance(raw_examples, list):
                raw_examples = []
            accepted_this_pass = 0
            pass_skips = candidate_batch.get("skipped", [])
            if not isinstance(pass_skips, list):
                pass_skips = []

            # Process in deterministic task order, irrespective of model ordering.
            for task in TASKS:
                task_key = (cid, pass_number, task)
                if task_key in existing_task_keys:
                    # A record was committed before a previous interruption.
                    # Preserve it and do not create a duplicate ID.
                    continue

                matches = [
                    x for x in raw_examples
                    if isinstance(x, dict) and str(x.get("task_type", "")).lower() == task
                ]
                if len(matches) > 1:
                    matches = matches[:1]
                    skipped += 1
                if not matches:
                    skipped += 1
                    continue

                candidate = matches[0]
                valid, reason, span, selected_ids = validate_candidate(
                    candidate, blocks, source_text, task
                )
                if not valid:
                    skipped += 1
                    pass_skips.append({"task_type": task, "reason": reason})
                    continue

                instruction = candidate["instruction"].strip()
                response_text = candidate["response"].strip()
                duplicate_match = None
                duplicate_score = 0.0
                for prior_id, prior_instruction in seen:
                    score = similarity(instruction, prior_instruction)
                    if score >= args.similarity_threshold:
                        duplicate_match = prior_id
                        duplicate_score = score
                        break

                if duplicate_match:
                    duplicates += 1
                    pass_skips.append({
                        "task_type": task,
                        "reason": "global_duplicate",
                        "duplicate_of": duplicate_match,
                        "similarity": round(duplicate_score, 4),
                    })
                    continue

                example_id = f"genv2_{cid}_p{pass_number:02d}_{task}"
                if example_id in existing_ids:
                    # Defensive guard: do not duplicate a previously committed ID.
                    skipped += 1
                    pass_skips.append({
                        "task_type": task,
                        "reason": "example_id_already_committed",
                    })
                    continue

                record = {
                    "example_id": example_id,
                    "batch_id": f"production_v2_pass_{pass_number:02d}",
                    "chunk_id": cid,
                    "instruction": instruction,
                    "response": response_text,
                    "evidence_block_ids": selected_ids,
                    "evidence_span": span,
                    "generator_model": args.model,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "task_type": task,
                    "generation_pass": pass_number,
                }
                append_jsonl(OUTPUT_FILE, record)
                generated_rows.append(record)
                existing_ids.add(example_id)
                existing_task_keys.add((cid, pass_number, task))
                seen.append((example_id, instruction))
                questions_by_chunk.setdefault(cid, []).append(instruction)
                accepted_this_pass += 1
                added += 1

            status = "complete" if accepted_this_pass else "no_examples_accepted"
            write_json(cp, {
                "chunk_id": cid,
                "pass_number": pass_number,
                "status": status,
                "accepted_count": accepted_this_pass,
                "skipped": pass_skips,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            })
            completed_passes += 1
            progress.update(1)
            time.sleep(args.delay)

    progress.close()
    summary = {
        "selected_chunks": len(selected),
        "passes_per_chunk": args.passes,
        "pass_slots": total_passes,
        "new_examples_added_this_run": added,
        "v2_dataset_total": len(generated_rows),
        "model_skips_or_validation_rejections": skipped,
        "global_duplicate_rejections": duplicates,
        "request_errors": errors,
        "completed_passes_this_run": completed_passes,
        "stopped_early": False,
        "original_raw_examples_read_only": len(old_rows),
        "prior_additional_examples_read_only": len(prior_rows),
        "original_raw_modified": False,
        "prior_additional_modified": False,
        "frozen_corpus_modified": False,
        "recovered_seed_file": str(RECOVERY_FILE.relative_to(ROOT)),
        "output_file": str(OUTPUT_FILE.relative_to(ROOT)),
        "checkpoint_dir": str(CHECKPOINT_DIR.relative_to(ROOT)),
    }
    write_json(SUMMARY_FILE, summary)
    print("\nRUN SUMMARY")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted. Completed examples and pass checkpoints are preserved.")
        raise SystemExit(130)