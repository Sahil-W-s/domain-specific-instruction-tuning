import argparse
import json
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "frozen" / "chunks_v1_frozen.jsonl"
OUTPUT_DIR = ROOT / "data" / "interim"
CHECKPOINT_DIR = OUTPUT_DIR / "production_checkpoints"
RAW_FILE = OUTPUT_DIR / "dataset_v0_raw.jsonl"
REQUEST_LOG = OUTPUT_DIR / "production_request_log.jsonl"

MODEL = "nvidia/nemotron-3.5-super-vl:free"

# Preserve the original 80 examples. Generate 920 new examples:
# 5 complete passes across all 91 chunks (910) + 5 extra chunk slots (10).
TARGET_EXISTING_EXAMPLES = 80
TARGET_NEW_EXAMPLES = 920
FINAL_TARGET = TARGET_EXISTING_EXAMPLES + TARGET_NEW_EXAMPLES

CHUNKS_PER_REQUEST = 2
EXAMPLES_PER_CHUNK = 2
MAX_TOKENS = 5000
TIMEOUT_SECONDS = 240
MAX_RETRIES = 2
FIRST_NEW_BATCH_NUMBER = 21

REQUIRED_EXAMPLE_FIELDS = {
    "example_id",
    "batch_id",
    "chunk_id",
    "instruction",
    "response",
    "evidence_block_ids",
    "evidence_span",
    "generator_model",
    "generated_at",
}


def load_jsonl(path):
    records = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSONL at {path}:{line_number}: {exc}"
                    ) from exc
    return records


def extract_evidence_blocks(source):
    """Split source into paragraph-like evidence blocks without rewriting text."""
    blocks = []
    pattern = re.compile(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", re.DOTALL)
    for match in pattern.finditer(source):
        block = match.group(0)
        if len(block.strip()) >= 30:
            blocks.append({"id": len(blocks) + 1, "text": block})
    return blocks


def normalize_question(text):
    return re.sub(r"\W+", " ", text.casefold()).strip()


def atomic_write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    temporary.replace(path)


def atomic_write_jsonl(path, records):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def checkpoint_sort_key(path):
    match = re.search(r"production_batch_(\d+)$", path.stem)
    if not match:
        raise ValueError(f"Unexpected checkpoint filename: {path.name}")
    return int(match.group(1))


def read_checkpoints():
    files = sorted(
        CHECKPOINT_DIR.glob("production_batch_*.json"),
        key=checkpoint_sort_key
    )
    all_examples = []
    seen_ids = set()
    checkpoint_by_batch = {}

    for path in files:
        with path.open("r", encoding="utf-8") as f:
            checkpoint = json.load(f)

        batch_id = checkpoint.get("batch_id")
        if not batch_id or path.name != f"{batch_id}.json":
            raise ValueError(f"Checkpoint filename/batch ID mismatch: {path.name}")
        if batch_id in checkpoint_by_batch:
            raise ValueError(f"Duplicate checkpoint batch ID: {batch_id}")

        examples = checkpoint.get("examples")
        if not isinstance(examples, list) or not examples:
            raise ValueError(f"Missing/empty examples list in {path.name}")

        for example in examples:
            missing = REQUIRED_EXAMPLE_FIELDS - set(example)
            if missing:
                raise ValueError(
                    f"{path.name}: missing fields {sorted(missing)}"
                )
            if example["batch_id"] != batch_id:
                raise ValueError(f"{path.name}: example batch_id mismatch")
            if example["example_id"] in seen_ids:
                raise ValueError(
                    f"Duplicate example ID across checkpoints: "
                    f"{example['example_id']}"
                )
            if not example["instruction"].strip() or not example["response"].strip():
                raise ValueError(f"Empty instruction/response in {path.name}")
            if not example["evidence_span"].strip():
                raise ValueError(f"Empty evidence span in {path.name}")

            seen_ids.add(example["example_id"])
            all_examples.append(example)

        checkpoint_by_batch[batch_id] = checkpoint

    return files, all_examples, checkpoint_by_batch


def rebuild_raw_dataset():
    """Rebuild raw JSONL from checkpoint source-of-truth, preserving all examples."""
    _, examples, _ = read_checkpoints()
    atomic_write_jsonl(RAW_FILE, examples)
    return examples


def append_request_log(record):
    with REQUEST_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_generation_plan(chunks):
    """
    Five passes across all frozen chunks, interleaved by document,
    plus five extra chunk slots. At two examples per slot this is 920.
    """
    by_document = {}
    for chunk in chunks:
        by_document.setdefault(chunk["doc_id"], []).append(chunk)

    if len(by_document) != 4:
        raise ValueError(f"Expected 4 training documents, found {len(by_document)}")

    for doc_id in by_document:
        by_document[doc_id].sort(key=lambda item: item["chunk_index"])

    plan = []
    max_count = max(len(items) for items in by_document.values())

    for _pass in range(5):
        for index in range(max_count):
            for doc_id in sorted(by_document):
                if index < len(by_document[doc_id]):
                    plan.append(by_document[doc_id][index])

    # Five additional slots: one for each document, plus one extra
    # for the document with the most chunks.
    doc_ids = sorted(by_document)
    extras = [by_document[doc_id][0] for doc_id in doc_ids]
    largest_doc = max(doc_ids, key=lambda doc_id: len(by_document[doc_id]))
    extras.append(by_document[largest_doc][1])
    plan.extend(extras)

    expected_slots = TARGET_NEW_EXAMPLES // EXAMPLES_PER_CHUNK
    if len(plan) != expected_slots:
        raise ValueError(
            f"Generation plan has {len(plan)} slots; expected {expected_slots}"
        )
    return plan


def build_prompt(chunk_data, prior_questions):
    sections = []
    for chunk in chunk_data:
        block_text = "\n\n".join(
            f'[BLOCK {block["id"]}]\n{block["text"]}'
            for block in chunk["blocks"]
        )
        previous = prior_questions.get(chunk["chunk_id"], [])
        previous_text = (
            "\n".join(f"- {question}" for question in previous)
            if previous else "(none)"
        )
        sections.append(
            f"""
CHUNK ID: {chunk["chunk_id"]}

PREVIOUSLY GENERATED QUESTIONS FOR THIS CHUNK.
Do not repeat or lightly paraphrase these:
{previous_text}

SOURCE BLOCKS:
{block_text}
"""
        )

    source_material = "\n\n".join(sections)
    expected_count = len(chunk_data) * EXAMPLES_PER_CHUNK

    return f"""
You are generating a high-quality technical instruction dataset about IoT.

Use only the source blocks supplied below. Do not use outside knowledge.

SOURCE MATERIAL:
{source_material}

TASK:
Generate exactly {EXAMPLES_PER_CHUNK} distinct question-answer examples
for EACH supplied chunk. There are {len(chunk_data)} chunks, so return
exactly {expected_count} examples.

STRICT QUALITY REQUIREMENTS:
1. Every answer must be fully supported by the evidence blocks selected for it.
2. Select the smallest set of blocks that directly supports the complete answer.
3. Every selected block must support a specific claim in the answer.
4. Never select evidence merely because it mentions the same topic.
5. Do not include claims that are not supported by the selected blocks.
6. Do not combine facts from different chunks.
7. Questions must be clear, technically meaningful, and answerable from the source.
8. Avoid duplicate questions and avoid paraphrasing previous questions.
9. Prefer questions about definitions, purpose, operation, components,
   relationships, distinctions, explicitly stated advantages or limitations.
10. Do not infer unstated causes, advantages, or relationships.
11. Do not ask about information absent from the source.
12. Return evidence block IDs only, never copied evidence text.
13. Use exact supplied chunk IDs and valid block IDs only.

Return ONLY valid JSON without Markdown fences:
{{
  "examples": [
    {{
      "chunk_id": "exact_chunk_id",
      "instruction": "Question",
      "response": "Answer",
      "evidence_block_ids": [1]
    }}
  ]
}}
"""


def parse_response(content):
    if not content:
        raise ValueError("Model returned empty content")

    content = content.strip()
    content = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        content,
        flags=re.IGNORECASE
    ).strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("No JSON object found in response")
        return json.loads(content[start:end + 1])


def validate_and_prepare(
    parsed,
    chunk_data,
    batch_id,
    actual_model,
    existing_question_keys,
):
    block_lookup = {
        chunk["chunk_id"]: {
            block["id"]: block["text"] for block in chunk["blocks"]
        }
        for chunk in chunk_data
    }

    examples = parsed.get("examples")
    if not isinstance(examples, list):
        raise ValueError("Missing examples list")

    expected_total = len(chunk_data) * EXAMPLES_PER_CHUNK
    if len(examples) != expected_total:
        raise ValueError(
            f"Expected {expected_total} examples, received {len(examples)}"
        )

    counts = {chunk["chunk_id"]: 0 for chunk in chunk_data}
    prepared = []
    batch_question_keys = set()

    for index, example in enumerate(examples, start=1):
        chunk_id = example.get("chunk_id")
        if chunk_id not in block_lookup:
            raise ValueError(f"Example {index}: unknown chunk ID {chunk_id}")

        instruction = example.get("instruction")
        response = example.get("response")
        block_ids = example.get("evidence_block_ids")

        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"Example {index}: empty instruction")
        if not isinstance(response, str) or not response.strip():
            raise ValueError(f"Example {index}: empty response")
        if not isinstance(block_ids, list) or not block_ids:
            raise ValueError(f"Example {index}: missing evidence IDs")

        try:
            block_ids = [int(value) for value in block_ids]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Example {index}: invalid evidence IDs") from exc

        if len(block_ids) != len(set(block_ids)):
            raise ValueError(f"Example {index}: duplicate evidence IDs")

        available_blocks = block_lookup[chunk_id]
        if any(block_id not in available_blocks for block_id in block_ids):
            raise ValueError(f"Example {index}: unknown evidence block")

        question_key = (chunk_id, normalize_question(instruction))
        if not question_key[1]:
            raise ValueError(f"Example {index}: question normalizes to empty")
        if question_key in existing_question_keys or question_key in batch_question_keys:
            raise ValueError(
                f"Duplicate normalized question in {chunk_id}: {instruction}"
            )
        batch_question_keys.add(question_key)

        evidence_span = "\n\n".join(
            available_blocks[block_id] for block_id in block_ids
        )

        counts[chunk_id] += 1
        prepared.append({
            "example_id": f"{batch_id}_example_{index:02d}",
            "batch_id": batch_id,
            "chunk_id": chunk_id,
            "instruction": instruction.strip(),
            "response": response.strip(),
            "evidence_block_ids": block_ids,
            "evidence_span": evidence_span,
            "generator_model": actual_model,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        })

    for chunk_id, count in counts.items():
        if count != EXAMPLES_PER_CHUNK:
            raise ValueError(
                f"{chunk_id}: expected {EXAMPLES_PER_CHUNK}, got {count}"
            )

    return prepared


def main():
    parser = argparse.ArgumentParser(
        description="Resume IoT instruction generation from existing checkpoints."
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=1,
        help="Maximum NEW batches to generate in this run. Default: 1 (safe smoke test).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all remaining batches after the smoke test has been reviewed.",
    )
    args = parser.parse_args()

    if args.max_batches < 1 and not args.all:
        raise ValueError("--max-batches must be at least 1")

    load_dotenv(ROOT / ".env", override=True)
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY missing from .env")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    chunks = load_jsonl(INPUT_FILE)
    if len(chunks) != 91:
        raise RuntimeError(
            f"Expected 91 frozen chunks, found {len(chunks)}. "
            "Do not proceed with a changed corpus."
        )

    _, checkpoint_examples, checkpoint_by_batch = read_checkpoints()

    original_examples = [
        example for example in checkpoint_examples
        if int(re.search(r"production_batch_(\d+)", example["batch_id"]).group(1)) <= 20
    ]
    new_examples_already_saved = [
        example for example in checkpoint_examples
        if int(re.search(r"production_batch_(\d+)", example["batch_id"]).group(1)) >= 21
    ]

    if len(original_examples) != TARGET_EXISTING_EXAMPLES:
        raise RuntimeError(
            f"Expected exactly {TARGET_EXISTING_EXAMPLES} original examples "
            f"in batches 001-020; found {len(original_examples)}."
        )

    if RAW_FILE.exists():
        raw_records = load_jsonl(RAW_FILE)
        raw_ids = {record.get("example_id") for record in raw_records}
        checkpoint_ids = {record["example_id"] for record in checkpoint_examples}
        if raw_ids and raw_ids != checkpoint_ids:
            raise RuntimeError(
                "Existing raw JSONL differs from checkpoint source-of-truth. "
                "Back it up and inspect before proceeding."
            )

    # Rebuild only after checking that any existing raw file agrees with
    # checkpoint contents. The checkpoints remain the source of truth.
    atomic_write_jsonl(RAW_FILE, checkpoint_examples)

    existing_question_keys = {
        (example["chunk_id"], normalize_question(example["instruction"]))
        for example in checkpoint_examples
    }
    prior_questions = {}
    for example in checkpoint_examples:
        prior_questions.setdefault(example["chunk_id"], []).append(
            example["instruction"]
        )

    plan = build_generation_plan(chunks)
    batches = [
        plan[index:index + CHUNKS_PER_REQUEST]
        for index in range(0, len(plan), CHUNKS_PER_REQUEST)
    ]

    if len(batches) != 230:
        raise RuntimeError(f"Expected 230 planned requests, found {len(batches)}")

    # Find the exact missing batch IDs. A saved checkpoint is never regenerated.
    pending = []
    for index, batch in enumerate(batches):
        batch_number = FIRST_NEW_BATCH_NUMBER + index
        batch_id = f"production_batch_{batch_number:03d}"
        if batch_id in checkpoint_by_batch:
            saved_examples = checkpoint_by_batch[batch_id]["examples"]
            expected_count = len(batch) * EXAMPLES_PER_CHUNK
            if len(saved_examples) != expected_count:
                raise RuntimeError(f"Incomplete checkpoint: {batch_id}")
            continue
        pending.append((batch_number, batch_id, batch))

    if args.all:
        selected_pending = pending
    else:
        selected_pending = pending[:args.max_batches]

    print("=" * 75)
    print("PRODUCTION GENERATION — RESUMABLE RUNNER")
    print("=" * 75)
    print("Model:", MODEL)
    print("Frozen chunks:", len(chunks))
    print("Original examples:", len(original_examples))
    print("New examples already checkpointed:", len(new_examples_already_saved))
    print("Current raw examples:", len(checkpoint_examples))
    print("Remaining planned batches:", len(pending))
    print("Batches this run:", len(selected_pending))
    print("Examples per batch:", CHUNKS_PER_REQUEST * EXAMPLES_PER_CHUNK)
    print("Final target:", FINAL_TARGET)
    print("Raw output:", RAW_FILE)

    if not selected_pending:
        print("No pending batches. Checkpointed generation work is complete.")
        return

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        timeout=TIMEOUT_SECONDS,
    )

    failed_batches = []

    for run_index, (batch_number, batch_id, batch) in enumerate(
        selected_pending, start=1
    ):
        chunk_data = [
            {
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "blocks": extract_evidence_blocks(chunk["content"]),
            }
            for chunk in batch
        ]

        if any(not item["blocks"] for item in chunk_data):
            raise RuntimeError(f"Chunk has no usable evidence blocks: {batch_id}")

        prompt = build_prompt(chunk_data, prior_questions)

        print("\n" + "=" * 75)
        print(f"REQUEST {run_index}/{len(selected_pending)} — {batch_id}")
        print("CHUNKS:", ", ".join(item["chunk_id"] for item in chunk_data))

        log_record = {
            "batch_id": batch_id,
            "requested_model": MODEL,
            "chunk_ids": [item["chunk_id"] for item in chunk_data],
            "status": "failed",
            "raw_response": None,
            "error": None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        prepared = None

        for attempt in range(1, MAX_RETRIES + 1):
            started = time.perf_counter()
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=MAX_TOKENS,
                    response_format={"type": "json_object"},
                    extra_body={"reasoning": {"effort": "none"}},
                    timeout=TIMEOUT_SECONDS,
                )

                latency = round(time.perf_counter() - started, 2)
                choice = response.choices[0]
                content = choice.message.content

                log_record["raw_response"] = content
                log_record["actual_model"] = response.model
                log_record["latency_seconds"] = latency
                log_record["finish_reason"] = choice.finish_reason
                log_record["usage"] = (
                    response.usage.model_dump() if response.usage else None
                )

                if choice.finish_reason == "length":
                    raise ValueError("Response truncated (finish_reason=length).")

                parsed = parse_response(content)
                prepared = validate_and_prepare(
                    parsed,
                    chunk_data,
                    batch_id,
                    response.model,
                    existing_question_keys,
                )

                log_record["status"] = "success"
                log_record["generated_examples"] = len(prepared)
                log_record["attempt"] = attempt
                break

            except Exception as exc:
                log_record["error"] = (
                    f"{type(exc).__name__}: {str(exc)[:1000]}"
                )
                log_record["attempt"] = attempt
                print(f"Attempt {attempt}/{MAX_RETRIES} failed: {log_record['error']}")
                if attempt < MAX_RETRIES:
                    time.sleep(2 * attempt)

        if prepared is None:
            failed_batches.append(batch_id)
            append_request_log(log_record)
            print("Batch not checkpointed; it remains eligible for retry.")
            continue

        checkpoint = {
            "batch_id": batch_id,
            "model": response.model,
            "chunk_ids": log_record["chunk_ids"],
            "latency_seconds": log_record["latency_seconds"],
            "usage": log_record["usage"],
            "examples": prepared,
        }
        checkpoint_path = CHECKPOINT_DIR / f"{batch_id}.json"
        atomic_write_json(checkpoint_path, checkpoint)

        # Update deduplication context immediately so later batches in this run
        # cannot repeat questions from newly saved examples.
        for example in prepared:
            existing_question_keys.add(
                (example["chunk_id"], normalize_question(example["instruction"]))
            )
            prior_questions.setdefault(example["chunk_id"], []).append(
                example["instruction"]
            )

        append_request_log(log_record)
        current_examples = rebuild_raw_dataset()

        print("SUCCESS:", len(prepared), "examples")
        print("FINISH:", log_record["finish_reason"])
        print("LATENCY:", log_record["latency_seconds"], "seconds")
        print("TOTAL RAW EXAMPLES:", len(current_examples))

    final_examples = rebuild_raw_dataset()
    document_counts = Counter(
        next(
            chunk["doc_id"]
            for chunk in chunks
            if chunk["chunk_id"] == example["chunk_id"]
        )
        for example in final_examples
    )

    print("\n" + "=" * 75)
    print("RUN SUMMARY")
    print("=" * 75)
    print("Raw examples:", len(final_examples), "/", FINAL_TARGET)
    print("New examples this run:", len(final_examples) - TARGET_EXISTING_EXAMPLES)
    print("Remaining to target:", max(0, FINAL_TARGET - len(final_examples)))
    print("Failed batches this run:", len(failed_batches))
    for doc_id, count in sorted(document_counts.items()):
        print(f"  {doc_id}: {count} examples")

    if failed_batches:
        print("Failed batch IDs:")
        for batch_id in failed_batches:
            print(" ", batch_id)

    if len(final_examples) == FINAL_TARGET:
        print("RAW GENERATION TARGET REACHED.")
    else:
        print("Target not yet reached. Rerun to resume missing batches.")

    print(
        "\nImportant: exact source provenance is verified here; "
        "semantic evidence relevance and factual faithfulness still require "
        "the independent judge and human audit."
    )


if __name__ == "__main__":
    main()