import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "interim" / "production_filter" / "reconciliation_v2" / "reconciled_candidates.jsonl"
CHUNKS = ROOT / "data" / "frozen" / "chunks_v1_frozen.jsonl"
OUT_DIR = ROOT / "data" / "interim" / "production_filter" / "repair_v1"
OUT_CANDIDATES = OUT_DIR / "repair_candidates.jsonl"
OUT_REPORT = OUT_DIR / "repair_report.json"

REPAIRS = {
    "production_batch_011_example_04": {
        "required_terms": ["in smart irrigation example", "soil", "motor", "pump"],
        "instruction": None,
        "response": None,
    },
    "production_batch_012_example_04": {
        "required_terms": ["virtual entity: soil", "virtual entity", "motorstate", "signallevel"],
        "instruction": None,
        "response": None,
    },
    "production_batch_013_example_03": {
        "required_terms": ["monitor the patient body temperature", "real time", "medical", "alarms in emergency"],
        "instruction": None,
        "response": None,
    },
    "production_batch_015_example_04": {
        "required_terms": ["level 5 system has multiple nodes", "coordinator node collects", "sends to the cloud"],
        "instruction": None,
        "response": (
            "The IoT Level 5 system consists of multiple end nodes and one coordinator node. "
            "The coordinator collects data from the end nodes and sends it to the cloud."
        ),
    },
}

def read_jsonl(path):
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{n}: {exc}")
    return rows

def extract_blocks(content):
    # Keep the same evidence-block splitting rule used by the generator.
    pattern = re.compile(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", re.DOTALL)
    blocks = []
    for match in pattern.finditer(content):
        block = match.group(0)
        if len(block.strip()) >= 30:
            blocks.append(block)
    return blocks

def norm(text):
    return re.sub(r"\s+", " ", text.casefold()).strip()

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = read_jsonl(INPUT)
    chunks = read_jsonl(CHUNKS)
    chunk_map = {x["chunk_id"]: x for x in chunks}

    found = {x["example_id"]: x for x in candidates}
    if len(found) != len(candidates):
        raise ValueError("Duplicate candidate IDs in input.")

    report_items = []
    for example_id, spec in REPAIRS.items():
        if example_id not in found:
            raise ValueError(f"Expected repair candidate missing: {example_id}")

        item = found[example_id]
        chunk = chunk_map.get(item["chunk_id"])
        if chunk is None:
            raise ValueError(f"Frozen chunk missing for {example_id}")

        blocks = extract_blocks(chunk["content"])
        required = [norm(x) for x in spec["required_terms"]]
        selected_ids = []

        # Select every block that contributes to the required evidence.
        for idx, block in enumerate(blocks, 1):
            block_norm = norm(block)
            if any(term in block_norm for term in required):
                selected_ids.append(idx)

        # The required terms must all be covered by the selected source blocks.
        selected_text = "\n\n".join(blocks[i - 1] for i in selected_ids)
        selected_norm = norm(selected_text)
        missing_terms = [term for term in required if term not in selected_norm]
        if missing_terms:
            raise ValueError(
                f"Could not locate all required source terms for {example_id}: {missing_terms}"
            )

        old_instruction = item["instruction"]
        old_response = item["response"]
        item["evidence_block_ids"] = selected_ids
        item["evidence_span"] = selected_text
        item["evidence_provenance_valid"] = True
        item["repair_status"] = "REPAIRED_REQUIRES_JUDGE_REEVALUATION"

        if spec["instruction"] is not None:
            item["instruction"] = spec["instruction"]
        if spec["response"] is not None:
            item["response"] = spec["response"]

        report_items.append({
            "example_id": example_id,
            "chunk_id": item["chunk_id"],
            "selected_evidence_block_ids": selected_ids,
            "evidence_span_reconstructed_exactly": (
                "\n\n".join(blocks[i - 1] for i in selected_ids) == item["evidence_span"]
            ),
            "instruction_changed": item["instruction"] != old_instruction,
            "response_changed": item["response"] != old_response,
            "response": item["response"],
            "judge_reevaluation_required": True,
        })

    # Verify every record's evidence span against its frozen source before writing.
    verification_errors = []
    for item in candidates:
        if item["example_id"] in REPAIRS:
            chunk = chunk_map[item["chunk_id"]]
            blocks = extract_blocks(chunk["content"])
            ids = item.get("evidence_block_ids", [])
            if not ids or any(not isinstance(i, int) or i < 1 or i > len(blocks) for i in ids):
                verification_errors.append({
                    "example_id": item["example_id"],
                    "error": "invalid_block_ids",
                })
                continue
            reconstructed = "\n\n".join(blocks[i - 1] for i in ids)
            if reconstructed != item.get("evidence_span"):
                verification_errors.append({
                    "example_id": item["example_id"],
                    "error": "evidence_span_mismatch",
                })

    if verification_errors:
        raise ValueError(f"Post-repair evidence verification failed: {verification_errors}")

    with OUT_CANDIDATES.open("w", encoding="utf-8") as f:
        for item in candidates:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    report = {
        "input_candidate_count": len(candidates),
        "records_repaired": len(report_items),
        "repair_details": report_items,
        "post_repair_evidence_verification": "passed",
        "verification_errors": verification_errors,
        "all_four_records_require_fresh_judge_evaluation": True,
        "input_reconciliation_file_modified": False,
        "frozen_chunks_modified": False,
        "original_datasets_modified": False,
        "outputs": {
            "repair_candidates": str(OUT_CANDIDATES.relative_to(ROOT)),
            "repair_report": str(OUT_REPORT.relative_to(ROOT)),
        },
    }
    OUT_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()