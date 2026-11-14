import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "interim" / "production_filter" / "repair_v1" / "repair_candidates.jsonl"
CHUNKS = ROOT / "data" / "frozen" / "chunks_v1_frozen.jsonl"
OUT_DIR = ROOT / "data" / "interim" / "production_filter" / "repair_v2"
OUT_FILE = OUT_DIR / "repair_candidates.jsonl"
REPORT = OUT_DIR / "repair_report.json"

TARGET = "production_batch_013_example_03"
ADD_BLOCKS = [5]

def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def extract_blocks(content):
    pattern = re.compile(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", re.DOTALL)
    return [
        m.group(0)
        for m in pattern.finditer(content)
        if len(m.group(0).strip()) >= 30
    ]

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = read_jsonl(INPUT)
    chunks = {x["chunk_id"]: x for x in read_jsonl(CHUNKS)}
    matches = [x for x in candidates if x["example_id"] == TARGET]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {TARGET}, found {len(matches)}")

    item = matches[0]
    chunk = chunks[item["chunk_id"]]
    blocks = extract_blocks(chunk["content"])

    existing = item.get("evidence_block_ids", [])
    new_ids = sorted(set(existing + ADD_BLOCKS))
    if any(not isinstance(i, int) or i < 1 or i > len(blocks) for i in new_ids):
        raise ValueError(f"Invalid block IDs: {new_ids}")

    old_evidence = item["evidence_span"]
    new_evidence = "\n\n".join(blocks[i - 1] for i in new_ids)

    required_phrases = [
        "captured and stored",
        "authorized personnel",
        "view and analyze the Data",
        "remotely at any time",
    ]
    normalized = " ".join(new_evidence.casefold().split())
    missing = [p for p in required_phrases if p.casefold() not in normalized]
    if missing:
        raise ValueError(f"Added evidence does not support expected claims: {missing}")

    item["evidence_block_ids"] = new_ids
    item["evidence_span"] = new_evidence
    item["evidence_provenance_valid"] = True
    item["repair_status"] = "REPAIRED_REQUIRES_JUDGE_REEVALUATION"

    # Keep every candidate unchanged except the target record.
    output = []
    for candidate in candidates:
        output.append(item if candidate["example_id"] == TARGET else candidate)

    # Verify exact reconstruction.
    rebuilt = "\n\n".join(blocks[i - 1] for i in item["evidence_block_ids"])
    if rebuilt != item["evidence_span"]:
        raise ValueError("Exact evidence reconstruction failed.")

    with OUT_FILE.open("w", encoding="utf-8") as f:
        for candidate in output:
            f.write(json.dumps(candidate, ensure_ascii=False) + "\n")

    report = {
        "target_example_id": TARGET,
        "previous_evidence_block_ids": existing,
        "added_evidence_block_ids": ADD_BLOCKS,
        "new_evidence_block_ids": new_ids,
        "new_evidence_span": new_evidence,
        "required_claims_supported_by_selected_evidence": True,
        "exact_reconstruction_verified": True,
        "instruction_unchanged": True,
        "response_unchanged": True,
        "requires_fresh_judge_evaluation": True,
        "candidate_count": len(output),
        "source_files_modified": False,
        "outputs": {
            "repair_candidates": str(OUT_FILE.relative_to(ROOT)),
            "report": str(REPORT.relative_to(ROOT)),
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()