import csv
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILTER_DIR = ROOT / "data" / "interim" / "production_filter"
PILOT_DIR = ROOT / "data" / "interim" / "improved_generation"

CANDIDATES_FILE = FILTER_DIR / "deduplicated_candidates.jsonl"
ORIGINAL_JUDGE_FILE = ROOT / "data" / "interim" / "groq_evaluation_results.jsonl"
PILOT_JUDGE_FILE = PILOT_DIR / "groq_pilot_judge_results.jsonl"
FROZEN_CHUNKS_FILE = ROOT / "data" / "frozen" / "chunks_v1_frozen.jsonl"

OUTPUT_DIR = FILTER_DIR / "reconciliation_v2"
OUTPUT_CANDIDATES = OUTPUT_DIR / "reconciled_candidates.jsonl"
OUTPUT_REPORT = OUTPUT_DIR / "reconciliation_report.json"
OUTPUT_CSV = OUTPUT_DIR / "reconciliation_review.csv"

SKYNET_ID = "improved_iot_cloud_case_studies_chunk_0014_classification"
SKYNET_INSTRUCTION = (
    "Identify the platform model and cloud deployment description explicitly "
    "stated for Skynet in the source material."
)
SKYNET_RESPONSE = (
    "Skynet runs as an IoT platform-as-a-service (PaaS) and as a private cloud "
    "through Docker."
)


def read_jsonl(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}, line {line_no}: {exc}")
    return records


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def evidence_blocks(source):
    pattern = re.compile(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", re.DOTALL)
    blocks = []
    for match in pattern.finditer(source):
        block = match.group(0)
        if len(block.strip()) >= 30:
            blocks.append(block)
    return blocks


def judge_passes_original(result):
    return (
        result.get("judge_status") == "success"
        and result.get("accepted") is True
        and result.get("evidence_provenance_valid") is True
        and isinstance(result.get("faithfulness_score"), (int, float))
        and result.get("faithfulness_score") >= 4
        and result.get("evidence_relevant") is True
        and result.get("unsupported_claims") is False
        and result.get("answer_correct_from_source") is True
        and result.get("answer_complete_from_source") is True
    )


def judge_passes_pilot(result):
    return (
        result.get("status") == "completed"
        and result.get("accepted") is True
        and isinstance(result.get("faithfulness_score"), (int, float))
        and result.get("faithfulness_score") >= 4
        and result.get("evidence_relevant") is True
        and result.get("unsupported_claims") is False
        and result.get("answer_correct_from_source") is True
        and result.get("answer_complete_from_source") is True
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    candidates = read_jsonl(CANDIDATES_FILE)
    original_results = read_jsonl(ORIGINAL_JUDGE_FILE)
    pilot_results = read_jsonl(PILOT_JUDGE_FILE)
    chunks = read_jsonl(FROZEN_CHUNKS_FILE)

    candidate_ids = [x.get("example_id") for x in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Duplicate example_id values found in candidate dataset.")

    original_map = {x["example_id"]: x for x in original_results}
    pilot_map = {x["example_id"]: x for x in pilot_results}
    chunk_map = {x["chunk_id"]: x for x in chunks}

    if len(original_map) != len(original_results):
        raise ValueError("Duplicate IDs found in original judge results.")
    if len(pilot_map) != len(pilot_results):
        raise ValueError("Duplicate IDs found in pilot judge results.")

    output_records = []
    review_rows = []
    counts = Counter()
    issues = []

    for candidate in candidates:
        example_id = candidate.get("example_id")
        chunk_id = candidate.get("chunk_id")
        source_chunk = chunk_map.get(chunk_id)

        provenance_ok = False
        provenance_reason = ""
        if source_chunk is None:
            provenance_reason = "chunk_id_not_found_in_frozen_corpus"
        else:
            source_text = source_chunk.get("content", "")
            blocks = evidence_blocks(source_text)
            block_ids = candidate.get("evidence_block_ids", [])
            if not isinstance(block_ids, list) or not block_ids:
                provenance_reason = "missing_or_invalid_evidence_block_ids"
            elif any(not isinstance(i, int) or i < 1 or i > len(blocks) for i in block_ids):
                provenance_reason = "evidence_block_id_out_of_range"
            else:
                reconstructed = "\n\n".join(blocks[i - 1] for i in block_ids)
                provenance_ok = reconstructed == candidate.get("evidence_span", "")
                provenance_reason = "verified" if provenance_ok else "evidence_span_mismatch"

        if example_id in original_map:
            judge = original_map[example_id]
            text_match = (
                candidate.get("instruction") == judge.get("instruction")
                and candidate.get("response") == judge.get("response")
            )
            judge_source = "original"
            judge_ok = judge_passes_original(judge)
        elif example_id in pilot_map:
            judge = pilot_map[example_id]
            # Pilot judge records contain IDs and scores, but not the judged text.
            text_match = True
            judge_source = "pilot"
            judge_ok = judge_passes_pilot(judge)
        else:
            judge = None
            text_match = False
            judge_source = "missing"
            judge_ok = False

        corrected = False
        if example_id == SKYNET_ID:
            candidate["instruction"] = SKYNET_INSTRUCTION
            candidate["response"] = SKYNET_RESPONSE
            corrected = True
            # The previous pilot judgement applies to the old wording, not this correction.
            judge_ok = False
            text_match = False
            judge_source = "pilot_rejudge_required"

        if not provenance_ok:
            issues.append({"example_id": example_id, "issue": provenance_reason})
        if not text_match and not corrected:
            issues.append({"example_id": example_id, "issue": "candidate_text_does_not_match_judged_text"})
        if judge is None:
            issues.append({"example_id": example_id, "issue": "judge_result_missing"})

        if corrected:
            status = "REJUDGE_REQUIRED"
        elif not provenance_ok:
            status = "PROVENANCE_REVIEW_REQUIRED"
        elif judge is None or not text_match:
            status = "JUDGE_MATCH_REVIEW_REQUIRED"
        elif judge_ok:
            status = "JUDGE_ACCEPTED"
        else:
            status = "JUDGE_REJECTED"

        counts[status] += 1
        candidate["reconciliation_status"] = status
        candidate["reconciliation_judge_source"] = judge_source
        candidate["reconciliation_provenance_verified"] = provenance_ok
        candidate["human_audit_correction_applied"] = corrected
        output_records.append(candidate)

        review_rows.append({
            "example_id": example_id,
            "chunk_id": chunk_id,
            "judge_source": judge_source,
            "judge_status": (
                judge.get("judge_status", judge.get("status", "missing"))
                if judge else "missing"
            ),
            "judge_accepted": judge_ok,
            "text_match": text_match,
            "provenance_verified": provenance_ok,
            "human_audit_correction_applied": corrected,
            "reconciliation_status": status,
            "faithfulness_score": judge.get("faithfulness_score") if judge else "",
        })

    write_jsonl(OUTPUT_CANDIDATES, output_records)

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(review_rows[0].keys()))
        writer.writeheader()
        writer.writerows(review_rows)

    report = {
        "candidate_count": len(candidates),
        "original_judge_result_count": len(original_results),
        "pilot_judge_result_count": len(pilot_results),
        "reconciliation_status_counts": dict(counts),
        "provenance_verified_count": sum(r["provenance_verified"] for r in review_rows),
        "human_audit_correction": {
            "example_id": SKYNET_ID,
            "applied": any(r["human_audit_correction_applied"] for r in review_rows),
            "instruction": SKYNET_INSTRUCTION,
            "response": SKYNET_RESPONSE,
            "requires_new_judge_evaluation": True,
        },
        "issues": issues,
        "outputs": {
            "reconciled_candidates": str(OUTPUT_CANDIDATES.relative_to(ROOT)),
            "review_csv": str(OUTPUT_CSV.relative_to(ROOT)),
            "report": str(OUTPUT_REPORT.relative_to(ROOT)),
        },
        "source_files_modified": False,
    }

    OUTPUT_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()