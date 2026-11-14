from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "interim" / "production_filter"
CANDIDATE_FILE = BASE / "repair_v2" / "repair_candidates.jsonl"
FROZEN_FILE = ROOT / "data" / "frozen" / "chunks_v1_frozen.jsonl"

# Higher-priority evaluations supersede older evaluations for the same example.
JUDGE_FILES = [
    ("original_groq", 10, ROOT / "data" / "interim" / "groq_evaluation_results.jsonl"),
    ("pilot_groq", 10, BASE.parent / "improved_generation" / "groq_pilot_judge_results.jsonl"),
    ("repair_v1_groq", 20, BASE / "repair_v1" / "judge" / "repair_judge_results.jsonl"),
    ("repair_v2_hma_groq", 30, BASE / "repair_v2" / "judge" / "hma_repair_judge_results.jsonl"),
]

OUTPUT_DIR = BASE / "final_reconciliation"
ACCEPTED_FILE = OUTPUT_DIR / "final_accepted_candidates.jsonl"
REJECTED_FILE = OUTPUT_DIR / "final_rejected_candidates.jsonl"
AUDIT_FILE = OUTPUT_DIR / "final_reconciliation_audit.jsonl"
SUMMARY_FILE = OUTPUT_DIR / "final_reconciliation_summary.json"


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


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def extract_evidence_blocks(source: str) -> list[dict]:
    blocks = []
    pattern = re.compile(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", re.DOTALL)
    for match in pattern.finditer(source):
        block = match.group(0)
        if len(block.strip()) >= 30:
            blocks.append({"id": len(blocks) + 1, "text": block})
    return blocks


def strict_judge_acceptance(result: dict) -> tuple[bool, list[str]]:
    reasons = []
    try:
        score = int(result.get("faithfulness_score", 0))
    except (TypeError, ValueError):
        score = 0
    if score < 4:
        reasons.append("faithfulness_score_below_4")
    if result.get("evidence_relevant") is not True:
        reasons.append("evidence_not_confirmed_relevant")
    if result.get("unsupported_claims") is not False:
        reasons.append("unsupported_claims_not_false")
    if result.get("answer_correct_from_source") is not True:
        reasons.append("answer_correctness_not_confirmed")
    if result.get("answer_complete_from_source") is not True:
        reasons.append("answer_completeness_not_confirmed")
    if result.get("status") not in (None, "completed", "success") and result.get("judge_status") not in (None, "success", "completed"):
        reasons.append("judge_not_completed")
    return not reasons, reasons


def main() -> None:
    for path in (CANDIDATE_FILE, FROZEN_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Required input file not found: {path}")

    candidates = read_jsonl(CANDIDATE_FILE)
    if len(candidates) != 110:
        raise RuntimeError(f"Expected 110 deduplicated candidates; found {len(candidates)}.")

    candidate_ids = [x.get("example_id") for x in candidates]
    if not all(candidate_ids) or len(set(candidate_ids)) != len(candidate_ids):
        raise RuntimeError("Candidate example_id values are missing or duplicated.")

    chunks = read_jsonl(FROZEN_FILE)
    chunk_map = {}
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        content = chunk.get("content")
        if chunk_id and isinstance(content, str):
            chunk_map[chunk_id] = content

    judge_by_id: dict[str, dict] = {}
    judge_source_by_id: dict[str, str] = {}
    judge_priority_by_id: dict[str, int] = {}
    judge_file_counts = {}
    missing_judge_files = []

    for source_name, priority, path in JUDGE_FILES:
        if not path.exists():
            missing_judge_files.append(str(path))
            continue
        rows = read_jsonl(path)
        judge_file_counts[source_name] = len(rows)
        for row in rows:
            example_id = row.get("example_id")
            if not example_id:
                continue
            old_priority = judge_priority_by_id.get(example_id, -1)
            if priority >= old_priority:
                judge_by_id[example_id] = row
                judge_source_by_id[example_id] = source_name
                judge_priority_by_id[example_id] = priority

    accepted, rejected, audits = [], [], []
    status_counts = Counter()
    provenance_failures = []
    missing_judges = []
    accepted_from_sources = Counter()
    rejected_reasons = Counter()

    for candidate in candidates:
        example_id = candidate["example_id"]
        chunk_id = candidate.get("chunk_id")
        source_text = chunk_map.get(chunk_id)
        provenance_ok = False
        provenance_reason = None

        if source_text is None:
            provenance_reason = "chunk_id_not_found_in_frozen_corpus"
        else:
            block_lookup = {b["id"]: b["text"] for b in extract_evidence_blocks(source_text)}
            ids = candidate.get("evidence_block_ids") or []
            if not ids:
                provenance_reason = "evidence_block_ids_missing"
            elif any(block_id not in block_lookup for block_id in ids):
                provenance_reason = "one_or_more_evidence_block_ids_invalid"
            else:
                reconstructed = "\n\n".join(block_lookup[block_id] for block_id in ids)
                if reconstructed != candidate.get("evidence_span", ""):
                    provenance_reason = "evidence_span_does_not_exactly_reconstruct"
                else:
                    provenance_ok = True

        judge = judge_by_id.get(example_id)
        judge_ok, judge_reasons = (False, ["judge_result_missing"])
        if judge is not None:
            judge_ok, judge_reasons = strict_judge_acceptance(judge)
        else:
            missing_judges.append(example_id)

        final_ok = provenance_ok and judge_ok
        if not provenance_ok:
            provenance_failures.append({"example_id": example_id, "reason": provenance_reason})

        record = dict(candidate)
        record["final_filter_status"] = "ACCEPT" if final_ok else "REJECT"
        record["final_judge_source"] = judge_source_by_id.get(example_id)
        record["final_judge_model"] = judge.get("judge_model") if judge else None
        record["final_faithfulness_score"] = judge.get("faithfulness_score") if judge else None
        record["final_rejection_reasons"] = (
            ([] if provenance_ok else [f"provenance:{provenance_reason}"])
            + ([] if judge_ok else [f"judge:{reason}" for reason in judge_reasons])
        )

        audit = {
            "example_id": example_id,
            "chunk_id": chunk_id,
            "provenance_verified": provenance_ok,
            "provenance_failure_reason": provenance_reason,
            "judge_result_found": judge is not None,
            "judge_source": judge_source_by_id.get(example_id),
            "judge_model": judge.get("judge_model") if judge else None,
            "faithfulness_score": judge.get("faithfulness_score") if judge else None,
            "judge_accepted_under_strict_rule": judge_ok,
            "judge_rejection_reasons": judge_reasons,
            "final_status": record["final_filter_status"],
        }
        audits.append(audit)

        if final_ok:
            accepted.append(record)
            accepted_from_sources[judge_source_by_id[example_id]] += 1
            status_counts["ACCEPT"] += 1
        else:
            rejected.append(record)
            status_counts["REJECT"] += 1
            for reason in record["final_rejection_reasons"]:
                rejected_reasons[reason] += 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(ACCEPTED_FILE, accepted)
    write_jsonl(REJECTED_FILE, rejected)
    write_jsonl(AUDIT_FILE, audits)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_count": len(candidates),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "acceptance_rate_percent": round(len(accepted) / len(candidates) * 100, 2),
        "provenance_verified_count": sum(1 for x in audits if x["provenance_verified"]),
        "provenance_failure_count": len(provenance_failures),
        "missing_judge_result_count": len(missing_judges),
        "judge_result_counts_by_file": judge_file_counts,
        "judge_sources_used_for_final_decision": dict(accepted_from_sources),
        "final_status_counts": dict(status_counts),
        "rejection_reason_counts": dict(rejected_reasons),
        "provenance_failures": provenance_failures,
        "missing_judge_example_ids": missing_judges,
        "missing_optional_judge_files": missing_judge_files,
        "acceptance_rule": (
            "Exact evidence reconstruction from frozen corpus AND "
            "faithfulness_score >= 4 AND evidence_relevant=true AND "
            "unsupported_claims=false AND answer_correct_from_source=true AND "
            "answer_complete_from_source=true"
        ),
        "human_audit": {
            "sample_size": 30,
            "human_accepted": 29,
            "human_rejected": 1,
            "rejected_example_id": "production_batch_011_example_03",
            "note": "Human-audit decision is reported separately; the approved duplicate is already excluded from the 110-candidate input."
        },
        "input_sha256": {
            "candidate_file": sha256(CANDIDATE_FILE),
            "frozen_corpus": sha256(FROZEN_FILE),
        },
        "outputs": {
            "accepted": str(ACCEPTED_FILE.relative_to(ROOT)),
            "rejected": str(REJECTED_FILE.relative_to(ROOT)),
            "audit": str(AUDIT_FILE.relative_to(ROOT)),
            "summary": str(SUMMARY_FILE.relative_to(ROOT)),
        },
        "source_files_modified": False,
        "raw_dataset_modified": False,
        "frozen_source_modified": False,
    }
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()