#!/usr/bin/env python3
"""
Reusable orchestrator for the IoT technical instruction dataset.

Examples (run from project root):
  python pipeline.py --stage status
  python pipeline.py --stage generate
  python pipeline.py --stage audit
  python pipeline.py --stage judge --input <reviewed_candidates.jsonl>
  python pipeline.py --stage prepare-human-audit
  python pipeline.py --stage apply-human-audit
  python pipeline.py --stage all-automated

Human review is an explicit gate. No stage silently approves examples.
All source datasets and frozen chunks are read-only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
DATA = ROOT / "data"
FROZEN = DATA / "frozen" / "chunks_v1_frozen.jsonl"
GENERATOR = SCRIPTS / "60_generate_production_v2_resumable.py"
AUDITOR = SCRIPTS / "61_audit_generation_v2.py"
JUDGE = SCRIPTS / "pipeline_judge.py"

V2 = DATA / "interim" / "production_generation_v2"
GENERATED = V2 / "dataset_v0_additional_v2_resumed.jsonl"
AUDIT_SUMMARY = V2 / "audit_v1" / "audit_summary.json"
JUDGE_DIR = V2 / "pipeline_judge"
JUDGE_RESULTS = JUDGE_DIR / "v2_judge_results.jsonl"
HUMAN_DIR = V2 / "human_audit_pipeline"
HUMAN_SAMPLE = HUMAN_DIR / "human_audit_sample.csv"
HUMAN_DECISIONS = HUMAN_DIR / "human_audit_decisions.csv"
HUMAN_APPROVED = HUMAN_DIR / "human_audit_approved.json"
STATE_FILE = V2 / "pipeline_state.json"

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}, line {n}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}, line {n}")
            rows.append(row)
    return rows

def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def load_state() -> dict[str, Any]:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"created_at": datetime.now(timezone.utc).isoformat(), "stages": {}}

def mark_stage(state: dict[str, Any], name: str, status: str, **details: Any) -> None:
    state.setdefault("stages", {})[name] = {
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **details,
    }
    write_json(STATE_FILE, state)

def run_script(script: Path, args: list[str] | None = None, env: dict[str, str] | None = None) -> None:
    if not script.exists():
        raise FileNotFoundError(f"Pipeline component missing: {script}")
    command = [sys.executable, str(script)] + (args or [])
    print("\n>", " ".join(command), flush=True)
    child_env = os.environ.copy()
    if env:
        child_env.update(env)
    result = subprocess.run(command, cwd=ROOT, env=child_env)
    if result.returncode != 0:
        raise RuntimeError(f"{script.name} failed with exit code {result.returncode}")

def stage_status() -> int:
    state = load_state()
    print(json.dumps(state, indent=2, ensure_ascii=False))
    print("\nFile status:")
    for label, path in [
        ("Frozen corpus", FROZEN),
        ("Generated v2", GENERATED),
        ("Structural audit", AUDIT_SUMMARY),
        ("Judge results", JUDGE_RESULTS),
        ("Human audit sample", HUMAN_SAMPLE),
        ("Human decisions", HUMAN_DECISIONS),
    ]:
        print(f"- {label}: {'present' if path.exists() else 'not present'}")
    return 0

def stage_generate(args: argparse.Namespace, state: dict[str, Any]) -> None:
    if not FROZEN.exists():
        raise FileNotFoundError(FROZEN)
    before = sha256(FROZEN)
    gen_args = ["--all-chunks", "--passes", str(args.passes), "--delay", str(args.delay)]
    if args.retry_skipped:
        gen_args.append("--retry-skipped")
    run_script(GENERATOR, gen_args)
    after = sha256(FROZEN)
    if before != after:
        raise RuntimeError("Frozen corpus hash changed during generation; stop.")
    rows = read_jsonl(GENERATED)
    mark_stage(state, "generate", "complete", output=str(GENERATED.relative_to(ROOT)),
               examples=len(rows), frozen_sha256=after)

def stage_audit(state: dict[str, Any]) -> None:
    if not GENERATED.exists():
        raise FileNotFoundError("Run generation first; generated v2 dataset is missing.")
    before = sha256(FROZEN)
    run_script(AUDITOR)
    after = sha256(FROZEN)
    if before != after:
        raise RuntimeError("Frozen corpus hash changed during audit; stop.")
    report = json.loads(AUDIT_SUMMARY.read_text(encoding="utf-8"))
    if report.get("status") != "PASS_STRUCTURAL_AUDIT":
        mark_stage(state, "audit", "failed", report=str(AUDIT_SUMMARY.relative_to(ROOT)))
        raise RuntimeError("Structural audit did not pass. Inspect audit_v1 reports.")
    mark_stage(state, "audit", "complete", examples=report["inputs"]["v2_count"],
               report=str(AUDIT_SUMMARY.relative_to(ROOT)), frozen_sha256=after)

def stage_judge(input_path: Path, delay: float, limit: int, state: dict[str, Any]) -> None:
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    rows = read_jsonl(input_path)
    if not rows:
        raise ValueError("Judge input contains no examples.")
    before = sha256(FROZEN)
    env = {
        "IOT_PIPELINE_JUDGE_INPUT": str(input_path),
        "IOT_PIPELINE_JUDGE_OUT_DIR": str(JUDGE_DIR),
    }
    judge_args = ["--delay", str(delay)]
    if limit:
        judge_args += ["--limit", str(limit)]
    run_script(JUDGE, judge_args, env)
    if sha256(FROZEN) != before:
        raise RuntimeError("Frozen corpus hash changed during judging.")
    results = read_jsonl(JUDGE_RESULTS) if JUDGE_RESULTS.exists() else []
    mark_stage(state, "judge", "in_progress" if len(results) < len(rows) else "complete",
               input=str(input_path), input_count=len(rows), judged_count=len(results),
               results=str(JUDGE_RESULTS.relative_to(ROOT)))

def stage_prepare_human_audit(state: dict[str, Any], sample_size: int, seed: int) -> None:
    candidates = read_jsonl(GENERATED)
    if not JUDGE_RESULTS.exists():
        raise FileNotFoundError("Run judge before preparing the human audit sample.")
    judgments = {str(r.get("example_id")): r for r in read_jsonl(JUDGE_RESULTS)}
    accepted = [r for r in candidates if judgments.get(str(r.get("example_id")), {}).get("verdict") == "ACCEPT"]
    if len(accepted) < sample_size:
        raise ValueError(f"Only {len(accepted)} judge-accepted examples; cannot sample {sample_size}.")
    # Deterministic stratified sample by task_type, then document/chunk.
    import random
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in accepted:
        groups.setdefault(str(row.get("task_type", "unknown")), []).append(row)
    for group in groups.values():
        rng.shuffle(group)
    selected = []
    task_names = sorted(groups)
    while len(selected) < sample_size:
        progressed = False
        for task in task_names:
            if groups[task] and len(selected) < sample_size:
                selected.append(groups[task].pop())
                progressed = True
        if not progressed:
            break
    HUMAN_DIR.mkdir(parents=True, exist_ok=True)
    fields = ["example_id", "task_type", "chunk_id", "instruction", "response",
              "evidence_span", "human_decision", "reviewer_notes"]
    with HUMAN_SAMPLE.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in selected:
            writer.writerow({k: row.get(k, "") for k in fields if k not in ("human_decision", "reviewer_notes")})
    mark_stage(state, "prepare_human_audit", "complete", sample_size=len(selected),
               accepted_pool=len(accepted), seed=seed, file=str(HUMAN_SAMPLE.relative_to(ROOT)))
    print(f"Human audit sample created: {HUMAN_SAMPLE}")
    print(f"Judge-accepted pool: {len(accepted)}; sample size: {len(selected)}")
    print("Fill human_decision with ACCEPT or REJECT and reviewer_notes, then save as human_audit_decisions.csv.")

def stage_apply_human_audit(state: dict[str, Any]) -> None:
    if not HUMAN_DECISIONS.exists():
        raise FileNotFoundError(f"Copy completed review to {HUMAN_DECISIONS}")
    decisions = list(csv.DictReader(HUMAN_DECISIONS.open("r", encoding="utf-8-sig", newline="")))
    if not decisions:
        raise ValueError("Human decisions file is empty.")
    valid = {"ACCEPT", "REJECT"}
    seen = set()
    for row in decisions:
        eid = row.get("example_id", "").strip()
        decision = row.get("human_decision", "").strip().upper()
        if not eid or eid in seen or decision not in valid:
            raise ValueError(f"Invalid/duplicate human decision row: {row}")
        seen.add(eid)
        row["human_decision"] = decision
    known = {str(r.get("example_id")) for r in read_jsonl(GENERATED)}
    unknown = seen - known
    if unknown:
        raise ValueError(f"Human review contains unknown IDs: {sorted(unknown)[:10]}")
    write_json(HUMAN_APPROVED, {
        "reviewed_count": len(decisions),
        "accepted": sum(r["human_decision"] == "ACCEPT" for r in decisions),
        "rejected": sum(r["human_decision"] == "REJECT" for r in decisions),
        "decisions": decisions,
        "note": "Human decisions recorded; this does not automatically override Groq rejection.",
    })
    mark_stage(state, "apply_human_audit", "complete", reviewed=len(decisions),
               accepted=sum(r["human_decision"] == "ACCEPT" for r in decisions),
               rejected=sum(r["human_decision"] == "REJECT" for r in decisions),
               file=str(HUMAN_APPROVED.relative_to(ROOT)))

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True,
        choices=["status", "generate", "audit", "judge", "prepare-human-audit", "apply-human-audit", "all-automated"])
    parser.add_argument("--passes", type=int, default=1, choices=(1, 2, 3, 4))
    parser.add_argument("--delay", type=float, default=15.0)
    parser.add_argument("--judge-delay", type=float, default=20.0)
    parser.add_argument("--judge-limit", type=int, default=0)
    parser.add_argument("--input", default=str(GENERATED.relative_to(ROOT)),
        help="JSONL input for the judge stage.")
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--retry-skipped", action="store_true")
    args = parser.parse_args()

    state = load_state()
    if args.stage == "status":
        return stage_status()
    try:
        if args.stage == "generate":
            stage_generate(args, state)
        elif args.stage == "audit":
            stage_audit(state)
        elif args.stage == "judge":
            stage_judge(Path(args.input), args.judge_delay, args.judge_limit, state)
        elif args.stage == "prepare-human-audit":
            stage_prepare_human_audit(state, args.sample_size, args.seed)
        elif args.stage == "apply-human-audit":
            stage_apply_human_audit(state)
        elif args.stage == "all-automated":
            stage_generate(args, state)
            stage_audit(state)
            # Judge only after structural audit passes.
            stage_judge(GENERATED, args.judge_delay, args.judge_limit, state)
            stage_prepare_human_audit(state, args.sample_size, args.seed)
            print("\nPipeline paused for human audit. Complete the CSV, then run --stage apply-human-audit.")
        return 0
    except Exception as exc:
        print(f"\nPIPELINE STOPPED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())