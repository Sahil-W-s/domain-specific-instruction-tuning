import json, hashlib, random
from pathlib import Path
from datetime import datetime, timezone
from datasets import load_dataset
from pydantic import BaseModel, ValidationError

ROOT = Path(".")
OUT = ROOT / "data" / "phase3"
FILES = {
    "raw": ROOT / "data/interim/dataset_v0_raw.jsonl",
    "prior": ROOT / "data/interim/improved_generation/dataset_v0_additional.jsonl",
    "v2": ROOT / "data/interim/production_generation_v2/dataset_v0_additional_v2_resumed.jsonl",
    "b4": ROOT / "data/frozen/dataset_v1_filtered.jsonl",
    "chunks": ROOT / "data/frozen/chunks_v1_frozen.jsonl",
}
EXPECTED_B4 = "9e541db71c7d5325e3817cd8c29469cfac7137aad0a399cb8b6a88a2e72007a3"
EXPECTED_CHUNKS = "67e90d3428ff0a421111e19916ceb2dfc007460ae047f18131131be2f80bb827"
SEED = 20261114

class Example(BaseModel):
    instruction: str
    input: str
    output: str
    source_id: str
    metadata: dict

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]

def write_jsonl(path, rows):
    with path.open("x", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def validate(rows, label):
    for i, row in enumerate(rows):
        try:
            Example(**row)
        except ValidationError as e:
            raise ValueError(f"{label} row {i}: {e}") from e

def convert_domain(row, label, chunks):
    for k in ("example_id", "instruction", "response", "chunk_id", "evidence_span"):
        if not row.get(k):
            raise ValueError(f"{label}: {row.get('example_id')} missing {k}")
    cid, evidence = row["chunk_id"], row["evidence_span"]
    if cid not in chunks or evidence not in chunks[cid]:
        raise ValueError(f"Evidence provenance failure: {row['example_id']}")
    return {
        "instruction": row["instruction"],
        "input": evidence,
        "output": row["response"],
        "source_id": cid,
        "metadata": {
            "example_id": row["example_id"],
            "task_type": row.get("task_type", "unspecified"),
            "dataset_version": label,
            "generator_model": row.get("generator_model"),
            "evidence_block_ids": row.get("evidence_block_ids", []),
            "evidence_span": evidence,
            "pool_source": row.get("pool_source"),
            "human_audit_status": row.get("human_audit_status"),
        },
    }

def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"{OUT} already contains files; refusing overwrite")
    OUT.mkdir(parents=True, exist_ok=True)

    for key, p in FILES.items():
        if not p.exists():
            raise FileNotFoundError(p)
    source_hashes = {k: digest(p) for k, p in FILES.items()}
    if source_hashes["b4"] != EXPECTED_B4 or source_hashes["chunks"] != EXPECTED_CHUNKS:
        raise ValueError("Frozen B4 or chunk corpus hash mismatch")

    raw, prior, v2, b4_rows, chunk_rows = [read_jsonl(FILES[k]) for k in ("raw","prior","v2","b4","chunks")]
    if (len(raw), len(prior), len(v2), len(b4_rows)) != (88,27,485,577):
        raise ValueError("Unexpected source record counts")

    chunks = {r["chunk_id"]: r["content"] for r in chunk_rows}
    if len(chunks) != len(chunk_rows):
        raise ValueError("Duplicate frozen chunk IDs")

    b3_src = raw + prior + v2
    ids = [r.get("example_id") for r in b3_src]
    if len(ids) != 600 or len(set(ids)) != 600 or None in ids:
        raise ValueError("B3 candidate IDs are not 600 unique IDs")

    b3 = [convert_domain(r, "B3_reconstructed_unfiltered", chunks) for r in b3_src]
    b4 = [convert_domain(r, "B4_frozen_filtered", chunks) for r in b4_rows]
    validate(b3, "B3")
    validate(b4, "B4")

    print("Downloading/loading Databricks Dolly 15k...")
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    dolly = [dict(r) for r in ds]
    if len(dolly) < 1577:
        raise ValueError(f"Dolly has {len(dolly)} rows; 1577 required")
    rng = random.Random(SEED)
    ix = list(range(len(dolly)))
    rng.shuffle(ix)
    b2_ix, b5_ix = ix[:1000], ix[1000:1577]
    if set(b2_ix) & set(b5_ix):
        raise AssertionError("B2/B5 Dolly subsets overlap")

    def cv_dolly(r, label):
        ins, ctx, out = (r.get("instruction") or "").strip(), (r.get("context") or "").strip(), (r.get("response") or "").strip()
        if not ins or not out:
            raise ValueError("Dolly row has empty instruction/response")
        return {
            "instruction": ins, "input": ctx, "output": out,
            "source_id": "dolly15k:" + str(r.get("category") or "uncategorized"),
            "metadata": {"dataset": "databricks/databricks-dolly-15k", "category": r.get("category"), "assignment": label}
        }

    b2 = [cv_dolly(dolly[i], "B2_generic_only") for i in b2_ix]
    b5g = [cv_dolly(dolly[i], "B5_general_mix") for i in b5_ix]
    b5d = [dict(r) for r in b4]
    b5 = b5d + b5g
    rng.shuffle(b5)
    for label, rows in (("B2",b2),("B3",b3),("B4",b4),("B5 general",b5g),("B5",b5)):
        validate(rows, label)

    outputs = {
        "b2_dolly_1000.jsonl": b2,
        "b3_iot_unfiltered_reconstructed_600.jsonl": b3,
        "b4_iot_filtered_577.jsonl": b4,
        "b5_domain_577.jsonl": b5d,
        "b5_general_577.jsonl": b5g,
        "b5_iot_plus_dolly_1154.jsonl": b5,
    }
    for name, rows in outputs.items():
        write_jsonl(OUT / name, rows)

    out_hashes = {name: digest(OUT/name) for name in outputs}
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_DATASET_PREPARATION",
        "pydantic_validation": "PASS",
        "counts": {"B2":len(b2),"B3":len(b3),"B4":len(b4),"B5_domain":len(b5d),"B5_general":len(b5g),"B5_total":len(b5)},
        "B3_label": "Reconstructed from 88 original + 27 prior additional + 485 V2 candidates; not frozen before filtering.",
        "Dolly": {"dataset":"databricks/databricks-dolly-15k","rows_available":len(dolly),"seed":SEED,"B2_count":len(b2_ix),"B5_count":len(b5_ix),"B2_B5_disjoint":True},
        "source_hashes":source_hashes,"output_hashes":out_hashes,
        "frozen_sources_modified":False,
    }
    (OUT/"phase3_validation_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))

if __name__ == "__main__":
    main()