# Reproducibility

## Frozen experiment archive

The experiment archive is `iot_cp_reproducibility_package_v1.zip`. Its final audit recorded 39 directory files and 39 matching ZIP entries. All 37 SHA-256 manifest entries passed; all four nested adapter ZIPs passed integrity checks; the outer ZIP passed integrity verification.

## Included artifacts

- `code/train_iot_baseline.py`
- Source training JSONL datasets and held-out IoT test
- `configuration/b1_fewshot_examples.json`
- `configuration/split_manifest.json`
- `configuration/phase3_validation_report.json`
- B2–B5 adapter ZIPs
- B2–B5 run metadata
- B0–B5 IoT predictions
- B0/B4/B5 raw and corrected MMLU predictions
- Evaluation summaries, per-example metrics, structured extraction metrics, and preliminary unanswerable assessment
- SHA-256 manifest

## External dependency

The base model weights are not included. Obtain `Qwen/Qwen2.5-1.5B-Instruct` separately and follow its terms.

## Historical environment

- PyTorch 2.11.0+cu130
- Transformers 5.20.2
- PEFT 0.21.2
- Accelerate 1.15.0
- Datasets 4.8.5
- bitsandbytes 0.50.2

Training ran on a Tesla T4 using FP16 compute.

## Reproduction gaps

The archive does not include inference/evaluation scripts, complete original inference prompt construction, the full scoring implementation, full trainer logs, trainer states, checkpoints, or loss curves. Saved predictions and metrics are preserved research artifacts, but exact end-to-end regeneration has not yet been demonstrated.

Do not overwrite the frozen archive. Save reconstructed scripts, independently recalculated metrics, and new experiments separately, with clear provenance labels.
