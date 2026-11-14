# Project Status and Remaining Work

## Preserved and completed

- B0–B5 experiment definitions and IoT predictions
- B2–B5 QLoRA training runs and adapter archives
- Dataset files and train/validation split manifest
- Structured extraction reevaluation
- MMLU retention evaluation for B0/B4/B5
- Preliminary unanswerable-response assessment
- Reproducibility archive and checksum verification

## Work still required before public release

- [ ] Human review of a representative 5–10% dataset sample, with review protocol and error count recorded
- [ ] Independent review of the 30 unanswerable-response labels
- [ ] Recover or reconstruct inference prompt templates and scoring code; label reconstructed components
- [ ] Add evaluation scripts and tests for prediction-file integrity
- [ ] Add dataset provenance, source licensing, and redistribution statements
- [ ] Decide whether datasets and adapters can legally be redistributed
- [ ] Prepare final research report and short demo
- [ ] Decide large-file strategy (Git LFS or release assets)
- [ ] Test the documented process in a clean environment where feasible

## Publication rule

The frozen experiment package is the historical source of truth. Never silently replace its predictions, metrics, or datasets. Store new analyses separately and label them as recalculated, reconstructed, or newly run.
