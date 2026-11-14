# Project Status and Remaining Work

## Preserved and completed

- [x] B0–B5 experiment definitions and IoT predictions
- [x] B2–B5 QLoRA training runs and adapter archives
- [x] Dataset files and train/validation split manifest
- [x] Structured extraction reevaluation
- [x] MMLU retention evaluation for B0/B4/B5
- [x] Dataset-level human review: two distinct 30-example batches (29/30 ACCEPT in the original-v0 batch; 28/30 ACCEPT in the follow-up candidate-pool batch)
- [x] Reproducibility archive and checksum verification
- [x] Original chunking, generation, filtering, judging, dataset-preparation, and training scripts added to the repository

## Work still required before public release

- [ ] Independent review of the 30 model responses to unanswerable questions
- [ ] Reconcile the separate orchestration bundle (`pipeline.py`, `pipeline_config.json`, `pipeline_judge.py`) with the scripts committed here
- [ ] Recover or reconstruct model-inference prompt templates and scoring code; label reconstructed components
- [ ] Add evaluation scripts and tests for prediction-file integrity
- [ ] Add dataset provenance, source licensing, and redistribution statements
- [ ] Decide whether datasets and adapters can legally be redistributed
- [ ] Prepare final research report and short demo
- [ ] Decide large-file strategy (Git LFS or release assets)
- [ ] Test the documented process in a clean environment where feasible

## Publication rule

The frozen experiment package is the historical source of truth. Never silently replace its predictions, metrics, or datasets. Store new analyses separately and label them as recalculated, reconstructed, or newly run.
