# Project Status and Remaining Work

## Preserved and committed

- [x] B0–B5 experiment definitions and IoT predictions are archived locally
- [x] B2–B5 QLoRA training runs and adapter archives are preserved in the frozen reproducibility ZIP
- [x] Dataset files and train/validation split manifest are preserved in the frozen ZIP
- [x] Structured extraction reevaluation is preserved
- [x] MMLU retention evaluation for B0/B4/B5 is preserved
- [x] Dataset-level human review: two distinct 30-example batches (29/30 ACCEPT in the original-v0 batch; 28/30 ACCEPT in the follow-up candidate-pool batch)
- [x] Reproducibility archive and checksum verification
- [x] Original chunking, generation, filtering, judging, dataset-preparation, and QLoRA training scripts committed
- [x] Dataset pipeline orchestrator, configuration, and cross-model judge script committed
- [x] Methodology, results, provenance, human-review, limitations, and script-catalog documentation committed

## Work still required before a strong public release

- [ ] Independently review the 30 model responses to unanswerable IoT questions
- [ ] Recover or reconstruct the B0–B5 model-inference prompt templates and scoring implementation; label reconstructed components
- [ ] Add an evaluation runner and automated tests for prediction coverage, test-set hash alignment, metric parsing, and summary regeneration
- [ ] Reconcile and test the orchestration runner in a clean local environment with the required frozen corpus and directory layout
- [ ] Add dataset provenance, source licensing, and redistribution statements
- [ ] Decide whether training datasets, held-out examples, human-review rows, and adapter weights can legally be redistributed
- [ ] Decide the large-artifact strategy (Git LFS, GitHub Releases, or external model/data hosting)
- [ ] Prepare the full research report and demonstration
- [ ] Run a clean-environment documentation/reproduction check

## Historical-artifact policy

The downloaded reproducibility ZIP is the historical source of truth. Never silently replace its predictions, metrics, datasets, or model adapters. Store new analyses separately and label them as recalculated, reconstructed, or newly run.

## Current repository scope

The public GitHub repository currently emphasizes source code, documentation, provenance summaries, and non-sensitive aggregate review results. The frozen training datasets, source-document corpus, raw review CSVs, prediction JSONL files, and adapter binaries have not been uploaded. This is intentional pending licensing review and a large-file strategy.
