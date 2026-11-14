# Script Catalog and Provenance

Scripts are historical implementation files recovered from project uploads. Their presence does not mean every script was used in every final experiment or that they can run without the original data directory and environment.

| Path | Role |
|---|---|
| `scripts/08b_create_semantic_chunks.py` | Semantic chunk construction |
| `scripts/28_evaluate_dataset.py` | Initial LLM-based dataset assessment |
| `scripts/29_generate_production_batch.py` | Initial production batch generation |
| `scripts/31_audit_generated_examples.py` | Generated-data audit |
| `scripts/32_audit_evidence_blocks.py` | Evidence block provenance audit |
| `scripts/36_evaluate_dataset_groq.py` | Groq judge evaluation |
| `scripts/37_inspect_rejected_examples.py` | Rejected-example inspection |
| `scripts/38_generate_improved.py` | Improved generation pilot |
| `scripts/41_generate_improved_safe.py` | Safer isolated generation |
| `scripts/42_audit_improved_pilot.py` | Improved pilot audit |
| `scripts/43_evaluate_improved_pilot_groq.py` | Improved pilot judge |
| `scripts/44_prepare_human_audit_30.py` | Prepare human-review sample |
| `scripts/45_production_heuristic_filter.py` | Production heuristic filtering |
| `scripts/46_apply_approved_deduplication.py` | Apply reviewed duplicate decisions |
| `scripts/48_reconcile_judge_results_fixed.py` | Reconcile judge results |
| `scripts/49_repair_evidence_candidates.py` | Create evidence-repair candidates |
| `scripts/50_judge_repaired_examples_groq.py` | Judge repaired candidates |
| `scripts/51_repair_hma_evidence.py` | Repair human/judge ambiguity evidence |
| `scripts/53_judge_hma_repair_groq_fixed.py` | Judge HMA repair candidates |
| `scripts/54_finalize_judge_reconciliation.py` | Finalize candidate reconciliation |
| `scripts/55_generate_production_v2.py` | Production generation v2 |
| `scripts/56_audit_generation_v2.py` | Production v2 audit |
| `scripts/57_apply_v2_manual_review.py` | Apply manual review decisions |
| `scripts/58_judge_v2_candidates_groq.py` | Cross-model judge for v2 candidates |
| `scripts/59_merge_and_audit_validated_pool.py` | Merge and audit candidate pool |
| `scripts/60_generate_production_v2_resumable.py` | Resumable production generation |
| `scripts/61_audit_generation_v2.py` | Comprehensive v2 audit |
| `training/prepare_phase3_datasets.py` | Earlier Phase 3 export |
| `training/prepare_phase3_datasets_v2.py` | Hardened Phase 3 export with exact evidence reconstruction |
| `training/train_iot_baseline.py` | QLoRA trainer for B2–B5 |
| `scripts/diagnostics/33_*.py`–`35d_*.py` | Judge tests and benchmark iterations |
| `scripts/diagnostics/39_*.py`, `40_*.py` | Provider/API diagnostics |
