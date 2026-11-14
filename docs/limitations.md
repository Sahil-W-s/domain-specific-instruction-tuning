# Limitations and Open Validation Items

1. **Small IoT evaluation:** 35 held-out items, with only 5 classification, 5 summarization, and 5 unanswerable examples.
2. **Small general benchmark:** 50 MMLU questions; only a limited proxy for general capability.
3. **Limited retention coverage:** Only B0, B4, and B5 have saved general-retention predictions.
4. **Preliminary manual assessment:** Unanswerable-response labels need independent human review.
5. **Dataset human-review gap:** The Project 7 rubric calls for a 5–10% human review sample to estimate error and judge agreement. The preserved package does not establish completion of this item.
6. **B3 provenance:** B3 is reconstructed, not a verified pre-filter snapshot of B4.
7. **Validation composition:** B3/B4 share IoT validation examples, while B5 validation combines IoT and general data. Their losses are not directly comparable.
8. **Incomplete training trace:** Full trainer logs, checkpoints, trainer state, and loss curves are absent.
9. **Incomplete inference trace:** Prompt construction and complete scoring code are not included, so exact prediction/metric regeneration has not been verified.
10. **Single-seed evidence:** The preserved results do not establish multi-seed robustness.
11. **Ablations not evidenced:** Dataset-size scaling, filter threshold, generator comparison, grounded-vs-ungrounded generation, LoRA rank/epoch, mixing-ratio sweeps, and judge-human agreement are not established by the preserved artifact inventory.

These limitations should remain visible in the final report. Do not claim statistical significance or complete rubric compliance where evidence is absent.
