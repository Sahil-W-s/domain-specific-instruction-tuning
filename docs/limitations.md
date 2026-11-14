# Limitations and Open Validation Items

1. **Small IoT evaluation:** 35 held-out items, with only 5 classification, 5 summarization, and 5 unanswerable examples.
2. **Small general benchmark:** 50 MMLU questions; only a limited proxy for general capability.
3. **Limited retention coverage:** Only B0, B4, and B5 have saved general-retention predictions.
4. **Preliminary model hallucination assessment:** The 30 labels for responses to unanswerable questions need independent human review.
5. **Dataset human review:** Two completed 30-example review batches are documented. The follow-up sample contains 6 previously validated records and 24 v2 records, with 28 ACCEPT and 2 REJECT. The samples come from different stages; do not pool them or present their rejection proportions as unbiased estimates for the final 577-example dataset without a defined sampling frame.
6. **B3 provenance:** B3 is reconstructed, not a verified pre-filter snapshot of B4.
7. **Validation composition:** B3/B4 share IoT validation examples, while B5 validation combines IoT and general data. Their losses are not directly comparable.
8. **Incomplete training trace:** Full trainer logs, checkpoints, trainer state, and loss curves are absent.
9. **Incomplete inference trace:** Complete held-out inference prompt construction and scoring implementation are not contained in the frozen experiment archive, so exact prediction/metric regeneration has not been verified.
10. **Single-seed evidence:** The preserved results do not establish multi-seed robustness.
11. **Ablations not evidenced:** Dataset-size scaling, filter-threshold sweeps, generator comparisons, grounded-vs-ungrounded generation comparison, LoRA rank/epoch sweeps, mixing-ratio sweeps, and judge-human agreement analyses are not established by the preserved artifact inventory.

These limitations should remain visible in the final report. Do not claim statistical significance or complete rubric compliance where evidence is absent.
