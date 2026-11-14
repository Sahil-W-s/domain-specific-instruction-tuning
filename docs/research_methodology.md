# Research Methodology

## Objective

Investigate whether a small instruction-tuned language model can be adapted to IoT-oriented tasks using curated domain instruction-response data, and whether mixing general instruction data may mitigate loss of general-domain capability.

## Model and adaptation

The base model is `Qwen/Qwen2.5-1.5B-Instruct`. B2–B5 were trained using 4-bit NF4 QLoRA with FP16 compute. LoRA rank was 16, alpha 32, dropout 0.05, targeting attention projections and MLP gate/up/down projections.

Each run used three epochs, learning rate 2e-4, seed 42, per-device batch size 1, gradient accumulation 8, cosine scheduling, warmup ratio 0.03, weight decay 0.01, and assistant-only loss masking. Maximum sequence length was 3,072.

## Baselines

- **B0:** Base model, zero-shot.
- **B1:** Base model with four fixed few-shot demonstrations selected from B4 training data.
- **B2:** General-only fine-tuning on 1,000 Dolly-style records.
- **B3:** Fine-tuning on a 600-record reconstructed unfiltered IoT candidate pool.
- **B4:** Fine-tuning on 577 filtered IoT examples.
- **B5:** Fine-tuning on 577 IoT and 577 general examples.

## Data partitioning and provenance

Recorded train/validation counts: B2 900/100, B3 542/58, B4 519/58, and B5 1,038/116. B3 and B4 share IoT validation examples. B5 validation includes both IoT and general-domain examples, so validation losses across these conditions are not directly comparable.

The held-out IoT test contains 35 items and was held apart from training and validation. The general-retention subset contains 50 MMLU test questions selected with seed 42 after excluding nine potentially overlapping source indices.

B3 is a reconstructed unfiltered candidate pool rather than a verified frozen pre-filter snapshot of B4. This limits causal interpretation of the B3/B4 comparison.

## Evaluation

IoT task metrics include exact match for classification, an initial strict string exact match for extraction, and token F1 for QA and summarization. Because strict string EM was sensitive to JSON formatting, extraction was separately reevaluated using JSON parsing, structured exact match, and element-level precision, recall, and F1. Element comparisons use normalized exact field/value matches, not semantic equivalence.

General retention was measured on 50 MMLU questions for B0, B4, and B5 only. Paired exact McNemar tests were reported for the three pairwise comparisons.

Five unanswerable IoT questions were evaluated for each baseline, producing 30 responses. The current labels are preliminary manual assessments and require independent review.

## Interpretation boundary

The evaluation sets are small. Results are descriptive and should not be generalized beyond the sampled tasks. No statistically significant general-retention difference was found in the reported paired comparisons. B3 is reconstructed and cannot be represented as a verified raw pre-filter version of B4.
