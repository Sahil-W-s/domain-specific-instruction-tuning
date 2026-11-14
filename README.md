# IoT Domain-Specific Instruction Tuning

Research project investigating whether validated, grounded IoT instruction-response data improves domain task performance while preserving general-domain capability.

**Base model:** `Qwen/Qwen2.5-1.5B-Instruct`  
**Adaptation:** 4-bit QLoRA with FP16 compute  
**Hardware:** NVIDIA Tesla T4  
**Project rubric:** Project 7 — Domain-Specific Instruction-Tuning Dataset Curation Framework.

> **Research status:** Core training and evaluation runs have been completed and their artifacts archived. This repository is the documentation baseline. Some reproduction components and human-review requirements remain open; see [limitations](docs/limitations.md) and [project status](docs/project_status.md).

## Research question

Does a validated, grounded domain dataset improve IoT accuracy over the base model and reconstructed unfiltered data, without unacceptable loss of general capability?

## Baselines

| ID | Description |
|---|---|
| B0 | Base model, zero-shot |
| B1 | Base model + four fixed few-shot demonstrations |
| B2 | QLoRA on 1,000 Dolly-style general instruction examples |
| B3 | QLoRA on 600 reconstructed unfiltered IoT candidates |
| B4 | QLoRA on 577 filtered IoT examples |
| B5 | QLoRA on 577 IoT + 577 general examples (1,154 total) |

Adapters are preserved for B2–B5. B0 and B1 use the base model without trained adapters.

**Provenance caution:** B3 is a reconstructed unfiltered candidate pool, not a verified pre-filter snapshot of B4. Do not describe B3/B4 as a perfectly controlled raw-versus-filtered pair.

## IoT held-out evaluation

The held-out test contains 35 examples: 12 QA, 8 extraction, 5 summarization, 5 unanswerable, and 5 classification. Answerable task results cover 30 items per baseline.

| Baseline | Initial Extraction EM* | Summarization Token F1 | QA Token F1 | Classification EM |
|---|---:|---:|---:|---:|
| B0 | 0.00% | 43.03% | 23.99% | 20.00% |
| B1 | 0.00% | 47.02% | 59.95% | 60.00% |
| B2 | 0.00% | 31.55% | 52.15% | 100.00% |
| B3 | 50.00% | 51.42% | 59.25% | 100.00% |
| B4 | 37.50% | 52.23% | 59.02% | 40.00% |
| B5 | 37.50% | 49.49% | 56.06% | 80.00% |

*The initial extraction string metric is sensitive to JSON formatting; see the separate structured extraction analysis below.

### Structured extraction

| Baseline | JSON parse success | Structured EM | Element F1 |
|---|---:|---:|---:|
| B0 | 8/8 | 37.50% | 62.92% |
| B1 | 8/8 | 37.50% | 65.15% |
| B2 | 8/8 | 37.50% | 55.69% |
| B3 | 8/8 | 50.00% | 68.61% |
| B4 | 7/8 | 37.50% | 59.33% |
| B5 | 8/8 | 37.50% | 64.03% |

Structured element matching uses normalized exact field/value matching, not semantic similarity. B3 led these extraction metrics on this small sample.

### General capability retention

A 50-question MMLU test subset was evaluated for B0, B4, and B5 only.

| Baseline | Correct | Accuracy |
|---|---:|---:|
| B0 | 31/50 | 62% |
| B4 | 27/50 | 54% |
| B5 | 33/50 | 66% |

Reported exact McNemar p-values: B0 vs B4 0.2188; B0 vs B5 0.7266; B4 vs B5 0.0703. None reached p<0.05. Differences are descriptive and do not establish statistically significant retention gains.

### Unanswerable responses

A preliminary manual assessment of 30 responses reported 25 unsupported/hallucinated, 3 safe abstentions, and 2 grounded-but-incomplete responses (83.33% pooled unsupported/hallucinated). This is not independently validated and must not be presented as a final hallucination rate until reviewed.

## Training configuration

- QLoRA: 4-bit NF4, double quantization, FP16 compute
- LoRA rank 16, alpha 32, dropout 0.05
- Targets: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- 3 epochs, learning rate 2e-4, seed 42
- Per-device batch size 1, gradient accumulation 8
- Cosine scheduler, warmup ratio 0.03, weight decay 0.01
- `paged_adamw_8bit`, gradient checkpointing, max sequence length 3,072
- Assistant-only loss masking

Historical environment: PyTorch 2.11.0+cu130, Transformers 5.20.2, PEFT 0.21.2, Accelerate 1.15.0, Datasets 4.8.5, bitsandbytes 0.50.2.

## Repository guide

- [Methodology](docs/research_methodology.md)
- [Results and analysis](docs/results_and_analysis.md)
- [Reproducibility](docs/reproducibility.md)
- [Limitations](docs/limitations.md)
- [Project status and remaining work](docs/project_status.md)

## Data, models, and licensing

Large datasets and adapter archives are not included in this initial documentation commit. Review source-document, dataset, and model redistribution terms before publishing any data or weights. The base model weights are external and are not part of the experiment archive. See [reproducibility notes](docs/reproducibility.md).

## Main interpretation

The evidence shows that domain fine-tuning changes task performance, but does not establish that filtering improved results over the reconstructed unfiltered candidate pool. B3 led structured extraction; B4 led summarization; B1 remained competitive on QA. B5 achieved the highest MMLU subset accuracy among the three evaluated models, but the paired test did not establish a statistically significant difference.
