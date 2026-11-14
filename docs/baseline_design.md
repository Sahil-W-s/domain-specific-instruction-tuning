# Baseline Design and Comparison Logic

The six baselines isolate different interventions. They are not all pairwise controlled experiments; dataset composition and total training-set size must be considered when interpreting differences.

## Baseline definitions

| ID | Training/prompt condition | What it represents |
|---|---|---|
| B0 | Base model, zero-shot | No project-specific prompting or fine-tuning |
| B1 | Base model + four fixed demonstrations | In-context learning baseline without weight updates |
| B2 | 1,000 Dolly-style general examples | Generic instruction-tuning baseline |
| B3 | 600 reconstructed unfiltered IoT candidates | IoT adaptation from a reconstructed candidate pool |
| B4 | 577 filtered IoT examples | IoT adaptation from the filtered dataset |
| B5 | 577 filtered IoT + 577 general examples | Mixed-domain training condition |

## Intended comparisons

| Comparison | Intended question | Important qualification |
|---|---|---|
| B0 vs B1 | What changes when four demonstrations are added? | Prompt-only intervention; no parameter update |
| B0 vs B2 | Does generic instruction tuning change performance? | General-domain training baseline |
| B0 vs B3 | Does IoT fine-tuning change IoT task performance? | B3 is reconstructed; interpret descriptively |
| B0 vs B4 | Does filtered IoT fine-tuning change domain performance and general retention? | IoT test has 35 examples; MMLU retention has 50 |
| B3 vs B4 | Does the filtered dataset outperform the unfiltered candidate pool? | Not a clean causal filtering ablation: B3 is not a verified frozen pre-filter snapshot of B4 and record counts differ |
| B4 vs B5 | Does adding general data change the domain/retention trade-off? | B5 has 1,154 total examples versus B4's 577; the comparison also changes total training-set size |
| B2 vs B5 | How does mixed IoT/general training compare with general-only training? | Both dataset size and domain composition differ |

## What the results show

- B1 is highly competitive on QA (59.95 Token F1) despite having no trainable adapter.
- B3 achieves the strongest structured extraction element F1 (68.61%) and a 50% structured exact match over eight extraction items.
- B4 achieves the strongest summarization Token F1 (52.23%) over five items.
- Classification EM is 100% for B2 and B3, but it is based on only five examples.
- B5 reaches 66% on the 50-question MMLU subset, compared with 62% for B0 and 54% for B4. Paired exact McNemar tests did not show statistically significant differences at p<0.05.

## Interpretation rules

1. Do not claim that filtering caused a measured performance improvement or degradation based only on B3 vs B4.
2. Do not compare validation loss across B3/B4 and B5 as if they used the same validation distribution.
3. Do not claim broad capability retention from 50 MMLU questions.
4. Report denominators with percentages, especially for five- and eight-example tasks.
5. Describe B5 as a mixed-data condition that changes both composition and total number of training examples.
