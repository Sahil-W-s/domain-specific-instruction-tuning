# Results and Analysis

## IoT task performance

The answerable held-out evaluation includes 30 items per baseline: 12 QA, 8 extraction, 5 summarization, and 5 classification.

B1 had the highest QA token F1 (59.95%), followed by B3 (59.25%) and B4 (59.02%). B4 had the highest summarization token F1 (52.23%), followed by B3 (51.42%). Classification EM was 100% for B2 and B3, but the task contains only five items.

## Structured extraction

Strict extraction string EM was misleading for JSON-formatted outputs, so a structured evaluation was conducted separately.

| Baseline | JSON parse | Structured EM | Element precision | Element recall | Element F1 |
|---|---:|---:|---:|---:|---:|
| B0 | 8/8 | 37.5% | 63.33% | 63.93% | 62.92% |
| B1 | 8/8 | 37.5% | 65.00% | 65.71% | 65.15% |
| B2 | 8/8 | 37.5% | 54.58% | 57.08% | 55.69% |
| B3 | 8/8 | 50.0% | 67.50% | 70.00% | 68.61% |
| B4 | 7/8 | 37.5% | 58.21% | 60.71% | 59.33% |
| B5 | 8/8 | 37.5% | 62.92% | 65.42% | 64.03% |

B3 led structured extraction in this sample. Field/value matching is exact after normalization and does not award semantic equivalence.

## General capability retention

| Baseline | Correct | Accuracy |
|---|---:|---:|
| B0 | 31/50 | 62% |
| B4 | 27/50 | 54% |
| B5 | 33/50 | 66% |

Exact McNemar comparisons:

| Comparison | First-only correct / second-only correct | p-value |
|---|---:|---:|
| B0 vs B4 | 5 / 1 | 0.2188 |
| B0 vs B5 | 3 / 5 | 0.7266 |
| B4 vs B5 | 1 / 7 | 0.0703 |

None reached p<0.05. B5's observed 12 percentage-point advantage over B4 is promising descriptively, but the paired comparison did not establish statistical significance. The 50-question subset is too small to support broad claims about general capability retention.

## Unanswerable responses

The preliminary manual assessment labels 25/30 responses unsupported or hallucinated, 3 safe abstentions, and 2 grounded but incomplete. Independent review is required before presenting these as validated rates.

## Training loss

Recorded run metadata reports:

| Baseline | Train loss | Evaluation loss | Runtime |
|---|---:|---:|---:|
| B2 | 1.4924 | 1.4890 | 1,726 s |
| B3 | 0.2199 | 0.3911 | 959 s |
| B4 | 0.2062 | 0.4171 | 927 s |
| B5 | 1.0409 | 1.0098 | 1,937 s |

Do not compare these losses across different dataset compositions as if they came from an identical validation distribution. B3/B4 share IoT validation examples; B5 validation includes general examples. Full trainer logs and loss curves are not archived, so these historical figures are supported by run metadata rather than complete trainer traces.

## Supported conclusion

The experiment demonstrates that domain fine-tuning changes task performance, but it does not establish that filtering improves performance over the reconstructed unfiltered candidate pool. B3 led structured extraction; B4 led summarization; B1 remained highly competitive on QA. B5 had the highest general-retention accuracy among B0/B4/B5, but paired testing did not establish a statistically significant difference.
