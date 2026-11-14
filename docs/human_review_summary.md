# Human Review Summary

Two distinct 30-example human-review batches are preserved in the local experiment records. They represent separate review stages and are not pooled.

## Batch A — original dataset v0

- Sample size: 30
- ACCEPT: 29
- REJECT: 1
- Within-sample acceptance: 96.7%
- Within-sample rejection: 3.3%
- The rejected record was identified as a near-duplicate.

## Batch B — follow-up candidate-pool review

- Sample size: 30
- Source composition: 6 previously validated records and 24 v2 records
- ACCEPT: 28
- REJECT: 2
- Within-sample acceptance: 93.3%
- Within-sample rejection: 6.7%

The two rejected records were flagged because:
1. One answer added a TCP-connection claim that was not explicitly supported by the cited evidence.
2. One instruction was declarative rather than an actionable question, and the response repeated it.

## Interpretation

These are separate sample frames from different stages and must not be pooled into a single 60-item rate. The rejection percentages are observed within the reviewed samples, not unbiased whole-dataset error estimates unless the population and sampling frame are established.

The follow-up review contains 30 completed decisions with reviewer notes. This addresses the dataset-level human-review item in the Project 7 checklist. The exact percentage of the final 577-record B4 dataset represented by the sample is not asserted because the sample mixes previously validated and v2 candidate records.

## Public-release caution

The full CSVs contain instructions, generated responses, and source evidence excerpts. They are retained in the local research record but not published here until source-document redistribution rights are confirmed.
