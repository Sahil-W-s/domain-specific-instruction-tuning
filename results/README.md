# Machine-Readable Results

`experiment_metrics_summary.json` contains the aggregate metrics transcribed from the frozen experiment results and run metadata. It preserves denominators, evaluation subset sizes, task categories, paired-test outcomes, and caveats.

Important interpretation limits:
- The initial extraction EM is not the preferred metric; use the structured extraction evaluation.
- The MMLU set contains only 50 questions and was evaluated for B0/B4/B5 only.
- The unanswerable-response labels are preliminary manual assessments.
- The run losses correspond to different validation compositions and should not be treated as directly comparable.
- Per-example inputs and predictions are not included in this public aggregate JSON.
