# Deterministic admission benchmark

Means across independent run seeds; controlled synthetic results.

| Phase | Admission method | Faulty admitted | Valid rejected | Downstream success | Violations | Verification calls |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| online | first_success | 100.0% | 0.0% | 45.3% | 17.3% | 48 |
| online | independent_composition | 0.0% | 0.0% | 100.0% | 0.0% | 495 |
| online | independent_replay | 0.0% | 0.0% | 100.0% | 0.0% | 570 |
| online | self_verification | 0.0% | 0.0% | 100.0% | 0.0% | 570 |
| static | first_success | 100.0% | 0.0% | 45.3% | 17.3% | 48 |
| static | independent_composition | 0.0% | 0.0% | 100.0% | 0.0% | 495 |
| static | independent_replay | 0.0% | 0.0% | 100.0% | 0.0% | 570 |
| static | self_verification | 0.0% | 0.0% | 100.0% | 0.0% | 570 |

95% descriptive run-bootstrap intervals, paired effect sizes, exact denominators, fault-specific results, and checkpoint curves are in report.json and per_run.csv.
Intervals are conditional on this small frozen candidate pool. Zero-width intervals do not establish general reliability.
Self-verification is an in-process checking surrogate, not an LLM self-critique experiment. It shares inputs and checks with independent replay; process separation alone should not change decisions here.
Budgets are matched by allocated maximum tests; actual work and process overhead are reported separately.
Online results use staged submissions from the frozen pool, not adaptive candidate generation.
Model tokens and provider costs are zero. Reward sensitivity is post-hoc, not evidence of reward-driven learning.
