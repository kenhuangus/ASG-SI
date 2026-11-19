# Recorded implementation results

These results were validated on 18 November 2026 with Python 3.12.14.
The evaluated implementation is commit `71657fac5c7e135daf8a88199cc6585186abdda3`.

- [Readable result tables](report.md)
- [Per-run metrics and denominators](per_run.csv)
- [Descriptive confidence intervals and paired differences](report.json)
- [Complete audit replay result](audit_report.json)
- [Composition summary](composition_summary.json)
- [Implementation progress PDF](ASG-SI_Implementation_Progress.pdf)
- [Full source and raw evidence archive](ASG-SI_implementation_and_results.zip)
- [File checksums](SHA256SUMS)

The archive contains the complete per-decision and per-task records, promotion
evidence, split manifests, library checkpoints, composition traces, source code,
and a patch. It was extracted into a fresh directory and the full audit replay
reproduced 400 admission bundles, 3,700 candidate test executions, and 7,000
downstream task results and rewards. Larger raw artifacts are kept in the
archive to keep this repository browsable.

To reproduce a new run from the repository root:

```bash
python3 -m unittest discover -s tests -v
python3 -m experiments.run --config configs/smoke.json --output runs/new_run
python3 -m experiments.report runs/new_run
python3 -m experiments.audit runs/new_run
```

To replay the supplied raw evidence with the recorded Python version, extract
the archive, change into its `ASG-SI` directory, and run:

```bash
python3 -m experiments.audit runs/review_run
```

These are controlled deterministic experiments with 10 constructed candidates,
not real-LLM training results. The in-process checking surrogate and independent
replay use identical tests and criteria and produce identical decisions.
Verification costs include actual tool calls and measured process overhead.
Intervals are conditional on the small frozen candidate pool. The progress PDF
records the implementation's pre-publication status; repository publication is
separate from the validated experiment.
