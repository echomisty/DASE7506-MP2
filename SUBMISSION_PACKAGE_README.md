# DASE7506 MP2 Submission Package

## Selected checkpoint

The archive contains the four answer-group rankers plus linear candidate ranker, frozen on 2026-10-15. Its existing public-Test score is 382/467 (81.7987%), recorded once on that checkpoint. Test was not rerun during this restoration. The later MLP alternative scored 377/467. The user requested restoration after seeing this comparison, so Test influenced package-level model selection. This is a documented deviation from the coursework rule; the historical score is not an unbiased estimate for a test-blind selected package.

## Matching evidence

- Validation: 745/798 (93.3584%); ambiguous Validation: 302/355 (85.0704%).
- Strict nested Train CV, recomputed for the linear candidate: 1743/2114 ambiguous questions (82.4503%).
- Validation resources: three runs, maximum 1.0063 seconds total. Historical Test resource record: one run, 1.2920 seconds total.
- Checkpoint arrays: 259,032 bytes uncompressed; course limit: 16 MiB.
- Raw Test predictions are not retained. The official aggregate score and resource record are included.

## Upload contents

- `code/`, `tests/`, `guide/`: implementation, training and scoring commands, checkpoint, tests, and assignment guide.
- `code/data/`: benchmark data, license, attribution, and manifest.
- `experiments/`: baselines, ablations, validation diagnostics, linear nested-CV record, and final configuration.
- `results/final/`: Validation predictions, score and resources; historical aggregate Test score and resources; SHA-256 records.
- `report/` and `output/pdf/`: reproducible report source, Chinese and English full reports (11 pages), and exact 10-page submission editions. The PDFs now match the restored linear checkpoint and document the Test-selection deviation.
- `AI_DISCLOSURE.md`, `PROJECT_OVERVIEW.md`, `REQUIREMENTS.md`: project overview, requirements, and AI/reuse disclosure.

For the coursework report, use the exact 10-page Chinese or English edition. The archive itself documents the Test-isolation issue; check course rules before submitting the 81.80% score as a test-blind result. Website submission also requires creating the prepared GitHub issue.
