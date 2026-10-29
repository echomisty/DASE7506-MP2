# DASE7506 MP2 Submission Package

## Selected checkpoint

The final package uses the four answer-group rankers plus the linear candidate ranker, frozen on 2026-10-15. This checkpoint's existing public-Test score is 382/467 (81.7987%), recorded once. The package was restored on 2026-10-28 after its Test score was compared with the later MLP alternative (377/467). After restoration, no model or checkpoint changes were made and Test was not rerun. Because the comparison informed the restoration, Test influenced final package-level version selection; 81.7987% is therefore a historical score, not an independent test-blind estimate.

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
