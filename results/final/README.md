# Restored Linear-Candidate Checkpoint

The package uses the linear-candidate stack frozen on 2026-10-15. Its existing one-time Test score is 382/467 = 81.7987%. Test was not rerun during restoration. The later MLP alternative scored 377/467; the user requested the linear version after seeing that comparison. Test therefore influenced package-level model selection, which does not satisfy the coursework's Test-isolation rule. The 81.80% record is historical and is not a test-blind final estimate.

- Validation: 745/798 = 93.3584%; ambiguous questions: 302/355 = 85.0704%.
- Strict nested Train CV: 1743/2114 ambiguous questions = 82.4503%.
- Test resources: one recorded run, 1.2920 seconds total; checkpoint arrays: 259,032 bytes uncompressed (16 MiB limit).
- `test-score.json` and `test-resource-report.json` contain the aggregate score and historical run record. Raw Test predictions are not included.
- Validation predictions, score, and resources were regenerated for the restored checkpoint.
