# DASE7506 MP2 Project Overview

## Project in one minute

This project builds a lightweight offline verifier for four language-model
solutions to each GSM8K question. The system does not generate a new solution.
It parses the supplied candidates' final numeric answers, groups candidates
that produce the same answer, ranks the answer groups, and returns one candidate
index from `0` to `3`.

The official metric is selection accuracy: a question is correct when the
selected candidate's parsed final numeric answer equals the reference answer.
Intermediate reasoning quality is not graded directly.

## Recommended method

1. Parse final answers with the same rules as the official scorer.
2. Group candidates by canonical parsed answer.
3. Resolve single-answer groups deterministically.
4. Rank ambiguous answer groups using consensus, trace structure,
   question-candidate consistency, and the supplied 1024-dimensional
   embeddings.
5. Select one deterministic candidate index from the winning answer group.

The learned component is a small stack trained only on the provided training
references. Four answer-group rankers capture aggregate evidence; a fifth
candidate-level ranker preserves the strongest individual reasoning trace
inside each answer group. Inference uses NumPy and the Python standard library
only. The frozen stack also includes a linear candidate-level ranker
whose group-level score joins the linear evidence in the final meta-ranker.

## Repository map

- `REQUIREMENTS.md`: project constraints and acceptance checklist.
- `guide/GUIDE.md`: authoritative coursework guide.
- `code/README.md`: authoritative installation and runner instructions.
- `code/submission.py`: official prediction entry point.
- `code/verifier/`: answer grouping, features, checkpoint loading, and ranking.
- `code/train.py`: train-only fitting and checkpoint export.
- `code/train_stacked.py`: trains the selected cross-fold stacked checkpoint.
- `code/evaluate_validation.py`: validation diagnostics and slice metrics.
- `code/run_experiments.py`: fixed train cross-validation and ablations.
- `code/run_stacking_experiments.py`: compact-stack development experiment.
- `code/run_candidate_stack_experiment.py`: selected candidate-level extension.
- `code/run_nested_stacking_cv.py`: strict nested train-only confirmation.
- `code/benchmark.py`: inference-asset and runtime checks.
- `experiments/`: compact, reproducible experiment records.
- `results/final/`: outputs and resource records matching the current checkpoint.
- `results/historical/`: earlier checkpoint-bound results kept for provenance.
- `report/build_reports.py`: one reproducible source for all four current PDFs.
- `output/pdf/DASE7506_MP2_Full_Report_{Chinese,English}.pdf`: complete
  continuously paginated technical reports (11 pages in both languages).
- `output/pdf/DASE7506_MP2_10_Page_Report_{Chinese,English}.pdf`: exact
  10-page bilingual submission reports with no section-level forced page breaks.
- `tests/`: contract, feature, leakage, and checkpoint tests.

## Environment

From `code/`:

```bash
source .venv/bin/activate
python --version
python check_data.py
```

The local project environment uses Python 3.12 and the pinned dependencies in
`code/requirements.txt`.

## Development protocol

- Fit weights and preprocessing statistics on `train` only.
- Use question-level train folds for stability checks.
- Use `validation` for method selection and ablation reporting.
- Do not inspect test errors or tune after the frozen test evaluation.
- Never use question IDs as model features or answer lookups.
- Keep the official data, parser, scorer, and runners unchanged.
- Keep total uncompressed inference assets at or below 16 MiB.
- Keep complete four-thread CPU prediction plus scoring at or below 30 seconds
  on the reference machine.

## Final reproduction flow

From `code/`, after the method and checkpoint are frozen:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python predict.py --method submission.py --checkpoint checkpoint/final \
  --split validation --output ../results/final/validation-predictions.json

python score_predictions.py --split validation \
  --predictions ../results/final/validation-predictions.json \
  --output ../results/final/validation-score.json
```

The linear-candidate checkpoint's public-Test score (382/467) was recorded in its original one-time evaluation on 2026-10-15. It was not rerun during the 2026-10-28 restoration. However, the restoration was requested after comparing this score with the later MLP checkpoint's score (377/467). Therefore Test influenced package-level model selection, contrary to the coursework rule; the restored package cannot be described as test-blind. Use an unseen audit bank for an independent comparison.
## Frozen result

- Strict nested train CV: 82.4503% on ambiguous questions (1743/2114), up from
  82.0719% for the previous four-group stack.
- Validation: 745/798, 93.3584%; ambiguous validation: 302/355, 85.0704%.
- Public Test: 382/467, 81.7987%, from the checkpoint's existing one-time
  scoring record. Test was not rerun during restoration; however, its comparison
  with the MLP score influenced package-level selection, which violates the
  coursework Test-isolation rule.
- Checkpoint arrays: 259,032 bytes uncompressed. The recorded Test
  prediction-plus-scoring time was 1.292 seconds.
- The MLP candidate extension remains an alternative experiment. Its score was
  compared with the linear checkpoint Test score before restoring the latter,
  so this package does not meet test-blind selection requirements.

The previous four-model stack's matching checkpoint and Test artifacts are
retained under `results/historical/previous-stack/`. Aggregate score and
resource records for the selected checkpoint are in `results/final/`; its raw
Test prediction vector is not included.

## Completion status

See `REQUIREMENTS.md` for the full definition of done. Temporary predictions,
debug logs, abandoned checkpoints, caches, and report build artifacts are not
final deliverables and must be removed before submission. The restored package
currently has a documented Test-isolation deviation.
