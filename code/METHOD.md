# Method and Reproduction

## Frozen method

The method parses candidate final answers with `common.parse_answer`, groups
equivalent numeric answers, and directly resolves questions with one valid
answer group. Ambiguous questions are ranked by a compact train-only stack.
Four answer-group ranker types use:

- answer-group support and plurality context;
- trace structure and question-candidate consistency;
- group-relative supplied embeddings;
- answer-group embedding cohesion.

Each group-level type contains five cross-fold models. A fifth candidate-level
linear ranker, also with five cross-fold models, scores individual traces using
candidate features and multiple-positive pairwise training; its answer-group
score is the maximum candidate score in that group. The selected version uses the five group scores, the linear candidate score,
and answer-group support in a six-value linear meta-ranker. Base L2 is `0.001`
for text and `0.1` for the other group rankers and linear candidate ranker;
meta-ranker L2 is `0.001`. The later MLP candidate extension remains a recorded
experiment and is not part of this selected checkpoint. See
`../experiments/final_config.json` for the freeze record.

## Environment

From `code/`:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python check_data.py
```

## Tests

From the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=code \
code/.venv/bin/python -m unittest discover -s tests -v
```

## Train

From `code/`:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python train_stacked.py --output checkpoint/final --folds 5 \
  --stack-l2 0.001 --include-candidate --candidate-l2 0.1 \
  --epochs 220 --learning-rate 0.03 --seed 7506
```

Only `data/gsm8k_train.jsonl` is opened by training. The exported checkpoint is
small NumPy arrays plus JSON metadata. `train.py` remains available to
reproduce the earlier single-ranker baselines.

## Train-only model selection

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python run_experiments.py \
  --output ../experiments/cross_validation.json \
  --feature-sets text embedding full_no_context full \
  --l2 0.001 0.01 0.1 --folds 5 --epochs 220 \
  --learning-rate 0.03 --seed 7506

OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python run_experiments.py \
  --output ../experiments/cv_regularization_extension.json \
  --feature-sets full --l2 0.03 0.3 1.0 3.0 \
  --folds 5 --epochs 220 --learning-rate 0.03 --seed 7506

python run_candidate_stack_experiment.py \
  --output ../experiments/improvement_candidate_stack.json --folds 5 \
  --stack-l2 0.001 0.003 0.01 0.03 0.1 0.3 --epochs 220 \
  --learning-rate 0.03 --seed 7506

python run_nested_stacking_cv.py \
  --output ../experiments/improvement_candidate_stack_nested_cv.json \
  --outer-folds 5 --inner-folds 4 --stack-l2 0.003 \
  --include-candidate --include-candidate-mlp --candidate-l2 0.1 \
  --candidate-mlp-hidden 8 --candidate-mlp-epochs 100 \
  --candidate-mlp-learning-rate 0.03 --candidate-mlp-l2 0.001 \
  --epochs 220 --learning-rate 0.03 --seed 7506
```

## Validation

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python predict.py --method submission.py --checkpoint checkpoint/final \
  --split validation \
  --output ../results/final/validation-predictions.json

python score_predictions.py --split validation \
  --predictions ../results/final/validation-predictions.json \
  --output ../results/final/validation-score.json
```

Validation diagnostics and fresh-process resource measurement:

```bash
python evaluate_validation.py --checkpoint checkpoint/final \
  --output ../experiments/validation_diagnostics.json

python benchmark.py --checkpoint checkpoint/final --split validation \
  --predictions ../results/final/validation-predictions.json \
  --score ../results/final/validation-score.json \
  --report ../results/final/validation-resource-report.json --repeats 3
```

## Test evaluation policy

The selected linear-candidate checkpoint has an existing Test record of
382/467 (81.7987%), from one scoring pass on 2026-10-15. Test was not rerun
during the 2026-10-28 restoration, and the raw Test prediction vector is not
retained. The later MLP alternative had its own one-time Test score of 377/467.
The user requested restoration of the linear checkpoint after seeing that
comparison. Thus, although each checkpoint's individual score was recorded
once, Test influenced this package-level model choice. This violates the
coursework rule that Test must not inform selection. The historical 81.80% score
should not be presented as an unbiased score for a test-blind selection process.
The matching aggregate score and resource record are in `../results/final/`.

The method accepts an unseen audit bank through the supplied `predict.py`
`--split audit --data-dir ...` interface and never opens references at
inference. The selected checkpoint reaches 93.3584% on validation, 81.7987% on
the previously scored public Test, and 82.4503% on strict nested ambiguous train CV.
Because the package restoration followed a public-Test comparison, the current
package is not test-blind. A future unseen audit bank is needed for an
independent comparison.
