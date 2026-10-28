# Experiments

This directory contains compact experiment records, not raw temporary outputs.

- `experiment_log.csv`: one row per formal experiment.
- `cross_validation.json`: fixed-fold results and chosen hyperparameters.
- `ablation_summary.csv`: final comparable validation ablations.
- `improvement_attempts.md`: compact ledger of accepted and rejected ideas.
- `improvement_candidate_stack.json`: development grid for the current stack.
- `improvement_candidate_stack_nested_cv.json`: strict nested train-only
  confirmation for the current stack.
- `final_config.json`: current frozen configuration and checkpoint identity.

Raw fold predictions, debug logs, and abandoned checkpoints belong under
ignored temporary directories and must be deleted after their aggregate results
are recorded.
