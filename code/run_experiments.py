"""Fixed question-level cross-validation for feature and regularization choices."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from common import DATA, PROTOCOL, load_inputs, read_jsonl
from verifier.development import (
    model_group_predictions,
    plurality_predictions,
    prepare_labeled_questions,
    stratified_fold_indices,
    summarize_predictions,
)
from verifier.features import FEATURE_SETS
from verifier.ranker import (
    fit_hardest_negative_ranker,
    fit_listwise_ranker,
    fit_pairwise_ranker,
)


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _std(values: list[float]) -> float:
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1)) if len(values) > 1 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-sets", nargs="+", choices=FEATURE_SETS, default=["full"])
    parser.add_argument("--l2", nargs="+", type=float, default=[0.01])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=7506)
    parser.add_argument(
        "--objective",
        choices=("pairwise", "listwise", "hardest"),
        default="pairwise",
    )
    parser.add_argument("--data-dir", type=Path, default=DATA)
    args = parser.parse_args()

    started = time.perf_counter()
    bank, arrays = load_inputs("train", args.data_dir)
    references = read_jsonl(args.data_dir / "gsm8k_train.jsonl")
    results = []
    plurality_summary = None

    for feature_set in args.feature_sets:
        examples = prepare_labeled_questions(
            bank, arrays["embeddings"], references, feature_set
        )
        folds = stratified_fold_indices(examples, args.folds, args.seed)
        if plurality_summary is None:
            plurality_summary = summarize_predictions(examples, plurality_predictions(examples))
        for l2 in args.l2:
            fold_results = []
            for fold_number, validation_indices in enumerate(folds):
                validation_set = set(validation_indices)
                training_examples = [
                    example
                    for index, example in enumerate(examples)
                    if index not in validation_set and example.ambiguous
                ]
                validation_examples = [examples[index] for index in validation_indices]
                fit_ranker = {
                    "pairwise": fit_pairwise_ranker,
                    "listwise": fit_listwise_ranker,
                    "hardest": fit_hardest_negative_ranker,
                }[args.objective]
                model, diagnostics = fit_ranker(
                    feature_set=feature_set,
                    feature_matrices=[example.features for example in training_examples],
                    correct_group_indices=[
                        example.correct_group_index for example in training_examples
                    ],
                    l2=l2,
                    epochs=args.epochs,
                    learning_rate=args.learning_rate,
                    seed=args.seed + fold_number,
                )
                predictions, margins = model_group_predictions(validation_examples, model)
                fold_results.append(
                    {
                        "fold": fold_number,
                        "metrics": summarize_predictions(
                            validation_examples, predictions, margins=margins
                        ),
                        "training": diagnostics,
                    }
                )
            ambiguous_accuracies = [
                fold["metrics"]["ambiguous"]["accuracy"] for fold in fold_results
            ]
            full_accuracies = [fold["metrics"]["full"]["accuracy"] for fold in fold_results]
            results.append(
                {
                    "feature_set": feature_set,
                    "l2": l2,
                    "folds": fold_results,
                    "mean_full_accuracy": _mean(full_accuracies),
                    "std_full_accuracy": _std(full_accuracies),
                    "mean_ambiguous_accuracy": _mean(ambiguous_accuracies),
                    "std_ambiguous_accuracy": _std(ambiguous_accuracies),
                }
            )

    ranked = sorted(
        results,
        key=lambda item: (
            item["mean_ambiguous_accuracy"],
            item["mean_full_accuracy"],
            -item["l2"],
        ),
        reverse=True,
    )
    output = {
        "protocol": PROTOCOL,
        "source_split": "train",
        "folds": args.folds,
        "seed": args.seed,
        "objective": args.objective,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "plurality_baseline": plurality_summary,
        "experiments": results,
        "selected": {
            "feature_set": ranked[0]["feature_set"],
            "l2": ranked[0]["l2"],
            "mean_full_accuracy": ranked[0]["mean_full_accuracy"],
            "mean_ambiguous_accuracy": ranked[0]["mean_ambiguous_accuracy"],
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["selected"], indent=2))


if __name__ == "__main__":
    main()
