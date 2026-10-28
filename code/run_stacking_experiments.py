"""OOF development experiment for complementary answer-group rankers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from common import DATA, PROTOCOL, load_inputs, read_jsonl
from verifier.answer_groups import plurality_group_index
from verifier.development import prepare_labeled_questions, stratified_fold_indices
from verifier.ranker import fit_pairwise_ranker


BASE_FEATURE_SETS = (
    "text",
    "embedding",
    "full_no_context",
    "full",
)
BASE_L2 = {
    "text": 0.001,
    "embedding": 0.1,
    "full_no_context": 0.1,
    "full": 0.1,
}


def _top_group(scores: np.ndarray, groups) -> int:
    return max(
        range(len(groups)),
        key=lambda index: (float(scores[index]), -groups[index].first_index),
    )


def _accuracy(examples, predictions: list[int]) -> float:
    return 100.0 * sum(
        prediction == example.correct_group_index
        for prediction, example in zip(predictions, examples)
    ) / len(examples)


def _margin(scores: np.ndarray) -> float:
    ordered = np.sort(np.asarray(scores, dtype=np.float64))
    return float(ordered[-1] - ordered[-2])


def _has_unique_plurality(example) -> bool:
    maximum = max(group.support for group in example.groups)
    return sum(group.support == maximum for group in example.groups) == 1


def _gated_predictions(examples, score_rows, threshold: float, unique_only: bool):
    predictions = []
    fallback_count = 0
    for example, scores in zip(examples, score_rows):
        prediction = _top_group(scores, example.groups)
        if _margin(scores) < threshold and (
            not unique_only or _has_unique_plurality(example)
        ):
            prediction = plurality_group_index(example.groups)
            fallback_count += 1
        predictions.append(prediction)
    return predictions, fallback_count


def _summaries(examples, predictions: list[int]) -> dict:
    full_hits = sum(
        prediction == example.correct_group_index
        for prediction, example in zip(predictions, examples)
    )
    ambiguous_pairs = [
        (prediction, example)
        for prediction, example in zip(predictions, examples)
        if example.ambiguous
    ]
    ambiguous_hits = sum(
        prediction == example.correct_group_index
        for prediction, example in ambiguous_pairs
    )
    return {
        "full": {
            "correct": full_hits,
            "n": len(examples),
            "accuracy": 100.0 * full_hits / len(examples),
        },
        "ambiguous": {
            "correct": ambiguous_hits,
            "n": len(ambiguous_pairs),
            "accuracy": 100.0 * ambiguous_hits / len(ambiguous_pairs),
        },
    }


def _stack_features(
    example,
    score_blocks: dict[str, np.ndarray],
    transform: str = "linear",
) -> np.ndarray:
    columns = []
    for feature_set in BASE_FEATURE_SETS:
        scores = np.asarray(score_blocks[feature_set], dtype=np.float32)
        columns.append(scores - scores.mean())
    support = np.asarray([group.support / 4.0 for group in example.groups], dtype=np.float32)
    columns.append(support - support.mean())
    linear = np.stack(columns, axis=1).astype(np.float32, copy=False)
    if transform == "linear":
        return linear
    expanded = [linear]
    expanded.append(linear * linear)
    expanded.append(
        np.stack(
            [
                linear[:, left] * linear[:, right]
                for left in range(linear.shape[1])
                for right in range(left + 1, linear.shape[1])
            ],
            axis=1,
        )
    )
    quadratic = np.concatenate(expanded, axis=1)
    return (quadratic - quadratic.mean(axis=0)).astype(np.float32, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--stack-l2", nargs="+", type=float, default=[0.0, 0.01, 0.1, 1.0])
    parser.add_argument(
        "--stack-transform",
        choices=("linear", "quadratic"),
        default="linear",
    )
    parser.add_argument(
        "--gate-thresholds",
        nargs="+",
        type=float,
        default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0],
    )
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=7506)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    args = parser.parse_args()

    started = time.perf_counter()
    bank, arrays = load_inputs("train", args.data_dir)
    references = read_jsonl(args.data_dir / "gsm8k_train.jsonl")
    examples_by_set = {
        feature_set: prepare_labeled_questions(
            bank, arrays["embeddings"], references, feature_set
        )
        for feature_set in BASE_FEATURE_SETS
    }
    examples = examples_by_set["full"]
    validation_bank, validation_arrays = load_inputs("validation", args.data_dir)
    validation_references = read_jsonl(args.data_dir / "gsm8k_validation.jsonl")
    validation_by_set = {
        feature_set: prepare_labeled_questions(
            validation_bank,
            validation_arrays["embeddings"],
            validation_references,
            feature_set,
        )
        for feature_set in BASE_FEATURE_SETS
    }
    validation_examples = validation_by_set["full"]
    folds = stratified_fold_indices(examples, args.folds, args.seed)
    fold_by_index = {
        index: fold_number
        for fold_number, fold_indices in enumerate(folds)
        for index in fold_indices
    }
    ambiguous_indices = [index for index, example in enumerate(examples) if example.ambiguous]
    oof_scores: dict[int, dict[str, np.ndarray]] = {
        index: {} for index in ambiguous_indices
    }
    base_training = []
    validation_score_sums = {
        feature_set: [
            np.zeros(len(example.groups), dtype=np.float64)
            if example.ambiguous
            else None
            for example in validation_examples
        ]
        for feature_set in BASE_FEATURE_SETS
    }

    for feature_number, feature_set in enumerate(BASE_FEATURE_SETS):
        feature_examples = examples_by_set[feature_set]
        for fold_number, held_out_indices in enumerate(folds):
            held_out_set = set(held_out_indices)
            training_examples = [
                example
                for index, example in enumerate(feature_examples)
                if index not in held_out_set and example.ambiguous
            ]
            model, diagnostics = fit_pairwise_ranker(
                feature_set,
                [example.features for example in training_examples],
                [example.correct_group_index for example in training_examples],
                l2=BASE_L2[feature_set],
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                seed=args.seed + 100 * feature_number + fold_number,
            )
            base_training.append(
                {
                    "feature_set": feature_set,
                    "fold": fold_number,
                    "best_epoch": diagnostics["best_epoch"],
                    "best_loss": diagnostics["best_loss"],
                }
            )
            for index in held_out_indices:
                example = feature_examples[index]
                if example.ambiguous:
                    oof_scores[index][feature_set] = model.score(example.features)
            for index, example in enumerate(validation_by_set[feature_set]):
                if example.ambiguous:
                    validation_score_sums[feature_set][index] += model.score(
                        example.features
                    )

    ambiguous_examples = [examples[index] for index in ambiguous_indices]
    stack_matrices = [
        _stack_features(examples[index], oof_scores[index], args.stack_transform)
        for index in ambiguous_indices
    ]
    record_folds = [fold_by_index[index] for index in ambiguous_indices]

    individual = {}
    for feature_set in BASE_FEATURE_SETS:
        predictions = [
            _top_group(oof_scores[index][feature_set], examples[index].groups)
            for index in ambiguous_indices
        ]
        individual[feature_set] = _accuracy(ambiguous_examples, predictions)

    stack_results = []
    oof_score_rows_by_l2 = {}
    for stack_l2 in args.stack_l2:
        predictions = [None] * len(ambiguous_examples)
        score_rows = [None] * len(ambiguous_examples)
        fold_metrics = []
        for fold_number in range(args.folds):
            training_positions = [
                position for position, fold in enumerate(record_folds) if fold != fold_number
            ]
            held_positions = [
                position for position, fold in enumerate(record_folds) if fold == fold_number
            ]
            model, diagnostics = fit_pairwise_ranker(
                "stack",
                [stack_matrices[position] for position in training_positions],
                [ambiguous_examples[position].correct_group_index for position in training_positions],
                l2=stack_l2,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                seed=args.seed + 1000 + fold_number,
            )
            fold_predictions = []
            for position in held_positions:
                scores = model.score(stack_matrices[position])
                prediction = _top_group(
                    scores,
                    ambiguous_examples[position].groups,
                )
                predictions[position] = prediction
                score_rows[position] = scores
                fold_predictions.append(prediction)
            fold_metrics.append(
                {
                    "fold": fold_number,
                    "n": len(held_positions),
                    "accuracy": _accuracy(
                        [ambiguous_examples[position] for position in held_positions],
                        fold_predictions,
                    ),
                    "weights": model.weights.tolist(),
                    "best_epoch": diagnostics["best_epoch"],
                }
            )
        if any(prediction is None for prediction in predictions):
            raise RuntimeError("A stacking prediction was not filled")
        stack_results.append(
            {
                "l2": stack_l2,
                "cross_fitted_ambiguous_accuracy": _accuracy(
                    ambiguous_examples, predictions
                ),
                "folds": fold_metrics,
            }
        )
        oof_score_rows_by_l2[stack_l2] = score_rows

    selected = max(
        stack_results,
        key=lambda item: (item["cross_fitted_ambiguous_accuracy"], -item["l2"]),
    )
    gate_results = []
    selected_oof_scores = oof_score_rows_by_l2[selected["l2"]]
    for unique_only in (False, True):
        for threshold in args.gate_thresholds:
            gated, fallback_count = _gated_predictions(
                ambiguous_examples,
                selected_oof_scores,
                threshold,
                unique_only,
            )
            gate_results.append(
                {
                    "threshold": threshold,
                    "unique_plurality_only": unique_only,
                    "fallback_count": fallback_count,
                    "cross_fitted_ambiguous_accuracy": _accuracy(
                        ambiguous_examples, gated
                    ),
                }
            )
    selected_gate = max(
        gate_results,
        key=lambda item: (
            item["cross_fitted_ambiguous_accuracy"],
            -item["fallback_count"],
            -item["threshold"],
        ),
    )
    final_stack_model, final_stack_diagnostics = fit_pairwise_ranker(
        "stack",
        stack_matrices,
        [example.correct_group_index for example in ambiguous_examples],
        l2=selected["l2"],
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed + 2000,
    )
    validation_base_scores = {}
    validation_base_metrics = {}
    for feature_set in BASE_FEATURE_SETS:
        scores = [
            value / args.folds if value is not None else None
            for value in validation_score_sums[feature_set]
        ]
        validation_base_scores[feature_set] = scores
        predictions = [
            0 if not example.ambiguous else _top_group(scores[index], example.groups)
            for index, example in enumerate(validation_examples)
        ]
        validation_base_metrics[feature_set] = _summaries(
            validation_examples, predictions
        )
    validation_stack_predictions = []
    validation_ambiguous_examples = []
    validation_stack_scores = []
    for index, example in enumerate(validation_examples):
        if not example.ambiguous:
            validation_stack_predictions.append(0)
            continue
        score_blocks = {
            feature_set: validation_base_scores[feature_set][index]
            for feature_set in BASE_FEATURE_SETS
        }
        stack_features = _stack_features(
            example, score_blocks, args.stack_transform
        )
        scores = final_stack_model.score(stack_features)
        validation_stack_predictions.append(
            _top_group(scores, example.groups)
        )
        validation_ambiguous_examples.append(example)
        validation_stack_scores.append(scores)
    validation_gated_ambiguous, validation_fallback_count = _gated_predictions(
        validation_ambiguous_examples,
        validation_stack_scores,
        selected_gate["threshold"],
        selected_gate["unique_plurality_only"],
    )
    validation_gated_predictions = []
    ambiguous_position = 0
    for example in validation_examples:
        if example.ambiguous:
            validation_gated_predictions.append(
                validation_gated_ambiguous[ambiguous_position]
            )
            ambiguous_position += 1
        else:
            validation_gated_predictions.append(0)
    output = {
        "protocol": PROTOCOL,
        "source_split": "train",
        "method": "oof_score_stacking",
        "base_feature_sets": list(BASE_FEATURE_SETS),
        "stack_features": [*BASE_FEATURE_SETS, "group_support"],
        "stack_transform": args.stack_transform,
        "folds": args.folds,
        "base_l2": BASE_L2,
        "stack_l2_candidates": args.stack_l2,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "ambiguous_questions": len(ambiguous_examples),
        "individual_oof_ambiguous_accuracy": individual,
        "stacking_results": stack_results,
        "selected": {
            "l2": selected["l2"],
            "cross_fitted_ambiguous_accuracy": selected[
                "cross_fitted_ambiguous_accuracy"
            ],
        },
        "confidence_gate": {
            "candidates": gate_results,
            "selected_on_train_oof": selected_gate,
            "validation_fallback_count": validation_fallback_count,
        },
        "validation_evaluation": {
            "base_cross_fold_ensembles": validation_base_metrics,
            "stack": _summaries(validation_examples, validation_stack_predictions),
            "confidence_gated_stack": _summaries(
                validation_examples, validation_gated_predictions
            ),
            "final_stack_training": final_stack_diagnostics,
        },
        "base_training": base_training,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(
        json.dumps(
            {
                "individual": individual,
                "selected": output["selected"],
                "validation": output["validation_evaluation"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
