"""Development-only stack of group rankers and a candidate-level ranker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from common import DATA, PROTOCOL, load_inputs, read_jsonl
from verifier.development import (
    prepare_labeled_candidates,
    prepare_labeled_questions,
    stratified_fold_indices,
)
from verifier.ranker import fit_multi_positive_pairwise_ranker, fit_pairwise_ranker


FEATURE_SETS = ("text", "embedding", "full_no_context", "full")
BASE_L2 = {"text": 0.001, "embedding": 0.1, "full_no_context": 0.1, "full": 0.1}
CANDIDATE_L2 = 0.1


def _top_group(example, scores):
    return max(
        range(len(example.groups)),
        key=lambda index: (float(scores[index]), -example.groups[index].first_index),
    )


def _candidate_group_scores(example, candidate_scores):
    return np.asarray(
        [max(float(candidate_scores[index]) for index in group.indices) for group in example.groups],
        dtype=np.float32,
    )


def _stack_matrix(example, scores_by_name):
    columns = []
    for name in (*FEATURE_SETS, "candidate"):
        scores = np.asarray(scores_by_name[name], dtype=np.float32)
        columns.append(scores - scores.mean())
    support = np.asarray([group.support / 4.0 for group in example.groups], dtype=np.float32)
    columns.append(support - support.mean())
    return np.stack(columns, axis=1).astype(np.float32, copy=False)


def _summary(examples, predictions):
    correct = [prediction == example.correct_group_index for example, prediction in zip(examples, predictions)]
    ambiguous_correct = [hit for hit, example in zip(correct, examples) if example.ambiguous]
    return {
        "full": {"correct": sum(correct), "n": len(correct), "accuracy": 100.0 * sum(correct) / len(correct)},
        "ambiguous": {"correct": sum(ambiguous_correct), "n": len(ambiguous_correct), "accuracy": 100.0 * sum(ambiguous_correct) / len(ambiguous_correct)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--stack-l2", nargs="+", type=float, required=True)
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=7506)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    args = parser.parse_args()

    started = time.perf_counter()
    train_bank, train_arrays = load_inputs("train", args.data_dir)
    train_references = read_jsonl(args.data_dir / "gsm8k_train.jsonl")
    validation_bank, validation_arrays = load_inputs("validation", args.data_dir)
    validation_references = read_jsonl(args.data_dir / "gsm8k_validation.jsonl")
    train_by_set = {
        feature_set: prepare_labeled_questions(
            train_bank, train_arrays["embeddings"], train_references, feature_set
        )
        for feature_set in FEATURE_SETS
    }
    validation_by_set = {
        feature_set: prepare_labeled_questions(
            validation_bank,
            validation_arrays["embeddings"],
            validation_references,
            feature_set,
        )
        for feature_set in FEATURE_SETS
    }
    examples = train_by_set["full"]
    validation_examples = validation_by_set["full"]
    candidate_matrices, candidate_positives = prepare_labeled_candidates(
        train_bank, train_arrays["embeddings"], train_references
    )
    validation_candidate_matrices, _ = prepare_labeled_candidates(
        validation_bank, validation_arrays["embeddings"], validation_references
    )
    folds = stratified_fold_indices(examples, args.folds, args.seed)
    fold_by_index = {
        index: fold for fold, indices in enumerate(folds) for index in indices
    }
    ambiguous_indices = [index for index, example in enumerate(examples) if example.ambiguous]
    oof_scores = {index: {} for index in ambiguous_indices}
    validation_sums = {
        name: [
            np.zeros(len(example.groups), dtype=np.float64) if example.ambiguous else None
            for example in validation_examples
        ]
        for name in (*FEATURE_SETS, "candidate")
    }

    for feature_number, feature_set in enumerate(FEATURE_SETS):
        for fold_number, held_indices in enumerate(folds):
            held_set = set(held_indices)
            training = [
                example
                for index, example in enumerate(train_by_set[feature_set])
                if index not in held_set and example.ambiguous
            ]
            model, _ = fit_pairwise_ranker(
                feature_set,
                [example.features for example in training],
                [example.correct_group_index for example in training],
                l2=BASE_L2[feature_set],
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                seed=args.seed + 100 * feature_number + fold_number,
            )
            for index in held_indices:
                if examples[index].ambiguous:
                    oof_scores[index][feature_set] = model.score(
                        train_by_set[feature_set][index].features
                    )
            for index, example in enumerate(validation_by_set[feature_set]):
                if example.ambiguous:
                    validation_sums[feature_set][index] += model.score(example.features)

    for fold_number, held_indices in enumerate(folds):
        held_set = set(held_indices)
        training_indices = [index for index in range(len(examples)) if index not in held_set]
        model, _ = fit_multi_positive_pairwise_ranker(
            "candidate",
            [candidate_matrices[index] for index in training_indices],
            [candidate_positives[index] for index in training_indices],
            l2=CANDIDATE_L2,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            seed=args.seed + 1000 + fold_number,
        )
        for index in held_indices:
            if examples[index].ambiguous:
                oof_scores[index]["candidate"] = _candidate_group_scores(
                    examples[index], model.score(candidate_matrices[index])
                )
        for index, example in enumerate(validation_examples):
            if example.ambiguous:
                validation_sums["candidate"][index] += _candidate_group_scores(
                    example, model.score(validation_candidate_matrices[index])
                )

    ambiguous_examples = [examples[index] for index in ambiguous_indices]
    matrices = [_stack_matrix(examples[index], oof_scores[index]) for index in ambiguous_indices]
    record_folds = [fold_by_index[index] for index in ambiguous_indices]
    validation_scores = {
        index: {
            name: validation_sums[name][index] / args.folds
            for name in (*FEATURE_SETS, "candidate")
        }
        for index, example in enumerate(validation_examples)
        if example.ambiguous
    }
    results = []
    for l2 in args.stack_l2:
        predictions = [None] * len(ambiguous_examples)
        for fold_number in range(args.folds):
            training_positions = [position for position, fold in enumerate(record_folds) if fold != fold_number]
            held_positions = [position for position, fold in enumerate(record_folds) if fold == fold_number]
            model, _ = fit_pairwise_ranker(
                "candidate_stack",
                [matrices[position] for position in training_positions],
                [ambiguous_examples[position].correct_group_index for position in training_positions],
                l2=l2,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                seed=args.seed + 2000 + fold_number,
            )
            for position in held_positions:
                predictions[position] = _top_group(
                    ambiguous_examples[position], model.score(matrices[position])
                )
        full_model, diagnostics = fit_pairwise_ranker(
            "candidate_stack",
            matrices,
            [example.correct_group_index for example in ambiguous_examples],
            l2=l2,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            seed=args.seed + 3000,
        )
        validation_predictions = []
        for index, example in enumerate(validation_examples):
            if not example.ambiguous:
                validation_predictions.append(0)
            else:
                validation_predictions.append(
                    _top_group(
                        example,
                        full_model.score(_stack_matrix(example, validation_scores[index])),
                    )
                )
        correct = sum(
            prediction == example.correct_group_index
            for prediction, example in zip(predictions, ambiguous_examples)
        )
        results.append(
            {
                "l2": l2,
                "cross_fitted_ambiguous_correct": correct,
                "cross_fitted_ambiguous_accuracy": 100.0 * correct / len(ambiguous_examples),
                "validation": _summary(validation_examples, validation_predictions),
                "full_stack_best_epoch": diagnostics["best_epoch"],
            }
        )
    selected = max(results, key=lambda item: (item["cross_fitted_ambiguous_accuracy"], -item["l2"]))
    output = {
        "protocol": PROTOCOL,
        "source_split": "train",
        "method": "four_group_models_plus_candidate_stack",
        "base_l2": BASE_L2,
        "candidate_l2": CANDIDATE_L2,
        "folds": args.folds,
        "seed": args.seed,
        "results": results,
        "selected_on_train_oof": selected,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
