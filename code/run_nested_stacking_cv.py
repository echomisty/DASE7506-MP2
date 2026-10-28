"""Strict nested train-only CV for group/candidate stacked rankers."""

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
from verifier.mlp import fit_multi_positive_mlp_ranker


FEATURE_SETS = ("text", "embedding", "full_no_context", "full")
BASE_L2 = {"text": 0.001, "embedding": 0.1, "full_no_context": 0.1, "full": 0.1}


def _top_group(example, scores: np.ndarray) -> int:
    return max(
        range(len(example.groups)),
        key=lambda index: (float(scores[index]), -example.groups[index].first_index),
    )


def _stack_matrix(example, score_blocks: dict[str, np.ndarray]) -> np.ndarray:
    columns = []
    names = (*FEATURE_SETS, "candidate") if "candidate" in score_blocks else FEATURE_SETS
    for feature_set in names:
        scores = np.asarray(score_blocks[feature_set], dtype=np.float32)
        columns.append(scores - scores.mean())
    support = np.asarray([group.support / 4.0 for group in example.groups], dtype=np.float32)
    columns.append(support - support.mean())
    return np.stack(columns, axis=1).astype(np.float32, copy=False)


def _candidate_group_scores(example, candidate_scores: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            max(float(candidate_scores[index]) for index in group.indices)
            for group in example.groups
        ],
        dtype=np.float32,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--stack-l2", type=float, default=0.01)
    parser.add_argument("--include-candidate", action="store_true")
    parser.add_argument("--include-candidate-mlp", action="store_true")
    parser.add_argument("--candidate-l2", type=float, default=0.1)
    parser.add_argument("--candidate-mlp-hidden", type=int, default=8)
    parser.add_argument("--candidate-mlp-epochs", type=int, default=100)
    parser.add_argument("--candidate-mlp-learning-rate", type=float, default=0.03)
    parser.add_argument("--candidate-mlp-l2", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=7506)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    args = parser.parse_args()
    if args.include_candidate_mlp and not args.include_candidate:
        parser.error("--include-candidate-mlp requires --include-candidate")

    started = time.perf_counter()
    bank, arrays = load_inputs("train", args.data_dir)
    references = read_jsonl(args.data_dir / "gsm8k_train.jsonl")
    examples_by_set = {
        feature_set: prepare_labeled_questions(
            bank, arrays["embeddings"], references, feature_set
        )
        for feature_set in FEATURE_SETS
    }
    examples = examples_by_set["full"]
    if args.include_candidate:
        candidate_matrices, candidate_positives = prepare_labeled_candidates(
            bank, arrays["embeddings"], references
        )
    outer_folds = stratified_fold_indices(examples, args.outer_folds, args.seed)
    total_correct = 0
    total_baseline_correct = 0
    total_ambiguous = 0
    fold_results = []

    for outer_fold, outer_held_indices in enumerate(outer_folds):
        outer_held_set = set(outer_held_indices)
        outer_train_indices = [
            index for index in range(len(examples)) if index not in outer_held_set
        ]
        outer_train_examples = [examples[index] for index in outer_train_indices]
        inner_local_folds = stratified_fold_indices(
            outer_train_examples,
            args.inner_folds,
            args.seed + 101 * (outer_fold + 1),
        )
        meta_train_scores = {
            index: {} for index in outer_train_indices if examples[index].ambiguous
        }
        score_names = (
            (*FEATURE_SETS, "candidate", "candidate_mlp")
            if args.include_candidate_mlp
            else (*FEATURE_SETS, "candidate")
            if args.include_candidate
            else FEATURE_SETS
        )
        held_score_sums = {
            index: {
                feature_set: np.zeros(len(examples[index].groups), dtype=np.float64)
                for feature_set in score_names
            }
            for index in outer_held_indices
            if examples[index].ambiguous
        }

        for feature_number, feature_set in enumerate(FEATURE_SETS):
            feature_examples = examples_by_set[feature_set]
            for inner_fold, inner_held_local in enumerate(inner_local_folds):
                inner_held_original = {
                    outer_train_indices[local_index] for local_index in inner_held_local
                }
                base_training = [
                    feature_examples[index]
                    for index in outer_train_indices
                    if index not in inner_held_original and feature_examples[index].ambiguous
                ]
                model, _ = fit_pairwise_ranker(
                    feature_set,
                    [example.features for example in base_training],
                    [example.correct_group_index for example in base_training],
                    l2=BASE_L2[feature_set],
                    epochs=args.epochs,
                    learning_rate=args.learning_rate,
                    seed=(
                        args.seed
                        + 10000 * outer_fold
                        + 100 * feature_number
                        + inner_fold
                    ),
                )
                for index in inner_held_original:
                    if feature_examples[index].ambiguous:
                        meta_train_scores[index][feature_set] = model.score(
                            feature_examples[index].features
                        )
                for index in outer_held_indices:
                    if feature_examples[index].ambiguous:
                        held_score_sums[index][feature_set] += model.score(
                            feature_examples[index].features
                        )

        if args.include_candidate:
            for inner_fold, inner_held_local in enumerate(inner_local_folds):
                inner_held_original = {
                    outer_train_indices[local_index]
                    for local_index in inner_held_local
                }
                candidate_training_indices = [
                    index
                    for index in outer_train_indices
                    if index not in inner_held_original
                ]
                model, _ = fit_multi_positive_pairwise_ranker(
                    "candidate",
                    [candidate_matrices[index] for index in candidate_training_indices],
                    [candidate_positives[index] for index in candidate_training_indices],
                    l2=args.candidate_l2,
                    epochs=args.epochs,
                    learning_rate=args.learning_rate,
                    seed=args.seed + 90000 + 100 * outer_fold + inner_fold,
                )
                for index in inner_held_original:
                    if examples[index].ambiguous:
                        meta_train_scores[index]["candidate"] = _candidate_group_scores(
                            examples[index], model.score(candidate_matrices[index])
                        )
                for index in outer_held_indices:
                    if examples[index].ambiguous:
                        held_score_sums[index]["candidate"] += _candidate_group_scores(
                            examples[index], model.score(candidate_matrices[index])
                        )

        if args.include_candidate_mlp:
            for inner_fold, inner_held_local in enumerate(inner_local_folds):
                inner_held_original = {
                    outer_train_indices[local_index]
                    for local_index in inner_held_local
                }
                mlp_training_indices = [
                    index for index in outer_train_indices
                    if index not in inner_held_original
                ]
                model, _ = fit_multi_positive_mlp_ranker(
                    [candidate_matrices[index] for index in mlp_training_indices],
                    [candidate_positives[index] for index in mlp_training_indices],
                    hidden_dim=args.candidate_mlp_hidden,
                    epochs=args.candidate_mlp_epochs,
                    learning_rate=args.candidate_mlp_learning_rate,
                    l2=args.candidate_mlp_l2,
                    seed=args.seed + 120000 + 100 * outer_fold + inner_fold,
                )
                for index in inner_held_original:
                    if examples[index].ambiguous:
                        meta_train_scores[index]["candidate_mlp"] = _candidate_group_scores(
                            examples[index], model.score(candidate_matrices[index])
                        )
                for index in outer_held_indices:
                    if examples[index].ambiguous:
                        held_score_sums[index]["candidate_mlp"] += _candidate_group_scores(
                            examples[index], model.score(candidate_matrices[index])
                        )

        meta_indices = sorted(meta_train_scores)
        meta_matrices = [
            _stack_matrix(examples[index], meta_train_scores[index])
            for index in meta_indices
        ]
        meta_model, diagnostics = fit_pairwise_ranker(
            "stack",
            meta_matrices,
            [examples[index].correct_group_index for index in meta_indices],
            l2=args.stack_l2,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            seed=args.seed + 50000 + outer_fold,
        )
        fold_correct = 0
        fold_baseline_correct = 0
        fold_ambiguous = 0
        for index in outer_held_indices:
            example = examples[index]
            if not example.ambiguous:
                continue
            averaged = {
                feature_set: held_score_sums[index][feature_set] / args.inner_folds
                for feature_set in score_names
            }
            baseline_prediction = _top_group(example, averaged["full"])
            prediction = _top_group(
                example,
                meta_model.score(_stack_matrix(example, averaged)),
            )
            fold_correct += int(prediction == example.correct_group_index)
            fold_baseline_correct += int(
                baseline_prediction == example.correct_group_index
            )
            fold_ambiguous += 1
        total_correct += fold_correct
        total_baseline_correct += fold_baseline_correct
        total_ambiguous += fold_ambiguous
        fold_results.append(
            {
                "fold": outer_fold,
                "correct": fold_correct,
                "n": fold_ambiguous,
                "ambiguous_accuracy": 100.0 * fold_correct / fold_ambiguous,
                "baseline_correct": fold_baseline_correct,
                "baseline_ambiguous_accuracy": (
                    100.0 * fold_baseline_correct / fold_ambiguous
                ),
                "meta_best_epoch": diagnostics["best_epoch"],
            }
        )

    output = {
        "protocol": PROTOCOL,
        "source_split": "train",
        "method": (
            "strict_nested_four_group_plus_linear_candidate_plus_mlp_candidate_stacking"
            if args.include_candidate_mlp
            else "strict_nested_four_group_plus_candidate_stacking"
            if args.include_candidate
            else "strict_nested_four_group_stacking"
        ),
        "feature_sets": list(FEATURE_SETS) + (
            ["candidate", "candidate_mlp"] if args.include_candidate_mlp
            else ["candidate"] if args.include_candidate else []
        ),
        "base_l2": BASE_L2,
        "candidate_l2": args.candidate_l2 if args.include_candidate else None,
        "candidate_mlp": (
            {
                "hidden_dim": args.candidate_mlp_hidden,
                "epochs": args.candidate_mlp_epochs,
                "learning_rate": args.candidate_mlp_learning_rate,
                "l2": args.candidate_mlp_l2,
            }
            if args.include_candidate_mlp else None
        ),
        "stack_l2": args.stack_l2,
        "outer_folds": args.outer_folds,
        "inner_folds": args.inner_folds,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "ambiguous_questions": total_ambiguous,
        "correct": total_correct,
        "nested_oof_ambiguous_accuracy": 100.0 * total_correct / total_ambiguous,
        "baseline_correct": total_baseline_correct,
        "baseline_nested_oof_ambiguous_accuracy": (
            100.0 * total_baseline_correct / total_ambiguous
        ),
        "folds": fold_results,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
