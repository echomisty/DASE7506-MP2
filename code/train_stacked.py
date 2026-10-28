"""Train and export the selected cross-fold stack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from common import DATA, PROTOCOL, load_inputs, read_jsonl
from verifier.checkpoint import save_stacked_checkpoint
from verifier.development import (
    prepare_labeled_candidates,
    prepare_labeled_questions,
    stratified_fold_indices,
)
from verifier.ranker import fit_multi_positive_pairwise_ranker, fit_pairwise_ranker
from verifier.mlp import fit_multi_positive_mlp_ranker
from verifier.stacked import STACK_FEATURE_SETS, StackedRanker, stack_feature_matrix


BASE_L2 = {"text": 0.001, "embedding": 0.1, "full_no_context": 0.1, "full": 0.1}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--stack-l2", type=float, default=0.01)
    parser.add_argument("--include-candidate", action="store_true")
    parser.add_argument("--candidate-l2", type=float, default=0.1)
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
        for feature_set in STACK_FEATURE_SETS
    }
    examples = examples_by_set["full"]
    if args.include_candidate:
        candidate_matrices, candidate_positives = prepare_labeled_candidates(
            bank, arrays["embeddings"], references
        )
    folds = stratified_fold_indices(examples, args.folds, args.seed)
    ambiguous_indices = [index for index, example in enumerate(examples) if example.ambiguous]
    oof_scores = {index: {} for index in ambiguous_indices}
    base_models = {}
    base_diagnostics = []

    for feature_number, feature_set in enumerate(STACK_FEATURE_SETS):
        feature_examples = examples_by_set[feature_set]
        models = []
        for fold_number, held_indices in enumerate(folds):
            held_set = set(held_indices)
            training = [
                example
                for index, example in enumerate(feature_examples)
                if index not in held_set and example.ambiguous
            ]
            model, diagnostics = fit_pairwise_ranker(
                feature_set,
                [example.features for example in training],
                [example.correct_group_index for example in training],
                l2=BASE_L2[feature_set],
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                seed=args.seed + 100 * feature_number + fold_number,
            )
            models.append(model)
            base_diagnostics.append(
                {
                    "feature_set": feature_set,
                    "fold": fold_number,
                    "best_epoch": diagnostics["best_epoch"],
                    "best_loss": diagnostics["best_loss"],
                }
            )
            for index in held_indices:
                if feature_examples[index].ambiguous:
                    oof_scores[index][feature_set] = model.score(
                        feature_examples[index].features
                    )
        base_models[feature_set] = tuple(models)

    candidate_models = []
    if args.include_candidate:
        for fold_number, held_indices in enumerate(folds):
            held_set = set(held_indices)
            training_indices = [
                index for index in range(len(examples)) if index not in held_set
            ]
            model, diagnostics = fit_multi_positive_pairwise_ranker(
                "candidate",
                [candidate_matrices[index] for index in training_indices],
                [candidate_positives[index] for index in training_indices],
                l2=args.candidate_l2,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                seed=args.seed + 1000 + fold_number,
            )
            candidate_models.append(model)
            base_diagnostics.append(
                {
                    "feature_set": "candidate",
                    "fold": fold_number,
                    "best_epoch": diagnostics["best_epoch"],
                    "best_loss": diagnostics["best_loss"],
                }
            )
            for index in held_indices:
                if examples[index].ambiguous:
                    candidate_scores = model.score(candidate_matrices[index])
                    oof_scores[index]["candidate"] = np.asarray(
                        [
                            max(
                                float(candidate_scores[candidate_index])
                                for candidate_index in group.indices
                            )
                            for group in examples[index].groups
                        ],
                        dtype=np.float32,
                    )

    candidate_mlp_models = []
    if args.include_candidate:
        for fold_number, held_indices in enumerate(folds):
            held_set = set(held_indices)
            training_indices = [
                index for index in range(len(examples)) if index not in held_set
            ]
            model, diagnostics = fit_multi_positive_mlp_ranker(
                [candidate_matrices[index] for index in training_indices],
                [candidate_positives[index] for index in training_indices],
                hidden_dim=8,
                epochs=100,
                learning_rate=args.learning_rate,
                l2=0.001,
                seed=args.seed + 3000 + fold_number,
            )
            candidate_mlp_models.append(model)
            base_diagnostics.append(
                {
                    "feature_set": "candidate_mlp",
                    "fold": fold_number,
                    **diagnostics,
                }
            )
            for index in held_indices:
                if examples[index].ambiguous:
                    candidate_scores = model.score(candidate_matrices[index])
                    oof_scores[index]["candidate_mlp"] = np.asarray(
                        [
                            max(float(candidate_scores[candidate_index]) for candidate_index in group.indices)
                            for group in examples[index].groups
                        ],
                        dtype=np.float32,
                    )

    stack_matrices = [
        stack_feature_matrix(examples[index].groups, oof_scores[index])
        for index in ambiguous_indices
    ]
    stack_model, stack_diagnostics = fit_pairwise_ranker(
        "stack",
        stack_matrices,
        [examples[index].correct_group_index for index in ambiguous_indices],
        l2=args.stack_l2,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed + 2000,
    )
    stacked = StackedRanker(
        base_models=base_models,
        stack_model=stack_model,
        candidate_models=tuple(candidate_models),
        candidate_mlp_models=tuple(candidate_mlp_models),
    )
    training_metadata = {
        "protocol": PROTOCOL,
        "source_split": "train",
        "total_questions": len(examples),
        "ambiguous_questions": len(ambiguous_indices),
        "method": (
            "four_group_models_plus_candidate_cross_fold_stack"
            if args.include_candidate
            else "four_model_cross_fold_stack"
        ),
        "feature_sets": list(STACK_FEATURE_SETS) + (
            ["candidate", "candidate_mlp"] if args.include_candidate else []
        ),
        "base_l2": BASE_L2,
        "candidate_l2": args.candidate_l2 if args.include_candidate else None,
        "candidate_mlp": (
            {"hidden_dim": 8, "epochs": 100, "learning_rate": args.learning_rate, "l2": 0.001}
            if args.include_candidate else None
        ),
        "folds": args.folds,
        "stack_l2": args.stack_l2,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "base_diagnostics": base_diagnostics,
        "stack_diagnostics": stack_diagnostics,
        "training_seconds": time.perf_counter() - started,
    }
    report = save_stacked_checkpoint(args.output, stacked, training_metadata)
    print(json.dumps({"training": training_metadata, "checkpoint": report}, indent=2))


if __name__ == "__main__":
    main()
