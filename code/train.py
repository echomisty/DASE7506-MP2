"""Fit and export the train-only answer-group ranker."""

import argparse
import json
from pathlib import Path
import time

from common import DATA, PROTOCOL, load_inputs, read_jsonl
from verifier.checkpoint import save_checkpoint
from verifier.development import prepare_labeled_questions
from verifier.features import FEATURE_SETS
from verifier.ranker import (
    fit_hardest_negative_ranker,
    fit_listwise_ranker,
    fit_pairwise_ranker,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-set", choices=FEATURE_SETS, default="full")
    parser.add_argument("--l2", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=240)
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
    examples = prepare_labeled_questions(
        bank, arrays["embeddings"], references, args.feature_set
    )
    ambiguous = [example for example in examples if example.ambiguous]
    fit_ranker = {
        "pairwise": fit_pairwise_ranker,
        "listwise": fit_listwise_ranker,
        "hardest": fit_hardest_negative_ranker,
    }[args.objective]
    model, diagnostics = fit_ranker(
        feature_set=args.feature_set,
        feature_matrices=[example.features for example in ambiguous],
        correct_group_indices=[example.correct_group_index for example in ambiguous],
        l2=args.l2,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    training_metadata = {
        "protocol": PROTOCOL,
        "source_split": "train",
        "total_questions": len(examples),
        "ambiguous_questions": len(ambiguous),
        "feature_set": args.feature_set,
        "l2": args.l2,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "objective": args.objective,
        "diagnostics": diagnostics,
        "training_seconds": time.perf_counter() - started,
    }
    checkpoint = save_checkpoint(args.output, model, training_metadata)
    print(json.dumps({"training": training_metadata, "checkpoint": checkpoint}, indent=2))


if __name__ == "__main__":
    main()
