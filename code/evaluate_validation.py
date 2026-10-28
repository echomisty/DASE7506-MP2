"""Validation-only diagnostics, slices, and controlled baseline comparisons."""

import argparse
import json
from pathlib import Path

from common import DATA, PROTOCOL, load_inputs, read_jsonl, score_choices
from verifier.answer_groups import build_answer_groups, plurality_group_index
from verifier.checkpoint import load_checkpoint
from verifier.development import (
    model_group_predictions,
    paired_bootstrap_difference,
    plurality_predictions,
    prepare_labeled_questions,
    summarize_predictions,
)
from verifier.pipeline import predict_choices
from verifier.stacked import STACK_FEATURE_SETS, StackedRanker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    args = parser.parse_args()

    bank, arrays = load_inputs("validation", args.data_dir)
    references = read_jsonl(args.data_dir / "gsm8k_validation.jsonl")
    model, metadata = load_checkpoint(args.checkpoint)
    analysis_feature_set = "full" if isinstance(model, StackedRanker) else model.feature_set
    examples = prepare_labeled_questions(
        bank, arrays["embeddings"], references, analysis_feature_set
    )
    if isinstance(model, StackedRanker):
        group_predictions = []
        margins = []
        for index, (row, example) in enumerate(zip(bank, examples)):
            if not example.ambiguous:
                group_predictions.append(0)
                margins.append(float("inf"))
                continue
            groups, parsed_answers = build_answer_groups(row)
            scores = model.score_groups(
                question=row["question"],
                candidate_texts=[candidate["text"] for candidate in row["candidates"]],
                embeddings=arrays["embeddings"][index],
                groups=groups,
                parsed_answers=parsed_answers,
            )
            predicted = max(
                range(len(groups)),
                key=lambda item: (float(scores[item]), -groups[item].first_index),
            )
            ordered = sorted(float(value) for value in scores)
            group_predictions.append(predicted)
            margins.append(ordered[-1] - ordered[-2])
    else:
        group_predictions, margins = model_group_predictions(examples, model)
    plurality_group_predictions = plurality_predictions(examples)

    ids = [row["id"] for row in bank]
    model_choices = predict_choices(bank, arrays, args.checkpoint)
    plurality_choices = []
    for row in bank:
        groups, _ = build_answer_groups(row)
        group_index = plurality_group_index(groups)
        plurality_choices.append(groups[group_index].first_index if groups else 0)
    longest_choices = [
        max(range(4), key=lambda index: len(row["candidates"][index]["text"]))
        for row in bank
    ]
    fixed_zero = [0] * len(bank)

    output = {
        "protocol": PROTOCOL,
        "split": "validation",
        "checkpoint": str(args.checkpoint),
        "feature_set": (
            list(STACK_FEATURE_SETS)
            + (["candidate"] if model.candidate_models else [])
            if isinstance(model, StackedRanker)
            else model.feature_set
        ),
        "checkpoint_training": metadata.get("training", {}),
        "official_scores": {
            "fixed_zero": score_choices(
                "validation", ids, fixed_zero, args.data_dir
            ),
            "longest_trace": score_choices(
                "validation", ids, longest_choices, args.data_dir
            ),
            "plurality": score_choices(
                "validation", ids, plurality_choices, args.data_dir
            ),
            "model": score_choices(
                "validation", ids, model_choices, args.data_dir
            ),
        },
        "group_metrics": {
            "plurality": summarize_predictions(examples, plurality_group_predictions),
            "model": summarize_predictions(
                examples, group_predictions, margins=margins
            ),
        },
        "paired_bootstrap_model_minus_plurality": {
            "full": paired_bootstrap_difference(
                examples,
                group_predictions,
                plurality_group_predictions,
                ambiguous_only=False,
            ),
            "ambiguous": paired_bootstrap_difference(
                examples,
                group_predictions,
                plurality_group_predictions,
                ambiguous_only=True,
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["official_scores"], indent=2))


if __name__ == "__main__":
    main()
