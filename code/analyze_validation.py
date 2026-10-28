"""Post-freeze validation error, confidence, and model-weight analysis."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from common import DATA, load_inputs, parse_answer, read_jsonl
from verifier.answer_groups import build_answer_groups, plurality_group_index
from verifier.checkpoint import load_checkpoint
from verifier.features import feature_names, group_feature_matrix
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
    if isinstance(model, StackedRanker):
        stack_names = list(STACK_FEATURE_SETS)
        if model.candidate_models:
            stack_names.append("candidate")
        stack_names.append("group_support")
        names = tuple(f"stack:{name}" for name in stack_names)
        interpretation_weights = model.stack_model.weights
    else:
        names = feature_names(model.feature_set)
        interpretation_weights = model.weights

    records = []
    group_count = Counter()
    unparseable_count = Counter()
    for index, (row, reference) in enumerate(zip(bank, references)):
        groups, parsed = build_answer_groups(row)
        target = parse_answer(reference["answer"])
        correct_group = next(i for i, group in enumerate(groups) if group.answer == target)
        if len(groups) == 1:
            predicted_group = 0
            scores = np.asarray([0.0])
            margin = None
        else:
            if isinstance(model, StackedRanker):
                scores = model.score_groups(
                    question=row["question"],
                    candidate_texts=[candidate["text"] for candidate in row["candidates"]],
                    embeddings=arrays["embeddings"][index],
                    groups=groups,
                    parsed_answers=parsed,
                )
            else:
                features = group_feature_matrix(
                    row["question"],
                    [candidate["text"] for candidate in row["candidates"]],
                    arrays["embeddings"][index],
                    groups,
                    parsed,
                    model.feature_set,
                )
                scores = model.score(features)
            predicted_group = max(
                range(len(groups)),
                key=lambda item: (float(scores[item]), -groups[item].first_index),
            )
            ordered = np.sort(scores)
            margin = float(ordered[-1] - ordered[-2])
        plurality = plurality_group_index(groups)
        group_count[len(groups)] += 1
        unparseable_count[sum(answer is None for answer in parsed)] += 1
        records.append(
            {
                "id": row["id"],
                "question": row["question"],
                "parsed_candidate_answers": parsed,
                "correct_answer": target,
                "predicted_answer": groups[predicted_group].answer,
                "plurality_answer": groups[plurality].answer,
                "model_correct": predicted_group == correct_group,
                "plurality_correct": plurality == correct_group,
                "answer_group_count": len(groups),
                "correct_support": groups[correct_group].support,
                "predicted_support": groups[predicted_group].support,
                "margin": margin,
            }
        )

    ambiguous = [record for record in records if record["answer_group_count"] >= 2]
    finite = [record for record in ambiguous if record["margin"] is not None]
    bins = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, float("inf"))]
    confidence = []
    for lower, upper in bins:
        selected = [record for record in finite if lower <= record["margin"] < upper]
        correct = sum(record["model_correct"] for record in selected)
        confidence.append(
            {
                "lower_inclusive": lower,
                "upper_exclusive": None if np.isinf(upper) else upper,
                "correct": correct,
                "n": len(selected),
                "accuracy": 100.0 * correct / len(selected) if selected else None,
            }
        )

    interpretable = [
        (name, float(weight))
        for name, weight in zip(names, interpretation_weights)
        if not name.startswith("embedding_delta:")
    ]
    block_norms = {}
    for prefix in ("consensus:", "trace_mean:", "trace_max:", "embedding_delta:", "embedding:"):
        values = [
            float(weight)
            for name, weight in zip(names, interpretation_weights)
            if name.startswith(prefix)
        ]
        if values:
            block_norms[prefix.rstrip(":")] = float(np.linalg.norm(values))

    high_confidence_errors = sorted(
        [record for record in finite if not record["model_correct"]],
        key=lambda record: record["margin"],
        reverse=True,
    )[:10]
    plurality_corrections = sorted(
        [
            record
            for record in finite
            if record["model_correct"] and not record["plurality_correct"]
        ],
        key=lambda record: record["margin"],
        reverse=True,
    )[:10]
    output = {
        "split": "validation",
        "post_freeze_analysis": True,
        "checkpoint_training": metadata.get("training", {}),
        "counts": {
            "questions": len(records),
            "ambiguous_questions": len(ambiguous),
            "model_errors": sum(not record["model_correct"] for record in records),
            "model_only_correct": sum(
                record["model_correct"] and not record["plurality_correct"]
                for record in records
            ),
            "plurality_only_correct": sum(
                not record["model_correct"] and record["plurality_correct"]
                for record in records
            ),
            "answer_group_count": dict(sorted(group_count.items())),
            "unparseable_candidate_count": dict(sorted(unparseable_count.items())),
        },
        "confidence_bins": confidence,
        "model_interpretation": {
            "feature_block_l2_norms": block_norms,
            "largest_positive_interpretable_weights": [
                {"feature": name, "weight": weight}
                for name, weight in sorted(interpretable, key=lambda item: item[1], reverse=True)[:12]
            ],
            "largest_negative_interpretable_weights": [
                {"feature": name, "weight": weight}
                for name, weight in sorted(interpretable, key=lambda item: item[1])[:12]
            ],
        },
        "high_confidence_errors": high_confidence_errors,
        "high_confidence_plurality_corrections": plurality_corrections,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["counts"], indent=2))


if __name__ == "__main__":
    main()
