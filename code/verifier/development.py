"""Train/validation-only preparation and metrics; never imported by inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common import parse_answer

from .answer_groups import AnswerGroup, build_answer_groups, plurality_group_index
from .features import candidate_feature_matrix, group_feature_matrix
from .ranker import LinearRanker


@dataclass(frozen=True)
class LabeledQuestion:
    source_index: int
    question_id: str
    groups: tuple[AnswerGroup, ...]
    features: np.ndarray
    correct_group_index: int
    plurality_group_index: int

    @property
    def ambiguous(self) -> bool:
        return len(self.groups) >= 2

    @property
    def correct_support(self) -> int:
        return self.groups[self.correct_group_index].support

    @property
    def has_unique_plurality(self) -> bool:
        supports = [group.support for group in self.groups]
        maximum = max(supports)
        return maximum > 1 and supports.count(maximum) == 1


def prepare_labeled_questions(
    bank: list[dict],
    embeddings: np.ndarray,
    references: list[dict],
    feature_set: str,
) -> list[LabeledQuestion]:
    """Create scorer-aligned answer-group examples from an allowed reference split."""

    embeddings = np.asarray(embeddings)
    if embeddings.shape != (len(bank), 4, 1024) or len(references) != len(bank):
        raise ValueError("Inputs and references are not aligned")
    examples: list[LabeledQuestion] = []
    for source_index, (row, reference) in enumerate(zip(bank, references)):
        if row["id"] != reference["id"] or row["question"] != reference["question"]:
            raise ValueError("Reference and candidate bank alignment mismatch")
        groups, parsed_answers = build_answer_groups(row)
        target = parse_answer(reference["answer"])
        correct = next(
            (index for index, group in enumerate(groups) if group.answer == target),
            None,
        )
        if correct is None:
            raise ValueError(f"No correct parseable candidate for {row['id']}")
        plurality = plurality_group_index(groups)
        features = group_feature_matrix(
            question=row["question"],
            candidate_texts=[candidate["text"] for candidate in row["candidates"]],
            embeddings=embeddings[source_index],
            groups=groups,
            parsed_answers=parsed_answers,
            feature_set=feature_set,
        )
        examples.append(
            LabeledQuestion(
                source_index=source_index,
                question_id=row["id"],
                groups=tuple(groups),
                features=features,
                correct_group_index=correct,
                plurality_group_index=int(plurality),
            )
        )
    return examples


def prepare_labeled_candidates(
    bank: list[dict],
    embeddings: np.ndarray,
    references: list[dict],
) -> tuple[list[np.ndarray], list[tuple[int, ...]]]:
    """Create candidate feature matrices with all scorer-correct positives."""

    embeddings = np.asarray(embeddings)
    if embeddings.shape != (len(bank), 4, 1024) or len(references) != len(bank):
        raise ValueError("Inputs and references are not aligned")
    matrices = []
    positive_indices = []
    for source_index, (row, reference) in enumerate(zip(bank, references)):
        if row["id"] != reference["id"] or row["question"] != reference["question"]:
            raise ValueError("Reference and candidate bank alignment mismatch")
        groups, parsed_answers = build_answer_groups(row)
        target = parse_answer(reference["answer"])
        positives = tuple(
            index for index, answer in enumerate(parsed_answers) if answer == target
        )
        if not positives:
            raise ValueError(f"No correct parseable candidate for {row['id']}")
        matrices.append(
            candidate_feature_matrix(
                question=row["question"],
                candidate_texts=[
                    candidate["text"] for candidate in row["candidates"]
                ],
                embeddings=embeddings[source_index],
                groups=groups,
                parsed_answers=parsed_answers,
            )
        )
        positive_indices.append(positives)
    return matrices, positive_indices


def stratified_fold_indices(
    examples: list[LabeledQuestion],
    folds: int,
    seed: int,
) -> list[list[int]]:
    """Deterministically distribute question-level strata across folds."""

    if folds < 2:
        raise ValueError("At least two folds are required")
    rng = np.random.default_rng(seed)
    buckets: dict[tuple[int, int], list[int]] = {}
    for index, example in enumerate(examples):
        key = (min(len(example.groups), 4), example.correct_support)
        buckets.setdefault(key, []).append(index)
    result = [[] for _ in range(folds)]
    offset = 0
    for key in sorted(buckets):
        indices = np.asarray(buckets[key], dtype=np.int64)
        rng.shuffle(indices)
        for position, index in enumerate(indices.tolist()):
            result[(offset + position) % folds].append(index)
        offset = (offset + len(indices)) % folds
    for fold in result:
        fold.sort()
    return result


def model_group_predictions(
    examples: list[LabeledQuestion],
    model: LinearRanker,
) -> tuple[list[int], list[float]]:
    predictions: list[int] = []
    margins: list[float] = []
    for example in examples:
        if len(example.groups) == 1:
            predictions.append(0)
            margins.append(float("inf"))
            continue
        scores = model.score(example.features)
        predicted = max(
            range(len(example.groups)),
            key=lambda index: (float(scores[index]), -example.groups[index].first_index),
        )
        sorted_scores = np.sort(scores)
        predictions.append(int(predicted))
        margins.append(float(sorted_scores[-1] - sorted_scores[-2]))
    return predictions, margins


def summarize_predictions(
    examples: list[LabeledQuestion],
    predictions: list[int],
    *,
    margins: list[float] | None = None,
) -> dict:
    if len(predictions) != len(examples):
        raise ValueError("Predictions and examples are not aligned")
    correct = np.asarray(
        [prediction == example.correct_group_index for prediction, example in zip(predictions, examples)],
        dtype=bool,
    )
    ambiguous = np.asarray([example.ambiguous for example in examples], dtype=bool)
    unique_plurality = np.asarray(
        [example.has_unique_plurality for example in examples], dtype=bool
    )
    plurality_correct = np.asarray(
        [example.plurality_group_index == example.correct_group_index for example in examples],
        dtype=bool,
    )

    def slice_result(mask: np.ndarray) -> dict:
        count = int(mask.sum())
        hits = int((correct & mask).sum())
        return {
            "correct": hits,
            "n": count,
            "accuracy": 100.0 * hits / count if count else None,
        }

    summary = {
        "full": slice_result(np.ones(len(examples), dtype=bool)),
        "ambiguous": slice_result(ambiguous),
        "unique_plurality": slice_result(ambiguous & unique_plurality),
        "no_unique_plurality": slice_result(ambiguous & ~unique_plurality),
        "correct_support": {
            str(support): slice_result(
                np.asarray([example.correct_support == support for example in examples], dtype=bool)
            )
            for support in range(1, 5)
        },
        "paired_vs_plurality": {
            "both_correct": int((correct & plurality_correct).sum()),
            "model_only_correct": int((correct & ~plurality_correct).sum()),
            "plurality_only_correct": int((~correct & plurality_correct).sum()),
            "both_wrong": int((~correct & ~plurality_correct).sum()),
        },
    }
    if margins is not None:
        finite = np.asarray([value for value in margins if np.isfinite(value)], dtype=np.float64)
        summary["confidence_margin"] = {
            "finite_n": int(finite.size),
            "mean": float(finite.mean()) if finite.size else None,
            "median": float(np.median(finite)) if finite.size else None,
        }
    return summary


def plurality_predictions(examples: list[LabeledQuestion]) -> list[int]:
    return [example.plurality_group_index for example in examples]


def paired_bootstrap_difference(
    examples: list[LabeledQuestion],
    first_predictions: list[int],
    second_predictions: list[int],
    *,
    ambiguous_only: bool,
    seed: int = 7506,
    samples: int = 10_000,
) -> dict:
    """Question-level 95% bootstrap interval for an accuracy difference."""

    if not (len(examples) == len(first_predictions) == len(second_predictions)):
        raise ValueError("Bootstrap inputs are not aligned")
    indices = [
        index
        for index, example in enumerate(examples)
        if not ambiguous_only or example.ambiguous
    ]
    first = np.asarray(
        [first_predictions[index] == examples[index].correct_group_index for index in indices],
        dtype=np.float32,
    )
    second = np.asarray(
        [second_predictions[index] == examples[index].correct_group_index for index in indices],
        dtype=np.float32,
    )
    paired = first - second
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(samples, dtype=np.float32)
    batch_size = 500
    for start in range(0, samples, batch_size):
        end = min(samples, start + batch_size)
        sampled = rng.integers(0, len(paired), size=(end - start, len(paired)))
        bootstrap[start:end] = paired[sampled].mean(axis=1)
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "n": len(paired),
        "samples": samples,
        "seed": seed,
        "accuracy_difference_percentage_points": 100.0 * float(paired.mean()),
        "ci95_lower_percentage_points": 100.0 * float(lower),
        "ci95_upper_percentage_points": 100.0 * float(upper),
    }
