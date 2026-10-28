"""Reference-free prediction pipeline used by submission.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .answer_groups import build_answer_groups, plurality_group_index
from .checkpoint import load_checkpoint
from .features import group_feature_matrix
from .ranker import LinearRanker
from .stacked import StackedRanker


def _choose_group(
    row: dict,
    embeddings: np.ndarray,
    model: LinearRanker | StackedRanker | None,
) -> int:
    groups, parsed_answers = build_answer_groups(row)
    if not groups:
        return 0
    if len(groups) == 1:
        return int(groups[0].first_index)
    if model is None:
        plurality = plurality_group_index(groups)
        return int(groups[plurality].first_index)

    candidate_texts = [candidate["text"] for candidate in row["candidates"]]
    if isinstance(model, StackedRanker):
        scores = model.score_groups(
            question=row["question"],
            candidate_texts=candidate_texts,
            embeddings=embeddings,
            groups=groups,
            parsed_answers=parsed_answers,
        )
    else:
        features = group_feature_matrix(
            question=row["question"],
            candidate_texts=candidate_texts,
            embeddings=embeddings,
            groups=groups,
            parsed_answers=parsed_answers,
            feature_set=model.feature_set,
        )
        scores = model.score(features)
    best_group = max(
        range(len(groups)),
        key=lambda index: (float(scores[index]), -groups[index].first_index),
    )
    return int(groups[best_group].first_index)


def predict_choices(bank: list[dict], arrays: dict, checkpoint: Path | None) -> list[int]:
    """Return one built-in Python candidate index per question."""

    embeddings = np.asarray(arrays["embeddings"])
    if embeddings.shape != (len(bank), 4, 1024):
        raise ValueError("Expected embeddings with shape [N, 4, 1024]")
    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain non-finite values")
    model = None
    if checkpoint is not None:
        model, _ = load_checkpoint(Path(checkpoint))
    choices = [
        _choose_group(row, embeddings[index], model)
        for index, row in enumerate(bank)
    ]
    return [int(choice) for choice in choices]
