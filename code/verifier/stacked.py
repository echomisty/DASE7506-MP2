"""Inference representation for the compact stacked answer-group ranker."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .answer_groups import AnswerGroup
from .features import candidate_feature_matrix, group_feature_matrix
from .mlp import MLPRanker
from .ranker import LinearRanker


STACK_FEATURE_SETS = ("text", "embedding", "full_no_context", "full")


def stack_feature_matrix(
    groups: list[AnswerGroup] | tuple[AnswerGroup, ...],
    score_blocks: dict[str, np.ndarray],
) -> np.ndarray:
    columns = []
    score_names = (
        (*STACK_FEATURE_SETS, "candidate", "candidate_mlp")
        if "candidate_mlp" in score_blocks
        else (*STACK_FEATURE_SETS, "candidate")
        if "candidate" in score_blocks
        else STACK_FEATURE_SETS
    )
    for feature_set in score_names:
        scores = np.asarray(score_blocks[feature_set], dtype=np.float32)
        if scores.shape != (len(groups),):
            raise ValueError("A base-score block is not aligned with answer groups")
        columns.append(scores - scores.mean())
    support = np.asarray([group.support / 4.0 for group in groups], dtype=np.float32)
    columns.append(support - support.mean())
    return np.stack(columns, axis=1).astype(np.float32, copy=False)


@dataclass(frozen=True)
class StackedRanker:
    base_models: dict[str, tuple[LinearRanker, ...]]
    stack_model: LinearRanker
    candidate_models: tuple[LinearRanker, ...] = ()
    candidate_mlp_models: tuple[MLPRanker, ...] = ()

    def __post_init__(self) -> None:
        if tuple(self.base_models) != STACK_FEATURE_SETS:
            raise ValueError("Stacked ranker base feature sets are invalid")
        counts = {len(models) for models in self.base_models.values()}
        if len(counts) != 1 or not counts or next(iter(counts)) <= 0:
            raise ValueError("Stacked ranker needs equal non-empty base ensembles")
        for feature_set, models in self.base_models.items():
            if any(model.feature_set != feature_set for model in models):
                raise ValueError("A base model has the wrong feature set")
        if self.candidate_models:
            if len(self.candidate_models) != next(iter(counts)):
                raise ValueError("Candidate and group ensembles must be the same size")
            if any(model.feature_set != "candidate" for model in self.candidate_models):
                raise ValueError("A candidate model has the wrong feature set")
        if self.candidate_mlp_models:
            if len(self.candidate_mlp_models) != next(iter(counts)):
                raise ValueError("Candidate MLP and group ensembles must be the same size")
            if not self.candidate_models:
                raise ValueError("Candidate MLP ensemble requires linear candidate models")
        expected_stack_dim = (
            len(STACK_FEATURE_SETS)
            + 1
            + bool(self.candidate_models)
            + bool(self.candidate_mlp_models)
        )
        if self.stack_model.weights.shape != (expected_stack_dim,):
            raise ValueError("Stack model dimension is invalid")

    @property
    def models_per_feature_set(self) -> int:
        return len(next(iter(self.base_models.values())))

    def score_groups(
        self,
        question: str,
        candidate_texts: list[str],
        embeddings: np.ndarray,
        groups: list[AnswerGroup],
        parsed_answers: list[str | None],
    ) -> np.ndarray:
        score_blocks = {}
        for feature_set, models in self.base_models.items():
            features = group_feature_matrix(
                question=question,
                candidate_texts=candidate_texts,
                embeddings=embeddings,
                groups=groups,
                parsed_answers=parsed_answers,
                feature_set=feature_set,
            )
            scores = np.stack([model.score(features) for model in models], axis=0)
            score_blocks[feature_set] = scores.mean(axis=0)
        if self.candidate_models or self.candidate_mlp_models:
            features = candidate_feature_matrix(
                question=question,
                candidate_texts=candidate_texts,
                embeddings=embeddings,
                groups=list(groups),
                parsed_answers=parsed_answers,
            )
            if self.candidate_models:
                candidate_scores = np.stack(
                    [model.score(features) for model in self.candidate_models], axis=0
                ).mean(axis=0)
                score_blocks["candidate"] = np.asarray(
                    [max(float(candidate_scores[index]) for index in group.indices) for group in groups],
                    dtype=np.float32,
                )
            if self.candidate_mlp_models:
                mlp_scores = np.stack(
                    [model.score(features) for model in self.candidate_mlp_models], axis=0
                ).mean(axis=0)
                score_blocks["candidate_mlp"] = np.asarray(
                    [max(float(mlp_scores[index]) for index in group.indices) for group in groups],
                    dtype=np.float32,
                )
        return self.stack_model.score(stack_feature_matrix(groups, score_blocks))
