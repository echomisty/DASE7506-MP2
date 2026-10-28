"""Small pairwise linear ranker with deterministic NumPy training/inference."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class LinearRanker:
    """Standardized linear scorer for answer-group feature vectors."""

    feature_set: str
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    weights: np.ndarray

    def __post_init__(self) -> None:
        dimension = self.weights.shape
        if self.feature_mean.shape != dimension or self.feature_scale.shape != dimension:
            raise ValueError("Ranker arrays have inconsistent shapes")
        if self.weights.ndim != 1:
            raise ValueError("Ranker arrays must be one-dimensional")
        if not all(
            np.isfinite(array).all()
            for array in (self.feature_mean, self.feature_scale, self.weights)
        ):
            raise ValueError("Ranker arrays must be finite")
        if np.any(self.feature_scale <= 0):
            raise ValueError("Ranker feature scales must be positive")

    @property
    def feature_dim(self) -> int:
        return int(self.weights.size)

    def score(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError("Feature matrix does not match the ranker")
        standardized = (features - self.feature_mean) / self.feature_scale
        scores = standardized @ self.weights
        if not np.isfinite(scores).all():
            raise ValueError("Ranker produced non-finite scores")
        return scores


def pairwise_accuracy(
    model: LinearRanker,
    feature_matrices: list[np.ndarray],
    correct_group_indices: list[int],
) -> float:
    """Return top-one answer-group accuracy for ambiguous questions."""

    correct = 0
    total = 0
    for features, expected in zip(feature_matrices, correct_group_indices):
        if len(features) < 2:
            continue
        scores = model.score(features)
        predicted = int(np.argmax(scores))
        correct += int(predicted == expected)
        total += 1
    return correct / total if total else float("nan")


def _weighted_pairwise_loss(
    differences: np.ndarray,
    sample_weights: np.ndarray,
    weights: np.ndarray,
    l2: float,
) -> float:
    margins = differences @ weights
    losses = np.logaddexp(0.0, -margins)
    data_loss = float(np.dot(sample_weights, losses) / sample_weights.sum())
    return data_loss + 0.5 * l2 * float(np.dot(weights, weights))


def fit_pairwise_ranker(
    feature_set: str,
    feature_matrices: list[np.ndarray],
    correct_group_indices: list[int],
    *,
    l2: float = 0.01,
    epochs: int = 240,
    learning_rate: float = 0.03,
    seed: int = 7506,
    tolerance: float = 1e-7,
) -> tuple[LinearRanker, dict]:
    """Fit a question-balanced pairwise logistic ranker.

    Each training question contributes total weight one, regardless of how many
    incorrect answer groups it contains. Only train-derived feature statistics
    are fitted and exported.
    """

    if len(feature_matrices) != len(correct_group_indices):
        raise ValueError("Feature matrices and labels are not aligned")
    if not feature_matrices:
        raise ValueError("No training questions were supplied")
    if l2 < 0 or epochs <= 0 or learning_rate <= 0:
        raise ValueError("Invalid ranker hyperparameters")

    usable = [
        (np.asarray(features, dtype=np.float32), int(correct))
        for features, correct in zip(feature_matrices, correct_group_indices)
        if len(features) >= 2
    ]
    if not usable:
        raise ValueError("Pairwise training needs ambiguous answer groups")
    dimension = usable[0][0].shape[1]
    if any(features.ndim != 2 or features.shape[1] != dimension for features, _ in usable):
        raise ValueError("Training feature dimensions are inconsistent")
    if any(not 0 <= correct < len(features) for features, correct in usable):
        raise ValueError("A correct answer-group label is out of range")

    all_features = np.concatenate([features for features, _ in usable], axis=0)
    feature_mean = all_features.mean(axis=0, dtype=np.float64).astype(np.float32)
    feature_scale = all_features.std(axis=0, dtype=np.float64).astype(np.float32)
    feature_scale = np.maximum(feature_scale, np.float32(1e-4))

    differences: list[np.ndarray] = []
    pair_weights: list[float] = []
    for features, correct in usable:
        negative_indices = [index for index in range(len(features)) if index != correct]
        question_weight = 1.0 / len(negative_indices)
        for negative in negative_indices:
            differences.append((features[correct] - features[negative]) / feature_scale)
            pair_weights.append(question_weight)
    difference_matrix = np.stack(differences).astype(np.float32, copy=False)
    sample_weights = np.asarray(pair_weights, dtype=np.float32)

    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 1e-4, size=dimension).astype(np.float32)
    first_moment = np.zeros_like(weights)
    second_moment = np.zeros_like(weights)
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    weight_total = float(sample_weights.sum())
    losses: list[float] = []
    best_loss = math.inf
    best_weights = weights.copy()
    best_epoch = 0
    stable_checks = 0
    completed_epochs = 0

    for epoch in range(1, epochs + 1):
        margins = difference_matrix @ weights
        probability_wrong_order = 1.0 / (1.0 + np.exp(np.clip(margins, -40.0, 40.0)))
        gradient_weights = sample_weights * probability_wrong_order
        gradient = -(difference_matrix.T @ gradient_weights) / weight_total + l2 * weights
        gradient = gradient.astype(np.float32, copy=False)

        first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
        second_moment = beta2 * second_moment + (1.0 - beta2) * (gradient * gradient)
        corrected_first = first_moment / (1.0 - beta1**epoch)
        corrected_second = second_moment / (1.0 - beta2**epoch)
        weights -= learning_rate * corrected_first / (np.sqrt(corrected_second) + epsilon)
        completed_epochs = epoch

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            loss = _weighted_pairwise_loss(
                difference_matrix, sample_weights, weights, l2
            )
            losses.append(loss)
            if math.isfinite(best_loss):
                relative_improvement = (best_loss - loss) / max(1.0, abs(best_loss))
            else:
                relative_improvement = math.inf
            if loss < best_loss:
                best_loss = loss
                best_weights = weights.copy()
                best_epoch = epoch
            stable_checks = stable_checks + 1 if relative_improvement < tolerance else 0
            if stable_checks >= 4:
                break

    weights = best_weights

    model = LinearRanker(
        feature_set=feature_set,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        weights=weights.astype(np.float32, copy=False),
    )
    diagnostics = {
        "questions": len(usable),
        "pairs": len(difference_matrix),
        "feature_dim": dimension,
        "l2": l2,
        "requested_epochs": epochs,
        "completed_epochs": completed_epochs,
        "best_epoch": best_epoch,
        "best_loss": best_loss,
        "learning_rate": learning_rate,
        "seed": seed,
        "loss_history": losses,
        "training_pairwise_top1_accuracy": pairwise_accuracy(
            model,
            [features for features, _ in usable],
            [correct for _, correct in usable],
        ),
    }
    return model, diagnostics


def fit_multi_positive_pairwise_ranker(
    feature_set: str,
    feature_matrices: list[np.ndarray],
    positive_indices: list[tuple[int, ...]],
    *,
    l2: float = 0.01,
    epochs: int = 240,
    learning_rate: float = 0.03,
    seed: int = 7506,
    tolerance: float = 1e-7,
) -> tuple[LinearRanker, dict]:
    """Fit a question-balanced candidate ranker with one or more positives."""

    if len(feature_matrices) != len(positive_indices) or not feature_matrices:
        raise ValueError("Candidate feature matrices and labels are not aligned")
    if l2 < 0 or epochs <= 0 or learning_rate <= 0:
        raise ValueError("Invalid ranker hyperparameters")
    usable = []
    for features, positives in zip(feature_matrices, positive_indices):
        features = np.asarray(features, dtype=np.float32)
        positives = tuple(sorted(set(int(index) for index in positives)))
        negatives = tuple(index for index in range(len(features)) if index not in positives)
        if positives and negatives:
            usable.append((features, positives, negatives))
    if not usable:
        raise ValueError("Multi-positive training needs positive and negative candidates")
    dimension = usable[0][0].shape[1]
    if any(
        features.ndim != 2 or features.shape[1] != dimension
        for features, _, _ in usable
    ):
        raise ValueError("Training feature dimensions are inconsistent")
    if any(
        positive < 0 or positive >= len(features)
        for features, positives, _ in usable
        for positive in positives
    ):
        raise ValueError("A positive candidate label is out of range")

    all_features = np.concatenate([features for features, _, _ in usable], axis=0)
    feature_mean = all_features.mean(axis=0, dtype=np.float64).astype(np.float32)
    feature_scale = all_features.std(axis=0, dtype=np.float64).astype(np.float32)
    feature_scale = np.maximum(feature_scale, np.float32(1e-4))

    differences = []
    pair_weights = []
    for features, positives, negatives in usable:
        question_weight = 1.0 / (len(positives) * len(negatives))
        for positive in positives:
            for negative in negatives:
                differences.append(
                    (features[positive] - features[negative]) / feature_scale
                )
                pair_weights.append(question_weight)
    difference_matrix = np.stack(differences).astype(np.float32, copy=False)
    sample_weights = np.asarray(pair_weights, dtype=np.float32)

    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 1e-4, size=dimension).astype(np.float32)
    first_moment = np.zeros_like(weights)
    second_moment = np.zeros_like(weights)
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    weight_total = float(sample_weights.sum())
    losses = []
    best_loss = math.inf
    best_weights = weights.copy()
    best_epoch = 0
    stable_checks = 0
    completed_epochs = 0

    for epoch in range(1, epochs + 1):
        margins = difference_matrix @ weights
        probability_wrong_order = 1.0 / (
            1.0 + np.exp(np.clip(margins, -40.0, 40.0))
        )
        gradient_weights = sample_weights * probability_wrong_order
        gradient = (
            -(difference_matrix.T @ gradient_weights) / weight_total + l2 * weights
        ).astype(np.float32, copy=False)
        first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
        second_moment = beta2 * second_moment + (1.0 - beta2) * (gradient * gradient)
        corrected_first = first_moment / (1.0 - beta1**epoch)
        corrected_second = second_moment / (1.0 - beta2**epoch)
        weights -= learning_rate * corrected_first / (
            np.sqrt(corrected_second) + epsilon
        )
        completed_epochs = epoch

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            loss = _weighted_pairwise_loss(
                difference_matrix, sample_weights, weights, l2
            )
            losses.append(loss)
            relative_improvement = (
                (best_loss - loss) / max(1.0, abs(best_loss))
                if math.isfinite(best_loss)
                else math.inf
            )
            if loss < best_loss:
                best_loss = loss
                best_weights = weights.copy()
                best_epoch = epoch
            stable_checks = stable_checks + 1 if relative_improvement < tolerance else 0
            if stable_checks >= 4:
                break

    model = LinearRanker(
        feature_set=feature_set,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        weights=best_weights.astype(np.float32, copy=False),
    )
    training_correct = sum(
        int(int(np.argmax(model.score(features))) in positives)
        for features, positives, _ in usable
    )
    diagnostics = {
        "objective": "multi_positive_pairwise_logistic",
        "questions": len(usable),
        "pairs": len(difference_matrix),
        "feature_dim": dimension,
        "l2": l2,
        "requested_epochs": epochs,
        "completed_epochs": completed_epochs,
        "best_epoch": best_epoch,
        "best_loss": best_loss,
        "learning_rate": learning_rate,
        "seed": seed,
        "loss_history": losses,
        "training_top1_accuracy": training_correct / len(usable),
    }
    return model, diagnostics


def fit_listwise_ranker(
    feature_set: str,
    feature_matrices: list[np.ndarray],
    correct_group_indices: list[int],
    *,
    l2: float = 0.01,
    epochs: int = 240,
    learning_rate: float = 0.03,
    seed: int = 7506,
    tolerance: float = 1e-7,
) -> tuple[LinearRanker, dict]:
    """Fit a question-balanced multiclass softmax answer-group ranker."""

    if len(feature_matrices) != len(correct_group_indices):
        raise ValueError("Feature matrices and labels are not aligned")
    if not feature_matrices:
        raise ValueError("No training questions were supplied")
    if l2 < 0 or epochs <= 0 or learning_rate <= 0:
        raise ValueError("Invalid ranker hyperparameters")

    usable = [
        (np.asarray(features, dtype=np.float32), int(correct))
        for features, correct in zip(feature_matrices, correct_group_indices)
        if len(features) >= 2
    ]
    if not usable:
        raise ValueError("Listwise training needs ambiguous answer groups")
    dimension = usable[0][0].shape[1]
    if any(features.ndim != 2 or features.shape[1] != dimension for features, _ in usable):
        raise ValueError("Training feature dimensions are inconsistent")
    if any(not 0 <= correct < len(features) for features, correct in usable):
        raise ValueError("A correct answer-group label is out of range")

    all_features = np.concatenate([features for features, _ in usable], axis=0)
    feature_mean = all_features.mean(axis=0, dtype=np.float64).astype(np.float32)
    feature_scale = all_features.std(axis=0, dtype=np.float64).astype(np.float32)
    feature_scale = np.maximum(feature_scale, np.float32(1e-4))
    standardized = ((all_features - feature_mean) / feature_scale).astype(
        np.float32, copy=False
    )

    slices = []
    offset = 0
    for features, correct in usable:
        end = offset + len(features)
        slices.append((offset, end, correct))
        offset = end

    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 1e-4, size=dimension).astype(np.float32)
    first_moment = np.zeros_like(weights)
    second_moment = np.zeros_like(weights)
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    losses: list[float] = []
    best_loss = math.inf
    best_weights = weights.copy()
    best_epoch = 0
    stable_checks = 0
    completed_epochs = 0

    for epoch in range(1, epochs + 1):
        scores = standardized @ weights
        score_gradient = np.zeros_like(scores)
        for start, end, correct in slices:
            local = scores[start:end]
            shifted = local - float(local.max())
            probabilities = np.exp(shifted)
            probabilities /= probabilities.sum()
            score_gradient[start:end] = probabilities
            score_gradient[start + correct] -= 1.0
        score_gradient /= len(slices)
        gradient = standardized.T @ score_gradient + l2 * weights
        gradient = gradient.astype(np.float32, copy=False)

        first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
        second_moment = beta2 * second_moment + (1.0 - beta2) * (gradient * gradient)
        corrected_first = first_moment / (1.0 - beta1**epoch)
        corrected_second = second_moment / (1.0 - beta2**epoch)
        weights -= learning_rate * corrected_first / (np.sqrt(corrected_second) + epsilon)
        completed_epochs = epoch

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            scores = standardized @ weights
            data_loss = 0.0
            for start, end, correct in slices:
                local = scores[start:end]
                maximum = float(local.max())
                data_loss += (
                    maximum
                    + math.log(float(np.exp(local - maximum).sum()))
                    - float(local[correct])
                )
            loss = data_loss / len(slices) + 0.5 * l2 * float(
                np.dot(weights, weights)
            )
            losses.append(loss)
            if math.isfinite(best_loss):
                relative_improvement = (best_loss - loss) / max(1.0, abs(best_loss))
            else:
                relative_improvement = math.inf
            if loss < best_loss:
                best_loss = loss
                best_weights = weights.copy()
                best_epoch = epoch
            stable_checks = stable_checks + 1 if relative_improvement < tolerance else 0
            if stable_checks >= 4:
                break

    model = LinearRanker(
        feature_set=feature_set,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        weights=best_weights.astype(np.float32, copy=False),
    )
    diagnostics = {
        "objective": "listwise_softmax",
        "questions": len(usable),
        "groups": len(all_features),
        "feature_dim": dimension,
        "l2": l2,
        "requested_epochs": epochs,
        "completed_epochs": completed_epochs,
        "best_epoch": best_epoch,
        "best_loss": best_loss,
        "learning_rate": learning_rate,
        "seed": seed,
        "loss_history": losses,
        "training_top1_accuracy": pairwise_accuracy(
            model,
            [features for features, _ in usable],
            [correct for _, correct in usable],
        ),
    }
    return model, diagnostics


def fit_hardest_negative_ranker(
    feature_set: str,
    feature_matrices: list[np.ndarray],
    correct_group_indices: list[int],
    *,
    l2: float = 0.01,
    epochs: int = 240,
    learning_rate: float = 0.03,
    seed: int = 7506,
    tolerance: float = 1e-7,
) -> tuple[LinearRanker, dict]:
    """Fit logistic ranking against only each question's highest-scoring error."""

    if len(feature_matrices) != len(correct_group_indices):
        raise ValueError("Feature matrices and labels are not aligned")
    if not feature_matrices:
        raise ValueError("No training questions were supplied")
    if l2 < 0 or epochs <= 0 or learning_rate <= 0:
        raise ValueError("Invalid ranker hyperparameters")
    usable = [
        (np.asarray(features, dtype=np.float32), int(correct))
        for features, correct in zip(feature_matrices, correct_group_indices)
        if len(features) >= 2
    ]
    if not usable:
        raise ValueError("Hard-negative training needs ambiguous answer groups")
    dimension = usable[0][0].shape[1]
    if any(features.ndim != 2 or features.shape[1] != dimension for features, _ in usable):
        raise ValueError("Training feature dimensions are inconsistent")
    if any(not 0 <= correct < len(features) for features, correct in usable):
        raise ValueError("A correct answer-group label is out of range")

    all_features = np.concatenate([features for features, _ in usable], axis=0)
    feature_mean = all_features.mean(axis=0, dtype=np.float64).astype(np.float32)
    feature_scale = all_features.std(axis=0, dtype=np.float64).astype(np.float32)
    feature_scale = np.maximum(feature_scale, np.float32(1e-4))
    standardized = ((all_features - feature_mean) / feature_scale).astype(
        np.float32, copy=False
    )
    slices = []
    offset = 0
    for features, correct in usable:
        end = offset + len(features)
        slices.append((offset, end, correct))
        offset = end

    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 1e-4, size=dimension).astype(np.float32)
    first_moment = np.zeros_like(weights)
    second_moment = np.zeros_like(weights)
    losses: list[float] = []
    best_loss = math.inf
    best_weights = weights.copy()
    best_epoch = 0
    stable_checks = 0
    completed_epochs = 0

    for epoch in range(1, epochs + 1):
        scores = standardized @ weights
        positive_indices = np.empty(len(slices), dtype=np.int64)
        negative_indices = np.empty(len(slices), dtype=np.int64)
        for index, (start, end, correct) in enumerate(slices):
            local = scores[start:end]
            wrong = [candidate for candidate in range(end - start) if candidate != correct]
            hardest = max(wrong, key=lambda candidate: (float(local[candidate]), -candidate))
            positive_indices[index] = start + correct
            negative_indices[index] = start + hardest
        margins = scores[positive_indices] - scores[negative_indices]
        probability_wrong = 1.0 / (1.0 + np.exp(np.clip(margins, -40.0, 40.0)))
        score_gradient = np.zeros_like(scores)
        pair_gradient = -probability_wrong / len(slices)
        np.add.at(score_gradient, positive_indices, pair_gradient)
        np.add.at(score_gradient, negative_indices, -pair_gradient)
        gradient = standardized.T @ score_gradient + l2 * weights
        gradient = gradient.astype(np.float32, copy=False)

        first_moment = 0.9 * first_moment + 0.1 * gradient
        second_moment = 0.999 * second_moment + 0.001 * (gradient * gradient)
        corrected_first = first_moment / (1.0 - 0.9**epoch)
        corrected_second = second_moment / (1.0 - 0.999**epoch)
        weights -= learning_rate * corrected_first / (np.sqrt(corrected_second) + 1e-8)
        completed_epochs = epoch

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            scores = standardized @ weights
            margins = []
            for start, end, correct in slices:
                local = scores[start:end]
                hardest_score = max(
                    float(local[candidate])
                    for candidate in range(end - start)
                    if candidate != correct
                )
                margins.append(float(local[correct]) - hardest_score)
            data_loss = float(np.logaddexp(0.0, -np.asarray(margins)).mean())
            loss = data_loss + 0.5 * l2 * float(np.dot(weights, weights))
            losses.append(loss)
            if math.isfinite(best_loss):
                relative_improvement = (best_loss - loss) / max(1.0, abs(best_loss))
            else:
                relative_improvement = math.inf
            if loss < best_loss:
                best_loss = loss
                best_weights = weights.copy()
                best_epoch = epoch
            stable_checks = stable_checks + 1 if relative_improvement < tolerance else 0
            if stable_checks >= 4:
                break

    model = LinearRanker(
        feature_set=feature_set,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        weights=best_weights.astype(np.float32, copy=False),
    )
    diagnostics = {
        "objective": "hardest_negative_logistic",
        "questions": len(usable),
        "groups": len(all_features),
        "feature_dim": dimension,
        "l2": l2,
        "requested_epochs": epochs,
        "completed_epochs": completed_epochs,
        "best_epoch": best_epoch,
        "best_loss": best_loss,
        "learning_rate": learning_rate,
        "seed": seed,
        "loss_history": losses,
        "training_top1_accuracy": pairwise_accuracy(
            model,
            [features for features, _ in usable],
            [correct for _, correct in usable],
        ),
    }
    return model, diagnostics
