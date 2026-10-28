"""Small deterministic NumPy MLP ranker for candidate-level evidence."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class MLPRanker:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    input_weights: np.ndarray
    hidden_bias: np.ndarray
    output_weights: np.ndarray
    output_bias: np.ndarray

    def __post_init__(self) -> None:
        dimension, hidden = self.input_weights.shape
        if self.feature_mean.shape != (dimension,) or self.feature_scale.shape != (dimension,):
            raise ValueError("MLP input statistics do not match the weight matrix")
        if self.hidden_bias.shape != (hidden,) or self.output_weights.shape != (hidden,):
            raise ValueError("MLP hidden-layer arrays have inconsistent shapes")
        if self.output_bias.shape != (1,):
            raise ValueError("MLP output bias has an invalid shape")
        arrays = (
            self.feature_mean, self.feature_scale, self.input_weights,
            self.hidden_bias, self.output_weights, self.output_bias,
        )
        if not all(np.isfinite(array).all() for array in arrays):
            raise ValueError("MLP checkpoint arrays must be finite")
        if np.any(self.feature_scale <= 0):
            raise ValueError("MLP feature scales must be positive")

    @property
    def feature_dim(self) -> int:
        return int(self.feature_mean.size)

    def score(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError("Candidate feature matrix does not match the MLP")
        x = (features - self.feature_mean) / self.feature_scale
        hidden = np.maximum(x @ self.input_weights + self.hidden_bias, 0.0)
        scores = hidden @ self.output_weights + self.output_bias[0]
        if not np.isfinite(scores).all():
            raise ValueError("MLP produced non-finite scores")
        return scores


def fit_multi_positive_mlp_ranker(
    feature_matrices: list[np.ndarray],
    positive_indices: list[tuple[int, ...]],
    *,
    hidden_dim: int = 8,
    epochs: int = 100,
    learning_rate: float = 0.03,
    l2: float = 0.001,
    seed: int = 7506,
) -> tuple[MLPRanker, dict]:
    """Fit a question-balanced pairwise MLP using training data only."""

    if len(feature_matrices) != len(positive_indices) or not feature_matrices:
        raise ValueError("Candidate features and labels are not aligned")
    if hidden_dim <= 0 or epochs <= 0 or learning_rate <= 0 or l2 < 0:
        raise ValueError("Invalid MLP training parameters")
    usable = []
    for features, positives in zip(feature_matrices, positive_indices):
        features = np.asarray(features, dtype=np.float32)
        positives = tuple(sorted(set(int(index) for index in positives)))
        if features.ndim != 2 or not np.isfinite(features).all():
            raise ValueError("Candidate features must be finite matrices")
        if any(index < 0 or index >= len(features) for index in positives):
            raise ValueError("A positive candidate index is out of range")
        negatives = tuple(index for index in range(len(features)) if index not in positives)
        if positives and negatives:
            usable.append((features, positives, negatives))
    if not usable:
        raise ValueError("MLP pairwise training needs positive and negative candidates")

    all_features = np.concatenate([features for features, _, _ in usable], axis=0)
    feature_mean = all_features.mean(axis=0, dtype=np.float64).astype(np.float32)
    feature_scale = np.maximum(
        all_features.std(axis=0, dtype=np.float64).astype(np.float32),
        np.float32(1e-4),
    )
    positive_rows = []
    negative_rows = []
    pair_weights = []
    for features, positives, negatives in usable:
        standardized = (features - feature_mean) / feature_scale
        pair_weight = 1.0 / (len(positives) * len(negatives))
        for positive in positives:
            for negative in negatives:
                positive_rows.append(standardized[positive])
                negative_rows.append(standardized[negative])
                pair_weights.append(pair_weight)
    x_positive = np.asarray(positive_rows, dtype=np.float32)
    x_negative = np.asarray(negative_rows, dtype=np.float32)
    sample_weights = np.asarray(pair_weights, dtype=np.float32)
    total_weight = float(sample_weights.sum())

    rng = np.random.default_rng(seed)
    dimension = feature_mean.size
    input_weights = rng.normal(
        0.0, math.sqrt(2.0 / dimension), size=(dimension, hidden_dim)
    ).astype(np.float32)
    hidden_bias = np.zeros(hidden_dim, dtype=np.float32)
    output_weights = rng.normal(0.0, 0.01, size=hidden_dim).astype(np.float32)
    output_bias = np.zeros(1, dtype=np.float32)
    params = [input_weights, hidden_bias, output_weights, output_bias]
    first_moments = [np.zeros_like(param) for param in params]
    second_moments = [np.zeros_like(param) for param in params]
    best_loss = math.inf
    best_params = [param.copy() for param in params]
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8

    for step in range(1, epochs + 1):
        z_positive = x_positive @ input_weights + hidden_bias
        z_negative = x_negative @ input_weights + hidden_bias
        h_positive = np.maximum(z_positive, 0.0)
        h_negative = np.maximum(z_negative, 0.0)
        scores_positive = h_positive @ output_weights + output_bias[0]
        scores_negative = h_negative @ output_weights + output_bias[0]
        margins = scores_positive - scores_negative
        probability_wrong = 1.0 / (1.0 + np.exp(np.clip(margins, -40.0, 40.0)))
        grad_margin = (-sample_weights * probability_wrong / total_weight).astype(np.float32)
        grad_output = h_positive.T @ grad_margin - h_negative.T @ grad_margin
        grad_output += l2 * output_weights
        grad_hidden_positive = grad_margin[:, None] * output_weights
        grad_hidden_negative = -grad_margin[:, None] * output_weights
        grad_z_positive = grad_hidden_positive * (z_positive > 0)
        grad_z_negative = grad_hidden_negative * (z_negative > 0)
        grad_input = x_positive.T @ grad_z_positive + x_negative.T @ grad_z_negative
        grad_input += l2 * input_weights
        gradients = [
            grad_input.astype(np.float32),
            (grad_z_positive.sum(axis=0) + grad_z_negative.sum(axis=0)).astype(np.float32),
            grad_output.astype(np.float32),
            np.zeros_like(output_bias),
        ]
        for index, (param, gradient) in enumerate(zip(params, gradients)):
            first_moments[index] = beta1 * first_moments[index] + (1.0 - beta1) * gradient
            second_moments[index] = beta2 * second_moments[index] + (1.0 - beta2) * (gradient * gradient)
            corrected_first = first_moments[index] / (1.0 - beta1**step)
            corrected_second = second_moments[index] / (1.0 - beta2**step)
            param -= learning_rate * corrected_first / (np.sqrt(corrected_second) + epsilon)
        if step == 1 or step % 5 == 0 or step == epochs:
            loss = float(
                np.dot(sample_weights, np.logaddexp(0.0, -margins)) / total_weight
                + 0.5 * l2 * (np.sum(input_weights * input_weights) + np.sum(output_weights * output_weights))
            )
            if loss < best_loss:
                best_loss = loss
                best_params = [param.copy() for param in params]

    model = MLPRanker(feature_mean, feature_scale, *best_params)
    return model, {
        "questions": len(usable),
        "pairs": int(len(sample_weights)),
        "feature_dim": int(dimension),
        "hidden_dim": int(hidden_dim),
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "l2": float(l2),
        "seed": int(seed),
        "best_loss": float(best_loss),
    }
