import tempfile
import unittest
from pathlib import Path

import numpy as np

from verifier.checkpoint import load_checkpoint, save_checkpoint, save_stacked_checkpoint
from verifier.features import CANDIDATE_FEATURE_DIM, feature_names
from verifier.ranker import LinearRanker
from verifier.mlp import MLPRanker
from verifier.stacked import STACK_FEATURE_SETS, StackedRanker


class CheckpointTests(unittest.TestCase):
    def test_round_trip_and_resource_report(self) -> None:
        dimension = len(feature_names("consensus"))
        model = LinearRanker(
            feature_set="consensus",
            feature_mean=np.zeros(dimension, dtype=np.float32),
            feature_scale=np.ones(dimension, dtype=np.float32),
            weights=np.arange(dimension, dtype=np.float32),
        )
        with tempfile.TemporaryDirectory() as directory:
            report = save_checkpoint(Path(directory), model, {"source_split": "train"})
            loaded, metadata = load_checkpoint(Path(directory))
            self.assertTrue(report["within_limit"])
            self.assertEqual(metadata["training"]["source_split"], "train")
            np.testing.assert_array_equal(loaded.weights, model.weights)

    def test_stacked_round_trip(self) -> None:
        base_models = {}
        for feature_set in STACK_FEATURE_SETS:
            dimension = len(feature_names(feature_set))
            base_models[feature_set] = tuple(
                LinearRanker(
                    feature_set=feature_set,
                    feature_mean=np.zeros(dimension, dtype=np.float32),
                    feature_scale=np.ones(dimension, dtype=np.float32),
                    weights=np.full(dimension, index + 1, dtype=np.float32),
                )
                for index in range(2)
            )
        stack_dimension = len(STACK_FEATURE_SETS) + 1
        stack_model = LinearRanker(
            feature_set="stack",
            feature_mean=np.zeros(stack_dimension, dtype=np.float32),
            feature_scale=np.ones(stack_dimension, dtype=np.float32),
            weights=np.arange(stack_dimension, dtype=np.float32),
        )
        model = StackedRanker(base_models=base_models, stack_model=stack_model)
        with tempfile.TemporaryDirectory() as directory:
            report = save_stacked_checkpoint(
                Path(directory), model, {"source_split": "train"}
            )
            loaded, metadata = load_checkpoint(Path(directory))
            self.assertTrue(report["within_limit"])
            self.assertIsInstance(loaded, StackedRanker)
            self.assertEqual(loaded.models_per_feature_set, 2)
            self.assertEqual(metadata["training"]["source_split"], "train")
            np.testing.assert_array_equal(
                loaded.stack_model.weights, stack_model.weights
            )

    def test_candidate_stacked_round_trip(self) -> None:
        base_models = {}
        for feature_set in STACK_FEATURE_SETS:
            dimension = len(feature_names(feature_set))
            base_models[feature_set] = (
                LinearRanker(
                    feature_set=feature_set,
                    feature_mean=np.zeros(dimension, dtype=np.float32),
                    feature_scale=np.ones(dimension, dtype=np.float32),
                    weights=np.ones(dimension, dtype=np.float32),
                ),
            )
        candidate_models = (
            LinearRanker(
                feature_set="candidate",
                feature_mean=np.zeros(CANDIDATE_FEATURE_DIM, dtype=np.float32),
                feature_scale=np.ones(CANDIDATE_FEATURE_DIM, dtype=np.float32),
                weights=np.ones(CANDIDATE_FEATURE_DIM, dtype=np.float32),
            ),
        )
        stack_dimension = len(STACK_FEATURE_SETS) + 2
        stack_model = LinearRanker(
            feature_set="stack",
            feature_mean=np.zeros(stack_dimension, dtype=np.float32),
            feature_scale=np.ones(stack_dimension, dtype=np.float32),
            weights=np.arange(stack_dimension, dtype=np.float32),
        )
        model = StackedRanker(
            base_models=base_models,
            stack_model=stack_model,
            candidate_models=candidate_models,
        )
        with tempfile.TemporaryDirectory() as directory:
            save_stacked_checkpoint(Path(directory), model, {"source_split": "train"})
            loaded, metadata = load_checkpoint(Path(directory))
            self.assertEqual(len(loaded.candidate_models), 1)
            self.assertEqual(metadata["candidate_model_count"], 1)
            np.testing.assert_array_equal(
                loaded.candidate_models[0].weights,
                candidate_models[0].weights,
            )

    def test_candidate_mlp_stacked_round_trip(self) -> None:
        base_models = {}
        for feature_set in STACK_FEATURE_SETS:
            dimension = len(feature_names(feature_set))
            base_models[feature_set] = (
                LinearRanker(
                    feature_set=feature_set,
                    feature_mean=np.zeros(dimension, dtype=np.float32),
                    feature_scale=np.ones(dimension, dtype=np.float32),
                    weights=np.ones(dimension, dtype=np.float32),
                ),
            )
        candidate_models = (
            LinearRanker(
                feature_set="candidate",
                feature_mean=np.zeros(CANDIDATE_FEATURE_DIM, dtype=np.float32),
                feature_scale=np.ones(CANDIDATE_FEATURE_DIM, dtype=np.float32),
                weights=np.ones(CANDIDATE_FEATURE_DIM, dtype=np.float32),
            ),
        )
        mlp_model = MLPRanker(
            feature_mean=np.zeros(CANDIDATE_FEATURE_DIM, dtype=np.float32),
            feature_scale=np.ones(CANDIDATE_FEATURE_DIM, dtype=np.float32),
            input_weights=np.ones((CANDIDATE_FEATURE_DIM, 8), dtype=np.float32),
            hidden_bias=np.zeros(8, dtype=np.float32),
            output_weights=np.ones(8, dtype=np.float32),
            output_bias=np.zeros(1, dtype=np.float32),
        )
        stack_dimension = len(STACK_FEATURE_SETS) + 3
        stack_model = LinearRanker(
            feature_set="stack",
            feature_mean=np.zeros(stack_dimension, dtype=np.float32),
            feature_scale=np.ones(stack_dimension, dtype=np.float32),
            weights=np.arange(stack_dimension, dtype=np.float32),
        )
        model = StackedRanker(
            base_models=base_models,
            stack_model=stack_model,
            candidate_models=candidate_models,
            candidate_mlp_models=(mlp_model,),
        )
        with tempfile.TemporaryDirectory() as directory:
            report = save_stacked_checkpoint(Path(directory), model, {"source_split": "train"})
            loaded, metadata = load_checkpoint(Path(directory))
            self.assertTrue(report["within_limit"])
            self.assertEqual(metadata["format"], "dase7506-mp2-stacked-ranker-v3")
            self.assertEqual(metadata["candidate_mlp_model_count"], 1)
            np.testing.assert_array_equal(
                loaded.candidate_mlp_models[0].input_weights,
                mlp_model.input_weights,
            )


if __name__ == "__main__":
    unittest.main()
