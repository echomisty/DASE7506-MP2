import unittest

import numpy as np

from verifier.answer_groups import build_answer_groups
from verifier.features import FEATURE_SETS, feature_names, group_feature_matrix


class FeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.row = {
            "id": "features",
            "question": "A car travels 10 miles in 2 hours. What is its speed?",
            "candidates": [
                {"index": 0, "text": r"10 / 2 = 5. Final answer: \boxed{5}", "code": None},
                {"index": 1, "text": r"The speed is \boxed{5} miles per hour.", "code": None},
                {"index": 2, "text": r"10 * 2 = 20, so \boxed{20}", "code": None},
                {"index": 3, "text": r"The result is \boxed{4}", "code": None},
            ],
        }
        rng = np.random.default_rng(7)
        self.embeddings = rng.normal(size=(4, 1024)).astype(np.float32)

    def test_all_feature_sets_are_finite_and_deterministic(self) -> None:
        groups, parsed = build_answer_groups(self.row)
        texts = [candidate["text"] for candidate in self.row["candidates"]]
        for feature_set in FEATURE_SETS:
            first = group_feature_matrix(
                self.row["question"], texts, self.embeddings, groups, parsed, feature_set
            )
            second = group_feature_matrix(
                self.row["question"], texts, self.embeddings, groups, parsed, feature_set
            )
            self.assertEqual(first.shape, (len(groups), len(feature_names(feature_set))))
            self.assertTrue(np.isfinite(first).all())
            np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
