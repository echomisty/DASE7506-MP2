import unittest

import numpy as np

from submission import predict


class SubmissionContractTests(unittest.TestCase):
    def test_plurality_fallback_returns_python_integers(self) -> None:
        bank = [
            {
                "id": "contract",
                "question": "What is 2 + 2?",
                "candidates": [
                    {"index": 0, "text": r"\boxed{4}", "code": None},
                    {"index": 1, "text": r"\boxed{5}", "code": None},
                    {"index": 2, "text": r"\boxed{4}", "code": None},
                    {"index": 3, "text": r"\boxed{6}", "code": None},
                ],
            }
        ]
        arrays = {"embeddings": np.zeros((1, 4, 1024), dtype=np.float32)}
        result = predict(bank, arrays, None)
        self.assertEqual(result, [0])
        self.assertIs(type(result[0]), int)


if __name__ == "__main__":
    unittest.main()
