import unittest

from verifier.answer_groups import build_answer_groups, plurality_group_index


def candidate(index: int, answer_text: str) -> dict:
    return {"index": index, "text": answer_text, "code": None}


class AnswerGroupTests(unittest.TestCase):
    def test_equivalent_numbers_share_a_group(self) -> None:
        row = {
            "id": "synthetic",
            "question": "What is one half?",
            "candidates": [
                candidate(0, r"Answer: \boxed{1/2}"),
                candidate(1, r"Answer: \boxed{0.5}"),
                candidate(2, r"Answer: \boxed{2}"),
                candidate(3, "No final numeric answer"),
            ],
        }
        groups, parsed = build_answer_groups(row)
        self.assertEqual(parsed, ["1/2", "1/2", "2", None])
        self.assertEqual([group.indices for group in groups], [(0, 1), (2,)])
        self.assertEqual(plurality_group_index(groups), 0)

    def test_plurality_tie_uses_first_candidate(self) -> None:
        row = {
            "id": "tie",
            "question": "Choose.",
            "candidates": [
                candidate(0, r"\boxed{10}"),
                candidate(1, r"\boxed{20}"),
                candidate(2, r"\boxed{10}"),
                candidate(3, r"\boxed{20}"),
            ],
        }
        groups, _ = build_answer_groups(row)
        self.assertEqual(plurality_group_index(groups), 0)


if __name__ == "__main__":
    unittest.main()
