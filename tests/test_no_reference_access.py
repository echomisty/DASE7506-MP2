import unittest
from pathlib import Path


class ReferenceIsolationTests(unittest.TestCase):
    def test_inference_sources_do_not_open_reference_banks(self) -> None:
        root = Path(__file__).resolve().parents[1] / "code"
        inference_files = [
            root / "submission.py",
            root / "verifier" / "__init__.py",
            root / "verifier" / "answer_groups.py",
            root / "verifier" / "checkpoint.py",
            root / "verifier" / "features.py",
            root / "verifier" / "pipeline.py",
            root / "verifier" / "ranker.py",
            root / "verifier" / "stacked.py",
        ]
        forbidden = ("gsm8k_", "score_choices(", "read_jsonl(", "load_bank(")
        for path in inference_files:
            source = path.read_text()
            for marker in forbidden:
                self.assertNotIn(marker, source, f"{path.name} contains {marker}")


if __name__ == "__main__":
    unittest.main()
