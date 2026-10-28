"""Fresh-process prediction, scoring, timing, and inference-asset reporting."""

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from verifier.checkpoint import checkpoint_report


def run(command: list[str], environment: dict[str, str]) -> tuple[float, str]:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    return time.perf_counter() - started, completed.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--method", type=Path, default=Path("submission.py"))
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Required explicit acknowledgement for a frozen test evaluation.",
    )
    args = parser.parse_args()
    if args.split == "test" and not args.allow_test:
        raise SystemExit("Refusing test evaluation without --allow-test")
    if args.repeats < 1:
        raise SystemExit("--repeats must be positive")

    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    predict_command = [
        sys.executable,
        "predict.py",
        "--method",
        str(args.method),
        "--checkpoint",
        str(args.checkpoint),
        "--split",
        args.split,
        "--output",
        str(args.predictions),
    ]
    score_command = [
        sys.executable,
        "score_predictions.py",
        "--split",
        args.split,
        "--predictions",
        str(args.predictions),
        "--output",
        str(args.score),
    ]

    runs = []
    for repeat in range(args.repeats):
        prediction_seconds, prediction_stdout = run(predict_command, environment)
        scoring_seconds, scoring_stdout = run(score_command, environment)
        runs.append(
            {
                "repeat": repeat + 1,
                "prediction_process_seconds": prediction_seconds,
                "scoring_process_seconds": scoring_seconds,
                "total_seconds": prediction_seconds + scoring_seconds,
                "prediction_stdout": prediction_stdout,
                "scoring_stdout": scoring_stdout,
            }
        )

    score = json.loads(args.score.read_text())
    totals = [item["total_seconds"] for item in runs]
    report = {
        "split": args.split,
        "repeats": args.repeats,
        "commands": {
            "prediction": predict_command,
            "scoring": score_command,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "thread_variables": {
                name: environment[name]
                for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
            },
        },
        "checkpoint": checkpoint_report(args.checkpoint),
        "runs": runs,
        "timing_summary": {
            "minimum_seconds": min(totals),
            "mean_seconds": sum(totals) / len(totals),
            "maximum_seconds": max(totals),
            "reference_limit_seconds": 30.0,
            "local_maximum_within_limit": max(totals) <= 30.0,
        },
        "official_score": score,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"timing": report["timing_summary"], "score": score}, indent=2))


if __name__ == "__main__":
    main()
