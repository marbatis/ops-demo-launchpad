from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from nemotron_reasoning_challenge.solver import solve_dataframe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=Path("/Users/marcelosilveira/iCloud Drive (Archive)/Documents/Playground/tmp/nemotron/train.csv"),
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=Path("/Users/marcelosilveira/iCloud Drive (Archive)/Documents/Playground/tmp/nemotron/test.csv"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("/Users/marcelosilveira/iCloud Drive (Archive)/Documents/Playground/output/nemotron_submission.csv"),
    )
    args = parser.parse_args()

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)
    submission = solve_dataframe(train_df, test_df)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output_csv, index=False)
    print(f"wrote {args.output_csv}")
    print(submission.to_string(index=False))


if __name__ == "__main__":
    main()
