from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from nemotron_reasoning_challenge.solver import NemotronReasoningSolver, detect_family


def run_holdout(train_csv: Path, eval_fraction: float = 0.2, seed: int = 7) -> None:
    df = pd.read_csv(train_csv).copy()
    df["family"] = df["prompt"].map(detect_family)

    train_parts = []
    eval_parts = []
    for _, group in df.groupby("family", sort=True):
        group = group.sample(frac=1.0, random_state=seed)
        split = int(len(group) * (1.0 - eval_fraction))
        train_parts.append(group.iloc[:split])
        eval_parts.append(group.iloc[split:])

    train_df = pd.concat(train_parts).reset_index(drop=True)
    eval_df = pd.concat(eval_parts).reset_index(drop=True)

    solver = NemotronReasoningSolver().fit(train_df)

    predictions = []
    for _, row in eval_df.iterrows():
        predictions.append(solver.solve(row["prompt"]))

    eval_df = eval_df.copy()
    eval_df["prediction"] = predictions
    eval_df["correct"] = eval_df["prediction"] == eval_df["answer"]

    overall = eval_df["correct"].mean()
    print(f"overall_accuracy={overall:.4f}")
    print()
    print("family_accuracy")
    family_scores = (
        eval_df.groupby("family")["correct"]
        .agg(["mean", "count", "sum"])
        .sort_index()
    )
    print(family_scores.to_string())
    print()
    print("sample_failures")
    failures = eval_df[~eval_df["correct"]].head(12)
    if failures.empty:
        print("none")
    else:
        for _, row in failures.iterrows():
            print(f"[{row['family']}] id={row['id']} pred={row['prediction']!r} gold={row['answer']!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=Path("/Users/marcelosilveira/iCloud Drive (Archive)/Documents/Playground/tmp/nemotron/train.csv"),
    )
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    run_holdout(args.train_csv, eval_fraction=args.eval_fraction, seed=args.seed)


if __name__ == "__main__":
    main()
