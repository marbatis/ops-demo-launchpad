from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from nemotron_reasoning_challenge.analyze_symbolic_failures import analyze_symbolic_failures
from nemotron_reasoning_challenge.solver import NemotronReasoningSolver, detect_family


def build_holdout(train_csv: Path, eval_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    return train_df, eval_df


def evaluate_current_solver(train_csv: Path, eval_fraction: float, seed: int) -> dict:
    train_df, eval_df = build_holdout(train_csv, eval_fraction=eval_fraction, seed=seed)
    solver = NemotronReasoningSolver().fit(train_df)
    predictions = [solver.solve(prompt) for prompt in eval_df["prompt"]]

    eval_df = eval_df.copy()
    eval_df["prediction"] = predictions
    eval_df["correct"] = eval_df["prediction"] == eval_df["answer"]

    family_scores = (
        eval_df.groupby("family")["correct"]
        .agg(["mean", "count", "sum"])
        .sort_index()
        .reset_index()
        .to_dict(orient="records")
    )
    family_score_map = {row["family"]: row["mean"] for row in family_scores}
    return {
        "overall_accuracy": float(eval_df["correct"].mean()),
        "family_scores": family_scores,
        "family_score_map": family_score_map,
    }


def compare_to_baseline(metrics: dict, baseline: dict | None) -> dict:
    if baseline is None:
        return {
            "baseline_present": False,
            "decision": "no_baseline",
            "overall_delta": None,
            "equation_delta": None,
        }

    baseline_overall = baseline["overall_accuracy"]
    baseline_equation = baseline["family_score_map"]["equation"]
    current_overall = metrics["overall_accuracy"]
    current_equation = metrics["family_score_map"]["equation"]
    overall_delta = current_overall - baseline_overall
    equation_delta = current_equation - baseline_equation

    if overall_delta > 0 or equation_delta > 0:
        decision = "keep"
    elif overall_delta < 0 or equation_delta < 0:
        decision = "revert"
    else:
        decision = "neutral"

    return {
        "baseline_present": True,
        "decision": decision,
        "overall_delta": overall_delta,
        "equation_delta": equation_delta,
    }


def load_baseline(baseline_json: Path | None) -> dict | None:
    if baseline_json is None or not baseline_json.exists():
        return None
    return json.loads(baseline_json.read_text())


def write_run_artifacts(
    out_dir: Path,
    label: str,
    metrics: dict,
    comparison: dict,
    failure_df: pd.DataFrame,
) -> tuple[Path, Path]:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    stem = f"{timestamp}-{label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{stem}.json"
    csv_path = out_dir / f"{stem}-symbolic-failures.csv"

    payload = {
        "label": label,
        "timestamp": timestamp,
        "metrics": {
            "overall_accuracy": metrics["overall_accuracy"],
            "family_score_map": metrics["family_score_map"],
        },
        "comparison": comparison,
        "top_failure_clusters": failure_df[~failure_df["correct"]]["primary_cluster"].value_counts().to_dict(),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    failure_df.to_csv(csv_path, index=False)
    return json_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=Path("/Users/marcelosilveira/iCloud Drive (Archive)/Documents/Playground/tmp/nemotron/train.csv"),
    )
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--label", type=str, default="manual")
    parser.add_argument(
        "--baseline-json",
        type=Path,
        default=Path("/Users/marcelosilveira/iCloud Drive (Archive)/Documents/Playground/nemotron_reasoning_challenge/baseline_metrics.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/Users/marcelosilveira/iCloud Drive (Archive)/Documents/Playground/output/nemotron_loop_runs"),
    )
    args = parser.parse_args()

    metrics = evaluate_current_solver(args.train_csv, eval_fraction=args.eval_fraction, seed=args.seed)
    baseline = load_baseline(args.baseline_json)
    comparison = compare_to_baseline(metrics, baseline)
    failure_df = analyze_symbolic_failures(args.train_csv, eval_fraction=args.eval_fraction, seed=args.seed)
    json_path, csv_path = write_run_artifacts(
        out_dir=args.out_dir,
        label=args.label,
        metrics=metrics,
        comparison=comparison,
        failure_df=failure_df,
    )

    print(f"overall_accuracy={metrics['overall_accuracy']:.6f}")
    print(f"equation_accuracy={metrics['family_score_map']['equation']:.6f}")
    print(f"decision={comparison['decision']}")
    if comparison["baseline_present"]:
        print(f"overall_delta={comparison['overall_delta']:.6f}")
        print(f"equation_delta={comparison['equation_delta']:.6f}")
    print(f"wrote_run_json={json_path}")
    print(f"wrote_failure_csv={csv_path}")


if __name__ == "__main__":
    main()
