from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

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


def primary_cluster(
    n_same_op: int,
    uniform_rhs_len: bool,
    pred: str | None,
    answer_visible_anywhere: bool,
) -> str:
    if n_same_op == 0:
        if pred is None:
            return "zero_shot_no_answer"
        return "zero_shot_copyish_fallback"
    if n_same_op == 1 and pred is None:
        if answer_visible_anywhere:
            return "one_shot_visible_but_unresolved"
        return "one_shot_novel_char_gap"
    if n_same_op >= 2 and pred is None:
        if uniform_rhs_len:
            return "multi_example_uniform_dsl_gap"
        return "multi_example_mixed_length_gap"
    if pred is not None:
        return "wrong_prior_or_partial_copy"
    return "other"


def analyze_symbolic_failures(train_csv: Path, eval_fraction: float, seed: int) -> pd.DataFrame:
    train_df, eval_df = build_holdout(train_csv, eval_fraction=eval_fraction, seed=seed)
    solver = NemotronReasoningSolver().fit(train_df)

    rows = []
    for _, row in eval_df.iterrows():
        if row["family"] != "equation" or solver.detect_equation_kind(row["prompt"]) != "symbolic":
            continue

        gold = str(row["answer"])
        pred = solver.solve(row["prompt"])
        target = solver.parse_equation_target(row["prompt"])
        target_op = target[2]
        pairs = solver.parse_equation_pairs(row["prompt"])
        same_op = [(lhs, rhs) for lhs, rhs in pairs if lhs[2] == target_op]
        same_op_rhs_lengths = sorted({len(rhs) for _, rhs in same_op})
        templates = solver.symbolic_subsequence_templates(same_op)
        visible_text = "".join(lhs + rhs for lhs, rhs in pairs) + target
        answer_chars = set(gold)
        pred_chars = set(pred) if pred is not None else set()
        cluster = primary_cluster(
            n_same_op=len(same_op),
            uniform_rhs_len=len(same_op_rhs_lengths) == 1,
            pred=pred,
            answer_visible_anywhere=answer_chars.issubset(set(visible_text)),
        )

        rows.append(
            {
                "id": row["id"],
                "target": target,
                "target_op": target_op,
                "gold": gold,
                "pred": pred,
                "correct": pred == gold,
                "n_pairs": len(pairs),
                "n_same_op": len(same_op),
                "same_op_rhs_lengths": ",".join(map(str, same_op_rhs_lengths)) if same_op_rhs_lengths else "",
                "same_op_uniform_len": len(same_op_rhs_lengths) == 1,
                "same_op_template_count": len(templates),
                "answer_len": len(gold),
                "answer_chars_all_in_target": answer_chars.issubset(set(target)),
                "answer_chars_all_visible_anywhere": answer_chars.issubset(set(visible_text)),
                "pred_is_none": pred is None,
                "pred_chars_all_in_target": bool(pred_chars) and pred_chars.issubset(set(target)),
                "primary_cluster": cluster,
            }
        )

    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame) -> None:
    misses = df[~df["correct"]].copy()
    print(f"symbolic_total={len(df)}")
    print(f"symbolic_correct={int(df['correct'].sum())}")
    print(f"symbolic_misses={len(misses)}")
    print()
    print("misses_by_primary_cluster")
    print(misses["primary_cluster"].value_counts().to_string())
    print()
    print("misses_by_target_op")
    print(misses["target_op"].value_counts().head(12).to_string())
    print()
    print("misses_by_same_op_count")
    print(misses["n_same_op"].value_counts().sort_index().to_string())
    print()
    print("sample_misses")
    cols = [
        "id",
        "target",
        "gold",
        "pred",
        "target_op",
        "n_same_op",
        "same_op_rhs_lengths",
        "primary_cluster",
    ]
    print(misses[cols].head(20).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=Path("/Users/marcelosilveira/iCloud Drive (Archive)/Documents/Playground/tmp/nemotron/train.csv"),
    )
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()

    analysis_df = analyze_symbolic_failures(
        train_csv=args.train_csv,
        eval_fraction=args.eval_fraction,
        seed=args.seed,
    )
    print_summary(analysis_df)
    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        analysis_df.to_csv(args.output_csv, index=False)
        print()
        print(f"wrote {args.output_csv}")


if __name__ == "__main__":
    main()
