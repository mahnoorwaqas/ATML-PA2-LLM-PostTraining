from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.logging_utils import save_json
from task5_feedback.rlvr import exact_reward
from task5_feedback.rlaif import PairwiseAIJudge

EXPECTED_VARIANTS = {
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
}

# (category, diagnostically-better variant, other variant, family)
#   reasoning : final answer held correct, reasoning degraded           -> S_reason
#   outcome   : reasoning ~fixed, designated final answer changed       -> S_outcome
#   style     : both correct; second adds irrelevant persuasive filler  -> preferring the filler is a WRONG preference
#               (a verifier tie is the legitimate/expected verifier behaviour)
PAIRS = [
    ("corrupt_reasoning", "clean_correct", "corrupt_reasoning_correct_final", "reasoning"),
    ("wrong_final_good_reasoning", "clean_correct", "good_reasoning_wrong_final", "outcome"),
    ("gold_distractor_wrong_final", "clean_correct", "gold_distractor_wrong_final", "outcome"),
    ("persuasive_filler", "clean_correct", "persuasive_filler_correct", "style"),
    ("filler_vs_wrong_final", "persuasive_filler_correct", "good_reasoning_wrong_final", "outcome_under_filler"),
]


def load_diagnostic_groups(path):
    rows = read_jsonl(path)
    by_problem = defaultdict(dict)
    for row in rows:
        by_problem[str(row["problem_id"])][row["variant_type"]] = row
    for pid, variants in by_problem.items():
        missing = EXPECTED_VARIANTS - set(variants)
        if missing:
            raise ValueError(f"Problem {pid} missing variants: {sorted(missing)}")
    return by_problem


def _first(row: dict, keys, what: str):
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    raise KeyError(f"cannot find {what}; row keys = {sorted(row)}")


def resp_of(row):
    return str(_first(row, ("response", "completion", "solution", "text", "answer_text"), "response text"))


def gold_of(row):
    return str(_first(row, ("gold_final", "gold", "gold_answer"), "gold answer")).replace(",", "")


def question_of(row):
    return str(_first(row, ("question", "problem", "prompt"), "question"))


def outcome_label(better_score: float, other_score: float) -> str:
    if better_score > other_score:
        return "better"
    if better_score == other_score:
        return "tie"
    return "wrong"


def judge_label(judge: PairwiseAIJudge, q: str, better: str, other: str, both_orders: bool):
    """Return list of labels (better/tie/wrong) from one or two argument orders."""
    labs = []
    p = judge.compare(q, better, other)  # "A" => better preferred
    labs.append({"A": "better", "TIE": "tie", "B": "wrong"}[p])
    if both_orders:
        p2 = judge.compare(q, other, better)  # "B" => better preferred
        labs.append({"B": "better", "TIE": "tie", "A": "wrong"}[p2])
    return labs


def summarize(df: pd.DataFrame, mech: str) -> pd.DataFrame:
    g = df.groupby("category")[mech].apply(
        lambda s: pd.Series({"better_rate": (s == "better").mean(), "tie_rate": (s == "tie").mean(), "wrong_rate": (s == "wrong").mean(), "n": len(s)})
    ).unstack()
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--single-order", action="store_true", help="judge each pair once (released orientation) instead of both argument orders")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    groups = load_diagnostic_groups(cfg["paths"]["task5_diagnostics"])
    print("Diagnostic problems:", len(groups))
    print("Variants/problem:", sorted(EXPECTED_VARIANTS))
    outdir = repo_path(cfg["results_dir"]) / "task5_feedback"
    outdir.mkdir(parents=True, exist_ok=True)
    judge = PairwiseAIJudge(cfg, outdir / "judge_cache.json")
    both = not args.single_order

    records = []
    for pid, v in groups.items():
        q = question_of(v["clean_correct"])
        gold = gold_of(v["clean_correct"])
        for cat, better_v, other_v, family in PAIRS:
            rb, ro = resp_of(v[better_v]), resp_of(v[other_v])
            # RLVR: exact binary verifier on the designated final answer
            vb, vo = exact_reward(rb, gold), exact_reward(ro, gold)
            ver = outcome_label(vb, vo)
            jl = judge_label(judge, q, rb, ro, both)
            records.append(
                {
                    "problem_id": pid, "category": cat, "family": family, "better_variant": better_v, "other_variant": other_v,
                    "verifier_score_better": vb, "verifier_score_other": vo, "verifier": ver,
                    "judge_orders": jl,
                    # per-order labels flattened to a single fractional outcome for the headline rates
                    "judge": max(set(jl), key=jl.count) if len(set(jl)) == 1 else "inconsistent",
                    "judge_better_frac": jl.count("better") / len(jl),
                    "judge_tie_frac": jl.count("tie") / len(jl),
                    "judge_wrong_frac": jl.count("wrong") / len(jl),
                }
            )
    write_jsonl(outdir / "perturbation_pairs.jsonl", records)
    df = pd.DataFrame(records)

    table_rows = []
    for cat in [p[0] for p in PAIRS]:
        sub = df[df["category"] == cat]
        table_rows.append(
            {
                "category": cat, "family": sub["family"].iloc[0], "n_pairs": len(sub),
                "verifier_better": float((sub["verifier"] == "better").mean()),
                "verifier_tie": float((sub["verifier"] == "tie").mean()),
                "verifier_wrong": float((sub["verifier"] == "wrong").mean()),
                "judge_better": float(sub["judge_better_frac"].mean()),
                "judge_tie": float(sub["judge_tie_frac"].mean()),
                "judge_wrong": float(sub["judge_wrong_frac"].mean()),
                "judge_order_inconsistent": float((sub["judge"] == "inconsistent").mean()),
            }
        )
    table = pd.DataFrame(table_rows)
    table.to_csv(outdir / "perturbation_table.csv", index=False)
    print(table.round(3).to_string(index=False))

    # S_reason = Pr[R(clean) > R(reasoning-corrupt)] ; S_outcome pooled over the two outcome-changing pairs
    # whose reasoning is held ~fixed (clean vs wrong-final, clean vs gold-distractor).
    def s_value(cats, mech):
        sub = df[df["category"].isin(cats)]
        if mech == "verifier":
            return float((sub["verifier"] == "better").mean()), float((sub["verifier"] == "tie").mean()), float((sub["verifier"] == "wrong").mean())
        return float(sub["judge_better_frac"].mean()), float(sub["judge_tie_frac"].mean()), float(sub["judge_wrong_frac"].mean())

    out = {"both_orders": both, "n_problems": len(groups), "table": table_rows}
    for name, cats in [("S_reason", ["corrupt_reasoning"]), ("S_outcome", ["wrong_final_good_reasoning", "gold_distractor_wrong_final"])]:
        for mech in ("verifier", "judge"):
            b, t, w = s_value(cats, mech)
            out[f"{name}_{mech}"] = {"better": b, "tie": t, "wrong": w}
    save_json(outdir / "perturbation_scores.json", out)
    print({k: v for k, v in out.items() if k.startswith("S_")})


if __name__ == "__main__":
    main()
