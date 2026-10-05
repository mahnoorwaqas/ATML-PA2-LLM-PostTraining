"""Task 2 report tables: standard-continuation trajectories + fork comparison + stability statistics.

Usage: python -m task2_ppo.summarize --config configs/ppo.yaml
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import load_json


def stability_stats(log: list[dict]) -> dict:
    r = np.array([x["reward_mean"] for x in log])
    kl = np.array([x["kl"] for x in log])
    gn = np.array([x["grad_norm"] for x in log])
    return {
        # Stability statistics (all defined once, same for every fork):
        "reward_diff_std": float(np.std(np.diff(r))) if len(r) > 2 else float("nan"),  # roughness of reward trajectory
        "kl_max": float(kl.max()),
        "kl_final": float(kl[-1]),
        "grad_norm_max": float(gn.max()),
        "clip_fraction_mean": float(np.mean([x["clip_fraction"] for x in log])),
        "affected_fraction_mean": float(np.mean([x["affected_fraction"] for x in log])),
        "value_loss_final": float(log[-1]["value_loss"]),
        "reward_first3": float(r[:3].mean()),
        "reward_last3": float(r[-3:].mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    d = repo_path(cfg["results_dir"])

    rows = []
    for tag in ["standard"] + [f"clip_{float(e):.2f}" for e in cfg["clip_values"]] + [f"kl_{float(k):.2f}" for k in cfg["kl_values"]]:
        lp, ep = d / f"{tag}_log.jsonl", d / f"{tag}_heldout_eval.json"
        row = {"run": tag}
        if lp.exists():
            row.update(stability_stats(read_jsonl(lp)))
        if ep.exists():
            e = load_json(ep)
            row.update(
                {
                    "heldout_rm": e["rm_score_mean"],
                    "heldout_kl": e["kl_token_weighted"],
                    "heldout_entropy": e["entropy_mean"],
                    "heldout_len_mean": e["length"]["mean"],
                    "heldout_len_std": e["length"]["std"],
                    "heldout_trunc": e["truncation_rate"],
                }
            )
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(d / "summary.csv", index=False)
    (d / "summary.md").write_text(df.to_markdown(index=False, floatfmt=".3f"), encoding="utf-8")
    print(df.to_string(index=False))

    ts = d / "standard_train_summary.json"
    if ts.exists():
        print("\nStandard continuation resources:", load_json(ts))


if __name__ == "__main__":
    main()
