"""Collect Task 1 results into report-ready tables and qualitative candidates.

Usage: python -m task1_dpo.summarize --config configs/dpo.yaml
Reads results/task1_dpo/*_eval.json written by evaluate.py and writes
results/task1_dpo/summary.csv, summary.md, qualitative_candidates.json.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from common.data import load_yaml, repo_path
from common.logging_utils import load_json, save_json

CONDITIONS = [
    ("sft_base", "SFT base (no adapter)"),
    ("standard", "DPO standard (1 epoch, beta=0.10)"),
    ("beta_0.03", "short fork beta=0.03"),
    ("beta_0.10", "short fork beta=0.10"),
    ("beta_0.30", "short fork beta=0.30"),
    ("length_balanced", "DPO length-balanced"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    d = repo_path(cfg["results_dir"])
    rows = []
    for key, label in CONDITIONS:
        p = d / f"{key}_eval.json"
        if not p.exists():
            continue
        r = load_json(p)
        g, h = r.get("generation", {}), r.get("heldout") or {}
        rows.append(
            {
                "condition": label,
                "budget": "full epoch" if key in {"standard", "length_balanced"} else ("short (%d ex.)" % cfg["short_ablation_examples"] if key.startswith("beta") else "-"),
                "beta": r["beta_for_loss"] if key != "sft_base" else None,
                "dpo_loss": h.get("dpo_loss"),
                "pref_acc": h.get("pref_accuracy"),
                "kl_token_weighted": g.get("kl_token_weighted"),
                "rm_score": g.get("rm_score_mean"),
                "len_mean": g.get("length", {}).get("mean"),
                "len_std": g.get("length", {}).get("std"),
                "wordlimit_compliance": r.get("word_limit", {}).get("compliance_rate"),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(d / "summary.csv", index=False)
    (d / "summary.md").write_text(df.to_markdown(index=False, floatfmt=".3f") if len(df) else "", encoding="utf-8")
    print(df.to_string(index=False))

    strata = {}
    for key in ("standard", "length_balanced"):
        p = d / f"{key}_eval.json"
        if p.exists():
            ls = load_json(p).get("length_stratified")
            if ls:
                strata[key] = {k: v["pref_accuracy"] for k, v in ls.items() if isinstance(v, dict)}
    if strata:
        sdf = pd.DataFrame(strata).T
        sdf.to_csv(d / "length_strata_accuracy.csv")
        print("\nStratified preference accuracy:\n", sdf.to_string())

    # Qualitative candidates: high RM score gain over base but much longer / non-compliant responses.
    base_p, std_p = d / "sft_base_generations.jsonl", d / "standard_generations.jsonl"
    if base_p.exists() and std_p.exists():
        base = [json.loads(l) for l in base_p.open(encoding="utf-8")]
        std = [json.loads(l) for l in std_p.open(encoding="utf-8")]
        cands = []
        for b, s in zip(base, std):
            cands.append(
                {
                    "prompt_id": s["prompt_id"],
                    "prompt": s["prompt"],
                    "rm_gain": s["rm_score"] - b["rm_score"],
                    "len_gain_tokens": s["n_tokens"] - b["n_tokens"],
                    "dpo_response": s["response"][:600],
                    "base_response": b["response"][:600],
                }
            )
        by_rm_len = sorted(cands, key=lambda x: (x["rm_gain"] > 0) * x["len_gain_tokens"], reverse=True)[:8]
        by_rm_gain = sorted(cands, key=lambda x: x["rm_gain"], reverse=True)[:8]
        save_json(d / "qualitative_candidates.json", {"rm_up_and_longer": by_rm_len, "largest_rm_gain": by_rm_gain})
        print("wrote qualitative candidates (YOU must read them and judge quality yourself)")


if __name__ == "__main__":
    main()
