from __future__ import annotations

import argparse

import pandas as pd

from common.data import load_yaml, repo_path
from common.logging_utils import load_json, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    d = repo_path(cfg["results_dir"]) / "task5_feedback"
    gsm, tr, pert = d / "gsm_summary.json", d / "transfer_summary.json", d / "perturbation_scores.json"
    out = {}

    rows = []
    if gsm.exists() and tr.exists():
        G, T = load_json(gsm), load_json(tr)
        for name in G["policies"]:
            g, t = G["policies"][name], T["policies"][name]
            rows.append(
                {
                    "policy": name,
                    "gsm_acc": g["exact_accuracy"], "svamp_acc": t["exact_accuracy"],
                    "acc_drop_abs": g["exact_accuracy"] - t["exact_accuracy"],
                    "acc_drop_rel": (g["exact_accuracy"] - t["exact_accuracy"]) / max(g["exact_accuracy"], 1e-9),
                    "gsm_format": g["format_compliance"], "svamp_format": t["format_compliance"],
                    "gsm_len": g["length_tokens"]["mean"], "svamp_len": t["length_tokens"]["mean"],
                    "svamp_len_over_gsm_len": t["length_tokens"]["mean"] / max(g["length_tokens"]["mean"], 1e-9),
                    "gsm_failures": g["failure_types"], "svamp_failures": t["failure_types"],
                }
            )
        df = pd.DataFrame(rows)
        df.to_csv(d / "in_vs_out_of_domain.csv", index=False)
        print(df.drop(columns=["gsm_failures", "svamp_failures"]).round(3).to_string(index=False))
        out["in_vs_out_of_domain"] = rows

        pw = []
        for ds, S in (("gsm", G), ("svamp", T)):
            for k, v in S.get("pairwise", {}).items():
                pw.append({"dataset": ds, "pair": k, **{kk: vv for kk, vv in v.items()}})
        if pw:
            pdf = pd.DataFrame(pw)
            pdf.to_csv(d / "pairwise_results.csv", index=False)
            print(pdf.round(3).to_string(index=False))
            out["pairwise"] = pw
    else:
        print("[info] run evaluate_math for both datasets first")

    if pert.exists():
        P = load_json(pert)
        out["diagnostics"] = {k: v for k, v in P.items() if k.startswith("S_")}
        print(pd.DataFrame(P["table"]).round(3).to_string(index=False))
        print({k: v for k, v in P.items() if k.startswith("S_")})

    save_json(d / "feedback_comparison.json", out)
    print("saved", d / "feedback_comparison.json")


if __name__ == "__main__":
    main()
