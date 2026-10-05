from __future__ import annotations

import argparse
import gc

import numpy as np
import torch

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import load_json, save_json
from common.metrics import safe_corr
from task3_grpo.continue_train import run_grpo
from task3_grpo.evaluate import evaluate_adapter

CONDITIONS = {"grpo_canonical": "grpo", "dr_grpo": "dr_grpo"}


def length_conditioned_stats(log: list[dict]) -> dict:
    """Statistics that directly test the normalization effect.

    Pools every completion across the fork's updates and splits at the pooled median length:
      * token_weight: effective per-token loss weight |A|/T (canonical) vs |A|/L_max (Dr. GRPO)
      * grad_norm_policy_term: per-completion gradient norm (needs --grad-stats)
      * corr(length, token_weight) and corr(length, grad_norm)
      * short/long gradient-share = share of the total per-completion gradient norm coming from
        completions shorter / longer than the median.
    """
    comps = [c for rec in log for c in rec["completions"]]
    L = np.array([c["length"] for c in comps])
    w = np.array([c["token_weight"] for c in comps])
    A = np.array([abs(c["advantage"]) for c in comps])
    med = float(np.median(L))
    short, long_ = L <= med, L > med
    out = {
        "n_completions": len(comps),
        "median_length": med,
        "mean_length_short": float(L[short].mean()) if short.any() else float("nan"),
        "mean_length_long": float(L[long_].mean()) if long_.any() else float("nan"),
        "corr_length_vs_token_weight": safe_corr(L, w),
        "mean_token_weight_short": float(w[short].mean()) if short.any() else float("nan"),
        "mean_token_weight_long": float(w[long_].mean()) if long_.any() else float("nan"),
        # total gradient "mass" a completion receives from the advantage-weighted token terms: |A| * T * weight
        "token_mass_ratio_long_over_short": float((A * L * w)[long_].sum() / max((A * L * w)[short].sum(), 1e-12)) if short.any() and long_.any() else float("nan"),
    }
    gn = np.array([c["grad_norm_policy_term"] if c["grad_norm_policy_term"] is not None else np.nan for c in comps])
    if not np.isnan(gn).all():
        ok = ~np.isnan(gn)
        out["corr_length_vs_grad_norm"] = safe_corr(L[ok], gn[ok])
        out["mean_grad_norm_short"] = float(np.nanmean(gn[short & ok])) if (short & ok).any() else float("nan")
        out["mean_grad_norm_long"] = float(np.nanmean(gn[long_ & ok])) if (long_ & ok).any() else float("nan")
        tot = np.nansum(gn)
        out["grad_share_short"] = float(np.nansum(gn[short & ok]) / tot) if tot > 0 else float("nan")
        out["grad_share_long"] = float(np.nansum(gn[long_ & ok]) / tot) if tot > 0 else float("nan")
    return out


def trajectory_stats(log: list[dict]) -> dict:
    g = lambda k: np.array([r[k] for r in log], dtype=float)
    return {
        "reward_first3": float(g("reward_mean")[:3].mean()),
        "reward_last3": float(g("reward_mean")[-3:].mean()),
        "kl_final": float(g("kl")[-1]),
        "kl_max": float(g("kl").max()),
        "length_first3": float(g("response_length")[:3].mean()),
        "length_last3": float(g("response_length")[-3:].mean()),
        "length_slope_per_update": float(np.polyfit(np.arange(len(log)), g("response_length"), 1)[0]) if len(log) > 2 else float("nan"),
        "grad_norm_mean": float(g("grad_norm").mean()),
        "uninformative_fraction": float(g("uninformative_group_fraction").mean()),
        "group_reward_std_mean": float(g("group_reward_std").mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--analyze-only", action="store_true", help="skip training/eval, only re-analyze saved logs")
    ap.add_argument("--no-grad-stats", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("Fork updates:", cfg["fork_updates"])
    print("Compare loss_type='grpo' vs loss_type='dr_grpo' from the identical supplied midpoint.")
    d = repo_path(cfg["results_dir"])

    if not args.analyze_only:
        for tag, loss_type in CONDITIONS.items():
            out = f"{cfg['fork_output_dir']}/{tag}"
            # Same midpoint, prompts, seed, reward, beta, epsilon, K, caps, generated-token budget; ONLY loss_type changes.
            run_grpo(args.config, output=out, updates=int(cfg["fork_updates"]), loss_type=loss_type,
                     run_name=tag, grad_stats=(bool(cfg.get("grad_stats_in_forks", True)) and not args.no_grad_stats))
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            evaluate_adapter(args.config, out, f"{tag}_heldout")

    summary = {}
    for tag in CONDITIONS:
        lp = d / f"{tag}_log.jsonl"
        if not lp.exists():
            continue
        log = read_jsonl(lp)
        entry = {"trajectory": trajectory_stats(log), "length_conditioned": length_conditioned_stats(log)}
        ep = d / f"{tag}_heldout_eval.json"
        if ep.exists():
            e = load_json(ep)
            entry["heldout"] = {
                "rm_score_mean": e["rm_score_mean"],
                "kl_token_weighted": e["kl_token_weighted"],
                "entropy_mean": e["entropy_mean"],
                "length_mean": e["length"]["mean"],
                "length_std": e["length"]["std"],
                "truncation_rate": e["truncation_rate"],
            }
        summary[tag] = entry
    save_json(d / "normalization_comparison.json", summary)
    print(__import__("json").dumps(summary, indent=2))


if __name__ == "__main__":
    main()
