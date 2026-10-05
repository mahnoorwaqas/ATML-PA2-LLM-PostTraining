from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json

CACHE_K = 8


def load_k8_cache(path):
    rows = read_jsonl(path)
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[str(row["source_index"])].append(row)
    # Instructor cache has 8 rows per prompt, one row per completion.
    bad = {pid: len(group) for pid, group in by_prompt.items() if len(group) < 8}
    if bad:
        raise ValueError(f"Expected at least K=8 cached completions per prompt; short groups: {bad}")
    for group in by_prompt.values():
        group.sort(key=lambda x: int(x.get("generation_index", 0)))
    return by_prompt


def regroup_equal_generation_budget(by_prompt, k: int):
    """Return K-sized groups while keeping total cached completions fixed.

    Rule (documented in the report): use exactly the first 8 completions (by generation_index) of every
    prompt, and split them into consecutive blocks of k (k=8 -> 1 group/prompt, k=4 -> 2, k=2 -> 4).
    The SAME completions and rewards are used for every k, so total generations (8 x #prompts) is
    identical; only the partition into comparison groups changes. Returns a list of
    {"prompt_id", "block", "rewards"}.
    """
    if CACHE_K % k != 0:
        raise ValueError(f"k={k} must divide {CACHE_K}")
    groups = []
    for pid, comps in by_prompt.items():
        comps = comps[:CACHE_K]
        for b in range(CACHE_K // k):
            block = comps[b * k : (b + 1) * k]
            groups.append({"prompt_id": pid, "block": b, "rewards": np.array([float(c["reward"]) for c in block])})
    return groups


def difficulty_bins(by_prompt, n_bins: int = 3):
    """Binning rule (defined ONCE, independent of k): rank prompts by the mean cached reward over their
    8 completions and cut into equal-sized terciles: hard (lowest mean reward), medium, easy."""
    pids = list(by_prompt)
    means = np.array([np.mean([float(c["reward"]) for c in by_prompt[p][:CACHE_K]]) for p in pids])
    order = np.argsort(means, kind="stable")
    names = ["hard", "medium", "easy"] if n_bins == 3 else [f"bin{i}" for i in range(n_bins)]
    bins = {}
    for rank, idx in enumerate(order):
        bins[pids[idx]] = names[min(rank * n_bins // len(pids), n_bins - 1)]
    return bins


def group_metrics(groups, tol: float, eps: float = 1e-6) -> dict:
    stds = np.array([g["rewards"].std() for g in groups])  # population std
    informative = stds > tol
    advs, centered = [], []
    for g in groups:
        r = g["rewards"]
        centered.append(r - r.mean())
        advs.append((r - r.mean()) / (r.std() + eps))
    advs, centered = np.concatenate(advs), np.concatenate(centered)
    return {
        "n_groups": len(groups),
        "informative_rate": float(informative.mean()),
        "uninformative_rate": float(1 - informative.mean()),
        "mean_group_reward_std": float(stds.mean()),
        "mean_group_reward_std_informative_only": float(stds[informative].mean()) if informative.any() else 0.0,
        # "Variance of the group-relative signal": (i) variance of the group-centered rewards r_k - mu_g
        # (unnormalised signal), (ii) variance of the normalised advantages A_k (=~ informative rate).
        "var_centered_reward": float(centered.var()),
        "var_normalised_advantage": float(advs.var()),
        "mean_abs_advantage": float(np.abs(advs).mean()),
    }


def bootstrap_ci(groups, tol, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    flags = np.array([g["rewards"].std() > tol for g in groups], dtype=float)
    pids = np.array([g["prompt_id"] for g in groups])
    uniq = np.unique(pids)
    per_prompt = {p: flags[pids == p] for p in uniq}
    vals = []
    for _ in range(n_boot):  # resample PROMPTS (groups from one prompt are not independent)
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        vals.append(np.mean(np.concatenate([per_prompt[p] for p in samp])))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--tol", type=float, default=None)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    tol = float(args.tol if args.tol is not None else cfg.get("informative_std_tol", 1e-6))
    by_prompt = load_k8_cache(cfg["group_cache"])
    print("Cached prompts:", len(by_prompt))
    print("Group sizes to analyze:", cfg["group_sizes"])
    first = next(iter(by_prompt.values()))
    print("Cache row keys:", sorted(first[0].keys()))

    all_r = np.array([float(c["reward"]) for g in by_prompt.values() for c in g[:CACHE_K]])
    print(f"reward range [{all_r.min():.3f}, {all_r.max():.3f}], distinct values: {len(np.unique(all_r))}")
    bins = difficulty_bins(by_prompt)

    out = {"tol": tol, "n_prompts": len(by_prompt), "total_generations": len(by_prompt) * CACHE_K,
           "reward_distinct_values": int(len(np.unique(all_r))), "by_k": {}}
    table = []
    for k in cfg["group_sizes"]:
        groups = regroup_equal_generation_budget(by_prompt, int(k))
        res = {"overall": group_metrics(groups, tol)}
        res["overall"]["informative_rate_ci95"] = bootstrap_ci(groups, tol)
        for b in ("hard", "medium", "easy"):
            sub = [g for g in groups if bins[g["prompt_id"]] == b]
            if sub:
                res[b] = group_metrics(sub, tol)
        out["by_k"][str(k)] = res
        for name, m in res.items():
            table.append({"K": k, "bin": name, **{kk: vv for kk, vv in m.items() if not isinstance(vv, list)}})

    df = pd.DataFrame(table)
    d = repo_path(cfg["results_dir"])
    d.mkdir(parents=True, exist_ok=True)
    df.to_csv(d / "group_size_study.csv", index=False)
    save_json(d / "group_size_study.json", out)
    print(df.to_string(index=False))
    print("saved", d / "group_size_study.json")


if __name__ == "__main__":
    main()
