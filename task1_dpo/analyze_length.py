from __future__ import annotations

import argparse

import numpy as np

from common.data import load_yaml, preference_responses, read_jsonl, repo_path
from common.logging_utils import save_json
from common.metrics import safe_corr, word_count
from task1_dpo.evaluate import evaluate_adapter, get_stratum
from task1_dpo.train import run_training


def dataset_length_stats(rows: list[dict]) -> dict:
    """Properties of the PREFERENCE DATA (not of any learned policy)."""
    lc = np.array([word_count(preference_responses(r)[0]) for r in rows], dtype=float)
    lr = np.array([word_count(preference_responses(r)[1]) for r in rows], dtype=float)
    diff = lc - lr
    strata = [get_stratum(r) for r in rows]
    return {
        "n": len(rows),
        "frac_chosen_longer": float(np.mean(lc > lr)),
        "mean_chosen_words": float(lc.mean()),
        "mean_rejected_words": float(lr.mean()),
        "mean_word_diff": float(diff.mean()),
        "median_len_ratio_chosen_over_rejected": float(np.median(lc / np.maximum(lr, 1))),
        "corr_chosen_indicator_vs_len_diff": safe_corr((diff > 0).astype(float), np.abs(diff)),
        "strata_counts": {s: strata.count(s) for s in ("preferred_longer", "matched", "rejected_longer")},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--skip-train", action="store_true", help="reuse an existing length-balanced adapter")
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    standard_train = read_jsonl(cfg["paths"]["dpo_standard_train"])
    balanced = read_jsonl(cfg["paths"]["dpo_length_train"])
    stratified = read_jsonl(cfg["paths"]["dpo_length_eval"])
    print("Length-balanced train rows:", len(balanced))
    print("Length-stratified eval rows:", len(stratified))

    out_dir = repo_path(cfg["results_dir"])
    save_json(
        out_dir / "dataset_length_stats.json",
        {
            "standard_train": dataset_length_stats(standard_train),
            "length_balanced_train": dataset_length_stats(balanced),
            "length_stratified_eval": dataset_length_stats(stratified),
        },
    )

    if not args.skip_train:
        run_training(
            args.config, "length_balanced", dataset_path=cfg["paths"]["dpo_length_train"],
            output_path=cfg["length_output"],
        )
    if not args.skip_eval:
        # Both models: stratified pair accuracy + generated length + word-limit compliance (same prompts/decoding).
        evaluate_adapter(args.config, cfg["standard_output"], "standard", with_strata=True)
        evaluate_adapter(args.config, cfg["length_output"], "length_balanced", with_strata=True)


if __name__ == "__main__":
    main()
