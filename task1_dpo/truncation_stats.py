"""Report how many DPO examples are affected by the 768-token limit, per dataset and length stratum.

Usage: python -m task1_dpo.truncation_stats
Writes results/task1_dpo/truncation_stats.json -- cite these counts when interpreting the length experiments.
"""
from __future__ import annotations

import argparse

import numpy as np

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json
from common.models import load_tokenizer
from task1_dpo.evaluate import get_stratum
from task1_dpo.preprocess import fit_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    tok = load_tokenizer(cfg["base_model"])
    L = int(cfg["max_sequence_length"])
    out = {"max_length": L}
    for key in ("dpo_standard_train", "dpo_standard_eval", "dpo_length_train", "dpo_length_eval"):
        rows = read_jsonl(cfg["paths"][key])
        reps = [fit_report(tok, r, L) for r in rows]
        strata = np.array([get_stratum(r) for r in rows])
        d = {
            "n": len(rows),
            "prompt_too_long_filtered": int(sum(r["prompt_too_long"] for r in reps)),
            "chosen_truncated": int(sum(r["chosen_truncated"] for r in reps)),
            "rejected_truncated": int(sum(r["rejected_truncated"] for r in reps)),
            "any_truncated": int(sum(r["chosen_truncated"] or r["rejected_truncated"] for r in reps)),
            "by_stratum": {},
        }
        for s in ("preferred_longer", "matched", "rejected_longer"):
            idx = np.where(strata == s)[0]
            if len(idx):
                d["by_stratum"][s] = {
                    "n": int(len(idx)),
                    "chosen_truncated": int(sum(reps[i]["chosen_truncated"] for i in idx)),
                    "rejected_truncated": int(sum(reps[i]["rejected_truncated"] for i in idx)),
                    "prompt_too_long": int(sum(reps[i]["prompt_too_long"] for i in idx)),
                }
        out[key] = d
        print(key, d)
    save_json(repo_path(cfg["results_dir"]) / "truncation_stats.json", out)


if __name__ == "__main__":
    main()
