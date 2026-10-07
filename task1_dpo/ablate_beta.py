from __future__ import annotations

import argparse
import gc

import torch

from common.data import load_yaml, repo_path
from task1_dpo.evaluate import evaluate_adapter
from task1_dpo.train import run_training


def beta_tag(b: float) -> str:
    return f"beta_{b:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--train-only", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    n_ex = int(cfg["short_ablation_examples"])
    print("Required beta values:", cfg["betas"])
    print("Short-run examples per condition:", n_ex)

    for beta in cfg["betas"]:
        tag = beta_tag(beta)
        out = f"{cfg['beta_output_dir']}/{tag}"
        if not args.eval_only:
            # Same data subset (first n_ex rows), seed, optimizer, LoRA config; ONLY beta changes.
            run_training(args.config, tag, output_path=out, beta=float(beta), max_examples=n_ex)
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        if not args.train_only:
            evaluate_adapter(args.config, out, tag, beta=float(beta), with_word_limit=False)
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None


if __name__ == "__main__":
    main()
