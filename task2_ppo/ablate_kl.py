from __future__ import annotations

import argparse
import gc

import torch

from common.data import load_yaml, repo_path
from task2_ppo.continue_train import run_ppo
from task2_ppo.evaluate import evaluate_adapter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("KL beta conditions:", cfg["kl_values"])
    print("Fork update budget:", cfg["fork_updates"])

    for beta in cfg["kl_values"]:
        tag = f"kl_{float(beta):.2f}"
        out = f"{cfg['fork_output_dir']}/{tag}"
        # The (eps=0.20, beta_KL=0.10) condition is identical to the eps=0.20 clipping fork: reuse it
        # if it already exists (same checkpoint, seed, prompts) instead of retraining.
        reuse = abs(float(beta) - float(cfg["kl_beta"])) < 1e-12 and repo_path(f"{cfg['fork_output_dir']}/clip_{float(cfg['clip_epsilon']):.2f}").exists()
        if reuse:
            print(f"[{tag}] identical to clip_{float(cfg['clip_epsilon']):.2f}; reuse its results (copying eval under this name)")
            import shutil

            res = repo_path(cfg["results_dir"])
            for suffix in ("_log.jsonl", "_train_summary.json"):
                src = res / f"clip_{float(cfg['clip_epsilon']):.2f}{suffix}"
                if src.exists():
                    shutil.copy(src, res / f"{tag}{suffix}")
            evaluate_adapter(args.config, f"{cfg['fork_output_dir']}/clip_{float(cfg['clip_epsilon']):.2f}", f"{tag}_heldout")
            continue
        if not args.eval_only:
            # Same checkpoint, prompts, seed, epsilon, budget; ONLY beta_KL changes.
            run_ppo(args.config, output=out, updates=int(cfg["fork_updates"]), clip_epsilon=float(cfg["clip_epsilon"]),
                    kl_beta=float(beta), run_name=tag)
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        evaluate_adapter(args.config, out, f"{tag}_heldout")


if __name__ == "__main__":
    main()
