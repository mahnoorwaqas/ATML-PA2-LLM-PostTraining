from __future__ import annotations

import argparse

from common.data import load_yaml, read_jsonl
from common.models import load_policy, load_reward_model, load_tokenizer
from common.policy_eval import evaluate_policy_generation


def load_evaluation_bundle(config_path: str, adapter: str | None):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["rl_prompt_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


def evaluate_adapter(config_path: str, adapter: str | None, name: str, greedy: bool = False):
    """Frozen held-out protocol: same prompts, decoding, 768-token cap, seed for every PPO condition."""
    cfg = load_yaml(config_path)
    rows = read_jsonl(cfg["paths"]["rl_prompt_eval"])
    return evaluate_policy_generation(
        cfg, adapter, name, rows, cfg["results_dir"],
        max_new_tokens=int(cfg["eval_max_response_length"]),
        n_prompts=int(cfg.get("eval_prompts", 64)),
        batch_size=int(cfg.get("eval_batch_size", 4)),
        greedy=greedy,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--adapter", required=True, help="adapter dir, or 'none' for the base policy")
    ap.add_argument("--name", default="standard")
    ap.add_argument("--greedy", action="store_true")
    args = ap.parse_args()
    adapter = None if args.adapter.lower() == "none" else args.adapter
    s = evaluate_adapter(args.config, adapter, f"{args.name}_heldout", args.greedy)
    print({k: s[k] for k in ("rm_score_mean", "kl_token_weighted", "entropy_mean", "truncation_rate")}, s["length"])


if __name__ == "__main__":
    main()
