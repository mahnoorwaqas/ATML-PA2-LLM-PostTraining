from __future__ import annotations

import argparse
import gc

import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import append_jsonl, save_json, set_seed
from common.metrics import sample_entropy
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode, trainable_parameters
from common.policy_eval import row_id
from common.train_utils import ResourceTracker, Stepper, detach_generation, disable_dropout, make_trainables_fp32, token_entropy
from task3_grpo.grpo import (
    group_relative_advantages,
    group_reward_stats,
    grpo_policy_loss,
    mask_truncated_sequences,
)


def prepare_grpo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["grpo_midpoint_policy"],
        trainable=True,
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])
    optimizer = AdamW(trainable_parameters(policy), lr=float(cfg["learning_rate"]))
    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "optimizer": optimizer,
    }


def per_completion_grad_norms(policy, seq, attn, pw, rids, old_lp, ref_lp, adv, tmask, eps, loss_type, max_len, params):
    """Gradient norm contributed by each completion's POLICY term (beta=0), scaled exactly as in the
    full batch loss (1/N for the mean over sequences). Used as the length-conditioned statistic that
    directly tests the effect of sequence normalization (1/T_k vs 1/L_max)."""
    n = seq.shape[0]
    norms = []
    for i in range(n):
        new_lp, _ = response_token_logprobs(policy, seq[i : i + 1], attn[i : i + 1], pw, rids[i : i + 1])
        loss_i, _ = grpo_policy_loss(
            new_lp, old_lp[i : i + 1], adv[i : i + 1], tmask[i : i + 1], ref_lp[i : i + 1],
            eps, 0.0, loss_type=loss_type, max_completion_length=max_len,
        )
        grads = torch.autograd.grad(loss_i / n, params, allow_unused=True)
        sq = sum(float((g.float() ** 2).sum().item()) for g in grads if g is not None)
        norms.append(sq**0.5)
        del new_lp, loss_i, grads
    return norms


def run_grpo(config_path: str, output: str | None = None, updates: int | None = None, loss_type: str = "grpo", run_name: str = "standard", grad_stats: bool = False):
    bundle = prepare_grpo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    out = repo_path(output or cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)

    tok, policy = bundle["tokenizer"], bundle["policy"]
    rm, rm_tok = bundle["reward_model"], bundle["reward_tokenizer"]
    rows_all = bundle["prompt_rows"]
    K = int(cfg["num_generations"])
    eps, beta = float(cfg["clip_epsilon"]), float(cfg["kl_beta"])
    max_len = int(cfg["max_completion_length"])
    ppu = int(cfg["prompts_per_update"])
    n_updates = int(cfg["updates"])
    tol = float(cfg.get("informative_std_tol", 1e-6))
    gen_cfg = cfg.get("generation", {})
    make_trainables_fp32(policy)
    params = trainable_parameters(policy)
    stepper = Stepper(bundle["optimizer"], params, float(cfg["max_grad_norm"]))
    tracker = ResourceTracker()

    results_dir = repo_path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / f"{run_name}_log.jsonl"
    if log_path.exists():
        log_path.unlink()
    print(f"[grpo:{run_name}] loss_type={loss_type} updates={n_updates} K={K} eps={eps} beta={beta}")

    for u in range(n_updates):
        idxs = [(u * ppu + j) % len(rows_all) for j in range(ppu)]
        rows = [rows_all[i] for i in idxs]
        ids = [row_id(r, i) for r, i in zip(rows, idxs)]
        prompts, group_ids = [], []
        for g, r in enumerate(rows):
            m = prompt_messages(r)
            prompts.extend([m] * K)
            group_ids.extend([g] * K)
        group_ids = torch.tensor(group_ids)

        gen = batch_generate(
            policy, tok, prompts,
            max_prompt_length=int(cfg["max_prompt_length"]),
            max_new_tokens=max_len,
            temperature=float(gen_cfg.get("temperature", 0.7)),
            top_p=float(gen_cfg.get("top_p", 0.9)),
            do_sample=bool(gen_cfg.get("do_sample", True)),
        )
        gen = detach_generation(gen)  # generate() ran in inference_mode; autograd needs normal tensors
        seq, attn, pw = gen["sequences"], gen["attention_mask"], gen["prompt_width"]
        rids, rmask = gen["response_ids"], gen["response_mask"]
        policy.train()
        disable_dropout(policy)

        with torch.no_grad():
            old_lp, logits = response_token_logprobs(policy, seq, attn, pw, rids)
            ent = float(token_entropy(logits, rmask).item())
            del logits
            with reference_mode(policy):
                ref_lp, _ = response_token_logprobs(policy, seq, attn, pw, rids)
            policy.train()
            disable_dropout(policy)
            rewards = score_reward_pairs(
                rm, rm_tok, prompts, gen["responses"], max_length=int(cfg.get("reward_max_length", 1280))
            ).to(old_lp.device).float()

        # Max-length completions are removed from the loss (but their rewards still enter the group statistics).
        tmask = mask_truncated_sequences(rmask, gen["truncated"]) if bool(cfg.get("mask_truncated_completions", True)) else rmask
        adv = group_relative_advantages(rewards, group_ids.to(rewards.device))
        stds, informative = group_reward_stats(rewards.cpu(), group_ids, tol)

        lengths = rmask.sum(-1).float()
        comp_gn = None
        if grad_stats and tmask.sum() > 0:
            comp_gn = per_completion_grad_norms(
                policy, seq, attn, pw, rids, old_lp, ref_lp, adv, tmask, eps, loss_type, max_len, params
            )

        new_lp, _ = response_token_logprobs(policy, seq, attn, pw, rids)
        loss, diag = grpo_policy_loss(new_lp, old_lp, adv, tmask, ref_lp, eps, beta, loss_type=loss_type, max_completion_length=max_len)
        stepper.backward(loss)
        gn = stepper.step()

        abs_adv = adv.abs()
        rec = {
            "update": u + 1,
            "loss_type": loss_type,
            "prompt_ids": ids,
            "reward_mean": float(rewards.mean().item()),
            "group_reward_std": float(sum(stds) / len(stds)),
            "uninformative_group_fraction": float(1.0 - sum(informative) / len(informative)),
            "policy_loss": float(loss.item()),
            "policy_term": float(diag["policy_term"].item()),
            "kl": float(diag["sampled_kl"].item()),
            "clip_fraction": float(diag["clip_fraction"].item()),
            "entropy": ent,
            "sample_entropy": float(sample_entropy(old_lp, tmask).item()),
            "grad_norm": gn,
            "response_length": float(lengths.mean().item()),
            "truncated_fraction": float(sum(gen["truncated"]) / len(gen["truncated"])),
            "n_masked_completions": int(sum(gen["truncated"])),
            "completions": [
                {
                    "length": float(lengths[i].item()),
                    "reward": float(rewards[i].item()),
                    "advantage": float(adv[i].item()),
                    "truncated": bool(gen["truncated"][i]),
                    # effective per-token loss weight (excluding the 1/N sequence mean): 1/T_k vs 1/L_max
                    "token_weight": float(abs_adv[i].item() / (max(lengths[i].item(), 1.0) if loss_type == "grpo" else max_len)),
                    "grad_norm_policy_term": (comp_gn[i] if comp_gn is not None else None),
                }
                for i in range(seq.shape[0])
            ],
            "elapsed_s": tracker.elapsed(),
        }
        if torch.cuda.is_available():
            rec["peak_vram_gb_so_far"] = torch.cuda.max_memory_allocated() / 1024**3
        append_jsonl(log_path, rec)
        print(
            f"[grpo:{run_name}] u{u+1:02d} R={rec['reward_mean']:+.3f} std={rec['group_reward_std']:.3f} "
            f"unin={rec['uninformative_group_fraction']:.2f} KL={rec['kl']:.4f} len={rec['response_length']:.0f} gn={gn:.2f}"
        )
        del gen, old_lp, ref_lp, new_lp, adv, rewards
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    policy.save_pretrained(str(out))
    summary = {
        "run_name": run_name,
        "loss_type": loss_type,
        "updates": n_updates,
        "K": K,
        "clip_epsilon": eps,
        "kl_beta": beta,
        "seed": int(cfg["seed"]),
        "adapter_path": str(out),
        **tracker.summary(),
    }
    save_json(results_dir / f"{run_name}_train_summary.json", summary)
    print("saved adapter ->", out)
    del policy, rm, bundle
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--loss-type", choices=["grpo", "dr_grpo"], default="grpo")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--grad-stats", action="store_true", help="also log per-completion gradient norms (K extra backward passes/update)")
    args = ap.parse_args()
    run_grpo(args.config, args.output, args.updates, args.loss_type, args.run_name, args.grad_stats)


if __name__ == "__main__":
    main()
