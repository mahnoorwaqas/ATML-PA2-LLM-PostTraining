from __future__ import annotations

import argparse
import gc

import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import append_jsonl, save_json, set_seed
from common.metrics import sampled_kl, sample_entropy
from common.models import (
    load_policy,
    load_reward_model,
    load_tokenizer,
    load_value_model,
    reference_mode,
    token_values,
    trainable_parameters,
    value_parameter_groups,
)
from common.policy_eval import row_id
from common.train_utils import ResourceTracker, Stepper, detach_generation, disable_dropout, make_trainables_fp32, token_entropy
from task2_ppo.ppo import (
    compute_gae,
    normalize_advantages,
    ppo_affected_fraction,
    ppo_policy_loss,
    shaped_rewards,
    value_mse_loss,
)


def prepare_ppo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["ppo_midpoint_policy"],
        trainable=True,
    )
    value_model = load_value_model(
        cfg,
        cfg["paths"]["ppo_midpoint_value"],
        train_mode=cfg.get("value_train_mode", "head_only"),
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])

    policy_optimizer = AdamW(
        trainable_parameters(policy),
        lr=float(cfg["policy_learning_rate"]),
    )
    value_optimizer = AdamW(
        value_parameter_groups(
            value_model,
            lora_lr=float(cfg["value_lora_learning_rate"]),
            head_lr=float(cfg["value_head_learning_rate"]),
        ),
        weight_decay=0.0,
    )

    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "value_model": value_model,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "policy_optimizer": policy_optimizer,
        "value_optimizer": value_optimizer,
    }


def values_on_response(value_model, seq, attn, prompt_width, T):
    """V(s_t) for response step t is read at sequence position prompt_width-1+t (state BEFORE emitting token t)."""
    v = token_values(value_model, seq, attn).float()
    return v[:, prompt_width - 1 : prompt_width - 1 + T]


def critic_diagnostics(values, returns, mask) -> dict:
    """How well does the critic predict the GAE return on this batch?"""
    m = mask.bool()
    v, r = values[m], returns[m]
    if v.numel() < 2:
        return {"value_mean": float("nan"), "return_mean": float("nan"), "value_mae": float("nan"), "explained_variance": float("nan")}
    var_r = r.var(unbiased=False)
    ev = float((1.0 - (r - v).var(unbiased=False) / var_r.clamp_min(1e-8)).item())
    return {
        "value_mean": float(v.mean().item()),
        "return_mean": float(r.mean().item()),
        "value_mae": float((v - r).abs().mean().item()),
        "explained_variance": ev,
    }


def run_ppo(config_path: str, output: str | None = None, updates: int | None = None, clip_epsilon: float | None = None, kl_beta: float | None = None, run_name: str = "standard"):
    bundle = prepare_ppo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    if clip_epsilon is not None:
        cfg["clip_epsilon"] = float(clip_epsilon)
    if kl_beta is not None:
        cfg["kl_beta"] = float(kl_beta)
    out = repo_path(output or cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)

    tok, policy, vmodel = bundle["tokenizer"], bundle["policy"], bundle["value_model"]
    # fp16 trainable params (e.g. the critic's scalar head) break GradScaler/AdamW; keep them in fp32.
    make_trainables_fp32(vmodel)
    make_trainables_fp32(policy)
    rm, rm_tok = bundle["reward_model"], bundle["reward_tokenizer"]
    rows_all = bundle["prompt_rows"]
    eps, beta_kl = float(cfg["clip_epsilon"]), float(cfg["kl_beta"])
    gamma, lam = float(cfg["gamma"]), float(cfg["gae_lambda"])
    gen_cfg = cfg.get("generation", {})
    ppu = int(cfg["prompts_per_update"])
    n_updates = int(cfg["updates"])

    results_dir = repo_path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / f"{run_name}_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    pol_step = Stepper(bundle["policy_optimizer"], trainable_parameters(policy), float(cfg["max_grad_norm"]))
    val_step = Stepper(bundle["value_optimizer"], trainable_parameters(vmodel), float(cfg["max_grad_norm"]))
    tracker = ResourceTracker()
    print(f"[ppo:{run_name}] updates={n_updates} eps={eps} beta_kl={beta_kl} prompts/update={ppu}")

    for u in range(n_updates):
        # ---- fixed, run-independent prompt schedule (identical across every fork) ----
        idxs = [(u * ppu + j) % len(rows_all) for j in range(ppu)]
        rows = [rows_all[i] for i in idxs]
        prompts = [prompt_messages(r) for r in rows]
        ids = [row_id(r, i) for r, i in zip(rows, idxs)]

        # ---- 1. on-policy rollout ----
        gen = batch_generate(
            policy, tok, prompts,
            max_prompt_length=int(cfg["max_prompt_length"]),
            max_new_tokens=int(cfg["max_response_length"]),
            temperature=float(gen_cfg.get("temperature", 0.7)),
            top_p=float(gen_cfg.get("top_p", 0.9)),
            do_sample=bool(gen_cfg.get("do_sample", True)),
        )
        gen = detach_generation(gen)  # generate() ran in inference_mode; autograd needs normal tensors
        seq, attn, pw = gen["sequences"], gen["attention_mask"], gen["prompt_width"]
        rids, rmask = gen["response_ids"], gen["response_mask"]
        T = rids.shape[1]
        policy.train()
        disable_dropout(policy)
        disable_dropout(vmodel)

        # ---- 2. old / reference log-probs, rewards, values ----
        with torch.no_grad():
            old_lp, logits = response_token_logprobs(policy, seq, attn, pw, rids)
            ent = float(token_entropy(logits, rmask).item())
            del logits
            with reference_mode(policy):
                ref_lp, _ = response_token_logprobs(policy, seq, attn, pw, rids)
            policy.train()
            disable_dropout(policy)

            raw_reward = score_reward_pairs(rm, rm_tok, prompts, gen["responses"], max_length=int(cfg["reward_max_length"]))
            raw_reward = raw_reward.to(old_lp.device).float()
            missing = torch.tensor([0.0 if e else 1.0 for e in gen["terminated_with_eos"]], device=old_lp.device)
            task_reward = raw_reward - float(cfg["missing_eos_penalty"]) * missing

            old_values = values_on_response(vmodel, seq, attn, pw, T)

            rewards = shaped_rewards(task_reward, old_lp, ref_lp, rmask, beta_kl)
            adv, returns = compute_gae(rewards, old_values, rmask, gamma, lam)
            adv_n = normalize_advantages(adv, rmask)
            kl_est = float(sampled_kl(old_lp, ref_lp, rmask).item())
            samp_ent = float(sample_entropy(old_lp, rmask).item())
            crit = critic_diagnostics(old_values, returns, rmask)

        # ---- 3. PPO epochs on this batch ----
        pl, vl, gns, cfr, aff = [], [], [], [], []
        for _ in range(int(cfg["ppo_epochs"])):
            new_lp, _ = response_token_logprobs(policy, seq, attn, pw, rids)
            loss_p, ratio, clip_frac = ppo_policy_loss(new_lp, old_lp, adv_n, rmask, eps)
            aff.append(float(ppo_affected_fraction(new_lp.detach(), old_lp, adv_n, rmask, eps).item()))
            pol_step.backward(loss_p)
            gns.append(pol_step.step())
            pl.append(float(loss_p.item()))
            cfr.append(float(clip_frac.item()))

            new_v = values_on_response(vmodel, seq, attn, pw, T)
            loss_v = value_mse_loss(new_v, returns, rmask)
            val_step.backward(float(cfg["value_coef"]) * loss_v)
            val_step.step()
            vl.append(float(loss_v.item()))

        mean = lambda x: sum(x) / max(len(x), 1)
        rec = {
            "update": u + 1,
            "prompt_ids": ids,
            "reward_mean": float(raw_reward.mean().item()),
            "task_reward_mean": float(task_reward.mean().item()),
            "kl": kl_est,
            "policy_loss": mean(pl),
            "value_loss": mean(vl),
            "entropy": ent,
            "sample_entropy": samp_ent,
            "grad_norm": mean(gns),
            "clip_fraction": mean(cfr),
            "affected_fraction": mean(aff),
            "clip_fraction_per_epoch": cfr,
            "response_length": float(rmask.sum(-1).float().mean().item()),
            "truncated_fraction": float(sum(gen["truncated"]) / max(len(gen["truncated"]), 1)),
            "adv_abs_mean": float(adv_n[rmask.bool()].abs().mean().item()),
            "elapsed_s": tracker.elapsed(),
            **{f"critic_{k}": v for k, v in crit.items()},
        }
        if torch.cuda.is_available():
            rec["peak_vram_gb_so_far"] = torch.cuda.max_memory_allocated() / 1024**3
        append_jsonl(log_path, rec)
        print(
            f"[ppo:{run_name}] u{u+1:02d} R={rec['reward_mean']:+.3f} KL={kl_est:.4f} "
            f"len={rec['response_length']:.0f} clip={rec['clip_fraction']:.3f} gn={rec['grad_norm']:.2f} "
            f"vloss={rec['value_loss']:.3f}"
        )
        del gen, old_lp, ref_lp, rewards, adv, returns, adv_n, new_lp, new_v
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    policy.save_pretrained(str(out))
    summary = {
        "run_name": run_name,
        "updates": n_updates,
        "clip_epsilon": eps,
        "kl_beta": beta_kl,
        "seed": int(cfg["seed"]),
        "adapter_path": str(out),
        **tracker.summary(),
    }
    save_json(results_dir / f"{run_name}_train_summary.json", summary)
    print("saved adapter ->", out)
    del policy, vmodel, rm, bundle
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--clip-epsilon", type=float)
    ap.add_argument("--kl-beta", type=float)
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_ppo(args.config, args.output, args.updates, args.clip_epsilon, args.kl_beta, args.run_name)


if __name__ == "__main__":
    main()
