from __future__ import annotations

import argparse
import gc

import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import response_token_logprobs, score_reward_pairs
from common.logging_utils import save_json, set_seed
from common.metrics import masked_mean
from common.models import (
    load_policy,
    load_reward_model,
    load_tokenizer,
    load_value_model,
    token_values,
    trainable_parameters,
)
from common.train_utils import Stepper, disable_dropout
from task2_ppo.ppo import (
    compute_gae,
    normalize_advantages,
    ppo_affected_fraction,
    ppo_policy_loss,
    shaped_rewards,
)


def load_cached_rollouts(path):
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")

    # Instructor iterations used two equivalent names for these fields. Normalize once here so
    # the student analysis code sees one stable interface.
    normalized = []
    for row in rows:
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalized.append(row)

    required = {"source_index", "response", "old_logprobs", "ref_logprobs"}
    if not required.issubset(normalized[0]):
        raise ValueError(f"Unexpected PPO cache schema; need at least {sorted(required)}")
    return normalized


def _as_list(x):
    if torch.is_tensor(x):
        return x.float().flatten().tolist()
    return [float(v) for v in x]


def prompt_lookup(cfg: dict):
    """Index the PPO prompt pool by position and by each identifier a row may carry.

    Each identifier type gets its own dict so that, e.g., a small-integer `id` column can never
    overwrite or shadow a `source_index` entry.
    """
    pool = read_jsonl(cfg["paths"]["rl_prompt_train"])
    table = {"pool": pool, "prompt_id": {}, "source_index": {}, "id": {}}
    for r in pool:
        for k in ("prompt_id", "source_index", "id"):
            if k in r:
                table[k].setdefault(str(r[k]), r)
    return table


def find_prompt_messages(table: dict, row: dict):
    msgs = row.get("messages") or row.get("prompt_messages")
    if msgs is not None:
        return msgs

    # 1) exact identifier match
    for k in ("prompt_id", "source_index"):
        if row.get(k) is not None and str(row[k]) in table[k]:
            return prompt_messages(table[k][str(row[k])])

    # 2) positional fallback: source_index is an index into the pool file
    pool = table["pool"]
    si = row.get("source_index")
    if si is not None and 0 <= int(si) < len(pool):
        cand = pool[int(si)]
        # refuse to guess if both sides carry a prompt_id and they disagree
        if row.get("prompt_id") is not None and cand.get("prompt_id") not in (None, row["prompt_id"]):
            raise KeyError(
                f"prompt_id mismatch at source_index={si}: cache has {row['prompt_id']}, "
                f"pool has {cand['prompt_id']} (cache was built from a different prompt pool)"
            )
        return prompt_messages(cand)

    raise KeyError(
        f"cannot find prompt for cache row (prompt_id={row.get('prompt_id')}, "
        f"source_index={si}); pool has {len(pool)} rows"
    )


def reconstruct_batch(cfg: dict, rows: list[dict], tokenizer, device):
    """Rebuild the fixed cached batch as padded tensors in the SAME layout batch_generate uses
    (left-padded prompt, right-padded response).

    Prompt messages come from the cache row if present, else from the PPO prompt pool via
    source_index. Response token ids come from the cache if stored, else from re-tokenizing the
    cached response text; lengths are checked against the cached log-prob vectors.
    """
    table = prompt_lookup(cfg)
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    max_p = int(cfg["max_prompt_length"])

    items, n_len_mismatch = [], 0
    for row in rows:
        msgs = find_prompt_messages(table, row)
        p_ids = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
        if hasattr(p_ids, "input_ids"):
            p_ids = p_ids["input_ids"]
        p_ids = list(p_ids)[-max_p:]

        old = _as_list(row["old_logprobs"])
        ref = _as_list(row["ref_logprobs"])
        if "response_ids" in row:
            r_ids = [int(t) for t in (row["response_ids"].tolist() if torch.is_tensor(row["response_ids"]) else row["response_ids"])]
        else:
            r_ids = tokenizer(str(row["response"]), add_special_tokens=False)["input_ids"]
            if len(r_ids) + 1 == len(old) and eos is not None:
                r_ids = r_ids + [eos]
        n = min(len(r_ids), len(old), len(ref))
        if n != len(old) or n != len(r_ids):
            n_len_mismatch += 1
        items.append((p_ids, r_ids[:n], old[:n], ref[:n], row))

    if n_len_mismatch:
        print(f"[warn] {n_len_mismatch}/{len(items)} cached rows had token/logprob length mismatches; truncated to common length")

    pw = max(len(p) for p, *_ in items)
    T = max(len(r) for _, r, *_ in items)
    B = len(items)
    seq = torch.full((B, pw + T), pad, dtype=torch.long)
    attn = torch.zeros((B, pw + T), dtype=torch.long)
    rmask = torch.zeros((B, T))
    old_lp = torch.zeros((B, T))
    ref_lp = torch.zeros((B, T))
    rids = torch.full((B, T), pad, dtype=torch.long)
    for i, (p, r, o, rf, _) in enumerate(items):
        seq[i, pw - len(p) : pw] = torch.tensor(p)
        attn[i, pw - len(p) : pw] = 1
        seq[i, pw : pw + len(r)] = torch.tensor(r)
        attn[i, pw : pw + len(r)] = 1
        rids[i, : len(r)] = torch.tensor(r)
        rmask[i, : len(r)] = 1.0
        old_lp[i, : len(o)] = torch.tensor(o)
        ref_lp[i, : len(rf)] = torch.tensor(rf)
    cached_values = None
    if "values" in items[0][4]:
        cached_values = torch.zeros((B, T))
        for i, (_p, r, _o, _f, row) in enumerate(items):
            v = _as_list(row["values"])[: len(r)]
            cached_values[i, : len(v)] = torch.tensor(v)
    batch = {k: v.to(device) for k, v in dict(seq=seq, attn=attn, rmask=rmask, old_lp=old_lp, ref_lp=ref_lp, rids=rids).items()}
    batch["pw"] = pw
    batch["T"] = T
    batch["items"] = items
    batch["cached_values"] = cached_values.to(device) if cached_values is not None else None
    return batch


def cached_rewards_and_values(cfg: dict, batch: dict, tokenizer):
    """Task rewards (cached if present, else reward model) and critic values for the cached batch."""
    rows = [it[4] for it in batch["items"]]
    cached = None
    for key in ("effective_terminal_reward", "task_reward", "reward", "rm_reward", "reward_score", "raw_reward", "raw_terminal_reward"):
        if key in rows[0]:
            cached = torch.tensor([float(r[key]) for r in rows])
            print(f"using cached rewards from key '{key}'")
            break
    if cached is None:
        rm, rm_tok = load_reward_model(cfg)
        table = prompt_lookup(cfg)
        prompts = [find_prompt_messages(table, row) for _p, _r, _o, _f, row in batch["items"]]
        resps = [str(row["response"]) for *_x, row in batch["items"]]
        cached = score_reward_pairs(rm, rm_tok, prompts, resps, max_length=int(cfg["reward_max_length"])).cpu().float()
        del rm
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if batch.get("cached_values") is not None:
        print("using cached critic values from the rollout cache")
        return cached.to(batch["seq"].device), batch["cached_values"]
    vm = load_value_model(cfg, cfg["paths"]["ppo_midpoint_value"], train_mode="frozen")
    vals = []
    with torch.no_grad():
        mb = 2
        for s in range(0, batch["seq"].shape[0], mb):
            v = token_values(vm, batch["seq"][s : s + mb], batch["attn"][s : s + mb]).float()
            vals.append(v[:, batch["pw"] - 1 : batch["pw"] - 1 + batch["T"]])
    values = torch.cat(vals, 0)
    del vm
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return cached.to(batch["seq"].device), values


def batch_logps(policy, batch, sl: slice):
    lp, _ = response_token_logprobs(
        policy, batch["seq"][sl], batch["attn"][sl], batch["pw"], batch["rids"][sl]
    )
    return lp


def eval_batch(policy, batch, adv, eps, mb=2, grad=False, stepper=None):
    """Clipped-surrogate value, clip fraction and affected-token fraction over the full cached batch.

    With grad=True also accumulates gradients (token-count-weighted across micro-batches so the
    result equals the full-batch masked mean) and returns the pre-step statistics."""
    B = batch["seq"].shape[0]
    tot_tokens = batch["rmask"].sum().clamp_min(1.0)
    surr = clip = aff = 0.0
    for s in range(0, B, mb):
        sl = slice(s, s + mb)
        m = batch["rmask"][sl]
        w = (m.sum() / tot_tokens).item()
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            new_lp = batch_logps(policy, batch, sl)
            loss, _ratio, cf = ppo_policy_loss(new_lp, batch["old_lp"][sl], adv[sl], m, eps)
            af = ppo_affected_fraction(new_lp.detach(), batch["old_lp"][sl], adv[sl], m, eps)
            if grad:
                stepper.backward(loss * w)
        surr += -float(loss.item()) * w  # clipped surrogate objective (to be maximised)
        clip += float(cf.item()) * w
        aff += float(af.item()) * w
    return {"surrogate": surr, "clip_fraction": clip, "affected_fraction": aff}


def run_cached_study(config_path: str, mb: int = 2, multipliers=(1.0, 10.0)):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    rows = load_cached_rollouts(cfg["cached_rollouts"])
    print("Cached PPO rollouts:", len(rows))
    print("Required epsilon values:", cfg["clip_values"])
    print("Cache keys:", sorted(rows[0].keys()))

    tokenizer = load_tokenizer(cfg["base_model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = reconstruct_batch(cfg, rows, tokenizer, device)
    task_reward, values = cached_rewards_and_values(cfg, batch, tokenizer)

    beta_kl = float(cfg["kl_beta"])
    rewards = shaped_rewards(task_reward, batch["old_lp"], batch["ref_lp"], batch["rmask"], beta_kl)
    adv, returns = compute_gae(rewards, values, batch["rmask"], float(cfg["gamma"]), float(cfg["gae_lambda"]))
    adv_n = normalize_advantages(adv, batch["rmask"])

    # ---- (a) static geometry: drift of the midpoint policy from the reference policy ----
    # rho_ref = pi_old / pi_ref. At the very first inner step the PPO ratio is exactly 1, so this
    # is the only zero-update measurement of "how often would eps-clipping bind on these tokens".
    rho_ref = torch.exp(batch["old_lp"] - batch["ref_lp"])
    static = {}
    for eps in cfg["clip_values"]:
        outside = ((rho_ref < 1 - eps) | (rho_ref > 1 + eps)).float()
        static[str(eps)] = {"drift_ratio_outside_fraction": float(masked_mean(outside, batch["rmask"]).item())}

    # ---- (b) dynamic geometry: inner gradient steps on the fixed batch ----
    policy = load_policy(cfg, adapter_path=cfg["paths"]["ppo_midpoint_policy"], trainable=True)
    disable_dropout(policy)
    init_state = {n: p.detach().clone() for n, p in policy.named_parameters() if p.requires_grad}
    base_lr = float(cfg["policy_learning_rate"])
    n_inner = int(cfg.get("inner_steps_cached", 4))

    results = {"n_rollouts": len(rows), "n_tokens": int(batch["rmask"].sum().item()), "static": static, "dynamic": {}}
    for mult in multipliers:
        for eps in cfg["clip_values"]:
            with torch.no_grad():
                for n, p in policy.named_parameters():
                    if p.requires_grad:
                        p.copy_(init_state[n])
            opt = AdamW(trainable_parameters(policy), lr=base_lr * mult)
            stepper = Stepper(opt, trainable_parameters(policy), float(cfg["max_grad_norm"]))
            traj = []
            for step in range(n_inner):
                stats = eval_batch(policy, batch, adv_n, float(eps), mb=mb, grad=True, stepper=stepper)
                stats["grad_norm"] = stepper.step()
                stats["inner_step"] = step
                traj.append(stats)
            # measurement after the last update (no further step)
            final = eval_batch(policy, batch, adv_n, float(eps), mb=mb, grad=False)
            traj.append({**final, "inner_step": n_inner, "grad_norm": None})
            results["dynamic"][f"lr_x{mult:g}_eps{eps}"] = {
                "lr": base_lr * mult,
                "eps": float(eps),
                "trajectory": traj,
                "final_clip_fraction": final["clip_fraction"],
                "final_affected_fraction": final["affected_fraction"],
                "final_surrogate": final["surrogate"],
                "mean_clip_fraction_over_steps": sum(t["clip_fraction"] for t in traj) / len(traj),
            }
            print(f"lr x{mult:g} eps={eps}: final clip={final['clip_fraction']:.4f} affected={final['affected_fraction']:.4f}")

    out = repo_path(cfg["results_dir"]) / "clipping_cached_batch.json"
    save_json(out, results)
    print("saved", out)
    del policy
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--micro-batch", type=int, default=2)
    ap.add_argument("--skip-cached", action="store_true", help="skip the cached-batch study")
    ap.add_argument("--skip-forks", action="store_true", help="skip the matched short continuation forks")
    ap.add_argument("--inspect", action="store_true", help="only print the cache schema and exit")
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    if args.inspect:
        rows = load_cached_rollouts(cfg["cached_rollouts"])
        print("rows:", len(rows))
        for k, v in rows[0].items():
            print(f"  {k}: {type(v).__name__} {tuple(v.shape) if torch.is_tensor(v) else (len(v) if hasattr(v, '__len__') else v)}")
        return

    if not args.skip_cached:
        run_cached_study(args.config, mb=args.micro_batch)

    if not args.skip_forks:
        from task2_ppo.continue_train import run_ppo
        from task2_ppo.evaluate import evaluate_adapter

        for eps in cfg["clip_values"]:
            tag = f"clip_{float(eps):.2f}"
            out = f"{cfg['fork_output_dir']}/{tag}"
            # Same checkpoint, prompts, seed, KL beta, generation budget; ONLY epsilon changes.
            run_ppo(args.config, output=out, updates=int(cfg["fork_updates"]), clip_epsilon=float(eps),
                    kl_beta=float(cfg["kl_beta"]), run_name=tag)
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            evaluate_adapter(args.config, out, f"{tag}_heldout")


if __name__ == "__main__":
    main()