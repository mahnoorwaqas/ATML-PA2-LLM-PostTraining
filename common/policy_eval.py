"""Common held-out generation evaluation for PPO / GRPO / DPO policies.

Everything that is compared across conditions (prompt set, decoding, cap, seed, RM, KL
estimator) is fixed here so conditions cannot drift apart.
"""
from __future__ import annotations

import gc
from pathlib import Path

import torch

from common.data import prompt_messages, repo_path
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import save_json, set_seed, append_jsonl
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode
from common.train_utils import seq_stats, token_entropy, ResourceTracker


def row_id(row: dict, idx: int):
    for k in ("source_index", "prompt_id", "id", "pair_id"):
        if k in row:
            return row[k]
    return idx


@torch.no_grad()
def generate_and_score(
    policy,
    tokenizer,
    rm,
    rm_tok,
    prompts: list[list[dict]],
    ids: list,
    cfg: dict,
    max_new_tokens: int,
    batch_size: int = 4,
    greedy: bool = False,
    max_prompt_length: int = 256,
    rm_max_length: int = 1280,
):
    """Generate one response per prompt; return per-example records.

    KL is the course sampled estimator: mean over response tokens of
    (log pi_policy - log pi_ref) on the policy's own samples, with the reference = base model
    (adapter disabled). Both token-averaged and sequence-summed values are stored per example.
    """
    gen_cfg = cfg.get("generation", {})
    records = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        gen = batch_generate(
            policy,
            tokenizer,
            chunk,
            max_prompt_length=max_prompt_length,
            max_new_tokens=max_new_tokens,
            temperature=float(gen_cfg.get("temperature", 0.7)),
            top_p=float(gen_cfg.get("top_p", 0.9)),
            do_sample=(not greedy) and bool(gen_cfg.get("do_sample", True)),
        )
        seq, attn, pw, rids, rmask = (
            gen["sequences"], gen["attention_mask"], gen["prompt_width"], gen["response_ids"], gen["response_mask"],
        )
        pol_lp, logits = response_token_logprobs(policy, seq, attn, pw, rids)
        ent_total = token_entropy(logits, rmask)  # batch-mean; per-example below
        with reference_mode(policy):
            ref_lp, _ = response_token_logprobs(policy, seq, attn, pw, rids)
        diff = (pol_lp - ref_lp) * rmask
        n_tok = rmask.sum(-1).clamp_min(1.0)
        # per-example entropy
        ents = []
        for b in range(logits.shape[0]):
            ents.append(float(token_entropy(logits[b : b + 1], rmask[b : b + 1]).item()))
        scores = score_reward_pairs(rm, rm_tok, chunk, gen["responses"], max_length=rm_max_length)
        for i in range(len(chunk)):
            records.append(
                {
                    "prompt_id": ids[start + i],
                    "response": gen["responses"][i],
                    "n_tokens": int(gen["response_lengths"][i]),
                    "truncated": bool(gen["truncated"][i]),
                    "terminated_with_eos": bool(gen["terminated_with_eos"][i]),
                    "rm_score": float(scores[i].item()),
                    "kl_token_mean": float((diff[i].sum() / n_tok[i]).item()),
                    "kl_seq_sum": float(diff[i].sum().item()),
                    "entropy": ents[i],
                }
            )
        del gen, logits, pol_lp, ref_lp
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return records


def summarize_records(records: list[dict]) -> dict:
    import numpy as np

    lens = [r["n_tokens"] for r in records]
    toks = np.array(lens, dtype=float)
    kl_tok_weighted = float(sum(r["kl_seq_sum"] for r in records) / max(toks.sum(), 1.0))
    return {
        "n": len(records),
        "rm_score_mean": float(np.mean([r["rm_score"] for r in records])),
        "rm_score_std": float(np.std([r["rm_score"] for r in records])),
        # Convention used throughout the report: token-weighted KL = sum(logratio)/sum(tokens)
        # (this equals common.metrics.sampled_kl over the whole evaluation set).
        "kl_token_weighted": kl_tok_weighted,
        "kl_per_sequence_token_mean": float(np.mean([r["kl_token_mean"] for r in records])),
        "kl_seq_sum_mean": float(np.mean([r["kl_seq_sum"] for r in records])),
        "entropy_mean": float(np.mean([r["entropy"] for r in records])),
        "length": seq_stats(lens),
        "truncation_rate": float(np.mean([r["truncated"] for r in records])),
        "eos_rate": float(np.mean([r["terminated_with_eos"] for r in records])),
    }


def evaluate_policy_generation(
    cfg: dict,
    adapter: str | None,
    name: str,
    rows: list[dict],
    results_dir: str,
    max_new_tokens: int,
    n_prompts: int | None,
    batch_size: int = 4,
    greedy: bool = False,
    extra: dict | None = None,
):
    """Load adapter, evaluate on the first n_prompts rows, save per-example + summary JSON."""
    set_seed(int(cfg["seed"]))
    if n_prompts is not None:
        rows = rows[: int(n_prompts)]
    prompts = [prompt_messages(r) for r in rows]
    ids = [row_id(r, i) for i, r in enumerate(rows)]

    tracker = ResourceTracker()
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=adapter, trainable=False)
    rm, rm_tok = load_reward_model(cfg)
    records = generate_and_score(
        policy, tokenizer, rm, rm_tok, prompts, ids, cfg,
        max_new_tokens=max_new_tokens, batch_size=batch_size, greedy=greedy,
        max_prompt_length=int(cfg.get("max_prompt_length", 256)),
        rm_max_length=int(cfg.get("reward_max_length", 1280)),
    )
    summary = summarize_records(records)
    summary.update(
        {
            "name": name,
            "adapter": adapter,
            "max_new_tokens": max_new_tokens,
            "greedy": greedy,
            "seed": int(cfg["seed"]),
            "prompt_ids": ids,
            "eval_resources": tracker.summary(),
        }
    )
    if extra:
        summary.update(extra)
    out = repo_path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"{name}_generations.jsonl").open("w", encoding="utf-8") as f:
        import json

        for p, r in zip(prompts, records):
            f.write(json.dumps({**r, "prompt": p}, ensure_ascii=False) + "\n")
    save_json(out / f"{name}_eval.json", summary)
    del policy, rm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary
