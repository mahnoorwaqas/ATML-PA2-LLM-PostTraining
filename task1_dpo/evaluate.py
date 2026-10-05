from __future__ import annotations

import argparse
import gc
import json

import numpy as np
import torch

from common.data import (
    load_yaml,
    prompt_messages_from_preference,
    preference_responses,
    read_jsonl,
    repo_path,
)
from common.generation import batch_generate, response_sequence_logprobs
from common.logging_utils import save_json, set_seed
from common.metrics import word_count, word_limit_compliance
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode
from common.policy_eval import generate_and_score, summarize_records
from common.train_utils import seq_stats
from task1_dpo.dpo import dpo_loss
from task1_dpo.preprocess import filter_fitting
from task1_dpo.train import make_collate

STRATUM_KEYS = ("length_stratum", "stratum", "length_bucket", "bucket", "length_group", "stratification")


def get_stratum(row: dict, ratio_hi: float = 1.25) -> str:
    """Return one of 'preferred_longer' | 'matched' | 'rejected_longer'.

    Uses the course-supplied label if the eval file has one; otherwise falls back to a word-length
    ratio rule (documented in the report if used)."""
    for k in STRATUM_KEYS:
        if k in row:
            v = str(row[k]).lower()
            if "prefer" in v or "chosen" in v or v in {"pref_longer", "chosen_longer", "long_chosen"}:
                return "preferred_longer"
            if "reject" in v:
                return "rejected_longer"
            if "match" in v or "similar" in v or "equal" in v:
                return "matched"
    yc, yr = preference_responses(row)
    lc, lr = max(word_count(yc), 1), max(word_count(yr), 1)
    if lc / lr >= ratio_hi:
        return "preferred_longer"
    if lr / lc >= ratio_hi:
        return "rejected_longer"
    return "matched"


@torch.no_grad()
def pair_logps(model, tokenizer, rows, max_len: int, batch_size: int):
    """Policy and reference (adapter disabled) summed response log-probs for chosen/rejected."""
    collate = make_collate(tokenizer, max_len)
    device = next(model.parameters()).device
    out = {"pol_c": [], "pol_r": [], "ref_c": [], "ref_r": []}
    for s in range(0, len(rows), batch_size):
        chosen, rejected = collate(rows[s : s + batch_size])
        chosen = {k: v.to(device) for k, v in chosen.items()}
        rejected = {k: v.to(device) for k, v in rejected.items()}
        pc, _, _ = response_sequence_logprobs(model, chosen)
        pr, _, _ = response_sequence_logprobs(model, rejected)
        with reference_mode(model):
            rc, _, _ = response_sequence_logprobs(model, chosen)
            rr, _, _ = response_sequence_logprobs(model, rejected)
        for k, v in zip(("pol_c", "pol_r", "ref_c", "ref_r"), (pc, pr, rc, rr)):
            out[k].extend(v.float().cpu().tolist())
    return {k: torch.tensor(v) for k, v in out.items()}


def pair_metrics(lp: dict, beta: float) -> dict:
    loss, diag = dpo_loss(lp["pol_c"], lp["pol_r"], lp["ref_c"], lp["ref_r"], beta)
    margin = (lp["pol_c"] - lp["pol_r"]) - (lp["ref_c"] - lp["ref_r"])
    return {
        "dpo_loss": float(loss.item()),
        "pref_accuracy": float((margin > 0).float().mean().item()),
        "margin_mean": float(margin.mean().item()),
        "n_pairs": int(margin.numel()),
    }


def load_evaluation_bundle(config_path: str, adapter: str | None):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["dpo_standard_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


@torch.no_grad()
def word_limit_eval(policy, tokenizer, cfg: dict, n_samples: int):
    rows = read_jsonl(cfg["paths"]["word_limit_prompts"])
    gen_cfg = cfg.get("generation", {})
    records = []
    for row in rows:
        prompt = row["messages"]
        text = prompt[-1]["content"]
        gen = batch_generate(
            policy, tokenizer, [prompt] * n_samples, max_prompt_length=256,
            max_new_tokens=int(cfg["max_generation_tokens"]),
            temperature=float(gen_cfg.get("temperature", 0.7)), top_p=float(gen_cfg.get("top_p", 0.9)),
            do_sample=True,
        )
        for resp, n_tok in zip(gen["responses"], gen["response_lengths"]):
            records.append(
                {
                    "prompt_id": row["prompt_id"],
                    "response": resp,
                    "words": word_count(resp),
                    "tokens": int(n_tok),
                    "compliant": word_limit_compliance(text, resp),
                }
            )
    comp = [r["compliant"] for r in records if r["compliant"] is not None]
    return {
        "compliance_rate": float(np.mean(comp)) if comp else float("nan"),
        "words": seq_stats([r["words"] for r in records]),
        "n_samples_per_prompt": n_samples,
        "n_total": len(records),
    }, records


def evaluate_adapter(
    config_path: str,
    adapter: str | None,
    name: str,
    beta: float | None = None,
    with_strata: bool = False,
    with_word_limit: bool = True,
    with_generation: bool = True,
):
    cfg = load_yaml(config_path)
    beta = float(cfg["beta"] if beta is None else beta)
    set_seed(int(cfg["seed"]))
    results_dir = repo_path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=adapter, trainable=False)
    result = {"name": name, "adapter": adapter, "beta_for_loss": beta, "seed": int(cfg["seed"])}

    max_len = int(cfg["max_sequence_length"])
    bs = int(cfg.get("eval_batch_size", 4))
    held, n_drop_held = filter_fitting(tokenizer, read_jsonl(cfg["paths"]["dpo_standard_eval"]), max_len)
    result["n_heldout_pairs"] = len(held)
    result["n_heldout_dropped_prompt_too_long"] = n_drop_held
    if adapter is not None:
        lp = pair_logps(policy, tokenizer, held, max_len, bs)
        result["heldout"] = pair_metrics(lp, beta)
        if with_strata:
            strat_rows, n_drop_strat = filter_fitting(tokenizer, read_jsonl(cfg["paths"]["dpo_length_eval"]), max_len)
            result["n_stratified_dropped_prompt_too_long"] = n_drop_strat
            lps = pair_logps(policy, tokenizer, strat_rows, max_len, bs)
            labels = np.array([get_stratum(r) for r in strat_rows])
            result["length_stratified"] = {}
            for s in ("preferred_longer", "matched", "rejected_longer"):
                idx = np.where(labels == s)[0]
                if len(idx) == 0:
                    continue
                sub = {k: v[idx] for k, v in lps.items()}
                result["length_stratified"][s] = pair_metrics(sub, beta)
            result["length_stratified"]["_overall"] = pair_metrics(lps, beta)
            result["length_stratified"]["_labels_from_file"] = any(k in strat_rows[0] for k in STRATUM_KEYS)
    else:
        result["heldout"] = None  # base policy: margin is identically zero by construction

    if with_generation:
        n = int(cfg.get("eval_generation_prompts", 100))
        rows = held[:n]
        prompts = [prompt_messages_from_preference(r) for r in rows]
        ids = [r.get("pair_id", r.get("source_index", i)) for i, r in enumerate(rows)]
        rm, rm_tok = load_reward_model(cfg)
        records = generate_and_score(
            policy, tokenizer, rm, rm_tok, prompts, ids, cfg,
            max_new_tokens=int(cfg["max_generation_tokens"]), batch_size=bs,
            max_prompt_length=256,
        )
        result["generation"] = summarize_records(records)
        result["generation"]["prompt_ids"] = ids
        with (results_dir / f"{name}_generations.jsonl").open("w", encoding="utf-8") as f:
            for p, r in zip(prompts, records):
                f.write(json.dumps({**r, "prompt": p}, ensure_ascii=False) + "\n")
        del rm
    if with_word_limit:
        wl, wl_records = word_limit_eval(policy, tokenizer, cfg, int(cfg.get("word_limit_samples", 8)))
        result["word_limit"] = wl
        with (results_dir / f"{name}_word_limit.jsonl").open("w", encoding="utf-8") as f:
            for r in wl_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    save_json(results_dir / f"{name}_eval.json", result)
    del policy
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter", required=True, help="adapter dir, or 'none' for the untouched base policy")
    ap.add_argument("--name", default="standard")
    ap.add_argument("--beta", type=float, help="beta used for the reported DPO loss (default: config beta)")
    ap.add_argument("--strata", action="store_true", help="also evaluate on the length-stratified held-out set")
    ap.add_argument("--skip-word-limit", action="store_true")
    ap.add_argument("--skip-generation", action="store_true")
    args = ap.parse_args()
    adapter = None if args.adapter.lower() == "none" else args.adapter
    evaluate_adapter(
        args.config, adapter, args.name, args.beta, args.strata,
        with_word_limit=not args.skip_word_limit, with_generation=not args.skip_generation,
    )


if __name__ == "__main__":
    main()
