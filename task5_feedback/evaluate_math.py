from __future__ import annotations

import argparse
import gc
import json
import re

import numpy as np
import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate
from common.logging_utils import save_json, set_seed
from common.models import load_policy, load_tokenizer
from common.train_utils import seq_stats
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward, extract_designated_final

PAIRS = [("rlaif", "sft"), ("rlvr", "sft"), ("rlaif", "rlvr")]


def policy_specs(cfg):
    return {
        "sft": None,
        "rlvr": cfg["policies"]["rlvr"],
        "rlaif": cfg["policies"]["rlaif"],
    }


def dataset_path(cfg, dataset: str):
    if dataset == "gsm":
        return cfg["paths"]["gsm_eval"]
    if dataset == "transfer":
        return cfg["paths"]["math_transfer_eval"]
    raise ValueError(dataset)


def load_math_evaluation(config_path: str, dataset: str):
    cfg = load_yaml(config_path)
    rows = read_jsonl(dataset_path(cfg, dataset))
    tokenizer = load_tokenizer(cfg["base_model"])
    return cfg, rows, tokenizer


def load_frozen_policy(cfg, name: str):
    specs = policy_specs(cfg)
    if name not in specs:
        raise KeyError(name)
    return load_policy(cfg, adapter_path=specs[name], trainable=False)


def gold_of(row: dict) -> str:
    if "gold_final" in row:
        return str(row["gold_final"]).replace(",", "")
    m = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", str(row.get("answer", "")))
    if m:
        return m.group(1).replace(",", "")
    raise KeyError(f"no gold answer field in row with keys {sorted(row)}")


def question_of(row: dict) -> str:
    if "question" in row:
        return str(row["question"])
    return prompt_messages(row)[-1]["content"]


def failure_type(correct: bool, fmt_ok: bool, truncated: bool) -> str:
    if correct:
        return "correct"
    if truncated and not fmt_ok:
        return "truncated_no_final"
    if not fmt_ok:
        return "format_missing"
    return "wrong_answer"


def generate_policy_responses(cfg, rows, tokenizer, name: str, batch_size: int = 8):
    """Greedy, matched decoding for every policy (same prompts, cap, tokenizer)."""
    model = load_frozen_policy(cfg, name)
    recs = []
    cap = int(cfg["math_max_new_tokens"])
    for s in range(0, len(rows), batch_size):
        chunk = rows[s : s + batch_size]
        gen = batch_generate(
            model, tokenizer, [prompt_messages(r) for r in chunk],
            max_prompt_length=512, max_new_tokens=cap, temperature=0.0, top_p=1.0, do_sample=False,
        )
        for r, resp, n, trunc in zip(chunk, gen["responses"], gen["response_lengths"], gen["truncated"]):
            gold = gold_of(r)
            final = extract_designated_final(resp)
            correct = bool(exact_reward(resp, gold))
            fmt_ok = final is not None
            recs.append(
                {
                    "source_index": r.get("source_index"),
                    "policy": name,
                    "question": question_of(r),
                    "gold_final": gold,
                    "response": resp,
                    "n_tokens": int(n),
                    "truncated": bool(trunc),
                    "pred_final": final,
                    "format_ok": fmt_ok,
                    "correct": correct,
                    "failure_type": failure_type(correct, fmt_ok, bool(trunc)),
                }
            )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return recs


def pairwise_stats(judge: PairwiseAIJudge, a_recs: list[dict], b_recs: list[dict]) -> dict:
    """AI pairwise win rate of A over B (win=1, tie=0.5, loss=0) + verifier-judge agreement."""
    wins = ties = losses = 0
    agree = disagree = judge_tie_on_distinct = n_distinct = 0
    judge_decides_on_equal = n_equal = 0
    prefers_longer = n_decided = 0
    per_example = []
    for a, b in zip(a_recs, b_recs):
        assert a["question"] == b["question"]
        pref = judge.compare(a["question"], a["response"], b["response"])
        score = {"A": 1.0, "TIE": 0.5, "B": 0.0}[pref]
        wins += pref == "A"; ties += pref == "TIE"; losses += pref == "B"
        va, vb = float(a["correct"]), float(b["correct"])
        if va != vb:
            n_distinct += 1
            better = "A" if va > vb else "B"
            if pref == better:
                agree += 1
            elif pref == "TIE":
                judge_tie_on_distinct += 1
            else:
                disagree += 1
        else:
            n_equal += 1
            judge_decides_on_equal += pref != "TIE"
        if pref != "TIE":
            n_decided += 1
            longer = "A" if a["n_tokens"] > b["n_tokens"] else ("B" if b["n_tokens"] > a["n_tokens"] else None)
            prefers_longer += (longer == pref) if longer else 0.5
        per_example.append({"source_index": a["source_index"], "judge": pref, "verifier_a": va, "verifier_b": vb})
    n = len(a_recs)
    return {
        "n": n,
        "win_rate_a": (wins + 0.5 * ties) / n,
        "wins": int(wins), "ties": int(ties), "losses": int(losses),
        "tie_rate": ties / n,
        "verifier_distinguishes_n": n_distinct,
        "judge_agrees_with_verifier": agree / n_distinct if n_distinct else float("nan"),
        "judge_disagrees_with_verifier": disagree / n_distinct if n_distinct else float("nan"),
        "judge_tie_when_verifier_distinguishes": judge_tie_on_distinct / n_distinct if n_distinct else float("nan"),
        "verifier_tied_n": n_equal,
        "judge_decisive_when_verifier_tied": judge_decides_on_equal / n_equal if n_equal else float("nan"),
        "judge_prefers_longer_among_decisive": prefers_longer / n_decided if n_decided else float("nan"),
        "_per_example": per_example,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--dataset", choices=["gsm", "transfer"], default="gsm")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--skip-judge", action="store_true")
    args = ap.parse_args()
    cfg, rows, tokenizer = load_math_evaluation(args.config, args.dataset)
    set_seed(int(cfg["seed"]))
    print("Rows:", len(rows), "dataset:", args.dataset)
    outdir = repo_path(cfg["results_dir"]) / "task5_feedback"
    outdir.mkdir(parents=True, exist_ok=True)

    recs = {}
    for name in policy_specs(cfg):
        f = outdir / f"{args.dataset}_{name}.jsonl"
        recs[name] = generate_policy_responses(cfg, rows, tokenizer, name, args.batch_size)
        write_jsonl(f, recs[name])
        print(f"[{name}] acc={np.mean([r['correct'] for r in recs[name]]):.3f}  fmt={np.mean([r['format_ok'] for r in recs[name]]):.3f}")

    summary = {"dataset": args.dataset, "n": len(rows), "max_new_tokens": int(cfg["math_max_new_tokens"]), "policies": {}, "pairwise": {}}
    for name, rr in recs.items():
        fails = {}
        for r in rr:
            fails[r["failure_type"]] = fails.get(r["failure_type"], 0) + 1
        summary["policies"][name] = {
            "exact_accuracy": float(np.mean([r["correct"] for r in rr])),
            "format_compliance": float(np.mean([r["format_ok"] for r in rr])),
            "length_tokens": seq_stats([r["n_tokens"] for r in rr]),
            "truncation_rate": float(np.mean([r["truncated"] for r in rr])),
            "failure_types": fails,
        }

    if not args.skip_judge:
        judge = PairwiseAIJudge(cfg, outdir / "judge_cache.json")
        for a, b in PAIRS:
            st = pairwise_stats(judge, recs[a], recs[b])
            per = st.pop("_per_example")
            write_jsonl(outdir / f"{args.dataset}_pairwise_{a}_vs_{b}.jsonl", per)
            summary["pairwise"][f"{a}_vs_{b}"] = st
            print(f"{a} vs {b}: win_rate={st['win_rate_a']:.3f} ties={st['ties']} agree_w_verifier={st['judge_agrees_with_verifier']:.3f}")
    save_json(outdir / f"{args.dataset}_summary.json", summary)
    print("saved", outdir / f"{args.dataset}_summary.json")


if __name__ == "__main__":
    main()
