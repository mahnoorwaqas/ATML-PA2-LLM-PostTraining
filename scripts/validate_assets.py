from __future__ import annotations

from collections import Counter

from common.data import load_yaml, repo_path, read_jsonl

cfg = load_yaml("configs/base.yaml")
required_data = [
    "dpo_standard_train", "dpo_standard_eval", "dpo_length_train", "dpo_length_eval",
    "word_limit_prompts", "rl_prompt_train", "rl_prompt_eval", "xstest",
    "gsm_train", "gsm_eval", "math_transfer_eval", "task5_diagnostics",
]
required_cached = ["ppo_rollout_cache", "grpo_k_cache"]
required_ckpt = [
    "ppo_midpoint_policy", "ppo_midpoint_value", "grpo_midpoint_policy",
    "rlvr_policy", "rlaif_policy",
]

missing = []
for key in required_data + required_cached + required_ckpt:
    p = repo_path(cfg["paths"][key])
    ok = p.exists()
    shown = p.relative_to(repo_path(".")) if ok else p
    print(f"{key:28s} {'OK' if ok else 'MISSING'}  {shown}")
    if not ok:
        missing.append(str(p))

if missing:
    raise SystemExit("Missing course assets:\n" + "\n".join(missing))

# Fixed small datasets.
word_rows = read_jsonl(cfg["paths"]["word_limit_prompts"])
if len(word_rows) != 10:
    raise SystemExit(f"Expected 10 word-limit prompts; found {len(word_rows)}")
transfer_rows = read_jsonl(cfg["paths"]["math_transfer_eval"])
if len(transfer_rows) != 100:
    raise SystemExit(f"Expected 100 fixed transfer examples; found {len(transfer_rows)}")
if not {"source_index", "question", "messages", "gold_final"}.issubset(transfer_rows[0]):
    raise SystemExit("Transfer-set schema mismatch")

t5 = read_jsonl(cfg["paths"]["task5_diagnostics"])
if len(t5) != 100:
    raise SystemExit(f"Expected 100 Task-5 diagnostic rows; found {len(t5)}")
if any(not bool(r.get("manual_validation", False)) for r in t5):
    raise SystemExit("Task-5 diagnostic set contains a row not marked manual_validation=true")

# GRPO cache: one row per completion, 8 rows/prompt in the released diagnostic cache.
grows = read_jsonl(cfg["paths"]["grpo_k_cache"])
if not grows:
    raise SystemExit("GRPO K-cache is empty")
required_grpo_keys = {"source_index", "generation_index", "completion", "reward"}
if not required_grpo_keys.issubset(grows[0]):
    raise SystemExit(f"GRPO K-cache schema mismatch: expected {sorted(required_grpo_keys)}")
counts = Counter(str(r["source_index"]) for r in grows)
if not counts or min(counts.values()) < 8:
    raise SystemExit(f"GRPO K-cache must contain at least 8 completions/prompt; min={min(counts.values()) if counts else 0}")

# PPO cache: tolerate either equivalent field naming convention from staff preparation.
try:
    import torch
    ppo = torch.load(repo_path(cfg["paths"]["ppo_rollout_cache"]), map_location="cpu", weights_only=False)
except Exception as exc:
    raise SystemExit(f"Could not load PPO rollout cache: {exc}")
if not isinstance(ppo, list) or not ppo:
    raise SystemExit("PPO rollout cache must be a non-empty list")
row = ppo[0]
if "old_logprobs" not in row and "old_policy_logprobs" not in row:
    raise SystemExit("PPO cache missing old-policy log-probabilities")
if "ref_logprobs" not in row and "reference_logprobs" not in row:
    raise SystemExit("PPO cache missing reference log-probabilities")
if not {"source_index", "response"}.issubset(row):
    raise SystemExit("PPO cache missing source_index/response")

print("All required course assets are present and release schemas are valid.")
