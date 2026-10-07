"""End-to-end CPU smoke test of Task 2 (and the GRPO loop, which shares the generation path).
Tiny random Qwen2 policy + tiny sequence-classification critic, fake tokenizer, synthetic data.
Run:  python -m tests.smoke_task2
"""
from __future__ import annotations

import json
import math
import random
import sys
import tempfile
from pathlib import Path

import torch
import yaml

from tests.smoke_task1 import FakeTok, PAD, EOS, VOCAB, tiny_base  # also patches tokenizer/RM/reward scorer
import common.models as cm
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import Qwen2Config, Qwen2ForSequenceClassification


def fake_policy(cfg, adapter_path=None, trainable=False, fresh_lora=False):
    base = tiny_base()
    if adapter_path and (Path(adapter_path) / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=trainable)
    else:  # stand-in for the supplied midpoint: fresh LoRA (B=0 => equals base policy)
        torch.manual_seed(1)
        model = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.05, target_modules=["q_proj", "v_proj"]))
    model.train() if trainable else model.eval()
    return model


def fake_value(cfg, checkpoint, train_mode="lora_head"):
    torch.manual_seed(2)
    c = Qwen2Config(vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                    num_key_value_heads=2, tie_word_embeddings=False, pad_token_id=PAD, num_labels=1)
    m = Qwen2ForSequenceClassification(c)
    if train_mode == "lora_head":
        m = get_peft_model(m, LoraConfig(task_type="SEQ_CLS", r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"]))
        m.train()
    elif train_mode == "frozen":
        for p in m.parameters():
            p.requires_grad_(False)
        m.eval()
    return m


cm.load_policy = fake_policy
cm.load_value_model = fake_value

from common.data import _deep_merge, load_yaml, write_jsonl  # noqa: E402
from common.logging_utils import set_seed  # noqa: E402
from task2_ppo import ablate_kl, analyze_clipping, summarize  # noqa: E402
from task2_ppo.continue_train import run_ppo  # noqa: E402
from task2_ppo.evaluate import evaluate_adapter  # noqa: E402


def check(c, m):
    print(("PASS  " if c else "FAIL  ") + m)
    if not c:
        raise SystemExit("smoke test failed: " + m)


def make_everything(tmp: Path):
    d = tmp / "data"
    d.mkdir()
    tok = FakeTok()
    rows = [{"prompt_id": f"p{i}", "source_index": i, "messages": [{"role": "user", "content": f"Tell me about topic {i} please."}]} for i in range(20)]
    write_jsonl(d / "pool_train.jsonl", rows)
    write_jsonl(d / "pool_eval.jsonl", [dict(r, source_index=100 + i) for i, r in enumerate(rows[:6])])
    rnd = random.Random(0)
    cache = []
    for i in range(6):
        text = "ab cd " * rnd.randint(2, 5)
        n = len(tok._ids(text)) + 1  # + EOS
        old = [-rnd.random() * 2 for _ in range(n)]
        cache.append({"source_index": i, "response": text, "old_logprobs": old, "ref_logprobs": [o + rnd.uniform(-0.4, 0.4) for o in old], "reward": rnd.uniform(-1, 1)})
    torch.save(cache, d / "ppo_rollout.pt")
    kc = []
    for i in range(5):
        for g in range(8):
            kc.append({"source_index": i, "generation_index": g, "reward": float((g + i) % 3 == 0), "response": "x"})
    write_jsonl(d / "grpo_k.jsonl", kc)

    over = {
        "updates": 3, "fork_updates": 2, "prompts_per_update": 2, "ppo_epochs": 2, "max_prompt_length": 64,
        "max_response_length": 8, "eval_max_response_length": 8, "reward_max_length": 128, "eval_prompts": 4, "eval_batch_size": 2,
        "inner_steps_cached": 2, "policy_learning_rate": 1e-3, "value_lora_learning_rate": 1e-3, "value_head_learning_rate": 1e-3,
        "clip_values": [0.05, 0.2, 0.5], "kl_values": [0.0, 0.1, 0.2],
        "output": str(tmp / "out/standard"), "fork_output_dir": str(tmp / "out"), "results_dir": str(tmp / "results"),
        "cached_rollouts": str(d / "ppo_rollout.pt"),
        "paths": {"rl_prompt_train": str(d / "pool_train.jsonl"), "rl_prompt_eval": str(d / "pool_eval.jsonl"),
                  "ppo_midpoint_policy": str(tmp / "none_policy"), "ppo_midpoint_value": str(tmp / "none_value")},
    }
    p = tmp / "ppo_smoke.yaml"
    p.write_text(yaml.safe_dump(_deep_merge(load_yaml("configs/ppo.yaml"), over)))

    g = {
        "updates": 2, "fork_updates": 2, "prompts_per_update": 2, "num_generations": 4, "max_prompt_length": 64,
        "max_completion_length": 8, "eval_max_completion_length": 8, "eval_prompts": 4, "eval_batch_size": 2,
        "learning_rate": 1e-3, "output": str(tmp / "out/grpo_standard"), "fork_output_dir": str(tmp / "out"),
        "results_dir": str(tmp / "results_grpo"), "group_cache": str(d / "grpo_k.jsonl"),
        "paths": over["paths"] | {"grpo_midpoint_policy": str(tmp / "none_policy")},
    }
    gp = tmp / "grpo_smoke.yaml"
    gp.write_text(yaml.safe_dump(_deep_merge(load_yaml("configs/grpo.yaml"), g)))
    return str(p), str(gp)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="ppo_smoke_"))
    cfg, gcfg = make_everything(tmp)
    R = tmp / "results"

    # --- fp32-trainable helper (critic head is fp16 on GPU; GradScaler cannot unscale fp16 grads)
    from common.train_utils import make_trainables_fp32

    head = torch.nn.Sequential(torch.nn.Linear(4, 1).half())
    make_trainables_fp32(head)
    out = head(torch.randn(2, 4).half())
    check(next(head.parameters()).dtype == torch.float32 and out.dtype == torch.float32, "fp16 trainable head -> fp32 params, forward still works")

    s = run_ppo(cfg, run_name="standard")
    log = [json.loads(l) for l in open(R / "standard_log.jsonl")]
    check(len(log) == 3, "3 PPO updates logged")
    need = {"reward_mean", "kl", "policy_loss", "value_loss", "entropy", "grad_norm", "clip_fraction", "affected_fraction", "response_length", "critic_explained_variance"}
    check(need <= set(log[0]), "all required PPO diagnostics logged")
    check(all(math.isfinite(v) for r in log for k, v in r.items() if isinstance(v, float)), "no NaN/inf in logs")
    check(abs(log[0]["kl"]) < 1e-5 and log[0]["clip_fraction"] == 0.0, f"update-1: policy==reference (KL~0) and ratio==1 (clip=0): kl={log[0]['kl']:.2e}")
    check((tmp / "out/standard/adapter_config.json").exists() and "wall_clock_seconds" in s, "adapter saved + time/VRAM summary")

    e = evaluate_adapter(cfg, str(tmp / "out/standard"), "standard_heldout")
    check({"rm_score_mean", "kl_token_weighted", "entropy_mean", "length"} <= set(e), "held-out eval summary")

    sys.argv = ["x", "--config", cfg, "--micro-batch", "2"]
    analyze_clipping.main()
    cb = json.load(open(R / "clipping_cached_batch.json"))
    check(set(cb["static"]) == {"0.05", "0.2", "0.5"} and len(cb["dynamic"]) == 6, "cached-batch study: 3 eps x 2 lr scales")
    check(cb["static"]["0.05"]["drift_ratio_outside_fraction"] >= cb["static"]["0.5"]["drift_ratio_outside_fraction"], "smaller eps => more tokens outside the clip range")
    for t in ("clip_0.05", "clip_0.20", "clip_0.50"):
        check((R / f"{t}_heldout_eval.json").exists() and (R / f"{t}_log.jsonl").exists(), f"fork {t} trained + evaluated")

    sys.argv = ["x", "--config", cfg]
    ablate_kl.main()
    for t in ("kl_0.00", "kl_0.10", "kl_0.20"):
        check((R / f"{t}_heldout_eval.json").exists(), f"fork {t} evaluated")
    summarize.main()
    check((R / "summary.csv").exists(), "Task 2 summary table written")

    # --- GRPO shares the generation path (same inference-tensor bug): short run for both loss types
    from task3_grpo.continue_train import run_grpo

    for lt in ("grpo", "dr_grpo"):
        run_grpo(gcfg, output=str(tmp / f"out/g_{lt}"), loss_type=lt, run_name=lt, grad_stats=True)
    gl = [json.loads(l) for l in open(tmp / "results_grpo/grpo_log.jsonl")]
    check(len(gl) == 2 and "uninformative_group_fraction" in gl[0], "GRPO loop runs (both normalisations)")
    print("\nALL TASK 2 / GRPO SMOKE CHECKS PASSED (tmp: %s)" % tmp)


if __name__ == "__main__":
    main()
