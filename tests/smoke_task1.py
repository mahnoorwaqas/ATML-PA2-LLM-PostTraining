"""End-to-end CPU smoke test of Task 1 using a tiny random Qwen2 + char-level tokenizer + synthetic data.

No downloads. Exercises the REAL task1_dpo code paths (train, evaluate, beta ablation, length study,
summaries, truncation stats) with common.models loaders monkey-patched.
Run:  python -m tests.smoke_task1
"""
from __future__ import annotations

import json
import math
import random
import shutil
import sys
import tempfile
from pathlib import Path

import torch
import yaml

# ---------------------------------------------------------------- fakes (patched BEFORE task imports)
import common.models as cm
import common.generation as cg
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM

PAD, EOS, VOCAB = 256, 257, 260


class FakeTok:
    pad_token_id, eos_token_id = PAD, EOS
    padding_side = "left"

    def _tmpl(self, messages, add_generation_prompt=True):
        s = "".join(f"<{m['role'][0]}>{m['content']}</s>" for m in messages)
        return s + ("<a>" if add_generation_prompt else "")

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, **kw):
        s = self._tmpl(messages, add_generation_prompt)
        return [ord(c) % 256 for c in s] if tokenize else s

    def _ids(self, t):
        return [ord(c) % 256 for c in t]

    def __call__(self, text, return_tensors=None, padding=False, truncation=False, max_length=None, add_special_tokens=False, **kw):
        single = isinstance(text, str)
        seqs = [self._ids(text)] if single else [self._ids(t) for t in text]
        if truncation and max_length:
            seqs = [s[:max_length] for s in seqs]
        if return_tensors == "pt":
            w = max(len(s) for s in seqs)
            ids = torch.tensor([[PAD] * (w - len(s)) + s for s in seqs])
            att = torch.tensor([[0] * (w - len(s)) + [1] * len(s) for s in seqs])
            return {"input_ids": ids, "attention_mask": att}
        return {"input_ids": seqs[0] if single else seqs}

    def decode(self, ids, skip_special_tokens=True):
        ids = ids.tolist() if torch.is_tensor(ids) else ids
        return "".join(chr(i) for i in ids if i < 256 or not skip_special_tokens)


def tiny_base():
    torch.manual_seed(0)  # identical base weights on every call, like loading the same checkpoint
    cfg = Qwen2Config(vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048,
                      tie_word_embeddings=True, pad_token_id=PAD, eos_token_id=EOS)
    return Qwen2ForCausalLM(cfg)


def fake_load_policy(cfg, adapter_path=None, trainable=False, fresh_lora=False):
    base = tiny_base()
    if adapter_path:
        model = PeftModel.from_pretrained(base, str(Path(adapter_path)), is_trainable=trainable)
    elif fresh_lora:
        torch.manual_seed(1)
        model = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.05, target_modules=["q_proj", "v_proj"]))
    else:
        model = base
    model.train() if trainable else model.eval()
    return model


cm.load_tokenizer = lambda *a, **k: FakeTok()
cm.load_policy = fake_load_policy
cm.load_reward_model = lambda cfg: (None, None)
cg.score_reward_pairs = lambda rm, rt, prompts, responses, max_length=0: torch.tensor([0.01 * len(r) for r in responses])

from common.data import write_jsonl  # noqa: E402
from task1_dpo import ablate_beta, analyze_length, summarize, truncation_stats  # noqa: E402
from task1_dpo.evaluate import evaluate_adapter  # noqa: E402
from task1_dpo.train import run_training  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def pair(i, chosen_n, rejected_n, prompt=None):
    prompt = prompt or f"Question {i}?"
    return {"pair_id": i,
            "chosen": [{"role": "user", "content": prompt}, {"role": "assistant", "content": "yes " * chosen_n}],
            "rejected": [{"role": "user", "content": prompt}, {"role": "assistant", "content": "no " * rejected_n}]}


def make_data(d: Path):
    rnd = random.Random(0)
    std_train = [pair(i, rnd.randint(2, 12), rnd.randint(2, 12)) for i in range(40)]
    std_train[3] = pair(3, 5, 5, prompt="x" * 300)          # prompt alone > max_length -> must be filtered
    std_train[7] = pair(7, 60, 4)                            # chosen response truncated at 128 tokens
    std_eval = [pair(100 + i, rnd.randint(2, 12), rnd.randint(2, 12)) for i in range(12)]
    len_train = [pair(200 + i, c, r) for i, (c, r) in enumerate([(12, 3)] * 4 + [(6, 6)] * 4 + [(3, 12)] * 4)]
    len_eval = [pair(300 + i, c, r) for i, (c, r) in enumerate([(12, 3)] * 6 + [(6, 6)] * 6 + [(3, 12)] * 6)]
    for name, rows in dict(dpo_standard_train=std_train, dpo_standard_eval=std_eval, dpo_length_balanced_train=len_train,
                           dpo_length_stratified_eval=len_eval).items():
        write_jsonl(d / f"{name}.jsonl", rows)
    shutil.copy(REPO / "data/word_limit_prompts.jsonl", d / "word_limit_prompts.jsonl")


def make_cfg(tmp: Path) -> str:
    from common.data import _deep_merge, load_yaml

    over = {
        "learning_rate": 3.0e-3, "batch_size": 2, "grad_accum_steps": 2, "epochs": 1,
        "short_ablation_examples": 8, "max_sequence_length": 128, "max_generation_tokens": 12,
        "eval_generation_prompts": 6, "eval_batch_size": 3, "word_limit_samples": 2,
        "standard_output": str(tmp / "out/standard"), "length_output": str(tmp / "out/length_balanced"),
        "beta_output_dir": str(tmp / "out"), "results_dir": str(tmp / "results"),
        "paths": {"dpo_standard_train": str(tmp / "data/dpo_standard_train.jsonl"),
                  "dpo_standard_eval": str(tmp / "data/dpo_standard_eval.jsonl"),
                  "dpo_length_train": str(tmp / "data/dpo_length_balanced_train.jsonl"),
                  "dpo_length_eval": str(tmp / "data/dpo_length_stratified_eval.jsonl"),
                  "word_limit_prompts": str(tmp / "data/word_limit_prompts.jsonl")},
    }
    cfg = _deep_merge(load_yaml("configs/dpo.yaml"), over)  # real merged config + tiny overrides
    p = tmp / "dpo_smoke.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return str(p)


def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        raise SystemExit(f"smoke test failed: {msg}")


def main():
    tmp = Path(tempfile.mkdtemp(prefix="dpo_smoke_"))
    (tmp / "data").mkdir()
    make_data(tmp / "data")
    cfg = make_cfg(tmp)
    R = tmp / "results"
    J = lambda p: json.load(open(R / p))

    # 0. truncation stats
    sys.argv = ["x", "--config", cfg]
    truncation_stats.main()
    ts = J("truncation_stats.json")
    check(ts["dpo_standard_train"]["prompt_too_long_filtered"] == 1, "overlength: 1 prompt-too-long pair detected")
    check(ts["dpo_standard_train"]["chosen_truncated"] >= 1, "overlength: truncated chosen response detected")

    # 1. standard training (full 'epoch') -> sign/plumbing check: step-1 loss must equal log 2
    s = run_training(cfg, "standard", output_path=str(tmp / "out/standard"))
    log = [json.loads(l) for l in open(R / "standard_train_log.jsonl")]
    check(abs(log[0]["loss"] - math.log(2)) < 1e-3, f"step-1 loss == log 2 (policy==reference): {log[0]['loss']:.4f}")
    check(log[-1]["loss"] < log[0]["loss"], f"training loss decreases: {log[0]['loss']:.3f} -> {log[-1]['loss']:.3f}")
    check(s["n_examples"] == 39 and len(s["example_ids"]) == 39, "filtered example set recorded (39 ids)")
    check((tmp / "out/standard/adapter_config.json").exists(), "adapter saved")

    # 2. evaluations
    evaluate_adapter(cfg, None, "sft_base")
    b = J("sft_base_eval.json")
    check(b["heldout"] is None and "generation" in b and "word_limit" in b, "sft_base: generation + word-limit, no pair metrics")
    evaluate_adapter(cfg, str(tmp / "out/standard"), "standard", with_strata=True)
    e = J("standard_eval.json")
    check(e["heldout"]["pref_accuracy"] > 0.5 and e["heldout"]["margin_mean"] > 0, f"held-out pref acc > 0.5: {e['heldout']['pref_accuracy']:.2f}")
    check(set(e["length_stratified"]) >= {"preferred_longer", "matched", "rejected_longer"}, "three length strata reported")
    check(all(e["length_stratified"][k]["n_pairs"] == 6 for k in ("preferred_longer", "matched", "rejected_longer")), "strata sizes 6/6/6")
    for k in ("rm_score_mean", "kl_token_weighted", "length", "entropy_mean", "prompt_ids"):
        check(k in e["generation"], f"generation summary has {k}")
    check({"compliance_rate", "words"} <= set(e["word_limit"]), "word-limit compliance + word stats")
    check(abs(e["generation"]["kl_token_weighted"]) > 0, "KL to reference nonzero after training")

    # 3. beta sweep
    sys.argv = ["x", "--config", cfg]
    ablate_beta.main()
    for bt in ("0.03", "0.10", "0.30"):
        r = J(f"beta_{bt}_eval.json")
        check(r["heldout"] is not None and "generation" in r and "word_limit" not in r and r["n_heldout_pairs"] == 12, f"beta {bt}: eval complete")
        check(json.load(open(R / f"beta_{bt}_train_summary.json"))["n_examples"] == 8, f"beta {bt}: 8-example short budget")

    # 4. length study
    sys.argv = ["x", "--config", cfg]
    analyze_length.main()
    for n in ("standard", "length_balanced"):
        r = J(f"{n}_eval.json")
        check({"preferred_longer", "matched", "rejected_longer"} <= set(r["length_stratified"]) and "word_limit" in r, f"{n}: strata + word-limit")
    ds = J("dataset_length_stats.json")
    check(ds["length_balanced_train"]["strata_counts"]["matched"] == 4, "dataset stratum counts computed")

    # 5. summaries
    sys.argv = ["x", "--config", cfg]
    summarize.main()
    import pandas as pd
    df = pd.read_csv(R / "summary.csv")
    check(len(df) == 6, f"summary has 6 conditions: {list(df['condition'])}")
    need = {"dpo_loss", "pref_acc", "kl_token_weighted", "rm_score", "len_mean", "len_std", "wordlimit_compliance", "budget"}
    check(need <= set(df.columns), "summary has all required columns")
    check(set(df["budget"]) >= {"full epoch", "short (8 ex.)"}, "budgets labelled (full epoch vs short)")
    for f in ("length_strata_accuracy.csv", "beta_monotonicity.json", "wordlimit_evidence.json", "qualitative_candidates.json", "dataset_length_stats.json"):
        check((R / f).exists(), f"artifact {f}")
    print("\nALL TASK 1 SMOKE CHECKS PASSED  (tmp dir: %s)" % tmp)


if __name__ == "__main__":
    main()
