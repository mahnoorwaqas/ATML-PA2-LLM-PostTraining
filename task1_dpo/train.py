from __future__ import annotations

import argparse
import math

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.generation import response_sequence_logprobs
from common.logging_utils import append_jsonl, save_json, set_seed
from common.models import load_policy, load_tokenizer, reference_mode, trainable_parameters
from common.train_utils import ResourceTracker, Stepper
from task1_dpo.dpo import dpo_loss
from task1_dpo.preprocess import filter_fitting


def make_collate(tokenizer, max_length):
    def collate(rows):
        chosen, rejected = [], []
        for row in rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            chosen.append(encode_prompt_response(tokenizer, prompt, yc, max_length))
            rejected.append(encode_prompt_response(tokenizer, prompt, yr, max_length))
        return pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected)
    return collate


def prepare_dpo_run(config_path: str, dataset_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    path = dataset_path or cfg["paths"]["dpo_standard_train"]
    rows = read_jsonl(path)
    tokenizer = load_tokenizer(cfg["base_model"])
    # Same overlength rule everywhere: drop examples whose prompt alone exceeds max_sequence_length.
    rows, n_dropped = filter_fitting(tokenizer, rows, int(cfg["max_sequence_length"]))
    print(f"[dpo] dropped {n_dropped} examples with prompt >= {cfg['max_sequence_length']} tokens ({path})")
    if max_examples is not None:
        rows = rows[: int(max_examples)]
    model = load_policy(cfg, trainable=True, fresh_lora=True)
    # Fixed shuffling seed so every condition sees the same example order (data-order matched).
    gen = torch.Generator().manual_seed(int(cfg["seed"]))
    loader = DataLoader(
        rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        generator=gen,
        collate_fn=make_collate(tokenizer, int(cfg["max_sequence_length"])),
    )
    optimizer = AdamW(
        trainable_parameters(model),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    return {
        "cfg": cfg,
        "rows": rows,
        "tokenizer": tokenizer,
        "model": model,
        "loader": loader,
        "optimizer": optimizer,
        "beta": float(cfg["beta"] if beta is None else beta),
    }


def _to_device(batch: dict, device):
    return {k: v.to(device) for k, v in batch.items()}


def run_training(config_path: str, run_name: str, dataset_path: str | None = None, output_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    bundle = prepare_dpo_run(config_path, dataset_path, beta, max_examples)
    cfg = bundle["cfg"]
    model, loader, optimizer, beta_val = bundle["model"], bundle["loader"], bundle["optimizer"], bundle["beta"]
    output = repo_path(output_path or cfg["standard_output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    results_dir = repo_path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / f"{run_name}_train_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    accum = int(cfg["grad_accum_steps"])
    epochs = int(cfg["epochs"])
    device = next(model.parameters()).device
    stepper = Stepper(optimizer, trainable_parameters(model), float(cfg["max_grad_norm"]))
    tracker = ResourceTracker()

    n_micro_total = epochs * len(loader)
    n_opt_steps = math.ceil(n_micro_total / accum)
    print(f"[dpo:{run_name}] examples={len(bundle['rows'])} beta={beta_val} micro-batches={n_micro_total} optimizer-steps={n_opt_steps}")

    model.train()
    micro, opt_step = 0, 0
    acc_stats = {"loss": 0.0, "acc": 0.0, "margin": 0.0, "n": 0}
    for epoch in range(epochs):
        for chosen, rejected in loader:
            chosen, rejected = _to_device(chosen, device), _to_device(rejected, device)

            # Reference = frozen base model (adapter disabled). No grad.
            with torch.no_grad(), reference_mode(model):
                ref_c, _, _ = response_sequence_logprobs(model, chosen)
                ref_r, _, _ = response_sequence_logprobs(model, rejected)
            model.train()

            pol_c, _, _ = response_sequence_logprobs(model, chosen)
            pol_r, _, _ = response_sequence_logprobs(model, rejected)
            loss, diag = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta_val)

            stepper.backward(loss / accum)
            micro += 1
            acc_stats["loss"] += float(loss.detach().item())
            acc_stats["acc"] += float(diag["preference_accuracy"].item())
            acc_stats["margin"] += float(diag["margin_mean"].item())
            acc_stats["n"] += 1

            last = micro == n_micro_total
            if micro % accum == 0 or last:
                grad_norm = stepper.step()
                opt_step += 1
                n = max(acc_stats["n"], 1)
                rec = {
                    "step": opt_step,
                    "epoch": epoch,
                    "beta": beta_val,
                    "loss": acc_stats["loss"] / n,
                    "train_pref_acc": acc_stats["acc"] / n,
                    "margin_mean": acc_stats["margin"] / n,
                    "grad_norm": grad_norm,
                    "step_skipped": stepper.skipped,
                    "lr": optimizer.param_groups[0]["lr"],
                }
                append_jsonl(log_path, rec)
                if opt_step % 5 == 0 or last:
                    print(f"[dpo:{run_name}] step {opt_step}/{n_opt_steps} loss={rec['loss']:.4f} acc={rec['train_pref_acc']:.3f} gn={grad_norm:.2f}")
                acc_stats = {"loss": 0.0, "acc": 0.0, "margin": 0.0, "n": 0}

    model.save_pretrained(str(output))
    summary = {
        "run_name": run_name,
        "beta": beta_val,
        "dataset": dataset_path or cfg["paths"]["dpo_standard_train"],
        "n_examples": len(bundle["rows"]),
        "overlength_rule": "prompt preserved, response truncated (+EOS); prompt-too-long examples filtered",
        "epochs": epochs,
        "optimizer_steps": opt_step,
        "learning_rate": float(cfg["learning_rate"]),
        "seed": int(cfg["seed"]),
        "adapter_path": str(output),
        **tracker.summary(),
    }
    save_json(results_dir / f"{run_name}_train_summary.json", summary)
    print("saved adapter ->", output)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--output")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--max-examples", type=int)
    args = ap.parse_args()
    run_training(args.config, args.run_name, args.dataset, args.output, args.beta, args.max_examples)


if __name__ == "__main__":
    main()
