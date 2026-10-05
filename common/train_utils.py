"""Shared training helpers (optimizer stepping, memory/time tracking, dropout control)."""
from __future__ import annotations

import time

import torch
import torch.nn as nn


def disable_dropout(model: nn.Module) -> None:
    """Put every Dropout module in eval mode while the rest of the model stays in train mode.

    Used for PPO/GRPO updates so that old/new log-probs are computed without dropout noise
    (the importance ratio is exactly 1 at the first inner step). Gradient checkpointing, which
    is gated on model.training, stays active.
    """
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.eval()


class Stepper:
    """Backward / clip / step wrapper with optional fp16 loss scaling.

    LoRA parameters are fp32 (PEFT autocasts adapters when the base is fp16), so GradScaler
    is applicable. If scaling produces inf/nan grads the step is skipped and `skipped` is set.
    """

    def __init__(self, optimizer, params, max_grad_norm: float, use_scaler: bool | None = None):
        self.optimizer = optimizer
        self.params = [p for p in params if p.requires_grad]
        self.max_grad_norm = float(max_grad_norm)
        if use_scaler is None:
            use_scaler = torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler("cuda", init_scale=2.0**12) if use_scaler and torch.cuda.is_available() else None
        self.skipped = False

    def backward(self, loss: torch.Tensor) -> None:
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def step(self) -> float:
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        gn = torch.nn.utils.clip_grad_norm_(self.params, self.max_grad_norm)
        gn = float(gn.item()) if torch.is_tensor(gn) else float(gn)
        self.skipped = False
        if self.scaler is not None:
            before = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.skipped = self.scaler.get_scale() < before
        else:
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return gn


class ResourceTracker:
    """Wall-clock time and peak VRAM for a run."""

    def __init__(self):
        self.start = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def elapsed(self) -> float:
        return time.perf_counter() - self.start

    def summary(self) -> dict:
        out = {"wall_clock_seconds": self.elapsed()}
        if torch.cuda.is_available():
            out["peak_vram_allocated_gb"] = torch.cuda.max_memory_allocated() / 1024**3
            out["peak_vram_reserved_gb"] = torch.cuda.max_memory_reserved() / 1024**3
            out["gpu"] = torch.cuda.get_device_name(0)
        else:
            out["peak_vram_allocated_gb"] = None
        return out


@torch.no_grad()
def token_entropy(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean full-vocabulary token entropy over valid response tokens.

    logits: [B, T, V]; mask: [B, T]. Computed per sequence to bound memory.
    """
    total, count = 0.0, 0.0
    for b in range(logits.shape[0]):
        n = int(mask[b].sum().item())
        if n == 0:
            continue
        lp = torch.log_softmax(logits[b, :n].float(), dim=-1)
        ent = -(lp.exp() * lp).sum(-1)
        total += float(ent.sum().item())
        count += n
    return torch.tensor(total / max(count, 1.0))


def seq_stats(values: list[float]) -> dict:
    import numpy as np

    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "iqr": float("nan"), "median": float("nan")}
    q1, q3 = np.percentile(a, [25, 75])
    return {"mean": float(a.mean()), "std": float(a.std()), "iqr": float(q3 - q1), "median": float(np.median(a))}
