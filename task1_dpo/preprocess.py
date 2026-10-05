"""Overlength handling for DPO (assignment note on the 768-token limit).

Strategy used in this submission (state it in the report):
  * Prompt preserved, response truncated from the right (the released/patched
    common.data.encode_prompt_response), with EOS re-appended to a truncated response.
  * Examples whose PROMPT alone is >= max_length cannot be encoded and are FILTERED OUT.
  * The same rule is applied to every DPO training and evaluation file/run, so all conditions
    see identical example sets.
"""
from __future__ import annotations

from common.data import (
    preference_responses,
    prompt_messages_from_preference,
)


def _prompt_len(tokenizer, row) -> int:
    ids = tokenizer.apply_chat_template(
        prompt_messages_from_preference(row), tokenize=True, add_generation_prompt=True
    )
    if hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    return len(ids)


def fit_report(tokenizer, row, max_length: int) -> dict:
    p = _prompt_len(tokenizer, row)
    yc, yr = preference_responses(row)
    lc = len(tokenizer(yc, add_special_tokens=False)["input_ids"]) + 1  # +1 EOS
    lr = len(tokenizer(yr, add_special_tokens=False)["input_ids"]) + 1
    budget = max_length - p
    return {
        "prompt_tokens": p,
        "prompt_too_long": p >= max_length,
        "chosen_tokens": lc,
        "rejected_tokens": lr,
        "chosen_truncated": p < max_length and lc > budget,
        "rejected_truncated": p < max_length and lr > budget,
    }


def filter_fitting(tokenizer, rows: list[dict], max_length: int):
    """Drop rows whose prompt alone does not fit. Returns (kept_rows, n_dropped)."""
    kept = [r for r in rows if _prompt_len(tokenizer, r) < max_length]
    return kept, len(rows) - len(kept)
