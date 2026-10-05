from __future__ import annotations

import torch
import torch.nn.functional as F


def dpo_loss(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float,
):
    """Return scalar DPO loss plus lightweight diagnostics.

    L = -E[ log sigmoid( beta * ( [log pi(y+|x) - log ref(y+|x)] - [log pi(y-|x) - log ref(y-|x)] ) ) ]

    All inputs are sequence log-probabilities summed over response tokens only.

    FIX (planted defect): the starter computed `beta * (policy_margin + ref_margin)`.
    The reference margin must be SUBTRACTED. The implicit-reward margin is
    (policy chosen - policy rejected) - (ref chosen - ref rejected).
    """
    policy_margin = policy_chosen_logp - policy_rejected_logp
    ref_margin = ref_chosen_logp - ref_rejected_logp

    # m_theta = [log pi/ref (y+)] - [log pi/ref (y-)] = policy_margin - ref_margin
    margin = policy_margin - ref_margin
    logits = beta * margin

    loss = -F.logsigmoid(logits).mean()

    return loss, {
        "logit_mean": logits.detach().mean(),
        "policy_margin_mean": policy_margin.detach().mean(),
        "margin_mean": margin.detach().mean(),
        "preference_accuracy": (margin > 0).float().mean().detach(),
    }
