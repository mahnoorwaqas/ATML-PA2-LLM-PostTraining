"""Unit tests validating the three corrected objectives against the equations in the assignment.
Run: python -m pytest tests -q   (CPU only, no model downloads)."""
import math

import torch
import torch.nn.functional as F

from task1_dpo.dpo import dpo_loss
from task2_ppo.ppo import compute_gae, ppo_affected_fraction, ppo_policy_loss
from task3_grpo.grpo import group_relative_advantages, grpo_policy_loss


# ---------------- Task 1: DPO ----------------
def test_dpo_zero_margin_at_reference():
    c, r = torch.tensor([-10.0, -20.0]), torch.tensor([-12.0, -25.0])
    loss, d = dpo_loss(c, r, c.clone(), r.clone(), beta=0.1)
    assert abs(loss.item() - math.log(2)) < 1e-6          # policy == reference -> log 2
    assert d["margin_mean"].item() == 0.0


def test_dpo_matches_closed_form_and_sign():
    pc, pr, rc, rr = torch.tensor([-5.0]), torch.tensor([-9.0]), torch.tensor([-8.0]), torch.tensor([-8.5])
    beta = 0.3
    margin = (pc - rc) - (pr - rr)                         # = 3 - (-0.5) = 3.5
    expected = -F.logsigmoid(beta * margin).mean()
    loss, d = dpo_loss(pc, pr, rc, rr, beta)
    assert torch.allclose(loss, expected)
    assert d["preference_accuracy"].item() == 1.0


def test_dpo_moving_toward_chosen_lowers_loss():
    rc, rr = torch.tensor([-8.0]), torch.tensor([-8.0])
    l_before, _ = dpo_loss(torch.tensor([-8.0]), torch.tensor([-8.0]), rc, rr, 0.1)
    l_after, _ = dpo_loss(torch.tensor([-6.0]), torch.tensor([-10.0]), rc, rr, 0.1)
    assert l_after < l_before


def test_dpo_reference_margin_is_subtracted():
    # same policy margin, larger reference margin -> LOWER implicit reward -> HIGHER loss
    pc, pr = torch.tensor([-5.0]), torch.tensor([-9.0])
    l_small, _ = dpo_loss(pc, pr, torch.tensor([-8.0]), torch.tensor([-8.0]), 0.1)
    l_big, _ = dpo_loss(pc, pr, torch.tensor([-5.0]), torch.tensor([-9.0]), 0.1)
    assert l_big > l_small


# ---------------- Task 2: PPO ----------------
def test_ppo_clip_is_pessimistic_min():
    mask = torch.ones(1, 1)
    adv = torch.tensor([[1.0]])
    old = torch.zeros(1, 1)
    new = torch.log(torch.tensor([[1.5]]))                 # ratio 1.5 > 1+eps
    loss, ratio, cf = ppo_policy_loss(new, old, adv, mask, eps=0.2)
    assert abs(loss.item() - (-1.2)) < 1e-5                # min(1.5, 1.2) = 1.2
    assert cf.item() == 1.0
    adv_n = torch.tensor([[-1.0]])
    loss2, _, _ = ppo_policy_loss(new, old, adv_n, mask, eps=0.2)
    assert abs(loss2.item() - 1.5) < 1e-5                  # A<0: min(-1.5, -1.2) = -1.5 (no clipping benefit)


def test_ppo_clipped_tokens_have_zero_gradient():
    new = torch.log(torch.tensor([[1.5]])).requires_grad_(True)
    loss, _, _ = ppo_policy_loss(new, torch.zeros(1, 1), torch.tensor([[1.0]]), torch.ones(1, 1), eps=0.2)
    loss.backward()
    assert new.grad.abs().item() == 0.0                    # binding clip => no gradient


def test_ppo_affected_fraction_is_binding_only():
    mask = torch.ones(1, 2)
    new = torch.log(torch.tensor([[1.5, 1.5]]))
    adv = torch.tensor([[1.0, -1.0]])                      # first binds, second does not
    f = ppo_affected_fraction(new, torch.zeros(1, 2), adv, mask, 0.2)
    assert abs(f.item() - 0.5) < 1e-6


def test_gae_single_step_and_terminal():
    r = torch.tensor([[0.0, 0.0, 1.0]])
    v = torch.tensor([[0.2, 0.4, 0.6]])
    m = torch.ones(1, 3)
    adv, ret = compute_gae(r, v, m, gamma=1.0, lam=1.0)
    # lambda=1, gamma=1 -> A_t = (sum future rewards) - V_t
    assert torch.allclose(adv, torch.tensor([[1.0 - 0.2, 1.0 - 0.4, 1.0 - 0.6]]), atol=1e-6)
    assert torch.allclose(ret, torch.ones(1, 3), atol=1e-6)


def test_gae_respects_padding():
    r = torch.tensor([[0.0, 1.0, 0.0]])
    v = torch.tensor([[0.0, 0.0, 5.0]])                    # garbage value at a padded position
    m = torch.tensor([[1.0, 1.0, 0.0]])
    adv, _ = compute_gae(r, v, m, 1.0, 0.95)
    assert adv[0, 2].item() == 0.0 and abs(adv[0, 1].item() - 1.0) < 1e-6


# ---------------- Task 3: GRPO ----------------
def test_group_advantages_are_per_group():
    r = torch.tensor([1.0, 0.0, 1.0, 0.0, 10.0, 10.0, 10.0, 10.0])
    g = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    a = group_relative_advantages(r, g)
    assert torch.allclose(a[:4], torch.tensor([1.0, -1.0, 1.0, -1.0]), atol=1e-4)
    assert torch.allclose(a[4:], torch.zeros(4), atol=1e-6)   # zero-variance group -> zero signal
    for gid in (0, 1):
        assert abs(a[g == gid].mean().item()) < 1e-6          # zero mean inside every group


def test_group_advantage_not_global():
    r = torch.tensor([0.0, 1.0, 100.0, 101.0])
    g = torch.tensor([0, 0, 1, 1])
    a = group_relative_advantages(r, g)
    assert torch.allclose(a, torch.tensor([-1.0, 1.0, -1.0, 1.0]), atol=1e-4)


def _loss(kind, L_short=2, L_long=8):
    T = L_long
    new = torch.zeros(2, T)
    mask = torch.zeros(2, T)
    mask[0, :L_short] = 1
    mask[1, :L_long] = 1
    adv = torch.tensor([1.0, 1.0])
    loss, d = grpo_policy_loss(new, new, adv, mask, new, 0.2, 0.0, loss_type=kind, max_completion_length=16)
    return loss.item(), d


def test_canonical_vs_dr_grpo_normalisation():
    lc, _ = _loss("grpo")        # each sequence contributes mean over its own tokens -> equal weight
    ld, _ = _loss("dr_grpo")     # constant 1/L_max -> longer sequence contributes more
    assert abs(lc - (-1.0)) < 1e-6
    assert abs(ld - (-(2 / 16 + 8 / 16) / 2)) < 1e-6


def test_masked_truncated_completion_contributes_nothing():
    new = torch.zeros(2, 4, requires_grad=True)
    mask = torch.tensor([[1.0, 1, 1, 1], [0.0, 0, 0, 0]])
    loss, _ = grpo_policy_loss(new, new.detach(), torch.tensor([1.0, 5.0]), mask, new.detach(), 0.2, 0.0)
    loss.backward()
    assert new.grad[1].abs().sum().item() == 0.0
