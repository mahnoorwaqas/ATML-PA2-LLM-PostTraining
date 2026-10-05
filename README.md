# ATML PA2 — LLM Post-Training (DPO, PPO, GRPO, Safety, RLVR vs RLAIF)

Individual submission for EE-5102/CS-6304 PA2. Report: `report/` (written by hand). Seed: 6304 (`configs/base.yaml`).

**Attribution.** Starter infrastructure (`common/`, `configs/`, `scripts/`, objective scaffolds, fixed judges/verifier)
comes from the course repo <https://github.com/AbDu11aHHH/ATML-PA2-LLM-PostTraining>. Everything listed under
"Implemented here" was written for this submission (with LLM coding assistance; every line reviewed). No other external code is reused.

## Objective fixes (one planted defect per core objective)
| Task | File | Defect | Fix |
|---|---|---|---|
| 1 | `task1_dpo/dpo.py` | `beta*(policy_margin + ref_margin)` | `beta*(policy_margin - ref_margin)` |
| 2 | `task2_ppo/ppo.py` | `torch.maximum(surr1, surr2)` | `torch.minimum(...)` (pessimistic clipped surrogate) |
| 3 | `task3_grpo/grpo.py` | advantages normalised over the whole batch | mean/std computed **within each prompt group** |

Validated by `python -m pytest tests -q` (CPU, 13 tests: closed-form DPO, clip geometry + zero gradient when binding, GAE, per-group advantages, 1/T vs 1/L_max normalisation, masked completions).

## Setup
```bash
python -m pip install -r requirements.txt
python -m scripts.download_assets && python -m scripts.validate_assets
python -m pytest tests -q
```

## Reproduce (run in this order; Task 4 only after 1–3 standard runs are frozen)
```bash
# ---- Task 1: DPO ----
python -m task1_dpo.truncation_stats                                 # how many pairs hit the 768-token limit (cite in report)
python -m task1_dpo.train --run-name standard                       # 1 epoch, beta=0.10 -> outputs/task1_dpo/standard
python -m task1_dpo.evaluate --adapter none --name sft_base --skip-word-limit   # base-policy generation reference
python -m task1_dpo.evaluate --adapter outputs/task1_dpo/standard --name standard --strata
python -m task1_dpo.ablate_beta                                      # betas 0.03/0.10/0.30, 600 examples each
python -m task1_dpo.analyze_length                                   # trains length_balanced, stratified + word-limit eval
python -m task1_dpo.summarize

# ---- Task 2: PPO ----
python -m task2_ppo.analyze_clipping --inspect                       # print cache schema first
python -m task2_ppo.continue_train --run-name standard               # 20 updates -> outputs/task2_ppo/standard
python -m task2_ppo.evaluate --adapter outputs/task2_ppo/standard --name standard
python -m task2_ppo.evaluate --adapter none --name base
python -m task2_ppo.analyze_clipping                                 # cached-batch study + eps forks (8 updates each)
python -m task2_ppo.ablate_kl                                        # beta_KL 0 / 0.10 / 0.20 forks
python -m task2_ppo.summarize

# ---- Task 3: GRPO ----
python -m task3_grpo.continue_train --run-name standard              # 20 updates -> outputs/task3_grpo/standard
python -m task3_grpo.evaluate --adapter outputs/task3_grpo/standard --name standard
python -m task3_grpo.analyze_group_size                              # K in {2,4,8}, no training
python -m task3_grpo.compare_normalization                           # canonical vs Dr. GRPO forks

# ---- Task 4: safety (frozen standard policies only) ----
python -m task4_safety.generate_responses
python -m task4_safety.make_audit_sheet                              # blind sheet -> label by hand BEFORE judging is read
python -m task4_safety.judge_responses
python -m task4_safety.evaluate_safety                               # re-run after saving manual_audit_labels.csv

# ---- Task 5: RLVR vs RLAIF ----
python -m task5_feedback.evaluate_math --dataset gsm
python -m task5_feedback.evaluate_math --dataset transfer
python -m task5_feedback.score_perturbations
python -m task5_feedback.compare_feedback
```
All outputs are JSON/CSV/JSONL under `results/`; adapters under `outputs/` (git-ignored).

## Implemented here
- `common/train_utils.py` (optimizer stepper, VRAM/time tracking, entropy), `common/policy_eval.py` (shared held-out generation/RM/KL evaluation)
- Task 1: `train.py`, `evaluate.py`, `ablate_beta.py`, `analyze_length.py`, `summarize.py`
- Task 2: `continue_train.py`, `evaluate.py`, `analyze_clipping.py`, `ablate_kl.py`, `summarize.py`
- Task 3: `continue_train.py`, `evaluate.py`, `analyze_group_size.py`, `compare_normalization.py`
- Task 4: `generate_responses.py`, `judge_responses.py` (main only), `make_audit_sheet.py` (blind sheet), `evaluate_safety.py`
- Task 5: `evaluate_math.py`, `score_perturbations.py`, `compare_feedback.py`

## Protocol decisions (cite in the report)
- **KL**: sampled estimator, response-token log-ratio vs the frozen base (adapter disabled); report `kl_token_weighted` (sum of log-ratios / total tokens) everywhere.
- **Group-size study**: first 8 cached completions/prompt, split into consecutive blocks of K (equal total generations); difficulty = terciles of per-prompt mean cached reward (hard/medium/easy), defined once.
- **Normalisation stat**: per-completion policy-term gradient norm split at the pooled median length, plus the effective per-token weight (1/T_k vs 1/L_max).
- **Fork budgets** are 8 updates (`fork_updates`), standard continuations 20; DPO short forks use the first 600 pairs.
- Held-out generation uses the first `eval_prompts` prompts of the fixed eval files (config keys added to the YAMLs); changing them changes every condition.

## DPO overlength handling (768 tokens)
Prompt preserved, response truncated from the right with EOS re-appended (the patched `common.data.encode_prompt_response`).
Pairs whose prompt alone is >= 768 tokens are **filtered** (`task1_dpo/preprocess.py`), identically for all DPO training and
evaluation sets. Counts per dataset/stratum: `results/task1_dpo/truncation_stats.json`. Truncation caps long responses, so it
compresses length differences between chosen/rejected on long pairs; discuss this when interpreting the length experiments.
