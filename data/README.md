# Course data

Most fixed course data are installed by:

```bash
python -m scripts.download_assets
```

The repository itself tracks only the two tiny fixed prompt/evaluation files that are not part of the large v3 post-training archive:

- `word_limit_prompts.jsonl` — 10 common prompts used in Task 1's explicit word-limit compliance analysis.
- `math_transfer_eval.jsonl` — fixed 100-example SVAMP transfer subset used in Task 5. The asset installer creates this deterministically from the first 100 examples in the official SVAMP challenge-set order if the file is absent.

The large release archive supplies:

- `dpo_standard_train.jsonl`
- `dpo_standard_eval.jsonl`
- `dpo_length_balanced_train.jsonl`
- `dpo_length_stratified_eval.jsonl`
- `rl_prompt_pool_train.jsonl`
- `rl_prompt_pool_eval.jsonl`
- `xstest_safety_prompts.csv`
- `gsm8k_rl_train.jsonl`
- `gsm8k_eval.jsonl`
- `task5_controlled_reward_diagnostics.jsonl`

Do not edit course-provided data. Save student-generated results under `results/` or `outputs/` instead.
