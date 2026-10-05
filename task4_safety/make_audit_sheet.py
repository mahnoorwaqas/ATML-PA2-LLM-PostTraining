from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path


def fixed_audit_ids(base_rows, per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    meta = pd.DataFrame(base_rows)
    ids = []
    for label in ["SAFE", "UNSAFE"]:
        pool = meta.loc[meta["benchmark_class"] == label, "xstest_id"].to_numpy()
        if len(pool) < per_class:
            raise ValueError(f"Not enough {label} rows for audit")
        ids.extend(rng.choice(pool, size=per_class, replace=False).tolist())
    return sorted(int(x) for x in ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--policies", nargs="*", default=["sft", "dpo", "ppo", "grpo"],
                    help="policies to include in the blind labelling sheet")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    src = outdir / "generated_sft.jsonl"
    if not src.exists():
        raise FileNotFoundError("Generate/save SFT responses first: " + str(src))
    ids = fixed_audit_ids(read_jsonl(src), int(cfg["manual_audit_per_class"]), int(cfg["seed"]))
    pd.DataFrame({"xstest_id": ids, "manual_label": [""] * len(ids)}).to_csv(outdir / "manual_audit_ids.csv", index=False)
    print("Wrote fixed audit IDs:", outdir / "manual_audit_ids.csv")

    # Blind labelling sheet: prompt + response only. No AI label, no benchmark_class/type, shuffled with a
    # fixed seed so policies are interleaved. Fill `manual_label` with one of
    # SAFE_ANSWER / JUSTIFIED_REFUSAL / UNSAFE_COMPLIANCE / OVER_REFUSAL / AMBIGUOUS and save as
    # manual_audit_labels.csv (same columns).
    rows = []
    for pol in args.policies:
        gp = outdir / f"generated_{pol}.jsonl"
        if not gp.exists():
            print(f"[skip] {gp} missing")
            continue
        for r in read_jsonl(gp):
            if int(r["xstest_id"]) in set(ids):
                rows.append({"xstest_id": int(r["xstest_id"]), "policy": pol, "prompt": r["prompt"], "response": r["response"], "manual_label": ""})
    sheet = pd.DataFrame(rows)
    if len(sheet):
        sheet = sheet.sample(frac=1.0, random_state=int(cfg["seed"])).reset_index(drop=True)
        sheet.insert(0, "row_id", range(len(sheet)))
        sheet.to_csv(outdir / "manual_audit_sheet.csv", index=False)
        print(f"Wrote blind sheet with {len(sheet)} rows -> manual_audit_sheet.csv")
        print("Label WITHOUT opening judged_*.jsonl, then save as manual_audit_labels.csv")


if __name__ == "__main__":
    main()
