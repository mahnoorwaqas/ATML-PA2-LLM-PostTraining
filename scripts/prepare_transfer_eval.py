from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from common.data import REPO_ROOT

# Official SVAMP challenge-set source from the authors' repository.
SVAMP_URL = "https://raw.githubusercontent.com/arkilpatel/SVAMP/main/SVAMP.json"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "math_transfer_eval.jsonl"
DEFAULT_N = 100


def _answer_text(value) -> str:
    """Stable text representation for the exact numeric verifier."""
    x = float(value)
    if x.is_integer():
        return str(int(x))
    return format(x, ".12g")


def build_transfer_rows(records: list[dict], n: int = DEFAULT_N) -> list[dict]:
    if len(records) < n:
        raise ValueError(f"SVAMP source contains {len(records)} rows; need at least {n}")

    rows = []
    # Fixed course subset: first N entries in the official SVAMP challenge-set order.
    for rec in records[:n]:
        body = str(rec["Body"]).strip()
        question = str(rec["Question"]).strip()
        full_question = f"{body} {question}".strip()
        sid = str(rec["ID"])
        gold = _answer_text(rec["Answer"])
        rows.append(
            {
                "source_index": f"svamp:{sid}",
                "prompt_id": f"svamp:{sid}",
                "dataset": "SVAMP",
                "question": full_question,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            full_question
                            + "\n\nShow your reasoning and end your response with exactly "
                            + "`#### <number>`."
                        ),
                    }
                ],
                "gold_final": gold,
                "svamp_id": sid,
                "svamp_type": str(rec.get("Type", "")),
            }
        )
    return rows


def materialize_transfer_eval(
    output: Path = DEFAULT_OUTPUT,
    n: int = DEFAULT_N,
    force: bool = False,
    url: str = SVAMP_URL,
) -> Path:
    output = Path(output)
    if output.exists() and not force:
        return output

    print(f"Downloading official SVAMP challenge set from: {url}")
    with urllib.request.urlopen(url, timeout=60) as response:
        records = json.loads(response.read().decode("utf-8"))

    rows = build_transfer_rows(records, n=n)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(output)
    print(f"Wrote fixed {len(rows)}-example SVAMP transfer set: {output}")
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--n", type=int, default=DEFAULT_N)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--url", default=SVAMP_URL)
    args = ap.parse_args()
    materialize_transfer_eval(args.output, n=args.n, force=args.force, url=args.url)


if __name__ == "__main__":
    main()
