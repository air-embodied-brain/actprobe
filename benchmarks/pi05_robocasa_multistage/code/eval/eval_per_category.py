"""
Per-category q-AUC table — paper Table 2-style.

Reads main_table.json (already has per_cat field per seed), aggregates across seeds.
If main_table.json doesn't exist or is stale, run eval_main_table.py first.

Usage:
  python code/eval/eval_per_category.py \
      --in  results/frozen/main_table.json \
      --out results/frozen/per_category.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from lib.categories import CATEGORIES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/frozen/main_table.json")
    ap.add_argument("--out", default="results/frozen/per_category.json")
    args = ap.parse_args()

    with open(args.inp) as f:
        mt = json.load(f)

    out = {"q": mt["q"], "categories": list(CATEGORIES.keys()),
           "results": {}}

    for method, by_split in mt["results"].items():
        out["results"][method] = {}
        for split, by_seed in by_split.items():
            seeds = {k: v for k, v in by_seed.items() if k.startswith("seed")}
            if not seeds:
                continue
            cat_vals = defaultdict(list)
            for sk, srec in seeds.items():
                pc = srec.get("per_cat", {})
                for cat, v in pc.items():
                    cat_vals[cat].append(v)
            agg = {}
            for cat, vals in cat_vals.items():
                agg[cat] = {
                    "mean": round(float(np.mean(vals)), 4),
                    "std":  round(float(np.std(vals)),  4),
                    "n_seeds": len(vals),
                }
            out["results"][method][split] = agg

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {args.out}")

    # Print table
    cats = list(CATEGORIES.keys()) + ["Overall"]
    print(f"\n{'='*100}")
    print(f"Per-category q={mt['q']} AUC (mean ± std over seeds)")
    print(f"{'='*100}")
    for split in ["allseen", "unseen"]:
        print(f"\n[{split}]")
        header = f"{'method':<18}  " + "  ".join(f"{c:>16}" for c in cats)
        print(header)
        print("-" * len(header))
        for method, by_split in out["results"].items():
            agg = by_split.get(split)
            if not agg:
                continue
            row = f"{method:<18}  "
            for cat in cats:
                if cat in agg:
                    row += f"{agg[cat]['mean']:>7.2f}±{agg[cat]['std']:.2f}  "
                else:
                    row += f"{'--':>16}  "
            print(row)


if __name__ == "__main__":
    main()
