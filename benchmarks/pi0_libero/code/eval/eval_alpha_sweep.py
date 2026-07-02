"""
Alpha sweep — paper Fig 7 curve data (regeneration).

This script regenerates the alpha-sweep curve data used for the paper figure.

Usage:
  source env.sh
  python code/eval/eval_alpha_sweep.py --out results/alpha_sweep_<tag>.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, balanced_accuracy_score

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "eval"))

from lib import data as data_mod
from lib import metrics as M
from lib import splits as S
from eval_main_table import run_method, ALL_METHODS, DEVICE  # noqa


ALPHAS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
          0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
Q_FRAC = 0.25


def alpha_sweep_for_run(test_r, val_r, task_min, alphas=ALPHAS, q=Q_FRAC):
    """Compute (alpha → metrics dict) for one (method, split, seed) run."""
    val_succ_q = []
    for r in val_r:
        if r["label"] != 0:
            continue
        ms = task_min.get(r["task_id"], r["length"])
        cutoff = M.task_min_cutoff(ms, q, r["length"])
        val_succ_q.append(M.quantile_max(r["scores"], r["length"], cutoff))

    test_q, test_labels = [], []
    test_records = []
    for r in test_r:
        ms = task_min.get(r["task_id"], r["length"])
        cutoff = M.task_min_cutoff(ms, q, r["length"])
        test_q.append(M.quantile_max(r["scores"], r["length"], cutoff))
        test_labels.append(r["label"])
        test_records.append({"task_id": r["task_id"], "length": r["length"],
                             "label": r["label"], "scores": r["scores"]})
    test_q = np.asarray(test_q); test_labels = np.asarray(test_labels)

    if len(val_succ_q) == 0 or len(set(test_labels)) < 2:
        return {f"{a:.2f}": None for a in alphas}

    out = {}
    for alpha in alphas:
        tau = M.fcp_tau(np.asarray(val_succ_q), alpha)
        preds = (test_q >= tau).astype(int)
        f1 = (f1_score(test_labels, preds) * 100
              if (preds.sum() + test_labels.sum()) > 0 else 0.0)
        bacc = balanced_accuracy_score(test_labels, preds) * 100
        tpr = (preds & (test_labels == 1)).sum() / max((test_labels == 1).sum(), 1) * 100
        fpr = (preds & (test_labels == 0)).sum() / max((test_labels == 0).sum(), 1) * 100

        det_metrics = M.detection_metrics(test_records, tau)
        out[f"{alpha:.2f}"] = {
            "tau":     round(float(tau), 6),
            "f1":      round(f1, 4),
            "bal_acc": round(bacc, 4),
            "tpr":     round(float(tpr), 4),
            "fpr":     round(float(fpr), 4),
            "tdet":    round(det_metrics["tdet"], 4) if not np.isnan(det_metrics["tdet"]) else None,
            "det_rate":round(det_metrics["det_rate"], 4) if not np.isnan(det_metrics["det_rate"]) else None,
            "n_det":   det_metrics["n_det"],
            "n_fail":  det_metrics["n_fail"],
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=ALL_METHODS)
    ap.add_argument("--splits",  nargs="+", default=["allseen", "unseen"])
    ap.add_argument("--seeds",   nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--q",       type=float, default=Q_FRAC)
    ap.add_argument("--out",     type=str,  required=True)
    args = ap.parse_args()

    print(f"Methods: {args.methods}")
    print(f"Splits:  {args.splits}")
    print(f"Seeds:   {args.seeds}")
    print(f"Alphas:  {ALPHAS}")

    print("\nLoading data...")
    eps_metrics = data_mod.load_metrics_logs()
    eps_hs      = data_mod.load_hs()
    task_embs   = data_mod.load_task_embeddings()
    print(f"  metrics: {len(eps_metrics)} tasks")
    print(f"  hs:      {len(eps_hs)} tasks")

    out = {"methods": args.methods, "splits": args.splits, "seeds": args.seeds,
           "alphas": ALPHAS, "q": args.q,
           "results": {}}

    for method in args.methods:
        out["results"][method] = {}
        for split in args.splits:
            out["results"][method][split] = {}
            for seed in args.seeds:
                tag = f"{method}/{split}/seed{seed}"
                t0 = time.time()
                try:
                    test_r, val_r, _ = run_method(
                        method, split, seed, eps_metrics, eps_hs, task_embs)
                except FileNotFoundError as e:
                    print(f"  [SKIP] {tag}  ({e.filename})")
                    continue
                except Exception as e:
                    print(f"  [ERR ] {tag}  {type(e).__name__}: {e}")
                    continue

                task_min = S.compute_task_min(test_r)
                sweep = alpha_sweep_for_run(test_r, val_r, task_min, ALPHAS, args.q)
                out["results"][method][split][f"seed{seed}"] = sweep

                a15 = sweep.get("0.15")
                if a15:
                    print(f"  {tag}: α=0.15  f1={a15['f1']:.2f}  "
                          f"bacc={a15['bal_acc']:.2f}  tdet={a15['tdet']:.3f}  "
                          f"({time.time() - t0:.1f}s)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
