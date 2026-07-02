"""
Alpha-sweep eval — paper fig3_fig7_fig8_fig9 / detection_curve_spec.md format.

For each (method, split), runs all 3 seeds, sweeps alpha across 0.0..1.0,
averages over seeds, writes one JSON matching:

Output structure:
  {
    "allseen": {<spec_method_key>: [{alpha, f1, f1_std, avg_det_time, avg_det_time_std,
                                     bal_acc, bal_acc_std, tpr, tnr}, ...]},
    "unseen":  {...},
  }

Method-key mapping (implementation → paper spec):
  actprobe → whole_2feat
  others (safe_lstm, safe_mlp, safe_lstm_tdqc, safe_mlp_tdqc, logpzo, cosine_knn) → same name

Usage:
  source env.sh
  python code/eval/eval_alpha_sweep.py --out results/frozen/alpha_sweep_curves.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "code"))

from lib import data as data_mod
from lib import metrics as M
from lib import splits as S

# Reuse run_method from eval_main_table.py (single source of truth for method execution)
sys.path.insert(0, str(ROOT / "code" / "eval"))
from eval_main_table import run_method, ALL_METHODS  # noqa: E402

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Implementation method key -> paper-spec method key
METHOD_KEY_MAP = {
    "actprobe":      "whole_2feat",
    "safe_lstm":      "safe_lstm",
    "safe_mlp":       "safe_mlp",
    "safe_lstm_tdqc": "safe_lstm_tdqc",
    "safe_mlp_tdqc":  "safe_mlp_tdqc",
    "logpzo":         "logpzo",
    "cosine_knn":     "cosine_knn",
}

# 101 alpha pts: 0.0, 0.01, ..., 0.99, 1.0
ALPHAS = [round(i * 0.01, 2) for i in range(0, 101)]


def alpha_sweep_for_seed(test_results, val_results, task_min_steps, q, alphas):
    """For one (method, split, seed) — returns dict[alpha → metrics]."""
    out = {}
    for alpha in alphas:
        m = M.f1_at_alpha(test_results, val_results, task_min_steps, alpha=alpha, q=q)
        # Detection time (T-det) at this tau: per-fail-ep, normalized.
        from sklearn.metrics import f1_score, balanced_accuracy_score  # noqa: F401
        # Re-extract tpr/tnr/T-det from raw scores using the calibrated tau
        tau = m["tau"]
        if not np.isfinite(tau):
            out[alpha] = {"f1": 0.0, "bal_acc": 0.5, "tpr": 0.0, "tnr": 1.0,
                          "avg_det_time": 1.0}
            continue
        # det time + tpr/tnr from test
        det_times = []
        tp = fn = fp = tn = 0
        for r in test_results:
            T = r["length"]
            ms = task_min_steps.get(r["task_id"], T)
            cutoff = M.task_min_cutoff(ms, q, T)
            sc = np.asarray(r["scores"])
            scq = M.quantile_max(sc, T, cutoff)
            detected = scq >= tau
            if r["label"] == 1:
                if detected:
                    tp += 1
                    # first hit step among scores up to T
                    hits = np.where(sc[:T] >= tau)[0]
                    det_times.append(hits[0] / T if len(hits) > 0 else 1.0)
                else:
                    fn += 1
                    det_times.append(1.0)
            else:
                if detected:
                    fp += 1
                else:
                    tn += 1
        tpr = tp / max(tp + fn, 1)
        tnr = tn / max(tn + fp, 1)
        avg_dt = float(np.mean(det_times)) if det_times else 1.0
        out[alpha] = {
            "f1":           m["f1"] / 100.0,        # f1_at_alpha returns 0-100, spec expects 0-1
            "bal_acc":      m["bal_acc"] / 100.0,
            "tpr":          tpr,
            "tnr":          tnr,
            "avg_det_time": avg_dt,
        }
    return out


def merge_seeds(seed_curves):
    """List of 3 alpha→metric dicts → spec list[{alpha, f1, f1_std, ...}]."""
    if not seed_curves:
        return []
    # collect per-alpha lists
    alphas = list(seed_curves[0].keys())
    merged = []
    for a in alphas:
        f1s   = [c[a]["f1"]      for c in seed_curves]
        bals  = [c[a]["bal_acc"] for c in seed_curves]
        tprs  = [c[a]["tpr"]     for c in seed_curves]
        tnrs  = [c[a]["tnr"]     for c in seed_curves]
        tdets = [c[a]["avg_det_time"] for c in seed_curves]
        merged.append({
            "alpha": a,
            "f1":              round(float(np.mean(f1s)),   4),
            "f1_std":          round(float(np.std(f1s)),    4),
            "avg_det_time":    round(float(np.mean(tdets)), 4),
            "avg_det_time_std":round(float(np.std(tdets)),  4),
            "bal_acc":         round(float(np.mean(bals)),  4),
            "bal_acc_std":     round(float(np.std(bals)),   4),
            "tpr":             round(float(np.mean(tprs)),  4),
            "tnr":             round(float(np.mean(tnrs)),  4),
        })
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=ALL_METHODS)
    ap.add_argument("--splits",  nargs="+", default=["allseen", "unseen"])
    ap.add_argument("--seeds",   nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--q",       type=float, default=0.25)
    ap.add_argument("--subsample_seed", type=int, default=42)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    print(f"Methods: {args.methods}")
    print(f"Splits:  {args.splits}")
    print(f"Seeds:   {args.seeds}")

    print("\nLoading data...")
    eps_metrics_full = data_mod.load_metrics_logs()
    eps_hs_full      = data_mod.load_hs()
    task_embs        = data_mod.load_task_embeddings()
    eps_metrics      = data_mod.subsample_50eps(eps_metrics_full, seed=args.subsample_seed)
    eps_hs           = data_mod.subsample_50eps(eps_hs_full,      seed=args.subsample_seed)
    print(f"  metrics: {sum(len(v) for v in eps_metrics.values())} eps")
    print(f"  hs:      {sum(len(v) for v in eps_hs.values())} eps")

    out = {split: {} for split in args.splits}

    for method in args.methods:
        spec_key = METHOD_KEY_MAP.get(method, method)
        for split in args.splits:
            seed_curves = []
            for seed in args.seeds:
                tag = f"{method}/{split}/seed{seed}"
                t0 = time.time()
                try:
                    test_r, val_r, _ = run_method(
                        method, split, seed, eps_metrics, eps_hs, task_embs)
                except Exception as e:
                    print(f"  [SKIP/ERR] {tag}: {type(e).__name__}: {e}")
                    continue
                task_min = S.compute_task_min(test_r)
                curve_dict = alpha_sweep_for_seed(test_r, val_r, task_min, args.q, ALPHAS)
                seed_curves.append(curve_dict)
                a15 = curve_dict[0.15]
                print(f"  {tag} @α=0.15: f1={a15['f1']:.3f} T-det={a15['avg_det_time']:.3f} "
                      f"({time.time() - t0:.1f}s)")
            if seed_curves:
                out[split][spec_key] = merge_seeds(seed_curves)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")

    # Summary @α=0.15
    print(f"\n{'='*70}")
    print("Summary @α=0.15 (3-seed mean)")
    print(f"{'='*70}")
    for split in args.splits:
        print(f"\n  {split}:")
        for method in args.methods:
            spec_key = METHOD_KEY_MAP.get(method, method)
            curve = out.get(split, {}).get(spec_key)
            if curve:
                pt = next(p for p in curve if abs(p["alpha"] - 0.15) < 0.001)
                print(f"    {spec_key:<20}  f1={pt['f1']:.3f}±{pt['f1_std']:.3f}  "
                      f"T-det={pt['avg_det_time']:.3f}")


if __name__ == "__main__":
    main()
