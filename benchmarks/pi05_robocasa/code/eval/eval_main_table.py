"""
Main Table evaluation — paper Table 1 (q=0.25 AUC, allseen + unseen).

Iterates: methods × splits × seeds → writes one JSON to results/.

Usage:
  source env.sh
  python code/eval/eval_main_table.py --out results/frozen/main_table.json
  # or partial:
  python code/eval/eval_main_table.py --methods actprobe safe_mlp --splits allseen --seeds 0
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
from lib.categories import TASK2CAT
from lib.methods import (actprobe, safe_mlp, safe_lstm, safe_tdqc,
                          logpzo, stac, cosine_knn)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CKPT_ROOT = ROOT / "checkpoints"

ALL_METHODS = [
    "actprobe", "safe_mlp", "safe_lstm",
    "safe_mlp_tdqc", "safe_lstm_tdqc",
    "logpzo", "stac", "cosine_knn",
]


def get_split(task_eps, split_type, seed):
    if split_type == "allseen":
        return S.split_allseen(task_eps, seed)
    elif split_type == "unseen":
        return S.split_unseen(task_eps, seed)
    raise ValueError(split_type)


def run_method(method, split_type, seed, eps_metrics, eps_hs_mp, task_embs):
    """Run one (method, split, seed). Returns (test_results, val_results)."""
    if method == "actprobe":
        sub = actprobe.subset_2feat(eps_metrics)
        sp = get_split(sub, split_type, seed)
        ckpt_path = CKPT_ROOT / "actprobe" / f"{split_type}_seed{seed}_best.pt"
        model, nm, ns = actprobe.load_ckpt(str(ckpt_path), device=DEVICE)
        test_r = actprobe.score_episodes(model, sp["test"], task_embs, nm, ns, DEVICE)
        val_r  = actprobe.score_episodes(model, sp["val"],  task_embs, nm, ns, DEVICE)
        del model

    elif method == "stac":
        sp = get_split(eps_metrics, split_type, seed)
        test_r = stac.score_episodes(sp["test"], variant="cumsum")
        val_r  = stac.score_episodes(sp["val"],  variant="cumsum")

    elif method in ("safe_mlp", "safe_lstm", "safe_mlp_tdqc", "safe_lstm_tdqc",
                     "logpzo", "cosine_knn"):
        sp = get_split(eps_hs_mp, split_type, seed)
        nm, ns = S.fit_norm(sp["train"], key="raw_hs")
        S.apply_norm(sp["train"], nm, ns, "raw_hs", "normed_hs")
        S.apply_norm(sp["val"],   nm, ns, "raw_hs", "normed_hs")
        S.apply_norm(sp["test"],  nm, ns, "raw_hs", "normed_hs")

        if method == "safe_mlp":
            ckpt = CKPT_ROOT / "safe_mlp" / f"{split_type}_seed{seed}.pt"
            model = safe_mlp.load_ckpt(str(ckpt), DEVICE)
            test_r = safe_mlp.score_episodes(model, sp["test"], DEVICE)
            val_r  = safe_mlp.score_episodes(model, sp["val"],  DEVICE)
            del model
        elif method == "safe_lstm":
            ckpt = CKPT_ROOT / "safe_lstm" / f"{split_type}_seed{seed}.pt"
            model = safe_lstm.load_ckpt(str(ckpt), DEVICE)
            test_r = safe_lstm.score_episodes(model, sp["test"], DEVICE)
            val_r  = safe_lstm.score_episodes(model, sp["val"],  DEVICE)
            del model
        elif method == "safe_mlp_tdqc":
            ckpt = CKPT_ROOT / "safe_mlp_tdqc" / f"{split_type}_seed{seed}.pt"
            model = safe_tdqc.load_mlp_ckpt(str(ckpt), DEVICE)
            test_r = safe_tdqc.score_mlp_episodes(model, sp["test"], DEVICE)
            val_r  = safe_tdqc.score_mlp_episodes(model, sp["val"],  DEVICE)
            del model
        elif method == "safe_lstm_tdqc":
            ckpt = CKPT_ROOT / "safe_lstm_tdqc" / f"{split_type}_seed{seed}.pt"
            model = safe_tdqc.load_lstm_ckpt(str(ckpt), DEVICE)
            test_r = safe_tdqc.score_lstm_episodes(model, sp["test"], DEVICE)
            val_r  = safe_tdqc.score_lstm_episodes(model, sp["val"],  DEVICE)
            del model
        elif method == "logpzo":
            ckpt = CKPT_ROOT / "logpzo" / f"{split_type}_seed{seed}.pt"
            net_s, net_f = logpzo.load_ckpt(str(ckpt), DEVICE)
            test_r = logpzo.score_episodes(net_s, net_f, sp["test"], DEVICE)
            val_r  = logpzo.score_episodes(net_s, net_f, sp["val"],  DEVICE)
            del net_s, net_f
        elif method == "cosine_knn":
            test_r = cosine_knn.score_episodes(sp["train"], sp["test"])
            val_r  = cosine_knn.score_episodes(sp["train"], sp["val"])
    else:
        raise ValueError(f"unknown method: {method}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return test_r, val_r, sp["train"]


def available_seeds(method, split_type):
    """Seeds with a checkpoint present, so default eval matches the shipped ckpts."""
    if method == "actprobe":
        cdir, pat = CKPT_ROOT / "actprobe", f"{split_type}_seed*_best.pt"
    elif method in ("safe_mlp", "safe_lstm", "safe_mlp_tdqc", "safe_lstm_tdqc", "logpzo"):
        cdir, pat = CKPT_ROOT / method, f"{split_type}_seed*.pt"
    else:                          # stac, cosine_knn -- no checkpoint
        return [0, 1, 2]
    seeds = sorted({int(p.stem.split("_seed")[1].split("_")[0]) for p in cdir.glob(pat)})
    return seeds or [0, 1, 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=ALL_METHODS)
    ap.add_argument("--splits",  nargs="+", default=["allseen", "unseen"])
    ap.add_argument("--seeds",   nargs="+", type=int, default=None,
                    help="default: auto-detect from trained checkpoints")
    ap.add_argument("--q",       type=float, default=0.25)
    ap.add_argument("--alpha",   type=float, default=0.15)
    ap.add_argument("--out",     type=str, required=True)
    args = ap.parse_args()

    print(f"Methods: {args.methods}")
    print(f"Splits:  {args.splits}")
    print(f"Seeds:   {args.seeds if args.seeds is not None else 'auto (from checkpoints)'}")
    print(f"q={args.q}, α={args.alpha}")

    print("\nLoading data...")
    eps_metrics = data_mod.load_metrics_logs()
    eps_hs_mp   = data_mod.load_hs_meanpool()
    task_embs   = data_mod.load_qwen3_embeddings()
    print(f"  metrics:  {len(eps_metrics)} tasks, "
          f"{sum(len(v) for v in eps_metrics.values())} eps")
    print(f"  hs_mp:    {len(eps_hs_mp)} tasks, "
          f"{sum(len(v) for v in eps_hs_mp.values())} eps")

    out = {"methods": args.methods, "splits": args.splits, "seeds": args.seeds if args.seeds is not None else "auto",
           "q": args.q, "alpha": args.alpha,
           "results": {}}

    for method in args.methods:
        out["results"][method] = {}
        for split in args.splits:
            out["results"][method][split] = {}
            seed_list = args.seeds if args.seeds is not None else available_seeds(method, split)
            for seed in seed_list:
                tag = f"{method}/{split}/seed{seed}"
                t0 = time.time()
                try:
                    test_r, val_r, train_eps = run_method(
                        method, split, seed, eps_metrics, eps_hs_mp, task_embs)
                except FileNotFoundError as e:
                    print(f"  [SKIP] {tag}  (ckpt missing: {e.filename})")
                    continue
                except Exception as e:
                    print(f"  [ERR ] {tag}  {type(e).__name__}: {e}")
                    continue

                # task_min from TEST split (paper protocol; train would miss unseen tasks)
                # task_min not needed for per-episode q-AUC
                ep_auc  = M.episode_auc(test_r) * 100
                qauc    = M.q_auc(test_r, q=args.q) * 100
                per_cat = M.q_auc_per_category(test_r, TASK2CAT, q=args.q)
                per_cat = {k: v * 100 for k, v in per_cat.items()}
                f1cp    = M.f1_at_alpha(test_r, val_r,
                                         alpha=args.alpha, q=args.q)

                out["results"][method][split][f"seed{seed}"] = {
                    "ep_auc":     round(ep_auc, 4),
                    "q_auc":      round(qauc, 4),
                    "per_cat":    {k: round(v, 4) for k, v in per_cat.items()},
                    "cp_f1":      round(f1cp["f1"], 4),
                    "cp_balacc":  round(f1cp["bal_acc"], 4),
                    "cp_tau":     round(f1cp["tau"], 6) if not np.isnan(f1cp["tau"]) else None,
                    "n_test":     len(test_r),
                    "elapsed_s":  round(time.time() - t0, 1),
                }
                print(f"  {tag}: ep_auc={ep_auc:.2f}  q_auc={qauc:.2f}  "
                      f"f1={f1cp['f1']:.2f}  ({time.time() - t0:.1f}s)")

        # Aggregate across seeds
        for split in args.splits:
            seed_results = out["results"][method][split]
            seeds = [v for k, v in seed_results.items() if k.startswith("seed")]
            if not seeds:
                continue
            qaucs = [s["q_auc"] for s in seeds]
            out["results"][method][split]["agg"] = {
                "q_auc_mean": round(float(np.mean(qaucs)), 4),
                "q_auc_std":  round(float(np.std(qaucs)),  4),
                "n_seeds":    len(qaucs),
            }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")

    # Summary
    print(f"\n{'='*70}")
    print(f"Summary (q={args.q} AUC, mean ± std over seeds)")
    print(f"{'='*70}")
    for method in args.methods:
        for split in args.splits:
            agg = out["results"][method][split].get("agg")
            if agg:
                print(f"  {method:<18} {split:<8}  "
                      f"{agg['q_auc_mean']:.2f} ± {agg['q_auc_std']:.2f}")


if __name__ == "__main__":
    main()
