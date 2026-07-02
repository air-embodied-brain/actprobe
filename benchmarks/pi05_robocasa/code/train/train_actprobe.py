"""Train ActProbe (2-feat ActProbeNet). Paper main method.

Hyperparameters (matches paper / ablation_architecture.py):
  AdamW lr=1e-3, wd=1e-4
  CosineAnnealingLR T_max=400
  BCE loss (uniform label, masked)
  Batch size 64
  400 epochs max, early stop after 50 epochs no-improvement
  Eval every 5 epochs

Output ckpt format:
  {"model_state_dict": ..., "norm_mean": [..], "norm_std": [..],
   "arch_variant": "full", "feat_idx": [0, 4],
   "feat_names": ["action_norm", "chunk_mse"]}

Usage:
  source env.sh
  python code/train/train_actprobe.py --split allseen --seed 0
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT.parent.parent))

from lib import data as data_mod
from lib import splits as S
from lib.methods.actprobe import (ActProbeNet, PAPER_FEAT_IDX, subset_2feat)
from lib_shared.metrics import q_auc

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CKPT_DIR = ROOT / "checkpoints" / "actprobe"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = 400
BATCH = 64
LR = 1e-3
WD = 1e-4
EVAL_EVERY = 5
PATIENCE = 50


class _DS(Dataset):
    def __init__(self, eps, task_embs, norm_mean, norm_std, max_len):
        self.eps = eps; self.task_embs = task_embs
        self.nm = norm_mean; self.ns = norm_std; self.max_len = max_len

    def __len__(self): return len(self.eps)

    def __getitem__(self, idx):
        ep = self.eps[idx]
        T = len(ep["raw"])
        feat = (ep["raw"] - self.nm) / (self.ns + 1e-7)
        ts   = (np.arange(T, dtype=np.float32) / 100.0).reshape(-1, 1)  # abs t/100 (pi0.5 decision)
        x    = np.hstack([feat, ts]).astype(np.float32)
        if self.max_len > T:
            x = np.vstack([x, np.zeros((self.max_len - T, x.shape[1]), dtype=np.float32)])
        return (torch.from_numpy(x),
                torch.from_numpy(self.task_embs[ep["instruction"]].astype(np.float32)),
                torch.tensor(T, dtype=torch.long),
                torch.tensor(float(ep["label"]), dtype=torch.float32),
                ep["task_id"])


def _collate(batch):
    xs, langs, lens, labs, tids = zip(*batch)
    return torch.stack(xs), torch.stack(langs), torch.stack(lens), torch.stack(labs), list(tids)


def train(splits, task_embs, seed, n_epochs=EPOCHS):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    nm, ns = S.fit_norm(splits["train"], key="raw")
    max_len = max(len(e["raw"]) for e in splits["train"] + splits["val"])

    tr_ld = DataLoader(_DS(splits["train"], task_embs, nm, ns, max_len),
                       BATCH, shuffle=True,  collate_fn=_collate, num_workers=2)
    va_ld = DataLoader(_DS(splits["val"],   task_embs, nm, ns, max_len),
                       BATCH, shuffle=False, collate_fn=_collate, num_workers=2)

    model = ActProbeNet().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    best_auc, best_state, no_improve = 0.0, None, 0
    for epoch in range(1, n_epochs + 1):
        model.train()
        for x, lang, lens, labs, _ in tr_ld:
            x, lang, lens, labs = (x.to(DEVICE), lang.to(DEVICE),
                                    lens.to(DEVICE), labs.to(DEVICE))
            sc   = model(x, lang, lens)
            mask = torch.arange(sc.shape[1], device=DEVICE).unsqueeze(0) < lens.unsqueeze(1)
            tgt  = labs.unsqueeze(1).expand_as(sc)
            loss = (nn.functional.binary_cross_entropy(
                        sc.clamp(1e-7, 1 - 1e-7), tgt, reduction="none")
                    * mask.float()).sum() / mask.float().sum()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()

        if epoch % EVAL_EVERY == 0:
            model.eval()
            va_results = []
            with torch.no_grad():
                for x, lang, lens, labs, tids in va_ld:
                    x, lang, lens = x.to(DEVICE), lang.to(DEVICE), lens.to(DEVICE)
                    sc = model(x, lang, lens)
                    for i, (T, lab, tid) in enumerate(zip(lens.tolist(), labs.tolist(), tids)):
                        va_results.append({"scores": sc[i, :T].cpu().numpy(),
                                           "length": T, "label": int(lab), "task_id": tid})
            auc = (q_auc(va_results, mode="taskmax", q=0.25)
                   if len(set(r["label"] for r in va_results)) > 1 else 0.5)
            if auc > best_auc + 1e-4:
                best_auc = auc; no_improve = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                no_improve += EVAL_EVERY
            if epoch % 50 == 0:
                print(f"  ep{epoch:3d}  val_q025taskmax={auc:.4f}  best={best_auc:.4f}  "
                      f"patience={no_improve}/{PATIENCE}")
            if no_improve >= PATIENCE:
                print(f"  Early stop at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    print(f"  Best val q025_taskmax: {best_auc:.4f}")
    return model, nm, ns, best_auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["allseen", "unseen"])
    ap.add_argument("--seeds",  nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=400)
    args = ap.parse_args()

    print("Loading metrics_logs...")
    eps = data_mod.load_metrics_logs()
    task_embs = data_mod.load_qwen3_embeddings()
    eps_2f = subset_2feat(eps, idx=PAPER_FEAT_IDX)

    for split in args.splits:
        for seed in args.seeds:
            print(f"=== ActProbe (2-feat) | split={split} seed={seed} ===")
            if split == "allseen":
                sp = S.split_allseen(eps_2f, seed=seed)
            else:
                sp = S.split_unseen(eps_2f, seed=seed)
            print(f"  train={len(sp['train'])}  val={len(sp['val'])}  test={len(sp['test'])}")
            model, nm, ns, best_auc = train(sp, task_embs, seed, n_epochs=args.epochs)
            out = CKPT_DIR / f"{split}_seed{seed}_best.pt"
            torch.save({"model_state_dict": model.state_dict(),
                        "norm_mean": nm.tolist(),
                        "norm_std":  ns.tolist(),
                        "arch_variant": "full",
                        "feat_idx": list(PAPER_FEAT_IDX),
                        "feat_names": ["action_norm", "chunk_mse"],
                        "val_q025taskmax": float(best_auc)},
                       out)
            print(f"Saved → {out}")


if __name__ == "__main__":
    main()
