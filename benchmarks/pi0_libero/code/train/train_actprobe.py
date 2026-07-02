"""
Train ActProbe (= ActProbe, 2-feat) for π0+LIBERO.

Reproduces the 6 ckpts in checkpoints/actprobe/ (3 seeds × 2 splits).
Implements 2-feat ActProbeNet per paper §3
ablation_architecture_2feat.py (variant=full).

Usage:
  source env.sh
  python code/train/train_actprobe.py --seeds 0 1 2 --splits allseen unseen
  # default --out: $PI0_CKPT/actprobe/{split}_seed{N}.pt
"""
import argparse
import random
import sys
import time
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
from lib.methods.actprobe import ActProbeNet, PAPER_FEAT_IDX
from lib_shared.metrics import q_auc

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparams (matches paper ablation_architecture_2feat.py "full" variant)
BATCH_SIZE = 64
LR         = 1e-3
WEIGHT_DEC = 1e-4
T_MAX      = 400
PATIENCE   = 50
N_EPOCHS   = 400


class WDataset(Dataset):
    def __init__(self, eps, nm, ns, task_embs, max_len):
        self.eps = list(eps); self.nm = nm; self.ns = ns
        self.task_embs = task_embs; self.max_len = max_len

    def __len__(self):
        return len(self.eps)

    def __getitem__(self, idx):
        ep = self.eps[idx]
        raw = ep["raw"]; T = len(raw)
        feat = (raw - self.nm) / (self.ns + 1e-7)
        ts   = (np.arange(T, dtype=np.float32) / 100.0).reshape(-1, 1)  # abs t/100, leak-free
        x    = np.hstack([feat, ts]).astype(np.float32)
        if T < self.max_len:
            x = np.vstack([x, np.zeros((self.max_len - T, x.shape[1]), np.float32)])
        return (torch.from_numpy(x),
                torch.from_numpy(self.task_embs[ep["task_id"]].astype(np.float32)),
                torch.tensor(T, dtype=torch.long),
                torch.tensor(float(ep["label"]), dtype=torch.float32),
                ep["task_id"])


def collate(b):
    xs, langs, lens, lbs, tids = zip(*b)
    return torch.stack(xs), torch.stack(langs), torch.stack(lens), torch.stack(lbs), list(tids)


def train_one(seed, split_type, eps_metrics, task_embs, ckpt_path):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    # Subset to 2-feat (action_norm + chunk_mse)
    sub = {tid: [{**e, "raw10": e["raw"], "raw": e["raw"][:, PAPER_FEAT_IDX].astype(np.float32)}
                  for e in eps]
           for tid, eps in eps_metrics.items()}

    # Split
    if split_type == "allseen":
        sp = S.split_allseen(sub, seed)
    else:
        sp = S.split_unseen(sub, seed)
    tr, va = sp["train"], sp["val"]
    print(f"    train={len(tr)} val={len(va)}")

    # Norm fit on train
    nm, ns = S.fit_norm(tr, key="raw")
    max_len = max(len(e["raw"]) for e in tr + va)

    tr_ld = DataLoader(WDataset(tr, nm, ns, task_embs, max_len),
                       batch_size=BATCH_SIZE, shuffle=True,
                       collate_fn=collate, num_workers=0)
    va_ld = DataLoader(WDataset(va, nm, ns, task_embs, max_len),
                       batch_size=BATCH_SIZE, shuffle=False,
                       collate_fn=collate, num_workers=0)

    model = ActProbeNet().to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DEC)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_MAX)
    best_auc, best_st, no_imp = 0.0, None, 0

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        for x, lang, lens, lbs, _ in tr_ld:
            x, lang, lens, lbs = (x.to(DEVICE), lang.to(DEVICE),
                                  lens.to(DEVICE), lbs.to(DEVICE))
            sc   = model(x, lang, lens)
            mask = torch.arange(sc.shape[1], device=DEVICE).unsqueeze(0) < lens.unsqueeze(1)
            tgt  = lbs.unsqueeze(1).expand_as(sc)
            loss = (nn.functional.binary_cross_entropy(
                sc.clamp(1e-7, 1 - 1e-7), tgt, reduction="none") * mask).sum() / mask.float().sum()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()

        if epoch % 5 == 0:
            model.eval()
            va_results = []
            with torch.no_grad():
                for x, lang, lens, lbs, tids in va_ld:
                    x, lang, lens = x.to(DEVICE), lang.to(DEVICE), lens.to(DEVICE)
                    sc = model(x, lang, lens)
                    for i, (T, lab, tid) in enumerate(zip(lens.tolist(), lbs.tolist(), tids)):
                        va_results.append({"scores": sc[i, :T].cpu().numpy(),
                                           "length": T, "label": int(lab), "task_id": tid})
            auc = (q_auc(va_results, mode="taskmax", q=0.25)
                   if len(set(r["label"] for r in va_results)) > 1 else 0.5)
            if auc > best_auc + 1e-4:
                best_auc, best_st, no_imp = auc, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
            else:
                no_imp += 5
            if epoch % 50 == 0:
                print(f"      ep{epoch:3d}  val_q025taskmax={auc:.4f}  best={best_auc:.4f}  patience={no_imp}/{PATIENCE}")
            if no_imp >= PATIENCE:
                print(f"      Early stop @ ep{epoch}")
                break

    if best_st:
        model.load_state_dict(best_st)
    model.eval()
    print(f"    best val q025_taskmax: {best_auc:.4f}")

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "norm_mean": nm.tolist(), "norm_std": ns.tolist(),
        "arch_variant": "full", "feat_indices": PAPER_FEAT_IDX,
        "val_q025taskmax": float(best_auc),
    }, ckpt_path)
    print(f"    Saved → {ckpt_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds",  nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--splits", nargs="+", default=["allseen", "unseen"])
    ap.add_argument("--out_dir", type=str,
                    default=str(ROOT / "checkpoints" / "actprobe"))
    args = ap.parse_args()

    print(f"Device: {DEVICE}")
    print("Loading data...")
    eps_metrics = data_mod.load_metrics_logs()
    task_embs   = data_mod.load_task_embeddings()
    print(f"  metrics: {len(eps_metrics)} tasks, {sum(len(v) for v in eps_metrics.values())} eps")

    for split_type in args.splits:
        for seed in args.seeds:
            print(f"\n=== {split_type} seed={seed} ===")
            ckpt_path = Path(args.out_dir) / f"{split_type}_seed{seed}_best.pt"
            t0 = time.time()
            train_one(seed, split_type, eps_metrics, task_embs, ckpt_path)
            print(f"  elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
