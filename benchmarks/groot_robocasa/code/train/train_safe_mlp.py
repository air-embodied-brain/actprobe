"""Train SAFE-MLP per Protocol §4.2.

Adam lr=1e-3, wd=1e-4, batch=512, 1000 epochs.
Hinge loss + cumsum: success → relu(s_t), failure → -s_t.
Class-frequency weighting.

Usage:
  source env.sh
  python code/train/train_safe_mlp.py --split allseen --seed 0
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "code"))

from lib import data as data_mod
from lib import splits as S
from lib.methods.safe_mlp import SafeMLP

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CKPT_DIR = ROOT / "checkpoints" / "safe_mlp"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = 1000
BATCH = 512
LR = 1e-3
WD = 1e-4
EVAL_EVERY = 20


class _DS(Dataset):
    def __init__(self, eps, normed_key="normed_hs"):
        self.eps = eps
        self.normed_key = normed_key
    def __len__(self): return len(self.eps)
    def __getitem__(self, i):
        ep = self.eps[i]
        return {"x": ep[self.normed_key], "T": ep["length"], "label": ep["label"]}


def _collate(batch):
    Tmax = max(b["T"] for b in batch)
    D = batch[0]["x"].shape[1]
    x = np.zeros((len(batch), Tmax, D), dtype=np.float32)
    lens = np.zeros(len(batch), dtype=np.int64)
    labels = np.zeros(len(batch), dtype=np.int64)
    for i, b in enumerate(batch):
        x[i, :b["T"]] = b["x"]
        lens[i] = b["T"]
        labels[i] = b["label"]
    succ = 1 - labels   # 1 = success
    return (torch.from_numpy(x), torch.from_numpy(labels), torch.from_numpy(succ),
            torch.from_numpy(lens))


def train(splits, seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    tr_ld = DataLoader(_DS(splits["train"]), batch_size=BATCH, shuffle=True,
                       collate_fn=_collate, num_workers=2)
    va_ld = DataLoader(_DS(splits["val"]), batch_size=BATCH, shuffle=False,
                       collate_fn=_collate, num_workers=2)

    model = SafeMLP(input_dim=1024, hidden=256).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

    n_fail = sum(1 for e in splits["train"] if e["label"] == 1)
    n_succ = len(splits["train"]) - n_fail
    n = len(splits["train"])
    w_fail = n / (n_fail + 1e-8)
    w_succ = n / (n_succ + 1e-8)

    best_auc, best_state = 0.0, None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x, lab, succ, lens in tr_ld:
            x = x.to(DEVICE); succ = succ.to(DEVICE); lens = lens.to(DEVICE)
            B, T = x.shape[:2]
            mask = (torch.arange(T, device=DEVICE).unsqueeze(0) < lens.unsqueeze(1)).float()

            s = model(x, lens)   # (B, T) cumsum

            is_succ = (succ == 1).float().unsqueeze(1)
            is_fail = (succ == 0).float().unsqueeze(1)
            losses = is_succ * torch.relu(s) + is_fail * (-s)
            seq_loss = (losses * mask).sum(1) / mask.sum(1).clamp(min=1)
            weighted = (seq_loss * is_fail.squeeze(1) * w_fail +
                        seq_loss * is_succ.squeeze(1) * w_succ)
            loss = weighted.sum() / B
            opt.zero_grad(); loss.backward(); opt.step()

        if epoch % EVAL_EVERY == 0:
            model.eval()
            ep_sc, ep_lb = [], []
            with torch.no_grad():
                for x, lab, succ, lens in va_ld:
                    x = x.to(DEVICE); lens = lens.to(DEVICE)
                    s = model(x, lens)
                    T = s.shape[1]
                    m = torch.arange(T, device=DEVICE).unsqueeze(0) < lens.unsqueeze(1)
                    ep_sc.extend(s.masked_fill(~m, -1e9).max(1).values.cpu().tolist())
                    ep_lb.extend(lab.tolist())
            auc = roc_auc_score(ep_lb, ep_sc) if len(set(ep_lb)) > 1 else 0.5
            if auc > best_auc + 1e-4:
                best_auc = auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if epoch % 200 == 0:
                print(f"  ep{epoch:4d}  val={auc:.4f}  best={best_auc:.4f}")

    if best_state: model.load_state_dict(best_state)
    model.eval()
    print(f"  Best val AUC: {best_auc:.4f}")
    return model, best_auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["allseen", "unseen"], required=True)
    ap.add_argument("--seed",  type=int, required=True)
    args = ap.parse_args()

    print(f"=== SAFE-MLP | split={args.split} seed={args.seed} ===")
    eps = data_mod.load_hs_meanpool()
    sp = (S.split_allseen if args.split == "allseen" else S.split_unseen)(eps, seed=args.seed)
    nm, ns = S.fit_norm(sp["train"], key="raw_hs")
    S.apply_norm(sp["train"], nm, ns, "raw_hs", "normed_hs")
    S.apply_norm(sp["val"],   nm, ns, "raw_hs", "normed_hs")
    print(f"  train={len(sp['train'])}  val={len(sp['val'])}  test={len(sp['test'])}")

    model, best_auc = train(sp, args.seed)

    out = CKPT_DIR / f"{args.split}_seed{args.seed}.pt"
    torch.save(model.state_dict(), out)
    print(f"Saved → {out}  (val_auc={best_auc:.4f})")


if __name__ == "__main__":
    main()
