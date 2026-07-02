"""Train SAFE-MLP-TDQC strict (paper-faithful TD(0) on hidden_states).

Architecture: IndepMLP (Linear → ReLU → Linear → Sigmoid, no cumsum).
Hyperparameters (paper sweep mlp_TD0):
  Adam lr=1e-4, wd=0, batch=64, 200 epochs
  StepLR step_size=200 gamma=0.8
  TDLoss + target network (sync every 10 batches)

Usage:
  source env.sh
  python code/train/train_safe_mlp_tdqc.py --split allseen --seed 0
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "code"))

from lib import data as data_mod
from lib import splits as S
from lib.methods.safe_tdqc import IndepMLP
from lib.tdqc_strict import TargetNet, tdqc_loss

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CKPT_DIR = ROOT / "checkpoints" / "safe_mlp_tdqc"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

LR = 1e-4
LAMBDA_REG = 0.0
LR_GAMMA = 0.8
LR_STEP_SIZE = 200
BATCH = 64
EPOCHS = 200
TARGET_UPDATE_FREQ = 10
PATIENCE = 50
EVAL_EVERY = 20


class _DS(Dataset):
    def __init__(self, eps, normed_key="normed_hs"):
        self.eps = eps; self.normed_key = normed_key
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
    succ = 1 - labels
    return (torch.from_numpy(x), torch.from_numpy(labels), torch.from_numpy(succ),
            torch.from_numpy(lens))


def train(splits, seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    tr_ld = DataLoader(_DS(splits["train"]), batch_size=BATCH, shuffle=True,
                       collate_fn=_collate, num_workers=0)
    va_ld = DataLoader(_DS(splits["val"]), batch_size=BATCH, shuffle=False,
                       collate_fn=_collate, num_workers=0)

    main = IndepMLP(input_dim=1024, n_layers=2, hidden_dim=256).to(DEVICE)
    target = TargetNet(main)
    opt = torch.optim.Adam(main.parameters(), lr=LR, weight_decay=LAMBDA_REG)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

    best_auc, best_state, no_improve = 0.0, None, 0
    for epoch in range(1, EPOCHS + 1):
        main.train()
        for x, lab, succ, lens in tr_ld:
            x = x.to(DEVICE); lens = lens.to(DEVICE)
            succ_lab = succ.to(DEVICE).float()
            B, Tmax = x.shape[:2]
            mask = (torch.arange(Tmax, device=DEVICE).unsqueeze(0) < lens.unsqueeze(1)).float()
            scores = main(x, lens)
            target_scores = target(x, lens)
            loss = tdqc_loss(scores, target_scores, succ_lab, mask)
            opt.zero_grad(); loss.backward(); opt.step()
            target.maybe_step(main, freq=TARGET_UPDATE_FREQ)
        sched.step()

        if epoch % EVAL_EVERY == 0:
            main.eval()
            ep_sc, ep_lb = [], []
            with torch.no_grad():
                for x, lab, succ, lens in va_ld:
                    x = x.to(DEVICE); lens = lens.to(DEVICE)
                    sc = main(x, lens)
                    Tm = sc.shape[1]
                    m = torch.arange(Tm, device=DEVICE).unsqueeze(0) < lens.unsqueeze(1)
                    ep_sc.extend(sc.masked_fill(~m, -1).max(1).values.cpu().numpy())
                    ep_lb.extend(lab.numpy())
            try:
                auc = roc_auc_score(ep_lb, ep_sc)
            except Exception:
                auc = 0.5
            if auc > best_auc + 1e-4:
                best_auc = auc
                best_state = {k: v.cpu().clone() for k, v in main.state_dict().items()}
                no_improve = 0
            else:
                no_improve += EVAL_EVERY
            print(f"  ep{epoch:4d}  val={auc:.4f}  best={best_auc:.4f}  "
                  f"patience={no_improve}/{PATIENCE}")
            if no_improve >= PATIENCE:
                print(f"  Early stop at epoch {epoch}")
                break

    if best_state: main.load_state_dict(best_state)
    main.eval()
    print(f"  Best val AUC: {best_auc:.4f}")
    return main, best_auc


def main_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["allseen", "unseen"], required=True)
    ap.add_argument("--seed",  type=int, required=True)
    args = ap.parse_args()

    print(f"=== SAFE-MLP-TDQC-strict | split={args.split} seed={args.seed} ===")
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
    main_cli()
