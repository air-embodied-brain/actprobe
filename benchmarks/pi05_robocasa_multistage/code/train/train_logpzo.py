"""Train LogPzO — flow-matching UNet on hidden states.

Two UNets (succ + fail) trained jointly. Score = ||hs+v_succ||² - ||hs+v_fail||².

Hyperparameters:
  Adam lr=1e-4, no wd, 200 epochs, chunk_size=512 per epoch.

Usage:
  source env.sh
  python code/train/train_logpzo.py --split allseen --seed 0
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "code"))

from lib import data as data_mod
from lib import splits as S
from lib.methods.logpzo import (_make_unet, _adjust_shape,
                                  score_episodes as logpzo_score)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CKPT_DIR = ROOT / "checkpoints" / "logpzo"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = 200
CHUNK_SIZE = 512
LR = 1e-4
EVAL_EVERY = 50


def _fm_loss(net, x):
    """Flow matching loss. x: (N, H', LOGPZO_IN_DIM)."""
    x1 = torch.randn_like(x)
    vtrue = x1 - x
    t = torch.rand(len(x), device=x.device).view(-1, 1, 1)
    xnow = x + t * vtrue
    vhat = net(xnow, (t.view(-1) * 100).long())
    return (vhat - vtrue).pow(2).mean()


def train(splits, seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    net_succ = _make_unet().to(DEVICE)
    net_fail = _make_unet().to(DEVICE)
    opt = torch.optim.Adam(list(net_succ.parameters()) + list(net_fail.parameters()), lr=LR)

    succ_np = np.vstack([e["normed_hs"] for e in splits["train"] if e["label"] == 0]).astype(np.float32)
    fail_np = np.vstack([e["normed_hs"] for e in splits["train"] if e["label"] == 1]).astype(np.float32)
    succ_t = torch.from_numpy(succ_np)
    fail_t = torch.from_numpy(fail_np)
    print(f"  train pool: {len(succ_t)} succ steps, {len(fail_t)} fail steps")

    best_auc, best_state = 0.0, None
    for epoch in range(1, EPOCHS + 1):
        net_succ.train(); net_fail.train()
        idx_s = torch.randperm(len(succ_t))[:CHUNK_SIZE]
        idx_f = torch.randperm(len(fail_t))[:CHUNK_SIZE]
        xs = _adjust_shape(succ_t[idx_s].to(DEVICE))
        xf = _adjust_shape(fail_t[idx_f].to(DEVICE))
        loss = _fm_loss(net_succ, xs) + _fm_loss(net_fail, xf)
        opt.zero_grad(); loss.backward(); opt.step()

        if epoch % EVAL_EVERY == 0:
            net_succ.eval(); net_fail.eval()
            val_results = logpzo_score(net_succ, net_fail, splits["val"], device=str(DEVICE))
            ep_labels = [r["label"] for r in val_results]
            ep_scores = [r["scores"].max() for r in val_results]
            auc = roc_auc_score(ep_labels, ep_scores) if len(set(ep_labels)) > 1 else 0.5
            if auc > best_auc + 1e-4:
                best_auc = auc
                best_state = {
                    "succ": {k: v.cpu().clone() for k, v in net_succ.state_dict().items()},
                    "fail": {k: v.cpu().clone() for k, v in net_fail.state_dict().items()},
                }
            print(f"  ep{epoch:4d}  val={auc:.4f}  best={best_auc:.4f}")

    if best_state:
        net_succ.load_state_dict(best_state["succ"])
        net_fail.load_state_dict(best_state["fail"])
    net_succ.eval(); net_fail.eval()
    print(f"  Best val AUC: {best_auc:.4f}")
    return net_succ, net_fail, best_auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["allseen", "unseen"], required=True)
    ap.add_argument("--seed",  type=int, required=True)
    args = ap.parse_args()

    print(f"=== LogPzO | split={args.split} seed={args.seed} ===")
    eps = data_mod.load_hs_meanpool()
    sp = (S.split_allseen if args.split == "allseen" else S.split_unseen)(eps, seed=args.seed)
    nm, ns = S.fit_norm(sp["train"], key="raw_hs")
    S.apply_norm(sp["train"], nm, ns, "raw_hs", "normed_hs")
    S.apply_norm(sp["val"],   nm, ns, "raw_hs", "normed_hs")
    print(f"  train={len(sp['train'])}  val={len(sp['val'])}  test={len(sp['test'])}")

    net_succ, net_fail, best_auc = train(sp, args.seed)

    out = CKPT_DIR / f"{args.split}_seed{args.seed}.pt"
    torch.save({"succ": net_succ.state_dict(), "fail": net_fail.state_dict()}, out)
    print(f"Saved → {out}  (val_auc={best_auc:.4f})")


if __name__ == "__main__":
    main()
