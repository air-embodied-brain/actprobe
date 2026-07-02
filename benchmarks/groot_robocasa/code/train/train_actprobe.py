"""Train ActProbe Group 2: fake_unseen val for OOD ckpt selection.

K tasks are held out from training entirely; all their episodes form fake_val.
Ckpt selection uses val_q025_{val_mode} on fake_val (OOD signal).
Uniform BCE loss, no gradient changes. Runs full EPOCHS, eval every EVAL_EVERY.

Applies to both allseen and unseen protocols:
  allseen : K tasks from all 22, fake_val tasks ARE in test (15% held-out)
  unseen  : K tasks from 15 seen tasks, fake_val tasks NOT in test (pure OOD)

Usage:
  source env.sh
  python code/train/train_actprobe.py --split allseen --seed 0 --fake-unseen 3 --val-mode taskmax
  python code/train/train_actprobe.py --split unseen  --seed 0 --fake-unseen 3 --val-mode taskmax
"""
import argparse
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT.parent.parent))

from lib import data as data_mod
from lib import splits as S
from lib.categories import EXCLUDE_TASKS
from lib.methods.actprobe import ActProbeNet, PAPER_FEAT_IDX, subset_2feat
from lib_shared.metrics import q_auc

DEVICE     = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPOCHS     = 300
BATCH      = 64
LR         = 1e-3
WD         = 1e-4
EVAL_EVERY = 10
Q          = 0.25


def _task_max(eps):
    tmax = defaultdict(int)
    for e in eps:
        L = len(e["raw"])
        if L > tmax[e["task_id"]]:
            tmax[e["task_id"]] = L
    return dict(tmax)


def make_splits_g2(task_eps, seed, split_type, n_fake):
    """Build train/val/fake_val/test splits for Group 2.

    fake_val: all episodes from n_fake held-out tasks (never trained on).
    train/val: remaining tasks, same ratios as original protocol.
    test: same as original protocol.
    """
    rng_fake = random.Random(seed + 9000)  # separate offset for fake selection

    if split_type == "allseen":
        # Original allseen split to get test
        base = S.split_allseen(task_eps, seed)
        # Trainable tasks: all non-excluded
        train_tasks = sorted([t for t in task_eps if t not in EXCLUDE_TASKS])
        rng_fake.shuffle(train_tasks)
        fake_tasks = set(train_tasks[:n_fake])
        real_tasks = [t for t in train_tasks if t not in fake_tasks]

        rng = random.Random(seed)
        sp = {"train": [], "val": [], "fake_val": [], "test": base["test"]}
        for tid in real_tasks:
            for lv in [0, 1]:
                grp = [e for e in task_eps[tid] if e["label"] == lv]
                rng.shuffle(grp)
                n = len(grp)
                ntr = int(n * 0.70)
                nva = int(n * 0.15)
                sp["train"].extend(grp[:ntr])
                sp["val"].extend(grp[ntr:ntr + nva])
        for tid in fake_tasks:
            sp["fake_val"].extend(task_eps[tid])

    else:  # unseen
        rng_unseen = random.Random(seed + 1000)
        all_tasks = sorted([t for t in task_eps if t not in EXCLUDE_TASKS])
        rng_unseen.shuffle(all_tasks)
        unseen_test = set(all_tasks[:S.N_UNSEEN])
        seen_tasks  = [t for t in all_tasks if t not in unseen_test]

        # Pick fake_unseen from seen_tasks
        rng_fake.shuffle(seen_tasks)
        fake_tasks = set(seen_tasks[:n_fake])
        real_tasks = [t for t in seen_tasks if t not in fake_tasks]

        rng = random.Random(seed)
        sp = {"train": [], "val": [], "fake_val": [], "test": []}
        for tid in real_tasks:
            for lv in [0, 1]:
                grp = [e for e in task_eps[tid] if e["label"] == lv]
                rng.shuffle(grp)
                n = len(grp)
                ntr = int(n * 0.70)
                sp["train"].extend(grp[:ntr])
                sp["val"].extend(grp[ntr:])
        for tid in fake_tasks:
            sp["fake_val"].extend(task_eps[tid])
        for tid in unseen_test:
            sp["test"].extend(task_eps[tid])

    return sp


class _DS(Dataset):
    def __init__(self, eps, task_embs, norm_mean, norm_std, max_len):
        self.eps = eps; self.task_embs = task_embs
        self.nm = norm_mean; self.ns = norm_std; self.max_len = max_len

    def __len__(self): return len(self.eps)

    def __getitem__(self, idx):
        ep = self.eps[idx]
        T  = len(ep["raw"])
        feat = (ep["raw"] - self.nm) / (self.ns + 1e-7)
        ts   = (np.arange(T, dtype=np.float32) / 100.0).reshape(-1, 1)
        x    = np.hstack([feat, ts]).astype(np.float32)
        if self.max_len > T:
            x = np.vstack([x, np.zeros((self.max_len - T, x.shape[1]), dtype=np.float32)])
        return (torch.from_numpy(x),
                torch.from_numpy(self.task_embs[ep["task_id"]].astype(np.float32)),
                torch.tensor(T, dtype=torch.long),
                torch.tensor(float(ep["label"]), dtype=torch.float32),
                ep["task_id"])


def _collate(batch):
    xs, langs, lens, labs, tids = zip(*batch)
    return torch.stack(xs), torch.stack(langs), torch.stack(lens), torch.stack(labs), list(tids)


def _q025_fake_val(model, eps_list, task_embs, nm, ns, max_len, val_mode):
    """q=0.25 AUC (taskmax or taskmin) on a list of episodes."""
    ds = _DS(eps_list, task_embs, nm, ns, max_len)
    ld = DataLoader(ds, BATCH, shuffle=False, collate_fn=_collate, num_workers=2)
    results = []
    model.eval()
    with torch.no_grad():
        for x, lang, lens, labs, tids in ld:
            x, lang, lens = x.to(DEVICE), lang.to(DEVICE), lens.to(DEVICE)
            sc = model(x, lang, lens)
            for i, (T, lab, tid) in enumerate(zip(lens.tolist(), labs.tolist(), tids)):
                results.append({"scores": sc[i, :T].cpu().numpy(),
                                "length": T, "label": int(lab), "task_id": tid})
    if len(set(r["label"] for r in results)) < 2:
        return 0.5
    return q_auc(results, mode=val_mode, q=Q)


def train(sp, task_embs, seed, val_mode):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    nm, ns  = S.fit_norm(sp["train"], key="raw")
    max_len = max(len(e["raw"]) for e in sp["train"] + sp["val"] + sp["fake_val"])

    tr_ld = DataLoader(_DS(sp["train"], task_embs, nm, ns, max_len),
                       BATCH, shuffle=True,  collate_fn=_collate, num_workers=2)

    model = ActProbeNet().to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_qtm, best_state_qtm = 0.0, None

    for epoch in range(1, EPOCHS + 1):
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
            q_tm = _q025_fake_val(model, sp["fake_val"], task_embs,
                                   nm, ns, max_len, val_mode)
            if q_tm > best_qtm + 1e-4:
                best_qtm = q_tm
                best_state_qtm = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  ep{epoch:3d}  fake_val_q025{val_mode}={q_tm:.4f}  best={best_qtm:.4f}")

    print(f"  Best fake_val q025_{val_mode}={best_qtm:.4f}")
    return nm, ns, best_state_qtm, best_qtm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits",      nargs="+", default=["allseen", "unseen"])
    ap.add_argument("--seeds",       nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--fake-unseen", type=int, default=3)
    ap.add_argument("--val-mode",    choices=["taskmax", "taskmin"], default="taskmax")
    args = ap.parse_args()

    ckpt_dir = ROOT / "checkpoints" / "actprobe"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    eps = data_mod.load_metrics_logs()
    task_embs = data_mod.load_task_embeddings()
    eps_2f = subset_2feat(eps, idx=PAPER_FEAT_IDX)

    for split in args.splits:
        for seed in args.seeds:
            print(f"=== ActProbe | split={split} seed={seed} "
                  f"fake_unseen={args.fake_unseen} val_mode={args.val_mode} ===")
            sp = make_splits_g2(eps_2f, seed, split, args.fake_unseen)
            print(f"  train={len(sp['train'])}  val={len(sp['val'])}"
                  f"  fake_val={len(sp['fake_val'])}  test={len(sp['test'])}")
            nm, ns, state, best_qtm = train(sp, task_embs, seed, args.val_mode)
            if state:
                out = ckpt_dir / f"{split}_seed{seed}_best.pt"
                torch.save({"model_state_dict": state,
                            "norm_mean": nm.tolist(), "norm_std": ns.tolist(),
                            "arch_variant": "full",
                            "feat_idx": list(PAPER_FEAT_IDX),
                            "feat_names": ["action_norm", "chunk_mse"],
                            f"val_q025{args.val_mode}": float(best_qtm),
                            "fake_unseen": args.fake_unseen,
                            "val_mode": args.val_mode}, out)
                print(f"Saved → {out}")


if __name__ == "__main__":
    main()
