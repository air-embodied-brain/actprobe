"""Aggregate per-episode OpenVLA HS pkls into per-task pkls + mean-pool over 7-token axis.

Source layout: $OPENVLA_ROLLOUTS_ROOT/task{N}--ep{M}--succ{S}.pkl
  Each pkl: {"hidden_states": (T, 7, 4096) bfloat16, "task_id": int, "eposide_idx": int,
             "episode_success": bool, "task_description": str, ...}

Output layout: data/hidden_states/task_{N}.pkl
  Each pkl: list[{"hidden_states": (T, 4096) f32, "episode_id": int, "task_id": int,
                  "success": bool, "task_description": str}]

Side effect: writes data/task_instructions.json {task_id_str: instruction}.

Usage:
  source env.sh
  python scripts/build_hidden_states.py
"""
import json
import os
import pickle
import re
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get("OPENVLA_LIBERO_ROOT", Path(__file__).resolve().parent.parent))
DATA = ROOT / "data"
HS_DIR = DATA / "hidden_states"
SRC = Path(os.environ["OPENVLA_ROLLOUTS_ROOT"])  # path to raw rollout pkls

FNAME_RE = re.compile(r"task(\d+)--ep(\d+)--succ(\d+)\.pkl")


def main():
    HS_DIR.mkdir(parents=True, exist_ok=True)
    by_task = {}
    instructions = {}

    for pkl_path in sorted(SRC.glob("task*--ep*--succ*.pkl")):
        m = FNAME_RE.match(pkl_path.name)
        if not m:
            continue
        tid = int(m.group(1))
        ep_idx = int(m.group(2))
        succ = int(m.group(3))

        with open(pkl_path, "rb") as f:
            d = pickle.load(f)
        hs = d.get("hidden_states")
        if hs is None:
            continue
        if isinstance(hs, torch.Tensor):
            hs = hs.float()
        else:
            hs = torch.tensor(hs, dtype=torch.float32)
        if hs.dim() == 3:
            hs = hs.mean(dim=1)  # (T, 7, 4096) → (T, 4096)
        arr = hs.numpy().astype(np.float32)
        if arr.shape[0] == 0:
            continue
        if arr.ndim != 2 or arr.shape[1] != 4096:
            raise ValueError(f"unexpected pooled shape {arr.shape} from {pkl_path.name}")

        ep = {
            "hidden_states":    arr,
            "episode_id":       ep_idx,
            "task_id":          tid,
            "success":          bool(succ) and bool(d.get("episode_success", succ)),
            "task_description": d.get("task_description", ""),
        }
        by_task.setdefault(tid, []).append(ep)

        if tid not in instructions and ep["task_description"]:
            instructions[f"task_{tid}"] = ep["task_description"]

    for tid, eps in sorted(by_task.items()):
        eps.sort(key=lambda e: e["episode_id"])
        out = HS_DIR / f"task_{tid}.pkl"
        with open(out, "wb") as f:
            pickle.dump(eps, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  task_{tid}: {len(eps)} eps  → {out.name}")

    instr_path = DATA / "task_instructions.json"
    with open(instr_path, "w") as f:
        json.dump(instructions, f, indent=2, ensure_ascii=False)
    print(f"\nWrote → {instr_path}")
    print(f"Total eps: {sum(len(v) for v in by_task.values())} across {len(by_task)} tasks")


if __name__ == "__main__":
    main()
