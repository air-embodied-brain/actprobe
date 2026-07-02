"""
Build hidden_states_meanpool/ from hidden_states/.

For each task pkl, mean-pool over the n_ode (axis=1) dimension:
  (T, 4, 1024) fp16  →  (T, 1024) fp16

Matches the released baseline-loader protocol:
  arr = hs.mean(axis=1).astype(np.float32)
"""
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC  = ROOT / "data" / "hidden_states"
DST  = ROOT / "data" / "hidden_states_meanpool"
DST.mkdir(parents=True, exist_ok=True)

pkls = sorted(SRC.glob("*.pkl"))
print(f"Found {len(pkls)} task pkls in {SRC}")

for p in pkls:
    with open(p, "rb") as f:
        eps = pickle.load(f)
    out_eps = []
    for ep in eps:
        hs = ep["hidden_states"]   # (T, 4, 1024) fp16
        if hs.ndim != 3 or hs.shape[1] != 4 or hs.shape[2] != 1024:
            print(f"  WARN {p.name}: unexpected shape {hs.shape}")
        mp = hs.mean(axis=1).astype(np.float16)   # (T, 1024) fp16
        out_eps.append({**ep, "hidden_states": mp})
    out_path = DST / p.name
    with open(out_path, "wb") as f:
        pickle.dump(out_eps, f, protocol=4)
    print(f"  {p.name}: {len(eps)} eps  →  {out_path.name}")

print(f"\nDone. mean-pooled pkls in {DST}")
