"""ActProbe (2-feat = action_norm + chunk_mse) — paper main method.

Architecture: ActProbeNet (LSTM + MLP-skip with lang-cond h0/c0).
Input: 2 feat + 1 timestamp = 3-dim per step.
Lang conditioning: Qwen3 emb (1024) → bottleneck(16) → LSTM h0/c0(32).

PAPER_FEAT_IDX is imported from lib.data and resolved by key name. See data.py
for the canonical 6-feature schema.

OpenVLA-specific note: state_dict key names match the released OpenVLA
ActProbe checkpoints (`lp`, `h0_lin`, `c0_lin`) and are not interchangeable
with the pi0_libero ActProbe module.
"""
import numpy as np
import torch
import torch.nn as nn

from lib.data import PAPER_FEAT_IDX

LANG_DIM   = 1024
N_FEAT     = len(PAPER_FEAT_IDX)   # = 2
HIDDEN     = 32
LANG_BN    = 16    # bottleneck
DROPOUT    = 0.4


class ActProbeNet(nn.Module):
    """LSTM + MLP-skip with lang-cond h0/c0.

    Uses checkpoint-compatible module names:
      lp:     Linear(LANG_DIM, LANG_BN)
      h0_lin: Linear(LANG_BN, HIDDEN)
      c0_lin: Linear(LANG_BN, HIDDEN)
      lstm:   LSTM(N_FEAT+1, HIDDEN)
      mlp:    Linear(HIDDEN+N_FEAT+1, 16) → 16 → 8 → 1
    """
    def __init__(self, n_feat=N_FEAT):
        super().__init__()
        self.lp     = nn.Linear(LANG_DIM, LANG_BN)
        self.h0_lin = nn.Linear(LANG_BN, HIDDEN)
        self.c0_lin = nn.Linear(LANG_BN, HIDDEN)
        self.lstm   = nn.LSTM(n_feat + 1, HIDDEN, batch_first=True)
        self.drop   = nn.Dropout(DROPOUT)
        self.mlp    = nn.Sequential(
            nn.Linear(HIDDEN + n_feat + 1, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x, lang_emb, lengths):
        # x: (B, T, n_feat+1) with timestamp last
        z  = torch.relu(self.lp(lang_emb))
        h0 = self.h0_lin(z).unsqueeze(0).contiguous()
        c0 = self.c0_lin(z).unsqueeze(0).contiguous()
        pk = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(pk, (h0, c0))
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
        out = self.drop(out)
        return torch.sigmoid(self.mlp(torch.cat([out, x], dim=-1)).squeeze(-1))


def load_ckpt(path, device="cuda:0"):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = ActProbeNet(n_feat=N_FEAT).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    nm = np.array(ckpt["norm_mean"], dtype=np.float32)
    ns = np.array(ckpt["norm_std"],  dtype=np.float32)
    return model, nm, ns


def subset_2feat(task_eps, idx=None):
    """Return new task_eps with raw subset to 2 feats. Preserve full 6-feat as `raw_full`."""
    if idx is None:
        idx = PAPER_FEAT_IDX
    out = {}
    for tid, eps in task_eps.items():
        out[tid] = []
        for e in eps:
            new = dict(e)
            new["raw_full"] = e["raw"]
            new["raw"]      = e["raw"][:, idx].astype(np.float32)
            out[tid].append(new)
    return out


TIMESTAMP_DIVISOR = 100.0  # absolute timestamp t/100 (length-leak-free for OpenVLA)


@torch.no_grad()
def score_episodes(model, episodes, task_embs, norm_mean, norm_std, device="cuda:0"):
    """Run ActProbe on a list of eps (with `raw` 2-feat).

    Timestamp = t / 100 (absolute, length-INdependent). Replaces the older
    np.linspace(0, 1, T) which leaks length on OpenVLA (fail eps timeout at T=520,
    succ eps shorter). Ckpts in checkpoints/actprobe_2feat/ are trained with this
    same `t/100` convention.

    Returns: list of {scores, label, length, task_id, episode_id}
    """
    results = []
    for ep in episodes:
        raw = ep["raw"]
        T = raw.shape[0]
        feat = (raw - norm_mean) / (norm_std + 1e-7)
        ts = (np.arange(T, dtype=np.float32) / TIMESTAMP_DIVISOR).reshape(-1, 1)
        x = torch.from_numpy(np.hstack([feat, ts]).astype(np.float32)).unsqueeze(0).to(device)
        lang = torch.from_numpy(task_embs[ep["task_id"]].astype(np.float32)).unsqueeze(0).to(device)
        lens = torch.tensor([T], dtype=torch.long, device=device)
        sc = model(x, lang, lens).squeeze(0).cpu().numpy()[:T]
        results.append({"scores": sc, "label": ep["label"],
                        "length": T, "task_id": ep["task_id"],
                        "episode_id": ep.get("episode_id")})
    return results
