"""ActProbe (2-feat = action_norm + chunk_mse) — paper main method.

Architecture: ActProbeNet `arch_variant=full`.
Input: 2 feat + 1 timestamp = 3-dim per step.
Lang conditioning: Qwen3 emb (1024) → bottleneck(16) → LSTM h0/c0(32).
"""
import numpy as np
import torch
import torch.nn as nn

LANG_DIM = 1024
N_FEAT = 2          # action_norm + chunk_mse
HIDDEN = 32
BOTTLENECK = 16
DROPOUT = 0.4

PAPER_FEAT_IDX = [0, 4]   # indices in 10-feat raw


class ActProbeNet(nn.Module):
    """LSTM + MLP-skip with lang-cond h0/c0. Matches paper main."""
    def __init__(self):
        super().__init__()
        self.lang_proj = nn.Sequential(nn.Linear(LANG_DIM, BOTTLENECK), nn.ReLU())
        self.h0_proj   = nn.Linear(BOTTLENECK, HIDDEN)
        self.c0_proj   = nn.Linear(BOTTLENECK, HIDDEN)
        self.lstm      = nn.LSTM(N_FEAT + 1, HIDDEN, num_layers=1, batch_first=True)
        self.drop      = nn.Dropout(DROPOUT)
        self.mlp       = nn.Sequential(
            nn.Linear(HIDDEN + N_FEAT + 1, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x, lang, lengths):
        # x: (B, T, N_FEAT+1)  with timestamp last
        z  = self.lang_proj(lang)
        h0 = self.h0_proj(z).unsqueeze(0).contiguous()
        c0 = self.c0_proj(z).unsqueeze(0).contiguous()
        pk = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(pk, (h0, c0))
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
        out = self.drop(out)
        return torch.sigmoid(self.mlp(torch.cat([out, x], dim=-1)).squeeze(-1))


def load_ckpt(path, device="cuda:0"):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = ActProbeNet().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    nm = np.array(ckpt["norm_mean"], dtype=np.float32)
    ns = np.array(ckpt["norm_std"],  dtype=np.float32)
    return model, nm, ns


def subset_2feat(task_eps, idx=PAPER_FEAT_IDX):
    """Return new task_eps with raw subset to 2 feats. Original `raw` (10-feat) preserved as `raw10`."""
    out = {}
    for tid, eps in task_eps.items():
        out[tid] = []
        for e in eps:
            new = dict(e)
            new["raw10"] = e["raw"]
            new["raw"]   = e["raw"][:, idx].astype(np.float32)
            out[tid].append(new)
    return out


@torch.no_grad()
def score_episodes(model, episodes, task_embs, norm_mean, norm_std, device="cuda:0"):
    """Run ActProbe on a list of eps (with `raw` 2-feat).

    Returns: list of {scores, label, length, task_id, episode_id}
    """
    results = []
    for ep in episodes:
        raw = ep["raw"]
        T = raw.shape[0]
        feat = (raw - norm_mean) / (norm_std + 1e-7)
        ts = (np.arange(T, dtype=np.float32) / 100.0).reshape(-1, 1)  # abs t/100, leak-free (pi0.5-aligned)
        x = torch.from_numpy(np.hstack([feat, ts]).astype(np.float32)).unsqueeze(0).to(device)
        lang = torch.from_numpy(task_embs[ep["task_id"]].astype(np.float32)).unsqueeze(0).to(device)
        lens = torch.tensor([T], dtype=torch.long, device=device)
        sc = model(x, lang, lens).squeeze(0).cpu().numpy()[:T]
        results.append({"scores": sc, "label": ep["label"],
                        "length": T, "task_id": ep["task_id"],
                        "episode_id": ep.get("episode_id")})
    return results
