"""SAFE-MLP-TDQC + SAFE-LSTM-TDQC (strict variant) — π0+LIBERO.

Pi0 与 GR00T 的差异：
- pi0 SAFE-MLP-TDQC 用与 plain SAFE-MLP **相同**的 nn.Sequential `net.*`
  state_dict structure（不是 GR00T 的 IndepMLP `projector.*`），但 forward 直接
  返回 per-step sigmoid（不 cumsum）。
- pi0 SAFE-LSTM-TDQC 与 plain SAFE-LSTM 同架构（鏡像 GR00T 行为）。

Both ckpts come from the `_strict` variant (paper default).
"""
import numpy as np
import torch
import torch.nn as nn

from lib.methods import safe_lstm


class SafeMLPNoSum(nn.Module):
    """Pi0 SAFE-MLP-TDQC：与 plain SafeMLP 同 state_dict，但 forward 返回 per-step sigmoid（无 cumsum）。"""
    def __init__(self, input_dim=4096, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, lengths=None):
        return torch.sigmoid(self.net(x).squeeze(-1))   # (B, T) — per-step sigmoid


def load_mlp_ckpt(path, device="cuda:0"):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model = SafeMLPNoSum(input_dim=4096, hidden=256).to(device)
    model.load_state_dict(sd)
    model.eval()
    return model


def load_lstm_ckpt(path, device="cuda:0"):
    """SAFE-LSTM-TDQC reuses plain SAFE-LSTM architecture."""
    return safe_lstm.load_ckpt(path, device=device)


@torch.no_grad()
def score_mlp_episodes(model, episodes, device="cuda:0", normed_key="normed_hs"):
    """SAFE-MLP-TDQC: per-step sigmoid score (no cumsum)."""
    results = []
    for ep in episodes:
        x = torch.from_numpy(ep[normed_key].astype(np.float32)).unsqueeze(0).to(device)
        s = model(x).squeeze(0).cpu().numpy()[:ep["length"]]
        results.append({"scores": s, "label": ep["label"],
                        "length": ep["length"], "task_id": ep["task_id"],
                        "episode_id": ep.get("episode_id")})
    return results


# SAFE-LSTM-TDQC reuses the plain LSTM scorer
score_lstm_episodes = safe_lstm.score_episodes
