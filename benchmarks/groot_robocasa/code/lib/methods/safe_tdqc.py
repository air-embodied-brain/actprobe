"""SAFE-MLP-TDQC + SAFE-LSTM-TDQC (strict variant).

LSTM-TDQC reuses SAFE-LSTM architecture (same nn.LSTM + nn.Linear head).
MLP-TDQC uses a different architecture: IndepMLP with `self.projector` and
sigmoid INSIDE the module. Outputs per-step sigmoid (no cumsum) — different
from SAFE-MLP which outputs cumsum.

Both ckpts come from the `_strict` variant (paper default).
"""
import numpy as np
import torch
import torch.nn as nn

from lib.methods import safe_lstm


class IndepMLP(nn.Module):
    """SAFE-MLP-TDQC architecture (paper-faithful, from train_safe_mlp_tdqc_strict.py).

    Per-step independent MLP with sigmoid INSIDE the module. Output is per-step
    failure probability (T,), not cumsum.
    """
    def __init__(self, input_dim=1024, n_layers=2, hidden_dim=256):
        super().__init__()
        layers = []
        if n_layers == 1:
            layers.append(nn.Linear(input_dim, 1))
        else:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            for _ in range(n_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, 1))
        layers.append(nn.Sigmoid())
        self.projector = nn.Sequential(*layers)

    def forward(self, x, lengths=None):
        # x: (B, T, D) → (B, T)
        return self.projector(x).squeeze(-1)


def load_mlp_ckpt(path, device="cuda:0"):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model = IndepMLP(input_dim=1024, n_layers=2, hidden_dim=256).to(device)
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
