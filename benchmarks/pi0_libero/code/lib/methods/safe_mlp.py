"""SAFE-MLP — paper §4.2 protocol.

Architecture: Linear(1024,256) → ReLU → Linear(256,1).
Forward: g_t = sigmoid(MLP(e_t)); s_t = cumsum(g_t).
No task conditioning.
"""
import numpy as np
import torch
import torch.nn as nn


class SafeMLP(nn.Module):
    def __init__(self, input_dim=1024, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, lengths=None):
        # x: (B, T, 1024)
        g = torch.sigmoid(self.net(x).squeeze(-1))   # (B, T)
        s = torch.cumsum(g, dim=1)                    # (B, T)
        return s


def load_ckpt(path, device="cuda:0"):
    """Returns model. Norm stats are NOT in ckpt — recompute from train split."""
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model = SafeMLP(input_dim=1024, hidden=256).to(device)
    model.load_state_dict(sd)
    model.eval()
    return model


@torch.no_grad()
def score_episodes(model, episodes, device="cuda:0", normed_key="normed_hs"):
    """Eps must have `normed_key` (z-score normalized hidden states (T, 1024))."""
    results = []
    for ep in episodes:
        x = torch.from_numpy(ep[normed_key].astype(np.float32)).unsqueeze(0).to(device)
        lens = torch.tensor([ep["length"]], dtype=torch.long, device=device)
        s = model(x, lens).squeeze(0).cpu().numpy()[:ep["length"]]
        results.append({"scores": s, "label": ep["label"],
                        "length": ep["length"], "task_id": ep["task_id"],
                        "episode_id": ep.get("episode_id")})
    return results
