"""SAFE-LSTM — paper §4.2 protocol.

Architecture: LSTM(1024 → 256) + Linear(256 → 1).
No task conditioning.
"""
import numpy as np
import torch
import torch.nn as nn


class SafeLSTM(nn.Module):
    def __init__(self, input_dim=1024, hidden=256):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden, num_layers=1, batch_first=True)
        self.fc  = nn.Linear(hidden, 1)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
        return torch.sigmoid(self.fc(h).squeeze(-1))


def load_ckpt(path, device="cuda:0"):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model = SafeLSTM(input_dim=1024, hidden=256).to(device)
    model.load_state_dict(sd)
    model.eval()
    return model


@torch.no_grad()
def score_episodes(model, episodes, device="cuda:0", normed_key="normed_hs"):
    results = []
    for ep in episodes:
        x = torch.from_numpy(ep[normed_key].astype(np.float32)).unsqueeze(0).to(device)
        lens = torch.tensor([ep["length"]], dtype=torch.long, device=device)
        s = model(x, lens).squeeze(0).cpu().numpy()[:ep["length"]]
        results.append({"scores": s, "label": ep["label"],
                        "length": ep["length"], "task_id": ep["task_id"],
                        "episode_id": ep.get("episode_id")})
    return results
