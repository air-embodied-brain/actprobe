"""LogPzO — UNet-based density estimator.

Two UNets (succ + fail) trained to denoise hidden states.
Score(t) = ||hs + v_succ(hs)||² - ||hs + v_fail(hs)||²
Episode score = per-step (no cumsum).
"""
import numpy as np
import torch

from lib.methods.conditional_unet1d import ConditionalUnet1D

LOGPZO_IN_DIM = 20


def _make_unet():
    return ConditionalUnet1D(
        input_dim=LOGPZO_IN_DIM,
        local_cond_dim=None,
        global_cond_dim=None,
        diffusion_step_embed_dim=128,
        down_dims=[256, 512, 1024],
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=False,
    )


def _adjust_shape(x):
    """Reshape (T, 1024) → (T, H, LOGPZO_IN_DIM=20). Pads to multiple of 4*20."""
    T = x.shape[0]
    total = x.shape[1]
    rem = total % LOGPZO_IN_DIM
    if rem:
        x = torch.cat([x, torch.zeros(T, LOGPZO_IN_DIM - rem, device=x.device)], 1)
        total += LOGPZO_IN_DIM - rem
    H = total // LOGPZO_IN_DIM
    if H % 4:
        extra = (4 - H % 4) * LOGPZO_IN_DIM
        x = torch.cat([x, torch.zeros(T, extra, device=x.device)], 1)
    return x.reshape(T, -1, LOGPZO_IN_DIM)


def load_ckpt(path, device="cuda:0"):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net_succ = _make_unet().to(device); net_succ.load_state_dict(ckpt["succ"])
    net_fail = _make_unet().to(device); net_fail.load_state_dict(ckpt["fail"])
    net_succ.eval(); net_fail.eval()
    return net_succ, net_fail


@torch.no_grad()
def score_episodes(net_succ, net_fail, episodes, device="cuda:0", normed_key="normed_hs"):
    results = []
    for ep in episodes:
        hs = torch.from_numpy(ep[normed_key].astype(np.float32)).to(device)
        T  = hs.shape[0]
        hs_adj = _adjust_shape(hs)
        ts = torch.zeros(T, device=device, dtype=torch.long)
        v_succ = net_succ(hs_adj, ts)
        v_fail = net_fail(hs_adj, ts)
        s_succ = (hs_adj + v_succ).reshape(T, -1).pow(2).sum(-1)
        s_fail = (hs_adj + v_fail).reshape(T, -1).pow(2).sum(-1)
        scores = (s_succ - s_fail).cpu().numpy()
        results.append({"scores": scores, "label": ep["label"],
                        "length": ep["length"], "task_id": ep["task_id"],
                        "episode_id": ep.get("episode_id")})
    return results
