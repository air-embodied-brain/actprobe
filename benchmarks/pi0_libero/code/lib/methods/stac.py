"""STAC-Single — deterministic, no ckpt.

Paper version: cumsum of chunk_mse per step.

⚠️ chunk_mse 列索引从 lib.data.METRIC_KEYS 取（key 驱动），不 hardcode。
"""
import numpy as np

from lib.data import METRIC_KEYS

CHUNK_MSE_IDX = METRIC_KEYS.index("chunk_mse")   # = 1 in pi0 canonical schema


def score_episodes(episodes, variant="cumsum"):
    """STAC scores per episode.

    Args:
      episodes: list of {raw: (T, ≥CHUNK_MSE_IDX+1), label, length, task_id}
      variant: "cumsum" (paper) | "perstep" (ablation)

    Returns:
      list of {scores, label, length, task_id, episode_id}
    """
    results = []
    for ep in episodes:
        chunk_mse = ep["raw"][:, CHUNK_MSE_IDX].astype(np.float32)
        T = len(chunk_mse)
        if variant == "cumsum":
            scores = np.cumsum(chunk_mse)
        elif variant == "perstep":
            scores = chunk_mse
        else:
            raise ValueError(f"unknown variant: {variant}")
        results.append({"scores": scores, "label": ep["label"],
                        "length": T, "task_id": ep["task_id"],
                        "episode_id": ep.get("episode_id")})
    return results
