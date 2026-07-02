"""STAC-Single — deterministic, no ckpt.

Paper version (current implementation): cumsum of chunk_mse per step.
Per-step variant (ablation): just chunk_mse without cumulation.

Reads `raw[:, 4]` (chunk_mse column) from metrics_logs episodes.
"""
import numpy as np

CHUNK_MSE_IDX = 4   # column index in 10-feat raw


def score_episodes(episodes, variant="cumsum"):
    """STAC scores per episode.

    Args:
      episodes: list of {raw: (T, ≥5), label, length, task_id}
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
