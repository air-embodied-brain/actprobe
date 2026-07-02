"""Cosine-kNN — deterministic, no ckpt.

Paper protocol (run_groot_baselines.py::eval_cosine_knn_quantile):
  - Gallery = mean-pool each train-success episode (one vector per ep), L2-norm
  - Per-step score = mean cosine distance to k=5 nearest gallery vectors
  - Episode score = cum_max (max-so-far) of per-step scores
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors


def score_episodes(train_eps, test_eps, k=5, hs_key="normed_hs"):
    """Cosine-kNN per-step scores with cum_max aggregation.

    Paper protocol (run_groot_baselines.eval_cosine_knn_quantile):
      - Input HS goes through z-score (apply_norm overwrites ep["raw"])
      - Gallery: mean-pool each succ ep into one vector, then L2-normalize
      - Per-step score: mean cosine distance to k=5 nearest gallery vectors
      - Episode score: cum_max (max-so-far) of per-step scores

    Args:
      train_eps: list with `hs_key` field (z-scored HS via apply_norm)
      test_eps:  list with `hs_key` field
      k: number of neighbors

    Returns:
      list of {scores: cum_max(per-step), label, length, task_id, episode_id}
    """
    # Gallery: ONE vector per train succ ep (mean-pool over time)
    gallery = []
    for ep in train_eps:
        if ep["label"] == 0:
            gallery.append(ep[hs_key].mean(axis=0))
    if not gallery:
        raise ValueError("no train succ episodes for kNN reference")
    gallery = np.stack(gallery).astype(np.float32)
    # L2-normalize per row (cosine == dot product on unit vectors)
    gallery_norm = gallery / (np.linalg.norm(gallery, axis=1, keepdims=True) + 1e-8)

    nn_model = NearestNeighbors(n_neighbors=min(k, len(gallery_norm)), metric="cosine")
    nn_model.fit(gallery_norm)

    results = []
    for ep in test_eps:
        raw = ep[hs_key].astype(np.float32)
        raw_norm = raw / (np.linalg.norm(raw, axis=1, keepdims=True) + 1e-8)
        dists, _ = nn_model.kneighbors(raw_norm)
        per_step = dists.mean(axis=1)            # (T,) avg cosine dist to k neighbors
        cum_max = np.maximum.accumulate(per_step).astype(np.float32)
        results.append({"scores": cum_max, "label": ep["label"],
                        "length": ep["length"], "task_id": ep["task_id"],
                        "episode_id": ep.get("episode_id")})
    return results
