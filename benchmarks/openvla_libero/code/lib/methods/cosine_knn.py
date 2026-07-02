"""Cosine-kNN — deterministic, no ckpt.

Reference vectors = train succ hidden states (mean-pool, normalized).
Score(t) = mean cosine distance to k=5 nearest reference vectors at step t.
Episode score = cumsum of per-step distances.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


def score_episodes(train_eps, test_eps, k=5, normed_key="normed_hs"):
    """Cosine-kNN cumsum scores.

    Args:
      train_eps: list with `normed_key` field (train split, used for reference)
      test_eps:  list with `normed_key` field (eps to score)
      k: number of neighbors
      normed_key: name of (T, 1024) normed HS field

    Returns:
      list of {scores: cumsum(per-step), label, length, task_id, episode_id}
    """
    train_succ = [e for e in train_eps if e["label"] == 0]
    if not train_succ:
        raise ValueError("no train succ episodes for kNN reference")
    ref = np.vstack([e[normed_key] for e in train_succ]).astype(np.float32)
    ref_n = normalize(ref, axis=1)

    nn_model = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
    nn_model.fit(ref_n)

    results = []
    for ep in test_eps:
        hs_n = normalize(ep[normed_key].astype(np.float32), axis=1)
        dists, _ = nn_model.kneighbors(hs_n)
        per_step = dists.mean(axis=1)
        cumsum = np.cumsum(per_step).astype(np.float32)
        results.append({"scores": cumsum, "label": ep["label"],
                        "length": ep["length"], "task_id": ep["task_id"],
                        "episode_id": ep.get("episode_id")})
    return results
