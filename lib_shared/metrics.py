"""Single source of truth for ActProbe evaluation metrics.

★ Only two q-AUC cutoff modes are supported (per project decision):
    - "perep":   cutoff = max(1, ceil(q * T_ep))           ← per-episode, used by pi0.5 / GR00T
    - "taskmin": cutoff = max(1, round((task_min-1)*q)+1)  ← per-task-min, used by pi0 / openvla / similar benchmarks

Each benchmark's `code/lib/metrics.py` should be a thin shim that picks one mode
via `functools.partial`. See benchmarks/<name>/code/lib/metrics.py for examples.

All other helpers (episode_max, fcp_tau, detect_step, detection_metrics, ...) are
benchmark-agnostic and shared verbatim.
"""
import math
import os
from functools import partial

import numpy as np
from sklearn.metrics import roc_auc_score

# Optional global override: ACTPROBE_Q_AUC_MODE=perep / taskmin.
# When set, all q_auc / q_auc_per_category / f1_at_alpha calls IGNORE the
# benchmark's default mode and use this one instead. Useful for cross-benchmark
# protocol comparisons (e.g., "what if all benchmarks used perep?").
_MODE_OVERRIDE = os.environ.get("ACTPROBE_Q_AUC_MODE")
if _MODE_OVERRIDE not in (None, "perep", "taskmin", "taskmax"):
    raise ValueError(f"ACTPROBE_Q_AUC_MODE must be 'perep' | 'taskmin' | 'taskmax', got {_MODE_OVERRIDE!r}")


# ════════════════════════════════════════════════════════════════════════════════
# Episode-level score reductions
# ════════════════════════════════════════════════════════════════════════════════

def episode_max(scores, length=None):
    """Episode score = max(score_t) over the actual episode length."""
    if length is None:
        length = len(scores)
    return float(scores[:length].max())


def quantile_max(scores, length, cutoff):
    """Truncated max: max(score_t) for t in [0, cutoff)."""
    cutoff = max(1, min(cutoff, length))
    return float(scores[:cutoff].max())


# ════════════════════════════════════════════════════════════════════════════════
# Cutoff formulas — ONLY two supported modes
# ════════════════════════════════════════════════════════════════════════════════

def perep_cutoff(T_ep, q):
    """Per-episode cutoff (pi0.5 / GR00T): ceil(q * T_ep)."""
    return max(1, int(math.ceil(q * T_ep)))


def task_min_cutoff(task_min, q, T_ep):
    """Task-min cutoff (pi0 / openvla / similar benchmarks): round((task_min-1)*q)+1, clamped to T_ep."""
    cutoff = round((task_min - 1) * q) + 1
    return max(1, min(cutoff, T_ep))


def task_max_cutoff(task_max, q, T_ep):
    """Task-max (timeout) cutoff: ceil(q * task_max), clamped to T_ep.
    task_max is the maximum episode length within the task (≈ timeout)."""
    cutoff = max(1, int(math.ceil(q * task_max)))
    return min(cutoff, T_ep)


# ════════════════════════════════════════════════════════════════════════════════
# AUC
# ════════════════════════════════════════════════════════════════════════════════

def episode_auc(test_results):
    """AUROC using max(score_t) per ep."""
    scores = [episode_max(r["scores"], r["length"]) for r in test_results]
    labels = [r["label"] for r in test_results]
    if len(set(labels)) < 2:
        return 0.5
    return float(roc_auc_score(labels, scores))


def q_auc(test_results, *, mode, task_min_steps=None, q=0.25):
    """★ q-AUC with explicit cutoff mode.

    Args:
      test_results: list of {scores: (T,), label, length, task_id}
      mode: "perep" or "taskmin"
      task_min_steps: dict[task_id, int]; required iff mode == "taskmin"
      q: cutoff fraction (paper main = 0.25)

    Returns: AUROC float in [0, 1].
    """
    if not test_results:
        return 0.5
    if _MODE_OVERRIDE is not None:
        mode = _MODE_OVERRIDE
    # Auto-compute task_min from test_results if mode=taskmin but not passed
    # (lets ACTPROBE_Q_AUC_MODE=taskmin work on benchmarks whose eval defaults to perep).
    if mode == "taskmin" and task_min_steps is None:
        from collections import defaultdict
        _tmin = defaultdict(lambda: float("inf"))
        for r in test_results:
            tid = r["task_id"]
            if r["length"] < _tmin[tid]:
                _tmin[tid] = r["length"]
        task_min_steps = {k: int(v) for k, v in _tmin.items()}
    # For taskmax mode: compute task_max from test_results (max ep length per task ≈ timeout)
    task_max_steps = None
    if mode == "taskmax":
        from collections import defaultdict
        _tmax = defaultdict(int)
        for r in test_results:
            tid = r["task_id"]
            if r["length"] > _tmax[tid]:
                _tmax[tid] = r["length"]
        task_max_steps = dict(_tmax)
    ep_scores, ep_labels = [], []
    for r in test_results:
        T = r["length"]
        if mode == "perep":
            cutoff = perep_cutoff(T, q)
        elif mode == "taskmin":
            ms = task_min_steps.get(r["task_id"], T)
            cutoff = task_min_cutoff(ms, q, T)
        elif mode == "taskmax":
            tmax = task_max_steps.get(r["task_id"], T)
            cutoff = task_max_cutoff(tmax, q, T)
        else:
            raise ValueError(f"unknown mode: {mode}")
        ep_scores.append(quantile_max(r["scores"], T, cutoff))
        ep_labels.append(r["label"])
    if len(set(ep_labels)) < 2:
        return 0.5
    return float(roc_auc_score(ep_labels, ep_scores))


def q_auc_per_category(test_results, task_to_category, *, mode,
                        task_min_steps=None, q=0.25):
    """Per-category q-AUC. Returns dict[cat, auc] + 'Overall'."""
    from collections import defaultdict
    cat_results = defaultdict(list)
    for r in test_results:
        cat = task_to_category.get(r["task_id"])
        if cat is None:
            continue
        cat_results[cat].append(r)
    out = {}
    for cat, results in cat_results.items():
        out[cat] = q_auc(results, mode=mode, task_min_steps=task_min_steps, q=q)
    out["Overall"] = q_auc(test_results, mode=mode,
                            task_min_steps=task_min_steps, q=q)
    return out


# ════════════════════════════════════════════════════════════════════════════════
# Functional CP threshold + detection
# ════════════════════════════════════════════════════════════════════════════════

def fcp_tau(succ_scores_max, alpha):
    """Functional Conformal Prediction threshold from val-success episode-max scores."""
    n = len(succ_scores_max)
    if n == 0:
        return float("inf")
    q = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(np.asarray(succ_scores_max), q))


def detect_step(scores, tau, max_step=None):
    """First step where scores >= tau (within max_step). Returns int index or None."""
    if max_step is None:
        max_step = len(scores)
    sc = np.asarray(scores[:max_step])
    hits = np.where(sc >= tau)[0]
    return int(hits[0]) if len(hits) > 0 else None


def detection_metrics(test_results, tau):
    """Detection rate + normalized T-det per episode."""
    per_ep = []
    n_fail, n_det = 0, 0
    tdets = []
    for r in test_results:
        T = r["length"]
        det = detect_step(r["scores"], tau, max_step=T)
        per_ep.append({"task_id": r["task_id"],
                       "ep_id": r.get("episode_id"),
                       "label": r["label"], "T": T,
                       "det_step": det,
                       "tdet_norm": (det / T) if det is not None else 1.0})
        if r["label"] == 1:
            n_fail += 1
            if det is not None:
                n_det += 1
                tdets.append(det / T)
            else:
                tdets.append(1.0)
    return {
        "det_rate": (n_det / n_fail) if n_fail > 0 else float("nan"),
        "tdet":    (float(np.mean(tdets)) if tdets else float("nan")),
        "n_det":   n_det,
        "n_fail":  n_fail,
        "per_ep":  per_ep,
    }


# ════════════════════════════════════════════════════════════════════════════════
# F1 / balanced accuracy at calibrated threshold
# ════════════════════════════════════════════════════════════════════════════════

def f1_at_alpha(test_results, val_results, *, mode,
                 task_min_steps=None, alpha=0.15, q=0.25):
    """F1 with τ from val-success q-scores at alpha. Same cutoff for val and test."""
    from sklearn.metrics import f1_score, balanced_accuracy_score
    if _MODE_OVERRIDE is not None:
        mode = _MODE_OVERRIDE

    # Auto-compute task_min / task_max from test+val if not passed
    if mode == "taskmin" and task_min_steps is None:
        from collections import defaultdict
        _tmin = defaultdict(lambda: float("inf"))
        for r in list(test_results) + list(val_results):
            if r["length"] < _tmin[r["task_id"]]:
                _tmin[r["task_id"]] = r["length"]
        task_min_steps = {k: int(v) for k, v in _tmin.items()}
    task_max_steps = None
    if mode == "taskmax":
        from collections import defaultdict
        _tmax = defaultdict(int)
        for r in list(test_results) + list(val_results):
            if r["length"] > _tmax[r["task_id"]]:
                _tmax[r["task_id"]] = r["length"]
        task_max_steps = dict(_tmax)

    def _cutoff(r):
        T = r["length"]
        if mode == "perep":
            return perep_cutoff(T, q)
        if mode == "taskmin":
            ms = task_min_steps.get(r["task_id"], T)
            return task_min_cutoff(ms, q, T)
        if mode == "taskmax":
            tm = task_max_steps.get(r["task_id"], T)
            return task_max_cutoff(tm, q, T)
        raise ValueError(mode)

    val_succ_q = []
    for r in val_results:
        if r["label"] != 0:
            continue
        val_succ_q.append(quantile_max(r["scores"], r["length"], _cutoff(r)))
    if not val_succ_q:
        return {"f1": 0.0, "bal_acc": 0.5, "tau": float("nan")}

    tau = fcp_tau(np.asarray(val_succ_q), alpha)

    test_q, test_labels = [], []
    for r in test_results:
        test_q.append(quantile_max(r["scores"], r["length"], _cutoff(r)))
        test_labels.append(r["label"])

    preds = (np.array(test_q) >= tau).astype(int)
    labels = np.array(test_labels)
    if len(set(labels)) < 2:
        return {"f1": 0.0, "bal_acc": 0.5, "tau": tau}
    return {
        "f1": float(f1_score(labels, preds)) * 100,
        "bal_acc": float(balanced_accuracy_score(labels, preds)) * 100,
        "tau": tau,
    }


# ════════════════════════════════════════════════════════════════════════════════
# Full-episode-max calibrated F1 (used by openvla; ⚠️ length-leaks)
# ════════════════════════════════════════════════════════════════════════════════

def f1_at_alpha_fullep(test_results, val_results, alpha=0.15):
    """Full-ep max + plain quantile (no fcp correction).

    Paper protocol for OpenVLA Table 1 bal_acc / t_det / f1@α=0.15 columns.

    ⚠️ Length-leaks via `r["scores"].max()` over the full episode (longer eps have
    more chances to peak above tau). Paper-faithful for OpenVLA only.
    """
    from sklearn.metrics import f1_score, balanced_accuracy_score

    val_succ_max = np.array([r["scores"][:r["length"]].max()
                              for r in val_results if r["label"] == 0])
    if len(val_succ_max) == 0:
        return {"f1": 0.0, "bal_acc": 0.5, "tau": float("nan"), "t_det": 1.0}

    tau = float(np.quantile(val_succ_max, 1 - alpha))

    test_max = np.array([r["scores"][:r["length"]].max() for r in test_results])
    labels = np.array([r["label"] for r in test_results])
    preds = (test_max >= tau).astype(int)

    det_times = []
    for r in test_results:
        if r["label"] != 1:
            continue
        sc = r["scores"][:r["length"]]
        if sc.max() >= tau:
            hits = np.where(sc >= tau)[0]
            det_times.append(hits[0] / r["length"])
        else:
            det_times.append(1.0)

    if len(set(labels)) < 2:
        return {"f1": 0.0, "bal_acc": 0.5, "tau": tau,
                "t_det": float(np.mean(det_times)) if det_times else 1.0}
    return {
        "f1": float(f1_score(labels, preds)) * 100,
        "bal_acc": float(balanced_accuracy_score(labels, preds)) * 100,
        "tau": tau,
        "t_det": float(np.mean(det_times)) if det_times else 1.0,
    }


# ════════════════════════════════════════════════════════════════════════════════
# Helpers for benchmark-local shims (avoid boilerplate)
# ════════════════════════════════════════════════════════════════════════════════

def bind_mode(mode):
    """Returns a dict of partial-bound q_auc / q_auc_per_category / f1_at_alpha.

    Usage in benchmarks/<bm>/code/lib/metrics.py:
        from lib_shared.metrics import *
        from lib_shared.metrics import bind_mode
        _b = bind_mode("perep")     # or "taskmin"
        q_auc                 = _b["q_auc"]
        q_auc_per_category    = _b["q_auc_per_category"]
        f1_at_alpha           = _b["f1_at_alpha"]
    """
    return {
        "q_auc":              partial(q_auc, mode=mode),
        "q_auc_per_category": partial(q_auc_per_category, mode=mode),
        "f1_at_alpha":        partial(f1_at_alpha, mode=mode),
    }
