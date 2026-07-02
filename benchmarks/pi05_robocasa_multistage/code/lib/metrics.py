"""Pi0.5+RoboCasa (single-stage) metrics — thin shim over lib_shared.

q-AUC cutoff: per-episode (perep).
"""
from lib_shared.metrics import (
    episode_max, quantile_max, episode_auc,
    perep_cutoff, task_min_cutoff,
    fcp_tau, detect_step, detection_metrics,
    q_auc as _q_auc,
    q_auc_per_category as _q_auc_per_category,
    f1_at_alpha as _f1_at_alpha,
)

# Back-compat alias (pi0.5 code used `per_episode_cutoff`)
per_episode_cutoff = perep_cutoff


def q_auc(test_results, q=0.25, _unused=None):
    return _q_auc(test_results, mode="taskmax", q=q)


def q_auc_per_category(test_results, task_to_category, q=0.25):
    return _q_auc_per_category(test_results, task_to_category, mode="taskmax", q=q)


def f1_at_alpha(test_results, val_results, alpha=0.15, q=0.25):
    return _f1_at_alpha(test_results, val_results, mode="taskmax", alpha=alpha, q=q)
