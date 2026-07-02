"""GR00T+RoboCasa metrics — thin shim over lib_shared.

q-AUC cutoff: per-episode (perep). Pi0.5-aligned.
"""
from lib_shared.metrics import (
    episode_max, quantile_max, episode_auc,
    perep_cutoff, task_min_cutoff,
    fcp_tau, detect_step, detection_metrics,
    q_auc as _q_auc,
    q_auc_per_category as _q_auc_per_category,
    f1_at_alpha as _f1_at_alpha,
)


def q_auc(test_results, task_min_steps=None, q=0.25):
    """Per-episode q-AUC. `task_min_steps` accepted for back-compat but ignored."""
    return _q_auc(test_results, mode="taskmax", q=q)


def q_auc_per_category(test_results, task_min_steps=None, task_to_category=None, q=0.25):
    """Per-category q-AUC. `task_min_steps` accepted for back-compat but ignored."""
    if task_to_category is None and task_min_steps is not None and isinstance(task_min_steps, dict):
        # old signature was q_auc_per_category(test, task_min, task2cat, q)
        # if no task_to_category passed positionally, this means caller used the dict
        # as positional arg-2. Need to be smart: assume task_to_category was actually 3rd.
        pass
    return _q_auc_per_category(test_results, task_to_category, mode="taskmax", q=q)


def f1_at_alpha(test_results, val_results, task_min_steps=None, alpha=0.15, q=0.25):
    return _f1_at_alpha(test_results, val_results, mode="taskmax", alpha=alpha, q=q)
