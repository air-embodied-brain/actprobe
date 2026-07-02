"""Pi0+LIBERO metrics — thin shim over lib_shared.

q-AUC cutoff: task_min.
"""
from lib_shared.metrics import (
    episode_max, quantile_max, episode_auc,
    perep_cutoff, task_min_cutoff,
    fcp_tau, detect_step, detection_metrics,
    q_auc as _q_auc,
    q_auc_per_category as _q_auc_per_category,
    f1_at_alpha as _f1_at_alpha,
)


def q_auc(test_results, task_min_steps, q=0.25):
    return _q_auc(test_results, mode="taskmax",
                  task_min_steps=task_min_steps, q=q)


def q_auc_per_category(test_results, task_min_steps, task_to_category, q=0.25):
    return _q_auc_per_category(test_results, task_to_category, mode="taskmax",
                                task_min_steps=task_min_steps, q=q)


def f1_at_alpha(test_results, val_results, task_min_steps, alpha=0.15, q=0.25):
    return _f1_at_alpha(test_results, val_results, mode="taskmax",
                        task_min_steps=task_min_steps, alpha=alpha, q=q)
