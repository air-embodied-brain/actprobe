"""
TDQC-strict utilities: paper-faithful TD(0) loss + target network.

The TD loss block is **transcribed verbatim** from the official TDQC repo
(`failure_prob/model/indep.py:131-202` and `failure_prob/model/lstm.py:164-222`),
with only mechanical glue changes: we accept tensors directly (instead of
omegaconf cfg + batch dict).

Paper-default hyperparameters (white-box, hidden_states/10-feat input):
    target_update_freq = 10        # sync target net every 10 train batches
    td_horizon         = 1          # TD(0)
    cumsum             = False      # output direct sigmoid prob
    final_act_layer    = "sigmoid"
    use_time_weighting = False     # not used in white-box main result
    lambda_bce_reg     = 0          # paper's main TDQC ablates BCE reg out

Score convention (paper): q ≈ P(failure)
    success_labels ∈ {0, 1}: 1 = success, 0 = failure
    terminal target = 1 - success_labels (= failure indicator)
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


# ════════════════════════════════════════════════════════════════════════════════
# §1. Target-network wrapper (paper §C.4 + indep.py:_copy_params_to_target)
# ════════════════════════════════════════════════════════════════════════════════

class TargetNet:
    """Holds a frozen copy of `main_net`. Sync via hard copy every K steps."""

    def __init__(self, main_net: nn.Module):
        self.target = copy.deepcopy(main_net)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.target.eval()
        self.steps = 0

    def sync_from(self, main_net: nn.Module):
        for tp, p in zip(self.target.parameters(), main_net.parameters()):
            tp.data.copy_(p.data)

    def maybe_step(self, main_net: nn.Module, freq: int = 10):
        self.steps += 1
        if self.steps % freq == 0:
            self.sync_from(main_net)

    def __call__(self, *args, **kwargs):
        with torch.no_grad():
            return self.target(*args, **kwargs)


# ════════════════════════════════════════════════════════════════════════════════
# §2. Paper-verbatim TD(0) target & loss
# ════════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def build_td0_targets(target_q_values: torch.Tensor,
                      success_labels: torch.Tensor,
                      valid_masks: torch.Tensor) -> torch.Tensor:
    """
    Build TD(0) targets exactly as in paper repo `IndepModel.forward_compute_loss`
    (failure_prob/model/indep.py:139-153) and `LstmModel.forward_compute_loss`
    (failure_prob/model/lstm.py:172-186).

    Args:
        target_q_values: (B, T)  output of frozen target net (not detached yet)
        success_labels:  (B,)    1 = success, 0 = failure
        valid_masks:     (B, T)  1 = valid step, 0 = padding

    Returns:
        target_scores:   (B, T)  per-step bootstrap target  (= q_target[t+1] with
                                 terminal anchored to 1 - success_label)
    """
    B, T = target_q_values.shape
    device = target_q_values.device

    target_q_values = target_q_values * valid_masks       # zero out padding
    last_valid_idx = valid_masks.sum(dim=1).long() - 1     # (B,)

    # Terminal anchoring: at last valid step, target = 1 - success_labels
    target_q_values[torch.arange(B, device=device), last_valid_idx] = 1 - success_labels.float()

    # If next position is within sequence bounds, also overwrite it (mirrors paper code)
    next_idx = last_valid_idx + 1
    within_bounds = next_idx < T
    if within_bounds.any():
        b_idx = torch.arange(B, device=device)[within_bounds]
        target_q_values[b_idx, next_idx[within_bounds]] = (1 - success_labels.float())[within_bounds]

    # Shift target by 1 step (target_t = q_target[t+1]); pad last column with itself
    target_with_terminal = torch.cat([target_q_values, target_q_values[:, -1:]], dim=-1)  # (B, T+1)
    target_scores = target_with_terminal[:, 1:]                                            # (B, T)
    return target_scores


def aggregate_monitor_loss(losses: torch.Tensor,
                           valid_masks: torch.Tensor,
                           success_labels: torch.Tensor,
                           weights=(1.0, 1.0)) -> torch.Tensor:
    """
    Per-step losses → per-seq → per-class weighted scalar.
    Paper convention: weights = (w_fail, w_success). Default (1, 1).
    Verbatim from `failure_prob/model/utils.py:aggregate_monitor_loss`.
    """
    B = losses.shape[0]
    seq_loss = (losses * valid_masks).sum(-1) / valid_masks.sum(-1).clamp(min=1.0)  # (B,)
    fail_mask = (success_labels == 0).float()
    succ_mask = (success_labels == 1).float()
    fail_loss = (fail_mask * seq_loss).sum()
    succ_loss = (succ_mask * seq_loss).sum()
    monitor = (weights[0] * fail_loss + weights[1] * succ_loss) / B
    return monitor


def tdqc_loss(main_net_scores: torch.Tensor,
              target_net_scores: torch.Tensor,
              success_labels: torch.Tensor,
              valid_masks: torch.Tensor,
              weights=(1.0, 1.0)) -> torch.Tensor:
    """
    Compute paper-faithful TD(0) loss.

    Args:
        main_net_scores:  (B, T)  sigmoid output of main net (q ≈ P(fail))
        target_net_scores:(B, T)  sigmoid output of target net (frozen)
        success_labels:   (B,)    1 = success, 0 = failure
        valid_masks:      (B, T)  1 = valid step

    Returns:
        scalar loss
    """
    target_scores = build_td0_targets(target_net_scores, success_labels, valid_masks).detach()
    losses = F.mse_loss(main_net_scores, target_scores, reduction='none')   # (B, T)
    return aggregate_monitor_loss(losses, valid_masks, success_labels, weights=weights)
