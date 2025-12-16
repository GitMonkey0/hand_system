import torch
import torch.nn as nn
import torch.nn.functional as F


def ce_loss(logits: torch.Tensor, targets: torch.Tensor):
    """
    logits: [B,40,26] or [B,T,40,26]
    targets: [B,40] or [B,T,40]
    """
    if logits.dim() == 3:
        b, p, c = logits.shape
        logits_flat = logits.view(b * p, c)
        targets_flat = targets.view(b * p)
    else:
        b, t, p, c = logits.shape
        logits_flat = logits.view(b * t * p, c)
        targets_flat = targets.view(b * t * p)
    return F.cross_entropy(logits_flat, targets_flat)