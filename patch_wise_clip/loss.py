import torch
import torch.nn.functional as F


# ───────────────────────── loss helpers ────────────────────────── #

def siglip_loss(sim: torch.Tensor) -> torch.Tensor:
    """
    Symmetric sigmoid-contrastive loss used in SigLIP.

    Args
    ----
    sim : Tensor, shape (B, B)
        Un–scaled cosine-similarity between every (image, text) pair.

    Returns
    -------
    Tensor – scalar loss.
    """
    b = sim.size(0)
    logits = sim                          # we will multiply by temperature outside
    labels = torch.eye(b, device=sim.device)

    # Weight positives so that ∑w_pos = ∑w_neg  ➜  pos_weight = #neg / #pos = b-1
    pos_weight = torch.full((), b - 1, device=sim.device)

    # BCEWithLogits does sigmoid internally
    loss = F.binary_cross_entropy_with_logits(
        logits,
        labels,
        pos_weight=pos_weight,
        reduction="mean",     # same as paper implementation
    )
    return loss

def _contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    """Standard contrastive cross‑entropy used in CLIP."""
    return F.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(sim: torch.Tensor) -> torch.Tensor:
    return (_contrastive_loss(sim) + _contrastive_loss(sim.t())) / 2.0