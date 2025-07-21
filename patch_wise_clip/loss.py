import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.nn


def _build_global_offsets(local_bs: int, device) -> torch.Tensor:
    """
    All‑gather local batch sizes and compute a global index offset for each worker.
    Works even with uneven last batches.
    """
    if not torch.distributed.is_initialized():
        return torch.arange(local_bs, device=device)

    world = torch.distributed.get_world_size()
    sizes = torch.tensor([local_bs], device=device)
    sizes_all = [torch.zeros_like(sizes) for _ in range(world)]
    torch.distributed.all_gather(sizes_all, sizes)
    sizes_all = torch.stack(sizes_all).cpu()
    offset = int(sizes_all[: torch.distributed.get_rank()].sum())
    return offset + torch.arange(local_bs, device=device)


class CLIPLoss(nn.Module):
    """
    Contrastive loss used by OpenAI CLIP but with:
    * gradient‑supporting all_gather (torch.distributed.nn)
    * uneven‑batch‑size safety
    """

    def __init__(self):
        super().__init__()
        self.cached_bs = None
        self.cached_global_bs = None
        self.register_buffer(
            "labels", torch.empty(0, dtype=torch.long), persistent=False
        )

    def _maybe_refresh_labels(self, local_bs, global_bs, device):
        if (local_bs != self.cached_bs) or (global_bs != self.cached_global_bs):
            self.labels = _build_global_offsets(local_bs, device)
            self.cached_bs, self.cached_global_bs = local_bs, global_bs

    def forward(self, img, txt, logit_scale):
        # normalise
        img = F.normalize(img, dim=-1, p=2)
        txt = F.normalize(txt, dim=-1, p=2)

        # distributed gather
        if torch.distributed.is_initialized():
            img_all = torch.cat(torch.distributed.nn.all_gather(img), dim=0)
            txt_all = torch.cat(torch.distributed.nn.all_gather(txt), dim=0)
            global_bs = img_all.size(0)
        else:
            img_all, txt_all, global_bs = img, txt, img.size(0)

        local_bs = img.size(0)
        self._maybe_refresh_labels(local_bs, global_bs, img.device)

        # logits
        logits_per_image = logit_scale.exp() * img @ txt_all.t()
        logits_per_text = logit_scale.exp() * txt @ img_all.t()

        loss = 0.5 * (
            F.cross_entropy(logits_per_image, self.labels)
            + F.cross_entropy(logits_per_text, self.labels)
        )

        with torch.no_grad():
            correct = (logits_per_image.argmax(dim=-1) == self.labels).sum()
            acc = correct.float() / local_bs * 100

        return loss, acc


class LossCalculator(nn.Module):
    """
    Computes:
      • global/pool‑level CLIP loss
      • patch‑level loss averaged over all patches
    """

    def __init__(self):
        super().__init__()
        self.clip_loss = CLIPLoss()

    def forward(
        self,
        img_pool,  # (B, D)
        img_patch_feats,  # (B, P, D)
        txt_pool,  # (B, D)
        txt_compressed_feats,  # (B, P, D)
        pool_scale,
        patch_scale,
        patch_diversity_scale=1.0,
    ):
        pool_loss, pool_acc = self.clip_loss(img_pool, txt_pool, pool_scale)
        img_patch_similarity = torch.einsum("bpd,bqd->bpq", img_patch_feats, img_patch_feats)
        img_patch_diversity = img_patch_similarity * patch_diversity_scale.exp()
        labels = torch.arange(img_patch_diversity.size(1), device=img_patch_diversity.device)
        labels = labels.repeat(img_patch_diversity.size(0), 1)
        img_patch_diversity = F.cross_entropy(img_patch_diversity, labels)
        img_patch_diversity = img_patch_diversity.mean()

        # Flatten patches so we gather only once
        B, P, D = img_patch_feats.shape
        if txt_compressed_feats is not None:
            total_patch_loss = 0.0
            for i in range(P):
                img_flat = img_patch_feats[:, i, :]
                txt_flat = txt_compressed_feats[:, i, :]
                patch_loss, _ = self.clip_loss(img_flat, txt_flat, patch_scale)
                total_patch_loss += patch_loss
            patch_loss = total_patch_loss / P  # average over patches
        else:
            patch_loss = torch.tensor(0.0)

        return pool_loss, patch_loss, pool_acc, img_patch_diversity, img_patch_similarity
