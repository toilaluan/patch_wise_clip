import torch
import torch.distributed as dist
import time
import wandb
import torch.nn.functional as F

def _contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    """Standard contrastive cross‑entropy used in CLIP."""
    return F.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(sim: torch.Tensor) -> torch.Tensor:
    return (_contrastive_loss(sim) + _contrastive_loss(sim.t())) / 2.0


def siglip_loss(similarity_matrix: torch.Tensor, labels: torch.Tensor, 
                temperature: float = 1.0, bias: float = 0.0) -> torch.Tensor:
    """
    SigLIP loss function optimized for distributed training.
    
    This implementation is designed to work with your existing training code
    where similarity matrices and labels are already computed.
    
    Args:
        similarity_matrix (torch.Tensor): Similarity scores of shape (batch_size, num_targets)
        labels (torch.Tensor): Boolean tensor of shape (batch_size, num_targets) 
                              where True indicates positive pairs
        temperature (float): Temperature parameter for scaling logits
        bias (float): Bias term added to logits before sigmoid
        
    Returns:
        torch.Tensor: SigLIP loss value
    """
    # Scale similarity scores
    logits = similarity_matrix / temperature + bias
    
    # Convert boolean labels to float for loss computation
    targets = labels.float()
    
    # Compute sigmoid loss
    # For positive pairs: -log(sigmoid(logits))
    # For negative pairs: -log(sigmoid(-logits)) = -log(1 - sigmoid(logits))
    pos_loss = -targets * F.logsigmoid(logits)
    neg_loss = -(1 - targets) * F.logsigmoid(-logits)
    
    loss = pos_loss + neg_loss
    
    # Return mean loss across all pairs
    return loss.mean()

def calculate_contrastive_loss(img_features, txt_features, labels, loss_fn):
    """Calculate contrastive loss between image and text features"""
    similarity = img_features @ txt_features.T
    return loss_fn(similarity, labels)

def calculate_patch_losses(img_patch_features, txt_features, labels, loss_fn):
    """Calculate average patch-level contrastive losses"""
    n_patches = img_patch_features.size(1)
    patch_losses = []
    
    for patch_idx in range(n_patches):
        patch_features = img_patch_features[:, patch_idx]
        patch_loss = calculate_contrastive_loss(patch_features, txt_features[:, patch_idx], labels, loss_fn)
        patch_losses.append(patch_loss)
    
    return sum(patch_losses) / n_patches

def calculate_distributed_losses(img_pool, img_patch_features, txt_pool, compressed_text_features, 
                               world_size, rank, device, loss_fn, pool_scale, patch_scale):
    """Calculate losses for distributed training"""
    gathered_txt_pool = torch.empty_like(txt_pool.repeat(world_size, 1))
    gathered_txt_patch_features = torch.empty_like(compressed_text_features.repeat(world_size, 1, 1))
    # txt_pool: (B, D)
    # compressed_text_features: (B, N, D)
    # gathered_txt_pool: (B*world_size, D)
    # gathered_txt_patch_features: (B*world_size, N, D)
    # labels: (B*world_size, B)
    dist.all_gather_into_tensor(gathered_txt_pool, txt_pool)
    dist.all_gather_into_tensor(gathered_txt_patch_features, compressed_text_features)
    labels = torch.zeros(img_pool.size(0) * world_size, img_pool.size(0), device=device)
    device_batch_size = img_pool.size(0)
    start_idx = rank * device_batch_size
    end_idx = start_idx + device_batch_size
    labels[start_idx:end_idx, :] = torch.eye(device_batch_size, device=device)
    
    pool_sim = gathered_txt_pool @ img_pool.T # (B*world_size, B)
    pool_loss = loss_fn(pool_sim * pool_scale.exp(), labels)
    patch_losses = []
    for i in range(img_patch_features.size(1)):
        patch_sim = gathered_txt_patch_features[:, i, :] @ img_patch_features[:, i, :].T # (B*world_size, B)
        patch_losses.append(loss_fn(patch_sim * patch_scale.exp(), labels))
    patch_loss = sum(patch_losses) / img_patch_features.size(1)
    return pool_loss, patch_loss

def calculate_local_losses(img_pool, img_patch_features, txt_pool, compressed_text_features, pool_scale, patch_scale):
    pool_sim = txt_pool @ img_pool.T # (B, B)
    pool_loss = clip_loss(pool_sim * pool_scale.exp())
    patch_losses = []
    for i in range(img_patch_features.size(1)):
        patch_sim = compressed_text_features[:, i, :] @ img_patch_features[:, i, :].T # (B, B)
        patch_losses.append(clip_loss(patch_sim * patch_scale.exp()))
    patch_loss = sum(patch_losses) / img_patch_features.size(1)
    return pool_loss, patch_loss

class LossCalculator:
    """Encapsulates loss calculation logic for cleaner code organization"""
    
    def __init__(self, world_size=1, rank=0):
        self.world_size = world_size
        self.rank = rank
        self.is_distributed = world_size > 1
    
    def calculate_losses(self, img_pool, img_patch_features, txt_pool, compressed_text_features, device, pool_scale, patch_scale):
        return calculate_local_losses(
            img_pool, img_patch_features, txt_pool, compressed_text_features,
            pool_scale, patch_scale
        )