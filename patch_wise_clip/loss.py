import torch
import torch.distributed as dist
import time
import wandb
import torch.nn.functional as F


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
        patch_loss = calculate_contrastive_loss(patch_features, txt_features, labels, loss_fn)
        patch_losses.append(patch_loss)
    
    return sum(patch_losses) / n_patches

def calculate_single_batch_losses(img_pool, img_patch_features, txt_pool, compressed_text_features, device, loss_fn):
    """Calculate losses for single GPU training"""
    batch_size = img_pool.size(0)
    labels = torch.eye(batch_size, device=device)
    
    # Pool-level loss
    pool_loss = calculate_contrastive_loss(img_pool, txt_pool, labels, loss_fn)
    
    # Patch-level loss
    patch_loss = calculate_patch_losses(img_patch_features, compressed_text_features, labels, loss_fn)
    
    return pool_loss, patch_loss


def calculate_distributed_losses(img_pool, img_patch_features, txt_pool, compressed_text_features, 
                               world_size, rank, device, loss_fn):
    """Calculate losses for distributed training"""
    local_batch_size = img_pool.size(0)
    shared_txt_pool = txt_pool
    shared_compressed_text_features = compressed_text_features
    dist.broadcast(shared_txt_pool, src=rank)
    dist.broadcast(shared_compressed_text_features, src=rank)
    all_pool_losses = []
    all_patch_losses = []
    for i in range(world_size):
        shared_txt_pool = dist.recv(shared_txt_pool, src=i)
        shared_compressed_text_features = dist.recv(shared_compressed_text_features, src=i)
        pool_loss, patch_loss = calculate_single_batch_losses(img_pool, img_patch_features, shared_txt_pool, shared_compressed_text_features, device, loss_fn)
        all_pool_losses.append(pool_loss)
        all_patch_losses.append(patch_loss)
    return sum(all_pool_losses) / world_size, sum(all_patch_losses) / world_size

class LossCalculator:
    """Encapsulates loss calculation logic for cleaner code organization"""
    
    def __init__(self, loss_fn=siglip_loss, world_size=1, rank=0):
        self.loss_fn = loss_fn
        self.world_size = world_size
        self.rank = rank
        self.is_distributed = world_size > 1
    
    def calculate_losses(self, img_pool, img_patch_features, txt_pool, compressed_text_features, device):
        """Main entry point for loss calculation"""
        if self.is_distributed:
            return calculate_distributed_losses(
                img_pool, img_patch_features, txt_pool, compressed_text_features,
                self.world_size, self.rank, device, self.loss_fn
            )
        else:
            return calculate_single_batch_losses(
                img_pool, img_patch_features, txt_pool, compressed_text_features,
                device, self.loss_fn
            )