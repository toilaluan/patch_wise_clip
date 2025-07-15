import os
import math
import time
import json
import logging
from pathlib import Path
from argparse import ArgumentParser
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from tqdm import tqdm
import wandb

from patch_wise_clip.model import PatchWiseCLIP
from patch_wise_clip.data import ImageNetDataset, ClipDataset
from patch_wise_clip.zeroshot_metadata import IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES
from patch_wise_clip.loss import LossCalculator

def setup_logging(local_rank: int, log_dir: str) -> logging.Logger:
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [Rank %(rank)d] - %(message)s'
    )
    
    # Setup logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (only for rank 0)
    if local_rank == 0:
        file_handler = logging.FileHandler(os.path.join(log_dir, 'training.log'))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Add rank to logger
    logger = logging.LoggerAdapter(logger, {'rank': local_rank})
    
    return logger


def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        
        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def create_transforms():
    """Create training and validation transforms"""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        normalize
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    
    return train_transform, val_transform


def create_scheduler(optimizer, total_steps: int, warmup_steps: int, min_lr: float, max_lr: float, rank: int = 0):
    """Create learning rate scheduler with warmup and cosine decay"""
    print(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}, Min LR: {min_lr}, Max LR: {max_lr}")
    
    def get_lr(step):
        if step < warmup_steps:
            # Linear warmup
            if rank == 0:
                print(f"Linear warmup: {step / warmup_steps}")
            return min_lr + (max_lr - min_lr) * step / warmup_steps
        else:
            # Cosine decay after warmup
            decay_steps = total_steps - warmup_steps
            current_decay_step = step - warmup_steps
            cosine_factor = 0.5 * (1 + math.cos(math.pi * current_decay_step / decay_steps))
            if rank == 0:
                print(f"Cosine factor: {cosine_factor}")
            return min_lr + (max_lr - min_lr) * cosine_factor
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)


def save_checkpoint(model, optimizer, scheduler, epoch: int, best_accuracy: float, 
                   args, save_dir: str, is_best: bool = False):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_accuracy': best_accuracy,
        'args': vars(args)
    }
    
    checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    
    if is_best:
        best_path = os.path.join(save_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)


def load_checkpoint(model, optimizer, scheduler, checkpoint_path: str, device: torch.device):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['epoch'], checkpoint['best_accuracy']


def train_one_epoch(model, optimizer, scheduler, train_loader, 
                   epoch: int, args, logger, device: torch.device, dtype: torch.dtype, local_rank: int = 0):
    """Train model for one epoch with refactored loss calculation"""
    model.train()

    total_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_parameters = sum(p.numel() for p in model.parameters())
    print(f"Total trainable parameters: {total_trainable_parameters / total_parameters * 100:.2f}%")
    
    # Initialize metrics tracking
    metrics = {
        'total_loss': 0.0,
        'pool_loss': 0.0,
        'patch_loss': 0.0,
        'num_batches': 0
    }
    
    # Parse loss weights
    loss_weights = [float(w) for w in args.loss_weights.split(',')]
    pool_weight, patch_weight = loss_weights[0], loss_weights[1]
    
    # Initialize loss calculator
    loss_calculator = LossCalculator(
        world_size=getattr(args, 'world_size', 1),
        rank=getattr(args, 'rank', 0)
    )
    
    start_time = time.time()
    
    for step, batch in enumerate(train_loader):
        batch_start_time = time.time()
        
        # Move data to device
        pixel_values, input_ids, attention_mask, meta_tensor = batch
        pixel_values = pixel_values.to(device, dtype=dtype)
        input_ids = input_ids.to(device, dtype=torch.long)
        attention_mask = attention_mask.to(device, dtype=torch.bool)
        meta_tensor = meta_tensor.to(device, dtype=dtype)

        optimizer.zero_grad()
        
        # Forward pass
        img_pool, img_patch_features = model.module.encode_image(pixel_values)
        txt_pool, compressed_text_features = model.module.encode_text(input_ids, attention_mask, meta=meta_tensor)
        
        # Calculate losses using refactored logic
        pool_loss, patch_loss = loss_calculator.calculate_losses(
            img_pool, img_patch_features, txt_pool, compressed_text_features, device, model.module.pool_scale, model.module.patch_scale
        )
        
        # Combined loss
        total_batch_loss = pool_loss * pool_weight + patch_loss * patch_weight
        
        # Backward pass
        total_batch_loss.backward()
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        # Gradient clipping
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        param_before = {name: param.clone() for name, param in model.named_parameters()}
        optimizer.step()
        # scheduler.step()

        # Update metrics
        metrics['total_loss'] += total_batch_loss.item()
        metrics['pool_loss'] += pool_loss.item()
        metrics['patch_loss'] += patch_loss.item()
        metrics['num_batches'] += 1
        
        # Logging
        if (step % args.log_interval == 0 or step == len(train_loader) - 1) and local_rank == 0:
            batch_time = time.time() - batch_start_time
            # current_lr = scheduler.get_last_lr()[0]
            current_lr = optimizer.param_groups[0]['lr']

            param_changes = {}
            for name, param in model.named_parameters():
                param_changes[name] = torch.norm(param - param_before[name]).item()
                
            max_change = max(param_changes.values())
            
            logger.info(
                f"Epoch {epoch}, Step {step}/{len(train_loader)}, "
                f"Total Loss: {total_batch_loss.item():.4f}, "
                f"Pool Loss: {pool_loss.item():.4f}, "
                f"Patch Loss: {patch_loss.item():.4f}, "
                f"LR: {current_lr:.6f}, "
                f"Batch Time: {batch_time:.3f}s, "
                f"Max Parameter Change: {max_change:.6f}, "
                f"Total Norm: {total_norm:.6f}"
            )
            
            # Wandb logging (only from rank 0)
            if local_rank == 0 and not args.disable_wandb:
                global_step = epoch * len(train_loader) + step
                wandb.log({
                    "train/total_loss": total_batch_loss.item(),
                    "train/pool_loss": pool_loss.item(),
                    "train/patch_loss": patch_loss.item(),
                    "train/learning_rate": current_lr,
                    "train/batch_time": batch_time,
                    "train/epoch": epoch,
                    "train/step": global_step,
                }, step=global_step)

    
    # Calculate epoch averages
    epoch_time = time.time() - start_time
    avg_loss = metrics['total_loss'] / metrics['num_batches']
    avg_pool_loss = metrics['pool_loss'] / metrics['num_batches']
    avg_patch_loss = metrics['patch_loss'] / metrics['num_batches']
    
    logger.info(
        f"Epoch {epoch} completed - "
        f"Avg Loss: {avg_loss:.4f}, "
        f"Avg Pool Loss: {avg_pool_loss:.4f}, "
        f"Avg Patch Loss: {avg_patch_loss:.4f}, "
        f"Time: {epoch_time:.2f}s"
    )
    
    # Log epoch-level metrics to wandb
    if local_rank == 0 and not args.disable_wandb:
        wandb.log({
            "train/epoch_avg_loss": avg_loss,
            "train/epoch_avg_pool_loss": avg_pool_loss,
            "train/epoch_avg_patch_loss": avg_patch_loss,
            "train/epoch_time": epoch_time,
        }, step=epoch * len(train_loader))
    
    return avg_loss, avg_pool_loss, avg_patch_loss


@torch.no_grad()
def validate(model: PatchWiseCLIP, val_loader, args, logger, device: torch.device, dtype: torch.dtype, local_rank: int = 0):
    """Validate model on ImageNet"""
    model.eval()
    
    logger.info("Starting validation...")
    
    # Prepare text features
    processor_kwargs = dict(
        padding="max_length", max_length=77, truncation=True, return_tensors="pt"
    )
    
    # Use all templates for better evaluation
    templates = OPENAI_IMAGENET_TEMPLATES
    texts = [template(c) for c in IMAGENET_CLASSNAMES for template in templates]
    
    # Process texts in batches
    batch_size = 1024
    all_txt_pools = []
    all_txt_latents = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing text templates"):
        batch_texts = texts[i:i+batch_size]
        proc_out = model.processor(text=batch_texts, **processor_kwargs).to(device)
        
        txt_pool, txt_latent = model.module.encode_text(
            input_ids=proc_out["input_ids"],
            attention_mask=proc_out["attention_mask"],
            meta=None,
        )
        
        all_txt_pools.append(txt_pool)
        all_txt_latents.append(txt_latent)
    
    # Concatenate and reshape
    txt_pools = torch.cat(all_txt_pools, dim=0)
    txt_latents = torch.cat(all_txt_latents, dim=0)
    
    # Reshape to (num_classes, num_templates, ...)
    num_classes = len(IMAGENET_CLASSNAMES)
    num_templates = len(templates)
    
    txt_pools = txt_pools.reshape(num_classes, num_templates, -1).mean(dim=1)
    txt_latents = txt_latents.reshape(num_classes, num_templates, txt_latents.size(-2), txt_latents.size(-1)).mean(dim=1)
    
    # Synchronize text features across processes
    if args.world_size > 1:
        dist.barrier()
        dist.all_reduce(txt_pools, op=dist.ReduceOp.AVG)
        dist.all_reduce(txt_latents, op=dist.ReduceOp.AVG)
    
    # Normalize features
    txt_pools = F.normalize(txt_pools, dim=-1)
    txt_latents = F.normalize(txt_latents, dim=-1)
    
    logger.info(f"Text pools shape: {txt_pools.shape}")
    logger.info(f"Text latents shape: {txt_latents.shape}")
    
    # Validation loop
    pool_correct = 0
    patch_correct = 0
    total_samples = 0
    
    for batch in tqdm(val_loader, desc="Validating"):
        pixel_values, labels, meta_tensor = batch
        pixel_values = pixel_values.to(device, dtype=dtype)
        labels = labels.to(device)
        
        img_pools, img_patches = model.module.encode_image(pixel_values)
        img_pools = F.normalize(img_pools, dim=-1)
        img_patches = F.normalize(img_patches, dim=-1)
        
        # Pool-based prediction
        pool_sim = torch.matmul(img_pools, txt_pools.T)
        pool_preds = pool_sim.argmax(dim=-1)
        
        # Patch-based prediction with voting
        B, N, D = img_patches.shape
        patch_preds = []
        
        for b in range(B):
            img_patch = img_patches[b]  # (N, D)
            
            # Calculate similarity between image patches and text patches
            patch_sim = torch.sum(img_patch.unsqueeze(0) * txt_latents, dim=-1)  # (num_classes, N)
            
            # Vote: each patch votes for the most similar class
            patch_votes = patch_sim.argmax(dim=0)  # (N,)
            
            # Final prediction: majority vote
            votes = torch.bincount(patch_votes, minlength=num_classes)
            patch_pred = votes.argmax()
            patch_preds.append(patch_pred)
        
        patch_preds = torch.stack(patch_preds)
        
        # Update counters
        pool_correct += (pool_preds == labels).sum().item()
        patch_correct += (patch_preds == labels).sum().item()
        total_samples += labels.size(0)
    
    # Gather results across processes
    if args.world_size > 1:
        pool_correct_tensor = torch.tensor(pool_correct, device=device)
        patch_correct_tensor = torch.tensor(patch_correct, device=device)
        total_samples_tensor = torch.tensor(total_samples, device=device)
        
        dist.all_reduce(pool_correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(patch_correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
        
        pool_correct = pool_correct_tensor.item()
        patch_correct = patch_correct_tensor.item()
        total_samples = total_samples_tensor.item()
    
    pool_accuracy = pool_correct / total_samples
    patch_accuracy = patch_correct / total_samples
    
    logger.info(f"Pool-based accuracy: {pool_accuracy:.4f}")
    logger.info(f"Patch-based voting accuracy: {patch_accuracy:.4f}")
    
    # Log validation metrics to wandb
    if local_rank == 0 and not args.disable_wandb:
        wandb.log({
            "val/pool_accuracy": pool_accuracy,
            "val/patch_accuracy": patch_accuracy,
            "val/best_accuracy": max(pool_accuracy, patch_accuracy),
        })
    
    return pool_accuracy, patch_accuracy


def main():
    parser = ArgumentParser()
    
    # Model arguments
    parser.add_argument("--clip_id", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--text_compressing_layers", type=int, default=0)
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    
    # Loss arguments
    parser.add_argument(
        "--loss_weights",
        type=str,
        default="0.6,0.4,0.0",
        help="Weights for loss components (pool_loss, patch_loss, ctf_loss)"
    )
    
    # Data arguments
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true", default=True)
    
    # Training configuration
    parser.add_argument("--training_dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--eval_interval", type=int, default=5, help="Evaluate every N epochs")
    parser.add_argument("--log_interval", type=int, default=100, help="Log every N steps")
    
    # Checkpoint arguments
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--resume", type=str, default="", help="Resume from checkpoint")
    parser.add_argument("--save_interval", type=int, default=10, help="Save checkpoint every N epochs")
    
    # Distributed training
    parser.add_argument("--seed", type=int, default=42)
    
    # Wandb logging
    parser.add_argument("--wandb_project", type=str, default="patch-wise-clip", help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default="", help="Wandb run name")
    parser.add_argument("--wandb_entity", type=str, default="", help="Wandb entity/team name")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable wandb logging")
    
    args = parser.parse_args()
    
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    args.rank = rank
    args.world_size = world_size
    args.local_rank = local_rank
    
    # Set device
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    
    # Set random seed
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    
    # Setup logging
    logger = setup_logging(local_rank, args.save_dir)
    
    # Initialize wandb (only on rank 0)
    if local_rank == 0 and not args.disable_wandb:
        wandb_config = {
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "min_lr": args.min_lr,
            "weight_decay": args.weight_decay,
            "warmup_steps": args.warmup_steps,
            "grad_clip": args.grad_clip,
            "loss_weights": args.loss_weights,
            "clip_id": args.clip_id,
            "text_compressing_layers": args.text_compressing_layers,
            "training_dtype": args.training_dtype,
            "world_size": world_size,
            "seed": args.seed,
        }
        
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name if args.wandb_run_name else None,
            entity=args.wandb_entity if args.wandb_entity else None,
            config=wandb_config,
            resume="allow" if args.resume else None
        )
    
    # Log arguments
    if local_rank == 0:
        logger.info(f"Training arguments: {json.dumps(vars(args), indent=2)}")
    
    # Set dtype
    if args.training_dtype == "bf16":
        dtype = torch.bfloat16
    elif args.training_dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    
    # Create transforms
    train_transform, val_transform = create_transforms()
    
    # Initialize model
    logger.info("Initializing model...")
    model = PatchWiseCLIP(
        clip_id=args.clip_id,
        text_compressing_layers=args.text_compressing_layers,
    )
    model.to(device, dtype=dtype)
    
    # Wrap model with DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # Initialize datasets
    logger.info("Initializing datasets...")
    train_ds = ClipDataset(
        pretrained_clip_id=args.clip_id,
        transform=train_transform,
    )
    val_ds = ImageNetDataset(
        pretrained_clip_id=args.clip_id,
        transform=val_transform,
    )
    
    # Create data loaders
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank) if world_size > 1 else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank) if world_size > 1 else None
    
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        shuffle=(train_sampler is None)
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        shuffle=False
    )
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98)
    )
    
    total_steps = args.epochs * len(train_loader)
    # scheduler = create_scheduler(optimizer, total_steps, args.warmup_steps, args.min_lr, args.lr, local_rank)
    scheduler = None
    # Load checkpoint if resuming
    start_epoch = 0
    best_accuracy = 0.0
    
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        start_epoch, best_accuracy = load_checkpoint(model, optimizer, scheduler, args.resume, device)
        logger.info(f"Resumed from epoch {start_epoch}, best accuracy: {best_accuracy:.4f}")
    
    # Create save directory
    if local_rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
    
    # Training loop
    logger.info("Starting training...")
    
    for epoch in range(start_epoch, args.epochs):
        # Set epoch for sampler
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        # Training
        train_loss, train_pool_loss, train_patch_loss = train_one_epoch(
            model, optimizer, scheduler, train_loader, epoch, args, logger, device, dtype, local_rank
        )
        
        # Validation
        pool_accuracy, patch_accuracy = 0.0, 0.0
        if (epoch + 1) % args.eval_interval == 0 or epoch == args.epochs - 1:
            pool_accuracy, patch_accuracy = validate(model, val_loader, args, logger, device, dtype, local_rank)
        
        # Save checkpoint
        current_accuracy = max(pool_accuracy, patch_accuracy)
        is_best = current_accuracy > best_accuracy
        
        if is_best:
            best_accuracy = current_accuracy
            logger.info(f"New best accuracy: {best_accuracy:.4f}")
            
            # Log new best accuracy to wandb
            if local_rank == 0 and not args.disable_wandb:
                wandb.log({
                    "val/new_best_accuracy": best_accuracy,
                    "val/best_accuracy_epoch": epoch,
                })
        
        if local_rank == 0:
            if (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1 or is_best:
                save_checkpoint(
                    model, optimizer, scheduler, epoch, best_accuracy, args,
                    args.save_dir, is_best
                )
    
    logger.info("Training completed!")
    
    # Finish wandb run
    if local_rank == 0 and not args.disable_wandb:
        wandb.finish()
    
    cleanup_distributed()


if __name__ == "__main__":
    main()