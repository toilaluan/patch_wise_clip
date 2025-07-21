#!/usr/bin/env python3
"""
Distributed training script for Patch-Wise CLIP.
This version:
- Uses AMP (fp16/bf16), gradient accumulation, and warmup+cosine LR
- Removes all known deadlocks and checkpoint pitfalls
- Works with torchrun / torch.distributed.launch
"""

import json
import logging
import math
import os
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from transformers import AutoProcessor
from tqdm import tqdm

import wandb
from patch_wise_clip.data import ClipDataset, ImageNetDataset
from patch_wise_clip.loss import LossCalculator
from patch_wise_clip.model import PatchWiseCLIP
from patch_wise_clip.zeroshot_metadata import (
    IMAGENET_CLASSNAMES,
    SIMPLE_IMAGENET_TEMPLATES,
)


def get_model(model):
    if isinstance(model, torch.nn.DataParallel) or isinstance(
        model, torch.nn.parallel.DistributedDataParallel
    ):
        return model.module
    else:
        return model


# --------------------------------------------------------------------------- #
# 1.  Logging & distributed helpers
# --------------------------------------------------------------------------- #
def setup_logging(rank: int, log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s | %(levelname)s | R%(rank)d | %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(),
            (
                logging.FileHandler(log_dir / "train.log")
                if rank == 0
                else logging.NullHandler()
            ),
        ],
    )
    logger = logging.getLogger()
    logger = logging.LoggerAdapter(logger, {"rank": rank})
    return logger


def setup_distributed() -> tuple[int, int, int]:
    if "RANK" not in os.environ:  # single-GPU
        return 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


# --------------------------------------------------------------------------- #
# 2. Transforms & scheduler
# --------------------------------------------------------------------------- #
def get_transforms(clip_id: str):
    processor = AutoProcessor.from_pretrained(clip_id)
    return processor


def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps: int, num_training_steps: int, min_lr_ratio: float = 0.0
):
    def lr_lambda(step: int):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = (step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps
        )
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (
            1 + math.cos(math.pi * progress)
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# --------------------------------------------------------------------------- #
# 3. Checkpointing utilities
# --------------------------------------------------------------------------- #
def save_checkpoint(
    model, optimizer, scheduler, epoch: int, best_acc: float, args, is_best: bool
):
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank != 0:
        return
    state = {
        "epoch": epoch,
        "model": (
            get_model(model).state_dict()
            if hasattr(model, "module")
            else model.state_dict()
        ),
        "optimizer": optimizer.state_dict(),
        "best_acc": best_acc,
        "args": vars(args),
    }
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    ckpt_path = Path(args.save_dir) / f"ckpt_ep{epoch:03d}.pth"
    torch.save(state, ckpt_path)
    if is_best:
        torch.save(state, Path(args.save_dir) / "best.pth")


def load_checkpoint(ckpt_path: str, model, optimizer, scheduler, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"]
    # strip 'module.' prefix if present
    if all(k.startswith("module.") for k in state):
        state = {k[7:]: v for k, v in state.items()}
    model.load_state_dict(state)
    optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["epoch"], ckpt["best_acc"]


# --------------------------------------------------------------------------- #
# 4. Training & validation loops
# --------------------------------------------------------------------------- #
def train_one_epoch(
    model,
    optimizer,
    scheduler,
    scaler,
    train_loader,
    epoch,
    args,
    logger,
    device,
    dtype,
    accum_steps,
):
    model.train()
    loss_calc = LossCalculator()
    pool_w, patch_w, patch_diversity_w = map(float, args.loss_weights.split(",")[:3])
    logger.info(f"Pool weight: {pool_w}, Patch weight: {patch_w}, Patch diversity weight: {patch_diversity_w}")

    metrics = {"loss": 0.0, "pl": 0.0, "pt": 0.0, "batches": 0, "acc": 0.0}

    for step, batch in enumerate(
        tqdm(train_loader, desc=f"E{epoch}", disable=args.rank != 0)
    ):
        pixel, ids, mask, meta = (t.to(device, non_blocking=True) for t in batch)
        ids, mask = ids.long(), mask.bool()

        with autocast(dtype=dtype, device_type="cuda"):
            # txt_pool, compressed_text_features, img_pool, img_patch = model(
            #     input_ids=ids,
            #     pixel_values=pixel,
            #     attention_mask=mask,
            #     meta_tensor=meta,
            # )
            img_pool, img_patch = get_model(model).encode_image(pixel)
            txt_pool, compressed_text_features = get_model(model).encode_text(
                ids, mask, meta=meta
            )
            pl, pt, acc, img_patch_diversity, img_patch_similarity = loss_calc(
                img_pool,
                img_patch,
                txt_pool,
                compressed_text_features,
                get_model(model).pool_scale,
                get_model(model).patch_scale,
                get_model(model).patch_diversity_scale,
            )
            loss = pool_w * pl + patch_w * pt + patch_diversity_w * img_patch_diversity

        scaler.scale(loss / accum_steps).backward()

        if (step + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        get_model(model).patch_scale.data.clamp_(0, 4.6052)
        get_model(model).pool_scale.data.clamp_(0, 4.6052)

        metrics["loss"] += loss.item()
        metrics["pl"] += pl.item()
        metrics["pt"] += pt.item()
        metrics["acc"] += acc.item()
        metrics["batches"] += 1

        if (
            step % args.log_interval == 0 or step == len(train_loader) - 1
        ) and args.rank == 0:

            lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"E{epoch}S{step:05d}| loss={loss.item():.4f} "
                f"pl={pl.item():.4f} pt={pt.item():.4f} lr={lr:.2e} img_patch_diversity={img_patch_diversity.item():.4f}"
            )
            if not args.disable_wandb:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/pool_loss": pl.item(),
                        "train/patch_loss": pt.item(),
                        "train/lr": lr,
                        "train/acc": acc.item(),
                        "train/pool_scale": get_model(model).pool_scale.item(),
                        "train/patch_scale": get_model(model).patch_scale.item(),
                        "train/img_patch_diversity": img_patch_diversity.item(),
                    },
                    step=epoch * len(train_loader) + step,
                )
    avg_loss = metrics["loss"] / metrics["batches"]
    avg_acc = metrics["acc"] / metrics["batches"]
    logger.info(f"Epoch {epoch} done | avg loss={avg_loss:.4f} avg acc={avg_acc:.4f}")
    return avg_loss, avg_acc


@torch.no_grad()
def validate(model, val_loader, args, logger, device, dtype, epoch):
    model.eval()
    logger.info("Running ImageNet validation…")

    # Build text features once (all templates)
    templates = SIMPLE_IMAGENET_TEMPLATES
    texts = [t(c) for c in IMAGENET_CLASSNAMES for t in templates]
    batch_size = 1024 * 4
    txt_pools, txt_latents = [], []

    for i in tqdm(
        range(0, len(texts), batch_size), desc="Text", disable=args.rank != 0
    ):
        batch = texts[i : i + batch_size]
        proc = (
            get_model(model)
            .processor(
                text=batch,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            .to(device)
        )
        pool, latent = get_model(model).encode_text(
            proc["input_ids"], proc["attention_mask"]
        )
        txt_pools.append(pool)
        if latent is not None:
            txt_latents.append(latent)

    num_cls, num_tmpl = len(IMAGENET_CLASSNAMES), len(templates)
    txt_pools = torch.cat(txt_pools)
    txt_pools = txt_pools.view(num_cls, num_tmpl, -1).mean(1)
    txt_pools = F.normalize(txt_pools, dim=-1, p=2)

    if txt_latents:
        txt_latents = torch.cat(txt_latents)
        txt_latents = txt_latents.view(
            num_cls, num_tmpl, txt_latents.size(-2), txt_latents.size(-1)
        ).mean(1)
        txt_latents = F.normalize(txt_latents, dim=-1, p=2)

    pool_corr = patch_corr = total = 0
    for pixel, label, _ in tqdm(val_loader, desc="Val", disable=args.rank != 0):
        pixel, label = pixel.to(device, dtype=dtype), label.to(device)
        img_pool, img_patch = get_model(model).encode_image(pixel)
        img_pool = F.normalize(img_pool, dim=-1, p=2)
        img_patch = F.normalize(img_patch, dim=-1, p=2)

        # pool prediction
        pool_pred = torch.argmax(img_pool @ txt_pools.T, dim=-1)
        pool_corr += (pool_pred == label).sum().item()

        # patch voting
        if isinstance(txt_latents, torch.Tensor):
            B, N, D = img_patch.shape
            patch_pred = []
            for b in range(B):
                sim = torch.sum(img_patch[b].unsqueeze(0) * txt_latents, dim=-1)  # (C, N)
                votes = torch.bincount(torch.argmax(sim, dim=0), minlength=num_cls)
                patch_pred.append(votes.argmax())
            patch_pred = torch.tensor(patch_pred, device=label.device)
            patch_corr += (patch_pred == label).sum().item()
        else:
            patch_corr += 0
        total += label.size(0)

    logger.info(
        f"[Rank {args.rank}] Pool acc={pool_corr / total:.4f} Patch acc={patch_corr / total:.4f}"
    )

    if args.world_size > 1:
        tensors = torch.tensor([pool_corr, patch_corr, total], device=device)
        dist.all_reduce(tensors, op=dist.ReduceOp.SUM)
        pool_corr, patch_corr, total = tensors.tolist()

    pool_acc = pool_corr / total
    patch_acc = patch_corr / total
    logger.info(f"Pool acc={pool_acc:.4f}  Patch acc={patch_acc:.4f}")
    if not args.disable_wandb and args.rank == 0:
        wandb.log(
            {
                "val/pool_acc": pool_acc,
                "val/pool_corr": pool_corr,
                "val/patch_corr": patch_corr,
                "val/total": total,
                "val/patch_acc": patch_acc,
            }
        )
    return pool_acc, patch_acc


# --------------------------------------------------------------------------- #
# 5. Main
# --------------------------------------------------------------------------- #
def main():
    parser = ArgumentParser()
    # fmt: off
    parser.add_argument("--clip_id", default="openai/clip-vit-base-patch32")
    parser.add_argument("--text_compressing_layers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--loss_weights", default="0.6,0.4", help="pool,patch")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true", default=True)
    parser.add_argument("--training_dtype", choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_dir", default="./checkpoints")
    parser.add_argument("--resume", default="")
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", default="patch-wise-clip")
    parser.add_argument("--wandb_run_name", default="")
    parser.add_argument("--wandb_entity", default="")
    parser.add_argument("--disable_wandb", action="store_true")
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    # fmt: on
    args = parser.parse_args()

    rank, world_size, local_rank = setup_distributed()
    args.rank, args.world_size, args.local_rank = rank, world_size, local_rank

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed + rank)

    device = torch.device("cuda", local_rank)

    # Logging + wandb
    logger = setup_logging(rank, Path(args.save_dir))
    if rank == 0 and not args.disable_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=args.wandb_run_name or None,
            config=vars(args),
            resume="allow" if args.resume else None,
        )

    # dtype
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map[args.training_dtype]

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_ds = ClipDataset(pretrained_clip_id=args.clip_id, transform=train_transform)
    val_ds = ImageNetDataset(pretrained_clip_id=args.clip_id, transform=val_transform)

    train_sampler = DistributedSampler(train_ds) if world_size > 1 else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if world_size > 1 else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=val_sampler,
        persistent_workers=True,
        prefetch_factor=4,
    )

    # model
    model = PatchWiseCLIP(
        clip_id=args.clip_id, text_compressing_layers=args.text_compressing_layers
    )
    model.to(device)
    model = torch.compile(model)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], gradient_as_bucket_view=True, static_graph=True)

    # optimizer & scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
        eps=1e-8,
    )

    total_steps = args.epochs * len(train_loader) // args.grad_accum_steps
    warmup_steps = args.warmup_steps
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # resume
    start_epoch, best_acc = 0, 0.0
    if args.resume:
        start_epoch, best_acc = load_checkpoint(
            args.resume, model, optimizer, scheduler, device
        )
        logger.info(f"Resumed from epoch {start_epoch}, best_acc={best_acc:.4f}")

    # scaler for AMP
    scaler = GradScaler(enabled=(dtype == torch.float16))

    # training loop
    logger.info(
        f"Start training for {args.epochs} epochs, {total_steps} steps, world_size={world_size}"
    )
    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss = train_one_epoch(
            model,
            optimizer,
            scheduler,
            scaler,
            train_loader,
            epoch,
            args,
            logger,
            device,
            dtype,
            args.grad_accum_steps,
        )

        if (epoch + 1) % args.eval_interval == 0 or epoch == args.epochs - 1:
            pool_acc, patch_acc = validate(
                model, val_loader, args, logger, device, dtype, epoch
            )
            curr_acc = max(pool_acc, patch_acc)
            is_best = curr_acc > best_acc
            if is_best:
                best_acc = curr_acc
                logger.info(f"New best acc={best_acc:.4f}")
        else:
            is_best = False

        if (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1 or is_best:
            save_checkpoint(model, optimizer, scheduler, epoch, best_acc, args, is_best)

    logger.info("Training finished!")
    if rank == 0 and not args.disable_wandb:
        wandb.finish()
    cleanup_distributed()


if __name__ == "__main__":
    main()
