from lightning_wrap import WrapPatchWiseClip
from data import ClipDataset, ImageNetDataset
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch import seed_everything
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument(
    "--pretrained-clip-id",
    type=str,
    default="openai/clip-vit-base-patch32",
    help="Pretrained CLIP model ID",
)

parser.add_argument(
    "--learning-rate",
    type=float,
    default=1e-4,
    help="Learning rate for the optimizer",
)

parser.add_argument(
    "--weight-decay",
    type=float,
    default=0,
    help="Weight decay for the optimizer",
)

parser.add_argument(
    "--max-epochs",
    type=int,
    default=32,
    help="Maximum number of epochs to train",
)

parser.add_argument(
    "--batch-size",
    type=int,
    default=64,
    help="Batch size for training",
)

parser.add_argument(
    "--num-workers",
    type=int,
    default=24,
    help="Number of workers for data loading",
)

parser.add_argument(
    "--n-register-encoder-layers",
    type=int,
    default=8,
    help="Number of layers to register for encoder",
)

parser.add_argument(
    "--loss-weights",
    type=str,
    default="0.6,0.4",
    help="Weights for the loss components (pool_loss, patch_loss). 1.0,0.0 means original CLIP",
)

args = parser.parse_args()

seed_everything(42)

wandb_logger = WandbLogger(project="Patch-Wise-CLIP")


train_dataset = ClipDataset(args.pretrained_clip_id, is_train=True)
val_dataset = ImageNetDataset(processor=train_dataset.processor)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

train_dl = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=24,
    pin_memory=True,
    drop_last=True,
)
val_dl = DataLoader(
    val_dataset, batch_size=args.batch_size, drop_last=True, num_workers=24
)

model = WrapPatchWiseClip(
    pretrained_clip_id=args.pretrained_clip_id,
    learning_rate=args.learning_rate,
    weight_decay=args.weight_decay,
    n_register_encoder_layers=args.n_register_encoder_layers,
    loss_weights=[float(w) for w in args.loss_weights.split(",")],
)

model.train()
model.clip.train()


trainer = L.Trainer(
    max_epochs=args.max_epochs,
    logger=wandb_logger,
    log_every_n_steps=5,
    callbacks=[
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            dirpath=f"checkpoints/{args.pretrained_clip_id.split('/')[-1]}/{args.n_register_encoder_layers}-{args.loss_weights}",
            filename="best-{epoch:02d}-{val_acc_global:.2f}",
            save_last=True,
            save_top_k=3,
            monitor="val_acc_global",
            mode="max",
            every_n_epochs=1,
        ),
    ],
    accelerator="gpu",
    devices=1,
    precision="bf16-mixed",
    accumulate_grad_batches=1,
)

trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
