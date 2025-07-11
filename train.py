from lightning_wrap import WrapPatchWiseClip
from data import ClipDataset, ImageNetDataset
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch import seed_everything
from argparse import ArgumentParser
from lightning.pytorch.strategies import DDPStrategy
from torchvision import transforms

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
    "--n-text-compressing-layers",
    type=int,
    default=8,
    help="Number of layers to register for encoder",
)

parser.add_argument(
    "--loss-weights",
    type=str,
    default="0.6,0.4,0.0",
    help="Weights for the loss components (pool_loss, patch_loss, ctf_loss). 1.0,0.0,0.0 means original CLIP",
)


args = parser.parse_args()

seed_everything(42)

wandb_logger = WandbLogger(project="Patch-Wise-CLIP")


normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    transforms.ToTensor(),
    normalize
])
val_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

train_dataset = ClipDataset(args.pretrained_clip_id, is_train=True, transform=train_transform)
val_dataset = ImageNetDataset(processor=train_dataset.processor, transform=val_transform)

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
    n_text_compressing_layers=args.n_text_compressing_layers,
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
            dirpath=f"checkpoints/{args.pretrained_clip_id.split('/')[-1]}/{args.n_text_compressing_layers}-{args.loss_weights}",
            filename="best-{epoch:02d}-{val_acc_global:.2f}",
            save_last=True,
            save_top_k=3,
            monitor="val_acc_global",
            mode="max",
            every_n_epochs=1,
        ),
    ],
    accelerator="gpu",
    precision="bf16-mixed",
    accumulate_grad_batches=1,
    devices=8, strategy=DDPStrategy(find_unused_parameters=True)
)

trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
