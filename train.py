from lightning_wrap import WrapPatchWiseClip
from data import ClipDataset, ImageNetDataset
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch import seed_everything

seed_everything(42)

wandb_logger = WandbLogger(project="Patch-Wise-CLIP")


MAX_EPOCHS = 32
BATCH_SIZE = 256
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0
PRETRAINED_CLIP_ID = "openai/clip-vit-base-patch32"


train_dataset = ClipDataset(PRETRAINED_CLIP_ID, is_train=True)
val_dataset = ImageNetDataset(processor=train_dataset.processor)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

train_dl = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=24,
    pin_memory=True,
    drop_last=True,
)
val_dl = DataLoader(val_dataset, batch_size=BATCH_SIZE, drop_last=True, num_workers=24)

total_steps = len(train_dl) * MAX_EPOCHS
model = WrapPatchWiseClip(
    pretrained_clip_id=PRETRAINED_CLIP_ID,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    total_steps=total_steps,
    warmup_steps=500,
    n_latent_layers=8
)

model.train()
model.latent_clip.train()


trainer = L.Trainer(
    max_epochs=MAX_EPOCHS,
    logger=wandb_logger,
    log_every_n_steps=5,
    callbacks=[LearningRateMonitor(logging_interval="step")],
    accelerator="gpu",
    devices=1,
    precision="bf16-mixed",
    accumulate_grad_batches=1,
    # gradient_clip_algorithm="norm",
    # gradient_clip_val=1.0,
)

trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
