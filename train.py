from lightning_wrap import WrapPatchWiseClip
from data import ClipDataset
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import WandbLogger


wandb_logger = WandbLogger(project="Patch-Wise-CLIP")


MAX_EPOCHS = 32
BATCH_SIZE = 256

pretrained_clip_id = "openai/clip-vit-base-patch32"


train_dataset = ClipDataset(pretrained_clip_id, is_train=True)
val_dataset = ClipDataset(pretrained_clip_id, is_train=False)

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
    pretrained_clip_id=pretrained_clip_id,
    learning_rate=1e-3,
    weight_decay=1e-4,
    total_steps=total_steps,
    warmup_steps=500,
)

trainer = L.Trainer(
    max_epochs=32,
    logger=wandb_logger,
    log_every_n_steps=5,
    gradient_clip_algorithm="norm",
    gradient_clip_val=1.0,
)

trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
