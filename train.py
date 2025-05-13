from lightning_wrap import WrapPatchWiseClip
from data import ClipDataset
from torch.utils.data import DataLoader
import lightning as L

pretrained_clip_id = "openai/clip-vit-base-patch32"

model = WrapPatchWiseClip(pretrained_clip_id=pretrained_clip_id, learning_rate=1e-4)

train_dataset = ClipDataset(pretrained_clip_id, is_train=True)
val_dataset = ClipDataset(pretrained_clip_id, is_train=False)

train_dl = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_dl = DataLoader(val_dataset, batch_size=1)


trainer = L.Trainer(max_epochs=1, limit_val_batches=50)

trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
