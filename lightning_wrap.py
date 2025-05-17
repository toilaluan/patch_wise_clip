import torch
import lightning as L
from latent_clip.model import LatentClip
from transformers.optimization import get_cosine_schedule_with_warmup


class WrapPatchWiseClip(L.LightningModule):
    def __init__(
        self,
        pretrained_clip_id: str = "openai/clip-vit-base-patch32",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        total_steps: int = 100000,
        warmup_steps: int = 1000,
    ):
        super().__init__()
        self.latent_clip = LatentClip(pretrained_clip_id, n_register_layers=12)
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        # for p in self.latent_clip.model.parameters():
        #     p.requires_grad = False
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(self, *args, **kwargs):
        return self.latent_clip(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        patch_loss = self.forward(**batch)
        self.log(
            "train/patch_loss", patch_loss, on_step=True, on_epoch=True, prog_bar=True
        )
        return patch_loss

    def validation_step(self, batch, batch_idx):
        patch_loss = self.forward(**batch)
        self.log("val/patch_loss", patch_loss, on_step=True, on_epoch=True)
        return patch_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        optimizer = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps,
        )
        return optimizer
