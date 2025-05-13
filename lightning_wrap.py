import torch
import lightning as L
from latent_clip.model import LatentClip


class WrapPatchWiseClip(L.LightningModule):
    def __init__(
        self,
        pretrained_clip_id: str = "openai/clip-vit-base-patch32",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
    ):
        super().__init__()
        self.latent_clip = LatentClip(
            pretrained_clip_id, pool_loss_weight=0.5, position_loss_weight=0.5
        )
        for p in self.latent_clip.model.parameters():
            p.requires_grad = False
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(self, *args, **kwargs):
        return self.latent_clip(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        total_loss, pool_loss, position_loss = self.forward(batch)
        self.log("train/total_loss", total_loss)
        self.log("train/pool_loss", pool_loss)
        self.log("train/pos_loss", position_loss)

    def validating_step(self, batch, batch_idx):
        total_loss, pool_loss, position_loss = self.forward(batch)
        self.log("val/total_loss", total_loss)
        self.log("val/pool_loss", pool_loss)
        self.log("val/pos_loss", position_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [p for p in self.parameters() if p.require_grad],
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return optimizer
