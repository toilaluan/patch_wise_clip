import torch
import lightning as L
from latent_clip.model import LatentCLIP
from transformers.optimization import get_cosine_schedule_with_warmup


POOL_W = 0.8
PATCH_W = 0.2

class WrapPatchWiseClip(L.LightningModule):
    def __init__(
        self,
        pretrained_clip_id: str = "openai/clip-vit-base-patch32",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        total_steps: int = 100000,
        warmup_steps: int = 1000,
        n_latent_layers: int = 0,
    ):
        super().__init__()
        self.latent_clip = LatentCLIP(pretrained_clip_id, latent_layers=n_latent_layers)
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(self, *args, **kwargs):
        return self.latent_clip(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        pool_loss, patch_loss = self.forward(**batch, return_loss=True)
        combined_loss = pool_loss * POOL_W + patch_loss * PATCH_W
        self.log(
            "train/total_loss", combined_loss, on_step=True, on_epoch=True, prog_bar=True
        )
        self.log("train/pool_loss", pool_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/patch_loss", patch_loss, on_step=True, on_epoch=True, prog_bar=True)
        return combined_loss

    def validation_step(self, batch, batch_idx):
        # For ImageNetDataset: batch is a dict, each key: batch_size
        # batch["pixel_values"]: (B, 3, H, W)
        # batch["input_ids_all"]: (B, num_classes, 77)
        # batch["attention_mask_all"]: (B, num_classes, 77)
        # batch["label"]: (B,)

        pixel_values = batch["pixel_values"]        # (B, 3, H, W)
        meta_tensor = batch.get("meta_tensor", None)
        labels = batch["label"]                     # (B,)

        # 1. Compute image pooled features
        img_pool, _ = self.latent_clip.encode_image(pixel_values)

        # 2. For each image in batch, get all candidate text features (num_classes, 77)
        num_classes = batch["input_ids_all"].shape[1]
        all_logits = []
        for i in range(pixel_values.size(0)):
            input_ids = batch["input_ids_all"][i]             # (num_classes, 77)
            attention_mask = batch["attention_mask_all"][i]   # (num_classes, 77)
            meta = meta_tensor[i] if meta_tensor is not None else None

            # Text features for all classes
            txt_pool, _ = self.latent_clip.encode_text(
                input_ids=input_ids,
                attention_mask=attention_mask,
                meta=meta.expand(num_classes, -1) if meta is not None else None
            )  # (num_classes, D)``

            # Compute similarity to current image
            logits = img_pool[i].unsqueeze(0) @ txt_pool.t() * self.latent_clip.model.logit_scale.exp()  # (1, num_classes)
            all_logits.append(logits.squeeze(0))  # (num_classes,)

        logits = torch.stack(all_logits, dim=0)  # (B, num_classes)
        preds = logits.argmax(dim=-1)            # (B,)
        acc = (preds == labels).float().mean()

        self.log("val/imagenet_acc", acc, on_epoch=True, prog_bar=True)

        # Optionally, return accuracy and loss for aggregation
        return {"acc": acc}


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        # scheduler = get_cosine_schedule_with_warmup(
        #     optimizer,
        #     num_warmup_steps=self.warmup_steps,
        #     num_training_steps=self.total_steps,
        # )
        return optimizer
