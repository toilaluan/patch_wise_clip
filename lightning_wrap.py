import torch
import lightning as L
from latent_clip.model import PatchWiseCLIP

POOL_W = 0.6
PATCH_W = 0.4


class WrapPatchWiseClip(L.LightningModule):
    def __init__(
        self,
        pretrained_clip_id: str = "openai/clip-vit-base-patch32",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        n_register_encoder_layers: int = 0,
        loss_weights: tuple = (POOL_W, PATCH_W),
    ):
        super().__init__()
        self.clip = PatchWiseCLIP(pretrained_clip_id, latent_layers=n_register_encoder_layers)
        self.lr = learning_rate
        self.wd = weight_decay
        self.loss_w = loss_weights
        self.save_hyperparameters("learning_rate", "weight_decay", "n_register_encoder_layers", "loss_weights")

    # ------------------------------ train ---------------------------------- #

    def forward(self, *args, **kwargs):
        return self.clip(*args, **kwargs)

    def training_step(self, batch, _):
        pool_loss, patch_loss = self(**batch, return_loss=True)
        total = pool_loss * self.loss_w[0] + patch_loss * self.loss_w[1]
        self.log_dict({
            "train/total": total,
            "train/pool": pool_loss,
            "train/patch": patch_loss,
        }, on_step=True, on_epoch=True, prog_bar=True)
        return total

    # ----------------------------- validation ------------------------------ #

    @torch.no_grad()
    def validation_step(self, batch, _):
        pixel_values = batch["pixel_values"]  # (B, 3, H, W)
        labels = batch["label"]
        meta_tensor = batch.get("meta_tensor", None)
        num_cls = batch["input_ids_all"].shape[1]

        img_pool, img_patch = self.clip.encode_image(pixel_values)
        all_global, all_patch = [], []  # logits for each image

        for i in range(pixel_values.size(0)):
            input_ids = batch["input_ids_all"][i]
            attention_mask = batch["attention_mask_all"][i]
            meta = meta_tensor[i] if meta_tensor is not None else None

            txt_pool, txt_latent = self.clip.encode_text(
                input_ids=input_ids,
                attention_mask=attention_mask,
                meta=meta.expand(num_cls, -1) if meta is not None else None,
            )

            # global similarity
            g_log = (img_pool[i].unsqueeze(0) @ txt_pool.t()) * self.clip.model.logit_scale.exp()  # (1, num_cls)
            all_global.append(g_log.squeeze(0))

            # patch similarity (if available)
            if txt_latent is not None:
                scale = self.clip.patch_logit_s.exp()
                p_sim = (img_patch[i].unsqueeze(0) * txt_latent).sum(-1) * scale  # (num_cls, N)
                all_patch.append(p_sim.sum(-1))  # (num_cls,)
            else:
                all_patch.append(torch.zeros(num_cls, device=self.device))

        global_logits = torch.stack(all_global)  # (B, num_cls)
        patch_logits = torch.stack(all_patch)    # (B, num_cls)
        combined_logits = global_logits + patch_logits

        preds_global = global_logits.argmax(-1)
        preds_comb = combined_logits.argmax(-1)
        preds_patch = patch_logits.argmax(-1)

        acc_global = (preds_global == labels).float().mean()
        acc_comb = (preds_comb == labels).float().mean()
        acc_patch = (preds_patch == labels).float().mean()

        self.log_dict({
            "val_imagenet_acc": acc_global,
            "val/acc_combined": acc_comb,
            "val/acc_patch": acc_patch,
        }, on_epoch=True, prog_bar=True)
        return {"acc_global": acc_global, "acc_combined": acc_comb}

    # ---------------------------- optimisers ------------------------------- #

    def configure_optimizers(self):
        opt = torch.optim.AdamW((p for p in self.parameters() if p.requires_grad), lr=self.lr, weight_decay=self.wd)
        return opt
