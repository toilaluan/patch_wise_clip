# ----------------------------------------------
# model_wrap.py
# ----------------------------------------------
import torch, lightning as L
from latent_clip.model import PatchWiseCLIP
from transformers import CLIPProcessor

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
        self.clip = PatchWiseCLIP(
            pretrained_clip_id, latent_layers=n_register_encoder_layers
        )
        self.processor = CLIPProcessor.from_pretrained(pretrained_clip_id)
        self.lr, self.wd, self.loss_w = learning_rate, weight_decay, loss_weights

        # Will be initialised in `setup()` when `stage == "validate"`
        self.register_buffer("txt_pool", torch.empty(0), persistent=False)
        self.register_buffer("txt_latent", torch.empty(0), persistent=False)

        self.save_hyperparameters(
            "learning_rate", "weight_decay", "n_register_encoder_layers", "loss_weights"
        )

    # ---------- text cache --------------------------------------------------
    def setup(self, stage: str) -> None:
        if stage != "validate":
            return

        # Build captions once per process
        from data import IMAGENET_CAPTIONS  # see dataset section below
        processor_kwargs = dict(
            padding="max_length", max_length=77, truncation=True, return_tensors="pt"
        )
        proc_out = self.processor(
            text=[f"An image of a {c}" for c in IMAGENET_CAPTIONS], **processor_kwargs
        ).to(self.device)

        with torch.no_grad():
            self.clip.eval()
            txt_pool, txt_latent = self.clip.encode_text(
                input_ids=proc_out["input_ids"],
                attention_mask=proc_out["attention_mask"],
                meta=None,
            )

        # Normalise once so later we can just use matmul
        self.txt_pool = torch.nn.functional.normalize(txt_pool, dim=-1)
        self.txt_latent = (
            torch.nn.functional.normalize(txt_latent, dim=-1) if txt_latent is not None else None
        )

        print(self.txt_pool.shape, self.txt_latent.shape)

    # ---------- train (unchanged) ------------------------------------------
    def forward(self, *a, **kw):
        return self.clip(*a, **kw)

    def training_step(self, batch, _):
        pool_loss, patch_loss = self(**batch, return_loss=True)
        total = pool_loss * self.loss_w[0] + patch_loss * self.loss_w[1]
        self.log_dict(
            {
                "train/total": total,
                "train/pool": pool_loss,
                "train/patch": patch_loss,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return total

    # ---------- fast validation --------------------------------------------
    def on_validation_epoch_start(self):
        # Set model to eval mode
        self.clip.eval()

        # Set up text cache
        if self.txt_pool.shape[0] == 0:
            self.setup("validate")

    @torch.no_grad()
    def validation_step(self, batch, _):
        img_pool, img_patch = self.clip.encode_image(batch["pixel_values"])

        # global logits
        scale = self.clip.model.logit_scale.exp()
        global_logits = img_pool @ self.txt_pool.T * scale          # (B, 1000)

        # patch logits
        if self.txt_latent is not None:
            patch_scale = self.clip.patch_logit_s.exp()
            # img_patch: (B, N, D)  txt_latent: (1000, N, D)
            # einsum -> (B, 1000, N)  →  sum over N  →  (B, 1000)
            # print(img_patch.shape, self.txt_latent.shape)
            # print(img_patch, img_patch.shape)
            # print(self.txt_latent, self.txt_latent.shape)
            patch_logits = (
                torch.einsum("bnd,knd->bk", img_patch, self.txt_latent)
            )
            # print(patch_logits)
        else:
            patch_logits = torch.zeros_like(global_logits)

        combined_logits = global_logits + patch_logits
        labels = batch["label"]

        preds_g = global_logits.argmax(-1)
        preds_c = combined_logits.argmax(-1)
        preds_p = patch_logits.argmax(-1)

        self.log_dict(
            {
                "val_acc_global": (preds_g == labels).float().mean(),
                "val/acc_comb": (preds_c == labels).float().mean(),
                "val/acc_patch": (preds_p == labels).float().mean(),
            },
            on_epoch=True,
            prog_bar=True,
        )

    # ---------- optim ------------------------------------------------------
    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            (p for p in self.parameters() if p.requires_grad),
            lr=self.lr,
            weight_decay=self.wd,
        )
        return opt
