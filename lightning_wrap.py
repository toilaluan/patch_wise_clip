import torch, lightning as L
from patch_wise_clip.model import PatchWiseCLIP
from transformers import CLIPProcessor

POOL_W = 0.6
PATCH_W = 0.4
CTF_W = 0.0


class WrapPatchWiseClip(L.LightningModule):
    def __init__(
        self,
        pretrained_clip_id: str = "openai/clip-vit-base-patch32",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        n_text_compressing_layers: int = 0,
        loss_weights: tuple = (POOL_W, PATCH_W, CTF_W),
        hf_output_id: str = "toilaluan/patch-wise-clip",
    ):
        super().__init__()
        self.hf_output_id = hf_output_id + f"-{n_text_compressing_layers}clayers-{loss_weights[0]}pool-{loss_weights[1]}patch-{loss_weights[2]}ctf"
        self.clip = PatchWiseCLIP(
            pretrained_clip_id, text_compressing_layers=n_text_compressing_layers
        )
        self.processor = CLIPProcessor.from_pretrained(pretrained_clip_id)
        self.lr, self.wd, self.loss_w = learning_rate, weight_decay, loss_weights

        # Will be initialised in `setup()` when `stage == "validate"`
        self.register_buffer("txt_pool", torch.empty(0), persistent=False)
        self.register_buffer("txt_latent", torch.empty(0), persistent=False)

        self.save_hyperparameters(
            "learning_rate", "weight_decay", "n_text_compressing_layers", "loss_weights"
        )
        self.best_val_acc = 0.0

    # ---------- text cache --------------------------------------------------
    def setup(self, stage: str) -> None:
        print(f"Setting up {stage}")
        if stage != "validate":
            return

        # Build captions once per process
        from data import IMAGENET_CAPTIONS  # see dataset section below
        processor_kwargs = dict(
            padding="max_length", max_length=77, truncation=True, return_tensors="pt"
        )
        labels = [str(i) for i in range(len(IMAGENET_CAPTIONS))]
        captions = [f"An image of a {IMAGENET_CAPTIONS[i]}" for i in labels]
        print(f"captions: {captions[:10]}")
        proc_out = self.processor(
            text=captions, **processor_kwargs
        ).to(self.device)
        # print(f"proc_out: {proc_out}")

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
        print(f"Will be pushing to {self.hf_output_id}")
        print(f"txt_pool shape: {self.txt_pool.shape}")
        print(f"txt_latent shape: {self.txt_latent.shape}")

    def forward(self, *a, **kw):
        return self.clip(*a, **kw)

    def training_step(self, batch, _):
        pool_loss, patch_loss, ctf_loss = self(**batch, return_loss=True)
        total = pool_loss * self.loss_w[0] + patch_loss * self.loss_w[1] + ctf_loss * self.loss_w[2]
        self.log_dict(
            {
                "train/total": total,
                "train/pool": pool_loss,
                "train/patch": patch_loss,
                "train/ctf": ctf_loss,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return total

    # ---------- fast validation --------------------------------------------
    def on_validation_epoch_start(self):
        self.clip.eval()
        self.validation_step_outputs = []
        # if self.txt_pool.shape[0] == 0:
        self.setup("validate")

    @torch.no_grad()
    def validation_step(self, batch, _):
        img_pool, img_patch = self.clip.encode_image(batch["pixel_values"])

        # global logits
        scale = self.clip.model.logit_scale.exp()
        global_logits = img_pool @ self.txt_pool.T * scale

        # patch logits
        if self.txt_latent is not None:
            # patch_scale = self.clip.patch_logit_s.exp() # Dont need scale for argmax
            patch_logits = (
                torch.einsum("bnd,knd->bk", img_patch, self.txt_latent)
            )
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
            sync_dist=True,
        )
        acc = (preds_c == labels).float().mean()
        self.validation_step_outputs.append(acc)
    
    def on_validation_epoch_end(self):
        acc = torch.stack(self.validation_step_outputs).mean()
        if acc > self.best_val_acc:
            self.best_val_acc = acc
            if self.trainer.is_global_zero:
                self.clip.model.push_to_hub(self.hf_output_id)

    # ---------- optim ------------------------------------------------------
    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            (p for p in self.parameters() if p.requires_grad),
            lr=self.lr,
            weight_decay=self.wd,
            betas=(0.9, 0.98),
        )
        return opt
