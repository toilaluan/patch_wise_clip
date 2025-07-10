import torch, lightning as L
from patch_wise_clip.model import PatchWiseCLIP
from patch_wise_clip.zeroshot_metadata import IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES
from transformers import CLIPProcessor
from tqdm import tqdm
import torch.distributed as dist

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
        print(f"Train steps: {self.trainer.estimated_stepping_batches}")
        if stage != "validate":
            return

        processor_kwargs = dict(
            padding="max_length", max_length=77, truncation=True, return_tensors="pt"
        )
        N_TEMPLATES = len(OPENAI_IMAGENET_TEMPLATES)
        N_CLASSES = len(IMAGENET_CLASSNAMES)
        local_rank = dist.get_rank()
        world_size = dist.get_world_size()
        templates = []
        for i, template in enumerate(OPENAI_IMAGENET_TEMPLATES[:N_TEMPLATES]):
            if i % world_size == local_rank:
                templates.append(template)
                print(f"Rank {local_rank} is processing template {i} of {N_TEMPLATES}")
        texts = [template(c) for c in IMAGENET_CLASSNAMES for template in templates]
        self.clip.eval()
        total_txt_pool = []
        total_txt_latent = []
        batch_texts = [texts[i:i+N_CLASSES] for i in range(0, len(texts), N_CLASSES)]

        for batch_texts in tqdm(batch_texts, desc="Processing texts"):
            proc_out = self.processor(
                text=batch_texts, **processor_kwargs
            ).to(self.device)
            with torch.no_grad():
                txt_pool, txt_latent = self.clip.encode_text(
                    input_ids=proc_out["input_ids"],
                    attention_mask=proc_out["attention_mask"],
                    meta=None,
                )
                total_txt_pool.append(txt_pool)
                total_txt_latent.append(txt_latent)
        txt_pool = torch.cat(total_txt_pool, dim=0)
        txt_latent = torch.cat(total_txt_latent, dim=0)

        txt_pool = txt_pool.reshape(N_CLASSES, len(templates), -1).mean(dim=1)
        txt_latent = txt_latent.reshape(N_CLASSES, len(templates), txt_latent.shape[-2], txt_latent.shape[-1]).mean(dim=1)
        dist.barrier()
        dist.all_reduce(txt_pool, op=dist.ReduceOp.AVG)
        dist.all_reduce(txt_latent, op=dist.ReduceOp.AVG)
        txt_pool = torch.nn.functional.normalize(txt_pool, dim=-1)
        txt_latent = (
            torch.nn.functional.normalize(txt_latent, dim=-1) if txt_latent is not None else None
        )
        self.txt_pool = txt_pool
        self.txt_latent = txt_latent
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
            # eps=1e-8,
        )
        # scheduler = CosineWarmupScheduler(opt, 2000, self.trainer.estimated_stepping_batches, verbose=True)
        return {
            "optimizer": opt,
            # "lr_scheduler": {
            #     "scheduler": scheduler,
            #     "interval": "step",
            #     "frequency": 1,
            # },
        }
