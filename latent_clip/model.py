"""
Lean(er) Latent-CLIP implementation
──────────────────────────────────
Key changes
-----------
1.  `encode_image`  – returns **pooled**  and **patch-wise** features.
2.  `encode_text`   – returns **pooled** and (optionally) **latent / register** features.
3.  `compute_loss`  – combines global CLIP loss **and** patch-alignment loss.
4.  `forward`       – pure inference that *optionally* also returns the three losses.

Everything else (imports, helper losses) stays as in your snippet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor, CLIPConfig
from mmdit.mmdit_generalized_pytorch import MMDiT
from x_transformers import CrossAttender

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    return (contrastive_loss(similarity) + contrastive_loss(similarity.t())) / 2


class LatentEmbedder(nn.Module):
    """
    Registers-as-tokens encoder (text → “patch-sized” latent tokens).
    """

    def __init__(self, hidden: int, image_hidden_size: int, n_tokens: int, layers: int = 1):
        super().__init__()
        self.register = nn.Parameter(torch.randn(n_tokens, hidden) * 0.02)
        # self.q_former = CrossAttender(
        #     dim=hidden,
        #     depth=layers,
        #     heads=8,
        #     use_rmsnorm=True,
        #     rotary_pos_emb=True,
        # )
        self.mmdit = MMDiT(
            depth=layers,
            dim_modalities=[hidden, hidden],
            dim_cond=256,
            qk_rmsnorm = True
        )
        self.out_proj = nn.Linear(hidden, image_hidden_size)
        self.to_cond = nn.Linear(2, 256, bias=False)

    def forward(self, text_hidden, mask, meta):
        y = self.register.expand(text_hidden.size(0), -1, -1)  # (B, N_reg, D)
        # out = self.q_former(
        #     y, context=text_hidden, context_mask=mask
        # )
        y, text_hidden = self.mmdit(
            modality_tokens=(y, text_hidden), modality_masks=(None, mask), time_cond=self.to_cond(meta)
        )
        out = self.out_proj(y)  # (B, N_reg, D)
        return out, y.mean(dim=1)  # (B, N_reg, D) | (B, D)


class LatentCLIP(nn.Module):
    """
    Refactored CLIP with optional patch-wise alignment.
    """

    def __init__(
        self, clip_id: str = "openai/clip-vit-base-patch32", latent_layers: int = 12
    ):
        super().__init__()
        config = CLIPConfig.from_pretrained(clip_id)
        self.model = CLIPModel(config)
        self.proc = CLIPProcessor.from_pretrained(clip_id)
        self.latent_layers = latent_layers

        if latent_layers:  # allocate only when needed
            img_conf = self.model.config.vision_config
            grid_sz = img_conf.image_size // img_conf.patch_size
            n_patches = grid_sz * grid_sz

            self.latent = LatentEmbedder(
                self.model.config.text_config.hidden_size,
                img_conf.hidden_size,
                n_tokens=n_patches,
                layers=latent_layers,
            )
            self.patch_logit_s = nn.Parameter(torch.zeros(n_patches))  # per-patch scale

    # ───────────────────────────── encode utilities ────────────────────────── #

    def encode_image(self, pixel_values, interpolate=False):
        v_out = self.model.vision_model(
            pixel_values,
            output_hidden_states=True,
            interpolate_pos_encoding=interpolate,
        )
        pooled = F.normalize(self.model.visual_projection(v_out.pooler_output), dim=-1)
        patches = F.normalize(v_out.last_hidden_state[:, 1:], dim=-1)  # drop CLS
        return pooled, patches  # (B, D), (B, N, D)

    def encode_text(self, input_ids, attention_mask, position_ids=None, meta=None):
        t_out = self.model.text_model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )
        pooled = F.normalize(self.model.text_projection(t_out.pooler_output), dim=-1)

        latents = None
        if self.latent_layers:
            latents, pool = self.latent(
                t_out.last_hidden_state, mask=attention_mask.bool(), meta=meta
            )
            # pooled = F.normalize(self.model.text_projection(pool), dim=-1)
            # pooled = latents.mean(dim=1)  # (B, N, D) → (B, D)
            # pooled = F.normalize(pooled, dim=-1)  # (B, D)
            latents = F.normalize(latents, dim=-1)
            # pooled = self.model.text_projection(pooled)  # (B, D) → (B, D)
        return pooled, latents  # (B, D), (B, N, D) | None

    # ─────────────────────────────── losses ────────────────────────────────── #

    def compute_loss(self, img_pool, txt_pool, img_patch=None, txt_latent=None):
        global_logits = img_pool @ txt_pool.t() * self.model.logit_scale.exp()
        pool_loss = clip_loss(global_logits)

        patch_loss = torch.tensor(0.0, device=img_pool.device)
        if img_patch is not None and txt_latent is not None:
            # loop over patch index for clarity; N is usually small (≤49)
            losses = []
            for i in range(img_patch.size(1)):
                logits = (txt_latent[:, i] @ img_patch[:, i].t()) * self.patch_logit_s[
                    i
                ].exp()
                losses.append(clip_loss(logits))
            patch_loss = torch.stack(losses).mean()

        return pool_loss, patch_loss

    # ─────────────────────────────── forward ───────────────────────────────── #

    def forward(
        self,
        input_ids,
        pixel_values,
        attention_mask=None,
        position_ids=None,
        meta_tensor=None,
        return_loss=False,
    ):
        img_pool, img_patch = self.encode_image(pixel_values)
        txt_pool, txt_latent = self.encode_text(
            input_ids, attention_mask, position_ids, meta_tensor
        )

        if return_loss:
            return self.compute_loss(img_pool, txt_pool, img_patch, txt_latent)

        # pure inference
        return {
            "text_pooled": txt_pool,  # (B, D)
            "text_latents": txt_latent,  # (B, N, D) | None
            "image_pooled": img_pool,  # (B, D)
            "image_patches": img_patch,  # (B, N, D)
        }
