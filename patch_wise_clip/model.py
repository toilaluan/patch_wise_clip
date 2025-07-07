import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from transformers import CLIPModel, CLIPProcessor, CLIPConfig
from mmdit.mmdit_generalized_pytorch import MMDiT
from .loss import clip_loss


# ───────────────────────── building blocks ─────────────────────── #

class TokenCompressor(nn.Module):
    """Registers‑as‑tokens encoder (text + registers → patch‑tokens)."""

    def __init__(self, hidden: int, img_hidden: int, n_tokens: int, layers: int = 1):
        super().__init__()
        self.register = nn.Parameter(torch.randn(n_tokens, hidden) * 0.02)
        self.mmdit = MMDiT(
            depth=layers,
            dim_modalities=[hidden, hidden],
            dim_cond=256,
            qk_rmsnorm=True,
        )
        self.to_cond = nn.Linear(2, 256, bias=False)
        self.out_proj = nn.Linear(hidden, img_hidden)

    def forward(self, txt_hidden: torch.Tensor, mask: torch.Tensor, meta: torch.Tensor):
        y = self.register.expand(txt_hidden.size(0), -1, -1)  # (B, N_reg, D)
        y, _ = self.mmdit(
            modality_tokens=(y, txt_hidden),
            modality_masks=(None, mask),
            time_cond=self.to_cond(meta),
        )
        y = self.out_proj(y)  # (B, N_reg, D_img)
        return F.normalize(y, dim=-1)  # (B, N, D)


class PatchWiseCLIP(nn.Module):
    """CLIP variant with optional patch‑wise alignment."""

    def __init__(self, clip_id: str = "openai/clip-vit-base-patch32", text_compressing_layers: int = 0):
        super().__init__()
        cfg = CLIPConfig.from_pretrained(clip_id)
        self.model = CLIPModel(cfg)
        self.proc = CLIPProcessor.from_pretrained(clip_id)
        self.text_compressing_layers = text_compressing_layers

        if text_compressing_layers > 0:
            v_cfg = self.model.config.vision_config
            grid = v_cfg.image_size // v_cfg.patch_size
            n_patches = grid * grid
            self.token_compressor = TokenCompressor(
                hidden=self.model.config.text_config.hidden_size,
                img_hidden=v_cfg.hidden_size,
                n_tokens=n_patches,
                layers=text_compressing_layers,
            )
            self.patch_logit_s = nn.Parameter(torch.zeros(n_patches))
        else:
            self.token_compressor = None
            self.patch_logit_s = None
        
    def configure_model(self):
        self.model = torch.compile(self.model)

    # ------------------------------- encoders ------------------------------ #

    def encode_image(self, pixel_values: torch.Tensor):
        out = self.model.vision_model(pixel_values, output_hidden_states=True, interpolate_pos_encoding=False)
        pooled = F.normalize(self.model.visual_projection(out.pooler_output), dim=-1)
        patches = F.normalize(out.last_hidden_state[:, 1:], dim=-1)  # remove CLS
        return pooled, patches  # (B, D), (B, N, D)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, position_ids=None, meta=None):
        out = self.model.text_model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )
        pooled = F.normalize(self.model.text_projection(out.pooler_output), dim=-1)

        if self.text_compressing_layers > 0:
            meta = meta if meta is not None else torch.zeros(len(input_ids), 2, device=input_ids.device)
            compressed_text_features = self.token_compressor(out.last_hidden_state, mask=attention_mask.bool(), meta=meta)
        else:
            compressed_text_features = None
        return pooled, compressed_text_features  # (B, D), (B, N, D)|None

    # ------------------------------- loss ---------------------------------- #

    def compute_loss(self, img_pool, txt_pool, img_patch=None, compressed_text_features=None):
        global_logits = img_pool @ txt_pool.t() * self.model.logit_scale.exp()
        pool_loss = clip_loss(global_logits)

        if img_patch is None or compressed_text_features is None:
            return pool_loss, torch.tensor(0.0, device=img_pool.device)

        scale = self.patch_logit_s.exp()  # (N,)
        patch_sim = (img_patch.unsqueeze(1) * compressed_text_features.unsqueeze(0)).sum(-1) * scale  # (B_img, B_txt, N)
        patch_logits = patch_sim.sum(-1)  # aggregate over patches → (B_img, B_txt)
        patch_loss = clip_loss(patch_logits)
        ctf_sim = (compressed_text_features.unsqueeze(1) * compressed_text_features.unsqueeze(0)).sum(-1).sum(-1)
        ctf_loss = clip_loss(ctf_sim)
        return pool_loss, patch_loss, ctf_loss

    # ------------------------------- forward -------------------------------- #

    def forward(self, *, input_ids, pixel_values, attention_mask=None, position_ids=None, meta_tensor=None, return_loss=False):
        img_pool, img_patch = self.encode_image(pixel_values)
        txt_pool, compressed_text_features = self.encode_text(input_ids, attention_mask, position_ids, meta_tensor)
        if return_loss:
            return self.compute_loss(img_pool, txt_pool, img_patch, compressed_text_features)
        return {
            "text_pooled": txt_pool,
            "text_latents": compressed_text_features,
            "image_pooled": img_pool,
            "image_patches": img_patch,
        }
