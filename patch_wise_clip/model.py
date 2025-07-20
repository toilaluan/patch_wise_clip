import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from transformers import CLIPModel, CLIPProcessor, CLIPConfig
from .transformer_model import Transformer
import numpy as np

# ───────────────────────── building blocks ─────────────────────── #


def _get_vector_norm(tensor: torch.Tensor) -> torch.Tensor:
    """
    This method is equivalent to tensor.norm(p=2, dim=-1, keepdim=True) and used to make
    model `executorch` exportable. See issue https://github.com/pytorch/executorch/issues/3566
    """
    square_tensor = torch.pow(tensor, 2)
    sum_tensor = torch.sum(square_tensor, dim=-1, keepdim=True)
    normed_tensor = torch.pow(sum_tensor, 0.5)
    return normed_tensor


class TokenCompressor(nn.Module):
    """Registers‑as‑tokens encoder (text + registers → patch‑tokens)."""

    def __init__(self, hidden: int, img_hidden: int, n_tokens: int, layers: int = 1):
        super().__init__()
        print(
            f"n_tokens: {n_tokens}, hidden: {hidden}, img_hidden: {img_hidden}, layers: {layers}"
        )
        self.register = nn.Parameter(torch.randn(n_tokens, hidden) * 0.02)
        self.encoder = Transformer(
            hidden_size=hidden,
            n_heads=8,
            n_layers=layers,
            mlp_ratio=4,
            eps=1e-5,
        )
        self.out_proj = nn.Linear(hidden, img_hidden)

    def forward(self, txt_hidden: torch.Tensor, mask: torch.Tensor, meta: torch.Tensor):
        y = self.register.expand(txt_hidden.size(0), -1, -1)
        mask = torch.cat(
            [mask, torch.ones(mask.size(0), self.register.size(0), device=mask.device)],
            dim=1,
        )
        y = torch.cat([txt_hidden, y], dim=1)
        y = self.encoder(y, mask)
        y = self.out_proj(y)
        return y[:, -self.register.size(0) :, :]


class PatchWiseCLIP(nn.Module):
    """CLIP variant with optional patch‑wise alignment."""

    def __init__(
        self,
        clip_id: str = "openai/clip-vit-base-patch32",
        text_compressing_layers: int = 0,
    ):
        super().__init__()
        cfg = CLIPConfig.from_pretrained(clip_id)
        self.model = CLIPModel(cfg)
        self.processor = CLIPProcessor.from_pretrained(clip_id)
        self.text_compressing_layers = text_compressing_layers
        self.pool_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.patch_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

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
        out = self.model.vision_model(
            pixel_values, output_hidden_states=True, interpolate_pos_encoding=False
        )
        pooled = self.model.visual_projection(out.pooler_output)
        pooled = pooled / _get_vector_norm(pooled)
        patches = out.last_hidden_state[:, 1:]  # remove CLS
        patches = patches / _get_vector_norm(patches)
        return pooled, patches  # (B, D), (B, N, D)

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids=None,
        meta=None,
    ):
        out = self.model.text_model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )
        pooled = self.model.text_projection(out.pooler_output)
        pooled = pooled / _get_vector_norm(pooled)

        if self.text_compressing_layers > 0:
            meta = (
                meta
                if meta is not None
                else torch.zeros(len(input_ids), 2, device=input_ids.device)
            )
            compressed_text_features = self.token_compressor(
                out.last_hidden_state, mask=attention_mask.bool(), meta=meta
            )
            compressed_text_features = compressed_text_features / _get_vector_norm(
                compressed_text_features
            )
        else:
            compressed_text_features = None
        return pooled, compressed_text_features

    # ------------------------------- forward -------------------------------- #

    def forward(
        self,
        *,
        input_ids,
        pixel_values,
        attention_mask=None,
        position_ids=None,
        meta_tensor=None,
        return_loss=False,
    ):
        img_pool, img_patch = self.encode_image(pixel_values)
        txt_pool, compressed_text_features = self.encode_text(
            input_ids, attention_mask, position_ids, meta_tensor
        )
        if return_loss:
            return self.compute_loss(
                img_pool, txt_pool, img_patch, compressed_text_features
            )
        return {
            "text_pooled": txt_pool,
            "text_latents": compressed_text_features,
            "image_pooled": img_pool,
            "image_patches": img_patch,
        }
