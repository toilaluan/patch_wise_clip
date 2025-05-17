# [Imports remain the same as the previous refactored version]
from transformers import CLIPProcessor, CLIPModel, CLIPConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.configuration_clip import CLIPTextConfig, CLIPVisionConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from typing import Optional, Tuple
from mmdit.mmdit_generalized_pytorch import MMDiT


# [contrastive_loss and clip_loss functions remain the same]
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


class LatentEmbedder(nn.Module):
    def __init__(self, hidden_size: int, n_register_tokens: int, n_layers: int = 1):
        super().__init__()
        self.register_embeds = nn.Parameter(
            torch.randn(n_register_tokens, hidden_size) * 0.02
        )

        self.blocks = MMDiT(
            depth=n_layers,
            dim_modalities=[hidden_size, hidden_size],
            dim_cond=256,
        )
        self.n_register_tokens = n_register_tokens
        self.hidden_size = hidden_size
        self.meta_tensor_projector = nn.Linear(2, 256, bias=False)
        self.meta_tensor_projector.weight.data.normal_(0, 0.02)

    def forward(self, x, attention_mask, meta_tensor=None):
        """
        Forward pass of the LatentEmbedder.

        Args:
            x: Input token embeddings tensor of shape (B, T_txt, D).

        Returns:
            A tensor containing the final states of the register tokens,
            shape (B, n_register_tokens, D).
        """
        B, T_txt, D = x.shape
        device = x.device
        cond = self.meta_tensor_projector(meta_tensor)
        # --- Register Token Embedding ---
        y = self.register_embeds.expand(B, -1, -1).to(device)

        x, y = self.blocks(
            modality_tokens=(x, y),
            modality_masks=(attention_mask, None),
            time_cond=cond,
        )

        # --- Extract Register Tokens ---
        return y


class LatentClip(nn.Module):
    """
    A modified CLIP model that incorporates a positional alignment loss
    between latent text representations (derived respecting padding) and
    image patch embeddings, in addition to the standard global contrastive loss.
    """

    def __init__(
        self,
        pretrained_clip_id: str = "openai/clip-vit-base-patch32",
        n_register_layers: int = 12,
    ):
        """
        [Init docstring remains the same]
        """
        super().__init__()
        print(f"Initializing LatentClip with {pretrained_clip_id}")
        # config = CLIPConfig.from_pretrained(pretrained_clip_id)
        # self.model = CLIPModel(config)
        self.model = CLIPModel.from_pretrained(pretrained_clip_id)
        self.processor = CLIPProcessor.from_pretrained(pretrained_clip_id)
        self.text_config: CLIPTextConfig = self.model.config.text_config
        self.image_config: CLIPVisionConfig = self.model.config.vision_config

        grid_size = self.image_config.image_size // self.image_config.patch_size
        n_patch_tokens = grid_size * grid_size

        print(f"Text hidden size: {self.text_config.hidden_size}")
        print(f"Vision hidden size: {self.image_config.hidden_size}")
        # Check compatibility *before* projection (relevant for positional loss)
        if self.text_config.hidden_size != self.image_config.hidden_size:
            print(
                f"Warning: Text encoder hidden size ({self.text_config.hidden_size}) != "
                f"Vision encoder hidden size ({self.image_config.hidden_size}). "
                f"Positional comparison uses encoder outputs directly."
            )
            # This might still work if CLIP implementation ensures they are comparable,
            # but it's worth noting.

        self.latent_embedder = LatentEmbedder(
            hidden_size=self.text_config.hidden_size,
            n_register_tokens=n_patch_tokens,
            n_layers=n_register_layers,
        )
        self.patch_logit_scale = nn.Parameter(torch.zeros((n_patch_tokens,)))
        self.latent_proj = nn.Linear(
            self.text_config.hidden_size, self.image_config.hidden_size, bias=False
        )
        print(f"Initialized LatentEmbedder with {n_patch_tokens} register tokens.")

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,  # This mask is now crucial
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = True,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = True,
        interpolate_pos_encoding: bool = False,
        meta_tensor: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        [Forward docstring remains the same, but note attention_mask importance]
        """
        if input_ids is None or pixel_values is None:
            raise ValueError("Both input_ids and pixel_values must be provided.")
        # Added check for attention_mask when needed
        if attention_mask is None and input_ids is not None:
            print(
                "Warning: input_ids provided but attention_mask is None. Assuming no padding."
            )
            # Create a default mask if needed, though processor usually provides it
            # attention_mask = torch.ones_like(input_ids)

        # 1. Get Encoder Outputs
        vision_outputs: BaseModelOutputWithPooling = self.model.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=True,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        text_outputs: BaseModelOutputWithPooling = self.model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,  # Pass mask to text model
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=True,
        )

        image_embeds_pooled = vision_outputs.pooler_output
        image_embeds_pooled = self.model.visual_projection(image_embeds_pooled)

        image_embeds_pooled_norm = image_embeds_pooled / torch.linalg.vector_norm(
            image_embeds_pooled, dim=-1, keepdim=True, ord=2
        )
        vision_hidden_states = (
            vision_outputs.last_hidden_state
        )  # (B, 1 + N_p, D_img) or (B, N_p, D_img)

        # 3. Positional Contrastive Loss
        text_hidden_states = text_outputs.last_hidden_state  # (B, T_txt, D_txt)

        # --- Extract patch embeddings [Remains the same] ---
        if vision_hidden_states.shape[1] == self.latent_embedder.n_register_tokens + 1:
            vision_patch_hidden_states = vision_hidden_states[:, 1:, :]
        elif vision_hidden_states.shape[1] == self.latent_embedder.n_register_tokens:
            vision_patch_hidden_states = vision_hidden_states
        else:
            raise ValueError(
                f"Vision hidden state seq len ({vision_hidden_states.shape[1]}) "
                f"mismatch with expected patches ({self.latent_embedder.n_register_tokens})."
            )
        latent_text_hidden_states = self.latent_embedder(
            text_hidden_states,
            attention_mask=attention_mask.bool(),
            meta_tensor=meta_tensor,
        )
        latent_text_hidden_states = self.latent_proj(latent_text_hidden_states)

        # Assert dimensions match [Remains the same]
        if latent_text_hidden_states.shape != vision_patch_hidden_states.shape:
            raise ValueError(
                f"Shape mismatch after LatentEmbedder: Latent Text {latent_text_hidden_states.shape} "
                f"vs Vision Patch {vision_patch_hidden_states.shape}"
            )

        B, N, D = latent_text_hidden_states.shape
        latent_text_norm = latent_text_hidden_states / torch.linalg.vector_norm(
            latent_text_hidden_states, dim=-1, keepdim=True, ord=2
        )
        vision_patch_norm = vision_patch_hidden_states / torch.linalg.vector_norm(
            vision_patch_hidden_states, dim=-1, keepdim=True, ord=2
        )

        t_permuted = latent_text_norm.permute(1, 0, 2)
        v_permuted_t = vision_patch_norm.permute(1, 2, 0)

        # Calculate clip_loss per position and average [Remains the same]
        positional_losses = []
        for i in range(N):
            logit_scale_i = self.patch_logit_scale[i].exp()
            logits_i = torch.matmul(t_permuted[i], v_permuted_t[i]) * logit_scale_i
            loss_i = clip_loss(logits_i)
            positional_losses.append(loss_i)
        position_clip_loss = torch.stack(positional_losses).mean()

        return position_clip_loss


# --- Example Usage (Basic Test) ---
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Test LatentEmbedder with Mask ---
    print("\n--- Testing LatentEmbedder with Mask ---")
    le_hidden_size = 768
    le_intermediate = le_hidden_size * 4
    le_n_registers = 49
    le_batch_size = 4
    le_seq_len = 77

    le_dummy_input = torch.rand(le_batch_size, le_seq_len, le_hidden_size).to(device)
    # Create a dummy mask with padding at the end for some samples
    le_dummy_mask = torch.ones(le_batch_size, le_seq_len, dtype=torch.long).to(device)
    le_dummy_mask[1, 60:] = 0  # Sample 1 has padding
    le_dummy_mask[3, 70:] = 0  # Sample 3 has padding

    latent_embedder = LatentEmbedder(
        hidden_size=le_hidden_size,
        intermediate_size=le_intermediate,
        n_register_tokens=le_n_registers,
    ).to(device)
    latent_embedder.eval()

    with torch.no_grad():
        le_output = latent_embedder(
            le_dummy_input, attention_mask=le_dummy_mask
        )  # Pass the mask

    print(f"LatentEmbedder Input shape: {le_dummy_input.shape}")
    print(f"LatentEmbedder Mask shape: {le_dummy_mask.shape}")
    print(f"LatentEmbedder Output shape: {le_output.shape}")
    assert le_output.shape == (le_batch_size, le_n_registers, le_hidden_size)
    print("LatentEmbedder with mask test passed.")

    # --- Test LatentClip ---
    print("\n--- Testing LatentClip ---")
    # [Rest of the LatentClip test code remains the same]
    # It implicitly uses the attention_mask generated by the processor
    try:
        model_id = "openai/clip-vit-base-patch32"
        latent_clip_model = LatentClip(pretrained_clip_id=model_id).to(device)
        for p in latent_clip_model.model.parameters():
            p.requires_grad = False
        latent_clip_model.eval()

        processor = CLIPProcessor.from_pretrained(model_id)
        dummy_texts = [
            "a photo of a red background",
            "a drawing of a blue background",
        ] * 2
        # Make one text longer to ensure padding occurs if max_length isn't hit
        dummy_texts[1] += " " + " ".join(["word"] * 20)
        dummy_images = [
            Image.new("RGB", (224, 224), color="red"),
            Image.new("RGB", (224, 224), color="blue"),
        ] * 2

        inputs = processor(
            text=dummy_texts,
            images=dummy_images,
            return_tensors="pt",
            padding="max_length",  # Pad to max length (e.g., 77)
            truncation=True,
        ).to(device)

        print(f"Model Input - input_ids shape: {inputs['input_ids'].shape}")
        print(f"Model Input - pixel_values shape: {inputs['pixel_values'].shape}")
        print(f"Model Input - attention_mask shape: {inputs['attention_mask'].shape}")
        # Verify mask contains zeros if padding occurred
        if torch.any(inputs["attention_mask"] == 0):
            print("Attention mask contains padding (zeros) as expected.")
        else:
            print("Attention mask does not contain padding (all ones).")

        with torch.no_grad():
            total_loss, pool_loss, position_loss = latent_clip_model(**inputs)

        print(f"\nLatentClip Output:")
        print(f"  Total Loss: {total_loss.item():.4f}")
        print(f"  Pool Loss: {pool_loss.item():.4f}")
        print(f"  Position Loss: {position_loss.item():.4f}")
        print("LatentClip forward pass test completed.")

    except Exception as e:
        import traceback

        print(f"\nError during LatentClip test: {e}")
        traceback.print_exc()
        print("Please ensure CLIP model weights can be downloaded and config matches.")
