import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentEmbedder(nn.Module):
    """
    Transforms arbitrary token length into "latent" tokens that has fixed, compressed size
    """

    def __init__(
        self,
        hidden_size: int = 384,
        intermediate_size=768,
        n_heads: int = 12,
        n_layers: int = 6,
        n_register_tokens: int = 32,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.n_register_tokens = n_register_tokens

        self.register_embeds = nn.Parameter(
            torch.zeros((n_register_tokens, hidden_size))
        )

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size, bias=False),
            nn.GELU(),
            nn.Linear(intermediate_size, intermediate_size, bias=False),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size, bias=False),
        )

    def forward(self, x):
        B, T, D = x.shape
        x = torch.cat([x, self.register_embeds.repeat(B, 1, 1)], dim=1)
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        out = self.ln_1(out + x)

        out = self.ln_2(self.mlp(out) + out)

        out = out[:, -self.n_register_tokens :, :]

        return out


if __name__ == "__main__":
    hidden_size = 384
    intermediate_size = hidden_size * 2
    n_heads = 6
    n_layers = 4
    n_register_tokens: int = 4
    x = torch.rand((1, 32, hidden_size))

    le = LatentEmbedder(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        n_heads=n_heads,
        n_layers=n_layers,
        n_register_tokens=n_register_tokens,
    )

    out = le(x)

    print(out)
