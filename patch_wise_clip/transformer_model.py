import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.dropout = dropout
        self.head_dim = hidden_size // n_heads
        self.to_q = nn.Linear(hidden_size, hidden_size)
        self.to_k = nn.Linear(hidden_size, hidden_size)
        self.to_v = nn.Linear(hidden_size, hidden_size)
        self.to_out = nn.Linear(hidden_size, hidden_size)

        self.q_norm = nn.RMSNorm(hidden_size, eps=1e-5)
        self.k_norm = nn.RMSNorm(hidden_size, eps=1e-5)

    def forward(self, x, mask=None, is_causal=False):
        batch_size, seq_len, dim = x.shape
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = q.view(q.size(0), -1, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(k.size(0), -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(v.size(0), -1, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Fix the mask processing
        if mask is not None:
            # Convert 1D mask to 2D attention mask
            # mask shape: [batch_size, seq_len] -> [batch_size, 1, seq_len, seq_len]
            mask = mask.unsqueeze(1) * mask.unsqueeze(2)  # [batch_size, seq_len, seq_len]
            mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]
            # Convert to boolean mask (True for positions to attend to)
            mask = mask.bool()
            if is_causal:
                mask = mask.tril(diagonal=0)
        
        out = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=mask, 
            dropout_p=self.dropout if self.training else 0, 
            is_causal=False
        )
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        out = self.to_out(out)
        return out

class FeedForward(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        mlp_ratio: int = 4,
    ):
        super().__init__()
        self.ff_1 = nn.Linear(hidden_size, hidden_size * mlp_ratio)
        self.ff_2 = nn.Linear(hidden_size * mlp_ratio, hidden_size)
        self.norm = nn.RMSNorm(hidden_size, eps=1e-5)
        self.gate_proj = nn.Linear(mlp_ratio * hidden_size, hidden_size * mlp_ratio * 2)

    def forward(self, x):
        x = self.ff_1(x)
        gates = self.gate_proj(x).chunk(2, dim=-1)
        x = F.silu(gates[0]) * gates[1]
        x = self.ff_2(x)
        x = self.norm(x)
        return x

class Block(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        dropout: float = 0.1,
        mlp_ratio: int = 4,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.attn = Attention(hidden_size, n_heads, dropout)
        self.norm1 = nn.RMSNorm(hidden_size, eps=eps)
        self.norm2 = nn.RMSNorm(hidden_size, eps=eps)
        self.mlp = FeedForward(hidden_size, mlp_ratio)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x

class Transformer(nn.Module):
    def __init__(
        self, 
        hidden_size: int,
        n_heads: int,
        n_layers: int,
        dropout: float = 0.1,
        mlp_ratio: int = 4,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            Block(hidden_size, n_heads, dropout, mlp_ratio, eps)
            for _ in range(n_layers)
        ])

    def forward(self, h, mask=None):
        for layer in self.layers:
            h = layer(h, mask)
        return h

if __name__ == "__main__":
    model = Transformer(
        hidden_size=512,
        n_heads=4,
        n_layers=2,
    ).to("cuda")
    h = torch.randn(2, 126, 512).to("cuda")
    mask = torch.ones(2, 126, device=h.device).to("cuda")
    print(model(h, mask).shape)