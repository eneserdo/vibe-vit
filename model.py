"""
Vision Transformer (ViT) — written explicitly for easy attention experimentation.

Key entry points:
  - Attention      : self-attention with stored attn_weights; swap this class to try variants
  - Block          : one transformer layer (norm → attn → norm → mlp)
  - VisionTransformer : full model
  - build_model(args) : convenience factory
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────────────
# Patch embedding
# ──────────────────────────────────────────────────────────────────────────────

class PatchEmbed(nn.Module):
    """Split image into non-overlapping patches and project to embed_dim."""

    def __init__(self, img_size: int = 64, patch_size: int = 8, in_chans: int = 3,
                 embed_dim: int = 192):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)  →  (B, num_patches, embed_dim)
        x = self.proj(x)          # (B, D, H/P, W/P)
        x = x.flatten(2)          # (B, D, N)
        x = x.transpose(1, 2)     # (B, N, D)
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Multi-head self-attention — fully explicit, no black-box fused kernels
# ──────────────────────────────────────────────────────────────────────────────

class Attention(nn.Module):
    """
    Standard scaled dot-product multi-head self-attention.

    After every forward pass, self.attn_weights holds the softmax attention
    map of shape (B, num_heads, N, N).  Override or read this tensor freely
    when experimenting with attention variants.
    """

    def __init__(self, dim: int, num_heads: int = 3,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Single fused projection for Q, K, V — split explicitly in forward()
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # Populated after every forward; shape (B, num_heads, N, N)
        self.attn_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        H, Dh = self.num_heads, self.head_dim

        # Project and split into Q, K, V
        qkv = self.qkv(x)                      # (B, N, 3*D)
        qkv = qkv.reshape(B, N, 3, H, Dh)      # (B, N, 3, H, Dh)
        qkv = qkv.permute(2, 0, 3, 1, 4)       # (3, B, H, N, Dh)
        q, k, v = qkv.unbind(0)                 # each (B, H, N, Dh)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Store for inspection / experiment hooks
        self.attn_weights = attn.detach()

        # Aggregate values and project
        out = attn @ v                          # (B, H, N, Dh)
        out = out.transpose(1, 2).reshape(B, N, D)  # (B, N, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Feed-forward MLP
# ──────────────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """Two-layer MLP with GELU activation (same as original ViT / DeiT)."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Transformer block
# ──────────────────────────────────────────────────────────────────────────────

class Block(nn.Module):
    """Pre-norm transformer block: LN → Attn → residual, LN → MLP → residual."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads,
                              attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, drop=proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Vision Transformer
# ──────────────────────────────────────────────────────────────────────────────

class VisionTransformer(nn.Module):
    """
    ViT-Tiny (default) tuned for Tiny ImageNet 64×64 images.

    Default config:
      patch_size=8  →  64 tokens
      embed_dim=192, depth=12, num_heads=3
    """

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        in_chans: int = 3,
        num_classes: int = 200,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # ── Patch embedding ───────────────────────────────────────────────────
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # ── CLS token + positional embeddings ────────────────────────────────
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        # ── Transformer blocks ────────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads,
                  mlp_ratio=mlp_ratio,
                  attn_drop=attn_drop_rate,
                  proj_drop=drop_rate)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # ── Classification head ───────────────────────────────────────────────
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    # ── Weight initialisation (ViT / DeiT convention) ─────────────────────────
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        x = self.patch_embed(x)                         # (B, N, D)

        cls = self.cls_token.expand(B, -1, -1)          # (B, 1, D)
        x = torch.cat([cls, x], dim=1)                  # (B, N+1, D)
        x = self.pos_drop(x + self.pos_embed)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls_out = x[:, 0]                               # (B, D)  — CLS token only
        return self.head(cls_out)

    # ── Convenience: return attention weights from all blocks ─────────────────
    def get_all_attn_weights(self) -> list[torch.Tensor]:
        """
        Returns a list of length `depth`, each tensor of shape (B, H, N+1, N+1).
        Call after forward().
        """
        return [block.attn.attn_weights for block in self.blocks]


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def build_model(args) -> VisionTransformer:
    model = VisionTransformer(
        img_size=args.img_size,
        patch_size=args.patch_size,
        num_classes=args.num_classes,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        drop_rate=args.drop_rate,
        attn_drop_rate=args.attn_drop_rate,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params / 1e6:.1f}M parameters  |  "
          f"patch_size={args.patch_size}  embed_dim={args.embed_dim}  "
          f"depth={args.depth}  heads={args.num_heads}")
    return model
