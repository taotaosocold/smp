"""DiT-style ε-prediction denoiser for motion windows (optionally terrain-conditioned).

Terrain conditioning uses per-frame spatial encoding (Conv2D over the height-map
grid) followed by cross-attention in each DiT block, so every motion frame can
attend to its corresponding terrain context.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Grid dimensions of the height map ──
GRID_X = 17
GRID_Y = 11


class _Timesteps(nn.Module):
  """Sinusoidal timestep features, cos first, sin second."""

  def __init__(self, num_channels: int = 256) -> None:
    super().__init__()
    if num_channels % 2 != 0:
      msg = f"_Timesteps requires even num_channels, got {num_channels}"
      raise ValueError(msg)
    self.num_channels = num_channels

  def forward(self, t: torch.Tensor) -> torch.Tensor:
    half = self.num_channels // 2
    exponent = -math.log(10000.0) * torch.arange(
      half, dtype=torch.float32, device=t.device
    )
    exponent = exponent / half
    emb = t.float()[:, None] * torch.exp(exponent)[None, :]
    return torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)


class _TimestepEmbedding(nn.Module):
  def __init__(self, in_channels: int, time_embed_dim: int) -> None:
    super().__init__()
    self.linear_1 = nn.Linear(in_channels, time_embed_dim)
    self.act = nn.SiLU()
    self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.linear_2(self.act(self.linear_1(x)))


class _SpatialTerrainEncoder(nn.Module):
  """Per-frame 2D Conv encoder for height maps.

  Input ``(B, W, 187)`` where 187 = 17×11, output ``(B, W, embed_dim)``.
  Each frame's height map is treated as a 1-channel 17×11 image so the
  encoder preserves the spatial structure (e.g. "higher on the left").
  """

  def __init__(self, embed_dim: int) -> None:
    super().__init__()
    self.conv = nn.Sequential(
      nn.Conv2d(1, 32, 3, padding=1),   # 17×11 → 17×11×32
      nn.SiLU(),
      nn.Conv2d(32, 64, 3, padding=1),  # 17×11×64
      nn.SiLU(),
    )
    self.pool = nn.AdaptiveAvgPool2d((4, 4))  # → 4×4×64 = 1024
    self.proj = nn.Linear(1024, embed_dim)

  def forward(self, terrain: torch.Tensor) -> torch.Tensor:
    B, W, _ = terrain.shape
    h = terrain.reshape(B * W, 1, GRID_X, GRID_Y)  # (B·W, 1, 17, 11)
    h = self.pool(self.conv(h))                     # (B·W, 64, 4, 4)
    h = self.proj(h.flatten(1))                     # (B·W, embed_dim)
    return h.reshape(B, W, -1)                      # (B, W, embed_dim)


class _AdaLayerNormSingle(nn.Module):
  """PixArt-α adaLN-single: produce (B, 1, 6·D) timestep modulation."""

  def __init__(self, embedding_dim: int) -> None:
    super().__init__()
    self.time_proj = _Timesteps(num_channels=256)
    self.timestep_embedder = _TimestepEmbedding(256, embedding_dim)
    self.silu = nn.SiLU()
    self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=True)

  def forward(self, t: torch.Tensor) -> torch.Tensor:
    t_emb = self.timestep_embedder(self.time_proj(t)).unsqueeze(1)
    return self.linear(self.silu(t_emb))


class _SinusoidalPositionalEmbedding(nn.Module):
  pe: torch.Tensor

  def __init__(self, embed_dim: int, max_seq_length: int = 32) -> None:
    super().__init__()
    position = torch.arange(max_seq_length).unsqueeze(1)
    div_term = torch.exp(
      torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
    )
    pe = torch.zeros(1, max_seq_length, embed_dim)
    pe[0, :, 0::2] = torch.sin(position * div_term)
    pe[0, :, 1::2] = torch.cos(position * div_term)
    self.register_buffer("pe", pe, persistent=False)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x + self.pe[:, : x.shape[1]]


class _SwiGLU(nn.Module):
  def __init__(self, dim: int, inner_dim: int, bias: bool = True) -> None:
    super().__init__()
    self.proj = nn.Linear(dim, inner_dim * 2, bias=bias)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    a, b = self.proj(x).chunk(2, dim=-1)
    return F.silu(a) * b


class _FeedForward(nn.Module):
  def __init__(
    self,
    dim: int,
    mult: int = 4,
    dropout: float = 0.0,
    bias: bool = True,
  ) -> None:
    super().__init__()
    inner_dim = dim * mult
    self.act = _SwiGLU(dim, inner_dim, bias=bias)
    self.dropout = nn.Dropout(dropout)
    self.proj_out = nn.Linear(inner_dim, dim, bias=bias)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.proj_out(self.dropout(self.act(x)))


class _DiTBlock(nn.Module):
  """Self-attn → cross-attn (optional) → SwiGLU FFN, all modulated by adaLN."""

  def __init__(
    self,
    dim: int,
    num_heads: int,
    head_dim: int,
    dropout: float = 0.0,
    norm_eps: float = 1e-5,
  ) -> None:
    super().__init__()
    self.dim = dim
    self.num_heads = num_heads
    self.head_dim = head_dim
    self.attn_dim = num_heads * head_dim

    # --- Self-attention ---
    self.norm1 = nn.LayerNorm(dim, eps=norm_eps, elementwise_affine=False)
    self.to_q = nn.Linear(dim, self.attn_dim, bias=False)
    self.to_k = nn.Linear(dim, self.attn_dim, bias=False)
    self.to_v = nn.Linear(dim, self.attn_dim, bias=False)
    self.to_out = nn.Linear(self.attn_dim, dim, bias=False)
    self.attn_dropout = nn.Dropout(dropout)

    # --- Cross-attention (motion → terrain) ---
    self.norm_cross = nn.LayerNorm(dim, eps=norm_eps, elementwise_affine=False)
    self.to_q_cross = nn.Linear(dim, self.attn_dim, bias=False)
    self.to_k_cross = nn.Linear(dim, self.attn_dim, bias=False)
    self.to_v_cross = nn.Linear(dim, self.attn_dim, bias=False)
    self.to_out_cross = nn.Linear(self.attn_dim, dim, bias=False)
    self.gate_cross = nn.Parameter(torch.zeros(1, 1, 1))

    # --- FFN ---
    self.norm2 = nn.LayerNorm(dim, eps=norm_eps, elementwise_affine=False)
    self.ff = _FeedForward(dim, mult=4, dropout=dropout)

    self.scale_shift_table = nn.Parameter(torch.randn(1, 1, 6, dim) / dim**0.5)

  def _attn(self, x: torch.Tensor) -> torch.Tensor:
    B, N, _ = x.shape
    h, d = self.num_heads, self.head_dim
    q = self.to_q(x).reshape(B, N, h, d).transpose(1, 2)
    k = self.to_k(x).reshape(B, N, h, d).transpose(1, 2)
    v = self.to_v(x).reshape(B, N, h, d).transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    out = out.transpose(1, 2).reshape(B, N, h * d)
    return self.attn_dropout(self.to_out(out))

  def _cross_attn(
    self, x: torch.Tensor, terrain_tokens: torch.Tensor
  ) -> torch.Tensor:
    B, N, _ = x.shape
    h, d = self.num_heads, self.head_dim
    q = self.to_q_cross(x).reshape(B, N, h, d).transpose(1, 2)
    k = self.to_k_cross(terrain_tokens).reshape(B, N, h, d).transpose(1, 2)
    v = self.to_v_cross(terrain_tokens).reshape(B, N, h, d).transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    out = out.transpose(1, 2).reshape(B, N, h * d)
    return self.to_out_cross(out)

  def forward(
    self,
    x: torch.Tensor,
    time_hidden_states: torch.Tensor,
    terrain_tokens: torch.Tensor | None = None,
  ) -> torch.Tensor:
    B = x.shape[0]
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
      self.scale_shift_table + time_hidden_states.reshape(B, 1, 6, -1)
    ).chunk(6, dim=-2)

    # Self-attention
    h = self.norm1(x)
    h = h * (1 + scale_msa.squeeze(-2)) + shift_msa.squeeze(-2)
    x = x + gate_msa.squeeze(-2) * self._attn(h)

    # Cross-attention (motion queries terrain)
    if terrain_tokens is not None:
      h_cross = self.norm_cross(x)
      x = x + self.gate_cross * self._cross_attn(h_cross, terrain_tokens)

    # FFN
    h = self.norm2(x)
    h = h * (1 + scale_mlp.squeeze(-2)) + shift_mlp.squeeze(-2)
    x = x + gate_mlp.squeeze(-2) * self.ff(h)

    return x


class DiffusionDenoiser(nn.Module):
  """ε-prediction DiT for motion windows (optionally terrain-conditioned).

  Pipeline: 1×1 Conv1d (channel mix, residual) → linear-in → adaLN-single
  timestep → additive sin-cos PE → ``num_layers`` DiT blocks (each with
  optional cross-attention to terrain tokens) → linear-out → 1×1 Conv1d
  (residual).

  Input:  ``x_t (B, W, feature_dim)``, ``t (B,)`` long timesteps,
          ``terrain (B, W, terrain_dim)`` optional
  Output: predicted noise ``(B, W, feature_dim)``
  """

  def __init__(
    self,
    feature_dim: int,
    window_size: int,
    d_model: int = 256,
    nhead: int = 4,
    num_layers: int = 2,
    dropout: float = 0.0,
    head_dim: int | None = None,
    terrain_dim: int | None = None,
  ) -> None:
    super().__init__()
    self.feature_dim = feature_dim
    self.window_size = window_size

    if head_dim is None:
      if d_model % nhead != 0:
        msg = (
          f"d_model ({d_model}) must be divisible by nhead ({nhead}) "
          f"when head_dim is unspecified"
        )
        raise ValueError(msg)
      head_dim = d_model // nhead
    self.num_heads = nhead
    self.head_dim = head_dim
    self.inner_dim = nhead * head_dim
    if self.inner_dim != d_model:
      msg = (
        f"d_model ({d_model}) must equal nhead·head_dim "
        f"({nhead}·{head_dim} = {self.inner_dim})"
      )
      raise ValueError(msg)

    self.preprocess_conv = nn.Conv1d(feature_dim, feature_dim, 1, bias=False)
    self.proj_in = nn.Linear(feature_dim, self.inner_dim, bias=False)

    if terrain_dim is not None:
      self.terrain_encoder = _SpatialTerrainEncoder(self.inner_dim)
    else:
      self.terrain_encoder = None

    self.adaln_single = _AdaLayerNormSingle(self.inner_dim)
    self.sequence_pos_encoder = _SinusoidalPositionalEmbedding(
      self.inner_dim, max_seq_length=max(window_size, 32)
    )
    self.blocks = nn.ModuleList(
      [
        _DiTBlock(
          dim=self.inner_dim,
          num_heads=nhead,
          head_dim=head_dim,
          dropout=dropout,
        )
        for _ in range(num_layers)
      ]
    )
    self.proj_out = nn.Linear(self.inner_dim, feature_dim, bias=False)
    self.postprocess_conv = nn.Conv1d(feature_dim, feature_dim, 1, bias=False)

  def forward(
    self,
    x_t: torch.Tensor,
    t: torch.Tensor,
    terrain: torch.Tensor | None = None,
  ) -> torch.Tensor:
    h = x_t.transpose(1, 2)
    h = self.preprocess_conv(h) + h
    h = h.transpose(1, 2)

    h = self.proj_in(h)

    terrain_tokens = None
    if terrain is not None and self.terrain_encoder is not None:
      terrain_tokens = self.terrain_encoder(terrain)  # (B, W, 256)

    time_hidden_states = self.adaln_single(t)
    h = self.sequence_pos_encoder(h)

    for block in self.blocks:
      h = block(h, time_hidden_states, terrain_tokens=terrain_tokens)

    h = self.proj_out(h)
    h = h.transpose(1, 2)
    h = self.postprocess_conv(h) + h
    return h.transpose(1, 2)
