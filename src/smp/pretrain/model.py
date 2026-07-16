"""DiT-style ε-prediction denoiser for motion windows (optionally terrain-conditioned).

Terrain conditioning uses cross-attention: Q comes from the noisy motion sequence,
K and V come from terrain features.  When ``terrain_dim`` is None the model falls
back to self-attention, yielding the original unconditional DiT.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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
  """Cross-attention → SwiGLU FFN, all modulated by adaLN.

  When ``terrain`` is None the block falls back to self-attention on the
  motion stream, matching the original unconditional DiT behaviour.

  Q always comes from the (modulated) motion stream.  K and V come from
  the terrain stream when available, otherwise from the motion stream
  (self-attention).
  """

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

    # Q projection (always from motion stream)
    self.norm1 = nn.LayerNorm(dim, eps=norm_eps, elementwise_affine=False)
    self.to_q = nn.Linear(dim, self.attn_dim, bias=False)

    # K, V projections (from terrain when available, otherwise motion)
    self.norm_kv = nn.LayerNorm(dim, eps=norm_eps, elementwise_affine=False)
    self.to_k = nn.Linear(dim, self.attn_dim, bias=False)
    self.to_v = nn.Linear(dim, self.attn_dim, bias=False)

    self.to_out = nn.Linear(self.attn_dim, dim, bias=False)
    self.attn_dropout = nn.Dropout(dropout)

    self.norm2 = nn.LayerNorm(dim, eps=norm_eps, elementwise_affine=False)
    self.ff = _FeedForward(dim, mult=4, dropout=dropout)

    self.scale_shift_table = nn.Parameter(torch.randn(1, 1, 6, dim) / dim**0.5)

  def _attn(self, x: torch.Tensor, terrain: torch.Tensor | None) -> torch.Tensor:
    """Cross-attention: Q from *x* (motion), K/V from *terrain* if given."""
    B, N, _ = x.shape
    h, d = self.num_heads, self.head_dim

    q = self.to_q(self.norm1(x)).reshape(B, N, h, d).transpose(1, 2)

    if terrain is not None:
      kv_src = self.norm_kv(terrain)
    else:
      kv_src = self.norm1(x)

    k = self.to_k(kv_src).reshape(B, N, h, d).transpose(1, 2)
    v = self.to_v(kv_src).reshape(B, N, h, d).transpose(1, 2)

    out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    out = out.transpose(1, 2).reshape(B, N, h * d)
    return self.attn_dropout(self.to_out(out))

  def forward(
    self,
    x: torch.Tensor,
    terrain: torch.Tensor | None,
    time_hidden_states: torch.Tensor,
  ) -> torch.Tensor:
    B = x.shape[0]
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
      self.scale_shift_table + time_hidden_states.reshape(B, 1, 6, -1)
    ).chunk(6, dim=-2)

    # adaLN modulation on Q pathway only
    h = self.norm1(x)
    h = h * (1 + scale_msa.squeeze(-2)) + shift_msa.squeeze(-2)
    x = x + gate_msa.squeeze(-2) * self._attn(h, terrain)

    h = self.norm2(x)
    h = h * (1 + scale_mlp.squeeze(-2)) + shift_mlp.squeeze(-2)
    x = x + gate_mlp.squeeze(-2) * self.ff(h)

    return x


class DiffusionDenoiser(nn.Module):
  """ε-prediction DiT for motion windows (optionally terrain-conditioned).

  Terrain conditioning uses **cross-attention**: terrain is projected to the
  model dimension and fed as K/V context to every DiT block.  Q always comes
  from the noisy motion stream.  When ``terrain_dim`` is None the model
  behaves as an unconditional self-attention DiT.

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
    self.terrain_dim = terrain_dim

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

    # Terrain projection: raw heights → inner_dim for K/V cross-attention
    if terrain_dim is not None:
      self.terrain_proj = nn.Sequential(
        nn.Linear(terrain_dim, self.inner_dim, bias=False),
        nn.LayerNorm(self.inner_dim, elementwise_affine=False),
        nn.SiLU(),
        nn.Linear(self.inner_dim, self.inner_dim, bias=False),
      )
    else:
      self.terrain_proj = None

    # Motion pathway (no terrain concat → input_dim == feature_dim)
    self.preprocess_conv = nn.Conv1d(feature_dim, feature_dim, 1, bias=False)
    self.proj_in = nn.Linear(feature_dim, self.inner_dim, bias=False)

    self.adaln_single = _AdaLayerNormSingle(self.inner_dim)
    self.sequence_pos_encoder = _SinusoidalPositionalEmbedding(
      self.inner_dim, max_seq_length=max(window_size, 32)
    )
    if terrain_dim is not None:
      self.terrain_pos_encoder = _SinusoidalPositionalEmbedding(
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
    # --- Motion pathway ---------------------------------------------------
    h = x_t  # (B, W, feature_dim)
    h = h.transpose(1, 2)  # (B, feature_dim, W)
    h = self.preprocess_conv(h) + h
    h = h.transpose(1, 2)  # (B, W, feature_dim)
    h = self.proj_in(h)  # (B, W, inner_dim)
    h = self.sequence_pos_encoder(h)

    # --- Terrain pathway --------------------------------------------------
    t_emb: torch.Tensor | None = None
    if terrain is not None and self.terrain_proj is not None:
      t_emb = self.terrain_proj(terrain)  # (B, W, inner_dim)
      t_emb = self.terrain_pos_encoder(t_emb)

    # --- DiT blocks (cross-attn only in the last block so early layers
    # preserve noise-driven diversity; the last layer injects terrain) ----
    time_hidden_states = self.adaln_single(t)
    for i, block in enumerate(self.blocks):
      t_block = t_emb if i == len(self.blocks) - 1 else None
      h = block(h, t_block, time_hidden_states)

    # --- Output projection ------------------------------------------------
    h = self.proj_out(h)  # (B, W, feature_dim)
    h = h.transpose(1, 2)  # (B, feature_dim, W)
    h = self.postprocess_conv(h) + h
    return h.transpose(1, 2)  # (B, W, feature_dim)
