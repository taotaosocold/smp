"""Lightweight CNN encoder for 17×11 terrain height maps.

Compresses 187 raw height samples into a compact feature vector that
preserves spatial structure (slopes, edges, steps) better than an MLP
on flattened heights.  Designed to run every sim step across thousands
of parallel envs.
"""

from __future__ import annotations

import torch
import torch.nn as nn

GRID_X = 17
GRID_Y = 11
CENTER_IDX = GRID_X // 2 * GRID_Y + GRID_Y // 2  # 93


class TerrainEncoder(nn.Module):
  """Small CNN: 17×11 grid → 32-dim terrain feature vector.

  Input is normalised robot-relative heights (each frame's height minus
  the terrain z at the pelvis), so the CNN sees local topography rather
  than absolute elevation.
  """

  def __init__(self, out_dim: int = 32) -> None:
    super().__init__()
    self.out_dim = out_dim

    self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
    self.conv3 = nn.Conv2d(16, 16, kernel_size=3, padding=1)

    self.pool = nn.AdaptiveAvgPool2d((3, 2))  # → (16, 3, 2) = 96
    self.fc = nn.Linear(16 * 3 * 2, out_dim)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """*x*: ``(N, 1, 17, 11)`` normalised height map → ``(N, out_dim)``."""
    h = torch.relu(self.conv1(x))
    h = torch.relu(self.conv2(h))
    h = torch.relu(self.conv3(h))
    h = self.pool(h)
    h = h.reshape(h.shape[0], -1)
    return torch.relu(self.fc(h))


def build_terrain_encoder(
  device: torch.device | str,
  out_dim: int = 32,
) -> TerrainEncoder:
  """Create a frozen terrain encoder (random weights, spatial inductive bias).

  The encoder is frozen so it acts as a deterministic feature extractor.
  Random-weight CNNs still provide useful spatial features due to the
  locality + translation-invariance priors of convolution.
  """
  encoder = TerrainEncoder(out_dim=out_dim).to(device)
  encoder.eval()
  encoder.requires_grad_(False)
  return encoder


def normalise_height_map(
  heights: torch.Tensor,
  height_scale: float = 0.5,
) -> torch.Tensor:
  """Normalise raw terrain heights ``(N, 187)`` → ``(N, 1, 17, 11)``.

  Centers on pelvis (pixel 93) and divides by ``height_scale`` so values
  are roughly in [-2, 2] for typical terrain variation.
  """
  centre = heights[:, CENTER_IDX:CENTER_IDX + 1]  # (N, 1)
  heights_rel = heights - centre
  heights_rel = heights_rel / height_scale
  return heights_rel.reshape(-1, 1, GRID_X, GRID_Y)
