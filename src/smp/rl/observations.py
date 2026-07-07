"""Observation functions for SMP RL tasks (terrain, privileged info, etc.)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def terrain_cnn_features(
  env: ManagerBasedRlEnv,
  sensor_name: str = "terrain_scan",
  height_scale: float = 0.5,
) -> torch.Tensor:
  """CNN-encoded terrain features ``(num_envs, 32)`` from the 17×11 height map.

  Reads the terrain scan sensor, normalises heights relative to the pelvis,
  and runs a frozen ``TerrainEncoder`` (lazily created on first call so it
  works before ``init_smp_state`` runs).
  """
  from smp.rl.nn.terrain_encoder import build_terrain_encoder, normalise_height_map

  if not hasattr(env, "_smp_terrain_encoder"):
    env._smp_terrain_encoder = build_terrain_encoder(env.device)  # type: ignore[attr-defined]

  sensor = env.scene.sensors[sensor_name]
  heights = sensor.data.hit_pos_w[..., 2]  # (num_envs, 187)
  normalised = normalise_height_map(heights, height_scale=height_scale)
  encoder = env._smp_terrain_encoder  # type: ignore[attr-defined]
  with torch.no_grad():
    return encoder(normalised)
