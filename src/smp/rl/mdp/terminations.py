"""SMP-specific termination functions (adapted from instinctlab parkour)."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def terrain_out_of_bounds(
    env: ManagerBasedRlEnv,
    distance_buffer: float = 3.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Terminate when the robot moves too close to the terrain edge."""
    terrain_cfg = env.cfg.scene.terrain
    if terrain_cfg.terrain_type == "plane":
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    gen_cfg = terrain_cfg.terrain_generator
    grid_width, grid_length = gen_cfg.size
    n_rows, n_cols = gen_cfg.num_rows, gen_cfg.num_cols
    border_width = gen_cfg.border_width
    map_width = n_rows * grid_width + 2 * border_width
    map_height = n_cols * grid_length + 2 * border_width

    asset = env.scene[asset_cfg.name]
    x_out = torch.abs(asset.data.root_link_pos_w[:, 0]) > 0.5 * map_width - distance_buffer
    y_out = torch.abs(asset.data.root_link_pos_w[:, 1]) > 0.5 * map_height - distance_buffer
    return torch.logical_or(x_out, y_out)


def root_height_below_env_origin_minimum(
    env: ManagerBasedRlEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Terminate when the root height above the env-origin terrain is too low."""
    asset = env.scene[asset_cfg.name]
    terrain_base = torch.clamp(env.scene.env_origins[:, 2], max=0.0)
    return asset.data.root_link_pos_w[:, 2] - terrain_base < minimum_height


def base_contact(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Terminate when the torso link experiences contact force above threshold."""
    sensor: ContactSensor = env.scene.sensors[sensor_name]
    data = sensor.data
    if data.force_history is not None:
        force_mag = torch.norm(data.force_history, dim=-1)
        return (force_mag > threshold).any(dim=-1).any(dim=-1)
    if data.found is not None:
        return torch.any(data.found, dim=-1)
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
