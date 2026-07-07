"""Utilities for SMP RL: denoiser loader, diff-normalizer, and feature buffer."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply_inverse,
  quat_conjugate,
  quat_mul,
  yaw_quat,
)

from smp.pretrain.model import DiffusionDenoiser
from smp.pretrain.scheduler import DDPMScheduler


def _quantile_normalize(
  x: torch.Tensor, q_low: torch.Tensor, q_high: torch.Tensor
) -> torch.Tensor:
  return 2.0 * (x - q_low) / (q_high - q_low + 1e-8) - 1.0


def load_denoiser(
  ckpt_path: str,
  device: torch.device | str,
  terrain_dim: int | None = None,
) -> tuple[
  DiffusionDenoiser,
  DDPMScheduler,
  torch.Tensor,
  torch.Tensor,
  int,
  int,
  int | None,
  torch.Tensor | None,
  torch.Tensor | None,
]:
  """Load a frozen pretrained denoiser checkpoint → ``(model, scheduler, q_low,
  q_high, feature_dim, window_size, terrain_dim, t_q_low, t_q_high)``.
  ``t_q_low``/``t_q_high`` are None for unconditional checkpoints."""
  device = torch.device(device)

  ckpt: dict[str, Any] = torch.load(ckpt_path, map_location=device, weights_only=False)
  cfg = ckpt["cfg"]
  feature_dim = int(cfg["feature_dim"])
  window_size = int(cfg["window_size"])

  _td = terrain_dim or cfg.get("terrain_dim")

  model = DiffusionDenoiser(
    feature_dim=feature_dim,
    window_size=window_size,
    d_model=int(cfg.get("d_model", 256)),
    nhead=int(cfg.get("nhead", 8)),
    num_layers=int(cfg.get("num_layers", 2)),
    dropout=float(cfg.get("dropout", 0.0)),
    terrain_dim=_td,
  ).to(device)
  state = ckpt.get("model_ema") or ckpt["model"]
  model.load_state_dict(state)
  model.eval()
  model.requires_grad_(False)

  scheduler = DDPMScheduler(
    num_timesteps=int(cfg.get("num_timesteps", 50)),
  ).to(device)

  q_low = torch.from_numpy(np.asarray(ckpt["q_low"], dtype=np.float32)).to(device)
  q_high = torch.from_numpy(np.asarray(ckpt["q_high"], dtype=np.float32)).to(device)

  t_q_low = ckpt.get("t_q_low")
  t_q_high = ckpt.get("t_q_high")
  if t_q_low is not None:
    t_q_low = torch.from_numpy(np.asarray(t_q_low, dtype=np.float32)).to(device)
  if t_q_high is not None:
    t_q_high = torch.from_numpy(np.asarray(t_q_high, dtype=np.float32)).to(device)

  return model, scheduler, q_low, q_high, feature_dim, window_size, _td, t_q_low, t_q_high


class DiffNormalizer:
  """Count-based running mean per diffusion timestep: equal-weighted, so it
  freezes as the count grows — a stable SDS-MSE reference, unlike a drifting EMA."""

  def __init__(
    self,
    num_timesteps: int,
    device: torch.device,
    min_value: float = 1e-4,
    max_count: int = 50_000_000,
  ) -> None:
    self.min_value = min_value
    self.max_count = max_count
    self.mean = torch.ones(num_timesteps, device=device)
    self.count = torch.zeros(num_timesteps, device=device, dtype=torch.long)

  def update_and_normalize(self, t: int, mse_per_env: torch.Tensor) -> torch.Tensor:
    """Record MSE values for timestep ``t``; return them divided by the mean."""
    if self.count[t] > self.max_count:
      # Freeze once stable (and avoid count overflow).
      return mse_per_env / self.mean[t].clamp(min=self.min_value)
    n = mse_per_env.numel()
    batch_mean = mse_per_env.mean()
    old_count = self.count[t].item()
    new_count = old_count + n
    if old_count == 0:
      self.mean[t] = batch_mean
    else:
      w_old = old_count / new_count
      w_new = n / new_count
      self.mean[t] = w_old * self.mean[t] + w_new * batch_mean
    self.count[t] = new_count
    return mse_per_env / self.mean[t].clamp(min=self.min_value)


class MotionFeatureBuffer:
  """Rolling per-env buffer of the last ``window_size`` kinematic samples;
  ``compute_features()`` returns a window anchored at the LAST frame's yaw-only
  local frame, layout (matching ``scripts/csv_to_npz.py``):

      ``[root_pos(3), root_rot(6), joint_pos(J), ee_pos(E*3),
         root_lin_vel(3), root_ang_vel(3)]``

  ``joint_vel`` is stored for symmetry but excluded from the output.  Positions
  use the caller's frame (SMP RL feeds env-origin-relative xy + terrain-relative z)."""

  def __init__(
    self,
    num_envs: int,
    window_size: int,
    num_joints: int,
    num_ee: int,
    device: torch.device | str,
    terrain_dim: int | None = None,
  ) -> None:
    self.num_envs = num_envs
    self.window_size = window_size
    self.num_joints = num_joints
    self.num_ee = num_ee
    self.device = torch.device(device)
    self.terrain_dim = terrain_dim

    self.root_pos_w = torch.zeros(num_envs, window_size, 3, device=self.device)
    self.root_quat_w = torch.zeros(num_envs, window_size, 4, device=self.device)
    self.root_quat_w[..., 0] = 1.0
    self.root_lin_vel_w = torch.zeros(num_envs, window_size, 3, device=self.device)
    self.root_ang_vel_w = torch.zeros(num_envs, window_size, 3, device=self.device)
    self.ee_pos_w = torch.zeros(num_envs, window_size, num_ee, 3, device=self.device)
    self.joint_pos = torch.zeros(num_envs, window_size, num_joints, device=self.device)
    self.joint_vel = torch.zeros(num_envs, window_size, num_joints, device=self.device)

    if terrain_dim is not None:
      self.terrain = torch.zeros(num_envs, window_size, terrain_dim, device=self.device)
    else:
      self.terrain = None

  def reset(
    self,
    env_ids: torch.Tensor,
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    root_lin_vel_w: torch.Tensor,
    root_ang_vel_w: torch.Tensor,
    ee_pos_w: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    terrain: torch.Tensor | None = None,
  ) -> None:
    """Fill all W slots of ``env_ids`` with a pre-sampled trajectory."""
    if env_ids.numel() == 0:
      return
    self.root_pos_w[env_ids] = root_pos_w
    self.root_quat_w[env_ids] = root_quat_w
    self.root_lin_vel_w[env_ids] = root_lin_vel_w
    self.root_ang_vel_w[env_ids] = root_ang_vel_w
    self.ee_pos_w[env_ids] = ee_pos_w
    self.joint_pos[env_ids] = joint_pos
    self.joint_vel[env_ids] = joint_vel
    if terrain is not None and self.terrain is not None:
      self.terrain[env_ids] = terrain

  def update(
    self,
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    root_lin_vel_w: torch.Tensor,
    root_ang_vel_w: torch.Tensor,
    ee_pos_w: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    terrain: torch.Tensor | None = None,
  ) -> None:
    """Shift left by one and append the new frame at index W-1."""
    self.root_pos_w = torch.roll(self.root_pos_w, shifts=-1, dims=1)
    self.root_quat_w = torch.roll(self.root_quat_w, shifts=-1, dims=1)
    self.root_lin_vel_w = torch.roll(self.root_lin_vel_w, shifts=-1, dims=1)
    self.root_ang_vel_w = torch.roll(self.root_ang_vel_w, shifts=-1, dims=1)
    self.ee_pos_w = torch.roll(self.ee_pos_w, shifts=-1, dims=1)
    self.joint_pos = torch.roll(self.joint_pos, shifts=-1, dims=1)
    self.joint_vel = torch.roll(self.joint_vel, shifts=-1, dims=1)
    self.root_pos_w[:, -1] = root_pos_w
    self.root_quat_w[:, -1] = root_quat_w
    self.root_lin_vel_w[:, -1] = root_lin_vel_w
    self.root_ang_vel_w[:, -1] = root_ang_vel_w
    self.ee_pos_w[:, -1] = ee_pos_w
    self.joint_pos[:, -1] = joint_pos
    self.joint_vel[:, -1] = joint_vel
    if terrain is not None and self.terrain is not None:
      self.terrain = torch.roll(self.terrain, shifts=-1, dims=1)
      self.terrain[:, -1] = terrain

  def get_terrain(self) -> torch.Tensor | None:
    """Return the current terrain window ``(num_envs, window_size, terrain_dim)``,
    or None if terrain conditioning is disabled."""
    return self.terrain

  def compute_features(self) -> torch.Tensor:
    """Return features ``(num_envs, W, 3+6+J+E*3+3+3)``, all anchored to the LAST
    frame's yaw-only local frame.  root_pos z is terrain-relative (height above
    terrain at pelvis xy)."""
    N = self.num_envs
    W = self.window_size
    E = self.num_ee

    anchor_pos_T = self.root_pos_w[:, -1]
    anchor_quat_T = self.root_quat_w[:, -1]
    yaw_T = yaw_quat(anchor_quat_T)
    heading_inv_T = quat_conjugate(yaw_T)
    heading_inv_T_W = heading_inv_T[:, None, :].expand(N, W, 4)
    yaw_T_W = yaw_T[:, None, :].expand(N, W, 4).reshape(-1, 4)

    root_offset = self.root_pos_w - anchor_pos_T[:, None, :]
    # xy heading-inv relative to anchor, z stays terrain-relative (height above terrain).
    root_pos_local = quat_apply_inverse(yaw_T_W, root_offset.reshape(-1, 3)).reshape(
      N, W, 3
    )
    root_pos_local = root_pos_local.clone()
    root_pos_local[..., 2] = self.root_pos_w[..., 2]

    # 6D rot is stacked [col0, col2] = [rotated-x-axis, rotated-z-axis].
    root_rot_local_quat = quat_mul(
      heading_inv_T_W.reshape(-1, 4),
      self.root_quat_w.reshape(-1, 4),
    ).reshape(N, W, 4)
    root_rot_mat = matrix_from_quat(root_rot_local_quat.reshape(-1, 4)).reshape(
      N, W, 3, 3
    )
    root_rot_6d = torch.cat([root_rot_mat[..., :, 0], root_rot_mat[..., :, 2]], dim=-1)

    ee_offset_w = self.ee_pos_w - self.root_pos_w[:, :, None, :]
    yaw_T_E = yaw_T[:, None, None, :].expand(N, W, E, 4).reshape(-1, 4)
    ee_pos_local = quat_apply_inverse(yaw_T_E, ee_offset_w.reshape(-1, 3)).reshape(
      N, W, E * 3
    )

    lin_vel_local = quat_apply_inverse(
      yaw_T_W, self.root_lin_vel_w.reshape(-1, 3)
    ).reshape(N, W, 3)
    ang_vel_local = quat_apply_inverse(
      yaw_T_W, self.root_ang_vel_w.reshape(-1, 3)
    ).reshape(N, W, 3)

    return torch.cat(
      [
        root_pos_local,
        root_rot_6d,
        self.joint_pos,
        ee_pos_local,
        lin_vel_local,
        ang_vel_local,
      ],
      dim=-1,
    )
