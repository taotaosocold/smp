"""Reward functions for SMP RL tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from smp.rl.utils import DiffNormalizer, MotionFeatureBuffer, _quantile_normalize

if TYPE_CHECKING:
  from collections.abc import Callable

  from mjlab.envs import ManagerBasedRlEnv

  TaskTerm = tuple["Callable[..., torch.Tensor]", float, dict]


def _update_buffer_from_sim(env: ManagerBasedRlEnv) -> None:
  """Push current sim kinematics onto the buffer, env-origin-relative
  (matching the feature frame) so features are placement-invariant.
  On first call after a non-GSI reset, fills all W slots with the current
  frame so ``compute_features()`` has a valid window from step one."""
  robot = env.scene["robot"]
  ee_indexes = env._smp_ee_indexes  # type: ignore[attr-defined]
  buffer: MotionFeatureBuffer = env._smp_buffer  # type: ignore[attr-defined]
  origins = env.scene.env_origins

  root_pos = robot.data.root_link_pos_w - origins
  root_quat = robot.data.root_link_quat_w
  root_lin_vel = robot.data.root_link_lin_vel_w
  root_ang_vel = robot.data.root_link_ang_vel_w
  ee_pos = robot.data.body_link_pos_w[:, ee_indexes] - origins[:, None, :]
  joint_pos = robot.data.joint_pos
  joint_vel = robot.data.joint_vel

  terrain = None
  terrain_sensor = env.scene.sensors.get("terrain_scan")
  if terrain_sensor is not None and buffer.terrain is not None:
    terrain = terrain_sensor.data.hit_pos_w[..., 2]  # (num_envs, 187)

  # Make root_pos z terrain-relative (height above terrain at pelvis xy).
  # Grid is 17×11, pelvis-centered → center index (8, 5) = 93 is directly below pelvis.
  if terrain is not None:
    terrain_height = terrain[:, 93]  # (num_envs,)
    root_pos = root_pos.clone()
    root_pos[..., 2] = robot.data.root_link_pos_w[..., 2] - terrain_height

  if getattr(env, "_smp_buffer_needs_init", False):
    # No GSI reset — fill all W slots with the current sim state so the
    # first SMP reward computation sees a consistent (repeated) window.
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    W = buffer.window_size
    buffer.reset(
      env_ids,
      root_pos[:, None, :].expand(-1, W, 3),
      root_quat[:, None, :].expand(-1, W, 4),
      root_lin_vel[:, None, :].expand(-1, W, 3),
      root_ang_vel[:, None, :].expand(-1, W, 3),
      ee_pos[:, None, :, :].expand(-1, W, buffer.num_ee, 3),
      joint_pos[:, None, :].expand(-1, W, buffer.num_joints),
      joint_vel[:, None, :].expand(-1, W, buffer.num_joints),
      terrain=terrain[:, None, :].expand(-1, W, terrain.shape[-1])
      if terrain is not None
      else None,
    )
    env._smp_buffer_needs_init = False  # type: ignore[attr-defined]
    return

  buffer.update(
    root_pos,
    root_quat,
    root_lin_vel,
    root_ang_vel,
    ee_pos,
    joint_pos,
    joint_vel,
    terrain=terrain,
  )


def smp_guidance_reward(
  env: ManagerBasedRlEnv,
  fixed_timesteps: tuple[int, ...] = (8, 15, 22),
  ws: float = 4.0,
  normalize: bool = True,
) -> torch.Tensor:
  """SDS-style guidance reward over fixed timesteps ``K``:
  ``exp(-w_s/|K| · Σ_{i∈K} ‖ε̂_i − ε_i‖²)``.  ``normalize`` divides each MSE by a
  ``DiffNormalizer`` running mean (policy-relative) vs. raw (absolute scale);
  always stashes the mean raw MSE on ``env._smp_raw_err``."""
  device = torch.device(env.device)
  model, scheduler, q_low, q_high, _, _, t_q_low, t_q_high = env._smp_bundle  # type: ignore[attr-defined]
  normalizer: DiffNormalizer = env._smp_normalizer  # type: ignore[attr-defined]
  buffer: MotionFeatureBuffer = env._smp_buffer  # type: ignore[attr-defined]
  _update_buffer_from_sim(env)

  features = buffer.compute_features()
  x_0 = 2.0 * (features - q_low) / (q_high - q_low + 1e-8) - 1.0
  num_envs = x_0.shape[0]

  terrain = None
  terrain_raw = buffer.get_terrain()
  if terrain_raw is not None and t_q_low is not None and t_q_high is not None:
    terrain = _quantile_normalize(terrain_raw, t_q_low, t_q_high)

  total_err = torch.zeros(num_envs, device=device)
  total_raw = torch.zeros(num_envs, device=device)
  with torch.no_grad():
    for t_scalar in fixed_timesteps:
      if not 0 <= t_scalar < scheduler.num_timesteps:
        msg = f"fixed_timestep {t_scalar} out of range [0, {scheduler.num_timesteps})"
        raise ValueError(msg)
      t = torch.full((num_envs,), t_scalar, dtype=torch.long, device=device)
      noise = torch.randn_like(x_0)
      x_t = scheduler.add_noise(x_0, noise, t)
      eps_hat = model(x_t, t, terrain=terrain)
      mse_per_env = ((eps_hat - noise) ** 2).mean(dim=(-1, -2))
      total_raw += mse_per_env
      if normalize:
        total_err += normalizer.update_and_normalize(t_scalar, mse_per_env)
      else:
        total_err += mse_per_env

  env._smp_raw_err = total_raw / len(fixed_timesteps)  # type: ignore[attr-defined]
  err = total_err / len(fixed_timesteps)
  return torch.exp(-err * ws)


def task_smp_product(
  env: ManagerBasedRlEnv,
  task_terms: tuple[TaskTerm, ...],
  fixed_timesteps: tuple[int, ...] = (8, 15, 22),
  ws: float = 6.0,
) -> torch.Tensor:
  """``(Σ wᵢ · taskᵢ(env)) · r_smp`` — multiplicative SMP gating; ``task_terms`` is
  a tuple of ``(func, weight, kwargs)``.  Calls ``smp_guidance_reward`` once (the
  sole SMP-buffer update), so it must be the task's only SMP reward term."""
  task = sum(w * func(env, **kw) for func, w, kw in task_terms)
  return task * smp_guidance_reward(env, fixed_timesteps=fixed_timesteps, ws=ws)
