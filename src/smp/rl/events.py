"""Startup + reset events for SMP RL.

Run from mjlab's event manager so the task stays a plain ``ManagerBasedRlEnv``.
Motion features carry no absolute root pose, so GSI writes a default root frame
(each env's origin, identity yaw) to sim and primes the feature buffer in an
env-origin-relative frame, so the SMP reward is invariant to env placement.
"""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import quat_apply, quat_mul, yaw_quat

from smp.rl.utils import DiffNormalizer, MotionFeatureBuffer, load_denoiser
from smp.sampling.feature_to_state import (
  EE_BODY_NAMES,
  NUM_EE,
  rot6d_to_quat,
  slice_features,
)

NUM_JOINTS = 29


def _maybe_compile(model, compile_model: bool, compile_mode: str | None):
  """``torch.compile`` ``model`` (no-op if ``compile_model`` false), working
  around the Inductor ``pad_mm`` TF32 crash by disabling shape padding."""
  if not compile_model:
    return model
  torch.set_float32_matmul_precision("high")
  try:
    import torch._inductor.config as _ic

    _ic.shape_padding = False
  except ImportError:
    pass
  if compile_mode is not None:
    return torch.compile(model, fullgraph=True, mode=compile_mode)
  return torch.compile(model, fullgraph=True)


def init_smp_state(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None = None,
  ckpt_path: str = "",
  gsi_buffer_size: int = 4096,
  gsi_batch_size: int = 256,
  compile_model: bool = True,
  compile_mode: str | None = None,
) -> None:
  """Startup-mode event: load the frozen denoiser, allocate the feature buffer +
  ``DiffNormalizer`` (stashed on the env), and pre-generate the GSI pool of
  ``gsi_buffer_size`` windows that ``gsi_reset`` samples from (amortizes the DDPM
  cost).  If ``compile_model``, the denoiser is ``torch.compile``-d and pre-warmed
  so Inductor compiles here, not on the first sim step."""
  del env_ids
  if not ckpt_path:
    msg = (
      "init_smp_state called without `ckpt_path`. Set it on the EventTermCfg: "
      "EventTermCfg(func=init_smp_state, mode='startup', "
      "params={'ckpt_path': '/path/to/pretrained.pt'})."
    )
    raise RuntimeError(msg)
  model, scheduler, q_low, q_high, feature_dim, window_size, t_q_low, t_q_high = (
    load_denoiser(ckpt_path, env.device)
  )
  model = _maybe_compile(model, compile_model, compile_mode)
  env._smp_bundle = (  # type: ignore[attr-defined]
    model,
    scheduler,
    q_low,
    q_high,
    feature_dim,
    window_size,
    t_q_low,
    t_q_high,
  )
  robot = env.scene["robot"]
  env._smp_ee_indexes = torch.tensor(  # type: ignore[attr-defined]
    robot.find_bodies(list(EE_BODY_NAMES), preserve_order=True)[0],
    dtype=torch.long,
    device=env.device,
  )
  terrain_dim = model.terrain_dim
  env._smp_buffer = MotionFeatureBuffer(  # type: ignore[attr-defined]
    num_envs=env.num_envs,
    window_size=window_size,
    num_joints=NUM_JOINTS,
    num_ee=NUM_EE,
    device=env.device,
    terrain_dim=terrain_dim,
  )
  env._smp_normalizer = DiffNormalizer(scheduler.num_timesteps, env.device)  # type: ignore[attr-defined]

  neutral_terrain: torch.Tensor | None = None
  if terrain_dim is not None and t_q_low is not None and t_q_high is not None:
    neutral_terrain = (
      ((t_q_low + t_q_high) / 2.0)
      .expand(1, window_size, terrain_dim)
      .to(env.device)
    )

  if gsi_buffer_size <= 0:
    msg = f"gsi_buffer_size must be positive, got {gsi_buffer_size}."
    raise ValueError(msg)
  pool_chunks: list[torch.Tensor] = []
  for start in range(0, gsi_buffer_size, gsi_batch_size):
    bsz = min(gsi_batch_size, gsi_buffer_size - start)
    terrain_batch = neutral_terrain.expand(bsz, -1, -1) if neutral_terrain is not None else None
    pool_chunks.append(_ddpm_sample(env, bsz, terrain=terrain_batch))
  env._smp_gsi_pool = torch.cat(pool_chunks, dim=0)  # type: ignore[attr-defined]

  if compile_model and env.num_envs != gsi_batch_size:
    # Warm the reward-path shape so its Inductor compile happens here.
    with torch.no_grad():
      dummy_x = torch.randn(env.num_envs, window_size, feature_dim, device=env.device)
      dummy_t = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
      _ = model(dummy_x, dummy_t)

  gsi_reset(env)


def _prime_sim_and_buffer(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  window: torch.Tensor,
  terrain: torch.Tensor | None = None,
) -> None:
  """Common GSI tail: write the window's last frame to sim, fill the feature
  buffer.  The buffer is env-origin-RELATIVE (placement-invariant features) while
  the sim write adds each env's origin so robots spread across the grid.
  ``joint_vel`` is finite-differenced from ``joint_pos`` (not in the window)."""
  n, W, _ = window.shape
  E = NUM_EE
  parts = slice_features(window)
  root_pos_local = parts["root_pos"]
  root_rot_6d = parts["root_rot"]
  joint_pos = parts["joint_pos"]
  ee_pos_local = parts["ee_pos"].reshape(n, W, E, 3)
  root_lin_vel_local = parts["root_lin_vel"]
  root_ang_vel_local = parts["root_ang_vel"]

  control_dt = float(env.cfg.sim.mujoco.timestep) * float(env.cfg.decimation)
  if W > 1:
    joint_vel = torch.zeros_like(joint_pos)
    joint_vel[:, :-1] = (joint_pos[:, 1:] - joint_pos[:, :-1]) / control_dt
    joint_vel[:, -1] = joint_vel[:, -2]
  else:
    joint_vel = torch.zeros_like(joint_pos)

  robot = env.scene["robot"]
  default_root = robot.data.default_root_state[env_ids].clone()
  default_pos = default_root[:, 0:3]
  default_quat = default_root[:, 3:7]
  yaw_T = yaw_quat(default_quat)
  yaw_T_W = yaw_T[:, None, :].expand(n, W, 4).reshape(-1, 4)

  local_xy = root_pos_local.clone()
  local_xy[..., 2] = 0.0
  world_offset_xy = quat_apply(yaw_T_W, local_xy.reshape(-1, 3)).reshape(n, W, 3)
  pelvis_pos_w = world_offset_xy.clone()
  pelvis_pos_w[..., 0] += default_pos[:, None, 0]
  pelvis_pos_w[..., 1] += default_pos[:, None, 1]
  pelvis_pos_w[..., 2] = root_pos_local[..., 2]

  root_rot_local_quat = rot6d_to_quat(root_rot_6d.reshape(-1, 6)).reshape(n, W, 4)
  pelvis_quat_w = quat_mul(yaw_T_W, root_rot_local_quat.reshape(-1, 4)).reshape(n, W, 4)

  lin_vel_w = quat_apply(yaw_T_W, root_lin_vel_local.reshape(-1, 3)).reshape(n, W, 3)
  ang_vel_w = quat_apply(yaw_T_W, root_ang_vel_local.reshape(-1, 3)).reshape(n, W, 3)

  yaw_T_E = yaw_T[:, None, None, :].expand(n, W, E, 4).reshape(-1, 4)
  ee_offset_w = quat_apply(yaw_T_E, ee_pos_local.reshape(-1, 3)).reshape(n, W, E, 3)
  ee_pos_w = ee_offset_w + pelvis_pos_w[:, :, None, :]

  # Buffer stays env-relative; the sim write is offset to each env's origin.
  origins = env.scene.env_origins[env_ids]
  last_root_state = torch.cat(
    [
      pelvis_pos_w[:, -1] + origins,
      pelvis_quat_w[:, -1],
      lin_vel_w[:, -1],
      ang_vel_w[:, -1],
    ],
    dim=-1,
  )
  robot.write_root_state_to_sim(last_root_state, env_ids=env_ids)
  robot.write_joint_state_to_sim(joint_pos[:, -1], joint_vel[:, -1], env_ids=env_ids)

  buf: MotionFeatureBuffer = env._smp_buffer  # type: ignore[attr-defined]
  buf.reset(
    env_ids,
    pelvis_pos_w,
    pelvis_quat_w,
    lin_vel_w,
    ang_vel_w,
    ee_pos_w,
    joint_pos,
    joint_vel,
    terrain=terrain,
  )


@torch.no_grad()
def _ddpm_sample(
  env: ManagerBasedRlEnv, n: int, terrain: torch.Tensor | None = None
) -> torch.Tensor:
  """Run DDPM ancestral sampling and return ``n`` denormalized windows.
  If ``terrain`` (B, W, terrain_dim) is given, cross-attention is used."""
  (
    model,
    scheduler,
    q_low,
    q_high,
    feature_dim,
    window_size,
    t_q_low,
    t_q_high,
  ) = env._smp_bundle  # type: ignore[attr-defined]
  x_t = torch.randn(n, window_size, feature_dim, device=env.device)
  terrain_norm: torch.Tensor | None = None
  if terrain is not None and t_q_low is not None and t_q_high is not None:
    terrain_norm = 2.0 * (terrain - t_q_low) / (t_q_high - t_q_low + 1e-8) - 1.0
  for t_int in reversed(range(scheduler.num_timesteps)):
    t = torch.full((n,), t_int, dtype=torch.long, device=env.device)
    eps = model(x_t, t, terrain=terrain_norm)
    x_t = scheduler.step(eps, x_t, t_int)
  return (x_t + 1.0) / 2.0 * (q_high - q_low) + q_low


@torch.no_grad()
def gsi_refresh(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None = None,
  num_samples: int = 1024,
  step_interval: int = 2400,
) -> None:
  """Step-mode event: every ``step_interval`` steps, FIFO-replace ``num_samples``
  GSI-pool windows with fresh DDPM samples so the init distribution stays fresh."""
  del env_ids
  cur = int(env.common_step_counter)
  if cur == 0 or (cur % step_interval) != 0:
    return

  pool: torch.Tensor = env._smp_gsi_pool  # type: ignore[attr-defined]
  pool_size = pool.shape[0]
  if num_samples > pool_size:
    msg = f"num_samples ({num_samples}) cannot exceed pool size ({pool_size})"
    raise ValueError(msg)

  terrain: torch.Tensor | None = None
  terrain_sensor = env.scene.sensors.get("terrain_scan")
  _, _, _, _, _, window_size, _, t_q_high = env._smp_bundle  # type: ignore[attr-defined]
  if terrain_sensor is not None and t_q_high is not None:
    heights = terrain_sensor.data.hit_pos_w[0, :, 2]  # (187,) world-frame z
    terrain = heights[None, None, :].expand(num_samples, window_size, -1)

  new_windows = _ddpm_sample(env, num_samples, terrain=terrain)
  head = int(getattr(env, "_smp_gsi_head", 0))
  end = head + num_samples
  if end <= pool_size:
    pool[head:end] = new_windows
  else:
    first = pool_size - head
    pool[head:] = new_windows[:first]
    pool[: end - pool_size] = new_windows[first:]
  env._smp_gsi_head = end % pool_size  # type: ignore[attr-defined]


@torch.no_grad()
def gsi_reset(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None = None) -> None:
  """Generative State Initialization: sample ``n`` windows from the GSI pool and
  prime sim + feature buffer from them.  Must run AFTER mjlab's ``reset_base``.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  n = int(env_ids.numel())
  if n == 0:
    return

  # Terrain-conditioned DDPM for training resets (cross-attention). At startup
  # the sensor may not have valid data yet, so fall back to the pool.
  terrain_sensor = env.scene.sensors.get("terrain_scan")
  _, _, _, _, _, window_size, _, t_q_high = env._smp_bundle  # type: ignore[attr-defined]
  terrain: torch.Tensor | None = None
  if terrain_sensor is not None and t_q_high is not None and int(env.common_step_counter) > 0:
    heights = terrain_sensor.data.hit_pos_w[env_ids, :, 2]  # (n, 187)
    terrain = heights[:, None, :].expand(n, window_size, -1)
    window = _ddpm_sample(env, n, terrain=terrain)
  else:
    pool: torch.Tensor = env._smp_gsi_pool  # type: ignore[attr-defined]
    idx = torch.randint(0, pool.shape[0], (n,), device=env.device)
    window = pool[idx]
  _prime_sim_and_buffer(env, env_ids, window, terrain=terrain)
