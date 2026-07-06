"""Startup + reset events for SMP RL.

Run from mjlab's event manager so the task stays a plain ``ManagerBasedRlEnv``.
Motion features carry no absolute root pose, so reset writes a default root frame
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
  compile_model: bool = True,
  compile_mode: str | None = None,
  terrain_dim: int | None = None,
) -> None:
  """Startup-mode event: load the frozen denoiser, allocate the feature buffer +
  ``DiffNormalizer`` (stashed on the env), and warm the Inductor compile."""
  del env_ids
  if not ckpt_path:
    msg = (
      "init_smp_state called without `ckpt_path`. Set it on the EventTermCfg: "
      "EventTermCfg(func=init_smp_state, mode='startup', "
      "params={'ckpt_path': '/path/to/pretrained.pt'})."
    )
    raise RuntimeError(msg)
  model, scheduler, q_low, q_high, feature_dim, window_size, _td, t_q_low, t_q_high = (
    load_denoiser(ckpt_path, env.device, terrain_dim=terrain_dim)
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
  env._smp_buffer = MotionFeatureBuffer(  # type: ignore[attr-defined]
    num_envs=env.num_envs,
    window_size=window_size,
    num_joints=NUM_JOINTS,
    num_ee=NUM_EE,
    device=env.device,
    terrain_dim=_td,
  )
  env._smp_normalizer = DiffNormalizer(scheduler.num_timesteps, env.device)  # type: ignore[attr-defined]
  env._smp_buffer_needs_init = True  # type: ignore[attr-defined]

  if compile_model:
    # Warm the reward-path shape so its Inductor compile happens here.
    with torch.no_grad():
      dummy_x = torch.randn(env.num_envs, window_size, feature_dim, device=env.device)
      dummy_t = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
      _warm_terrain = None
      if _td is not None and t_q_low is not None and t_q_high is not None:
        _warm_terrain = ((t_q_low + t_q_high) / 2).expand(
          env.num_envs, window_size, _td
        )
      _ = model(dummy_x, dummy_t, terrain=_warm_terrain)


def reset_smp_buffer_flag(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None = None,
) -> None:
  """Reset-mode event: flag the buffer to be filled with current sim state."""
  env._smp_buffer_needs_init = True  # type: ignore[attr-defined]
