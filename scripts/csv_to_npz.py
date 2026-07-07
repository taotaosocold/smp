"""Convert CSV motion files to windowed NPZ files.

Each output NPZ contains a ``windows`` array of shape ``(N, window_size, F)``
with the per-frame layout (59 dims for G1):

  root_pos        (3)              xy in last-frame heading-inv frame
                                    relative to last root; z terrain-relative
                                    (height above terrain at pelvis xy)
  root_rot        (6)              6D tan-norm of heading_inv(T) ⊗ root_quat[t]
  joint_pos       (num_joints=29)  raw joint angles
  ee_pos          (num_ee*3=15)    end-effectors, per-frame root offset,
                                    last-frame heading-inv rotation
  root_lin_vel    (3)              last-frame heading-inv
  root_ang_vel    (3)              last-frame heading-inv

The anchor frame for every spatial quantity is the LAST window frame's
yaw-only local frame (origin at pelvis_T, x-axis = heading_T direction).

Usage:
  uv run scripts/csv_to_npz.py --input-dir datasets/csv --output-dir datasets/npz
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.scripts.csv_to_npz import MotionLoader as CsvMotionLoader
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply_inverse,
  quat_conjugate,
  quat_mul,
  yaw_quat,
)

from smp.utils import detect_device

# Joint name order matches the CSV column order — the 29 G1 joints.
JOINT_NAMES: tuple[str, ...] = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)

NUM_JOINTS = len(JOINT_NAMES)

# Tracked end-effector bodies. ``torso_link`` proxies the head (head is
# rigidly attached to the torso so the kinematic signal is the same).
# Order must match the online RL feature buffer.
EE_BODY_NAMES: tuple[str, ...] = (
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "torso_link",
  "left_wrist_yaw_link",
  "right_wrist_yaw_link",
)
NUM_EE = len(EE_BODY_NAMES)


@dataclass
class Cfg:
  input_dir: str = "datasets/csv"
  """Directory of input CSV motion files."""
  output_dir: str = "datasets/npz"
  """Directory to write output NPZ window files."""
  window_size: int = 10
  """Number of frames per window."""
  stride: int = 1
  """Stride between consecutive windows."""
  input_fps: int = 30
  """CSV frame rate."""
  output_fps: int = 50
  """Output (and sim) frame rate after interpolation."""
  device: str = ""
  """Compute device. Empty = auto (cuda if available else cpu)."""
  shard_index: int = 0
  """Index of this shard (for parallel runs). Files are sliced as [shard_index::num_shards]."""
  num_shards: int = 1
  """Total number of shards (for parallel runs)."""


def _setup_sim(device: str) -> tuple[Simulation, Scene]:
  """Build the G1 sim once."""
  sim_cfg = SimulationCfg()
  env_cfg = unitree_g1_flat_tracking_env_cfg()
  scene = Scene(env_cfg.scene, device=device)
  model = scene.compile()
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)
  return sim, scene


@torch.no_grad()
def _fk_motion(
  csv_path: Path,
  sim: Simulation,
  scene: Scene,
  joint_indexes: torch.Tensor,
  ee_indexes: torch.Tensor,
  input_fps: int,
  output_fps: int,
) -> tuple[
  torch.Tensor,  # base_pos
  torch.Tensor,  # base_quat
  torch.Tensor,  # base_lin_vel
  torch.Tensor,  # base_ang_vel
  torch.Tensor,  # ee_pos  (T, num_ee, 3)
  torch.Tensor,  # joint_pos  (T, num_joints)
  torch.Tensor,  # joint_vel  (T, num_joints)
]:
  """Replay a CSV through the sim, returning interpolated base state and
  FK'd world-frame positions for the end-effector bodies."""
  motion = CsvMotionLoader(
    motion_file=str(csv_path),
    input_fps=input_fps,
    output_fps=output_fps,
    device=sim.device,
  )
  robot: Entity = scene["robot"]

  ee_pos_list: list[torch.Tensor] = []

  scene.reset()
  for _ in range(motion.output_frames):
    state, _ = motion.get_next_state()
    base_pos, base_rot, base_lin_vel, base_ang_vel, dof_pos, dof_vel = state

    root_states = robot.data.default_root_state.clone()
    root_states[:, 0:3] = base_pos
    root_states[:, :2] += scene.env_origins[:, :2]
    root_states[:, 3:7] = base_rot
    root_states[:, 7:10] = base_lin_vel
    root_states[:, 10:] = base_ang_vel
    robot.write_root_state_to_sim(root_states)

    joint_pos_full = robot.data.default_joint_pos.clone()
    joint_vel_full = robot.data.default_joint_vel.clone()
    joint_pos_full[:, joint_indexes] = dof_pos
    joint_vel_full[:, joint_indexes] = dof_vel
    robot.write_joint_state_to_sim(joint_pos_full, joint_vel_full)

    sim.forward()
    scene.update(sim.mj_model.opt.timestep)

    ee_pos_list.append(robot.data.body_link_pos_w[0, ee_indexes].clone())

  return (
    motion.motion_base_poss,
    motion.motion_base_rots,
    motion.motion_base_lin_vels,
    motion.motion_base_ang_vels,
    torch.stack(ee_pos_list),
    motion.motion_dof_poss,
    motion.motion_dof_vels,
  )


def _tan_norm_from_quat(quat: torch.Tensor) -> torch.Tensor:
  """Convert quaternion (wxyz) to 6D tan-norm.

  Stacks the rotation matrix's first column (rotated x-axis) and third
  column (rotated z-axis).  This is the "tangent + normal" 6D
  representation, NOT Zhou-2019's "first two columns" form.  Input
  ``(..., 4)``, output ``(..., 6)`` as ``[col0_xyz, col2_xyz]``.
  """
  mat = matrix_from_quat(quat)
  col0 = mat[..., :, 0]
  col2 = mat[..., :, 2]
  return torch.cat([col0, col2], dim=-1)


def _compute_windows(
  base_pos: torch.Tensor,
  base_quat: torch.Tensor,
  base_lin_vel: torch.Tensor,
  base_ang_vel: torch.Tensor,
  ee_pos: torch.Tensor,
  joint_pos: torch.Tensor,
  window_size: int,
  stride: int,
) -> torch.Tensor | None:
  """Slice into windows and compute the per-frame motion features.

  All spatial quantities anchored to the LAST window frame's yaw-only local
  frame (origin at pelvis_T, heading = yaw_T).  Joint velocities are NOT
  part of the feature output.

  Returns ``(num_windows, window_size, 3+6+J+E*3+3+3)`` or ``None`` if the
  input is too short.
  """
  T = base_pos.shape[0]
  if T < window_size:
    return None

  E = ee_pos.shape[1]
  J = joint_pos.shape[1]
  starts = torch.arange(
    0, T - window_size + 1, stride, device=base_pos.device, dtype=torch.long
  )
  offsets = torch.arange(window_size, device=base_pos.device, dtype=torch.long)
  win_idx = starts[:, None] + offsets[None, :]
  N, W = win_idx.shape[0], window_size

  flat_idx = win_idx.reshape(-1)
  win_base_pos = base_pos.index_select(0, flat_idx).reshape(N, W, 3)
  win_base_quat = base_quat.index_select(0, flat_idx).reshape(N, W, 4)
  win_base_lin_vel = base_lin_vel.index_select(0, flat_idx).reshape(N, W, 3)
  win_base_ang_vel = base_ang_vel.index_select(0, flat_idx).reshape(N, W, 3)
  win_ee_pos = ee_pos.index_select(0, flat_idx).reshape(N, W, E, 3)
  win_joint = joint_pos.index_select(0, flat_idx).reshape(N, W, J)

  anchor_pos_T = win_base_pos[:, -1, :]
  anchor_quat_T = win_base_quat[:, -1, :]
  yaw_T = yaw_quat(anchor_quat_T)
  heading_inv_T_WF = quat_conjugate(yaw_T)[:, None, :].expand(N, W, 4).reshape(-1, 4)
  yaw_T_W = yaw_T[:, None, :].expand(N, W, 4).reshape(-1, 4)

  # root_pos: xy in heading-inv frame, z terrain-relative (height above terrain).
  root_offset = win_base_pos - anchor_pos_T[:, None, :]
  root_pos_local = quat_apply_inverse(yaw_T_W, root_offset.reshape(-1, 3)).reshape(
    N, W, 3
  )
  root_pos_local = root_pos_local.clone()
  root_pos_local[..., 2] = win_base_pos[..., 2]

  # root_rot: tan-norm of heading_inv(T) ⊗ root_quat[t].
  root_rot_local_quat = quat_mul(
    heading_inv_T_WF, win_base_quat.reshape(-1, 4)
  ).reshape(N, W, 4)
  root_rot_6d = _tan_norm_from_quat(root_rot_local_quat)

  # EE: (ee[t] - root[t]) rotated into the last-frame heading-inv frame.
  ee_offset_w = win_ee_pos - win_base_pos[:, :, None, :]
  yaw_T_E = yaw_T[:, None, None, :].expand(N, W, E, 4).reshape(-1, 4)
  ee_pos_local = quat_apply_inverse(yaw_T_E, ee_offset_w.reshape(-1, 3)).reshape(
    N, W, E * 3
  )

  lin_vel_local = quat_apply_inverse(yaw_T_W, win_base_lin_vel.reshape(-1, 3)).reshape(
    N, W, 3
  )
  ang_vel_local = quat_apply_inverse(yaw_T_W, win_base_ang_vel.reshape(-1, 3)).reshape(
    N, W, 3
  )

  return torch.cat(
    [
      root_pos_local,
      root_rot_6d,
      win_joint,
      ee_pos_local,
      lin_vel_local,
      ang_vel_local,
    ],
    dim=-1,
  )


def main(cfg: Cfg) -> None:
  if not cfg.device:
    cfg.device = detect_device()
  print(f"Device: {cfg.device}")

  in_dir = Path(cfg.input_dir)
  out_dir = Path(cfg.output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  csv_files = sorted(in_dir.glob("*.csv"))
  if not csv_files:
    msg = f"No CSV files found in {in_dir}"
    raise FileNotFoundError(msg)
  if cfg.num_shards > 1:
    csv_files = csv_files[cfg.shard_index :: cfg.num_shards]
    print(f"Shard {cfg.shard_index}/{cfg.num_shards}: {len(csv_files)} files")

  sim, scene = _setup_sim(cfg.device)
  robot: Entity = scene["robot"]
  joint_indexes = torch.tensor(
    robot.find_joints(list(JOINT_NAMES), preserve_order=True)[0],
    dtype=torch.long,
    device=sim.device,
  )
  ee_indexes = torch.tensor(
    robot.find_bodies(list(EE_BODY_NAMES), preserve_order=True)[0],
    dtype=torch.long,
    device=sim.device,
  )

  feature_dims = [3, 6, NUM_JOINTS, NUM_EE * 3, 3, 3]
  total_feature_dim = sum(feature_dims)

  print(f"Files: {len(csv_files)} in {in_dir}")
  print(f"Output: {out_dir}")
  print(f"Window: size={cfg.window_size} stride={cfg.stride} fps={cfg.output_fps}")
  print(f"End-effectors: {NUM_EE} {EE_BODY_NAMES} | Joints: {NUM_JOINTS}")
  print(
    f"Feature dim: {total_feature_dim} "
    f"(= 3 root_pos + 6 root_rot + {NUM_JOINTS} joint_pos + {NUM_EE * 3} "
    f"ee_pos + 3 lin_vel + 3 ang_vel)"
  )

  for i, csv_path in enumerate(csv_files):
    print(f"\n[{i + 1}/{len(csv_files)}] {csv_path.name}")
    (
      base_pos,
      base_quat,
      base_lin_vel,
      base_ang_vel,
      ee_pos,
      joint_pos,
      joint_vel,
    ) = _fk_motion(
      csv_path,
      sim,
      scene,
      joint_indexes,
      ee_indexes,
      input_fps=cfg.input_fps,
      output_fps=cfg.output_fps,
    )
    if joint_pos.shape[-1] != NUM_JOINTS:
      msg = (
        f"{csv_path.name}: expected {NUM_JOINTS} dof columns, got {joint_pos.shape[-1]}"
      )
      raise ValueError(msg)
    del joint_vel
    windows = _compute_windows(
      base_pos,
      base_quat,
      base_lin_vel,
      base_ang_vel,
      ee_pos,
      joint_pos,
      cfg.window_size,
      cfg.stride,
    )
    if windows is None:
      print(f"  [SKIP] too short for window_size={cfg.window_size}")
      continue

    out_path = out_dir / f"{csv_path.stem}.npz"
    np.savez_compressed(
      out_path,
      windows=windows.cpu().numpy().astype(np.float32),
      fps=np.array([cfg.output_fps], dtype=np.float32),
      window_size=np.array([cfg.window_size], dtype=np.int32),
      stride=np.array([cfg.stride], dtype=np.int32),
      ee_body_names=np.array(EE_BODY_NAMES),
      feature_dims=np.array(feature_dims, dtype=np.int32),
    )
    print(f"  saved {out_path.name}: windows={tuple(windows.shape)}")


if __name__ == "__main__":
  main(tyro.cli(Cfg))
