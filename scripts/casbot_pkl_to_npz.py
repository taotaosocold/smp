"""Convert CasBot pkl motion files to windowed NPZ files for SMP pretraining.

Adapted from csv_to_npz.py: same windowing + feature layout, but reads pkl
(pickle dict) instead of CSV and uses CasBot 25-DOF kinematics instead of G1.

Each output NPZ contains a ``windows`` array of shape ``(N, window_size, 55)``
with the per-frame layout:

  root_pos        (3)              xy in last-frame heading-inv frame
                                    relative to last root; z terrain-relative
  root_rot        (6)              6D tan-norm of heading_inv(T) ⊗ root_quat[t]
  joint_pos       (num_joints=25)  raw joint angles
  ee_pos          (num_ee*3=15)    end-effectors, per-frame root offset,
                                    last-frame heading-inv rotation
  root_lin_vel    (3)              last-frame heading-inv
  root_ang_vel    (3)              last-frame heading-inv

Usage:
  python scripts/casbot_pkl_to_npz.py --input-dir datasets/pkl --output-dir datasets/npz
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch
import tyro

from mjlab.utils.lab_api.math import (
  axis_angle_from_quat,
  matrix_from_quat,
  quat_apply_inverse,
  quat_conjugate,
  quat_mul,
  quat_slerp,
  yaw_quat,
)
from smp.utils import detect_device

# ---------------------------------------------------------------------------
# CasBot-specific constants
# ---------------------------------------------------------------------------

JOINT_NAMES: tuple[str, ...] = (
  "left_leg_pelvic_pitch_joint",
  "left_leg_pelvic_roll_joint",
  "left_leg_pelvic_yaw_joint",
  "left_leg_knee_pitch_joint",
  "left_leg_ankle_pitch_joint",
  "left_leg_ankle_roll_joint",
  "right_leg_pelvic_pitch_joint",
  "right_leg_pelvic_roll_joint",
  "right_leg_pelvic_yaw_joint",
  "right_leg_knee_pitch_joint",
  "right_leg_ankle_pitch_joint",
  "right_leg_ankle_roll_joint",
  "waist_yaw_joint",
  "head_yaw_joint",
  "head_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_pitch_joint",
  "right_wrist_yaw_joint",
)
NUM_JOINTS = len(JOINT_NAMES)

# End-effector body (link) names — must match the URDF link names.
# Two feet, head proxy, two hands (same structure as G1's 5 EEs).
EE_BODY_NAMES: tuple[str, ...] = (
  "left_leg_ankle_roll_link",
  "right_leg_ankle_roll_link",
  "waist_yaw_link",
  "left_wrist_yaw_link",
  "right_wrist_yaw_link",
)
NUM_EE = len(EE_BODY_NAMES)

# Default URDF path (relative to the whole_body_tracking project).
# Override with --urdf-path if your layout differs.
_DEFAULT_URDF = (
  "/home/casbot/Desktop/whole_body_tracking/source/whole_body_tracking/"
  "whole_body_tracking/assets/casbot_skeleton_description/urdf/"
  "casbot_skeleton_25dof.urdf"
)


# ---------------------------------------------------------------------------
# Mujoco FK helpers
# ---------------------------------------------------------------------------

class _CasbotFK:
  """Lightweight FK via mujoco (no Omniverse needed)."""

  def __init__(self, urdf_path: str) -> None:
    self._model = mujoco.MjModel.from_xml_path(urdf_path)
    self._data = mujoco.MjData(self._model)

    # Joint qpos addresses for the 25 actuated joints.
    self._joint_qposadr = np.array(
      [
        self._model.jnt_qposadr[
          mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
        ]
        for name in JOINT_NAMES
      ],
      dtype=np.int32,
    )
    # Body IDs for the 5 end-effectors.
    self._ee_body_ids = np.array(
      [mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, n) for n in EE_BODY_NAMES],
      dtype=np.int32,
    )

  def forward(
    self,
    root_pos: np.ndarray,   # (T, 3)
    root_quat: np.ndarray,  # (T, 4)  wxyz
    dof_pos: np.ndarray,    # (T, 25)
  ) -> np.ndarray:
    """Run FK for every frame, return EE positions ``(T, 5, 3)`` in world frame."""
    T = root_pos.shape[0]
    ee_pos = np.empty((T, NUM_EE, 3), dtype=np.float32)
    for t in range(T):
      qpos = np.zeros(self._model.nq, dtype=np.float64)
      qpos[0:3] = root_pos[t]
      qpos[3:7] = root_quat[t]
      qpos[self._joint_qposadr] = dof_pos[t].astype(np.float64)
      self._data.qpos[:] = qpos
      mujoco.mj_forward(self._model, self._data)
      ee_pos[t] = self._data.xpos[self._ee_body_ids].astype(np.float32)
    return ee_pos


# ---------------------------------------------------------------------------
# Motion feature helpers (same as csv_to_npz.py)
# ---------------------------------------------------------------------------

def _tan_norm_from_quat(quat: torch.Tensor) -> torch.Tensor:
  """Convert quaternion (wxyz) to 6D tan-norm.

  Stacks the rotation matrix's first column (rotated x-axis) and third
  column (rotated z-axis).  Input ``(..., 4)``, output ``(..., 6)`` as
  ``[col0_xyz, col2_xyz]``.
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

  All spatial quantities are anchored to the LAST window frame's yaw-only
  local frame (origin at pelvis_T, heading = yaw_T).
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

  # root_pos: xy in heading-inv frame, z terrain-relative.
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


# ---------------------------------------------------------------------------
# pkl motion loader (adapted from casbot_pkl_to_npz.py MotionLoader)
# ---------------------------------------------------------------------------

def _load_pkl_motion(
  pkl_path: Path,
  input_fps: int,
  output_fps: int,
  device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
  """Load a pkl file, interpolate to output_fps, compute velocities.

  Returns ``(base_pos, base_quat, base_lin_vel, base_ang_vel, dof_pos)``
  each of shape ``(T, *)`` on ``device``, or ``None`` if too short.
  """
  with open(pkl_path, "rb") as f:
    motion = pickle.load(f)

  root_pos = torch.from_numpy(motion["root_pos"]).float().to(device)
  # pkl stores xyzw; convert to wxyz.
  root_rot = torch.from_numpy(motion["root_rot"]).float().to(device)
  root_rot = root_rot[:, [3, 0, 1, 2]]
  dof_pos = torch.from_numpy(motion["dof_pos"]).float().to(device)

  T = root_pos.shape[0]
  if T < 4:
    return None

  input_dt = 1.0 / input_fps
  output_dt = 1.0 / output_fps
  duration = (T - 1) * input_dt
  times = torch.arange(0, duration, output_dt, device=device, dtype=torch.float32)
  out_frames = times.shape[0]

  # Interpolation indices.
  phase = times / duration
  idx0 = (phase * (T - 1)).floor().long()
  idx1 = torch.minimum(idx0 + 1, torch.tensor(T - 1))
  blend = (phase * (T - 1) - idx0.float()).unsqueeze(1)

  base_pos = root_pos[idx0] * (1 - blend) + root_pos[idx1] * blend

  # SLERP for quaternions.
  base_quat = torch.zeros(out_frames, 4, device=device)
  for i in range(out_frames):
    base_quat[i] = quat_slerp(root_rot[idx0[i]], root_rot[idx1[i]], blend[i, 0].item())

  dof_pos_out = dof_pos[idx0] * (1 - blend) + dof_pos[idx1] * blend

  # Finite-difference velocities (central difference where possible).
  base_lin_vel = torch.zeros_like(base_pos)
  base_lin_vel[1:-1] = (base_pos[2:] - base_pos[:-2]) / (2 * output_dt)
  base_lin_vel[0] = (base_pos[1] - base_pos[0]) / output_dt
  base_lin_vel[-1] = (base_pos[-1] - base_pos[-2]) / output_dt

  # Angular velocity from SO(3) central difference.
  base_ang_vel = torch.zeros(out_frames, 3, device=device)
  for i in range(1, out_frames - 1):
    q_rel = quat_mul(base_quat[i + 1], quat_conjugate(base_quat[i - 1]))
    base_ang_vel[i] = axis_angle_from_quat(q_rel) / (2 * output_dt)
  q_rel0 = quat_mul(base_quat[1], quat_conjugate(base_quat[0]))
  base_ang_vel[0] = axis_angle_from_quat(q_rel0) / output_dt
  q_rel1 = quat_mul(base_quat[-1], quat_conjugate(base_quat[-2]))
  base_ang_vel[-1] = axis_angle_from_quat(q_rel1) / output_dt

  return base_pos, base_quat, base_lin_vel, base_ang_vel, dof_pos_out


# ---------------------------------------------------------------------------
# Config & main
# ---------------------------------------------------------------------------

@dataclass
class Cfg:
  input_dir: str = "datasets/pkl"
  """Directory of input pkl motion files."""
  output_dir: str = "datasets/npz"
  """Directory to write output NPZ window files."""
  urdf_path: str = _DEFAULT_URDF
  """Path to casbot_skeleton_25dof.urdf."""
  window_size: int = 10
  """Number of frames per window."""
  stride: int = 1
  """Stride between consecutive windows."""
  input_fps: int = 60
  """pkl frame rate."""
  output_fps: int = 50
  """Output (and sim) frame rate after interpolation."""
  device: str = ""
  """Compute device. Empty = auto (cuda if available else cpu)."""
  shard_index: int = 0
  """Index of this shard (for parallel runs)."""
  num_shards: int = 1
  """Total number of shards (for parallel runs)."""


def main(cfg: Cfg) -> None:
  if not cfg.device:
    cfg.device = detect_device()
  print(f"Device: {cfg.device}")

  in_dir = Path(cfg.input_dir)
  out_dir = Path(cfg.output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  pkl_files = sorted(in_dir.glob("*.pkl"))
  if not pkl_files:
    msg = f"No pkl files found in {in_dir}"
    raise FileNotFoundError(msg)
  if cfg.num_shards > 1:
    pkl_files = pkl_files[cfg.shard_index :: cfg.num_shards]
    print(f"Shard {cfg.shard_index}/{cfg.num_shards}: {len(pkl_files)} files")

  device = torch.device(cfg.device)
  fk = _CasbotFK(cfg.urdf_path)

  feature_dims = [3, 6, NUM_JOINTS, NUM_EE * 3, 3, 3]
  total_feature_dim = sum(feature_dims)

  print(f"Files: {len(pkl_files)} in {in_dir}")
  print(f"Output: {out_dir}")
  print(f"Window: size={cfg.window_size} stride={cfg.stride} fps={cfg.output_fps}")
  print(f"End-effectors: {NUM_EE} {EE_BODY_NAMES} | Joints: {NUM_JOINTS}")
  print(
    f"Feature dim: {total_feature_dim} "
    f"(= 3 root_pos + 6 root_rot + {NUM_JOINTS} joint_pos + {NUM_EE * 3} "
    f"ee_pos + 3 lin_vel + 3 ang_vel)"
  )

  for i, pkl_path in enumerate(pkl_files):
    print(f"\n[{i + 1}/{len(pkl_files)}] {pkl_path.name}")

    loaded = _load_pkl_motion(pkl_path, cfg.input_fps, cfg.output_fps, device)
    if loaded is None:
      print(f"  [SKIP] too short (need >= 4 frames)")
      continue
    base_pos, base_quat, base_lin_vel, base_ang_vel, dof_pos = loaded

    # FK: compute end-effector world positions.
    pos_np = base_pos.cpu().numpy()
    quat_np = base_quat.cpu().numpy()
    dof_np = dof_pos.cpu().numpy()
    ee_np = fk.forward(pos_np, quat_np, dof_np)
    ee_pos = torch.from_numpy(ee_np).float().to(device)

    windows = _compute_windows(
      base_pos, base_quat, base_lin_vel, base_ang_vel,
      ee_pos, dof_pos, cfg.window_size, cfg.stride,
    )
    if windows is None:
      print(f"  [SKIP] too short for window_size={cfg.window_size}")
      continue

    out_path = out_dir / f"{pkl_path.stem}.npz"
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
