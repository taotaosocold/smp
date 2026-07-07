"""Convert SMP feature windows back to robot states for viz + GSI.

Per-frame motion features: root pose (pos + 6D rot), joint angles,
end-effector positions, and root velocities.  All spatial quantities are
expressed in the LAST window frame's yaw-only local frame (origin at
pelvis_T, heading = yaw_T).  root_pos z is terrain-relative.
"""

from __future__ import annotations

import torch
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_conjugate,
  quat_from_matrix,
  quat_mul,
  yaw_quat,
)

NUM_JOINTS = 29

# Tracked end-effector bodies. ``torso_link`` proxies the head (mjlab's G1
# asset has no separate head body, but the head is rigidly attached to the
# torso so the kinematic signal is the same).  Order must match
# ``scripts/csv_to_npz.py::EE_BODY_NAMES``.
EE_BODY_NAMES: tuple[str, ...] = (
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "torso_link",
  "left_wrist_yaw_link",
  "right_wrist_yaw_link",
)
NUM_EE = len(EE_BODY_NAMES)


def slice_features(frame: torch.Tensor) -> dict[str, torch.Tensor]:
  """Slice a feature vector into named components.

  Layout (matches ``scripts/csv_to_npz.py::_compute_windows``):
    [0:3]                   root_pos       xy in last-frame heading-inv, z terrain-relative
    [3:9]                   root_rot       6D tan-norm of heading_inv(T) ⊗ root_quat[t]
    [9:9+J]                 joint_pos      raw joint angles (J = 29 for G1)
    [9+J:9+J+E*3]           ee_pos         per-frame root offset, last-frame
                                           heading-inv rotation (E = 5)
    [9+J+E*3:12+J+E*3]      root_lin_vel   last-frame heading-inv
    [12+J+E*3:15+J+E*3]     root_ang_vel   last-frame heading-inv
  """
  J = NUM_JOINTS
  E = NUM_EE
  expected = 3 + 6 + J + E * 3 + 3 + 3
  if frame.shape[-1] != expected:
    msg = f"expected feature_dim={expected}; got {frame.shape[-1]}"
    raise ValueError(msg)
  joint_pos_end = 9 + J
  ee_pos_end = joint_pos_end + E * 3
  lin_vel_end = ee_pos_end + 3
  ang_vel_end = lin_vel_end + 3
  return {
    "root_pos": frame[..., 0:3],
    "root_rot": frame[..., 3:9],
    "joint_pos": frame[..., 9:joint_pos_end],
    "ee_pos": frame[..., joint_pos_end:ee_pos_end],
    "root_lin_vel": frame[..., ee_pos_end:lin_vel_end],
    "root_ang_vel": frame[..., lin_vel_end:ang_vel_end],
  }


def rot6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
  """Convert 6D tan-norm ``[col0, col2]`` to a 3×3 rotation matrix.

  Inverse of the tan-norm encoder in ``scripts/csv_to_npz.py`` — the input
  is the rotation matrix's first column (rotated x-axis) followed by its
  third column (rotated z-axis).  Reconstructs the missing middle column
  via ``col1 = col2 × col0`` and re-orthogonalizes with Gram-Schmidt for
  robustness to numerical drift.
  """
  col0 = d6[..., :3]
  col2 = d6[..., 3:6]
  col0 = torch.nn.functional.normalize(col0, dim=-1)
  col2 = col2 - (col0 * col2).sum(dim=-1, keepdim=True) * col0
  col2 = torch.nn.functional.normalize(col2, dim=-1)
  # Right-handed: col0 × col1 = col2 → col1 = col2 × col0.
  col1 = torch.cross(col2, col0, dim=-1)
  return torch.stack([col0, col1, col2], dim=-1)


def rot6d_to_quat(d6: torch.Tensor) -> torch.Tensor:
  """Convert 6D rotation representation to (w, x, y, z) quaternion."""
  return quat_from_matrix(rot6d_to_matrix(d6))


def window_to_pelvis_trajectory(
  window: torch.Tensor,
  anchor_pelvis_pos_w: torch.Tensor,
  anchor_pelvis_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Reconstruct pelvis pose + joint pos for each frame in a window.

  ``root_pos`` z is terrain-relative (height above terrain).  Callers must
  add terrain height to convert back to world z when needed.

  Args:
    window: (W, F) denormalized feature window.
    anchor_pelvis_pos_w: (3,) world position of the pelvis at frame T (last).
    anchor_pelvis_quat_w: (4,) world quaternion (wxyz) at frame T.

  Returns:
    pelvis_pos_w:  (W, 3) — z is terrain-relative
    pelvis_quat_w: (W, 4) wxyz
    joint_pos:     (W, J)
  """
  parts = slice_features(window)
  root_pos_local = parts["root_pos"]
  root_rot_6d = parts["root_rot"]
  W = window.shape[0]

  anchor_pelvis_pos_w = anchor_pelvis_pos_w.to(window)
  anchor_pelvis_quat_w = anchor_pelvis_quat_w.to(window)
  yaw_T = yaw_quat(anchor_pelvis_quat_w[None]).squeeze(0)

  # Invert root_pos_local: rotate xy back into world, add pelvis_T xy, keep z.
  local_xy = root_pos_local.clone()
  local_xy[..., 2] = 0.0
  world_offset_xy = quat_apply(yaw_T[None].expand(W, 4), local_xy)
  pelvis_pos_w = world_offset_xy + anchor_pelvis_pos_w[None, :3]
  pelvis_pos_w = pelvis_pos_w.clone()
  pelvis_pos_w[..., 2] = root_pos_local[..., 2]

  # Invert root_rot_local: root_quat_w[t] = yaw_T ⊗ rot_local[t].
  root_rot_local_quat = rot6d_to_quat(root_rot_6d)
  pelvis_quat_w = quat_mul(yaw_T[None].expand(W, 4), root_rot_local_quat)

  return pelvis_pos_w, pelvis_quat_w, parts["joint_pos"]


def window_to_ee_trajectories(
  window: torch.Tensor,
  pelvis_pos_w: torch.Tensor,
  pelvis_quat_w: torch.Tensor,
) -> torch.Tensor:
  """Reconstruct world-frame end-effector positions across a window.

  EE positions are stored as ``(ee[t] - root[t])`` rotated into the
  last-frame heading-inv frame.  Lifting needs the per-frame pelvis world
  position and the anchor yaw_T (defined by ``pelvis_quat_w[-1]``).

  Returns: ``(W, E, 3)`` world-frame EE positions.
  """
  parts = slice_features(window)
  W = window.shape[0]
  E = NUM_EE
  ee_pos_local = parts["ee_pos"].reshape(W, E, 3)

  yaw_T = yaw_quat(pelvis_quat_w[-1:])
  yaw_T_E = yaw_T.expand(W, 4)[:, None, :].expand(W, E, 4).reshape(-1, 4)

  ee_offset_w = quat_apply(yaw_T_E, ee_pos_local.reshape(-1, 3)).reshape(W, E, 3)
  return ee_offset_w + pelvis_pos_w[:, None, :]


def tan_norm_from_quat(quat: torch.Tensor) -> torch.Tensor:
  """Convert quaternion (wxyz) to 6D tan-norm.

  Stacks the rotation matrix's first column (rotated x-axis) and third
  column (rotated z-axis).
  """
  mat = matrix_from_quat(quat)
  col0 = mat[..., :, 0]
  col2 = mat[..., :, 2]
  return torch.cat([col0, col2], dim=-1)


def heading_inv_quat(quat: torch.Tensor) -> torch.Tensor:
  """Yaw-only inverse of a world quaternion."""
  return quat_conjugate(yaw_quat(quat))
