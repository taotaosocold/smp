"""Convert height-map NPZ motion files to windowed NPZ files for conditional diffusion.

Each input NPZ contains per-frame G1 motion data + height_map (terrain).  This
script windows both the motion and terrain data, producing output NPZs with:

  motion_windows  (N, window_size, 59)   — same 59-dim feature layout as csv_to_npz
  terrain         (N, window_size, 187)  — height_map per frame (17×11 grid)

Motion features are anchored to the LAST window frame's yaw-only local frame
(identical to csv_to_npz.py).  Terrain stays in each frame's own local frame.

Usage:
  python scripts/height_map_to_npz.py \
    --input-dir datasets/walk_g1_height_map_npz \
    --output-dir datasets/conditional_npz
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from mjlab.utils.lab_api.math import (
    matrix_from_quat,
    quat_apply_inverse,
    quat_conjugate,
    quat_mul,
    yaw_quat,
)

from smp.utils import detect_device

# ── Body indices into body_*_w arrays (world excluded, so MJCF idx - 1) ──
PELVIS_IDX = 0
EE_IDXS = (6, 12, 15, 22, 29)  # left/right ankle, torso, left/right wrist
NUM_JOINTS = 29
NUM_EE = len(EE_IDXS)

# ── Feature dimension breakdown ──
FEATURE_DIMS = (3, 6, NUM_JOINTS, NUM_EE * 3, 3, 3)  # 59 total

# ── Terrain grid ──
GRID_X = 17
GRID_Y = 11
HEIGHT_MAP_DIM = GRID_X * GRID_Y  # 187


@dataclass
class Cfg:
    input_dir: str = "datasets/walk_g1_height_map_npz"
    """Directory of input NPZ files (height-map motion data)."""
    output_dir: str = "datasets/conditional_npz"
    """Directory to write output windowed NPZ files."""
    window_size: int = 10
    """Number of frames per window."""
    stride: int = 1
    """Stride between consecutive windows."""
    fps: int = 50
    """Input fps (data is already at this frame rate; no resampling)."""
    device: str = ""
    """Compute device. Empty = auto."""
    shard_index: int = 0
    """Index of this shard (for parallel runs)."""
    num_shards: int = 1
    """Total number of shards (for parallel runs)."""


def _tan_norm_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (wxyz) to 6D tan-norm: stacked [col0, col2] of rot matrix."""
    mat = matrix_from_quat(quat)
    col0 = mat[..., :, 0]
    col2 = mat[..., :, 2]
    return torch.cat([col0, col2], dim=-1)


def _compute_windows(
    base_pos: torch.Tensor,       # (T, 3)
    base_quat: torch.Tensor,      # (T, 4)
    base_lin_vel: torch.Tensor,   # (T, 3)
    base_ang_vel: torch.Tensor,   # (T, 3)
    ee_pos: torch.Tensor,         # (T, E, 3)
    joint_pos: torch.Tensor,      # (T, 29)
    height_map: torch.Tensor,     # (T, 187)
    window_size: int,
    stride: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Window motion and terrain data.  Returns (motion, terrain) or (None, None).

    Motion features anchored to the LAST frame's yaw-only local frame.
    Terrain stays in each frame's own local frame.
    """
    T = base_pos.shape[0]
    if T < window_size:
        return None, None

    E = ee_pos.shape[1]
    J = joint_pos.shape[1]

    starts = torch.arange(0, T - window_size + 1, stride, device=base_pos.device, dtype=torch.long)
    offsets = torch.arange(window_size, device=base_pos.device, dtype=torch.long)
    win_idx = starts[:, None] + offsets[None, :]  # (N, W)
    N, W = win_idx.shape[0], window_size

    flat_idx = win_idx.reshape(-1)

    # Gather motion data
    win_base_pos = base_pos.index_select(0, flat_idx).reshape(N, W, 3)
    win_base_quat = base_quat.index_select(0, flat_idx).reshape(N, W, 4)
    win_base_lin_vel = base_lin_vel.index_select(0, flat_idx).reshape(N, W, 3)
    win_base_ang_vel = base_ang_vel.index_select(0, flat_idx).reshape(N, W, 3)
    win_ee_pos = ee_pos.index_select(0, flat_idx).reshape(N, W, E, 3)
    win_joint = joint_pos.index_select(0, flat_idx).reshape(N, W, J)

    # ── Anchor to last frame's yaw-only local frame ──
    anchor_pos_T = win_base_pos[:, -1, :]          # (N, 3)
    anchor_quat_T = win_base_quat[:, -1, :]        # (N, 4)
    yaw_T = yaw_quat(anchor_quat_T)                # (N, 4)
    heading_inv_T_WF = quat_conjugate(yaw_T)[:, None, :].expand(N, W, 4).reshape(-1, 4)
    yaw_T_W = yaw_T[:, None, :].expand(N, W, 4).reshape(-1, 4)

    # root_pos: xy in heading-invariant frame, z in world
    root_offset = win_base_pos - anchor_pos_T[:, None, :]  # (N, W, 3)
    root_pos_local = quat_apply_inverse(yaw_T_W, root_offset.reshape(-1, 3)).reshape(N, W, 3)
    root_pos_local = root_pos_local.clone()
    root_pos_local[..., 2] = win_base_pos[..., 2]

    # root_rot: heading_inv(T) ⊗ root_quat[t] → 6D tan-norm
    root_rot_local_quat = quat_mul(
        heading_inv_T_WF, win_base_quat.reshape(-1, 4)
    ).reshape(N, W, 4)
    root_rot_6d = _tan_norm_from_quat(root_rot_local_quat)

    # EE: (ee[t] - root[t]) rotated into last-frame heading-invariant frame
    ee_offset_w = win_ee_pos - win_base_pos[:, :, None, :]  # (N, W, E, 3)
    yaw_T_E = yaw_T[:, None, None, :].expand(N, W, E, 4).reshape(-1, 4)
    ee_pos_local = quat_apply_inverse(yaw_T_E, ee_offset_w.reshape(-1, 3)).reshape(N, W, E * 3)

    # Velocities: rotated into last-frame heading-invariant frame
    lin_vel_local = quat_apply_inverse(yaw_T_W, win_base_lin_vel.reshape(-1, 3)).reshape(N, W, 3)
    ang_vel_local = quat_apply_inverse(yaw_T_W, win_base_ang_vel.reshape(-1, 3)).reshape(N, W, 3)

    motion = torch.cat(
        [root_pos_local, root_rot_6d, win_joint, ee_pos_local, lin_vel_local, ang_vel_local],
        dim=-1,
    )  # (N, W, 59)

    # Window terrain (keep in per-frame local frame, no rotation)
    win_terrain = height_map.index_select(0, flat_idx).reshape(N, W, HEIGHT_MAP_DIM)

    return motion, win_terrain


def main(cfg: Cfg) -> None:
    if not cfg.device:
        cfg.device = detect_device()
    print(f"Device: {cfg.device}")

    in_dir = Path(cfg.input_dir)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(in_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No NPZ files found in {in_dir}")
    if cfg.num_shards > 1:
        npz_files = npz_files[cfg.shard_index :: cfg.num_shards]
        print(f"Shard {cfg.shard_index}/{cfg.num_shards}: {len(npz_files)} files")

    total_motion_dim = sum(FEATURE_DIMS)
    print(f"Files: {len(npz_files)} in {in_dir}")
    print(f"Output: {out_dir}")
    print(f"Window: size={cfg.window_size} stride={cfg.stride}")
    print(f"Motion dim: {total_motion_dim} (= {' + '.join(str(d) for d in FEATURE_DIMS)})")
    print(f"Terrain dim: {HEIGHT_MAP_DIM} (= {GRID_X}×{GRID_Y} grid)")

    total_windows = 0

    for i, npz_path in enumerate(npz_files):
        print(f"\n[{i + 1}/{len(npz_files)}] {npz_path.name}")

        data = np.load(npz_path, allow_pickle=False)

        # Validate
        T = data["joint_pos"].shape[0]
        if T < cfg.window_size:
            print(f"  [SKIP] too short ({T} < {cfg.window_size})")
            continue

        # Extract motion data → torch tensors on device
        device = torch.device(cfg.device)
        base_pos = torch.from_numpy(data["body_pos_w"][:, PELVIS_IDX, :].astype(np.float32)).to(device)
        base_quat = torch.from_numpy(data["body_quat_w"][:, PELVIS_IDX, :].astype(np.float32)).to(device)
        base_lin_vel = torch.from_numpy(data["body_lin_vel_w"][:, PELVIS_IDX, :].astype(np.float32)).to(device)
        base_ang_vel = torch.from_numpy(data["body_ang_vel_w"][:, PELVIS_IDX, :].astype(np.float32)).to(device)

        ee_pos_list = [data["body_pos_w"][:, idx, :].astype(np.float32) for idx in EE_IDXS]
        ee_pos = torch.from_numpy(np.stack(ee_pos_list, axis=1)).to(device)  # (T, E, 3)

        joint_pos = torch.from_numpy(data["joint_pos"].astype(np.float32)).to(device)

        height_map = torch.from_numpy(data["height_map"].astype(np.float32)).to(device)

        motion, terrain = _compute_windows(
            base_pos, base_quat, base_lin_vel, base_ang_vel,
            ee_pos, joint_pos, height_map,
            cfg.window_size, cfg.stride,
        )

        if motion is None:
            print(f"  [SKIP] too short for window_size={cfg.window_size}")
            continue

        n_windows = motion.shape[0]
        total_windows += n_windows

        out_path = out_dir / f"{npz_path.stem}.npz"
        np.savez_compressed(
            out_path,
            motion_windows=motion.cpu().numpy().astype(np.float32),
            terrain=terrain.cpu().numpy().astype(np.float32),
            fps=np.array([cfg.fps], dtype=np.float32),
            window_size=np.array([cfg.window_size], dtype=np.int32),
            stride=np.array([cfg.stride], dtype=np.int32),
            feature_dims=np.array(FEATURE_DIMS, dtype=np.int32),
            grid_x=data["grid_x"].astype(np.float32),
            grid_y=data["grid_y"].astype(np.float32),
        )
        print(f"  saved {out_path.name}: motion={tuple(motion.shape)} terrain={tuple(terrain.shape)}")

    print(f"\nDone. Total windows: {total_windows}")


if __name__ == "__main__":
    main(tyro.cli(Cfg))
