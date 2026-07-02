"""Unconditionally generate CasBot motion windows with a trained SMP
diffusion model and visualize the predicted trajectory in a viser viewer.

Adapted from generate_viz.py for CasBot 25-DOF:
  - 55-dim features instead of 59
  - CasBot URDF loaded via mujoco instead of mjlab G1 sim
  - Same DDPM sampling + trajectory reconstruction logic

Usage:
  python scripts/casbot_generate_viz.py --ckpt-path logs/pretrain/.../pretrained.pt
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch
import tyro
import viser
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_conjugate,
  quat_from_matrix,
  quat_mul,
  yaw_quat,
)
from mjlab.viewer.viser.scene import MjlabViserScene

from smp.pretrain.model import DiffusionDenoiser
from smp.pretrain.scheduler import DDPMScheduler
from smp.utils import detect_device

# ---------------------------------------------------------------------------
# CasBot constants
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

EE_BODY_NAMES: tuple[str, ...] = (
  "left_leg_ankle_roll_link",
  "right_leg_ankle_roll_link",
  "head_pitch_link",
  "left_wrist_yaw_link",
  "right_wrist_yaw_link",
)
NUM_EE = len(EE_BODY_NAMES)

_DEFAULT_URDF = (
  "/home/casbot/Desktop/whole_body_tracking/source/whole_body_tracking/"
  "whole_body_tracking/assets/casbot_skeleton_description/urdf/"
  "casbot_skeleton_25dof.urdf"
)

# ---------------------------------------------------------------------------
# Feature-to-state (CasBot-adapted from smp.sampling.feature_to_state)
# ---------------------------------------------------------------------------


def _slice_features(frame: torch.Tensor) -> dict[str, torch.Tensor]:
  """Slice a CasBot feature vector into named components.

  Layout (matches casbot_pkl_to_npz.py):
    [0:3]                   root_pos
    [3:9]                   root_rot (6D tan-norm)
    [9:9+J]                 joint_pos (J=25)
    [9+J:9+J+E*3]           ee_pos (E=5)
    [9+J+E*3:12+J+E*3]      root_lin_vel
    [12+J+E*3:15+J+E*3]     root_ang_vel
  """
  J = NUM_JOINTS
  E = NUM_EE
  expected = 3 + 6 + J + E * 3 + 3 + 3
  if (d := frame.shape[-1]) != expected:
    msg = f"expected feature_dim={expected}; got {d}"
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


def _rot6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
  """Convert 6D tan-norm [col0, col2] → 3×3 rotation matrix."""
  col0 = d6[..., :3]
  col2 = d6[..., 3:6]
  col0 = torch.nn.functional.normalize(col0, dim=-1)
  col2 = col2 - (col0 * col2).sum(dim=-1, keepdim=True) * col0
  col2 = torch.nn.functional.normalize(col2, dim=-1)
  col1 = torch.cross(col2, col0, dim=-1)
  return torch.stack([col0, col1, col2], dim=-1)


def _rot6d_to_quat(d6: torch.Tensor) -> torch.Tensor:
  return quat_from_matrix(_rot6d_to_matrix(d6))


def _window_to_pelvis_trajectory(
  window: torch.Tensor,
  anchor_pelvis_pos_w: torch.Tensor,
  anchor_pelvis_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Reconstruct world-frame pelvis pose + joint pos for each frame."""
  parts = _slice_features(window)
  root_pos_local = parts["root_pos"]
  root_rot_6d = parts["root_rot"]
  W = window.shape[0]

  anchor_pelvis_pos_w = anchor_pelvis_pos_w.to(window)
  anchor_pelvis_quat_w = anchor_pelvis_quat_w.to(window)
  yaw_T = yaw_quat(anchor_pelvis_quat_w[None]).squeeze(0)

  local_xy = root_pos_local.clone()
  local_xy[..., 2] = 0.0
  world_offset_xy = quat_apply(yaw_T[None].expand(W, 4), local_xy)
  pelvis_pos_w = world_offset_xy + anchor_pelvis_pos_w[None, :3]
  pelvis_pos_w = pelvis_pos_w.clone()
  pelvis_pos_w[..., 2] = root_pos_local[..., 2]

  root_rot_local_quat = _rot6d_to_quat(root_rot_6d)
  pelvis_quat_w = quat_mul(yaw_T[None].expand(W, 4), root_rot_local_quat)

  return pelvis_pos_w, pelvis_quat_w, parts["joint_pos"]


def _window_to_ee_trajectories(
  window: torch.Tensor,
  pelvis_pos_w: torch.Tensor,
  pelvis_quat_w: torch.Tensor,
) -> torch.Tensor:
  """Reconstruct world-frame EE positions across a window."""
  parts = _slice_features(window)
  W = window.shape[0]
  E = NUM_EE
  ee_pos_local = parts["ee_pos"].reshape(W, E, 3)

  yaw_T = yaw_quat(pelvis_quat_w[-1:])
  yaw_T_E = yaw_T.expand(W, 4)[:, None, :].expand(W, E, 4).reshape(-1, 4)
  ee_offset_w = quat_apply(yaw_T_E, ee_pos_local.reshape(-1, 3)).reshape(W, E, 3)
  return ee_offset_w + pelvis_pos_w[:, None, :]


# ---------------------------------------------------------------------------
# Mujoco CasBot helpers
# ---------------------------------------------------------------------------


class _CasbotSim:
  """Minimal mujoco sim for CasBot (replaces mjlab Entity/Simulation)."""

  def __init__(self, urdf_path: str) -> None:
    self.model = mujoco.MjModel.from_xml_path(urdf_path)
    self.data = mujoco.MjData(self.model)
    # Joint qpos addresses for the 25 actuated joints.
    self._joint_qposadr = np.array(
      [
        self.model.jnt_qposadr[
          mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        ]
        for name in JOINT_NAMES
      ],
      dtype=np.int32,
    )

  def write_pose(
    self,
    pelvis_pos: np.ndarray,    # (3,)
    pelvis_quat_wxyz: np.ndarray,  # (4,)
    joint_pos: np.ndarray,     # (25,)
  ) -> None:
    """Write a single-frame pose to mujoco data and run FK."""
    self.data.qpos[0:3] = pelvis_pos
    self.data.qpos[3:7] = pelvis_quat_wxyz
    self.data.qpos[self._joint_qposadr] = joint_pos
    self.data.qvel[:] = 0.0
    mujoco.mj_forward(self.model, self.data)


# ---------------------------------------------------------------------------
# Config & main
# ---------------------------------------------------------------------------


@dataclass
class Cfg:
  ckpt_path: str = ""
  """Path to a local SMP diffusion checkpoint .pt file."""
  urdf_path: str = _DEFAULT_URDF
  """Path to casbot_skeleton_25dof.urdf."""
  device: str = ""
  """Compute device. Empty = auto."""
  fps: float = 50.0
  """Playback frame rate."""


def _build_model_and_scheduler(
  ckpt: dict, device: torch.device
) -> tuple[DiffusionDenoiser, DDPMScheduler, np.ndarray, np.ndarray]:
  cfg = ckpt["cfg"]
  model = DiffusionDenoiser(
    feature_dim=cfg["feature_dim"],
    window_size=cfg["window_size"],
    d_model=cfg.get("d_model", 256),
    nhead=cfg.get("nhead", 8),
    num_layers=cfg.get("num_layers", 2),
    dropout=cfg.get("dropout", 0.0),
  ).to(device)
  state = ckpt.get("model_ema") or ckpt["model"]
  model.load_state_dict(state)
  model.eval()
  scheduler = DDPMScheduler(
    num_timesteps=cfg.get("num_timesteps", 50),
  ).to(device)
  return model, scheduler, ckpt["q_low"], ckpt["q_high"]


def _quantile_denormalize(
  x: torch.Tensor, q_low: torch.Tensor, q_high: torch.Tensor
) -> torch.Tensor:
  return (x + 1.0) / 2.0 * (q_high - q_low) + q_low


@torch.no_grad()
def _run_generate(
  model: DiffusionDenoiser,
  scheduler: DDPMScheduler,
  q_low: np.ndarray,
  q_high: np.ndarray,
  window_size: int,
  feature_dim: int,
  device: torch.device,
) -> torch.Tensor:
  """Unconditional DDPM ancestral sampling. Returns (W, F) denormalized on CPU."""
  x_t = torch.randn(1, window_size, feature_dim, device=device)
  for t in reversed(range(scheduler.num_timesteps)):
    t_batch = torch.full((1,), t, dtype=torch.long, device=device)
    eps = model(x_t, t_batch)
    x_t = scheduler.step(eps, x_t, t)
  q_low_t = torch.from_numpy(q_low).float().to(device)
  q_high_t = torch.from_numpy(q_high).float().to(device)
  return _quantile_denormalize(x_t.squeeze(0), q_low_t, q_high_t).cpu()


def main(cfg: Cfg) -> None:
  device_str = cfg.device or detect_device()
  device = torch.device(device_str)
  print(f"Device: {device_str}")

  ckpt = torch.load(cfg.ckpt_path, map_location=device, weights_only=False)
  model, scheduler, q_low, q_high = _build_model_and_scheduler(ckpt, device)
  print(f"Loaded checkpoint epoch={ckpt.get('epoch')} from {cfg.ckpt_path}")

  feature_dim = int(ckpt["cfg"]["feature_dim"])
  window_size = int(ckpt["cfg"]["window_size"])
  print(
    f"  feature_dim={feature_dim}  window_size={window_size}  "
    f"num_timesteps={scheduler.num_timesteps}"
  )

  # Set up CasBot via mujoco.
  sim = _CasbotSim(cfg.urdf_path)
  print(f"Loaded CasBot URDF: {cfg.urdf_path}")

  # Anchor: use the model's default qpos0 (standing pose).
  anchor_pelvis_pos = np.array(sim.model.qpos0[0:3], dtype=np.float32)
  anchor_pelvis_quat = np.array(sim.model.qpos0[3:7], dtype=np.float32)

  def run() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pred_denorm = _run_generate(
      model, scheduler, q_low, q_high, window_size, feature_dim, device
    )
    p_pos, p_quat, p_joint = _window_to_pelvis_trajectory(
      pred_denorm,
      torch.from_numpy(anchor_pelvis_pos),
      torch.from_numpy(anchor_pelvis_quat),
    )
    ee_pos = _window_to_ee_trajectories(pred_denorm, p_pos, p_quat)
    return (
      p_pos.cpu().numpy(),
      p_quat.cpu().numpy(),
      p_joint.cpu().numpy(),
      ee_pos.cpu().numpy(),
    )

  state: dict = {"pred": run()}

  server = viser.ViserServer()
  viser_scene = MjlabViserScene(server, sim.model, num_envs=1)
  viser_scene.debug_visualization_enabled = True

  ee_points = server.scene.add_point_cloud(
    name="/fixed_bodies/predicted_ee_positions",
    points=np.zeros((NUM_EE, 3), dtype=np.float32),
    colors=np.tile(np.array([255, 80, 0], dtype=np.uint8), (NUM_EE, 1)),
    point_size=0.03,
  )

  with server.gui.add_folder("Generate"):
    frame_slider = server.gui.add_slider(
      "Frame", min=0, max=window_size - 1, step=1, initial_value=0
    )
    play_btn = server.gui.add_button("Play / Pause")
    resample_btn = server.gui.add_button("Resample")

  playing = {"v": True}

  @play_btn.on_click
  def _(_evt) -> None:
    playing["v"] = not playing["v"]

  @resample_btn.on_click
  def _(_evt) -> None:
    state["pred"] = run()

  def render(frame: int) -> None:
    p_pos, p_quat, p_joint, ee_pos = state["pred"]
    sim.write_pose(p_pos[frame], p_quat[frame], p_joint[frame])
    viser_scene.update_from_arrays(
	      body_xpos=np.asarray(sim.data.xpos).reshape(1, -1, 3),
	      body_xmat=np.asarray(sim.data.xmat).reshape(1, -1, 3, 3),
	      qpos=np.asarray(sim.data.qpos)[None],
      env_idx=0,
    )
    ee_points.points = ee_pos[frame]
    viser_scene.refresh_visualization()

  print("Viser server running. Open the printed URL in a browser.")
  dt_play = 1.0 / cfg.fps
  try:
    while True:
      render(int(frame_slider.value))
      if playing["v"]:
        nxt = (int(frame_slider.value) + 1) % window_size
        frame_slider.value = nxt
      time.sleep(dt_play)
  except KeyboardInterrupt:
    print("Shutting down.")


if __name__ == "__main__":
  main(tyro.cli(Cfg))
