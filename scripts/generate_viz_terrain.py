"""Terrain-conditioned motion generation with a trained SMP diffusion model.

Like generate_viz.py, but passes a height-map terrain window to the denoiser
so the generated motion is conditioned on the terrain underneath each frame.

Usage:
  python scripts/generate_viz_terrain.py \
    --ckpt-path logs/pretrain/lafan_g1_height_map/.../checkpoint_00600.pt \
    --data-dir datasets/walk_g1_height_map_npz
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
import viser
from mjlab.entity import Entity
from mjlab.viewer.viser.scene import MjlabViserScene

from smp.pretrain.model import DiffusionDenoiser
from smp.pretrain.scheduler import DDPMScheduler
from smp.sampling.feature_to_state import (
  NUM_EE,
  window_to_ee_trajectories,
  window_to_pelvis_trajectory,
)
from smp.utils import detect_device

GRID_X = 17
GRID_Y = 11
CENTER_IDX = GRID_X // 2 * GRID_Y + GRID_Y // 2  # 93 = terrain directly below pelvis

def _load_grid_xy(data_dir: str) -> tuple[np.ndarray, np.ndarray]:
  """Load grid_x, grid_y from the first NPZ file and build local (x,y) offsets."""
  npz_files = sorted(Path(data_dir).glob("*.npz"))
  with np.load(npz_files[0], allow_pickle=False) as d:
    gx = d["grid_x"].astype(np.float32)
    gy = d["grid_y"].astype(np.float32)
  gxv, gyv = np.meshgrid(gx, gy)
  return np.stack([gxv.ravel(), gyv.ravel()], axis=-1)  # (187, 2)


@dataclass
class Cfg:
  ckpt_path: str = ""
  """Path to a terrain-conditioned SMP checkpoint .pt file."""
  data_dir: str = "datasets/walk_g1_height_map_npz"
  """Directory of windowed NPZ files with terrain data."""
  device: str = ""
  """Compute device. Empty = auto."""
  fps: float = 50.0
  """Playback frame rate."""


def _build_model_and_scheduler(
  ckpt: dict, device: torch.device
) -> tuple[DiffusionDenoiser, DDPMScheduler, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  cfg = ckpt["cfg"]
  terrain_dim = cfg.get("terrain_dim")
  model = DiffusionDenoiser(
    feature_dim=cfg["feature_dim"],
    window_size=cfg["window_size"],
    d_model=cfg.get("d_model", 256),
    nhead=cfg.get("nhead", 4),
    num_layers=cfg.get("num_layers", 2),
    dropout=cfg.get("dropout", 0.0),
    terrain_dim=terrain_dim,
  ).to(device)
  state = ckpt.get("model_ema") or ckpt["model"]
  model.load_state_dict(state)
  model.eval()
  scheduler = DDPMScheduler(num_timesteps=cfg.get("num_timesteps", 50)).to(device)
  t_q_low = ckpt.get("t_q_low")
  t_q_high = ckpt.get("t_q_high")
  return model, scheduler, ckpt["q_low"], ckpt["q_high"], t_q_low, t_q_high


def _load_terrain_samples(data_dir: str, device: torch.device) -> torch.Tensor:
  """Load all terrain windows from the dataset into a single tensor (T, W, 187)."""
  npz_files = sorted(Path(data_dir).glob("*.npz"))
  chunks = []
  for f in npz_files:
    with np.load(f, allow_pickle=False) as d:
      chunks.append(d["terrain"].astype(np.float32))
  return torch.from_numpy(np.concatenate(chunks, axis=0)).to(device)


def _quantile_denormalize(
  x: torch.Tensor, q_low: torch.Tensor, q_high: torch.Tensor
) -> torch.Tensor:
  return (x + 1.0) / 2.0 * (q_high - q_low) + q_low


def _quantile_normalize(
  x: torch.Tensor, q_low: torch.Tensor, q_high: torch.Tensor
) -> torch.Tensor:
  return 2.0 * (x - q_low) / (q_high - q_low) - 1.0


@torch.no_grad()
def _run_generate(
  model: DiffusionDenoiser,
  scheduler: DDPMScheduler,
  q_low: np.ndarray,
  q_high: np.ndarray,
  t_q_low: np.ndarray | None,
  t_q_high: np.ndarray | None,
  terrain_raw: torch.Tensor,  # (1, W, 187) raw terrain heights
  window_size: int,
  feature_dim: int,
  device: torch.device,
) -> torch.Tensor:
  """Terrain-conditioned DDPM ancestral sampling. Returns (W, F) on CPU."""
  # Normalize terrain for the model
  if t_q_low is not None and t_q_high is not None:
    t_low = torch.from_numpy(t_q_low).float().to(device)
    t_high = torch.from_numpy(t_q_high).float().to(device)
    terrain = _quantile_normalize(terrain_raw, t_low, t_high)
  else:
    terrain = terrain_raw

  x_t = torch.randn(1, window_size, feature_dim, device=device)
  for t in reversed(range(scheduler.num_timesteps)):
    t_batch = torch.full((1,), t, dtype=torch.long, device=device)
    eps = model(x_t, t_batch, terrain=terrain)
    x_t = scheduler.step(eps, x_t, t)

  q_low_t = torch.from_numpy(q_low).float().to(device)
  q_high_t = torch.from_numpy(q_high).float().to(device)
  return _quantile_denormalize(x_t.squeeze(0), q_low_t, q_high_t).cpu()


def _setup_g1_sim(device: str):
  from mjlab.scene import Scene
  from mjlab.sim.sim import Simulation, SimulationCfg
  from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg

  sim_cfg = SimulationCfg()
  env_cfg = unitree_g1_flat_tracking_env_cfg()
  scene = Scene(env_cfg.scene, device=device)
  model = scene.compile()
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)
  return sim, scene


def _write_pose_to_robot(
  robot: Entity,
  pelvis_pos: np.ndarray,
  pelvis_quat_wxyz: np.ndarray,
  joint_pos: np.ndarray,
  device: str,
) -> None:
  root = robot.data.default_root_state.clone()
  root[:, 0:3] = torch.as_tensor(pelvis_pos, device=device, dtype=root.dtype)
  root[:, 3:7] = torch.as_tensor(pelvis_quat_wxyz, device=device, dtype=root.dtype)
  robot.write_root_state_to_sim(root)
  jp = robot.data.default_joint_pos.clone()
  jp[:] = torch.as_tensor(joint_pos, device=device, dtype=jp.dtype)
  jv = robot.data.default_joint_vel.clone()
  robot.write_joint_state_to_sim(jp, jv)


def main(cfg: Cfg) -> None:
  device_str = cfg.device or detect_device()
  device = torch.device(device_str)
  print(f"Device: {device_str}")

  ckpt = torch.load(cfg.ckpt_path, map_location=device, weights_only=False)
  model, scheduler, q_low, q_high, t_q_low, t_q_high = _build_model_and_scheduler(ckpt, device)
  print(f"Loaded checkpoint epoch={ckpt.get('epoch')} from {cfg.ckpt_path}")

  feature_dim = int(ckpt["cfg"]["feature_dim"])
  window_size = int(ckpt["cfg"]["window_size"])
  print(f"  feature_dim={feature_dim}  window_size={window_size}  "
        f"num_timesteps={scheduler.num_timesteps}  terrain_dim={ckpt['cfg'].get('terrain_dim')}")

  # Load all terrain windows + grid geometry
  all_terrains = _load_terrain_samples(cfg.data_dir, device)
  local_xy = _load_grid_xy(cfg.data_dir)  # (187, 2) local-frame grid offsets
  print(f"Loaded {all_terrains.shape[0]} terrain windows from {cfg.data_dir}")
  print(f"Grid: {local_xy[:, 0].max() - local_xy[:, 0].min():.1f}m x "
        f"{local_xy[:, 1].max() - local_xy[:, 1].min():.1f}m")

  sim_device = device_str
  sim, scene = _setup_g1_sim(sim_device)
  robot: Entity = scene["robot"]
  mj_model = sim.mj_model

  anchor_pelvis_pos = robot.data.default_root_state[0, 0:3].detach().cpu()
  anchor_pelvis_quat = robot.data.default_root_state[0, 3:7].detach().cpu()

  # State: track current terrain index and generated motion
  terrain_idx = {"v": 0}

  def run(terrain_raw: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, torch.Tensor]:
    pred_denorm = _run_generate(
      model, scheduler, q_low, q_high, t_q_low, t_q_high,
      terrain_raw, window_size, feature_dim, device,
    )
    p_pos, p_quat, p_joint = window_to_pelvis_trajectory(
      pred_denorm, anchor_pelvis_pos, anchor_pelvis_quat,
    )
    # root_pos[2] is already absolute pelvis z (terrain is center-subtracted,
    # so center cell = 0 → root_pos[2] = pelvis_world_z − 0 = pelvis_world_z).
    ee_pos = window_to_ee_trajectories(pred_denorm, p_pos, p_quat)
    return (
      p_pos.cpu().numpy(),
      p_quat.cpu().numpy(),
      p_joint.cpu().numpy(),
      ee_pos.cpu().numpy(),
      terrain_raw.cpu(),
    )

  terrain_sample = all_terrains[0:1]
  state: dict = {"pred": run(terrain_sample)}

  server = viser.ViserServer()
  viser_scene = MjlabViserScene(server, mj_model, num_envs=1)
  viser_scene.debug_visualization_enabled = True

  ee_points = server.scene.add_point_cloud(
    name="/fixed_bodies/predicted_ee_positions",
    points=np.zeros((NUM_EE, 3), dtype=np.float32),
    colors=np.tile(np.array([255, 80, 0], dtype=np.uint8), (NUM_EE, 1)),
    point_size=0.03,
  )

  # Height map point cloud
  hm_points_handle = server.scene.add_point_cloud(
    name="/fixed_bodies/height_map",
    points=np.zeros((GRID_X * GRID_Y, 3), dtype=np.float32),
    colors=np.full((GRID_X * GRID_Y, 3), np.array([100, 200, 100], dtype=np.uint8)),
    point_size=0.02,
  )

  with server.gui.add_folder("Generate"):
    frame_slider = server.gui.add_slider(
      "Frame", min=0, max=window_size - 1, step=1, initial_value=0
    )
    play_btn = server.gui.add_button("Play / Pause")
    resample_btn = server.gui.add_button("Resample Motion")
    next_terrain_btn = server.gui.add_button("Next Terrain")
    terrain_display = server.gui.add_number(
      "Terrain index", initial_value=0, disabled=True
    )

  playing = {"v": True}

  @play_btn.on_click
  def _(_evt) -> None:
    playing["v"] = not playing["v"]

  @resample_btn.on_click
  def _(_evt) -> None:
    t_raw = all_terrains[terrain_idx["v"]:terrain_idx["v"] + 1]
    state["pred"] = run(t_raw)

  @next_terrain_btn.on_click
  def _(_evt) -> None:
    ridx = int(torch.randint(0, all_terrains.shape[0], (1,)).item())
    terrain_idx["v"] = ridx
    terrain_display.value = ridx
    t_raw = all_terrains[ridx:ridx + 1]
    state["pred"] = run(t_raw)

  def render(frame: int) -> None:
    p_pos, p_quat, p_joint, ee_pos, terrain_raw = state["pred"]
    _write_pose_to_robot(robot, p_pos[frame], p_quat[frame], p_joint[frame], sim_device)
    sim.forward()
    wd = sim.wp_data
    viser_scene.update_from_arrays(
      body_xpos=np.asarray(wd.xpos.numpy()),
      body_xmat=np.asarray(wd.xmat.numpy()),
      qpos=np.asarray(wd.qpos.numpy()),
      env_idx=0,
    )
    ee_points.points = ee_pos[frame]

    # Height map: rotate by pelvis yaw + offset to pelvis xy + make z relative
    hm = terrain_raw[0, frame].numpy()                     # (187,)
    pq = p_quat[frame]                                      # (4,) wxyz
    w, x, y, z = pq
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    world_xy = local_xy.copy()
    world_xy_rot = np.empty_like(world_xy)
    world_xy_rot[:, 0] = world_xy[:, 0] * cos_y - world_xy[:, 1] * sin_y
    world_xy_rot[:, 1] = world_xy[:, 0] * sin_y + world_xy[:, 1] * cos_y
    world_xy_rot[:, 0] += p_pos[frame, 0]
    world_xy_rot[:, 1] += p_pos[frame, 1]
    # Terrain heights are center-subtracted (center cell = 0 at ground level).
    pts = np.stack([world_xy_rot[:, 0], world_xy_rot[:, 1], hm], axis=-1)
    hm_points_handle.points = pts.astype(np.float32)

    viser_scene.refresh_visualization()

  print("Viser server running. Open the printed URL.")
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
