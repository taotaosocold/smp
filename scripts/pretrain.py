"""Diffusion model pretraining loop (unconditional or terrain-conditioned).

When ``--terrain-data-dir`` is given, uses ``ConditionalMotionDataset`` and
automatically builds a terrain-conditioned denoiser.  Otherwise falls back to
the original unconditional path.
"""

from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import tyro
from torch.utils.data import DataLoader, random_split

from smp.pretrain.dataset import ConditionalMotionDataset, MotionWindowDataset
from smp.pretrain.model import DiffusionDenoiser
from smp.pretrain.pretrain_cfg import PretrainCfg
from smp.pretrain.scheduler import DDPMScheduler
from smp.utils import count_parameters, seed_everything


class _Ema:
  """Exponential moving average shadow of a model.

  Standard formula: ``θ_ema ← decay·θ_ema + (1−decay)·θ``, applied in-place
  over every entry of ``state_dict()`` (covers params and buffers).
  """

  def __init__(self, model: torch.nn.Module, decay: float) -> None:
    self.decay = decay
    self.shadow = copy.deepcopy(model)
    self.shadow.eval()
    for p in self.shadow.parameters():
      p.requires_grad_(False)

  @torch.no_grad()
  def update(self, model: torch.nn.Module) -> None:
    src = model.state_dict()
    dst = self.shadow.state_dict()
    for k, v_src in src.items():
      v_dst = dst[k]
      if v_dst.is_floating_point():
        v_dst.mul_(self.decay).add_(v_src.detach(), alpha=1.0 - self.decay)
      else:
        v_dst.copy_(v_src)


def _diffusion_loss(
  model: torch.nn.Module | DiffusionDenoiser,
  scheduler: DDPMScheduler,
  x_0: torch.Tensor,
  num_noise_samples: int,
  terrain: torch.Tensor | None = None,
) -> torch.Tensor:
  """DDPM ε-prediction L1 loss with multiple noise samples per data point.

  Each sample in the batch is paired with ``num_noise_samples`` random
  (timestep, noise) draws, giving lower-variance gradients than a single
  draw without the cost of exhausting all T timesteps.
  """
  B = x_0.shape[0]
  K = num_noise_samples
  # (B, W, F) → (B*K, W, F)
  x_0_exp = x_0[:, None].expand(B, K, *x_0.shape[1:]).reshape(B * K, *x_0.shape[1:])
  t = scheduler.sample_timesteps(B * K, x_0.device)
  noise = torch.randn_like(x_0_exp)
  x_t = scheduler.add_noise(x_0_exp, noise, t)

  if terrain is not None:
    terrain_exp = terrain[:, None].expand(B, K, *terrain.shape[1:]).reshape(B * K, *terrain.shape[1:])
  else:
    terrain_exp = None

  return F.l1_loss(model(x_t, t, terrain=terrain_exp), noise)


def _save_checkpoint(
  path: Path,
  epoch: int,
  model: DiffusionDenoiser,
  dataset: MotionWindowDataset | ConditionalMotionDataset,
  feature_dim: int,
  cfg: PretrainCfg,
  optimizer: torch.optim.Optimizer | None = None,
  ema: _Ema | None = None,
) -> None:
  data: dict[str, Any] = {
    "epoch": epoch,
    "model": model.state_dict(),
    "q_low": dataset.q_low,
    "q_high": dataset.q_high,
    "cfg": {
      **vars(cfg),
      "feature_dim": feature_dim,
      "window_size": dataset.window_size,
    },
  }
  if hasattr(dataset, "has_terrain") and dataset.has_terrain:
    data["t_q_low"] = dataset.t_q_low          # type: ignore[attr-defined]
    data["t_q_high"] = dataset.t_q_high         # type: ignore[attr-defined]
    data["cfg"]["terrain_dim"] = dataset.terrain_dim  # type: ignore[attr-defined]
  if optimizer is not None:
    data["optimizer"] = optimizer.state_dict()
  if ema is not None:
    data["model_ema"] = ema.shadow.state_dict()
  torch.save(data, path)


def pretrain(cfg: PretrainCfg) -> Path:
  """Run diffusion pretraining."""
  seed_everything(cfg.seed)
  print(f"[INFO] seed={cfg.seed}")
  device = torch.device(cfg.device)

  # ── Dataset (auto-detect conditional vs unconditional) ──
  data_dir = cfg.data_dir
  probe_files = sorted(Path(data_dir).glob("*.npz"))
  if not probe_files:
    raise FileNotFoundError(f"No NPZ files found in {data_dir}")

  import numpy as _np
  _probe = _np.load(str(probe_files[0]), allow_pickle=False)
  is_conditional = "terrain" in _probe or "motion_windows" in _probe

  if is_conditional:
    print("[INFO] Detected terrain-conditioned data, using ConditionalMotionDataset")
    tnorm = cfg.terrain_norm_stats_file or None
    dataset = ConditionalMotionDataset(
      data_dir,
      norm_stats_file=cfg.norm_stats_file if cfg.norm_stats_file else None,
      terrain_norm_stats_file=tnorm,
    )
    terrain_dim = dataset.terrain_dim
  else:
    print("[INFO] Using unconditional MotionWindowDataset")
    dataset = MotionWindowDataset(data_dir, norm_stats_file=cfg.norm_stats_file)
    terrain_dim = None

  feature_dim = dataset.feature_dim
  window_size = dataset.window_size

  n_train = int(len(dataset) * cfg.train_split)
  n_val = len(dataset) - n_train
  extra = f", terrain_dim={terrain_dim}" if terrain_dim else ""
  print(
    f"Dataset: {len(dataset)} windows, n_train={n_train}, n_val={n_val}, "
    f"feature_dim={feature_dim}, window_size={window_size}{extra}"
  )

  train_set, val_set = random_split(dataset, [n_train, n_val])
  pin_memory = device.type == "cuda"
  train_loader = DataLoader(
    train_set,
    batch_size=cfg.batch_size,
    shuffle=True,
    pin_memory=pin_memory,
  )
  val_loader = DataLoader(
    val_set, batch_size=cfg.batch_size, shuffle=False, pin_memory=pin_memory
  )

  model = DiffusionDenoiser(
    feature_dim=feature_dim,
    window_size=window_size,
    d_model=cfg.d_model,
    nhead=cfg.nhead,
    num_layers=cfg.num_layers,
    dropout=cfg.dropout,
    terrain_dim=terrain_dim,
  ).to(device)
  scheduler = DDPMScheduler(
    num_timesteps=cfg.num_timesteps,
  ).to(device)
  if terrain_dim:
    print(f"Denoiser (terrain-conditioned): {count_parameters(model):,} params")
  else:
    print(f"Denoiser: {count_parameters(model):,} params")

  optimizer = torch.optim.AdamW(
    model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
  )

  ema = _Ema(model, decay=cfg.ema_decay) if cfg.use_ema else None
  if ema is not None:
    print(f"EMA enabled (decay={cfg.ema_decay})")

  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  save_dir = Path(cfg.log_dir) / cfg.name / timestamp
  save_dir.mkdir(parents=True, exist_ok=True)

  wandb_run = None
  if cfg.use_wandb:
    import wandb

    wandb_run = wandb.init(project=cfg.wandb_project, name=cfg.name, config=vars(cfg))

  for epoch in range(cfg.num_epochs):
    model.train()
    epoch_loss = torch.zeros((), device=device)
    n_batches = 0

    for batch in train_loader:
      if is_conditional:
        x_0, terrain = batch
        terrain = terrain.to(device, non_blocking=pin_memory)
      else:
        x_0 = batch
        terrain = None
      x_0 = x_0.to(device, non_blocking=pin_memory)

      loss = _diffusion_loss(model, scheduler, x_0, cfg.num_noise_samples, terrain)

      optimizer.zero_grad()
      loss.backward()
      if cfg.max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
      optimizer.step()
      if ema is not None:
        ema.update(model)

      epoch_loss += loss.detach()
      n_batches += 1

    avg_loss = (epoch_loss / max(n_batches, 1)).item()

    if epoch % cfg.log_interval == 0:
      eval_model = ema.shadow if ema is not None else model
      val_loss = _validate(
        eval_model, scheduler, val_loader, device, pin_memory, cfg.num_noise_samples,
        has_terrain=is_conditional,
      )
      print(f"Epoch {epoch:4d} | train={avg_loss:.6f} | val={val_loss:.6f}")
      if wandb_run is not None:
        wandb_run.log({"epoch": epoch, "train/loss": avg_loss, "val/loss": val_loss})

    if epoch % cfg.save_interval == 0 or epoch == cfg.num_epochs - 1:
      ckpt_path = save_dir / f"checkpoint_{epoch:05d}.pt"
      _save_checkpoint(
        ckpt_path, epoch, model, dataset, feature_dim, cfg, optimizer, ema
      )
      if wandb_run is not None:
        wandb_run.save(str(ckpt_path), base_path=str(save_dir))

  final_path = save_dir / "pretrained.pt"
  _save_checkpoint(
    final_path, cfg.num_epochs, model, dataset, feature_dim, cfg, ema=ema
  )
  print(f"Saved final checkpoint to {final_path}")

  if wandb_run is not None:
    wandb_run.save(str(final_path), base_path=str(save_dir))
    wandb_run.finish()

  return final_path


@torch.no_grad()
def _validate(
  model: torch.nn.Module | DiffusionDenoiser,
  scheduler: DDPMScheduler,
  val_loader: DataLoader,
  device: torch.device,
  pin_memory: bool,
  num_noise_samples: int,
  has_terrain: bool = False,
) -> float:
  model.eval()
  total = torch.zeros((), device=device)
  n = 0
  for batch in val_loader:
    if has_terrain:
      x_0, terrain = batch
      terrain = terrain.to(device, non_blocking=pin_memory)
    else:
      x_0 = batch
      terrain = None
    x_0 = x_0.to(device, non_blocking=pin_memory)
    total += _diffusion_loss(model, scheduler, x_0, num_noise_samples, terrain)
    n += 1
  return (total / max(n, 1)).item()


if __name__ == "__main__":
  pretrain(tyro.cli(PretrainCfg))
