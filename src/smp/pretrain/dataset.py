"""Motion window datasets for diffusion pretraining (unconditional & conditional)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class MotionWindowDataset(Dataset[torch.Tensor]):
  """Loads pre-windowed NPZs produced by scripts/csv_to_npz.py.

  Normalization uses pre-computed q01/q99 quantiles (from
  ``scripts/compute_norm_stats.py``) to map features to [-1, 1].
  """

  def __init__(
    self,
    data_dir: str | Path,
    norm_stats_file: str | Path | None = None,
  ) -> None:
    npz_files = sorted(Path(data_dir).glob("*.npz"))
    if not npz_files:
      msg = f"No NPZ files found in {data_dir}"
      raise FileNotFoundError(msg)

    chunks: list[np.ndarray] = []
    expected_shape: tuple[int, int] | None = None
    for npz_file in npz_files:
      with np.load(npz_file, allow_pickle=False) as npz:
        windows = npz["windows"].astype(np.float32, copy=False)
      if windows.ndim != 3:
        msg = (
          f"{npz_file.name}: 'windows' has shape {windows.shape}, expected (N, W, S)"
        )
        raise ValueError(msg)
      if expected_shape is None:
        expected_shape = (int(windows.shape[1]), int(windows.shape[2]))
      elif (windows.shape[1], windows.shape[2]) != expected_shape:
        msg = (
          f"{npz_file.name}: shape {windows.shape} mismatches "
          f"first file's (*, {expected_shape[0]}, {expected_shape[1]})"
        )
        raise ValueError(msg)
      chunks.append(windows)

    assert expected_shape is not None
    self.window_size, self.feature_dim = expected_shape

    data = np.concatenate(chunks, axis=0)

    if norm_stats_file is not None:
      stats = np.load(norm_stats_file, allow_pickle=False)
      self.q_low = stats["q_low"].astype(np.float32)
      self.q_high = stats["q_high"].astype(np.float32)
    else:
      # Fallback: compute from data directly.
      flat = data.reshape(-1, self.feature_dim)
      self.q_low = np.percentile(flat, 1, axis=0).astype(np.float32)
      self.q_high = np.percentile(flat, 99, axis=0).astype(np.float32)
      span = self.q_high - self.q_low
      tiny = span < 1e-6
      if tiny.any():
        self.q_high[tiny] = self.q_low[tiny] + 1.0

    # Normalize
    data = 2.0 * (data - self.q_low) / (self.q_high - self.q_low) - 1.0

    self.windows = torch.from_numpy(data)

  def denormalize(self, x: torch.Tensor) -> torch.Tensor:
    q_low = torch.from_numpy(self.q_low).to(x.device, x.dtype)
    q_high = torch.from_numpy(self.q_high).to(x.device, x.dtype)
    return (x + 1.0) / 2.0 * (q_high - q_low) + q_low

  def __len__(self) -> int:
    return self.windows.shape[0]

  def __getitem__(self, idx: int) -> torch.Tensor:
    return self.windows[idx]


class ConditionalMotionDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
  """Loads windowed NPZs produced by ``scripts/height_map_to_npz.py``.

  Each NPZ contains ``motion_windows`` (N, W, F_m) and ``terrain`` (N, W, F_t).
  Separate per-feature q01/q99 normalization is applied to motion and terrain.
  ``__getitem__`` returns ``(motion, terrain)``.
  """

  def __init__(
    self,
    data_dir: str | Path,
    norm_stats_file: str | Path | None = None,
    terrain_norm_stats_file: str | Path | None = None,
  ) -> None:
    self.has_terrain = True
    npz_files = sorted(Path(data_dir).glob("*.npz"))
    if not npz_files:
      raise FileNotFoundError(f"No NPZ files found in {data_dir}")

    motion_chunks: list[np.ndarray] = []
    terrain_chunks: list[np.ndarray] = []
    expected_m_shape: tuple[int, int] | None = None
    expected_t_shape: tuple[int, int] | None = None

    for npz_file in npz_files:
      with np.load(npz_file, allow_pickle=False) as npz:
        mw = npz["motion_windows"].astype(np.float32, copy=False)
        tr = npz["terrain"].astype(np.float32, copy=False)

      if mw.ndim != 3:
        raise ValueError(f"{npz_file.name}: 'motion_windows' has shape {mw.shape}")
      if tr.ndim != 3:
        raise ValueError(f"{npz_file.name}: 'terrain' has shape {tr.shape}")

      if expected_m_shape is None:
        expected_m_shape = (int(mw.shape[1]), int(mw.shape[2]))
        expected_t_shape = (int(tr.shape[1]), int(tr.shape[2]))
      elif (mw.shape[1], mw.shape[2]) != expected_m_shape:
        raise ValueError(f"{npz_file.name}: motion shape {mw.shape} mismatches")
      elif (tr.shape[1], tr.shape[2]) != expected_t_shape:
        raise ValueError(f"{npz_file.name}: terrain shape {tr.shape} mismatches")
      elif mw.shape[0] != tr.shape[0]:
        raise ValueError(f"{npz_file.name}: motion ({mw.shape[0]}) & terrain ({tr.shape[0]}) count mismatch")

      motion_chunks.append(mw)
      terrain_chunks.append(tr)

    assert expected_m_shape is not None and expected_t_shape is not None
    self.window_size, self.feature_dim = expected_m_shape
    _, self.terrain_dim = expected_t_shape

    motion_data = np.concatenate(motion_chunks, axis=0)
    terrain_data = np.concatenate(terrain_chunks, axis=0)

    # ── Motion normalization ──
    if norm_stats_file is not None:
      stats = np.load(norm_stats_file, allow_pickle=False)
      self.q_low = stats["q_low"].astype(np.float32)
      self.q_high = stats["q_high"].astype(np.float32)
    else:
      flat = motion_data.reshape(-1, self.feature_dim)
      self.q_low = np.percentile(flat, 1, axis=0).astype(np.float32)
      self.q_high = np.percentile(flat, 99, axis=0).astype(np.float32)
      span = self.q_high - self.q_low
      tiny = span < 1e-6
      if tiny.any():
        self.q_high[tiny] = self.q_low[tiny] + 1.0

    motion_data = 2.0 * (motion_data - self.q_low) / (self.q_high - self.q_low) - 1.0
    self.windows = torch.from_numpy(motion_data)

    # ── Terrain normalization (per grid-cell, across all frames) ──
    if terrain_norm_stats_file is not None:
      tstats = np.load(terrain_norm_stats_file, allow_pickle=False)
      self.t_q_low = tstats["q_low"].astype(np.float32)
      self.t_q_high = tstats["q_high"].astype(np.float32)
    else:
      flat_t = terrain_data.reshape(-1, self.terrain_dim)
      self.t_q_low = np.percentile(flat_t, 1, axis=0).astype(np.float32)
      self.t_q_high = np.percentile(flat_t, 99, axis=0).astype(np.float32)
      tspan = self.t_q_high - self.t_q_low
      ttiny = tspan < 1e-6
      if ttiny.any():
        self.t_q_high[ttiny] = self.t_q_low[ttiny] + 1.0

    terrain_data = 2.0 * (terrain_data - self.t_q_low) / (self.t_q_high - self.t_q_low) - 1.0
    self.terrains = torch.from_numpy(terrain_data)

  def denormalize_motion(self, x: torch.Tensor) -> torch.Tensor:
    q_low = torch.from_numpy(self.q_low).to(x.device, x.dtype)
    q_high = torch.from_numpy(self.q_high).to(x.device, x.dtype)
    return (x + 1.0) / 2.0 * (q_high - q_low) + q_low

  def __len__(self) -> int:
    return self.windows.shape[0]

  def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    return self.windows[idx], self.terrains[idx]
