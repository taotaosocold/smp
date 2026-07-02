"""Motion window dataset for diffusion pretraining."""

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
    # 获得所有npz文件
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
    # 将所有数据归一化
    data = 2.0 * (data - self.q_low) / (self.q_high - self.q_low) - 1.0

    self.windows = torch.from_numpy(data)
  # 将数据反归一化
  def denormalize(self, x: torch.Tensor) -> torch.Tensor:
    q_low = torch.from_numpy(self.q_low).to(x.device, x.dtype)
    q_high = torch.from_numpy(self.q_high).to(x.device, x.dtype)
    return (x + 1.0) / 2.0 * (q_high - q_low) + q_low

  def __len__(self) -> int:
    return self.windows.shape[0]

  def __getitem__(self, idx: int) -> torch.Tensor:
    return self.windows[idx]
