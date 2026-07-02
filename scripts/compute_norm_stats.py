"""Compute per-feature q01/q99 quantiles from all NPZ window files.

Scans all ``*.npz`` files in the input directory, concatenates all windows,
and computes the 1st and 99th percentile per feature dimension. Saves the
result as a small ``.npz`` file for use by the training loop and RL reward.

Usage:
  uv run scripts/compute_norm_stats.py --input-dir datasets/npz --output datasets/norm_stats.npz
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro


@dataclass
class Cfg:
  input_dir: str = "datasets/npz"
  """Directory containing windowed NPZ files."""
  output: str = "datasets/norm_stats.npz"
  """Output path for the quantile stats file."""
  q_low: float = 0.01
  """Lower quantile (default 1st percentile)."""
  q_high: float = 0.99
  """Upper quantile (default 99th percentile)."""


def main(cfg: Cfg) -> None:
  # 加载运动数据集
  in_dir = Path(cfg.input_dir)
  npz_files = sorted(in_dir.glob("*.npz"))
  if not npz_files:
    raise FileNotFoundError(f"No NPZ files in {in_dir}")

  chunks: list[np.ndarray] = []
  for f in npz_files:
    with np.load(f, allow_pickle=False) as data:
      windows = data["windows"]  # (N, W, F)
      # Flatten window dim → treat each frame independently.
      # 将 (N, W, F) 变形为 (N*W, F)，这样每个时间步（每一帧）都被当作一个独立的样本
      chunks.append(windows.reshape(-1, windows.shape[-1]))
      print(f"  {f.name}: {windows.shape[0]} windows, {windows.shape[-1]} features")
  # 把所有文件的帧拼成一个巨大的矩阵 all_frames，形状 (总帧数, F)
  all_frames = np.concatenate(chunks, axis=0).astype(np.float64)
  print(f"\nTotal frames: {all_frames.shape[0]}, feature dim: {all_frames.shape[1]}")
  # 对每一列（即每一个特征维度）分别计算全数据集的 1% 和 99% 分位数，也就是每个特征排序，然后假设有1000个数据，那就是获得第10个和第990个个数据的值
  # 得到两个长度为 F 的向量 q_low 和 q_high
  q_low = np.percentile(all_frames, cfg.q_low * 100, axis=0).astype(np.float32)
  q_high = np.percentile(all_frames, cfg.q_high * 100, axis=0).astype(np.float32)

  # Prevent zero-range (constant features) → set a minimum span.
  # 如果某个特征的变化范围极小（接近常数），就用 q_low + 1.0 替代 q_high，保证后续归一化时分母不为零
  span = q_high - q_low
  tiny = span < 1e-6
  if tiny.any():
    print(
      f"  WARNING: {tiny.sum()} features have near-zero range, using fallback span=1.0"
    )
    q_high[tiny] = q_low[tiny] + 1.0
  # 最终将 q_low 和 q_high 存入 norm_stats.npz
  out_path = Path(cfg.output)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez(out_path, q_low=q_low, q_high=q_high)
  print(f"\nSaved {out_path}: q_low/q_high shape ({q_low.shape[0]},)")
  print(f"  q_low  range: [{q_low.min():.4f}, {q_low.max():.4f}]")
  print(f"  q_high range: [{q_high.min():.4f}, {q_high.max():.4f}]")


if __name__ == "__main__":
  main(tyro.cli(Cfg))
