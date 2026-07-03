"""Compute per-feature q01/q99 quantiles from all NPZ window files.

Scans all ``*.npz`` files in the input directory, concatenates all windows,
and computes the 1st and 99th percentile per feature dimension.

Auto-detects unconditional (``windows`` key) vs conditional (``motion_windows``
+ ``terrain`` keys) formats.  For conditional data, also computes terrain stats
and saves them to a separate ``.npz`` file (``<output>_terrain.npz``).

Usage:
  # Unconditional
  python scripts/compute_norm_stats.py --input-dir datasets/npz --output datasets/norm_stats.npz

  # Conditional (also writes datasets/norm_stats_terrain.npz)
  python scripts/compute_norm_stats.py --input-dir datasets/conditional_npz --output datasets/norm_stats.npz
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
  """Output path for motion quantile stats."""
  q_low: float = 0.01
  """Lower quantile (default 1st percentile)."""
  q_high: float = 0.99
  """Upper quantile (default 99th percentile)."""


def _compute_quantiles(frames: np.ndarray, q_low: float, q_high: float) -> tuple[np.ndarray, np.ndarray]:
  ql = np.percentile(frames, q_low * 100, axis=0).astype(np.float32)
  qh = np.percentile(frames, q_high * 100, axis=0).astype(np.float32)
  span = qh - ql
  tiny = span < 1e-6
  if tiny.any():
    print(f"  WARNING: {tiny.sum()} features have near-zero range, using fallback span=1.0")
    qh[tiny] = ql[tiny] + 1.0
  return ql, qh


def main(cfg: Cfg) -> None:
  in_dir = Path(cfg.input_dir)
  npz_files = sorted(in_dir.glob("*.npz"))
  if not npz_files:
    raise FileNotFoundError(f"No NPZ files in {in_dir}")

  # Detect format
  _probe = np.load(str(npz_files[0]), allow_pickle=False)
  is_conditional = "terrain" in _probe or "motion_windows" in _probe

  if is_conditional:
    m_key = "motion_windows"
    t_key = "terrain"
    print(f"[INFO] Detected conditional data format (keys: {m_key}, {t_key})")
  else:
    m_key = "windows"
    print(f"[INFO] Detected unconditional data format (key: {m_key})")

  # Load motion data
  motion_chunks: list[np.ndarray] = []
  terrain_chunks: list[np.ndarray] = []

  for f in npz_files:
    with np.load(f, allow_pickle=False) as data:
      mw = data[m_key]
      motion_chunks.append(mw.reshape(-1, mw.shape[-1]))
      if is_conditional:
        tr = data[t_key]
        terrain_chunks.append(tr.reshape(-1, tr.shape[-1]))
      print(f"  {f.name}: {mw.shape[0]} windows, {mw.shape[-1]} motion dims", end="")
      if is_conditional:
        print(f", {tr.shape[-1]} terrain dims")
      else:
        print()

  all_motion = np.concatenate(motion_chunks, axis=0).astype(np.float64)
  print(f"\nTotal motion frames: {all_motion.shape[0]}, feature dim: {all_motion.shape[1]}")

  q_low, q_high = _compute_quantiles(all_motion, cfg.q_low, cfg.q_high)
  out_path = Path(cfg.output)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez(out_path, q_low=q_low, q_high=q_high)
  print(f"\nSaved {out_path}: q_low/q_high shape ({q_low.shape[0]},)")
  print(f"  q_low  range: [{q_low.min():.4f}, {q_low.max():.4f}]")
  print(f"  q_high range: [{q_high.min():.4f}, {q_high.max():.4f}]")

  if is_conditional:
    all_terrain = np.concatenate(terrain_chunks, axis=0).astype(np.float64)
    print(f"\nTotal terrain frames: {all_terrain.shape[0]}, feature dim: {all_terrain.shape[1]}")
    t_q_low, t_q_high = _compute_quantiles(all_terrain, cfg.q_low, cfg.q_high)

    t_out_path = out_path.parent / f"{out_path.stem}_terrain.npz"
    np.savez(t_out_path, q_low=t_q_low, q_high=t_q_high)
    print(f"\nSaved {t_out_path}: q_low/q_high shape ({t_q_low.shape[0]},)")
    print(f"  q_low  range: [{t_q_low.min():.4f}, {t_q_low.max():.4f}]")
    print(f"  q_high range: [{t_q_high.min():.4f}, {t_q_high.max():.4f}]")


if __name__ == "__main__":
  main(tyro.cli(Cfg))
