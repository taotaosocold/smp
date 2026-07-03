"""Diffusion model pretraining loop."""
"""
我来大概说一下这个扩散模型的思路，首先一个运动序列假设(N,F)其中N是帧数,F是观测,然后我们。我们只取其中的部分观测,关节、根位置3、6D根朝向6、根线角速度6、末端
位置头双手双脚5*3。然后我们弄一个窗口W,每滑动一帧我们获得一个窗口，也就是这个数据会变成(N-W+1,W,F)其中对于每一个窗口其根位置xy是相对这个窗口的最后一帧的xy
位置,朝向也是这个yaw也是相对的,然后末端位置也是相对于根位置的相对位置。然后每一个epoch扔给扩散模型(B,W,F),这个扩散模型是两层的自注意力transformer架构。
然后这个数据会直接复制K份为(B*K,W,F)，然后我们采集的噪声是(B*K)也就是对于每一个窗口我们的噪声强度是一致的,但是我们在训练的时候会给同一个窗口复制K份来给予
K种不同的噪声强度。然后加上位置编码模型输出预测噪声,然后做损失就这样。详细的模型架构可以看model.py
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

from smp.pretrain.dataset import MotionWindowDataset
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
) -> torch.Tensor:
  """DDPM ε-prediction L1 loss with multiple noise samples per data point.

  Each sample in the batch is paired with ``num_noise_samples`` random
  (timestep, noise) draws, giving lower-variance gradients than a single
  draw without the cost of exhausting all T timesteps.
  """
  B = x_0.shape[0]
  # 每个窗口的噪声采样数
  K = num_noise_samples
  # (B, W, F) → (B*K, W, F)
  # 把窗口复制K份，也就是同一个干净数据对应10个不同的(t, \varepsilon)组合，loss求平均后梯度方差更小
  x_0_exp = x_0[:, None].expand(B, K, *x_0.shape[1:]).reshape(B * K, *x_0.shape[1:])
  # 随机采样10240个时间步，每个来自Uniform(0, 49)。不同数据点、不同K之间独立采样，保证梯度多样性
  # 这个t的shape为(B*K)
  t = scheduler.sample_timesteps(B * K, x_0.device)
  # 生成纯高斯分布的噪声10240个，即(10240, 10, 59)
  noise = torch.randn_like(x_0_exp)
  # 给干净的数据添加不同强度的噪声
  x_t = scheduler.add_noise(x_0_exp, noise, t)
  # 模型预测的噪声和实际噪声做L1损失
  return F.l1_loss(model(x_t, t), noise)


def _save_checkpoint(
  path: Path,
  epoch: int,
  model: DiffusionDenoiser,
  dataset: MotionWindowDataset,
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
  # 这个dataset有__getitem__(idx)函数，每次去拜访索引后获得的是(window_size, feature_dim)的归一化后的数据
  # 里面也含有反归一化的函数
  dataset = MotionWindowDataset(cfg.data_dir, norm_stats_file=cfg.norm_stats_file)
  feature_dim = dataset.feature_dim
  window_size = dataset.window_size
  # n_train 和 n_val 是用来划分训练集和验证集的样本数量
  n_train = int(len(dataset) * cfg.train_split)
  n_val = len(dataset) - n_train
  print(
    f"Dataset: {len(dataset)} windows, n_train={n_train}, n_val={n_val}, "
    f"feature_dim={feature_dim}, window_size={window_size}"
  )
  # 把原始的 MotionWindowDataset 按 [n_train, n_val] 的长度随机切分成两个新的 Subset 对象
  train_set, val_set = random_split(dataset, [n_train, n_val])
  pin_memory = device.type == "cuda"
  # 为训练集创建一个数据加载器， batch_size: 每个批次返回多少个窗口
  # shuffle=True: 每个 epoch 都会随机打乱训练样本的顺序，这是训练的标准做法，防止模型记忆数据顺序
  train_loader = DataLoader(
    train_set,
    batch_size=cfg.batch_size,
    shuffle=True,
    pin_memory=pin_memory,
  )
  # 测试集同理
  val_loader = DataLoader(
    val_set, batch_size=cfg.batch_size, shuffle=False, pin_memory=pin_memory
  )
  # 获得扩散模型
  model = DiffusionDenoiser(
    feature_dim=feature_dim,
    window_size=window_size,
    d_model=cfg.d_model,
    nhead=cfg.nhead,
    num_layers=cfg.num_layers,
    dropout=cfg.dropout,
  ).to(device)
  # DDPM 噪声调度器，其实就是设置了那个噪声的步数以及每一个等级的噪声强度
  scheduler = DDPMScheduler(
    num_timesteps=cfg.num_timesteps,
  ).to(device)
  print(f"Denoiser: {count_parameters(model):,} params")
  # 优化器
  optimizer = torch.optim.AdamW(
    model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
  )
  # 如果启用了 EMA（cfg.use_ema=True），会创建一个 _Ema 实例，维护模型参数的影子副本
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
  # 开始训练
  for epoch in range(cfg.num_epochs):
    model.train()
    epoch_loss = torch.zeros((), device=device)
    n_batches = 0
    # 每一次获得一个批次的训练集。形状为（B，10，59）10是帧数，59是观测维度，B是批次默认为1024
    for batch in train_loader:
      # 干净的数据转换到gpu上
      x_0 = batch.to(device, non_blocking=pin_memory)
      # 经过扩散模型获得损失
      loss = _diffusion_loss(model, scheduler, x_0, cfg.num_noise_samples)
      # 反向传播
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
        eval_model, scheduler, val_loader, device, pin_memory, cfg.num_noise_samples
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
  val_loader: DataLoader[torch.Tensor],
  device: torch.device,
  pin_memory: bool,
  num_noise_samples: int,
) -> float:
  model.eval()
  total = torch.zeros((), device=device)
  n = 0
  for batch in val_loader:
    x_0 = batch.to(device, non_blocking=pin_memory)
    total += _diffusion_loss(model, scheduler, x_0, num_noise_samples)
    n += 1
  return (total / max(n, 1)).item()


if __name__ == "__main__":
  pretrain(tyro.cli(PretrainCfg))
