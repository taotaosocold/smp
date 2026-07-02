# SMP — Score-Matching Motion Priors (reproduction)

A reproduction of **SMP: Reusable Score-Matching Motion Priors for Physics-Based
Character Control** (Mu et al., 2025) on the **Unitree G1** humanoid — the
original MimicKit implementation does not include a G1 setup, so this repo ports
the method to G1 end to end (motion features, priors, tasks, and rewards).

A small diffusion model (DDPM) is pretrained on motion windows; its **frozen
score** is then reused as an SDS-style *guidance reward* during PPO, so a policy
learns naturalistic motion for a downstream task without any per-task motion
clip or adversarial discriminator.

This is a reproduction for a course project. It re-implements the SMP idea on top
of [**mjlab**](https://github.com/mujocolab/mjlab) (the `ManagerBasedRlEnv` and
`mjlab.scripts.train` / `play` entrypoints are reused). The original method and
reference implementation are:

- **Paper:** SMP, Mu et al. 2025 — [arXiv:2512.03028](https://arxiv.org/abs/2512.03028) · [project page](https://yxmu.foo/smp-page/)
- **Original code:** [`xbpeng/MimicKit`](https://github.com/xbpeng/MimicKit) (see `docs/README_SMP.md`)

> The main intentional divergence from the original is the reward composition —
> see [Reward design](#reward-design-task--smp) below.

## Provided pretrained priors

To let you skip pretraining and run RL directly, **three pretrained diffusion
priors are shipped** in `datasets/pretrain_ckpt/`. Each task's env config already
points its `init_smp_state` event at the right one, so no setup is needed:

| Checkpoint                       | Trained on            | Used by                          |
| -------------------------------- | --------------------- | -------------------------------- |
| `pretrained_loco.pt`             | walk / jog / run      | `Smp-Forward-G1`                 |
| `pretrained_lafan_run.pt`        | LAFAN run subset      | `Smp-Steering-G1`, `Smp-Location-G1` |
| `pretrained_getup_f2s2.pt`       | get-up (fall→stand)   | `Smp-Getup-G1`                   |

## Setup

Use [conda](https://docs.conda.io/) (or [mamba](https://mamba.readthedocs.io/))
to create the environment:

```bash
conda create -n smp python=3.12
conda install -n smp -c pytorch -c conda-forge pytorch numpy matplotlib wandb tyro scikit-learn ruff
conda activate smp
pip install "mjlab @ git+https://github.com/mujocolab/mjlab.git@731fa27"
```

## Pipeline

1. **Data processing** (CSV → windowed NPZ → normalization stats) — _TODO (docs pending)._
2. **Diffusion pretraining** (DDPM ε-predictor on motion windows) — _TODO (docs pending)._
   You can skip this entirely using the [shipped checkpoints](#provided-pretrained-priors).
3. **RL** (PPO with the frozen prior as a guidance reward) — documented below.

---
## pretrain
```
python scripts/compute_norm_stats.py --input-dir datasets/npz --output datasets/norm_stats.npz
python scripts/pretrain.py --data-dir /path/npz_folder --norm-stats-file datasets/norm_stats.npz --name pretrain --no-use-wandb
python scripts/generate_viz.py --ckpt-path logs/pretrain/pretrain/20260529_190236/checkpoint_01999.pt
```


## RL

Four downstream tasks are registered with `mjlab.tasks.registry` (importing
`smp.rl.tasks` self-registers them):

| Task              | Demo | Description                              |
| ----------------- | :--: | ---------------------------------------- |
| `Smp-Forward-G1`  | <img src="https://raw.githubusercontent.com/SUZ-tsinghua/smp/assets/forward.gif" width="200"/> | walk / jog / run at a commanded `+x` speed |
| `Smp-Steering-G1` | <img src="https://raw.githubusercontent.com/SUZ-tsinghua/smp/assets/steering.gif" width="200"/> | track a commanded velocity + facing direction |
| `Smp-Location-G1` | <img src="https://raw.githubusercontent.com/SUZ-tsinghua/smp/assets/location.gif" width="200"/> | walk to a world-frame xy goal |
| `Smp-Getup-G1`    | <img src="https://raw.githubusercontent.com/SUZ-tsinghua/smp/assets/getup.gif" width="200"/> | stand up from a fallen pose |

### Train / play

```bash
# Train (checkpoints land under logs/)
python scripts/train.py Smp-Forward-G1 --env.scene.num-envs=4096

# Play a trained policy from a W&B run
python scripts/play.py Smp-Forward-G1 --wandb-run-path <org>/<project>/<run> --num-envs 4
```

Swap the task id for any of the four. Because the priors are shipped and already
wired into each env config, no editing is required before training.

### Reward design: `task × SMP`

Every task uses a single **multiplicative** reward term, `task_smp_product`:

```
r  =  ( Σᵢ wᵢ · taskᵢ(s) )  ×  r_smp(s)
```

where `r_smp = exp(−wₛ/|K| · Σ_{i∈K} ‖ε̂_i − ε_i‖²)` is the SDS guidance reward
(the frozen denoiser's ε-prediction error at a fixed set of diffusion timesteps
`K`, per-timestep normalized).

This is the **key divergence from the original SMP / MimicKit**, which combines
the two **additively** and balances them with separate weights
(`task_reward_weight`, `smp_reward_weight`):

```
# original (additive):     r = task_reward_weight · task  +  smp_reward_weight · r_smp
# here     (multiplicative): r = task · r_smp
```

We want the policy to **complete the task _while_ keeping the SMP reward high** —
which is exactly what a product expresses: it is large only when *both* factors
are large, and collapses toward 0 if *either* drops. This makes reward tuning
**easier and more robust**:

- **No task-vs-prior weight to balance.** The additive form needs a
  `task_reward_weight : smp_reward_weight` ratio whose sweet spot shifts per task
  (and per training stage); the product removes that knob entirely.
- **Neither term can be farmed alone.** Additively, a policy can max one term and
  ignore the other — e.g. stand still looking natural (high prior, no task
  progress) or lunge at the goal off-manifold (high task, low prior). With the
  product both failure modes score ≈ 0, so the only way to earn reward is to do
  the task *and* stay on the motion manifold.

Per-task `taskᵢ` components (each weighted, summed, then gated by `r_smp`):

- **Forward** — velocity tracking only: `exp(−s·‖v_cmd − v_xy‖²)`, zeroed when the
  velocity projects backwards onto the target direction. Fixed `+x` heading,
  commanded speed 0.5–5 m/s.
- **Steering** — `0.5·` velocity tracking `+ 0.5·` facing alignment
  `max(face_dir · heading, 0)`; randomized target direction + facing, speed 0.5–2 m/s.
- **Location** — position tracking only: `exp(−s·‖xy_goal − xy_robot‖)` toward a
  periodically resampled world-frame goal (uses `ws=4`).
- **Get-up** — `0.7·` upward head velocity `+ 0.3·` head-height tracking, each
  `exp(−s·max(target − ·, 0)²)`, from a fallen GSI start.

### Generative State Initialization (GSI)

On every reset, an init state is drawn from a pool of windows pre-sampled from the
frozen prior; its last frame seeds the sim state and the whole window primes the
online feature buffer, so `r_smp` is meaningful from step 0. Each env is reset to
its own scene origin while the feature buffer is kept **env-origin-relative**, so
the guidance reward is invariant to where the env sits in the world grid.

### Motion features

The guidance reward scores a rolling window of motion features rebuilt online by
`smp.rl.utils.MotionFeatureBuffer`, matching the pretraining layout (59-dim/frame
for G1), anchored to the last frame's yaw-only local frame:

```
[root_pos(3), root_rot(6), joint_pos(29), ee_pos(15), root_lin_vel(3), root_ang_vel(3)]
```

## Citation & acknowledgements

This repository reproduces SMP; please cite the original work and credit the
reference implementation:

- **SMP** — Mu et al., *Reusable Score-Matching Motion Priors for Physics-Based Character Control*, 2025. [arXiv:2512.03028](https://arxiv.org/abs/2512.03028)
- **MimicKit** — the original SMP implementation: <https://github.com/xbpeng/MimicKit>
- **mjlab** — RL environment backbone: <https://github.com/mujocolab/mjlab>
