"""SMP terrain-steering task — registers ``Smp-Steering-G1-Terrain`` on import."""

from mjlab.tasks.registry import register_mjlab_task

from smp.rl.rl_cfg import unitree_g1_smp_ppo_runner_cfg
from smp.rl.tasks.terrain_steering.terrain_steering_env_cfg import (
  g1_terrain_steering_smp_env_cfg,
)

_terrain_steering_rl = unitree_g1_smp_ppo_runner_cfg()
_terrain_steering_rl.experiment_name = "smp_terrain_steering_g1"
_terrain_steering_rl.run_name = "smp_terrain_steering_g1"

register_mjlab_task(
  task_id="Smp-Steering-G1-Terrain",
  env_cfg=g1_terrain_steering_smp_env_cfg(play=False),
  play_env_cfg=g1_terrain_steering_smp_env_cfg(play=True),
  rl_cfg=_terrain_steering_rl,
)

__all__ = [
  "g1_terrain_steering_smp_env_cfg",
]
