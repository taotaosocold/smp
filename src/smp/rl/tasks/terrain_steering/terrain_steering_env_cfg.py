"""G1 steering task with SMP guidance + terrain conditioning.

Same steering task as ``Smp-Steering-G1`` but uses a terrain-conditioned
diffusion model and a non-flat terrain (Perlin noise by default).
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from smp.rl.env_cfg import g1_smp_env_cfg
from smp.rl.rewards import task_smp_product
from smp.rl.tasks.steering import mdp


def g1_terrain_steering_smp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build the G1 steering env cfg with terrain-conditioned SMP guidance."""
  cfg = g1_smp_env_cfg(
    play=play,
    terrain_type="generator",
    terrain_conditioned=True,
    terrain_dim=187,
  )

  # --- Commands ------------------------------------------------------------
  cfg.commands["steering"] = mdp.SteeringCommandCfg(
    entity_name="robot",
    resampling_time_range=(3.0, 8.0),
    rand_tar_dir=True,
    rand_face_dir=True,
    tar_speed_min=0.5,
    tar_speed_max=2.0,
    debug_vis=True,
  )

  # --- Observations --------------------------------------------------------
  command_obs = ObservationTermCfg(
    func=mdp.generated_commands,
    params={"command_name": "steering"},
  )
  cfg.observations["actor"].terms["command"] = command_obs
  cfg.observations["critic"].terms["command"] = command_obs

  # --- Rewards -------------------------------------------------------------
  cfg.rewards["task_smp_product"] = RewardTermCfg(
    func=task_smp_product,
    weight=1.0,
    params={
      "task_terms": (
        (
          mdp.steering_target_velocity,
          0.5,
          {"command_name": "steering", "vel_err_scale": 1.0},
        ),
        (mdp.steering_face_direction, 0.5, {"command_name": "steering"}),
      ),
    },
  )

  # --- Events --------------------------------------------------------------
  cfg.events["init_smp_state"].params["ckpt_path"] = (
    "logs/pretrain/lafan_g1_walk_with_terrain/20260706_104243/pretrained.pt"
  )

  # --- Terminations --------------------------------------------------------
  # ``root_height`` (root_height_below_env_origin_minimum) is already in the
  # base config with minimum_height=0.5, matching parkour.

  return cfg
