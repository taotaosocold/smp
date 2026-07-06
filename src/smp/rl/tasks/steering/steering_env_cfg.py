"""G1 steering task with SMP guidance.

Each env gets a target xy direction + speed and a target facing direction,
periodically resampled.  Reward = vel-track + face-align, on top of the SMP
guidance reward inherited from ``g1_smp_env_cfg``.
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


def g1_steering_smp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build the G1 steering env cfg with SMP guidance."""
  # 直接先获得g1_smp_env_cfg实例对象，然后在这基础上去修改或添加观测、终止、奖励等等
  cfg = g1_smp_env_cfg(play=play)

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
  # task = 0.7·velocity tracking + 0.3·face alignment, gated by SMP.
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
    "datasets/pretrain_ckpt/pretrained_lafan_run.pt"
  )

  # --- Terminations --------------------------------------------------------
  cfg.terminations["base_too_low"] = TerminationTermCfg(
    func=mdp.root_height_below_minimum,
    params={
      "minimum_height": 0.3,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )

  return cfg
