"""G1 forward task — fixed +x heading, variable target speed.

A specialization of the steering task with the world-frame target and face
directions pinned to ``+x``.
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


def g1_forward_smp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build the G1 forward env cfg with SMP guidance."""
  cfg = g1_smp_env_cfg(play=play)

  # --- Commands ------------------------------------------------------------
  cfg.commands["steering"] = mdp.SteeringCommandCfg(
    entity_name="robot",
    resampling_time_range=(3.0, 8.0),
    rand_tar_dir=False,
    rand_face_dir=False,
    tar_speed_min=0.5,
    tar_speed_max=5.0,
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
  # task = velocity tracking, gated by SMP.
  cfg.rewards["task_smp_product"] = RewardTermCfg(
    func=task_smp_product,
    weight=1.0,
    params={
      "task_terms": (
        (
          mdp.steering_target_velocity,
          1.0,
          {"command_name": "steering", "vel_err_scale": 0.5},
        ),
      ),
    },
  )

  # --- Events --------------------------------------------------------------
  cfg.events["init_smp_state"].params["ckpt_path"] = (
    "datasets/pretrain_ckpt/pretrained_loco.pt"
  )

  # --- Terminations --------------------------------------------------------
  # ``root_height`` (root_height_below_env_origin_minimum) is already in the
  # base config with minimum_height=0.5, matching parkour.

  return cfg
