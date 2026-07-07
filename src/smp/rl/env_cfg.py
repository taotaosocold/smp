"""Shared G1 + SMP guidance env config.

The SMP feature buffer and frozen denoiser are attached to the stock
``ManagerBasedRlEnv`` via the startup/reset events in ``smp.rl.events``.
Per-task configs extend this with task-specific commands/observations/rewards.
"""

from __future__ import annotations

from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg, mdp
from mjlab.envs.mdp import dr, time_out
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity.mdp import illegal_contact
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from smp.rl.events import (
  gsi_refresh,
  gsi_reset,
  init_smp_state,
)


def g1_smp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build the shared G1 + SMP env cfg (denoiser ckpt path set on
  ``init_smp_state`` below; override it from the task config)."""

  # --- Observations --------------------------------------------------------
  actor_terms = {
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  critic_terms = {
    **actor_terms,
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
      history_length=10,
    ),
  }

  # --- Actions --------------------------------------------------------
  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=G1_ACTION_SCALE,
      use_default_offset=True,
    )
  }

  # --- Commands ------------------------------------------------------------
  commands: dict[str, CommandTermCfg] = {}

  # --- Events --------------------------------------------------------------
  events = {
    "init_smp_state": EventTermCfg(
      func=init_smp_state,
      mode="startup",
      params={
        "ckpt_path": "logs/pretrain/pretrained.pt",
        "gsi_buffer_size": 4096,
        "gsi_batch_size": 1024,
        "compile_model": True,
        "compile_mode": "max-autotune",
      },
    ),
    "gsi_reset": EventTermCfg(func=gsi_reset, mode="reset", params={}),
    "gsi_refresh": EventTermCfg(
      func=gsi_refresh,
      mode="step",
      params={"num_samples": 1024, "step_interval": 2400},
    ),
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(1.0, 3.0),
      params={
        "velocity_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (-0.4, 0.4),
          "roll": (-0.52, 0.52),
          "pitch": (-0.52, 0.52),
          "yaw": (-0.78, 0.78),
        },
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot", geom_names=r"^(left|right)_foot[1-7]_collision$"
        ),
        "operation": "abs",
        "ranges": (0.3, 1.2),
        "shared_random": True,  # All foot geoms share the same friction.
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.015, 0.015),
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
        "operation": "add",
        "ranges": {
          0: (-0.025, 0.025),
          1: (-0.025, 0.025),
          2: (-0.03, 0.03),
        },
      },
    ),
  }

  # --- Rewards -------------------------------------------------------------
  # Empty by design: each task adds its own ``task_smp_product`` term
  # (task reward × SMP guidance).
  rewards: dict[str, RewardTermCfg] = {}

  # --- Sensors -------------------------------------------------------------
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )

  # --- Terminations --------------------------------------------------------
  terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
    "self_collision": TerminationTermCfg(
      func=illegal_contact,
      params={"sensor_name": self_collision_cfg.name},
    ),
  }

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": get_g1_robot_cfg()},
      num_envs=1,
      extent=2.0,
      sensors=(self_collision_cfg,),
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="torso_link",
      distance=3.0,
      elevation=-5.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=35,
      njmax=1500,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    episode_length_s=20.0,
  )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.events.pop("push_robot", None)
    cfg.events.pop("gsi_refresh", None)
    cfg.events["init_smp_state"].params["compile_model"] = False
    cfg.events["init_smp_state"].params["gsi_buffer_size"] = 1024

  return cfg
