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
from mjlab.sensor.builtin_sensor import ObjRef
from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg
from mjlab.sensor.raycast_sensor import GridPatternCfg, RayCastSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg, TerrainGeneratorCfg
from mjlab.terrains.config import (
  flat,
  pyramid_stairs,
  random_spread_boxes,
  stepping_stones,
)
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from smp.rl.events import init_smp_state, reset_smp_buffer_flag
from smp.rl.mdp.terminations import (
  base_contact,
  root_height_below_env_origin_minimum,
  terrain_out_of_bounds,
)


def g1_smp_env_cfg(
  play: bool = False,
  terrain_type: str = "plane",
  terrain_conditioned: bool = False,
  terrain_dim: int | None = None,
) -> ManagerBasedRlEnvCfg:
  """Build the shared G1 + SMP env cfg (denoiser ckpt path set on
  ``init_smp_state`` below; override it from the task config).

  Args:
    play: If True, disable domain randomisation and set infinite episode length.
    terrain_type: ``"plane"`` (flat) or ``"generator"`` (procedural terrain).
    terrain_conditioned: If True, add a ``RayCastSensor`` for terrain sensing
      and pass ``terrain_dim`` to the SMP denoiser.
    terrain_dim: Height-map dimension (e.g. 187 for 17×11 grid). Only used
      when ``terrain_conditioned=True``.
  """

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
        "ckpt_path": "logs/pretrain/lafan_g1_walk_with_terrain/20260706_104243/pretrained.pt",
        "compile_model": True,
        "compile_mode": "max-autotune",
        "terrain_dim": terrain_dim,
      },
    ),
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.1, 0.1),
          "y": (-0.1, 0.1),
          "yaw": (-0.1, 0.1),
        },
        "velocity_range": {
          "x": (-0.2, 0.2),
          "y": (-0.2, 0.2),
          "z": (-0.2, 0.2),
          "roll": (-0.2, 0.2),
          "pitch": (-0.2, 0.2),
          "yaw": (-0.2, 0.2),
        },
      },
    ),
    "reset_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.15, 0.15),
        "velocity_range": (0.0, 0.0),
      },
    ),
    "reset_smp_buffer": EventTermCfg(
      func=reset_smp_buffer_flag,
      mode="reset",
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
        "shared_random": True,
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

  base_contact_cfg = ContactSensorCfg(
    name="base_contact",
    primary=ContactMatch(
      mode="subtree", pattern="torso_link", entity="robot"
    ),
    secondary=None,
    fields=("force",),
    reduce="none",
    num_slots=1,
    history_length=3,
  )

  sensors: tuple = (self_collision_cfg, base_contact_cfg)
  if terrain_conditioned:
    terrain_scan_cfg = RayCastSensorCfg(
      name="terrain_scan",
      frame=ObjRef(type="body", name="torso_link", entity="robot"),
      pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
      ray_alignment="yaw",
      max_distance=30.0,
      exclude_parent_body=True,
      include_geom_groups=(0,),
    )
    sensors = (self_collision_cfg, base_contact_cfg, terrain_scan_cfg)

  # --- Terminations --------------------------------------------------------
  terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
    "terrain_out_bound": TerminationTermCfg(
      func=terrain_out_of_bounds,
      time_out=True,
      params={"distance_buffer": 2.0},
    ),
    "base_contact": TerminationTermCfg(
      func=base_contact,
      params={
        "sensor_name": base_contact_cfg.name,
        "threshold": 1.0,
      },
    ),
    "bad_orientation": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": 1.0},
    ),
    "root_height": TerminationTermCfg(
      func=root_height_below_env_origin_minimum,
      params={"minimum_height": 0.5},
    ),
  }

  if terrain_type == "generator":
    terrain_cfg = TerrainEntityCfg(
      terrain_type="generator",
      terrain_generator=TerrainGeneratorCfg(
        size=(8.0, 8.0),
        num_rows=1,
        num_cols=1,
        sub_terrains={
          "stairs": pyramid_stairs(
            proportion=0.30,
            step_height_range=(0.05, 0.15),
            step_width=0.3,
            platform_width=2.0,
          ),
          "boxes": random_spread_boxes(
            proportion=0.30,
            num_boxes=60,
            box_height_range=(0.05, 0.25),
          ),
          "stones": stepping_stones(
            proportion=0.25,
            stone_height=0.15,
            stone_height_variation=0.10,
            floor_depth=0.0,
          ),
          "flat": flat(proportion=0.15),
        },
      ),
    )
  else:
    terrain_cfg = TerrainEntityCfg(terrain_type="plane")

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=terrain_cfg,
      entities={"robot": get_g1_robot_cfg()},
      num_envs=1,
      extent=2.0,
      sensors=sensors,
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
    cfg.events["init_smp_state"].params["compile_model"] = False
    cfg.terminations.pop("root_height", None)

  return cfg
