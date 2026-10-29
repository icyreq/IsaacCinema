import math
from pathlib import Path

from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.sim import CuboidCfg, PreviewSurfaceCfg
from isaaclab.managers.command_manager import CommandTermCfg
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg

from .commands import TaskAwareVelocityCommand, SphericalInterpolationCommand
from .hard_goal_filters import ManipHardGoalBounds


_REPO_ROOT = Path(__file__).resolve().parents[3]
REACHABLE_TCP_SET_DIR = _REPO_ROOT / "assets"


@configclass
class TaskAwareVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = TaskAwareVelocityCommand
    probs: list[float] = [0.25, 0.75]
    # forward-backward curriculum
    enable_backward_motion: bool = False
    forward_prob: float = 0.5
    forward_lin_vel_x: tuple[float, float] = (0.0, 1.0)
    backward_lin_vel_x: tuple[float, float] = (-1.0, 0.0)

    random_zero_command_event_rate: float = 0.0
    """Per-environment emergency-brake event rate for loco-manip commands, in Hz."""
    random_zero_command_duration_range: tuple[float, float] = (0.0, 0.0)
    """Uniform duration range, in seconds, for each short zero-command brake event."""


@configclass
class SphericalInterpolationCommandCfg(CommandTermCfg):
    """Configuration for spherical interpolation end-effector commands."""

    class_type: type = SphericalInterpolationCommand
    asset_name: str = "robot"
    ee_body_name: str = "eef_link"
    hard_goal_bounds_type: type = ManipHardGoalBounds

    # position sampling in polar coordinates
    radius_range: tuple[float, float] = (0.2, 0.7)
    pitch_range: tuple[float, float] = (-2 * math.pi / 5, 2 * math.pi / 5)
    yaw_range: tuple[float, float] = (-3 * math.pi / 5, 3 * math.pi / 5)
    # orientation sampling (euler angles relative to base frame)
    ee_roll_range: tuple[float, float] = (-math.pi / 3, math.pi / 3)
    ee_pitch_range: tuple[float, float] = (-math.pi / 3, math.pi / 3)
    ee_yaw_range: tuple[float, float] = (-math.pi / 3, math.pi / 3)

    # Benchmark grouping is set by the evaluation script; training keeps the single base group.
    benchmark_env_groups: tuple[str, ...] = ("base",)
    benchmark_num_envs_per_group: int = 0
    reachable_pose_set_path: str = str(REACHABLE_TCP_SET_DIR / "reachable_tcp_pose_set.pt")
    reachable_target_task_ids: tuple[int, ...] = (0, 1)
    reachable_position_perturbation_radius_m: float = 0.0
    reachable_orientation_perturbation_radius_rad: float = 0.0

    debug_vis: bool = True
    show_current_ee_pose: bool = False

    ee_cmd_vis_x_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/ee_cmd_x",
        markers={
            "xaxis": CuboidCfg(size=(0.2, 0.01, 0.01), visual_material=PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)))
        },
    )
    ee_cmd_vis_y_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/ee_cmd_y",
        markers={
            "yaxis": CuboidCfg(size=(0.01, 0.2, 0.01), visual_material=PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)))
        },
    )
    ee_cmd_vis_z_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/ee_cmd_z",
        markers={
            "zaxis": CuboidCfg(size=(0.01, 0.01, 0.2), visual_material=PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)))
        },
    )
    # Current EE pose markers are smaller and color-shifted to distinguish them from target commands.
    ee_current_vis_x_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/ee_current_x",
        markers={
            "xaxis": CuboidCfg(
                size=(0.14, 0.008, 0.008), visual_material=PreviewSurfaceCfg(diffuse_color=(1.0, 0.55, 0.20))
            )
        },
    )
    ee_current_vis_y_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/ee_current_y",
        markers={
            "yaxis": CuboidCfg(
                size=(0.008, 0.14, 0.008), visual_material=PreviewSurfaceCfg(diffuse_color=(0.55, 1.0, 0.20))
            )
        },
    )
    ee_current_vis_z_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/ee_current_z",
        markers={
            "zaxis": CuboidCfg(
                size=(0.008, 0.008, 0.14), visual_material=PreviewSurfaceCfg(diffuse_color=(0.20, 0.80, 1.0))
            )
        },
    )
