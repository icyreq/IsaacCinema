from __future__ import annotations


from typing import TYPE_CHECKING


import torch


from isaaclab.envs import mdp


from isaaclab.assets import Articulation


from isaaclab.managers import SceneEntityCfg


from isaaclab.sensors import ContactSensor


from isaaclab.utils.math import quat_error_magnitude


from .observations import trot_foot_phase


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time_task_aware(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float,
    sensor_cfg: SceneEntityCfg,
    air_time_threshold: float,
) -> torch.Tensor:
    """Reward long foot swings on landing without penalizing short steps."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    swing = (last_air_time - air_time_threshold) * first_contact
    reward = torch.sum(torch.clamp(swing, min=0.0), dim=1)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > command_threshold
    return reward


def feet_slide(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def stand_still_joint_deviation_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    command = env.command_manager.get_command(command_name)
    # Penalize motion when command is nearly zero.
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)


def action_rate_l2_separated(env: ManagerBasedRLEnv, action_start_idx: int, action_end_idx: int) -> torch.Tensor:
    """Penalize the rate of change of a specific subset of actions using an L2 squared kernel."""
    curr_action = env.action_manager.action[:, action_start_idx:action_end_idx]
    prev_action = env.action_manager.prev_action[:, action_start_idx:action_end_idx]
    return torch.sum(torch.square(curr_action - prev_action), dim=1)


def combined_pose_command_error_exp(
    env: ManagerBasedRLEnv, pos_std: float, ori_std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w = command[:, :3]
    des_quat_w = command[:, 3:7]
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]
    pos_error = torch.norm(curr_pos_w - des_pos_w, dim=-1)
    ori_error = quat_error_magnitude(curr_quat_w, des_quat_w)
    return torch.exp(-(pos_error / pos_std + ori_error / ori_std))


def track_lin_vel_xy_exp_task_aware(
    env: ManagerBasedRLEnv, std: float, command_name: str, command_threshold: float = 0.1
) -> torch.Tensor:
    """Reward linear velocity tracking only when locomotion is actually commanded."""
    command = env.command_manager.get_command(command_name)
    active = torch.norm(command[:, :3], dim=1) > command_threshold
    return mdp.track_lin_vel_xy_exp(env, std=std, command_name=command_name) * active


def track_ang_vel_z_exp_task_aware(
    env: ManagerBasedRLEnv, std: float, command_name: str, command_threshold: float = 0.1
) -> torch.Tensor:
    """Reward yaw-rate tracking only when locomotion is actually commanded."""
    command = env.command_manager.get_command(command_name)
    active = torch.norm(command[:, :3], dim=1) > command_threshold
    return mdp.track_ang_vel_z_exp(env, std=std, command_name=command_name) * active


def leg_joint_position_penalty_task_aware(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Leg default-posture penalty during locomotion."""
    command = env.command_manager.get_command(command_name)
    active = torch.norm(command[:, :3], dim=1) > command_threshold
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    default_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.linalg.norm(joint_pos - default_pos, dim=1)
    return reward * active


def leg_contact_force_task_aware(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    command_threshold: float = 0.1,
    force_threshold: float = 5.0,
) -> torch.Tensor:
    """
    Rewards a 1.0 reward if all 4 feet maintain a minimum vertical contact force.
    """
    command = env.command_manager.get_command(command_name)
    is_manipulation = torch.norm(command[:, :3], dim=1) < command_threshold
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    # extract only the z component of contact forces
    vertical_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    all_in_contact = torch.all(vertical_forces > force_threshold, dim=1).float()
    return all_in_contact * is_manipulation


def task_aware_base_tilt_tolerance_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, max_angle_rad: float = 0.52, command_threshold: float = 0.1
) -> torch.Tensor:
    """
    Allows free pitching/rolling up to 'max_angle_rad' (e.g., ~20 degrees).
    Only penalizes tilt that exceeds this safety threshold.
    """
    command = env.command_manager.get_command(command_name)
    is_manipulation = torch.norm(command[:, :3], dim=1) < command_threshold
    projected_gravity = env.scene["robot"].data.projected_gravity_b
    # The norm of the x and y components is roughly sin(tilt_angle)
    current_tilt_sin = torch.norm(projected_gravity[:, :2], dim=-1)
    current_tilt_rad = torch.asin(torch.clamp(current_tilt_sin, -1.0, 1.0))
    error = torch.clamp(current_tilt_rad - max_angle_rad, min=0.0)
    reward = torch.exp(-error / std)
    return reward * is_manipulation


def _desired_contact_states(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float,
    frequency_hz: float,
    duration: float,
    kappa: float,
    phase_offsets: tuple[float, float, float, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute smooth desired contacts and per-foot phases."""
    foot_phase = trot_foot_phase(
        env,
        command_name,
        command_threshold,
        frequency_hz,
        phase_offsets=phase_offsets,
    )

    duration = min(max(duration, 1.0e-6), 1.0 - 1.0e-6)
    stance_phase = foot_phase * (0.5 / duration)
    swing_phase = 0.5 + (foot_phase - duration) * (0.5 / (1.0 - duration))
    warped_phase = torch.where(foot_phase < duration, stance_phase, swing_phase)

    normal_cdf = torch.distributions.Normal(0.0, kappa).cdf
    phase = torch.remainder(warped_phase, 1.0)
    desired_contact = normal_cdf(phase) * (1.0 - normal_cdf(phase - 0.5)) + normal_cdf(phase - 1.0) * (
        1.0 - normal_cdf(phase - 1.5)
    )
    return desired_contact, foot_phase


def _locomotion_command_active(env: ManagerBasedRLEnv, command_name: str, command_threshold: float) -> torch.Tensor:
    """True when base velocity command magnitude exceeds the locomotion threshold."""
    command = env.command_manager.get_command(command_name)
    return torch.norm(command[:, :3], dim=1) > command_threshold


def tracking_contacts_shaped_force(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    command_threshold: float = 0.1,
    frequency_hz: float = 3.0,
    duration: float = 0.5,
    kappa: float = 0.07,
    gait_force_sigma: float = 100.0,
) -> torch.Tensor:
    """Penalize foot contact force during desired swing phases."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    desired_contact, _ = _desired_contact_states(env, command_name, command_threshold, frequency_hz, duration, kappa)
    foot_forces = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1)
    reward = -(1.0 - desired_contact) * (1.0 - torch.exp(-(foot_forces.square()) / gait_force_sigma))
    reward = torch.mean(reward, dim=1)
    return reward * _locomotion_command_active(env, command_name, command_threshold)


def tracking_contacts_shaped_vel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    command_threshold: float = 0.1,
    frequency_hz: float = 3.0,
    duration: float = 0.5,
    kappa: float = 0.07,
    gait_vel_sigma: float = 10.0,
) -> torch.Tensor:
    """Penalize foot velocity during desired stance phases."""
    desired_contact, _ = _desired_contact_states(env, command_name, command_threshold, frequency_hz, duration, kappa)
    asset: Articulation = env.scene[asset_cfg.name]
    foot_velocities = torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :], dim=-1)
    reward = -desired_contact * (1.0 - torch.exp(-(foot_velocities.square()) / gait_vel_sigma))
    reward = torch.mean(reward, dim=1)
    return reward * _locomotion_command_active(env, command_name, command_threshold)


def feet_clearance_cmd_linear(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    command_threshold: float = 0.1,
    frequency_hz: float = 3.0,
    duration: float = 0.5,
    kappa: float = 0.07,
    footswing_height: float = 0.04,
    foot_radius: float = 0.02,
) -> torch.Tensor:
    """Penalize swing-foot height error against a triangular phase target.

    Uses the desired_contact-weighted mean of stance feet as the height reference
    instead of the single minimum foot, which avoids the self-referential oscillation
    that arises when the minimum foot changes between steps.
    """
    desired_contact, foot_phase = _desired_contact_states(
        env, command_name, command_threshold, frequency_hz, duration, kappa
    )
    phase = 1.0 - torch.abs(1.0 - torch.clamp((foot_phase * 2.0) - 1.0, 0.0, 1.0) * 2.0)
    asset: Articulation = env.scene[asset_cfg.name]
    foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]

    stance_ref = (foot_height * desired_contact).sum(dim=1, keepdim=True) / desired_contact.sum(
        dim=1, keepdim=True
    ).clamp(min=1.0e-6)
    relative_foot_height = foot_height - stance_ref

    target_height = footswing_height * phase + foot_radius
    reward = torch.sum((target_height - relative_foot_height).square() * (1.0 - desired_contact), dim=1)
    return reward * _locomotion_command_active(env, command_name, command_threshold)
