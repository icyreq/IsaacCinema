from __future__ import annotations


from typing import TYPE_CHECKING


import math


import torch


from isaaclab.managers import SceneEntityCfg


from isaaclab.utils.math import (
    euler_xyz_from_quat,
    matrix_from_quat,
    quat_apply_inverse,
    quat_inv,
    quat_mul,
    yaw_quat,
)


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


ROBODUET_TROT_PHASE_OFFSETS = (0.5, 0.0, 0.0, 0.5)


def _quat_to_rot6d(quat: torch.Tensor) -> torch.Tensor:
    """Legacy project rot6d convention retained for existing baseline checkpoints."""
    rot_mat = matrix_from_quat(quat)  # (N, 3, 3)
    return rot_mat[..., :2].transpose(1, 2).contiguous().view(quat.shape[0], 6)


def ee_position_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Computes the 3D position error vector between the EE and the command."""
    asset = env.scene[asset_cfg.name]
    # ee command is in world frame
    command_w = env.command_manager.get_command(command_name)
    target_pos_w = command_w[:, 0:3]
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]
    # gravity-aligned base frame
    root_quat_w = asset.data.root_quat_w
    flat_quat_w = yaw_quat(root_quat_w)

    error_w = target_pos_w - curr_pos_w
    # Rotate the error into the EE Local Frame
    error_local = quat_apply_inverse(flat_quat_w, error_w)
    return error_local


def ee_orientation_error_6d(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Computes the 3D orientation error between the current end-effector and the target.
    Returns the vector part of the error quaternion, representing the rotation axis scaled by angle.
    """
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    des_quat_w = command[:, 3:7]
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]

    quat_error = quat_mul(quat_inv(curr_quat_w), des_quat_w)

    # Enforce shortest path rotation
    sign = torch.sign(quat_error[:, 0]).unsqueeze(-1)
    # Handle the rare edge case where qw is exactly 0
    sign = torch.where(sign == 0.0, torch.ones_like(sign), sign)

    return _quat_to_rot6d(sign * quat_error)


def trot_foot_phase(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.1,
    frequency_hz: float = 3.0,
    phase_offsets: tuple[float, float, float, float] | None = None,
) -> torch.Tensor:
    """Return per-foot normalized trot phase"""
    command = env.command_manager.get_command(command_name)
    active = torch.norm(command[:, :3], dim=1) >= command_threshold

    time_s = env.episode_length_buf.to(dtype=torch.float32, device=env.device) * env.step_dt
    base_phase = torch.remainder(frequency_hz * time_s, 1.0)

    # Keep the project's existing default phase for baseline checkpoint compatibility.
    offsets = (0.0, 0.5, 0.5, 0.0) if phase_offsets is None else phase_offsets
    phase_offset_tensor = torch.tensor(offsets, device=env.device)
    foot_phase = torch.remainder(base_phase.unsqueeze(1) + phase_offset_tensor.unsqueeze(0), 1.0)

    # freezes zero-velocity commands at phase 0.25, yielding sin(2*pi*phase)=1 for all feet.
    stand_phase = torch.full_like(foot_phase, 0.25)
    return torch.where(active.unsqueeze(1), foot_phase, stand_phase)


def trot_clock_phase(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.1,
    frequency_hz: float = 3.0,
    phase_offsets: tuple[float, float, float, float] | None = None,
) -> torch.Tensor:
    """Return 4D sine clock inputs for a fixed trot."""
    foot_phase = trot_foot_phase(
        env,
        command_name,
        command_threshold,
        frequency_hz,
        phase_offsets=phase_offsets,
    )
    return torch.sin(2.0 * math.pi * foot_phase)
