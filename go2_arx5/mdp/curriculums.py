from __future__ import annotations


import torch


from collections.abc import Sequence


from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def enable_backward_motion_when_tracking(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "base_track_lin_vel_xy_exp",
    command_name: str = "base_velocity",
    threshold: float = 0.8,
) -> torch.Tensor:
    command_term = env.command_manager.get_term(command_name)

    if command_term.cfg.enable_backward_motion:
        return torch.tensor(1.0, device=env.device)

    env_ids = torch.as_tensor(env_ids, device=env.device)
    task_ids = command_term.task_assignment[env_ids]
    loco_env_ids = env_ids[task_ids == 1]

    if len(loco_env_ids) == 0:
        return torch.tensor(0.0, device=env.device)

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][loco_env_ids]) / env.max_episode_length_s

    desired_reward = reward_term.weight
    if reward > desired_reward * threshold:
        command_term.cfg.enable_backward_motion = True
        print(
            f"[Curriculum] Enabled backward motion: "
            f"{reward_term_name}={reward.item():.3f}, target={desired_reward * threshold:.3f}"
        )

    return torch.tensor(float(command_term.cfg.enable_backward_motion), device=env.device)
