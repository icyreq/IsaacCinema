from __future__ import annotations
import math
import numpy as np
import torch
import warp as wp
from typing import TYPE_CHECKING
import omni
import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp import UniformVelocityCommand
from isaaclab.managers.command_manager import CommandTerm
from isaaclab.assets import Articulation
from isaaclab.markers import VisualizationMarkers
from isaaclab.envs import ManagerBasedEnv
import isaaclab.sim as sim_utils
from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh
from pxr import UsdGeom

from ..tcp_command_sampling import (
    cartesian_to_spherical,
    load_reachable_tcp_pose_set,
    sample_local_pose_perturbations,
    sample_reachable_tcp_poses,
)

if TYPE_CHECKING:
    from .config import TaskAwareVelocityCommandCfg, SphericalInterpolationCommandCfg


MANIP_TASK_ID = 0
LOCO_MANIP_TASK_ID = 1


class TaskAwareVelocityCommand(UniformVelocityCommand):
    """Assign tasks (0 = Pure Manipulation, 1 = Loco-Manipulation) upon spawn."""

    cfg: TaskAwareVelocityCommandCfg

    def __init__(self, cfg, env):
        super().__init__(cfg, env)

        probs = torch.tensor(self.cfg.probs, device=self.device)
        self.probs = probs / probs.sum()
        self.active_task_buffer = self._sample_tasks(self.num_envs)

        # Store the non-braking command so short emergency-brake events can restore it immediately.
        self.nominal_vel_command_b = self.vel_command_b.clone()
        self.random_zero_command_time_left = torch.zeros(self.num_envs, device=self.device)
        self.metrics["random_zero_command_active"] = torch.zeros(self.num_envs, device=self.device)

        # Validate the brake-event configuration before training starts.
        if self.cfg.random_zero_command_event_rate < 0.0:
            raise ValueError("random_zero_command_event_rate must be non-negative.")
        if self.cfg.random_zero_command_duration_range[0] < 0.0:
            raise ValueError("random_zero_command_duration_range lower bound must be non-negative.")
        if self.cfg.random_zero_command_duration_range[1] < self.cfg.random_zero_command_duration_range[0]:
            raise ValueError("random_zero_command_duration_range upper bound must be >= lower bound.")

        logging_probs = self.probs * 100
        print(
            f"Task assignment: pure manipulation {logging_probs[0]:.1f}%, "
            f"loco-manipulation: {logging_probs[1]:.1f}% "
        )

    def _sample_tasks(self, num_envs: int) -> torch.Tensor:
        """Sample task ids from the configured categorical distribution."""
        return torch.multinomial(self.probs, num_envs, replacement=True)

    def __str__(self) -> str:
        msg = super().__str__()
        if self.cfg.random_zero_command_event_rate > 0.0:
            msg += f"\n\tRandom zero-command event rate: {self.cfg.random_zero_command_event_rate} Hz"
            msg += f"\n\tRandom zero-command duration range: {self.cfg.random_zero_command_duration_range} s"
        return msg

    def reset(self, env_ids=None) -> dict[str, float]:
        # Clear active brake timers at episode boundaries. Task ids are redrawn on command resample.
        resolved_env_ids = slice(None) if env_ids is None else env_ids
        self.random_zero_command_time_left[resolved_env_ids] = 0.0
        return super().reset(env_ids)

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        # tasks are fixed for each environment
        assigned_tasks = self.active_task_buffer[env_ids]

        # apply forward-backward motion curriculum
        loco_envs = env_ids[assigned_tasks == LOCO_MANIP_TASK_ID]
        if len(loco_envs) > 0:
            if self.cfg.enable_backward_motion:
                forward_mask = torch.rand(len(loco_envs), device=self.device) < self.cfg.forward_prob
                forward_envs = loco_envs[forward_mask]
                backward_envs = loco_envs[~forward_mask]
                if len(forward_envs) > 0:
                    self.vel_command_b[forward_envs, 0] = torch.empty(len(forward_envs), device=self.device).uniform_(
                        *self.cfg.forward_lin_vel_x
                    )
                if len(backward_envs) > 0:
                    self.vel_command_b[backward_envs, 0] = torch.empty(len(backward_envs), device=self.device).uniform_(
                        *self.cfg.backward_lin_vel_x
                    )
            else:
                self.vel_command_b[loco_envs, 0] = torch.empty(len(loco_envs), device=self.device).uniform_(
                    *self.cfg.forward_lin_vel_x
                )

        manip_envs = env_ids[assigned_tasks == MANIP_TASK_ID]
        self.vel_command_b[manip_envs] = 0.0

        # Save the task-aware command before any short emergency-brake override is applied.
        self.nominal_vel_command_b[env_ids] = self.vel_command_b[env_ids]

    def _update_metrics(self):
        super()._update_metrics()

        # Track the episode fraction spent in emergency-brake events for logging.
        if self.cfg.random_zero_command_event_rate > 0.0:
            self.metrics["random_zero_command_active"] += (
                self.random_zero_command_time_left > 0.0
            ).float() / self._env.max_episode_length

    def _update_command(self):
        # Rebuild the command from the normal task-aware path before sampling brake events.
        self.vel_command_b.copy_(self.nominal_vel_command_b)
        super()._update_command()
        self.nominal_vel_command_b.copy_(self.vel_command_b)

        if self.cfg.random_zero_command_event_rate <= 0.0:
            return

        # Advance existing timers and sample new short brake events in physical time.
        dt = self._env.step_dt
        active_before_update = self.random_zero_command_time_left > 0.0
        self.random_zero_command_time_left = torch.clamp(self.random_zero_command_time_left - dt, min=0.0)

        event_probability = 1.0 - math.exp(-self.cfg.random_zero_command_event_rate * dt)
        eligible_env_ids = (
            ((self.random_zero_command_time_left <= 0.0) & (self.active_task_buffer == LOCO_MANIP_TASK_ID))
            .nonzero(as_tuple=False)
            .flatten()
        )
        if len(eligible_env_ids) > 0:
            random_values = torch.rand(len(eligible_env_ids), device=self.device)
            new_event_env_ids = eligible_env_ids[random_values <= event_probability]
            if len(new_event_env_ids) > 0:
                durations = torch.empty(len(new_event_env_ids), device=self.device)
                self.random_zero_command_time_left[new_event_env_ids] = durations.uniform_(
                    *self.cfg.random_zero_command_duration_range
                )
        else:
            new_event_env_ids = torch.empty(0, dtype=torch.long, device=self.device)

        # While a brake event is active, expose an immediate zero base-velocity command.
        active_env_ids = active_before_update.nonzero(as_tuple=False).flatten()
        if len(active_env_ids) > 0:
            self.vel_command_b[active_env_ids, :] = 0.0
        if len(new_event_env_ids) > 0:
            self.vel_command_b[new_event_env_ids, :] = 0.0

    @property
    def task_assignment(self):
        return self.active_task_buffer


class SphericalInterpolationCommand(CommandTerm):
    cfg: SphericalInterpolationCommandCfg

    def __init__(self, cfg: SphericalInterpolationCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]

        # Dynamically find the configured end-effector link index for Forward Kinematics.
        self.ee_body_idx = self.robot.find_bodies(self.cfg.ee_body_name)[0][0]
        self.hard_goal_bounds = self.cfg.hard_goal_bounds_type

        # Initialize buffers (Start point vs Target point)
        self.target_spherical_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_quat_b = torch.zeros(self.num_envs, 4, device=self.device)

        self.pos_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.quat_command_b = torch.zeros(self.num_envs, 4, device=self.device)
        self.pos_command_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.quat_command_w = torch.zeros(self.num_envs, 4, device=self.device)

        self.quat_command_b[:, 0] = 1.0
        self.quat_command_w[:, 0] = 1.0

        """ approximate bounding box for Go2+ARX (plus some safety buffer) """
        # TODO xzm: make acquisition automatic
        safe_buffer = 0.1
        self.box_x = [-0.25 - safe_buffer, 0.34 + safe_buffer]
        self.box_y = [-0.15 - safe_buffer, 0.15 + safe_buffer]
        self.box_z = [-0.35 - safe_buffer, 0.27 + safe_buffer]
        self.max_target_sample_rounds = 64
        self.pitch_solve_iterations = 10
        self.manip_terrain_z_w = torch.zeros(self.num_envs, device=self.device)
        self.manip_terrain_z_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._terrain_wp_mesh: wp.Mesh | None = None

        # Reachable benchmark targets are loaded lazily only when the evaluation config requests them.
        self._reachable_pose_set = None
        self._benchmark_group_by_env = self._build_benchmark_group_by_env()
        self._reachable_target_task_ids = tuple(int(task_id) for task_id in self.cfg.reachable_target_task_ids)

        supported_task_ids = {MANIP_TASK_ID, LOCO_MANIP_TASK_ID}
        unsupported_task_ids = set(self._reachable_target_task_ids) - supported_task_ids
        if unsupported_task_ids:
            raise ValueError(
                f"Unsupported reachable_target_task_ids {sorted(unsupported_task_ids)}; "
                f"supported task ids are {sorted(supported_task_ids)}."
            )
        if len(self._reachable_target_task_ids) != len(set(self._reachable_target_task_ids)):
            raise ValueError("reachable_target_task_ids must not contain duplicates.")
        if self.cfg.reachable_position_perturbation_radius_m < 0.0:
            raise ValueError("reachable_position_perturbation_radius_m must be non-negative.")
        if self.cfg.reachable_orientation_perturbation_radius_rad < 0.0:
            raise ValueError("reachable_orientation_perturbation_radius_rad must be non-negative.")

    def _build_benchmark_group_by_env(self) -> tuple[str, ...]:
        """Expand the configured benchmark groups into one label per environment."""
        groups = tuple(self.cfg.benchmark_env_groups)
        if len(groups) == 0:
            raise ValueError("benchmark_env_groups must contain at least one group.")

        num_per_group = int(self.cfg.benchmark_num_envs_per_group)
        if num_per_group <= 0:
            return tuple([groups[0]] * self.num_envs)

        expected_num_envs = num_per_group * len(groups)
        if expected_num_envs != self.num_envs:
            raise ValueError(
                "benchmark_num_envs_per_group and benchmark_env_groups do not match num_envs: "
                f"{num_per_group} * {len(groups)} != {self.num_envs}."
            )

        group_by_env: list[str] = []
        for group_name in groups:
            group_by_env.extend([group_name] * num_per_group)
        return tuple(group_by_env)

    def _ensure_reachable_pose_sets(self) -> None:
        """Load the full reachable pose set once before sampling benchmark targets."""
        if self._reachable_pose_set is None:
            self._reachable_pose_set = load_reachable_tcp_pose_set(self.cfg.reachable_pose_set_path, self.device)

    def _benchmark_group_mask(self, env_ids: torch.Tensor, group_name: str) -> torch.Tensor:
        """Return a mask over the resampled env_ids for a named benchmark group."""
        env_group_names = [self._benchmark_group_by_env[int(env_id)] for env_id in env_ids.tolist()]
        return torch.tensor([name == group_name for name in env_group_names], dtype=torch.bool, device=self.device)

    def _apply_reachable_targets(self, reachable_env_ids: torch.Tensor) -> None:
        """Sample reachable-group EE targets from the full cached TCP pose set."""
        if len(reachable_env_ids) == 0:
            return
        self._ensure_reachable_pose_sets()

        target_pos_b, target_quat_b = sample_reachable_tcp_poses(
            self._reachable_pose_set[0],
            self._reachable_pose_set[1],
            len(reachable_env_ids),
            self.device,
            self.target_spherical_b.dtype,
        )
        target_pos_b, target_quat_b = sample_local_pose_perturbations(
            target_pos_b,
            target_quat_b,
            self.cfg.reachable_position_perturbation_radius_m,
            self.cfg.reachable_orientation_perturbation_radius_rad,
        )
        self.target_spherical_b[reachable_env_ids] = cartesian_to_spherical(target_pos_b)
        self.target_quat_b[reachable_env_ids] = target_quat_b

    def _reachable_task_mask(self, task_ids: torch.Tensor) -> torch.Tensor:
        """Return which task assignments use the offline reachable target set."""
        mask = torch.zeros_like(task_ids, dtype=torch.bool)
        for task_id in self._reachable_target_task_ids:
            mask |= task_ids == task_id
        return mask

    def _refresh_manip_terrain_height(self, env_ids: torch.Tensor) -> None:
        """Approximate manipulation terrain height once at reset from the standing root height."""
        if len(env_ids) == 0:
            return
        root_z_w = self.robot.data.root_pos_w[env_ids, 2]
        nominal_base_height = self.robot.data.default_root_state[env_ids, 2]
        self.manip_terrain_z_w[env_ids] = root_z_w - nominal_base_height
        self.manip_terrain_z_initialized[env_ids] = True

    def _terrain_height_for_sampling(
        self,
        env_ids: torch.Tensor,
        root_pos_w: torch.Tensor,
        current_tasks: torch.Tensor,
    ) -> torch.Tensor:
        """Return cached manip terrain height, with a current-root fallback for other tasks."""
        terrain_z_w = root_pos_w[:, 2] - self.robot.data.default_root_state[env_ids, 2]
        manip_mask = current_tasks == 0
        if manip_mask.any():
            terrain_z_w[manip_mask] = self.manip_terrain_z_w[env_ids[manip_mask]]
        return terrain_z_w

    def _ensure_terrain_warp_mesh(self) -> wp.Mesh | None:
        """Create a cached warp mesh for exact terrain-height raycasts."""
        if self._terrain_wp_mesh is not None:
            return self._terrain_wp_mesh
        terrain = getattr(self._env.scene, "terrain", None)
        if terrain is None or not getattr(terrain, "terrain_prim_paths", None):
            return None

        # IsaacLab imports generated terrain under /World/ground/terrain with a Mesh child.
        mesh_prim = None
        for terrain_prim_path in terrain.terrain_prim_paths:
            mesh_prim = sim_utils.get_first_matching_child_prim(
                terrain_prim_path, lambda prim: prim.GetTypeName() == "Mesh"
            )
            if mesh_prim is not None and mesh_prim.IsValid():
                break
        if mesh_prim is None or not mesh_prim.IsValid():
            raise RuntimeError("Generated terrain exists, but no terrain mesh prim was found for height raycasts.")

        # Read the static terrain mesh in world coordinates so command sampling can raycast arbitrary target XYs.
        mesh = UsdGeom.Mesh(mesh_prim)
        points = np.asarray(mesh.GetPointsAttr().Get())
        transform_matrix = np.array(omni.usd.get_world_transform_matrix(mesh_prim)).T
        points = np.matmul(points, transform_matrix[:3, :3].T)
        points += transform_matrix[:3, 3]
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        self._terrain_wp_mesh = convert_to_warp_mesh(points, indices, device=self.device)
        return self._terrain_wp_mesh

    def _terrain_height_at_world_xy(
        self,
        xy_w: torch.Tensor,
        approximate_terrain_z_w: torch.Tensor,
    ) -> torch.Tensor:
        """Return terrain height at world XY using raycast, with configured approximation when no terrain exists."""
        if len(xy_w) == 0:
            return torch.empty(0, device=self.device, dtype=approximate_terrain_z_w.dtype)

        terrain_mesh = self._ensure_terrain_warp_mesh()
        if terrain_mesh is None:
            return approximate_terrain_z_w

        # Cast from high above each target XY so the hit height is the actual generated terrain surface.
        ray_starts = torch.zeros((len(xy_w), 3), device=self.device, dtype=approximate_terrain_z_w.dtype)
        ray_starts[:, :2] = xy_w.to(dtype=approximate_terrain_z_w.dtype)
        ray_starts[:, 2] = 100.0
        ray_dirs = torch.zeros_like(ray_starts)
        ray_dirs[:, 2] = -1.0
        ray_hits = raycast_mesh(ray_starts, ray_dirs, terrain_mesh)[0]
        terrain_z = ray_hits[:, 2]

        hit_mask = torch.isfinite(terrain_z)
        if not hit_mask.all():
            raise RuntimeError("Terrain height raycast missed the generated terrain mesh.")
        return terrain_z

    def _solve_pitch_for_terrain_relative_height(
        self,
        radius: torch.Tensor,
        yaw: torch.Tensor,
        desired_z_rel: torch.Tensor,
        anchor_pos_w: torch.Tensor,
        anchor_quat_w: torch.Tensor,
        approximate_terrain_z_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Solve pitch so each target sits at the sampled height above its own terrain XY."""
        pitch_limit = math.pi / 2.0 - 1.0e-4
        pitch_low = torch.full_like(radius, -pitch_limit)
        pitch_high = torch.full_like(radius, pitch_limit)
        terrain_z = approximate_terrain_z_w
        pos_w = anchor_pos_w.clone()

        # Bisection is robust for the vertical residual and only runs when EE commands are resampled.
        for _ in range(self.pitch_solve_iterations):
            pitch_mid = 0.5 * (pitch_low + pitch_high)
            pos_b = torch.stack(
                [
                    radius * torch.cos(pitch_mid) * torch.cos(yaw),
                    radius * torch.cos(pitch_mid) * torch.sin(yaw),
                    radius * torch.sin(pitch_mid),
                ],
                dim=-1,
            )
            pos_w = anchor_pos_w + math_utils.quat_apply(anchor_quat_w, pos_b)
            terrain_z = self._terrain_height_at_world_xy(pos_w[:, :2], approximate_terrain_z_w)
            residual = pos_w[:, 2] - terrain_z - desired_z_rel
            pitch_high = torch.where(residual > 0.0, pitch_mid, pitch_high)
            pitch_low = torch.where(residual <= 0.0, pitch_mid, pitch_low)

        pitch = 0.5 * (pitch_low + pitch_high)
        pos_b = torch.stack(
            [
                radius * torch.cos(pitch) * torch.cos(yaw),
                radius * torch.cos(pitch) * torch.sin(yaw),
                radius * torch.sin(pitch),
            ],
            dim=-1,
        )
        pos_w = anchor_pos_w + math_utils.quat_apply(anchor_quat_w, pos_b)
        terrain_z = self._terrain_height_at_world_xy(pos_w[:, :2], approximate_terrain_z_w)
        z_rel = pos_w[:, 2] - terrain_z
        return pitch, pos_b, pos_w, z_rel

    def _sample_manip_hard_spherical_terrain_relative(
        self,
        count: int,
        anchor_pos_w: torch.Tensor,
        anchor_quat_w: torch.Tensor,
        approximate_terrain_z_w: torch.Tensor,
        dtype: torch.dtype,
        ground_pick_prob: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample hard goals whose requested height is tied to terrain under the target point."""
        r_samp, desired_z_rel, y_samp = self.hard_goal_bounds.sample_manip_hard_radius_height_yaw(
            count,
            self.device,
            dtype,
            ground_pick_prob=ground_pick_prob,
        )
        p_samp, pos_b, pos_w, z_rel = self._solve_pitch_for_terrain_relative_height(
            r_samp,
            y_samp,
            desired_z_rel,
            anchor_pos_w,
            anchor_quat_w,
            approximate_terrain_z_w,
        )
        return r_samp, p_samp, y_samp, pos_b, pos_w, z_rel

    def _apply_fallback_targets(
        self,
        sample_idx: torch.Tensor,
        env_ids: torch.Tensor,
        root_pos_w: torch.Tensor,
        root_quat_w: torch.Tensor,
        terrain_z_w: torch.Tensor,
        r_final: torch.Tensor,
        p_final: torch.Tensor,
        y_final: torch.Tensor,
    ) -> None:
        """Fill any remaining samples with terrain-relative high-reach targets."""
        if len(sample_idx) == 0:
            return
        fallback_env_ids = env_ids[sample_idx]

        # Use the same yaw-only nominal-height anchor that converts sampled commands to world frame.
        anchor_pos_w = root_pos_w.clone()
        anchor_pos_w[:, 2] = self.robot.data.default_root_state[env_ids, 2]
        anchor_quat_w = math_utils.yaw_quat(root_quat_w)
        fallback_r, fallback_p, fallback_y, _, _, _ = self._sample_manip_hard_spherical_terrain_relative(
            len(sample_idx),
            anchor_pos_w[sample_idx],
            anchor_quat_w[sample_idx],
            terrain_z_w[sample_idx],
            r_final.dtype,
            ground_pick_prob=0.0,
        )
        r_final[sample_idx] = fallback_r
        p_final[sample_idx] = fallback_p
        y_final[sample_idx] = fallback_y
        print(
            "[WARN] EE target sampler hit max rounds; "
            f"used terrain-relative high-reach fallback for {len(fallback_env_ids)} envs."
        )

    def _apply_random_targets(self, env_ids: torch.Tensor, current_tasks: torch.Tensor) -> None:
        """Sample base benchmark EE targets with the original rejection-sampling logic."""
        if len(env_ids) == 0:
            return
        root_pos_w = self.robot.data.root_pos_w[env_ids]
        root_quat_w = self.robot.data.root_quat_w[env_ids]
        terrain_z_w = self._terrain_height_for_sampling(env_ids, root_pos_w, current_tasks)

        # Sample and validate targets in the same anchor frame used by _update_command().
        anchor_pos_w = root_pos_w.clone()
        anchor_pos_w[:, 2] = self.robot.data.default_root_state[env_ids, 2]
        anchor_quat_w = math_utils.yaw_quat(root_quat_w)

        """ Rejection sampling of target points """
        sampler = torch.empty(len(env_ids), device=self.device)
        num_to_sample = len(env_ids)
        valid_mask = torch.zeros(num_to_sample, dtype=torch.bool, device=self.device)
        r_final = torch.zeros(num_to_sample, device=self.device)
        p_final = torch.zeros(num_to_sample, device=self.device)
        y_final = torch.zeros(num_to_sample, device=self.device)

        rounds = 0
        while not valid_mask.all() and rounds < self.max_target_sample_rounds:
            invalid_idx = (~valid_mask).nonzero(as_tuple=True)[0]
            num_invalid = len(invalid_idx)
            tasks_for_invalid = current_tasks[invalid_idx]
            manip_candidate_mask = tasks_for_invalid == 0

            r_samp = torch.empty(num_invalid, device=self.device).uniform_(*self.cfg.radius_range)
            p_samp = torch.empty(num_invalid, device=self.device).uniform_(*self.cfg.pitch_range)
            y_samp = torch.empty(num_invalid, device=self.device).uniform_(*self.cfg.yaw_range)
            if manip_candidate_mask.any():
                manip_invalid_idx = invalid_idx[manip_candidate_mask]
                hard_r, hard_p, hard_y, hard_pos_b, hard_pos_w, hard_z_rel = (
                    self._sample_manip_hard_spherical_terrain_relative(
                        len(manip_invalid_idx),
                        anchor_pos_w[manip_invalid_idx],
                        anchor_quat_w[manip_invalid_idx],
                        terrain_z_w[manip_invalid_idx],
                        r_samp.dtype,
                    )
                )
                r_samp[manip_candidate_mask] = hard_r
                p_samp[manip_candidate_mask] = hard_p
                y_samp[manip_candidate_mask] = hard_y

            x_b = r_samp * torch.cos(p_samp) * torch.cos(y_samp)
            y_b = r_samp * torch.cos(p_samp) * torch.sin(y_samp)
            z_b = r_samp * torch.sin(p_samp)
            pos_b = torch.stack([x_b, y_b, z_b], dim=-1)

            # (1) check for self-collision
            in_x = (x_b > self.box_x[0]) & (x_b < self.box_x[1])
            in_y = (y_b > self.box_y[0]) & (y_b < self.box_y[1])
            in_z = (z_b > self.box_z[0]) & (z_b < self.box_z[1])
            self_collision = in_x & in_y & in_z

            # (2) check for ground collision
            pos_w = anchor_pos_w[invalid_idx] + math_utils.quat_apply(anchor_quat_w[invalid_idx], pos_b)
            z_rel = pos_w[:, 2] - terrain_z_w[invalid_idx]
            if manip_candidate_mask.any():
                pos_b[manip_candidate_mask] = hard_pos_b
                pos_w[manip_candidate_mask] = hard_pos_w
                z_rel[manip_candidate_mask] = hard_z_rel
            ground_collision = z_rel < self.hard_goal_bounds.GROUND_WORLD_Z_MIN

            # (3) hard goal filter for pure-manipulation envs: ground-pick or high-reach frontal arc.
            is_hard_goal = self.hard_goal_bounds.is_manip_hard_goal_from_samples(
                r_samp, y_samp, x_b, z_rel, self_collision
            )

            # If Task == 0 (Manipulation), it MUST be a hard goal.
            # If Task == 1 (Loco-manip), ANY valid point is acceptable.
            satisfies_task_condition = torch.where(
                tasks_for_invalid == 0, is_hard_goal, torch.ones_like(is_hard_goal, dtype=torch.bool)
            )

            # Combine all validity masks
            valid_this_round = ~(self_collision | ground_collision) & satisfies_task_condition

            valid_original_idx = invalid_idx[valid_this_round]
            r_final[valid_original_idx] = r_samp[valid_this_round]
            p_final[valid_original_idx] = p_samp[valid_this_round]
            y_final[valid_original_idx] = y_samp[valid_this_round]

            valid_mask[invalid_idx] = valid_this_round
            rounds += 1

        if not valid_mask.all():
            remaining_idx = (~valid_mask).nonzero(as_tuple=True)[0]
            self._apply_fallback_targets(
                remaining_idx,
                env_ids,
                root_pos_w,
                root_quat_w,
                terrain_z_w,
                r_final,
                p_final,
                y_final,
            )

        self.target_spherical_b[env_ids, 0] = r_final
        self.target_spherical_b[env_ids, 1] = p_final
        self.target_spherical_b[env_ids, 2] = y_final

        roll = sampler.uniform_(*self.cfg.ee_roll_range)
        pitch = sampler.uniform_(*self.cfg.ee_pitch_range)
        yaw = sampler.uniform_(*self.cfg.ee_yaw_range)
        self.target_quat_b[env_ids] = math_utils.quat_from_euler_xyz(roll, pitch, yaw)

    def _resample_command(self, env_ids: torch.Tensor):
        """
        Independently resamples the EE goal. Every environment, whether manipulation or
        loco-manipulation, will have an ee goal.
        """
        # Read the latest task tags from the velocity command manager (may switch mid-episode).
        current_tasks = self._env.command_manager.get_term("base_velocity").task_assignment[env_ids]
        manip_envs = env_ids[current_tasks == 0]
        if len(manip_envs) > 0:
            needs_refresh = (self.command_counter[manip_envs] == 0) | ~self.manip_terrain_z_initialized[manip_envs]
            self._refresh_manip_terrain_height(manip_envs[needs_refresh])

        # Route each benchmark group to its own target source so base sampling stays unchanged.
        reachable_mask = self._benchmark_group_mask(env_ids, "reachable") & self._reachable_task_mask(current_tasks)
        self._apply_random_targets(env_ids[~reachable_mask], current_tasks[~reachable_mask])
        self._apply_reachable_targets(env_ids[reachable_mask])

    def _update_command(self):
        interp_spherical = self.target_spherical_b

        r, p, y = interp_spherical[:, 0], interp_spherical[:, 1], interp_spherical[:, 2]
        self.pos_command_b[:, 0] = r * torch.cos(p) * torch.cos(y)
        self.pos_command_b[:, 1] = r * torch.cos(p) * torch.sin(y)
        self.pos_command_b[:, 2] = r * torch.sin(p)

        self.quat_command_b = self.target_quat_b

        """ transformation of goal points/orientations to world frame (move with base) """
        curr_root_pos = self.robot.data.root_pos_w.clone()
        curr_root_pos[:, 2] = self.robot.data.default_root_state[:, 2]  # Lock nominal height
        curr_flat_quat = math_utils.yaw_quat(self.robot.data.root_quat_w)

        self.pos_command_w = curr_root_pos + math_utils.quat_apply(curr_flat_quat, self.pos_command_b)
        self.quat_command_w = math_utils.quat_mul(curr_flat_quat, self.quat_command_b)

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set visibility of markers."""
        if debug_vis:
            if not hasattr(self, "ee_cmd_vis_x"):
                self.ee_cmd_vis_x = VisualizationMarkers(self.cfg.ee_cmd_vis_x_cfg)
                self.ee_cmd_vis_y = VisualizationMarkers(self.cfg.ee_cmd_vis_y_cfg)
                self.ee_cmd_vis_z = VisualizationMarkers(self.cfg.ee_cmd_vis_z_cfg)
            if self.cfg.show_current_ee_pose and not hasattr(self, "ee_current_vis_x"):
                self.ee_current_vis_x = VisualizationMarkers(self.cfg.ee_current_vis_x_cfg)
                self.ee_current_vis_y = VisualizationMarkers(self.cfg.ee_current_vis_y_cfg)
                self.ee_current_vis_z = VisualizationMarkers(self.cfg.ee_current_vis_z_cfg)
            self.ee_cmd_vis_x.set_visibility(True)
            self.ee_cmd_vis_y.set_visibility(True)
            self.ee_cmd_vis_z.set_visibility(True)
            if hasattr(self, "ee_current_vis_x"):
                self.ee_current_vis_x.set_visibility(self.cfg.show_current_ee_pose)
                self.ee_current_vis_y.set_visibility(self.cfg.show_current_ee_pose)
                self.ee_current_vis_z.set_visibility(self.cfg.show_current_ee_pose)
        else:
            if hasattr(self, "ee_cmd_vis_x"):
                self.ee_cmd_vis_x.set_visibility(False)
                self.ee_cmd_vis_y.set_visibility(False)
                self.ee_cmd_vis_z.set_visibility(False)
            if hasattr(self, "ee_current_vis_x"):
                self.ee_current_vis_x.set_visibility(False)
                self.ee_current_vis_y.set_visibility(False)
                self.ee_current_vis_z.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Update the marker positions dynamically."""
        if not self.cfg.debug_vis or not hasattr(self, "ee_cmd_vis_x"):
            return
        if not self.robot.is_initialized:
            return

        num_envs = self.target_spherical_b.shape[0]
        offset = self.cfg.ee_cmd_vis_x_cfg.markers["xaxis"].size[0] / 2
        offset_x_b = torch.tensor([offset, 0.0, 0.0], device=self.device).repeat(num_envs, 1)
        offset_y_b = torch.tensor([0.0, offset, 0.0], device=self.device).repeat(num_envs, 1)
        offset_z_b = torch.tensor([0.0, 0.0, offset], device=self.device).repeat(num_envs, 1)

        """ task-conditioned visualization of final goal point """
        curr_root_pos = self.robot.data.root_pos_w.clone()
        curr_root_pos[:, 2] = self.robot.data.default_root_state[:, 2]
        curr_flat_quat = math_utils.yaw_quat(self.robot.data.root_quat_w)

        # Convert final spherical target to cartesian base frame
        r, p, y = self.target_spherical_b[:, 0], self.target_spherical_b[:, 1], self.target_spherical_b[:, 2]
        target_pos_b = torch.stack(
            [r * torch.cos(p) * torch.cos(y), r * torch.cos(p) * torch.sin(y), r * torch.sin(p)], dim=-1
        )

        # Transform to world space using the FROZEN anchor frame
        target_pos_w = curr_root_pos + math_utils.quat_apply(curr_flat_quat, target_pos_b)
        target_quat_w = math_utils.quat_mul(curr_flat_quat, self.target_quat_b)

        # Calculate offsets for the final marker
        tgt_offset_x_w = math_utils.quat_apply(target_quat_w, offset_x_b)
        tgt_offset_y_w = math_utils.quat_apply(target_quat_w, offset_y_b)
        tgt_offset_z_w = math_utils.quat_apply(target_quat_w, offset_z_b)

        self.ee_cmd_vis_x.visualize(target_pos_w + tgt_offset_x_w, target_quat_w)
        self.ee_cmd_vis_y.visualize(target_pos_w + tgt_offset_y_w, target_quat_w)
        self.ee_cmd_vis_z.visualize(target_pos_w + tgt_offset_z_w, target_quat_w)

        """ current EE visualization for inspecting tracking error in the viewer """
        if not self.cfg.show_current_ee_pose or not hasattr(self, "ee_current_vis_x"):
            return

        # Place the current-pose axes at the configured end-effector pose from simulation state.
        curr_offset = self.cfg.ee_current_vis_x_cfg.markers["xaxis"].size[0] / 2
        curr_offset_x_b = torch.tensor([curr_offset, 0.0, 0.0], device=self.device).repeat(num_envs, 1)
        curr_offset_y_b = torch.tensor([0.0, curr_offset, 0.0], device=self.device).repeat(num_envs, 1)
        curr_offset_z_b = torch.tensor([0.0, 0.0, curr_offset], device=self.device).repeat(num_envs, 1)

        curr_pos_w = self.robot.data.body_pos_w[:, self.ee_body_idx]
        curr_quat_w = self.robot.data.body_quat_w[:, self.ee_body_idx]
        curr_offset_x_w = math_utils.quat_apply(curr_quat_w, curr_offset_x_b)
        curr_offset_y_w = math_utils.quat_apply(curr_quat_w, curr_offset_y_b)
        curr_offset_z_w = math_utils.quat_apply(curr_quat_w, curr_offset_z_b)

        self.ee_current_vis_x.visualize(curr_pos_w + curr_offset_x_w, curr_quat_w)
        self.ee_current_vis_y.visualize(curr_pos_w + curr_offset_y_w, curr_quat_w)
        self.ee_current_vis_z.visualize(curr_pos_w + curr_offset_z_w, curr_quat_w)

    def _update_metrics(self):
        pass

    @property
    def command(self) -> torch.Tensor:
        """Returns the [x, y, z, qw, qx, qy, qz] command in the world (anchor) frame."""
        return torch.cat([self.pos_command_w, self.quat_command_w], dim=-1)

    def __str__(self):
        msg = "SphericalInterpolationCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
