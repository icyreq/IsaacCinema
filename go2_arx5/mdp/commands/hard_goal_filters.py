from __future__ import annotations

import math
import torch


class ManipHardGoalBounds:
    """Bound parameters and predicates for ground-pick / high-reach hard goals.

    Goals are defined in base-link spherical coordinates. A manipulation hard goal must
    fall into either the ground-pick arc or the high-reach frontal arc.
    """

    # Self-collision exclusion box in robot base frame.
    safe_buffer = 0.1
    SELF_COLLISION_BOX_X = [-0.25 - safe_buffer, 0.34 + safe_buffer]
    SELF_COLLISION_BOX_Y = [-0.15 - safe_buffer, 0.15 + safe_buffer]
    SELF_COLLISION_BOX_Z = [-0.35 - safe_buffer, 0.27 + safe_buffer]

    FORWARD_X_MIN = 0.12
    FORWARD_YAW_HALF_RAD = math.radians(60.0)

    # Ground-pick: far forward-down reaches (large radius, moderate pitch).
    GROUND_RADIUS_MIN = 0.48
    GROUND_RADIUS_MAX = 0.92
    # clearance above the terrain directly below the target.
    GROUND_WORLD_Z_MIN = 0.05
    GROUND_WORLD_Z_MAX = 0.22

    HIGH_RADIUS_MIN = 0.48
    HIGH_RADIUS_MAX = 0.92
    HIGH_WORLD_Z_MIN = 0.62
    HIGH_WORLD_Z_MAX = 0.78

    GROUND_PITCH_RANGE = (-0.62, -0.22)
    HIGH_PITCH_RANGE = (0.05, 0.62)
    HARD_GOAL_YAW_RANGE = (-FORWARD_YAW_HALF_RAD, FORWARD_YAW_HALF_RAD)

    @classmethod
    def is_in_self_collision_box(cls, pos_b: torch.Tensor) -> torch.Tensor:
        x, y, z = pos_b[:, 0], pos_b[:, 1], pos_b[:, 2]
        in_x = (x > cls.SELF_COLLISION_BOX_X[0]) & (x < cls.SELF_COLLISION_BOX_X[1])
        in_y = (y > cls.SELF_COLLISION_BOX_Y[0]) & (y < cls.SELF_COLLISION_BOX_Y[1])
        in_z = (z > cls.SELF_COLLISION_BOX_Z[0]) & (z < cls.SELF_COLLISION_BOX_Z[1])
        return in_x & in_y & in_z

    @classmethod
    def is_forward_cone_b(cls, pos_b: torch.Tensor) -> torch.Tensor:
        x = pos_b[:, 0]
        y = pos_b[:, 1]
        yaw = torch.atan2(y, x)
        return (x > cls.FORWARD_X_MIN) & (torch.abs(yaw) < cls.FORWARD_YAW_HALF_RAD)

    @classmethod
    def is_ground_pick_goal(cls, pos_b: torch.Tensor, pos_w: torch.Tensor) -> torch.Tensor:
        radius = torch.linalg.norm(pos_b, dim=1)
        return (
            cls.is_forward_cone_b(pos_b)
            & ~cls.is_in_self_collision_box(pos_b)
            & (radius >= cls.GROUND_RADIUS_MIN)
            & (radius <= cls.GROUND_RADIUS_MAX)
            & (pos_w[:, 2] >= cls.GROUND_WORLD_Z_MIN)
            & (pos_w[:, 2] <= cls.GROUND_WORLD_Z_MAX)
        )

    @classmethod
    def is_high_reach_goal(cls, pos_b: torch.Tensor, pos_w: torch.Tensor) -> torch.Tensor:
        radius = torch.linalg.norm(pos_b, dim=1)
        return (
            cls.is_forward_cone_b(pos_b)
            & ~cls.is_in_self_collision_box(pos_b)
            & (radius >= cls.HIGH_RADIUS_MIN)
            & (radius <= cls.HIGH_RADIUS_MAX)
            & (pos_w[:, 2] >= cls.HIGH_WORLD_Z_MIN)
            & (pos_w[:, 2] <= cls.HIGH_WORLD_Z_MAX)
        )

    @classmethod
    def is_manip_hard_goal(cls, pos_b: torch.Tensor, pos_w: torch.Tensor) -> torch.Tensor:
        """Return True when the target lies in either hard manipulation arc."""
        return cls.is_ground_pick_goal(pos_b, pos_w) | cls.is_high_reach_goal(pos_b, pos_w)

    @classmethod
    def is_manip_hard_goal_from_samples(
        cls,
        radius: torch.Tensor,
        yaw: torch.Tensor,
        x_b: torch.Tensor,
        z_rel: torch.Tensor,
        self_collision: torch.Tensor,
    ) -> torch.Tensor:
        forward = (x_b > cls.FORWARD_X_MIN) & (torch.abs(yaw) < cls.FORWARD_YAW_HALF_RAD)
        valid_common = forward & ~self_collision
        ground_pick = (
            (radius >= cls.GROUND_RADIUS_MIN)
            & (radius <= cls.GROUND_RADIUS_MAX)
            & (z_rel >= cls.GROUND_WORLD_Z_MIN)
            & (z_rel <= cls.GROUND_WORLD_Z_MAX)
        )
        high_reach = (
            (radius >= cls.HIGH_RADIUS_MIN)
            & (radius <= cls.HIGH_RADIUS_MAX)
            & (z_rel >= cls.HIGH_WORLD_Z_MIN)
            & (z_rel <= cls.HIGH_WORLD_Z_MAX)
        )
        return valid_common & (ground_pick | high_reach)

    @staticmethod
    def _sample_uniform(
        count: int,
        low: float,
        high: float,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.empty(count, device=device, dtype=dtype).uniform_(low, high)

    @classmethod
    def sample_manip_hard_spherical(
        cls,
        count: int,
        device: torch.device | str,
        dtype: torch.dtype,
        ground_pick_prob: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (radius, pitch, yaw) from hard-goal proposal ranges.

        These ranges are proposals, not guarantees: callers should still pass the result
        through :meth:`is_manip_hard_goal_from_samples` because world-frame height can
        vary with root pose and because self-collision is checked after Cartesian conversion.
        """
        if count <= 0:
            empty = torch.empty(0, device=device, dtype=dtype)
            return empty, empty, empty

        ground_mask = torch.rand(count, device=device) < ground_pick_prob
        high_mask = ~ground_mask

        radius = torch.empty(count, device=device, dtype=dtype)
        pitch = torch.empty(count, device=device, dtype=dtype)
        yaw = cls._sample_uniform(count, *cls.HARD_GOAL_YAW_RANGE, device, dtype)

        num_ground = int(ground_mask.sum().item())
        if num_ground > 0:
            radius[ground_mask] = cls._sample_uniform(
                num_ground, cls.GROUND_RADIUS_MIN, cls.GROUND_RADIUS_MAX, device, dtype
            )
            pitch[ground_mask] = cls._sample_uniform(num_ground, *cls.GROUND_PITCH_RANGE, device, dtype)

        num_high = int(high_mask.sum().item())
        if num_high > 0:
            radius[high_mask] = cls._sample_uniform(num_high, cls.HIGH_RADIUS_MIN, cls.HIGH_RADIUS_MAX, device, dtype)
            pitch[high_mask] = cls._sample_uniform(num_high, *cls.HIGH_PITCH_RANGE, device, dtype)

        return radius, pitch, yaw

    @classmethod
    def sample_manip_hard_radius_height_yaw(
        cls,
        count: int,
        device: torch.device | str,
        dtype: torch.dtype,
        ground_pick_prob: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample hard-goal radius, terrain-relative height, and yaw before solving pitch.

        This keeps the hard-goal distribution independent from the terrain query method:
        callers can bind the sampled height to either an approximate or exact terrain height.
        """
        if count <= 0:
            empty = torch.empty(0, device=device, dtype=dtype)
            return empty, empty, empty

        # Split samples into the two hard-goal families while preserving the existing probability.
        ground_mask = torch.rand(count, device=device) < ground_pick_prob
        high_mask = ~ground_mask

        radius = torch.empty(count, device=device, dtype=dtype)
        desired_z_rel = torch.empty(count, device=device, dtype=dtype)
        yaw = cls._sample_uniform(count, *cls.HARD_GOAL_YAW_RANGE, device, dtype)

        # Ground-pick goals are far forward and close to the local terrain surface.
        num_ground = int(ground_mask.sum().item())
        if num_ground > 0:
            radius[ground_mask] = cls._sample_uniform(
                num_ground, cls.GROUND_RADIUS_MIN, cls.GROUND_RADIUS_MAX, device, dtype
            )
            desired_z_rel[ground_mask] = cls._sample_uniform(
                num_ground, cls.GROUND_WORLD_Z_MIN, cls.GROUND_WORLD_Z_MAX, device, dtype
            )

        # High-reach goals are also far forward, but target a higher terrain-relative band.
        num_high = int(high_mask.sum().item())
        if num_high > 0:
            radius[high_mask] = cls._sample_uniform(num_high, cls.HIGH_RADIUS_MIN, cls.HIGH_RADIUS_MAX, device, dtype)
            desired_z_rel[high_mask] = cls._sample_uniform(
                num_high, cls.HIGH_WORLD_Z_MIN, cls.HIGH_WORLD_Z_MAX, device, dtype
            )

        return radius, desired_z_rel, yaw

    @classmethod
    def sample_manip_hard_spherical_relative_height(
        cls,
        count: int,
        root_z_w: torch.Tensor,
        root_quat_w: torch.Tensor,
        terrain_z_w: torch.Tensor,
        device: torch.device | str,
        dtype: torch.dtype,
        ground_pick_prob: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample hard-goal spheres by first sampling terrain-relative target height.

        Given a target height above terrain, this solves pitch from the current root
        orientation so proposals are already close to the height predicate.
        """
        if count <= 0:
            empty = torch.empty(0, device=device, dtype=dtype)
            return empty, empty, empty

        # Reuse the same hard-goal distribution, then solve pitch against the provided terrain height.
        radius, desired_z_rel, yaw = cls.sample_manip_hard_radius_height_yaw(
            count, device, dtype, ground_pick_prob=ground_pick_prob
        )

        # Isaac Lab quaternions are (w, x, y, z). Only the world-z row is needed.
        quat = root_quat_w.to(dtype=dtype)
        qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        rot_zx = 2.0 * (qx * qz - qw * qy)
        rot_zy = 2.0 * (qy * qz + qw * qx)
        rot_zz = 1.0 - 2.0 * (qx * qx + qy * qy)

        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        horizontal_z_gain = rot_zx * cos_yaw + rot_zy * sin_yaw
        vertical_z_gain = rot_zz

        desired_delta_z = terrain_z_w.to(dtype=dtype) + desired_z_rel - root_z_w.to(dtype=dtype)
        gain_norm = torch.sqrt(horizontal_z_gain.square() + vertical_z_gain.square()).clamp_min(1.0e-6)
        pitch_offset = torch.atan2(horizontal_z_gain, vertical_z_gain)
        pitch_arg = torch.clamp(desired_delta_z / radius / gain_norm, -0.999, 0.999)
        pitch = torch.asin(pitch_arg) - pitch_offset

        pitch_limit = math.pi / 2.0 - 1.0e-4
        pitch = torch.clamp(pitch, -pitch_limit, pitch_limit)

        return radius, pitch, yaw
