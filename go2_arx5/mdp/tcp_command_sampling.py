"""Reachable target sampling used by the Go2 + ARX5 controller."""
from pathlib import Path
import torch
from isaaclab.utils.math import quat_apply, quat_from_angle_axis, quat_mul


def load_reachable_tcp_pose_set(path, device):
    data = torch.load(Path(path), map_location="cpu", weights_only=False)
    return data["tcp_pos_b"].to(device), data["tcp_quat_b"].to(device)

def sample_reachable_tcp_poses(
    reachable_tcp_pos_b: torch.Tensor,
    reachable_tcp_quat_b: torch.Tensor,
    num_samples: int,
    device: str | torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniformly sample TCP target poses from the loaded reachable pose tensors."""
    if num_samples < 0:
        raise ValueError(f"num_samples must be non-negative, got {num_samples}")
    if reachable_tcp_pos_b.shape[0] != reachable_tcp_quat_b.shape[0]:
        raise ValueError(
            "Reachable TCP pose tensors must have the same length: "
            f"pos={reachable_tcp_pos_b.shape[0]}, quat={reachable_tcp_quat_b.shape[0]}"
        )

    # Draw indices on the tensor's resident device, then return command-ready dtype/device tensors.
    pose_count = reachable_tcp_pos_b.shape[0]
    sample_indices = torch.randint(pose_count, (num_samples,), device=reachable_tcp_pos_b.device)
    target_pos_b = reachable_tcp_pos_b[sample_indices].to(device=device, dtype=dtype)
    target_quat_b = reachable_tcp_quat_b[sample_indices].to(device=device, dtype=dtype)
    return target_pos_b, target_quat_b


def _sample_uniform_ball(
    num_samples: int,
    radius: float,
    device: str | torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Sample vectors uniformly by volume from a three-dimensional ball."""
    if radius < 0.0:
        raise ValueError(f"Ball radius must be non-negative, got {radius}.")
    if radius == 0.0:
        return torch.zeros((num_samples, 3), device=device, dtype=dtype)

    directions = torch.nn.functional.normalize(
        torch.randn((num_samples, 3), device=device, dtype=dtype),
        dim=-1,
    )
    radii = radius * torch.rand(num_samples, device=device, dtype=dtype).pow(1.0 / 3.0)
    return directions * radii.unsqueeze(-1)


def apply_local_pose_perturbations(
    seed_pos_b: torch.Tensor,
    seed_quat_b: torch.Tensor,
    delta_pos_local: torch.Tensor,
    delta_rotvec_local: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply translation and rotation-vector perturbations in each seed TCP frame."""
    if seed_pos_b.shape != delta_pos_local.shape or seed_pos_b.shape[-1] != 3:
        raise ValueError(
            "seed_pos_b and delta_pos_local must have matching [N, 3] shapes, got "
            f"{tuple(seed_pos_b.shape)} and {tuple(delta_pos_local.shape)}."
        )
    if delta_rotvec_local.shape != seed_pos_b.shape:
        raise ValueError(
            "delta_rotvec_local must match seed_pos_b, got "
            f"{tuple(delta_rotvec_local.shape)} and {tuple(seed_pos_b.shape)}."
        )
    if seed_quat_b.shape != (seed_pos_b.shape[0], 4):
        raise ValueError(f"seed_quat_b must have shape [N, 4], got {tuple(seed_quat_b.shape)}.")

    target_pos_b = seed_pos_b + quat_apply(seed_quat_b, delta_pos_local)
    delta_angle = torch.linalg.norm(delta_rotvec_local, dim=-1)
    delta_quat_local = quat_from_angle_axis(delta_angle, delta_rotvec_local)
    target_quat_b = quat_mul(seed_quat_b, delta_quat_local)
    return target_pos_b, target_quat_b


def sample_local_pose_perturbations(
    seed_pos_b: torch.Tensor,
    seed_quat_b: torch.Tensor,
    position_radius_m: float,
    orientation_radius_rad: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample local-frame pose perturbations around a batch of reachable TCP poses."""
    num_samples = seed_pos_b.shape[0]
    delta_pos_local = _sample_uniform_ball(
        num_samples,
        position_radius_m,
        seed_pos_b.device,
        seed_pos_b.dtype,
    )
    delta_rotvec_local = _sample_uniform_ball(
        num_samples,
        orientation_radius_rad,
        seed_pos_b.device,
        seed_pos_b.dtype,
    )
    return apply_local_pose_perturbations(seed_pos_b, seed_quat_b, delta_pos_local, delta_rotvec_local)


def cartesian_to_spherical(pos_b: torch.Tensor) -> torch.Tensor:
    """Convert base-frame xyz targets to the command's [radius, pitch, yaw] convention."""
    if pos_b.ndim != 2 or pos_b.shape[1] != 3:
        raise ValueError(f"Expected base-frame positions with shape [N, 3], got {tuple(pos_b.shape)}")

    xy_radius = torch.linalg.norm(pos_b[:, :2], dim=-1)
    radius = torch.linalg.norm(pos_b, dim=-1)
    pitch = torch.atan2(pos_b[:, 2], xy_radius)
    yaw = torch.atan2(pos_b[:, 1], pos_b[:, 0])
    return torch.stack((radius, pitch, yaw), dim=-1)
