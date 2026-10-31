"""Vector geometry shared by the command renderer and numerical tests."""
import numpy as np


def rotate(quaternions, vectors):
    """Apply unit wxyz quaternions to broadcast-compatible XYZ vectors."""
    q = np.asarray(quaternions)
    t = 2*np.cross(q[..., 1:], vectors)
    return vectors + q[..., :1]*t + np.cross(q[..., 1:], t)


def segment_pose(start, end, radius):
    """Map a unit +X cylinder to each segment, including antiparallel segments."""
    delta = np.asarray(end)-start
    length = np.linalg.norm(delta, axis=-1)
    direction = np.zeros_like(delta)
    np.divide(delta, length[..., None], out=direction, where=length[..., None] > 0)
    quaternion = np.stack([1+direction[..., 0], np.zeros_like(length), -direction[..., 2], direction[..., 1]], axis=-1)
    opposite = direction[..., 0] < -1+1e-12
    quaternion[opposite] = [0., 0., 0., 1.]
    quaternion /= np.linalg.norm(quaternion, axis=-1, keepdims=True)
    scales = np.stack([length, np.full_like(length, radius), np.full_like(length, radius)], axis=-1)
    return quaternion, scales


def yaw_rotation(quaternions):
    q = np.asarray(quaternions)
    w, x, y, z = np.moveaxis(q, -1, 0)
    yaw = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    flat = np.stack([np.cos(yaw/2), np.zeros_like(yaw), np.zeros_like(yaw), np.sin(yaw/2)], axis=-1)
    return yaw, flat
