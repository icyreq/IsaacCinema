"""World-space camera moves; fixed lens, no image resizing."""
import math
import numpy as np
from pxr import UsdGeom


def smootherstep(t):
    t = np.clip(t, 0., 1.)
    return t * t * t * (t * (6. * t - 15.) + 10.)


class Camera:
    def __init__(self, env, settings, width, height):
        self.settings = settings
        robot = env.scene["robot"]
        positions = robot.data.root_pos_w.cpu().numpy()
        task_ids = env.command_manager.get_term("base_velocity").active_task_buffer.cpu().numpy()
        candidates = np.flatnonzero(task_ids == 0)
        focus = int(candidates[np.argmin(np.linalg.norm(positions[candidates, :2], axis=1))])
        self.target = positions[focus] + np.array([0., 0., settings["target_height"]])
        w, x, y, z = robot.data.root_quat_w[focus].cpu().numpy()
        yaw = math.atan2(2 * (w*z+x*y), 1-2*(y*y+z*z))
        offset = np.asarray(settings["close_direction"], dtype=float)
        offset *= settings["close_distance"] / np.linalg.norm(offset)
        self.near = np.array([math.cos(yaw)*offset[0]-math.sin(yaw)*offset[1],
                              math.sin(yaw)*offset[0]+math.cos(yaw)*offset[1], offset[2]])
        lateral = np.array([-self.near[1], self.near[0], 0.])
        lateral /= np.linalg.norm(lateral)
        self.end_target = self.target + settings["lateral_distance"] * lateral
        self.end_distance = math.hypot(settings["end_horizontal_distance"], settings["end_height"])
        self.end_pitch = math.atan2(settings["end_height"], settings["end_horizontal_distance"])
        camera = UsdGeom.Camera(env.sim.stage.GetPrimAtPath(env.cfg.viewer.cam_prim_path))
        camera.GetHorizontalApertureAttr().Set(36.)
        camera.GetVerticalApertureAttr().Set(36. * height / width)
        camera.GetClippingRangeAttr().Set((.05, 3000.))
        self.env = env

    def pose(self, t):
        s = self.settings
        if s["mode"] == "single":
            phase = 0.
        elif s["mode"] == "crowd":
            phase = 1.
        else:
            phase = smootherstep((t - s["closeup_seconds"]) / s["move_seconds"])
        target = (1.-phase)*self.target + phase*self.end_target
        near_distance = np.linalg.norm(self.near)
        pitch = (1.-phase)*np.arctan2(self.near[2], np.linalg.norm(self.near[:2])) + phase*self.end_pitch
        yaw = np.arctan2(self.near[1], self.near[0]) + phase*np.deg2rad(s["orbit_degrees"])
        distance = np.exp((1.-phase)*np.log(near_distance) + phase*np.log(self.end_distance))
        direction = np.array([np.cos(pitch)*np.cos(yaw), np.cos(pitch)*np.sin(yaw), np.sin(pitch)])
        return target + distance*direction, target

    def update(self, t):
        eye, target = self.pose(t)
        self.env.sim.set_camera_view(eye, target)
