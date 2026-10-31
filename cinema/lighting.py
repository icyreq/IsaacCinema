"""Independent soft studio lighting and per-robot downlights."""
from pxr import Gf, UsdGeom, UsdLux
from isaaclab.assets import AssetBaseCfg
import isaaclab.sim as sim


def configure_lighting(cfg, settings):
    cfg.scene.dome_light.spawn.intensity = settings['ambient']
    cfg.scene.dome_light.spawn.color = tuple(settings['ambient_color'])
    if settings['mode'] == 'studio':
        cfg.scene.filming_key = AssetBaseCfg(
            prim_path='/World/FilmingKey',
            spawn=sim.DistantLightCfg(intensity=settings['intensity'],
                                     color=tuple(settings['color']), angle=settings['angle']),
            init_state=AssetBaseCfg.InitialStateCfg(rot=tuple(settings['orientation'])),
        )
    elif settings['mode'] == 'robot_spotlights':
        def spawn(prim_path, light_cfg, translation=None, orientation=None, **kwargs):
            prim = sim.spawn_light(prim_path, light_cfg, translation, orientation, **kwargs)
            shape = UsdLux.ShapingAPI.Apply(prim)
            shape.CreateShapingConeAngleAttr(settings['cone_angle'])
            shape.CreateShapingConeSoftnessAttr(settings['cone_softness'])
            return prim
        cfg.scene.robot_spotlight = AssetBaseCfg(
            prim_path='{ENV_REGEX_NS}/RobotSpotlight',
            spawn=sim.DiskLightCfg(func=spawn, intensity=settings['intensity'],
                                  radius=settings['radius'], normalize=True, color=tuple(settings['color'])),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0., 0., settings['height']+.32)),
        )
    else:
        raise ValueError(f"Unknown lighting mode: {settings['mode']}")


class RobotLights:
    def __init__(self, env, height):
        self.env, self.height = env, height
        self.origins = env.scene.env_origins.cpu().numpy()
        self.ops = []
        for i in range(env.num_envs):
            prim = env.sim.stage.GetPrimAtPath(f'/World/envs/env_{i}/RobotSpotlight')
            self.ops.append(next(op for op in UsdGeom.Xformable(prim).GetOrderedXformOps()
                                 if op.GetOpType() == UsdGeom.XformOp.TypeTranslate))

    def update(self):
        positions = self.env.scene['robot'].data.root_pos_w.cpu().numpy()
        for op, position, origin in zip(self.ops, positions, self.origins):
            p = position - origin
            op.Set(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])+self.height))
