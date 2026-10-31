"""Instanced, live visualization of actual base and end-effector commands."""
import numpy as np
from pxr import Gf, UsdGeom, UsdShade
from .appearance import surface
from .overlay_geometry import rotate, segment_pose, yaw_rotation


class CommandOverlays:
    def __init__(self, env, settings):
        self.env, self.settings = env, settings
        stage = env.sim.stage
        root = '/World/CinemaCommands'
        self.instances = UsdGeom.PointInstancer.Define(stage, root)
        self.prototypes = []
        colors = [('Velocity', settings['velocity']['color']),
                  ('Goal', settings['end_effector']['color']),
                  ('AxisX', settings['end_effector']['axis_colors'][0]),
                  ('AxisY', settings['end_effector']['axis_colors'][1]),
                  ('AxisZ', settings['end_effector']['axis_colors'][2])]
        for name, color in colors:
            mat = surface(stage, f'/World/CinemaMaterials/{name}', color, emission=settings['emission'])
            kinds = ('sphere',) if name == 'Goal' else ('shaft', 'head')
            for kind in kinds:
                path = root+'/prototypes/'+name+kind
                if kind == 'sphere':
                    prim = UsdGeom.Sphere.Define(stage, path); prim.CreateRadiusAttr(1.)
                else:
                    angle = np.linspace(0, 2*np.pi, 12, endpoint=False)
                    points = []
                    for x, r in [(0., 1.), (1., 0. if kind == 'head' else 1.)]:
                        points.extend(np.stack([np.full(12, x), r*np.cos(angle), r*np.sin(angle)], axis=1).tolist())
                    indices = [j for i in range(12) for j in (i, (i+1)%12, (i+1)%12+12, i+12)]
                    prim = UsdGeom.Mesh.Define(stage, path)
                    prim.CreatePointsAttr(points); prim.CreateFaceVertexCountsAttr([4]*12)
                    prim.CreateFaceVertexIndicesAttr(indices); prim.CreateSubdivisionSchemeAttr('none')
                UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(mat)
                self.prototypes.append(path)
        self.instances.CreatePrototypesRel().SetTargets(self.prototypes)

    def update(self):
        cfg = self.settings
        robot = self.env.scene['robot']
        pos = robot.data.root_pos_w.cpu().numpy()
        yaw, flat = yaw_rotation(robot.data.root_quat_w.cpu().numpy())
        velocity = self.env.command_manager.get_term('base_velocity').command.cpu().numpy()
        ee = self.env.command_manager.get_term('ee_pose')
        blocks = []

        def rods(a, b, radius, prototype):
            a, b = np.broadcast_arrays(a, b)
            a, b = a.reshape(-1, 3), b.reshape(-1, 3)
            quat, scale = segment_pose(a, b, radius)
            visible = scale[:, 0] > 0
            blocks.append((a[visible], quat[visible], scale[visible], np.full(visible.sum(), prototype)))

        def arrows(a, vectors, radius, prototype):
            length = np.linalg.norm(vectors, axis=-1, keepdims=True)
            direction = np.zeros_like(vectors)
            np.divide(vectors, length, out=direction, where=length > 0)
            end = a+vectors
            neck = end-direction*np.minimum(cfg['arrow_head_length'], length*.3)
            rods(a, neck, radius, prototype)
            rods(neck, end, radius*3, prototype+1)

        if cfg['velocity']['enabled']:
            v = np.c_[velocity[:, :2], np.zeros(len(pos))]
            anchor = pos+rotate(flat, np.asarray(cfg['velocity']['anchor']))
            arrows(anchor, rotate(flat, v)*cfg['velocity']['scale_seconds'], cfg['velocity']['radius'], 0)
        if cfg['turn']['enabled']:
            turn = cfg['turn']; span = velocity[:, 2]*turn['scale_seconds']
            center = pos.copy(); center[:, 2] = turn['world_height']
            angles = yaw[:, None]+turn['start_angle']+span[:, None]*np.linspace(0, 1, turn['segments']+1)
            points = center[:, None, :]+turn['radius']*np.stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=-1)
            moving = span != 0
            rods(points[moving, :-1], points[moving, 1:], turn['line_radius'], 0)
            tangent = np.sign(span)[:, None]*np.stack([-np.sin(angles[:, -1]), np.cos(angles[:, -1]), np.zeros(len(pos))], axis=-1)
            rods(points[moving, -1]-tangent[moving]*.08, points[moving, -1]+tangent[moving]*.04, turn['line_radius']*3, 1)
        if cfg['end_effector']['enabled']:
            goal = ee.pos_command_w.cpu().numpy(); q = ee.quat_command_w.cpu().numpy()
            settings = cfg['end_effector']
            blocks.append((goal, np.tile([1., 0., 0., 0.], (len(goal), 1)),
                           np.full_like(goal, settings['radius']), np.full(len(goal), 2)))
            for axis in range(3):
                arrows(goal, rotate(q, np.eye(3)[axis])*settings['axis_length'], settings['axis_radius'], 3+axis*2)
        if not blocks:
            self.instances.GetProtoIndicesAttr().Set([])
            return
        positions, quaternions, scales, indices = [np.concatenate(items) for items in zip(*blocks)]
        self.instances.GetProtoIndicesAttr().Set(indices.astype(np.int32).tolist())
        self.instances.GetPositionsAttr().Set(positions.astype(np.float32))
        self.instances.GetOrientationsAttr().Set([Gf.Quath(float(q[0]), Gf.Vec3h(*q[1:].astype(float))) for q in quaternions])
        self.instances.GetScalesAttr().Set(scales.astype(np.float32))
