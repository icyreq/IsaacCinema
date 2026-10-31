"""Render-only terrain and robot materials; no changes to collision geometry."""
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade
import isaaclab.sim as sim


def surface(stage, path, color, roughness=.85, metallic=0., emission=0.):
    sim.spawn_preview_surface(path, sim.PreviewSurfaceCfg(
        diffuse_color=tuple(color), roughness=roughness, metallic=metallic,
        emissive_color=tuple(c*emission for c in color)))
    return UsdShade.Material(stage.GetPrimAtPath(path))


class Appearance:
    def __init__(self, env, settings):
        self.textures = TemporaryDirectory(prefix='isaaccinema-materials-')
        self.stage = env.sim.stage
        ground = settings['ground']
        if ground['type'] == 'grid':
            self._grid(env.cfg.scene.terrain.prim_path, ground)
        elif ground['type'] != 'asset':
            raise ValueError(f"Unknown ground type: {ground['type']}")
        # Override ARX finishes in the live stage only. Source USD layers stay intact.
        finishes = settings['robot_materials']
        if finishes != 'asset':
            found = set()
            for prim in self.stage.Traverse():
                if prim.IsA(UsdShade.Shader) and prim.GetName() == 'Surface':
                    name = prim.GetParent().GetName().removeprefix('ARX_')
                    if name in finishes:
                        shader = UsdShade.Shader(prim); finish = finishes[name]
                        shader.GetInput('diffuseColor').Set(Gf.Vec3f(*finish['color']))
                        shader.GetInput('metallic').Set(finish['metallic'])
                        shader.GetInput('roughness').Set(finish['roughness'])
                        found.add(name)
            if found != set(finishes):
                raise ValueError(f'ARX materials missing in stage: {set(finishes)-found}')

    def _grid(self, terrain_path, cfg):
        spacing, width = cfg['spacing'], cfg['line_width']
        if not 0 < width < spacing:
            raise ValueError('Grid line_width must be between zero and spacing')
        n = cfg['texture_resolution']
        coords = (np.arange(n)+.5) / n
        on_line = np.minimum(coords, 1-coords) < width/spacing/2
        pixels = np.empty((n, n, 3), dtype=np.uint8)
        pixels[:] = cfg['color_srgb']; pixels[on_line, :] = cfg['line_color_srgb']; pixels[:, on_line] = cfg['line_color_srgb']
        texture = Path(self.textures.name)/'grid.png'
        Image.fromarray(pixels).save(texture)
        mat = surface(self.stage, '/World/CinemaMaterials/Grid', (0., 0., 0.), cfg['roughness'])
        shader = UsdShade.Shader(self.stage.GetPrimAtPath(str(mat.GetPath())+'/Shader'))
        tex = UsdShade.Shader.Define(self.stage, str(mat.GetPath())+'/Texture')
        tex.CreateIdAttr('UsdUVTexture')
        tex.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(str(texture))
        tex.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set('sRGB')
        for axis in ('S', 'T'):
            tex.CreateInput('wrap'+axis, Sdf.ValueTypeNames.Token).Set('repeat')
        reader = UsdShade.Shader.Define(self.stage, str(mat.GetPath())+'/UV')
        reader.CreateIdAttr('UsdPrimvarReader_float2')
        reader.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
        reader.CreateOutput('result', Sdf.ValueTypeNames.Float2)
        tex.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), 'result')
        tex.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)
        shader.GetInput('diffuseColor').ConnectToSource(tex.ConnectableAPI(), 'rgb')
        terrain = self.stage.GetPrimAtPath(terrain_path)
        if not terrain:
            raise ValueError(f'Missing terrain: {terrain_path}')
        transforms = UsdGeom.XformCache()
        count = 0
        for prim in Usd.PrimRange(terrain):
            if prim.IsA(UsdGeom.Mesh):
                points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get())
                transform = np.asarray(transforms.GetLocalToWorldTransform(prim))
                world = np.c_[points, np.ones(len(points))] @ transform
                uv = UsdGeom.PrimvarsAPI(prim).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
                uv.Set([Gf.Vec2f(float(p[0]/spacing), float(p[1]/spacing)) for p in world])
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.strongerThanDescendants)
                count += 1
        if not count:
            raise ValueError('Grid appearance requires a mesh terrain')

    def close(self):
        self.textures.cleanup()
