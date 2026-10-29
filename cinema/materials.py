"""Edit the already bound ARX materials in this project's USD asset."""
import yaml
from pxr import Usd, UsdShade, Gf


def apply_arm_materials(root):
    path = root/"assets/go2_arx5/configuration/go2_arx5_stanford_base.usd"
    stage = Usd.Stage.Open(str(path))
    finishes = yaml.safe_load((root/"go2_arx5/materials.yaml").read_text())
    for name, finish in finishes.items():
        shader = UsdShade.Shader(stage.GetPrimAtPath(f"/go2_arx5_stanford/Looks/ARX_{name}/Surface"))
        shader.GetInput("diffuseColor").Set(Gf.Vec3f(*finish["color"]))
        shader.GetInput("metallic").Set(finish["metallic"])
        shader.GetInput("roughness").Set(finish["roughness"])
    stage.GetRootLayer().Save()
    print(f"Updated ARX materials: {path}", flush=True)
