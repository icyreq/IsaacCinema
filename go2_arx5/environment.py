"""Build the current robot environment from its control configuration."""
from pathlib import Path
from dataclasses import is_dataclass
from types import SimpleNamespace
import yaml

import isaaclab.sim as sim
import isaaclab.terrains as terrains
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import JointPositionActionCfg, UniformVelocityCommandCfg
from isaaclab.managers import (
    ObservationGroupCfg, ObservationTermCfg, EventTermCfg, RewardTermCfg,
    TerminationTermCfg, CurriculumTermCfg, SceneEntityCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import string_to_callable
from isaaclab.utils.noise import GaussianNoiseCfg

from .actuator import UnitreeGo2ActuatorCfg
from .mdp.commands.config import TaskAwareVelocityCommandCfg, SphericalInterpolationCommandCfg

ROOT = Path(__file__).resolve().parents[1]


class ControlConfigLoader(yaml.FullLoader):
    pass


ControlConfigLoader.add_constructor(
    "tag:yaml.org,2002:python/object/apply:builtins.slice",
    lambda loader, node: slice(*loader.construct_sequence(node)),
)


def configure(instance, values):
    for key, value in values.items():
        current = instance.get(key) if isinstance(instance, dict) else getattr(instance, key)
        if isinstance(value, dict) and is_dataclass(current):
            configure(current, value)
            continue
        if isinstance(value, dict) and isinstance(current, dict) and any(is_dataclass(v) for v in current.values()):
            for name, fields in value.items():
                configure(current[name], fields)
            continue
        if callable(current) and isinstance(value, str):
            value = string_to_callable(value)
        if isinstance(instance, dict):
            instance[key] = value
        else:
            setattr(instance, key, value)
    return instance


def term(kind, values):
    values = dict(values)
    values["func"] = string_to_callable(values["func"])
    values["params"] = {
        key: SceneEntityCfg(**value) if key in ("asset_cfg", "sensor_cfg") else value
        for key, value in values.get("params", {}).items()
    }
    if values.get("noise") is not None:
        values["noise"] = configure(GaussianNoiseCfg(), values["noise"])
    return kind(**values)


def build_environment(control_file, device, seed):
    saved = yaml.load(Path(control_file).read_text(), Loader=ControlConfigLoader)
    cfg = ManagerBasedRLEnvCfg()
    cfg.seed = seed
    managers = {"scene", "observations", "actions", "commands", "events", "rewards", "terminations", "curriculum"}
    configure(cfg, {key: value for key, value in saved.items() if key not in managers})
    cfg.seed = seed
    cfg.sim.device = device

    cfg.scene = InteractiveSceneCfg(num_envs=1, env_spacing=3.)
    cfg.scene.robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim.UsdFileCfg(
            usd_path=str(ROOT / "assets/go2_arx5/go2_arx5_stanford.usd"),
            rigid_props=sim.RigidBodyPropertiesCfg(),
            articulation_props=sim.ArticulationRootPropertiesCfg(),
        ),
        actuators={"legs": UnitreeGo2ActuatorCfg(joint_names_expr=[".*"]),
                   "arm": DelayedPDActuatorCfg(joint_names_expr=["joint[1-6]"])},
    )
    cfg.scene.contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*")
    cfg.scene.terrain = terrains.TerrainImporterCfg(
        terrain_generator=terrains.TerrainGeneratorCfg(sub_terrains={
            "flat": terrains.MeshPlaneTerrainCfg(), "random_rough": terrains.HfRandomUniformTerrainCfg(),
        }),
        visual_material=sim.PreviewSurfaceCfg(),
        physics_material=sim.RigidBodyMaterialCfg(),
    )
    cfg.scene.dome_light = AssetBaseCfg(prim_path="/World/DomeLight", spawn=sim.DomeLightCfg())
    configure(cfg.scene, saved["scene"])
    cfg.scene.robot.spawn.usd_path = str(ROOT / "assets/go2_arx5/go2_arx5_stanford.usd")

    cfg.observations = {}
    for name, values in saved["observations"].items():
        group = ObservationGroupCfg()
        for key, value in values.items():
            setattr(group, key, term(ObservationTermCfg, value) if isinstance(value, dict) and "func" in value else value)
        cfg.observations[name] = group
    cfg.actions = {name: configure(JointPositionActionCfg(), value) for name, value in saved["actions"].items()}
    cfg.commands = SimpleNamespace(**{
        "base_velocity": configure(TaskAwareVelocityCommandCfg(ranges=UniformVelocityCommandCfg.Ranges()), saved["commands"]["base_velocity"]),
        "ee_pose": configure(SphericalInterpolationCommandCfg(), saved["commands"]["ee_pose"]),
    })
    cfg.commands.ee_pose.reachable_pose_set_path = str(ROOT / "assets/reachable_tcp_pose_set.pt")
    for name, kind in [("events", EventTermCfg), ("rewards", RewardTermCfg),
                       ("terminations", TerminationTermCfg), ("curriculum", CurriculumTermCfg)]:
        setattr(cfg, name, {
            key: term(kind, value) for key, value in saved[name].items()
            if value is not None and not (name == "rewards" and value["weight"] == 0.)
        })
    for name in ("observations", "actions", "events", "rewards", "terminations", "curriculum"):
        setattr(cfg, name, SimpleNamespace(**getattr(cfg, name)))
    return cfg
