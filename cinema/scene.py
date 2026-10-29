"""Apply the filming layout and render settings to the robot environment."""
import math
from .lighting import configure_lighting


def configure_scene(cfg, settings, duration):
    scene, render = settings["scene"], settings["render"]
    cfg.scene.num_envs = scene["num_envs"]
    cfg.scene.env_spacing = scene["spacing"]
    cfg.scene.terrain.use_terrain_origins = False
    cfg.scene.terrain.env_spacing = scene["spacing"]
    terrain = cfg.scene.terrain.terrain_generator
    side = math.ceil(math.sqrt(scene["num_envs"]))
    terrain.num_rows = terrain.num_cols = math.ceil((side*scene["spacing"]+12)/terrain.size[0])
    cfg.commands.base_velocity.debug_vis = False
    cfg.commands.ee_pose.debug_vis = False
    cfg.commands.base_velocity.resampling_time_range = tuple(scene["velocity_command_interval"])
    cfg.commands.ee_pose.resampling_time_range = tuple(scene["ee_command_interval"])
    if settings["camera"]["mode"] == "single":
        cfg.commands.base_velocity.probs = [1., 0.]
    configure_lighting(cfg, settings["lighting"])
    cfg.sim.render.enable_direct_lighting = True
    cfg.sim.render.enable_shadows = True
    cfg.sim.render.enable_reflections = True
    cfg.sim.render.enable_global_illumination = True
    cfg.sim.render.samples_per_pixel = render["samples_per_pixel"]
    cfg.sim.render.antialiasing_mode = "TAA"
    cfg.sim.render.enable_dlssg = False
    cfg.sim.render.carb_settings = {"/rtx/sceneDb/ambientLightIntensity": 0.0,
                                   "/rtx/scenedb/skipMostLights": False}
    cfg.viewer.origin_type = "world"
    cfg.viewer.resolution = (render["width"], render["height"])
    cfg.viewer.eye = (3., -3., 2.)
    cfg.viewer.lookat = (0., 0., .5)
    cfg.episode_length_s = max(cfg.episode_length_s, scene["warmup"]+duration+cfg.sim.dt*cfg.decimation)
