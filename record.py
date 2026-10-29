"""Record Go2 + ARX5 policy demonstrations in Isaac Lab."""
import argparse
from pathlib import Path
import yaml
from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", type=Path, default=ROOT/"presets/orbit.yaml")
parser.add_argument("--output", type=Path, default=ROOT/"outputs/demo.mp4")
parser.add_argument("--checkpoint", type=Path)
parser.add_argument("--duration", type=float)
parser.add_argument("--num-envs", type=int)
parser.add_argument("--width", type=int)
parser.add_argument("--height", type=int)
parser.add_argument("--mode", choices=["orbit", "single", "crowd"])
parser.add_argument("--preview-at", type=float, help="Save one PNG at this point on the camera path, after simulation warmup")
parser.add_argument("--apply-materials", action="store_true", help="Update the project USD from go2_arx5/materials.yaml and exit")
preliminary, _ = parser.parse_known_args()
render_defaults = yaml.safe_load(preliminary.config.read_text())["render"]
parser.set_defaults(width=render_defaults["width"], height=render_defaults["height"])
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
launcher = AppLauncher(args, multi_gpu=False)

import imageio.v2 as imageio
import torch
from isaaclab.envs import ManagerBasedRLEnv
from go2_arx5.environment import build_environment
from go2_arx5.policy import Controller
from cinema.camera import Camera
from cinema.lighting import RobotLights
from cinema.scene import configure_scene


def main():
    if args.apply_materials:
        from cinema.materials import apply_arm_materials
        apply_arm_materials(ROOT)
        return
    settings = yaml.safe_load(args.config.read_text())
    for value, section, key in [(args.num_envs, "scene", "num_envs"),
                                (args.width, "render", "width"), (args.height, "render", "height"),
                                (args.mode, "camera", "mode")]:
        if value is not None:
            settings[section][key] = value
    shot = settings["camera"]
    duration = args.duration if args.duration is not None else sum(shot[k] for k in ("closeup_seconds", "move_seconds", "hold_seconds"))
    controller = Controller(args.checkpoint or ROOT/settings["robot"]["checkpoint"], settings["robot"]["actor"], args.device)
    cfg = build_environment(ROOT/settings["robot"]["control"], args.device, settings["scene"]["seed"])
    configure_scene(cfg, settings, duration)
    env = ManagerBasedRLEnv(cfg=cfg, render_mode="rgb_array")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with torch.inference_mode():
            obs, _ = env.reset()
            for _ in range(round(settings["scene"]["warmup"]/env.step_dt)):
                obs, *_ = env.step(controller(obs["policy"]))
            render = settings["render"]
            camera = Camera(env, shot, render["width"], render["height"])
            lights = RobotLights(env, settings["lighting"]["height"])
            camera.update(args.preview_at or 0.)
            lights.update()
            env.render()
            for _ in range(96):
                env.sim.render()
            if args.preview_at is not None:
                imageio.imwrite(args.output, env.render())
                print(f"Preview: {args.output}", flush=True)
                return
            fps = round(1/env.step_dt)
            frames = round(duration/env.step_dt)
            with imageio.get_writer(args.output, fps=fps, codec="libx264", macro_block_size=1,
                                    ffmpeg_params=["-crf", str(render["crf"]), "-preset", render["encoder_preset"],
                                                   "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-threads", "4"]) as writer:
                for i in range(frames):
                    camera.update(i*env.step_dt)
                    obs, *_ = env.step(controller(obs["policy"]))
                    lights.update()
                    writer.append_data(env.render())
                    if (i+1) % fps == 0:
                        print(f"Recording {(i+1)/fps:.0f}/{duration:g} s", flush=True)
            print(f"Video: {args.output}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
    launcher.app.close()
