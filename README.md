# IsaacCinema — Go2 + ARX5

独立的 Isaac Lab 拍摄项目。加载现有策略做推理，使用原生渲染输出视频；没有训练入口、PPO、优化器、训练缓存或自动复现记录。

默认效果沿用已完成的 4K demo：4096 个机器人、黑银 ARX 材质、每个机器人一盏柔光灯、暗环境、原生 3840×2160 / 50 fps。

## 开始录制

Git 仓库包含代码、配置和机器人 USD 资产，不包含 checkpoint、目标集合和录制输出。首次克隆后，请从当前拍摄环境或你自己的控制项目复制以下文件到仓库中的对应位置：

- `models/best_agent.pt`：当前 Go2 + ARX5 策略 checkpoint，也可以使用 `--checkpoint` 指定外部文件。
- `assets/reachable_tcp_pose_set.pt`：与当前控制配置匹配的可达目标集合。

当前 4090 上这两个文件已位于 `/home/lx/IsaacCinema/` 的对应目录中；从其他机器迁移时，可用 `scp 4090-20:/home/lx/IsaacCinema/assets/reachable_tcp_pose_set.pt assets/` 复制目标集合。模型目标目录可先用 `mkdir -p models` 创建。

在 4090 上：

```bash
cd /home/lx/IsaacCinema
./run.sh --output outputs/demo_4k.mp4
```

默认镜头：前 3 秒近景，随后 7 秒拉远、绕转 90° 并横移 8 米，最后固定相机继续仿真 10 秒。输出总长 20 秒。

单狗固定视角和群体固定视角：

```bash
./run.sh --mode single --num-envs 1 --duration 20 --output outputs/single.mp4
./run.sh --mode crowd --duration 20 --output outputs/crowd.mp4
```

低分辨率试录及静态构图预览：

```bash
./run.sh --width 1920 --height 1080 --output outputs/demo_1080p.mp4
./run.sh --preview-at 0 --output outputs/close.png
./run.sh --preview-at 10 --output outputs/crowd.png
```

`--preview-at` 只选择镜头路径上的位置；机器人姿态来自预热结束时。正式视频中每帧都推进真实仿真。

用其他显卡启动：

```bash
GPU=1 ./run.sh --output outputs/demo_gpu1.mp4
```

## 调整拍摄

编辑 `presets/orbit.yaml`。运镜时长、近景距离、旋转角度、平移距离、灯光、场景数量和画质参数都在这一份文件中。

- `camera.mode`：`orbit` / `single` / `crowd`。
- `camera.closeup_seconds`、`move_seconds`、`hold_seconds`：近景、运镜、固定远景时间。
- `render.width`、`height`：直接设置原生渲染尺寸，没有后期放大。
- `render.samples_per_pixel`：直接光照采样数，默认 16。
- `render.crf`、`encoder_preset`：默认 H.264、CRF 14、medium。

当前机器人控制步长为 0.02 秒，因此每次控制和渲染输出一帧，对应 50 fps。

## 换模型和控制逻辑

```bash
./run.sh --checkpoint /absolute/path/best_agent.pt --output outputs/new_policy.mp4
```

当前支持 Go2 + ARX5 的双分支策略 checkpoint，读取其中 `policy` 的 `actor.*` 权重和 `observation_preprocessor`。critic 和 optimizer 状态不会参与推理。

同结构模型只需替换 checkpoint。网络层宽变化时修改 YAML 的 `robot.actor`；观测、动作语义或网络结构变化时，同步修改 `go2_arx5/` 中相应代码和配置。

`go2_arx5/environment.yaml` 是运行模型所需的控制配置，包括观测顺序与历史、执行器、动作映射、随机化、目标采样和重置条件。环境保留了原有任务调度所需的计算，但不进行学习。

## 材质

`assets/go2_arx5/` 已包含机器狗原材质和黑银 ARX 材质，录制时直接加载。请保留整个目录及其中的 `configuration/` USD 图层。

调整机械臂配色、金属度或粗糙度：编辑 `go2_arx5/materials.yaml`，然后显式应用一次：

```bash
./run.sh --apply-materials
```

这个命令只更新本项目的 ARX 材质。银色盖板为 link3，外壳为 link2，夹爪为 link6，其余机械臂关节使用 DarkAnodized。

## 文件布局

```text
record.py                 录制入口
run.sh                    激活 Isaac Lab 环境并选择显卡
presets/orbit.yaml        拍摄设置
cinema/                   相机、灯光、场景和材质工具
go2_arx5/                 策略推理及当前机器人的仿真控制代码
models/best_agent.pt      模型
assets/go2_arx5/           完整 USD 资产
assets/reachable_tcp_pose_set.pt  控制所需的可达目标集合
outputs/                  视频或显式请求的预览图
```

项目可以整体移动，不引用旧拍摄项目或 H100 的文件。使用现有 Isaac Lab 2.3.2 / Isaac Sim 5.1 环境；`run.sh` 中默认环境路径对应当前 4090。换机器时可通过 `ISAACLAB_ROOT`、`ISAAC_RUNTIME`、`ISAAC_ASSETS` 环境变量指定安装位置。`TMPDIR` 默认使用当前服务器可写的 `/home/lx/tmp`，迁移时可按需指定。
