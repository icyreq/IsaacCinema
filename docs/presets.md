# 模块化拍摄预设

外观、指令可视化和相机相互独立。`presets/*.yaml` 保存一次拍摄的组合，同时独立保存模型、场景数量、控制指令采样周期及画质设置。例如 `presets/blue_grid_orbit.yaml`：

```yaml
appearance: blue_grid
commands: overhead
camera: close_orbit
```

## 使用

在已配置 Isaac Lab 的服务器上：

```bash
cd /home/lx/IsaacCinema
# 蓝黑网格 + 头顶速度箭头 + 近景环绕，1920×1080 / 50 fps。
GPU=2 ./run.sh --config presets/blue_grid_orbit.yaml --output outputs/blue_grid.mp4

# 只更换镜头：侧面固定朝向，跟随所选机器人的位置。
GPU=2 ./run.sh --config presets/blue_grid_orbit.yaml --camera fixed_side --output outputs/side.mp4

# 静态预览；preview-at 选择运镜时间，机器人状态来自预热结束时。
GPU=2 ./run.sh --config presets/blue_grid_orbit.yaml --camera fixed_side --preview-at 0 --output outputs/side.png

# 原有默认：聚光灯、原有地面、关闭指令标记、近景后拉远。
GPU=2 ./run.sh --output outputs/original.mp4

# 任意组合：在蓝黑网格拍摄配置上关闭标记或切换外观。
GPU=2 ./run.sh --config presets/blue_grid_orbit.yaml --commands none --output outputs/clean.mp4
GPU=2 ./run.sh --config presets/blue_grid_orbit.yaml --appearance spotlight --output outputs/spotlight.mp4
```

模块名对应 `presets/appearance/`、`presets/commands/`、`presets/camera/` 中的 YAML。也可以在拍摄配置内直接写一个模块的完整 mapping，或指定 YAML 文件路径。带扩展名的相对路径相对于拍摄配置文件所在目录；绝对路径直接使用。命令行覆盖替换整个模块，不做隐式继承或合并。缺少文件或配置项会报错。

## 外观

- `appearance/blue_grid.yaml`：0.5 m 网格，蓝黑地面，灰蓝网格线，柔和环境光与大角度方向光。
- `appearance/spotlight.yaml`：保留原来的机器人聚光灯和资产自带地面。
- `ground`、`lighting`、`robot_materials` 在外观文件内分开设置。
- 网格按地形的世界 XY 坐标投影，沿用原有地形起伏与碰撞；不覆盖成另一块平地。
- `color_srgb`、`line_color_srgb` 是 0–255 的 sRGB 值；`spacing`、`line_width` 单位为米。
- `robot_materials: asset` 沿用 USD 自带材质。需要覆盖 ARX 材质时，可改成以下 mapping，材质名与 `go2_arx5/materials.yaml` 对应：

```yaml
robot_materials:
  DarkAnodized:
    color: [0.16, 0.17, 0.18]  # 线性 RGB
    metallic: 0.70
    roughness: 0.30
```

材质覆盖只写入运行中的场景，不保存到机器人 USD 文件。程序生成的网格贴图保存在临时目录，录制结束后清理。

## 指令可视化

`commands/overhead.yaml` 分别控制三个开关及尺寸、颜色：

- `velocity`：实际机体平移速度指令，转换到机器人 yaw 对应的世界方向。`anchor: [0.28, 0, 0.50]` 是相对机体原点、随 yaw 旋转的偏移，位于头顶上方。箭头长度 = 速度模长 × `scale_seconds`，零指令不显示直箭头。
- `turn`：实际角速度指令，弧度 = 角速度 × `scale_seconds`；正负号决定方向。弧线位于 `world_height` 指定的地面上方高度。
- `end_effector`：橙色点和 XYZ 姿态轴显示策略当前跟踪的**插值后末端指令** `pos_command_w / quat_command_w`，不是机械臂实际位置，也不是尚未插值到达的最终采样点。

正式录制每帧读取真实指令并更新 USD 实例，不改变控制输出。`commands/none.yaml` 完全关闭这些标记。

## 相机

- `camera/orbit_pullback.yaml`：原有 3 秒近景、7 秒环绕拉远、10 秒固定远景。
- `camera/close_orbit.yaml`：2.2 m 距离、35° 俯角、20 秒完整环绕，跟随机器人位置。
- `camera/fixed_side.yaml`：相同距离和俯角，固定在侧面方向，跟随机器人位置。
- `follow: false` 固定世界空间的运镜路径；`true` 在路径上加上所选机器人自预热结束后的位移，不跟随机身滚转或俯仰。
- `focus` 可选 `nearest`、`manipulation`、`locomotion`，选择符合条件且离世界原点最近的机器人；无匹配对象时明确报错。
- `focal_length: null` 保留资产镜头默认焦距；数值单位为毫米。
- `scene.task_probabilities` 与相机独立；`null` 沿用控制配置的任务分布，`[1, 0]` 为纯操作，`[0, 1]` 为移动操作。单机器人拍摄应明确配好任务分布与 `focus`。

保留旧命令 `--mode single` 的组合含义：固定近景且设置纯操作任务。使用 `--camera fixed_side` 只切换相机，不改控制任务。`--duration` 决定截取多长的视频，不重新缩放模块中的运镜时间。

## 验证

```bash
source /home/lx/IsaacLab/lab/bin/activate
python -m unittest discover -s tests -v
```

测试覆盖模块组合、路径解析、控制配置独立性和标记方向的几何计算。实际画面需通过上面的预览或短视频命令验证。
