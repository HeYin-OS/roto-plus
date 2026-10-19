# Roto-Plus

基于笔画的视频转描（Rotoscoping）：用户只在首帧画一笔，系统自动将笔画沿视频传播，并贴合目标边缘。

## 项目简介

整条流水线由四个各司其职的小模型/算法组成：

| 环节 | 模型 | 作用 |
|---|---|---|
| 目标检测 | **YOLO11m** (COCO 预训练) | 文本 → 目标框，替代 Florence-2 |
| 目标分割 | **SAM2.1** (视频传播) | 框 → 全序列二值 mask |
| 光流估计 | **RAFT-Large** (torchvision) | 相邻帧位移场，笔画粗传播 |
| 边缘贴合 | **DP + FDoG**（核心算法） | 在光流基础上把笔画吸附到目标边缘 |

主要贡献：
- **检测-分割-光流-贴边四段式流水线**：每个环节可独立替换与评测（`experiment/compare_detectors.py` 提供了 YOLO vs Florence-2 的 GT-free 对比，结论：box IoU≥0.97、下游 mask IoU≥0.999，YOLO 快 2~5 倍）。
- **SAM2 mask 的 Anchor & Grow 后处理**：双阈值（高阈值核心 + 低阈值候选）配合连通域条件生长，切断与背景的粘连、保留完整轮廓。
- **DP 能量最小化 + FDoG 边缘响应的笔画贴边算法**：在光流传播的基础上，用动态规划沿笔画采样候选点使能量最小，将粗笔画吸附到真实目标轮廓；辅以拓扑修复（自交叉检测、方向修正、点数规整）。

代码结构：

```
reasoning/   # 模型推理：YOLO 检测、SAM2 mask、RAFT 光流
utils/       # 核心算法：DP+FDoG 贴边、KDTree、视频处理
experiment/  # 实验：主传播策略对比、检测器对比、误差曲线
stroke/      # 笔画数据与绘制工具
tests/       # 68 项测试（单元 + 真数据集成）
```

## 部署环境

要求：Windows / Linux，Python 3.10，NVIDIA GPU（CUDA 12.1），[uv](https://docs.astral.sh/uv/)。

```bash
uv run python scripts/init_sam2.py   # 拉取 sam2_git（钉在验证过的 commit）+ 下载 checkpoint
uv sync                             # torch 2.5.1+cu121、SAM2（本地 sam2_git editable）等全部就位
uv run pytest                       # 68 项测试（含 CUDA 检查、真数据集成）
```

Windows 注意：
- SAM2 编译扩展已跳过（`SAM2_BUILD_CUDA=0` 是默认行为），首次 import 出现 `_C` 警告属正常。
- 控制台中文/emoji 输出需要 `PYTHONUTF8=1`（已建议设为用户级环境变量）。

## 数据（DAVIS）

数据集放在 `targets/` 下（已在 .gitignore 中，不入库）：

```
targets/
├── videos/            # 原始 DAVIS mp4
├── images/<目标名>/    # 拆帧 jpg（由脚本生成）
```

首次使用：

```bash
# 1. 把 DAVIS 视频放到 targets/videos/（如 bear.mp4、car.mp4）
# 2. 拆帧（每个视频取中间至多 70 帧，裁剪到 8 的倍数分辨率）
uv run python targets/convert_videos_to_frames.py --videos-dir targets/videos --images-dir targets/images
```

切换处理目标：改 `config/test_video_init.yaml` 里的 `url_head`（如 `./targets/images/car`），全流程脚本都从这里读路径。

SAM2 checkpoint：由 `scripts/init_sam2.py` 自动下载到 `sam2_git/checkpoints/`（`sam2.1_hiera_base_plus.pt`、`sam2.1_hiera_large.pt`），无需手动操作。

## 构建光流与 mask 缓存

```bash
# 光流：RAFT 逐对帧计算，缓存到 caches/flow/<目标名>.pt
uv run python reasoning/compute_RAFT.py

# mask：YOLO 首帧检测 -> SAM2 视频传播，缓存到 caches/mask/<目标名>.pt
uv run python reasoning/mask_predictor.py
```

## 造 stroke

对首帧手绘一笔（OpenCV 交互窗口，左键拖动，`q` 退出），存到 `stroke/<目标名>/stroke_XX.npy`：

```bash
uv run python stroke/make_stroke.py
```

## 跑实验

```bash
# 主实验：加载缓存与笔画，逐帧对比各传播策略（flow / snapping / fitted 等），
# 交互窗口查看，并将各策略结果导出为 GIF/PNG 到 results/
uv run python experiment/test_stroke.py

# 检测器对比实验（YOLO vs Florence-2，GT-free）
uv run python experiment/compare_detectors.py --targets bear blackswan car cows --max_frames 20

# 边界贴合误差曲线
uv run python experiment/plot_boundary_adherence_error.py
```

测试：

```bash
uv run pytest                  # 全部 68 项
uv run pytest -m "not slow"    # 跳过真实推理的慢测试
uv run pytest -m integration   # 仅真数据集成测试
```
