# GTA-V-Autopilot

面向 GTA V 游戏画面的 Python 计算机视觉实验：目标检测（YOLO）、车道线检测、多目标跟踪（SORT）与 Windows 键盘控制。

## 在线展示

**[视觉展示页 →](https://unstoppablecurry.github.io/GTA-V-Autopilot/)**

演示视频与截图说明见 GitHub Pages 站点；页面仅描述仓库中已有素材的可见内容，不含未验证的性能声明。

## 仓库结构（摘要）

| 模块 | 说明 |
|------|------|
| `gtav.py` | 主入口：屏幕捕获与视觉处理循环 |
| `MutilCarDection*.py` | YOLO 车辆检测（Darknet / PyTorch） |
| `visualization_by_pytorch.py` | 车道线检测与可视化 |
| `kalman.py` | SORT 跟踪 + 卡尔曼滤波 |
| `fastlanedetection/` | 车道线模型 |
| `yolo-coco/` | YOLO 配置 |
| `有限状态机/` | 车速数据采集与实验脚本 |

## 演示素材

- `demo.jpg` — 静态截图
- `video/project_video.mp4` — 真实道路车载视角（无叠加）
- `video/project_video2.mp4` — GTA V 游戏内画面（无叠加）
- `wtx.mp4` — 视觉管线输出（含车道线与检测框叠加）

## 说明

本项目 README 仍在完善中。请以源码与上述演示素材为准；不对自动驾驶能力或系统性能作任何承诺。
