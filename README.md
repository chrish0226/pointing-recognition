# 指差确认：自然动作优化版

当前规则为 `natural_pointing_v5`，模型沿用 RTMW3D-L / YOLOX-s TensorRT FP16。前、左、右以人体朝向为基准，每人累计三向各一次后红框变绿，通过后继续累计；R 重置，Q/Esc 退出。

## 本次改变

针对低位、屈肘和连续换向的自然动作，增加 `natural_pointing.py`：

- 小幅侧指融合前臂与可靠食指二维点，不再仅由肩到腕的横移判断；双手同时操作时不使用这项补充，已有侧身前向辅助优先。
- 侧指转前指结合前臂投影缩短、手腕向内移动、抬手高度一致与三维前伸证据，允许不放下手连续换向。
- 手部点缺失、低分、弯曲或位置不合理时回退到既有身体规则；鼻耳仍仅辅助朝向。

模型引擎、三维解码、无人工 ROI、每人独立累计以及红绿框规则沿用基线。运行规则不读取视频编号、预期次数、人工时段或固定左→右→前顺序。

## 运行

本机使用现有环境、模型及引擎：

```powershell
& "E:\粉末冶金\指差检测\run_realtime.cmd" --source 0
```

测试视频按原始节奏播放：

```powershell
.\run_realtime.cmd --source "视频的完整路径.mp4" --file-realtime
```

[实时使用说明](实时推理使用说明.md) · [几何判定与三维坐标说明](三维关键点与几何判定说明.md)

## 验证

85 项自动测试通过，新增真实自然动作的顺序、确认时段、整理手套不计数、普通摆手不计数，以及手部/鼻耳缺失回退检查。已有侧身前指等 76 项检查保留。回归数据来自参与调参的素材，不是独立现场准确率测试。

完整视频重新推理及剩余错误见 [当前验证记录](verification/natural_pointing_v5/verification.json)。结果视频和逐帧日志保存在本机 `outputs/natural_pointing_v5`；未上传工厂视频。主规则和新增辅助模块分别记录 SHA256，并生成整套规则指纹，避免只记录主文件版本。

批量检查工具 `tools/verify_videos.py` 接收本地清单，例如：

```json
{"example": {"source": "D:/videos/example.mp4"}}
```

```powershell
.\.venv_rtmw3d_trt\Scripts\python.exe tools/verify_videos.py --manifest sources.json
.\.venv_rtmw3d_trt\Scripts\python.exe -m unittest discover -s . -p "test_*.py" -q
```

工具执行完整检测/跟踪/姿态/规则，导出 H.264，并验证逐帧次数、红绿状态、参考轴和视频完整解码。这些用于检查程序和导出一致性，识别正确性仍需逐次与人工动作核对。

## GitHub 基线与部署范围

优化前原版已保存在 `main` 的基线提交及标签 `baseline-side-forward-v4-20261026`；优化在 `improve/natural-pointing-v5` 分支。基线内容、模型指纹及旧结果保存在 [BASELINE.md](BASELINE.md) 和 `baselines/side_forward_v4`，可以准确回退。

代码仓库不包含虚拟环境、模型、TensorRT 引擎和工厂视频；新机器的准备事项见基线说明。Jetson、物理摄像头及 RTSP 性能未在本次实测，TensorRT 引擎需要在目标 GPU/软件环境中构建。
