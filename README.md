# 指差确认：侧身前指优化版

当前规则为 `side_forward_v4`。模型仍为 RTMW3D-L / YOLOX-s TensorRT FP16，前、左、右以人体朝向为基准。每人累计三向各一次后红框变绿，通过后继续累计；R 重置，Q/Esc 退出。

## 本次修改

侧身时，单目模型可能把双肩深度差压到接近零，甚至交换肩部左右标签，使身体参考轴错误地接近正面。本版在原身体几何规则上增加可选的朝向辅助：利用同一模型已有的鼻、耳点和该人的肩髋投影比例，辅助侧身朝向；不新增面部模型，不要求转头确认，也不因缺少鼻耳点就拒绝动作。

同一模型的食指二维点只在侧身时提供可选的矛盾检查，避免把手指朝向相反的动作直接算为前指。新增前指高度稳定检查、侧身短时观察及放下手时的重复方向抑制。极短投影手臂的深度分配还会经过骨段一致性比较；三维坐标仍是未标定估计。

本次用户确认 22:32:55 视频为 7 组左→右→前。逐次前指与剩余误判见 [本次优化结果](E:/粉末冶金/指差检测/outputs/side_forward_v4/优化说明.md)。左右混淆和其他场景漏检尚未全部解决，不能把有限素材上的改善视为现场准确率保证。此前 `outputs/person_relative_v3` 下的所有视频和报告均为旧规则结果。

## 启动

```powershell
& "E:\粉末冶金\指差检测\run_realtime.cmd" --source 0
```

重新查看本次视频：

```powershell
& "E:\粉末冶金\指差检测\run_realtime.cmd" --source "D:\ASDF\Pictures\Camera Roll\WIN_20261025_22_32_55_Pro.mp4" --file-realtime
```

[实时推理使用说明](E:/粉末冶金/指差检测/实时推理使用说明.md) · [几何判定说明](E:/粉末冶金/指差检测/三维关键点与几何判定说明.md)

## 验证与历史

76 项测试通过，包含真实侧身/背侧前指、短投影前指、抬手过渡不加前、放下手不加右、鼻耳缺失时回退、输入点位不被修改、诊断日志序列化等。真实关键点回归片段来自本次调参视频，不是独立准确率测试。

本次整段推理和既有正面三组、9.mp4 的检查记录见 [验证记录](E:/粉末冶金/指差检测/verification/side_forward_v4/verification.json)。模型引擎沿用当前电脑的构建；Jetson、摄像头、RTSP 性能未在本次验证。

七段旧视频已用当前 v4 完整重跑，共 1922 帧：[查看结果视频](E:/粉末冶金/指差检测/outputs/side_forward_v4/test_videos/README.md) · [次数、逐轨迹事件及 v3 对照](E:/粉末冶金/指差检测/outputs/side_forward_v4/test_videos/核对说明.md)。9.mp4 三向次数一致，3.mp4 的 ID1 左右次数一致；其他主体仍有未识别方向、混淆或多计，当前规则在这些场景中仍不稳定。

旧七段 v3 结果：[查看历史视频清单](E:/粉末冶金/指差检测/outputs/person_relative_v3/test_videos/README.md)。旧 12:40 恢复检查仍在 `verification/restore_1240`。修改前备份位置记录在 [backup_record.json](E:/粉末冶金/指差检测/runtime/side_forward_20261025/backup_record.json)。

运行测试：

```powershell
Set-Location "E:\粉末冶金\指差检测"
.\.venv_rtmw3d_trt\Scripts\python.exe -m unittest discover -s . -p "test_*.py" -q
```
