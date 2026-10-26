# 优化前的代码基线

本次基线为 `side_forward_v4`，保留用户要求“先上传当前代码，再优化”时的实现。识别代码尚未加入针对 8.mp4 自然动作的改动。

`balanced_pointing_rules.py` 原文件 SHA256：

`373694867936f7785aa1ede8116081c6d69c28827d8db9bbb6b6b439df9b84c6`

已知问题：8.mp4 的左指在约 2.83 秒被记为前，右指在约 3.13 秒确认，真正前指未形成新计数。其他七段结果以及此前相机视频的汇总保存在 `baselines/side_forward_v4`，供后续比较；次数一致不能代替逐次动作核对。

## 恢复与运行

代码、测试、关键点回归夹具和现有 RTMLib 源码均纳入版本控制。RTMLib 的第三方许可证保留在 `rtmlib-main/LICENSE`。

本机现有启动方式：

```powershell
.\run_realtime.cmd --source 0
```

新电脑需要单独准备兼容的 Python、CUDA/PyTorch/TensorRT 环境。将模型放入以下本地目录，并使用 `rtmw_trt_runtime.py` 在目标 GPU 上重新构建 TensorRT 引擎：

```text
models/rtmw3d_l_384x288/rtmw3d_l.onnx
models/detector_yolox_s/end2end.onnx
models/rtmw3d_l_tensorrt_fp16/rtmw3d_l_fp16.engine
models/rtmw3d_l_tensorrt_fp16/yolox_s_fp16.engine
```

模型、引擎、虚拟环境、工厂视频及生成的逐帧日志保留在本地，不进入代码仓库。原有使用说明中的绝对路径指向原电脑。`baselines/side_forward_v4/environment.json` 记录运行版本和模型指纹，不包含账号凭据。

测试：

```powershell
python -m unittest discover -s . -p "test_*.py" -q
```

后续优化从此基线创建独立分支，保留基线提交与标签，便于准确回退。
