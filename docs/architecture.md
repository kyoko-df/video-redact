# Architecture

目标流水线：

```text
file / RTSP
    -> demux
    -> NVDEC
    -> CUDA preprocess
    -> TensorRT detector
    -> tracker + policy
    -> CUDA redact
    -> NVENC
    -> file / HLS / RTSP
```

## 设计边界

- `video-redact-core` 不依赖 CUDA 或 FFmpeg，保存领域类型和可测试的 CPU 参考算法。
- `video-redact-cuda` 只处理已经解码的帧；当前用 RGB24 验证 kernel，接入 NVDEC 后扩展为 NV12/P010。
- `video-redact-cli` 只负责参数、I/O 和后端装配，业务策略不放进 CLI。
- 检测器和跟踪器将作为独立 trait 接入，避免把 TensorRT 生命周期耦合到视频解码器。

## 性能原则

- 解码后的帧尽量始终停留在 GPU。
- 预处理、推理、脱敏和编码共享 CUDA stream/event 进行同步。
- 批处理离线文件时优化吞吐，实时流模式则限制队列长度并优先控制尾延迟。
- CPU 实现是正确性基线，不作为最终性能路径。

## 验收指标

- 正确性：ROI 外像素保持不变，ROI 内只读取合法像素。
- 性能：分别记录 decode、preprocess、inference、redact、encode 耗时。
- 稳定性：输入损坏、流中断或单路 OOM 不应拖垮其他任务。
- 隐私：保存低置信度检测记录，并支持人工复核输出。

