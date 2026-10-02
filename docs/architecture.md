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

当前已落地的第一条文件流水线：

```text
MP4 file
    -> FFmpeg demux + software decode
    -> RGB24 stdout pipe
    -> Detector (static ROI | YuNet ONNX CPU|CoreML) -> policy (confidence/padding/review)
    -> CPU/CUDA Redactor (mosaic)
    -> RGB24 stdin pipe
    -> FFmpeg libx264 encode + input audio stream copy
    -> MP4 file
```

该路径优先建立端到端正确性、进程失败处理和可测试的模块边界。它按输入平均帧率输出恒定帧率视频；可变帧率时间戳、零拷贝和硬件编解码属于下一阶段。

YuNet CPU 检测器（`video-redact-detect`，CLI `--features yunet`）在 `Detector` 内部完成全部预处理：RGB24 按比例缩放并居中 padding 成 640x640，双线性半像素采样转 BGR NCHW f32（0..=255 不归一化），运行 tract ONNX 推理后按名字取 stride 8/16/32 的 `cls`/`obj`/`bbox` 头解码，`sqrt(cls*obj)` 过滤置信度、逆变换回原图坐标并做贪心 NMS（IoU>0.3，top-5000）。检测器的预处理、推理与后处理耗时统一计入 `StageTimings.inference`。

可选的 `coreml` feature（仅 `macos`+`aarch64`）通过 `ort` crate 动态加载官方 ONNX Runtime dylib（`ort::init_from`，路径只来自显式 `--ort-library`/`VIDEO_REDACT_ORT_DYLIB`，不自动搜索或下载），把同一个 YuNet ONNX 和完全相同的预处理/解码/NMS 代码放到 CoreML execution provider 上执行。会话固定 `ModelFormat=MLProgram`、`MLComputeUnits=CPUAndGPU`、`RequireStaticInputShapes=1`、Level3 图优化并禁用 CPU fallback；加载是进程全局的，`video-redact-detect` 用互斥锁序列化并只允许同一个 canonical runtime 路径。CoreML 只接管模型推理，帧传输与脱敏仍是 RGB24 CPU 管道——不是零拷贝 GPU 路径。

## 设计边界

- `video-redact-core` 不依赖 CUDA 或 FFmpeg，保存领域类型和可测试的 CPU 参考算法。
- `video-redact-ffmpeg` 管理探测、解码/编码子进程和逐帧调度，不包含具体脱敏策略。
- `video-redact-cuda` 只处理已经解码的帧；当前用 RGB24 验证 kernel，接入 NVDEC 后扩展为 NV12/P010。
- `video-redact-detect` 默认只依赖 `video-redact-core` 与 `tract-onnx`，是纯 Rust CPU 推理路径；可选 `coreml` feature（macOS arm64）追加 `ort` 动态加载的 ONNX Runtime/CoreML 推理后端。不引入 OpenCV/Python 运行时，也不做 CUDA/TensorRT 推理。
- `video-redact-cli` 只负责参数、I/O 和后端装配，业务策略不放进 CLI。
- 检测器通过 `Detector` trait 接入（静态 ROI 与 YuNet ONNX 已实现），策略层独立做置信度过滤、区域外扩和审核记录；跟踪器与 TensorRT 实现将复用同一接缝，不耦合到视频解码器。

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
