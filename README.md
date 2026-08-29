# video-redact

面向视频文件和实时流的 GPU 隐私脱敏网关。Rust 负责流水线、调度和服务层，CUDA 负责帧预处理与 ROI 马赛克，后续接入 TensorRT 完成人脸和车牌检测。

当前仓库是可运行的第一阶段脚手架：

- `video-redact-core`：RGB24 帧、ROI、CPU 参考实现和后端接口。
- `video-redact-cuda`：通过 `cudarc` + NVRTC 加载真实 CUDA ROI 马赛克 kernel。
- `video-redact-cli`：生成测试图并通过 CPU 或 CUDA 后端脱敏。

## 快速开始

在任意平台验证 CPU 路径：

```bash
cargo run -p video-redact-cli -- demo --output demo.ppm --backend cpu
cargo test --workspace
```

在装有 NVIDIA 驱动及 CUDA 12.8 runtime/toolkit 的 Linux 主机上验证 CUDA 路径：

```bash
cargo run -p video-redact-cli --features cuda -- \
  demo --output demo.ppm --backend cuda
```

CUDA 依赖采用动态加载，因此编译主机不需要静态链接 CUDA；运行 CUDA 路径仍需要 NVIDIA GPU、驱动以及 NVRTC 动态库。若目标机器使用不同 CUDA 版本，请调整 `crates/video-redact-cuda/Cargo.toml` 中的 `cuda-12080` feature。

## CLI

```text
video-redact info
video-redact demo [--output demo.ppm] [--backend cpu|cuda]
```

`demo` 会生成一张 RGB 测试图，并对两个矩形区域应用马赛克。PPM 可以被多数图像工具打开，也可以使用 FFmpeg 转换：

```bash
ffmpeg -i demo.ppm demo.png
```

## 近期路线

1. 接入 FFmpeg，完成 MP4 demux/decode/encode。
2. 将 CPU 帧替换为 NVDEC 产生的 GPU frame，消除主机往返拷贝。
3. 接入 ONNX/TensorRT 人脸及车牌检测模型。
4. 增加目标跟踪、低置信度审核和 JSON 审计报告。
5. 支持 RTSP、多路并发、Prometheus 指标和容器部署。

详细边界与模块关系见 [`docs/architecture.md`](docs/architecture.md)。

