# video-redact

面向视频文件和实时流的 GPU 隐私脱敏网关。Rust 负责流水线、调度和服务层，CUDA 负责帧预处理与 ROI 马赛克，后续接入 TensorRT 完成人脸和车牌检测。

当前仓库是可运行的第一阶段脚手架：

- `video-redact-core`：RGB24 帧、ROI、CPU 参考实现和后端接口。
- `video-redact-cuda`：通过 `cudarc` + NVRTC 加载真实 CUDA ROI 马赛克 kernel。
- `video-redact-ffmpeg`：通过 FFmpeg 解码 RGB24 帧、逐帧脱敏并编码 H.264 MP4。
- `video-redact-cli`：运行测试图或 MP4 文件脱敏流水线。

## 快速开始

在任意平台验证 CPU 路径：

```bash
cargo run -p video-redact-cli -- demo --output demo.ppm --backend cpu
cargo test --workspace
```

处理一条 MP4（需要 `ffmpeg` 和 `ffprobe` 在 `PATH` 中）：

```bash
cargo run -p video-redact-cli -- redact \
  --input input.mp4 \
  --output redacted.mp4 \
  --roi 120,80,420,320 \
  --roi 700,100,920,260 \
  --block-size 16 \
  --backend cpu
```

ROI 使用半开区间 `left,top,right,bottom`，会应用到视频的每一帧；超出画面的部分会自动裁剪。重复执行并替换已有输出时添加 `--overwrite`。

在装有 NVIDIA 驱动及 CUDA 12.8 runtime/toolkit 的 Linux 主机上验证 CUDA 路径：

```bash
cargo run -p video-redact-cli --features cuda -- \
  demo --output demo.ppm --backend cuda

VIDEO_REDACT_REQUIRE_CUDA=1 \
  cargo test -p video-redact-cuda --features cuda
```

CUDA 依赖采用动态加载，因此编译主机不需要静态链接 CUDA；运行 CUDA 路径仍需要 NVIDIA GPU、驱动以及 NVRTC 动态库。若目标机器使用不同 CUDA 版本，请调整 `crates/video-redact-cuda/Cargo.toml` 中的 `cuda-12080` feature。
CUDA 测试会逐像素比对 CPU 参考实现，覆盖不完整马赛克块、裁剪及重叠 ROI；没有设置 `VIDEO_REDACT_REQUIRE_CUDA` 时，无 CUDA 的开发机会跳过真机比对。

## CLI

```text
video-redact info
video-redact demo [--output demo.ppm] [--backend cpu|cuda]
video-redact redact --input PATH --output PATH --roi L,T,R,B [OPTIONS]
```

`demo` 会生成一张 RGB 测试图，并对两个矩形区域应用马赛克。PPM 可以被多数图像工具打开，也可以使用 FFmpeg 转换：

```bash
ffmpeg -i demo.ppm demo.png
```

`redact` 当前处理第一条视频流，用输入平均帧率驱动恒定帧率编码，视频编码器为 `libx264`、像素格式为 `yuv420p`；输入音频流直接复制到输出。可通过 `VIDEO_REDACT_FFMPEG` 和 `VIDEO_REDACT_FFPROBE` 环境变量指定可执行文件路径。

这是有意保留的第一条正确性基线：FFmpeg 与脱敏后端之间使用 RGB24 管道，因此 CPU 和现有 CUDA 实现都能工作，但仍有主机内存拷贝，且可变帧率输入会被归一化。后续 NVDEC/NVENC 路径将替换该传输层。

## 近期路线

1. 将 RGB24 管道替换为 NVDEC/NVENC GPU frame，消除主机往返拷贝。
2. 接入 ONNX/TensorRT 人脸及车牌检测模型。
3. 增加目标跟踪、低置信度审核和 JSON 审计报告。
4. 支持 RTSP、多路并发、Prometheus 指标和容器部署。

详细边界与模块关系见 [`docs/architecture.md`](docs/architecture.md)。
