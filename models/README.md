# models/

推理模型文件的本地存放目录。模型二进制不提交到仓库（`.gitignore` 已忽略 `*.onnx`），代码也不会自动联网下载；需要手工按下面的命令获取。

## face_detection_yunet_2023mar.onnx

- 用途：`--detector yunet` 人脸检测（OpenCV Zoo YuNet 2023-03，固定 640x640 BGR 输入）。
- 官方来源：<https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet>
- License：MIT（`opencv_zoo` 仓库 `models/face_detection_yunet/LICENSE`，Copyright (c) 2020 Shiqi Yu）。授权全文已附带在本目录 [`YUNET-LICENSE`](YUNET-LICENSE)。
- 手动下载（需联网）：

  ```bash
  curl -fL -o models/face_detection_yunet_2023mar.onnx \
    https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
  ```

- 校验：`shasum -a 256` 应为 `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`（232,589 字节）。
