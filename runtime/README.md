# ONNX Runtime binaries (CoreML execution)

`video-redact-detect` uses the Rust `ort` crate with `load-dynamic`: the ONNX
Runtime library is **not** linked at build time. The user supplies an official
prebuilt runtime at runtime via `--ort-library` or `VIDEO_REDACT_ORT_DYLIB`.

## Which runtime

ONNX Runtime **1.24.3**, macOS arm64, from the official Microsoft release:

    https://github.com/microsoft/onnxruntime/releases/download/v1.24.3/onnxruntime-osx-arm64-1.24.3.tgz

`ort 2.0.0-rc.12` is pinned with `api-24`; this runtime reports API version
`1.24.3` and passes the crate's compatibility check.

## Manual download (no automatic downloads in code)

    mkdir -p runtime
    curl -fL --retry 3 \
      https://github.com/microsoft/onnxruntime/releases/download/v1.24.3/onnxruntime-osx-arm64-1.24.3.tgz \
      -o runtime/onnxruntime-osx-arm64-1.24.3.tgz
    tar -xzf runtime/onnxruntime-osx-arm64-1.24.3.tgz -C runtime/

## Integrity

Archive SHA-256 (verify before extracting):

    c255663d40755f84b1b86373bdb9870bb65f3a2c3d779b3d7ae31aaa00cebb4f

## Trust

`--ort-library`/`VIDEO_REDACT_ORT_DYLIB` loads a native dynamic library into
the process — executable code. Only point it at the official Microsoft build
verified above (or a build you trust); never at a file of unknown provenance.
The crate pre-checks that it loads, exports `OrtGetApiBase`, reports a
compatible `1.24+` version and serves the ORT API version the crate needs.

## License

ONNX Runtime is MIT licensed. The full license text ships inside the archive
and is preserved at `runtime/onnxruntime-osx-arm64-1.24.3/LICENSE`
(plus `ThirdPartyNotices.txt`).

## Usage

    video-redact redact --detector yunet --model models/face_detection_yunet_2023mar.onnx \
      --inference-backend coreml \
      --ort-library runtime/onnxruntime-osx-arm64-1.24.3/lib/libonnxruntime.1.24.3.dylib \
      --backend cpu --input in.mp4 --output out.mp4

Only the YuNet model inference runs through ONNX Runtime/CoreML; decode,
preprocess, mosaic and encode remain CPU/Rust. The first CoreML session
compiles the model, which takes noticeable time.
