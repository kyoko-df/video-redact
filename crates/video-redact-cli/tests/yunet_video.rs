//! End-to-end YuNet redaction over a real video, run explicitly with:
//!
//! ```bash
//! cargo test -p video-redact-cli --features yunet --test yunet_video -- --ignored
//! ```
//!
//! Requires `VIDEO_REDACT_YUNET_MODEL` (the YuNet ONNX file) and
//! `VIDEO_REDACT_FACE_IMAGE` (a public image containing a face). Artifacts go
//! to `VIDEO_REDACT_E2E_DIR` or a fresh per-process temp directory.

#![cfg(feature = "yunet")]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn run(program: &str, args: &[&str], paths: &[&Path]) -> Output {
    let mut command = Command::new(program);
    command.args(args);
    for path in paths {
        command.arg(path);
    }
    command.output().expect("failed to spawn process")
}

fn require_success(output: &Output, what: &str) {
    assert!(
        output.status.success(),
        "{what} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn ffprobe_value(path: &Path, entries: &str) -> String {
    let output = run(
        "ffprobe",
        &[
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            entries,
            "-of",
            "default=noprint_wrappers=1",
        ],
        &[path],
    );
    require_success(&output, "ffprobe");
    String::from_utf8_lossy(&output.stdout).trim().to_owned()
}

#[test]
#[ignore = "requires VIDEO_REDACT_YUNET_MODEL and VIDEO_REDACT_FACE_IMAGE"]
fn yunet_redacts_a_face_video() {
    // Running with `--ignored` without the required fixtures fails loudly
    // instead of silently passing.
    let model = std::env::var_os("VIDEO_REDACT_YUNET_MODEL")
        .map(PathBuf::from)
        .expect("VIDEO_REDACT_YUNET_MODEL must point at face_detection_yunet_2023mar.onnx");
    let image = std::env::var_os("VIDEO_REDACT_FACE_IMAGE")
        .map(PathBuf::from)
        .expect("VIDEO_REDACT_FACE_IMAGE must point at a face image");

    // VIDEO_REDACT_E2E_DIR is a parent directory; each run creates a unique
    // child so artifacts are never overwritten.
    let parent =
        std::env::var_os("VIDEO_REDACT_E2E_DIR").map_or_else(std::env::temp_dir, PathBuf::from);
    let work = parent.join(format!(
        "run-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis()
    ));
    std::fs::create_dir_all(&work).unwrap();
    let input = work.join("input.mp4");
    let output_path = work.join("redacted.mp4");

    // Two identical 5fps frames encoded as H.264 in an MP4 container.
    let build = Command::new("ffmpeg")
        .args(["-v", "error", "-loop", "1", "-framerate", "5", "-i"])
        .arg(&image)
        .args(["-frames:v", "2", "-c:v", "libx264", "-pix_fmt", "yuv420p"])
        .arg(&input)
        .output()
        .unwrap();
    require_success(&build, "ffmpeg input build");

    let cli = Command::new(env!("CARGO_BIN_EXE_video-redact"))
        .args(["redact", "--input"])
        .arg(&input)
        .args(["--output"])
        .arg(&output_path)
        .args([
            "--detector",
            "yunet",
            "--min-confidence",
            "0.5",
            "--padding",
            "8",
            "--backend",
            "cpu",
        ])
        .arg("--model")
        .arg(&model)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&cli.stdout);
    require_success(&cli, "video-redact");
    println!("{stdout}");

    assert!(stdout.contains("2 frames"), "{stdout}");
    let redacted: u64 = stdout
        .split("regions redacted: ")
        .nth(1)
        .and_then(|rest| rest.split(',').next())
        .and_then(|value| value.trim().parse().ok())
        .expect("report must list redacted regions");
    assert!(redacted > 0, "expected redacted regions in: {stdout}");

    let probed = ffprobe_value(&output_path, "stream=codec_name,width,height");
    assert!(probed.contains("codec_name=h264"), "{probed}");
    assert!(probed.contains("width=512"), "{probed}");
    assert!(probed.contains("height=512"), "{probed}");
    let frames = run(
        "ffprobe",
        &[
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
        ],
        &[&output_path],
    );
    require_success(&frames, "ffprobe frame count");
    let frame_count = String::from_utf8_lossy(&frames.stdout).trim().to_owned();
    assert_eq!(frame_count, "2", "probe fields: {probed}");

    // Keep first-frame PNGs of input and output for visual verification.
    for (video, png) in [(&input, "before.png"), (&output_path, "after.png")] {
        let extract = Command::new("ffmpeg")
            .args(["-v", "error", "-i"])
            .arg(video)
            .args(["-frames:v", "1"])
            .arg(work.join(png))
            .output()
            .unwrap();
        require_success(&extract, "ffmpeg frame extract");
    }
    println!("artifacts in {}", work.display());
}
