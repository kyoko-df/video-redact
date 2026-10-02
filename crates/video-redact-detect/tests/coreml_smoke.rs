//! Real-model CoreML smoke test, compared against the frozen OpenCV 4.12
//! YuNet references with the wider CoreML tolerances (internal precision may
//! differ from the CPU reference).
//!
//! Requires FFmpeg/FFprobe on PATH, the ONNX Runtime dylib and the downloaded
//! fixtures; run explicitly with:
//!
//! ```bash
//! VIDEO_REDACT_ORT_DYLIB=/path/libonnxruntime.1.24.3.dylib \
//! VIDEO_REDACT_YUNET_MODEL=/path/face_detection_yunet_2023mar.onnx \
//! VIDEO_REDACT_FACE_IMAGE=/path/lena.jpg \
//! VIDEO_REDACT_WIDE_FACE_IMAGE=/path/wide-lena.png \
//! cargo test -p video-redact-detect --features coreml --test coreml_smoke -- --ignored
//! ```

#![cfg(all(feature = "coreml", target_os = "macos", target_arch = "aarch64"))]

use std::path::{Path, PathBuf};
use std::process::Command;

use video_redact_core::{DetectionLabel, Detector, RgbFrame};
use video_redact_detect::{CoreMlOptions, YuNetDetector};

const COORD_TOLERANCE: u32 = 2;
const CONFIDENCE_TOLERANCE: f64 = 0.01;

fn probe_size(path: &Path) -> (u32, u32) {
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
        ])
        .arg(path)
        .output()
        .expect("ffprobe failed to run");
    assert!(
        output.status.success(),
        "ffprobe failed on {}",
        path.display()
    );
    let text = String::from_utf8_lossy(&output.stdout);
    let mut parts = text.trim().split(',');
    let width = parts.next().unwrap().parse().unwrap();
    let height = parts.next().unwrap().parse().unwrap();
    (width, height)
}

fn decode_rgb(path: &Path) -> RgbFrame {
    let (width, height) = probe_size(path);
    let output = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(path)
        .args(["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
        .output()
        .expect("ffmpeg failed to run");
    assert!(
        output.status.success(),
        "ffmpeg failed on {}",
        path.display()
    );
    RgbFrame::new(width, height, output.stdout).unwrap()
}

fn assert_matches_reference(
    frame: &RgbFrame,
    detections: &[video_redact_core::Detection],
    expected: [u32; 4],
    expected_confidence: f64,
) {
    assert_eq!(
        detections.len(),
        1,
        "expected exactly one face, got {detections:?}"
    );
    let detection = detections[0];
    assert_eq!(detection.label, DetectionLabel::Face);
    for (actual, reference) in [
        (detection.rect.left, expected[0]),
        (detection.rect.top, expected[1]),
        (detection.rect.right, expected[2]),
        (detection.rect.bottom, expected[3]),
    ] {
        assert!(
            actual.abs_diff(reference) <= COORD_TOLERANCE,
            "rect {:?} vs reference {expected:?}",
            detection.rect
        );
    }
    assert!(
        (f64::from(detection.confidence) - expected_confidence).abs() <= CONFIDENCE_TOLERANCE,
        "confidence {} vs reference {expected_confidence}",
        detection.confidence
    );
    assert!(detection.confidence >= 0.5);
    assert!(detection.rect.right <= frame.width() && detection.rect.bottom <= frame.height());
}

#[test]
#[ignore = "requires VIDEO_REDACT_ORT_DYLIB, VIDEO_REDACT_YUNET_MODEL and fixture images"]
fn yunet_coreml_matches_opencv_reference() {
    // Running with `--ignored` without the required fixtures fails loudly
    // instead of silently passing.
    let library = std::env::var_os("VIDEO_REDACT_ORT_DYLIB")
        .map(PathBuf::from)
        .expect("VIDEO_REDACT_ORT_DYLIB must point at libonnxruntime.dylib");
    let model = std::env::var_os("VIDEO_REDACT_YUNET_MODEL")
        .map(PathBuf::from)
        .expect("VIDEO_REDACT_YUNET_MODEL must point at face_detection_yunet_2023mar.onnx");
    let face_image = std::env::var_os("VIDEO_REDACT_FACE_IMAGE")
        .map(PathBuf::from)
        .expect("VIDEO_REDACT_FACE_IMAGE must point at the 512x512 face fixture");

    // VIDEO_REDACT_COREML_ARTIFACT_DIR is a parent directory (created if
    // missing); each run creates an exclusive child so a fresh machine with
    // only the env fixtures works, and artifacts are never overwritten.
    let parent = std::env::var_os("VIDEO_REDACT_COREML_ARTIFACT_DIR")
        .map_or_else(std::env::temp_dir, PathBuf::from);
    std::fs::create_dir_all(&parent).unwrap();
    let work = parent.join(format!(
        "run-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&work).unwrap();
    let mut options = CoreMlOptions::new(library);
    options.verbose = true;
    options.profile_prefix = Some(work.join("coreml"));
    let mut detector = YuNetDetector::new_coreml(&model, 0.5, &options).unwrap();

    // Square 512x512 lena.jpg, run twice; both runs must satisfy the same
    // frozen reference bounds.
    let square = decode_rgb(&face_image);
    assert_eq!((square.width(), square.height()), (512, 512));
    for run in 1..=2 {
        let detections = detector.detect(&square, 0).unwrap();
        println!("square run {run} detections: {detections:?}");
        assert_matches_reference(
            &square,
            &detections,
            [209, 181, 352, 388],
            0.913_480_460_643_768_3,
        );
    }

    // Optional 768x512 wide fixture -> frozen reference rect [330,174,482,395].
    if let Some(wide) = std::env::var_os("VIDEO_REDACT_WIDE_FACE_IMAGE").map(PathBuf::from) {
        let wide_frame = decode_rgb(&wide);
        assert_eq!((wide_frame.width(), wide_frame.height()), (768, 512));
        let detections = detector.detect(&wide_frame, 0).unwrap();
        println!("wide detections: {detections:?}");
        assert_matches_reference(
            &wide_frame,
            &detections,
            [330, 174, 482, 395],
            0.907_168_328_762_054_4,
        );
    } else {
        // The wide fixture stays optional; only its check is skipped.
        eprintln!("skipping wide fixture check: VIDEO_REDACT_WIDE_FACE_IMAGE is not set");
    }

    // All-black 640x640 input must produce no detections.
    let black = RgbFrame::new(640, 640, vec![0; 640 * 640 * 3]).unwrap();
    let detections = detector.detect(&black, 0).unwrap();
    println!("black detections: {detections:?}");
    assert!(detections.is_empty(), "expected no faces in black input");

    let path = detector
        .end_profiling()
        .unwrap()
        .expect("profiling was enabled; end_profiling must return a path");
    println!("profiling written to {}", path.display());
    assert!(path.is_file(), "profile file missing: {}", path.display());
    assert!(
        std::fs::metadata(&path).unwrap().len() > 0,
        "profile file is empty: {}",
        path.display()
    );
    // Profiling ends exactly once; a second call reports None.
    assert_eq!(detector.end_profiling().unwrap(), None);
}
