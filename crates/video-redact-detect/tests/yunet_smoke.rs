//! Real-model smoke test comparing against frozen OpenCV 4.12 YuNet results.
//!
//! Requires FFmpeg/FFprobe on PATH and the downloaded fixtures; run
//! explicitly with:
//!
//! ```bash
//! VIDEO_REDACT_YUNET_MODEL=/path/face_detection_yunet_2023mar.onnx \
//! VIDEO_REDACT_FACE_IMAGE=/path/lena.jpg \
//! VIDEO_REDACT_WIDE_FACE_IMAGE=/path/wide-lena.png \
//! cargo test -p video-redact-detect --test yunet_smoke -- --ignored
//! ```

use std::path::{Path, PathBuf};
use std::process::Command;

use video_redact_core::{DetectionLabel, Detector, RgbFrame};
use video_redact_detect::YuNetDetector;

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
            actual.abs_diff(reference) <= 1,
            "rect {:?} vs reference {expected:?}",
            detection.rect
        );
    }
    assert!(
        (f64::from(detection.confidence) - expected_confidence).abs() <= 0.001,
        "confidence {} vs reference {expected_confidence}",
        detection.confidence
    );
    assert!(detection.confidence >= 0.5);
    assert!(detection.rect.right <= frame.width() && detection.rect.bottom <= frame.height());
}

#[test]
#[ignore = "requires VIDEO_REDACT_YUNET_MODEL and fixture images"]
fn yunet_matches_opencv_reference() {
    // Running with `--ignored` without the required fixtures fails loudly
    // instead of silently passing.
    let model = std::env::var_os("VIDEO_REDACT_YUNET_MODEL")
        .map(PathBuf::from)
        .expect("VIDEO_REDACT_YUNET_MODEL must point at face_detection_yunet_2023mar.onnx");
    let face_image = std::env::var_os("VIDEO_REDACT_FACE_IMAGE")
        .map(PathBuf::from)
        .expect("VIDEO_REDACT_FACE_IMAGE must point at the 512x512 face fixture");

    let mut detector = YuNetDetector::new(&model, 0.5).unwrap();

    // Square 512x512 lena.jpg -> OpenCV reference rect [209,181,352,388].
    let square = decode_rgb(&face_image);
    assert_eq!((square.width(), square.height()), (512, 512));
    let detections = detector.detect(&square, 0).unwrap();
    println!("square detections: {detections:?}");
    assert_matches_reference(
        &square,
        &detections,
        [209, 181, 352, 388],
        0.913_480_460_643_768_3,
    );

    // Optional 768x512 wide fixture -> OpenCV reference rect [330,174,482,395].
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
}
