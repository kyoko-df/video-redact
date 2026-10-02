use std::env;
use std::error::Error;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use video_redact_core::{
    CpuRedactor, Detector, Rect, RedactionEffect, RedactionPolicy, Redactor, RgbFrame,
    StaticDetector,
};
use video_redact_ffmpeg::{PipelineOptions, PipelineReport, redact_video};

const HELP: &str = "\
video-redact — GPU video privacy redaction scaffold

USAGE:
    video-redact info
    video-redact demo [--output PATH] [--backend cpu|cuda]
    video-redact redact --input PATH --output PATH [--roi L,T,R,B ...] [OPTIONS]

REDACT OPTIONS:
    --roi L,T,R,B       Static half-open ROI; may be supplied more than once
    --detector NAME     Detector: static or yunet (default: static)
    --model PATH        YuNet ONNX model path; required by --detector yunet
    --min-confidence F  Confidence needed to redact a detection (default: 0.5)
    --padding N         Pixels added around each detection (default: 0)
    --block-size N      Mosaic block size in pixels (default: 16)
    --backend NAME      Redaction backend: cpu or cuda (default: cpu)
    --overwrite         Replace the output file if it already exists
";

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        Some("info") => {
            println!("video-redact {}", env!("CARGO_PKG_VERSION"));
            println!("cuda feature: {}", cfg!(feature = "cuda"));
            Ok(())
        }
        Some("demo") => {
            let args = args.collect::<Vec<_>>();
            run_demo(&args)
        }
        Some("redact") => {
            let args = args.collect::<Vec<_>>();
            run_video_redact(&args)
        }
        Some("help" | "--help" | "-h") | None => {
            print!("{HELP}");
            Ok(())
        }
        Some(command) => Err(format!("unknown command `{command}`\n\n{HELP}").into()),
    }
}

fn run_video_redact(args: &[String]) -> Result<(), Box<dyn Error>> {
    let parsed = parse_video_args(args)?;
    // The detector (and any model file) is loaded before the FFmpeg pipeline
    // starts, so a bad --model fails before the output is touched.
    let mut detector = build_detector(&parsed)?;
    let mut policy = RedactionPolicy::new(parsed.min_confidence, parsed.padding)?;
    let options = PipelineOptions {
        input: parsed.input,
        output: parsed.output,
        effect: RedactionEffect::Mosaic {
            block_size: parsed.block_size,
        },
        overwrite: parsed.overwrite,
    };

    let report = match parsed.backend.as_str() {
        "cpu" => redact_video(&mut *detector, &mut policy, &mut CpuRedactor, &options)?,
        "cuda" => redact_video_with_cuda(&mut *detector, &mut policy, &options)?,
        other => return Err(format!("unsupported backend `{other}`; use cpu or cuda").into()),
    };

    print_pipeline_report(&options, &report, &parsed.backend);
    Ok(())
}

fn build_detector(parsed: &VideoArgs) -> Result<Box<dyn Detector>, Box<dyn Error>> {
    match parsed.detector.as_str() {
        "static" => Ok(Box::new(StaticDetector::new(parsed.regions.clone()))),
        "yunet" => build_yunet_detector(parsed),
        other => Err(format!("unsupported detector `{other}`; use static or yunet").into()),
    }
}

#[cfg(feature = "yunet")]
fn build_yunet_detector(parsed: &VideoArgs) -> Result<Box<dyn Detector>, Box<dyn Error>> {
    let model = parsed
        .model
        .as_ref()
        .ok_or("--detector yunet requires --model PATH")?;
    Ok(Box::new(video_redact_detect::YuNetDetector::new(
        model,
        parsed.min_confidence,
    )?))
}

#[cfg(not(feature = "yunet"))]
fn build_yunet_detector(parsed: &VideoArgs) -> Result<Box<dyn Detector>, Box<dyn Error>> {
    Err(format!(
        "detector `yunet` is not available (model {:?}); rebuild with `--features yunet`",
        parsed.model
    )
    .into())
}

#[derive(Debug)]
struct VideoArgs {
    input: PathBuf,
    output: PathBuf,
    regions: Vec<Rect>,
    detector: String,
    model: Option<PathBuf>,
    min_confidence: f32,
    padding: u32,
    block_size: u32,
    backend: String,
    overwrite: bool,
}

fn parse_video_args(args: &[String]) -> Result<VideoArgs, Box<dyn Error>> {
    let mut input = None;
    let mut output = None;
    let mut regions = Vec::new();
    let mut detector = String::from("static");
    let mut model = None;
    let mut min_confidence = 0.5_f32;
    let mut padding = 0_u32;
    let mut block_size = 16;
    let mut backend = String::from("cpu");
    let mut overwrite = false;
    let mut index = 0;

    while index < args.len() {
        match args[index].as_str() {
            "--input" => {
                index += 1;
                input = Some(PathBuf::from(
                    args.get(index).ok_or("--input requires a path")?,
                ));
            }
            "--output" => {
                index += 1;
                output = Some(PathBuf::from(
                    args.get(index).ok_or("--output requires a path")?,
                ));
            }
            "--roi" => {
                index += 1;
                regions.push(parse_rect(
                    args.get(index).ok_or("--roi requires L,T,R,B")?,
                )?);
            }
            "--detector" => {
                index += 1;
                detector.clone_from(
                    args.get(index)
                        .ok_or("--detector requires static or yunet")?,
                );
            }
            "--model" => {
                index += 1;
                model = Some(PathBuf::from(
                    args.get(index).ok_or("--model requires a path")?,
                ));
            }
            "--min-confidence" => {
                index += 1;
                min_confidence = args
                    .get(index)
                    .ok_or("--min-confidence requires a value in [0, 1]")?
                    .parse::<f32>()
                    .map_err(|_| "--min-confidence requires a value in [0, 1]")?;
                if !(0.0..=1.0).contains(&min_confidence) {
                    return Err("--min-confidence requires a value in [0, 1]".into());
                }
            }
            "--padding" => {
                index += 1;
                padding = args
                    .get(index)
                    .ok_or("--padding requires a non-negative integer")?
                    .parse::<u32>()
                    .map_err(|_| "--padding requires a non-negative integer")?;
            }
            "--block-size" => {
                index += 1;
                block_size = args
                    .get(index)
                    .ok_or("--block-size requires a positive integer")?
                    .parse::<u32>()
                    .map_err(|_| "--block-size requires a positive integer")?;
                if block_size == 0 {
                    return Err("--block-size must be greater than zero".into());
                }
            }
            "--backend" => {
                index += 1;
                backend.clone_from(args.get(index).ok_or("--backend requires cpu or cuda")?);
            }
            "--overwrite" => overwrite = true,
            argument => return Err(format!("unknown redact argument `{argument}`").into()),
        }
        index += 1;
    }

    check_detector_args(&detector, model.as_ref(), &regions)?;
    match backend.as_str() {
        "cpu" | "cuda" => {}
        other => return Err(format!("unsupported backend `{other}`; use cpu or cuda").into()),
    }

    Ok(VideoArgs {
        input: input.ok_or("redact requires --input PATH")?,
        output: output.ok_or("redact requires --output PATH")?,
        regions,
        detector,
        model,
        min_confidence,
        padding,
        block_size,
        backend,
        overwrite,
    })
}

fn check_detector_args(
    detector: &str,
    model: Option<&PathBuf>,
    regions: &[Rect],
) -> Result<(), Box<dyn Error>> {
    match detector {
        "static" => {
            if regions.is_empty() {
                return Err("redact requires at least one --roi".into());
            }
            if model.is_some() {
                return Err("--model is only valid with --detector yunet".into());
            }
        }
        "yunet" => {
            if model.is_none() {
                return Err("--detector yunet requires --model PATH".into());
            }
            if !regions.is_empty() {
                return Err("--detector yunet does not accept --roi".into());
            }
        }
        other => {
            return Err(format!("unsupported detector `{other}`; use static or yunet").into());
        }
    }
    Ok(())
}

fn parse_rect(value: &str) -> Result<Rect, Box<dyn Error>> {
    let coordinates = value
        .split(',')
        .map(str::parse::<u32>)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| format!("invalid ROI `{value}`; expected L,T,R,B"))?;
    let [left, top, right, bottom] = coordinates.as_slice() else {
        return Err(format!("invalid ROI `{value}`; expected L,T,R,B").into());
    };
    if left >= right || top >= bottom {
        return Err(format!("invalid ROI `{value}`; right and bottom must be greater").into());
    }
    Ok(Rect::new(*left, *top, *right, *bottom))
}

fn print_pipeline_report(options: &PipelineOptions, report: &PipelineReport, backend: &str) {
    println!(
        "wrote {}: {} frames, {}x{} @ {} fps, {backend} backend",
        options.output.display(),
        report.frames_processed,
        report.video.width,
        report.video.height,
        report.video.frame_rate,
    );
    println!(
        "detections: {}, regions redacted: {}, flagged for review: {}",
        report.detections,
        report.redacted_regions,
        report.review_records.len(),
    );
    let timings = &report.timings;
    println!(
        "timings: probe {:.2?}, decode {:.2?}, inference {:.2?}, redact {:.2?}, encode {:.2?}, total {:.2?}",
        timings.probe,
        timings.decode,
        timings.inference,
        timings.redact,
        timings.encode,
        report.elapsed,
    );
}

#[cfg(feature = "cuda")]
fn redact_video_with_cuda(
    detector: &mut dyn Detector,
    policy: &mut RedactionPolicy,
    options: &PipelineOptions,
) -> Result<PipelineReport, Box<dyn Error>> {
    let mut redactor = video_redact_cuda::CudaRedactor::new(0)?;
    Ok(redact_video(detector, policy, &mut redactor, options)?)
}

#[cfg(not(feature = "cuda"))]
fn redact_video_with_cuda(
    _detector: &mut dyn Detector,
    _policy: &mut RedactionPolicy,
    _options: &PipelineOptions,
) -> Result<PipelineReport, Box<dyn Error>> {
    Err("CUDA support is disabled; rebuild with `--features cuda`".into())
}

fn run_demo(args: &[String]) -> Result<(), Box<dyn Error>> {
    let mut output = PathBuf::from("demo.ppm");
    let mut backend = "cpu";
    let mut index = 0;

    while index < args.len() {
        match args[index].as_str() {
            "--output" => {
                index += 1;
                output = PathBuf::from(args.get(index).ok_or("--output requires a path")?);
            }
            "--backend" => {
                index += 1;
                backend = args
                    .get(index)
                    .ok_or("--backend requires cpu or cuda")?
                    .as_str();
            }
            argument => return Err(format!("unknown demo argument `{argument}`").into()),
        }
        index += 1;
    }

    let mut frame = demo_frame()?;
    let regions = [Rect::new(46, 34, 142, 118), Rect::new(178, 58, 284, 154)];
    let effect = RedactionEffect::Mosaic { block_size: 12 };

    match backend {
        "cpu" => CpuRedactor.redact(&mut frame, &regions, effect)?,
        "cuda" => redact_with_cuda(&mut frame, &regions, effect)?,
        other => return Err(format!("unsupported backend `{other}`; use cpu or cuda").into()),
    }

    write_ppm(&output, &frame)?;
    println!("wrote {} using the {backend} backend", output.display());
    Ok(())
}

#[cfg(feature = "cuda")]
fn redact_with_cuda(
    frame: &mut RgbFrame,
    regions: &[Rect],
    effect: RedactionEffect,
) -> Result<(), Box<dyn Error>> {
    let mut redactor = video_redact_cuda::CudaRedactor::new(0)?;
    redactor.redact(frame, regions, effect)?;
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn redact_with_cuda(
    _frame: &mut RgbFrame,
    _regions: &[Rect],
    _effect: RedactionEffect,
) -> Result<(), Box<dyn Error>> {
    Err("CUDA support is disabled; rebuild with `--features cuda`".into())
}

fn demo_frame() -> Result<RgbFrame, Box<dyn Error>> {
    const WIDTH: u32 = 320;
    const HEIGHT: u32 = 180;
    let mut pixels = Vec::with_capacity(WIDTH as usize * HEIGHT as usize * 3);

    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            pixels.push(u8::try_from(x * 255 / WIDTH).unwrap_or(u8::MAX));
            pixels.push(u8::try_from(y * 255 / HEIGHT).unwrap_or(u8::MAX));
            pixels.push(u8::try_from((x + y) * 255 / (WIDTH + HEIGHT)).unwrap_or(u8::MAX));
        }
    }

    Ok(RgbFrame::new(WIDTH, HEIGHT, pixels)?)
}

fn write_ppm(path: &Path, frame: &RgbFrame) -> Result<(), Box<dyn Error>> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    write!(writer, "P6\n{} {}\n255\n", frame.width(), frame.height())?;
    writer.write_all(frame.data())?;
    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::float_cmp)]
    fn parses_repeated_rois_and_options() {
        let args = [
            "--input",
            "in.mp4",
            "--output",
            "out.mp4",
            "--roi",
            "1,2,30,40",
            "--roi",
            "50,60,70,80",
            "--block-size",
            "12",
            "--overwrite",
        ]
        .map(String::from);
        let parsed = parse_video_args(&args).unwrap();

        assert_eq!(parsed.input, PathBuf::from("in.mp4"));
        assert_eq!(parsed.regions.len(), 2);
        assert_eq!(parsed.block_size, 12);
        assert_eq!(parsed.min_confidence, 0.5);
        assert_eq!(parsed.padding, 0);
        assert!(parsed.overwrite);
    }

    #[test]
    fn rejects_missing_roi() {
        let args = ["--input", "in.mp4", "--output", "out.mp4"].map(String::from);
        let error = parse_video_args(&args).unwrap_err();
        assert!(error.to_string().contains("--roi"));
    }

    #[test]
    fn parses_yunet_detector_without_roi() {
        let args = [
            "--input",
            "in.mp4",
            "--output",
            "out.mp4",
            "--detector",
            "yunet",
            "--model",
            "models/face_detection_yunet_2023mar.onnx",
        ]
        .map(String::from);
        let parsed = parse_video_args(&args).unwrap();

        assert_eq!(parsed.detector, "yunet");
        assert_eq!(
            parsed.model.as_deref(),
            Some(Path::new("models/face_detection_yunet_2023mar.onnx"))
        );
        assert!(parsed.regions.is_empty());
    }

    #[test]
    fn rejects_yunet_without_model() {
        let args = [
            "--input",
            "in.mp4",
            "--output",
            "out.mp4",
            "--detector",
            "yunet",
        ]
        .map(String::from);
        let error = parse_video_args(&args).unwrap_err();
        assert!(error.to_string().contains("--model"), "{error}");
    }

    #[test]
    fn rejects_unknown_detector() {
        let args = [
            "--input",
            "in.mp4",
            "--output",
            "out.mp4",
            "--detector",
            "hog",
        ]
        .map(String::from);
        let error = parse_video_args(&args).unwrap_err();
        assert!(
            error.to_string().contains("unsupported detector"),
            "{error}"
        );
    }

    #[test]
    fn rejects_model_with_static_detector() {
        let args = [
            "--input",
            "in.mp4",
            "--output",
            "out.mp4",
            "--roi",
            "1,2,30,40",
            "--model",
            "model.onnx",
        ]
        .map(String::from);
        let error = parse_video_args(&args).unwrap_err();
        assert!(error.to_string().contains("--model"), "{error}");
    }

    #[test]
    fn rejects_roi_with_yunet() {
        let args = [
            "--input",
            "in.mp4",
            "--output",
            "out.mp4",
            "--detector",
            "yunet",
            "--model",
            "model.onnx",
            "--roi",
            "1,2,30,40",
        ]
        .map(String::from);
        let error = parse_video_args(&args).unwrap_err();
        assert!(error.to_string().contains("--roi"), "{error}");
    }

    #[test]
    fn rejects_out_of_range_min_confidence() {
        for value in ["1.5", "-0.1", "nan", "abc"] {
            let args = [
                "--input",
                "in.mp4",
                "--output",
                "out.mp4",
                "--roi",
                "1,2,3,4",
                "--min-confidence",
                value,
            ]
            .map(String::from);
            assert!(
                parse_video_args(&args).is_err(),
                "expected `{value}` to be rejected"
            );
        }
    }

    #[test]
    fn rejects_unknown_backend_early() {
        let args = [
            "--input",
            "in.mp4",
            "--output",
            "out.mp4",
            "--roi",
            "1,2,3,4",
            "--backend",
            "opencl",
        ]
        .map(String::from);
        let error = parse_video_args(&args).unwrap_err();
        assert!(error.to_string().contains("unsupported backend"), "{error}");
    }

    #[cfg(not(feature = "yunet"))]
    #[test]
    fn yunet_detector_reports_missing_feature() {
        let args = [
            "--input",
            "in.mp4",
            "--output",
            "out.mp4",
            "--detector",
            "yunet",
            "--model",
            "model.onnx",
        ]
        .map(String::from);
        let parsed = parse_video_args(&args).unwrap();
        let error = build_detector(&parsed).err().unwrap();
        assert!(error.to_string().contains("--features yunet"), "{error}");
    }

    #[cfg(feature = "yunet")]
    #[test]
    fn missing_model_fails_before_output() {
        // Unique path: a stale file from an earlier run must not mask a
        // regression that creates the output before loading the model.
        let output = std::env::temp_dir().join(format!(
            "video-redact-missing-model-{}-{:?}.mp4",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let args = [
            "--input",
            "in.mp4",
            "--output",
            output.to_str().unwrap(),
            "--detector",
            "yunet",
            "--model",
            "/nonexistent/yunet.onnx",
        ]
        .map(String::from);
        let error = run_video_redact(&args).err().unwrap();
        assert!(error.to_string().contains("model"), "{error}");
        assert!(!output.exists(), "output must not be created: {error}");
    }

    #[test]
    fn rejects_inverted_roi() {
        let error = parse_rect("10,20,5,30").unwrap_err();
        assert!(error.to_string().contains("right and bottom"));
    }
}
