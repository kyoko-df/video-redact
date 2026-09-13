use std::env;
use std::error::Error;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use video_redact_core::{CpuRedactor, Rect, RedactionEffect, Redactor, RgbFrame};
use video_redact_ffmpeg::{PipelineOptions, PipelineReport, redact_video};

const HELP: &str = "\
video-redact — GPU video privacy redaction scaffold

USAGE:
    video-redact info
    video-redact demo [--output PATH] [--backend cpu|cuda]
    video-redact redact --input PATH --output PATH --roi L,T,R,B [OPTIONS]

REDACT OPTIONS:
    --roi L,T,R,B       Static half-open ROI; may be supplied more than once
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
    let options = PipelineOptions {
        input: parsed.input,
        output: parsed.output,
        regions: parsed.regions,
        effect: RedactionEffect::Mosaic {
            block_size: parsed.block_size,
        },
        overwrite: parsed.overwrite,
    };

    let report = match parsed.backend.as_str() {
        "cpu" => redact_video(&mut CpuRedactor, &options)?,
        "cuda" => redact_video_with_cuda(&options)?,
        other => return Err(format!("unsupported backend `{other}`; use cpu or cuda").into()),
    };

    print_pipeline_report(&options, &report, &parsed.backend);
    Ok(())
}

#[derive(Debug)]
struct VideoArgs {
    input: PathBuf,
    output: PathBuf,
    regions: Vec<Rect>,
    block_size: u32,
    backend: String,
    overwrite: bool,
}

fn parse_video_args(args: &[String]) -> Result<VideoArgs, Box<dyn Error>> {
    let mut input = None;
    let mut output = None;
    let mut regions = Vec::new();
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

    Ok(VideoArgs {
        input: input.ok_or("redact requires --input PATH")?,
        output: output.ok_or("redact requires --output PATH")?,
        regions,
        block_size,
        backend,
        overwrite,
    })
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
    let timings = &report.timings;
    println!(
        "timings: probe {:.2?}, decode {:.2?}, redact {:.2?}, encode {:.2?}, total {:.2?}",
        timings.probe, timings.decode, timings.redact, timings.encode, report.elapsed,
    );
}

#[cfg(feature = "cuda")]
fn redact_video_with_cuda(options: &PipelineOptions) -> Result<PipelineReport, Box<dyn Error>> {
    let mut redactor = video_redact_cuda::CudaRedactor::new(0)?;
    Ok(redact_video(&mut redactor, options)?)
}

#[cfg(not(feature = "cuda"))]
fn redact_video_with_cuda(_options: &PipelineOptions) -> Result<PipelineReport, Box<dyn Error>> {
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
        assert!(parsed.overwrite);
    }

    #[test]
    fn rejects_inverted_roi() {
        let error = parse_rect("10,20,5,30").unwrap_err();
        assert!(error.to_string().contains("right and bottom"));
    }
}
