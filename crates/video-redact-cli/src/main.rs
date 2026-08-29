use std::env;
use std::error::Error;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use video_redact_core::{CpuRedactor, Rect, RedactionEffect, Redactor, RgbFrame};

const HELP: &str = "\
video-redact — GPU video privacy redaction scaffold

USAGE:
    video-redact info
    video-redact demo [--output PATH] [--backend cpu|cuda]
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
        Some("help" | "--help" | "-h") | None => {
            print!("{HELP}");
            Ok(())
        }
        Some(command) => Err(format!("unknown command `{command}`\n\n{HELP}").into()),
    }
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
