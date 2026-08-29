//! FFmpeg-backed video decoding and encoding pipeline.
//!
//! The first implementation deliberately uses the stable `ffmpeg`/`ffprobe`
//! command-line boundary. Frames cross that boundary as tightly packed RGB24,
//! which lets both the CPU reference backend and the current CUDA backend share
//! exactly the same pipeline. A later NVDEC/NVENC implementation can replace
//! this module without changing the domain-level redaction API.

use std::error::Error;
use std::ffi::{OsStr, OsString};
use std::fmt::{Display, Formatter};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, ExitStatus, Stdio};

use video_redact_core::{Rect, RedactError, RedactionEffect, Redactor, RgbFrame};

/// Runtime settings for a file-to-file redaction job.
#[derive(Clone, Debug)]
pub struct PipelineOptions {
    pub input: PathBuf,
    pub output: PathBuf,
    pub regions: Vec<Rect>,
    pub effect: RedactionEffect,
    pub overwrite: bool,
}

/// Information reported by `ffprobe` for the first video stream.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VideoInfo {
    pub width: u32,
    pub height: u32,
    pub frame_rate: FrameRate,
}

/// A positive rational frame rate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FrameRate {
    numerator: u32,
    denominator: u32,
}

impl Display for FrameRate {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}/{}", self.numerator, self.denominator)
    }
}

/// Summary of a completed pipeline run.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PipelineReport {
    pub frames_processed: u64,
    pub video: VideoInfo,
}

/// Runs MP4/video decode -> RGB24 ROI redaction -> H.264 MP4 encode.
///
/// The first video stream is processed. If the input contains audio, all input
/// audio streams are stream-copied into the output. Input metadata is copied as
/// well. The output is encoded as H.264 (`libx264`) with `yuv420p` pixel format.
///
/// # Errors
///
/// Returns an error when the paths or stream metadata are invalid, `FFmpeg` is not
/// installed, either `FFmpeg` process fails, a frame is truncated, or the selected
/// redaction backend fails.
pub fn redact_video(
    redactor: &mut dyn Redactor,
    options: &PipelineOptions,
) -> Result<PipelineReport, FfmpegError> {
    validate_paths(options)?;
    let ffmpeg = executable("VIDEO_REDACT_FFMPEG", "ffmpeg");
    let ffprobe = executable("VIDEO_REDACT_FFPROBE", "ffprobe");
    let video = probe_video(&ffprobe, &options.input)?;
    let frame_len = frame_len(&video)?;

    let mut decoder = spawn_decoder(&ffmpeg, &options.input)?;
    let mut encoder = match spawn_encoder(&ffmpeg, options, &video) {
        Ok(encoder) => encoder,
        Err(error) => {
            stop_child(&mut decoder.child);
            return Err(error);
        }
    };

    let stream_result = process_frames(
        &mut decoder.stdout,
        &mut encoder.stdin,
        frame_len,
        &video,
        redactor,
        &options.regions,
        options.effect,
    );

    // Closing encoder stdin is the end-of-stream signal for its rawvideo input.
    drop(encoder.stdin);

    let frames_processed = match stream_result {
        Ok(frames_processed) => frames_processed,
        Err(error) => {
            stop_child(&mut decoder.child);
            stop_child(&mut encoder.child);
            return Err(error);
        }
    };

    let decoder_status = decoder.child.wait().map_err(FfmpegError::Io)?;
    let encoder_status = encoder.child.wait().map_err(FfmpegError::Io)?;
    // Always reap both children before reporting either process failure.
    ensure_success("ffmpeg decoder", decoder_status)?;
    ensure_success("ffmpeg encoder", encoder_status)?;

    Ok(PipelineReport {
        frames_processed,
        video,
    })
}

fn executable(variable: &str, fallback: &str) -> OsString {
    std::env::var_os(variable).unwrap_or_else(|| OsString::from(fallback))
}

fn validate_paths(options: &PipelineOptions) -> Result<(), FfmpegError> {
    if options.input == options.output {
        return Err(FfmpegError::InvalidConfiguration(
            "input and output paths must be different".into(),
        ));
    }
    if !options.input.is_file() {
        return Err(FfmpegError::InvalidConfiguration(format!(
            "input is not a file: {}",
            options.input.display()
        )));
    }
    if options.output.exists() && !options.overwrite {
        return Err(FfmpegError::InvalidConfiguration(format!(
            "output already exists: {} (pass --overwrite to replace it)",
            options.output.display()
        )));
    }
    if options.regions.is_empty() {
        return Err(FfmpegError::InvalidConfiguration(
            "at least one ROI is required".into(),
        ));
    }
    Ok(())
}

fn probe_video(ffprobe: &OsStr, input: &Path) -> Result<VideoInfo, FfmpegError> {
    let output = Command::new(ffprobe)
        .args([
            OsStr::new("-v"),
            OsStr::new("error"),
            OsStr::new("-select_streams"),
            OsStr::new("v:0"),
            OsStr::new("-show_entries"),
            OsStr::new("stream=width,height,avg_frame_rate,r_frame_rate"),
            OsStr::new("-of"),
            OsStr::new("default=noprint_wrappers=1:nokey=0"),
        ])
        .arg(input)
        .output()
        .map_err(|error| command_io_error("ffprobe", error))?;

    if !output.status.success() {
        return Err(FfmpegError::ProcessFailed {
            process: "ffprobe",
            status: output.status,
            detail: String::from_utf8_lossy(&output.stderr).trim().to_owned(),
        });
    }

    let text = std::str::from_utf8(&output.stdout)
        .map_err(|_| FfmpegError::InvalidProbe("ffprobe output is not UTF-8".into()))?;
    parse_video_info(text)
}

fn parse_video_info(text: &str) -> Result<VideoInfo, FfmpegError> {
    let mut width = None;
    let mut height = None;
    let mut average_frame_rate = None;
    let mut nominal_frame_rate = None;

    for line in text.lines() {
        let Some((key, value)) = line.trim().split_once('=') else {
            continue;
        };
        match key {
            "width" => width = value.parse::<u32>().ok(),
            "height" => height = value.parse::<u32>().ok(),
            "avg_frame_rate" => average_frame_rate = parse_frame_rate(value),
            "r_frame_rate" => nominal_frame_rate = parse_frame_rate(value),
            _ => {}
        }
    }

    let width = width
        .filter(|value| *value > 0)
        .ok_or_else(|| FfmpegError::InvalidProbe("first video stream has no valid width".into()))?;
    let height = height.filter(|value| *value > 0).ok_or_else(|| {
        FfmpegError::InvalidProbe("first video stream has no valid height".into())
    })?;
    let frame_rate = average_frame_rate.or(nominal_frame_rate).ok_or_else(|| {
        FfmpegError::InvalidProbe("first video stream has no usable frame rate".into())
    })?;

    Ok(VideoInfo {
        width,
        height,
        frame_rate,
    })
}

fn parse_frame_rate(value: &str) -> Option<FrameRate> {
    let (numerator, denominator) = value.split_once('/')?;
    let numerator = numerator.parse::<u32>().ok()?;
    let denominator = denominator.parse::<u32>().ok()?;
    (numerator > 0 && denominator > 0).then_some(FrameRate {
        numerator,
        denominator,
    })
}

fn frame_len(video: &VideoInfo) -> Result<usize, FfmpegError> {
    usize::try_from(video.width)
        .ok()
        .and_then(|width| width.checked_mul(3))
        .and_then(|stride| {
            usize::try_from(video.height)
                .ok()
                .and_then(|height| stride.checked_mul(height))
        })
        .ok_or_else(|| FfmpegError::InvalidProbe("video dimensions overflow memory size".into()))
}

struct Decoder {
    child: Child,
    stdout: std::process::ChildStdout,
}

fn spawn_decoder(ffmpeg: &OsStr, input: &Path) -> Result<Decoder, FfmpegError> {
    let mut child = Command::new(ffmpeg)
        .args([
            OsStr::new("-v"),
            OsStr::new("error"),
            OsStr::new("-nostdin"),
            OsStr::new("-noautorotate"),
            OsStr::new("-i"),
        ])
        .arg(input)
        .args([
            OsStr::new("-map"),
            OsStr::new("0:v:0"),
            OsStr::new("-f"),
            OsStr::new("rawvideo"),
            OsStr::new("-pix_fmt"),
            OsStr::new("rgb24"),
            OsStr::new("pipe:1"),
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|error| command_io_error("ffmpeg decoder", error))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| FfmpegError::InvalidConfiguration("failed to open decoder stdout".into()))?;
    Ok(Decoder { child, stdout })
}

struct Encoder {
    child: Child,
    stdin: ChildStdin,
}

fn spawn_encoder(
    ffmpeg: &OsStr,
    options: &PipelineOptions,
    video: &VideoInfo,
) -> Result<Encoder, FfmpegError> {
    let size = format!("{}x{}", video.width, video.height);
    let frame_rate = video.frame_rate.to_string();
    let overwrite = if options.overwrite { "-y" } else { "-n" };
    let mut child = Command::new(ffmpeg)
        .args([
            OsStr::new("-v"),
            OsStr::new("error"),
            OsStr::new(overwrite),
            OsStr::new("-f"),
            OsStr::new("rawvideo"),
            OsStr::new("-pix_fmt"),
            OsStr::new("rgb24"),
            OsStr::new("-video_size"),
            OsStr::new(&size),
            OsStr::new("-framerate"),
            OsStr::new(&frame_rate),
            OsStr::new("-i"),
            OsStr::new("pipe:0"),
            OsStr::new("-i"),
        ])
        .arg(&options.input)
        .args([
            OsStr::new("-map"),
            OsStr::new("0:v:0"),
            OsStr::new("-map"),
            OsStr::new("1:a?"),
            OsStr::new("-map_metadata"),
            OsStr::new("1"),
            OsStr::new("-c:v"),
            OsStr::new("libx264"),
            OsStr::new("-preset"),
            OsStr::new("medium"),
            OsStr::new("-crf"),
            OsStr::new("23"),
            OsStr::new("-pix_fmt"),
            OsStr::new("yuv420p"),
            OsStr::new("-c:a"),
            OsStr::new("copy"),
            OsStr::new("-movflags"),
            OsStr::new("+faststart"),
        ])
        .arg(&options.output)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|error| command_io_error("ffmpeg encoder", error))?;
    let stdin = child
        .stdin
        .take()
        .ok_or_else(|| FfmpegError::InvalidConfiguration("failed to open encoder stdin".into()))?;
    Ok(Encoder { child, stdin })
}

#[allow(clippy::too_many_arguments)]
fn process_frames(
    decoder: &mut impl Read,
    encoder: &mut impl Write,
    frame_len: usize,
    video: &VideoInfo,
    redactor: &mut dyn Redactor,
    regions: &[Rect],
    effect: RedactionEffect,
) -> Result<u64, FfmpegError> {
    let mut frames_processed = 0_u64;
    let mut data = vec![0_u8; frame_len];
    loop {
        if !read_frame(decoder, &mut data)? {
            break;
        }
        let mut frame = RgbFrame::new(video.width, video.height, data)?;
        redactor.redact(&mut frame, regions, effect)?;
        encoder.write_all(frame.data()).map_err(FfmpegError::Io)?;
        data = frame.into_data();
        frames_processed = frames_processed.saturating_add(1);
    }
    Ok(frames_processed)
}

fn read_frame(reader: &mut impl Read, data: &mut [u8]) -> Result<bool, FfmpegError> {
    let mut filled = 0;
    while filled < data.len() {
        match reader.read(&mut data[filled..]) {
            Ok(0) if filled == 0 => return Ok(false),
            Ok(0) => {
                return Err(FfmpegError::TruncatedFrame {
                    expected: data.len(),
                    actual: filled,
                });
            }
            Ok(read) => filled += read,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(FfmpegError::Io(error)),
        }
    }
    Ok(true)
}

fn ensure_success(process: &'static str, status: ExitStatus) -> Result<(), FfmpegError> {
    if status.success() {
        Ok(())
    } else {
        Err(FfmpegError::ProcessFailed {
            process,
            status,
            detail: String::new(),
        })
    }
}

fn stop_child(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}

fn command_io_error(process: &'static str, error: io::Error) -> FfmpegError {
    if error.kind() == io::ErrorKind::NotFound {
        FfmpegError::MissingExecutable(process)
    } else {
        FfmpegError::Io(error)
    }
}

#[derive(Debug)]
pub enum FfmpegError {
    InvalidConfiguration(String),
    InvalidProbe(String),
    MissingExecutable(&'static str),
    ProcessFailed {
        process: &'static str,
        status: ExitStatus,
        detail: String,
    },
    TruncatedFrame {
        expected: usize,
        actual: usize,
    },
    Io(io::Error),
    Redaction(RedactError),
}

impl Display for FfmpegError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfiguration(message) => formatter.write_str(message),
            Self::InvalidProbe(message) => write!(formatter, "invalid ffprobe output: {message}"),
            Self::MissingExecutable(process) => write!(
                formatter,
                "{process} executable was not found; install FFmpeg or set VIDEO_REDACT_FFMPEG/VIDEO_REDACT_FFPROBE"
            ),
            Self::ProcessFailed {
                process,
                status,
                detail,
            } if detail.is_empty() => write!(formatter, "{process} exited with {status}"),
            Self::ProcessFailed {
                process,
                status,
                detail,
            } => write!(formatter, "{process} exited with {status}: {detail}"),
            Self::TruncatedFrame { expected, actual } => write!(
                formatter,
                "decoder returned a truncated RGB frame: expected {expected} bytes, got {actual}"
            ),
            Self::Io(error) => write!(formatter, "video pipeline I/O error: {error}"),
            Self::Redaction(error) => write!(formatter, "redaction failed: {error}"),
        }
    }
}

impl Error for FfmpegError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Redaction(error) => Some(error),
            _ => None,
        }
    }
}

impl From<RedactError> for FfmpegError {
    fn from(error: RedactError) -> Self {
        Self::Redaction(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use video_redact_core::CpuRedactor;

    #[test]
    fn parses_ffprobe_video_stream() {
        let info =
            parse_video_info("width=1920\nheight=1080\navg_frame_rate=30000/1001\n").unwrap();
        assert_eq!(
            info,
            VideoInfo {
                width: 1920,
                height: 1080,
                frame_rate: FrameRate {
                    numerator: 30_000,
                    denominator: 1_001,
                },
            }
        );
    }

    #[test]
    fn rejects_zero_frame_rate() {
        let error = parse_video_info("width=2\nheight=2\navg_frame_rate=0/0\nr_frame_rate=0/0\n")
            .unwrap_err();
        assert!(error.to_string().contains("frame rate"));
    }

    #[test]
    fn falls_back_to_nominal_frame_rate() {
        let info =
            parse_video_info("width=2\nheight=2\navg_frame_rate=0/0\nr_frame_rate=24/1\n").unwrap();
        assert_eq!(info.frame_rate.to_string(), "24/1");
    }

    #[test]
    fn detects_truncated_frame() {
        let error = read_frame(&mut &b"short"[..], &mut [0; 10]).unwrap_err();
        assert!(matches!(
            error,
            FfmpegError::TruncatedFrame {
                expected: 10,
                actual: 5
            }
        ));
    }

    #[test]
    fn processes_rgb_frames_through_redactor() {
        let video = VideoInfo {
            width: 2,
            height: 2,
            frame_rate: FrameRate {
                numerator: 25,
                denominator: 1,
            },
        };
        let input: Vec<u8> = (0..24).collect();
        let mut output = Vec::new();
        let count = process_frames(
            &mut input.as_slice(),
            &mut output,
            12,
            &video,
            &mut CpuRedactor,
            &[Rect::new(0, 0, 2, 2)],
            RedactionEffect::Mosaic { block_size: 2 },
        )
        .unwrap();

        assert_eq!(count, 2);
        assert_eq!(&output[..12], &[4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6]);
        assert_eq!(
            &output[12..],
            &[16, 17, 18, 16, 17, 18, 16, 17, 18, 16, 17, 18]
        );
    }
}
