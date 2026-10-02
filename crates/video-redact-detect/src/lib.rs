//! `YuNet` ONNX face detection on the CPU via `tract`.
//!
//! Wraps the `OpenCV` Zoo `face_detection_yunet_2023mar` model behind the
//! [`Detector`] trait. Every frame is letterboxed onto a fixed 640x640 BGR
//! input, and the `cls`/`obj`/`bbox` heads at strides 8, 16 and 32 are decoded
//! into [`DetectionLabel::Face`] detections with greedy `NMS`.
//!
//! Geometry math clamps to the valid range before narrowing casts and relies
//! on Rust's saturating float-to-int conversions; indexing is `usize`/`i64`
//! only. The pedantic cast lints are allowed once for the whole module.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use std::fmt::{Debug, Formatter};
use std::path::Path;

use tract_onnx::prelude::{
    DatumType, Framework, InferenceModel, InferenceModelExt, TValue, Tensor, TypedModel,
    TypedSimplePlan, tvec,
};
use tract_onnx::tract_hir::infer::Factoid;
use tract_onnx::tract_hir::internal::DimLike;
use video_redact_core::{Detection, DetectionLabel, Detector, Rect, RedactError, RgbFrame};

/// Square edge of the model's fixed-size input tensor.
const INPUT_SIZE: usize = 640;
/// Anchor grid strides, one per detection head.
const STRIDES: [u32; 3] = [8, 16, 32];
/// `IoU` above which a weaker candidate is suppressed by a stronger one.
const NMS_IOU_THRESHOLD: f32 = 0.3;
/// Number of strongest candidates fed to `NMS`.
const NMS_TOP_K: usize = 5000;
/// Output head kinds in the order each stride consumes them.
const HEAD_KINDS: [&str; 3] = ["cls", "obj", "bbox"];

/// `YuNet` face detector backed by the `OpenCV` Zoo ONNX model running on
/// `tract`.
///
/// The ONNX model is loaded and optimized once at construction; `detect`
/// reuses the prepared plan for every frame.
pub struct YuNetDetector {
    plan: TypedSimplePlan<TypedModel>,
    /// Model output positions for `cls`/`obj`/`bbox` at each stride, so heads
    /// are found by name rather than by the model's output order.
    head_outputs: Vec<usize>,
    min_confidence: f32,
}

impl Debug for YuNetDetector {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("YuNetDetector")
            .field("min_confidence", &self.min_confidence)
            .finish_non_exhaustive()
    }
}

/// Geometry of the letterboxed model input relative to the source frame.
#[derive(Clone, Copy, Debug)]
struct Letterbox {
    /// Padded columns left of the resized image.
    offset_x: u32,
    /// Padded rows above the resized image.
    offset_y: u32,
    /// Resized width over the original width.
    scale_x: f32,
    /// Resized height over the original height.
    scale_y: f32,
    frame_width: u32,
    frame_height: u32,
}

/// Head tensors of one stride in model output order.
struct Head<'a> {
    stride: u32,
    cls: &'a [f32],
    obj: &'a [f32],
    bbox: &'a [f32],
}

/// One decoded anchor in source-frame coordinates, before NMS.
#[derive(Clone, Copy, Debug)]
struct Candidate {
    left: f32,
    top: f32,
    right: f32,
    bottom: f32,
    score: f32,
}

impl YuNetDetector {
    /// Loads the `YuNet` ONNX model from `model_path`.
    ///
    /// `min_confidence` is the minimum `sqrt(cls * obj)` score a candidate
    /// needs to be reported (`cls`/`obj`/`bbox` are the model's head names).
    ///
    /// # Errors
    ///
    /// Returns [`RedactError::InvalidConfidence`] when `min_confidence` is
    /// outside `[0.0, 1.0]`, and [`RedactError::Backend`] when the model file
    /// cannot be loaded or does not expose the expected `YuNet` interface.
    pub fn new(model_path: impl AsRef<Path>, min_confidence: f32) -> Result<Self, RedactError> {
        if !min_confidence.is_finite() || !(0.0..=1.0).contains(&min_confidence) {
            return Err(RedactError::InvalidConfidence);
        }
        let path = model_path.as_ref();
        let model = tract_onnx::onnx().model_for_path(path).map_err(|error| {
            backend(format!(
                "failed to load ONNX model {}: {error}",
                path.display()
            ))
        })?;
        validate_input(&model)?;
        let head_outputs = resolve_head_outputs(&model)?;
        let typed = model
            .into_optimized()
            .map_err(|error| backend(format!("failed to optimize ONNX model: {error}")))?;
        validate_outputs(&typed, &head_outputs)?;
        let plan = typed
            .into_runnable()
            .map_err(|error| backend(format!("failed to prepare ONNX model: {error}")))?;
        Ok(Self {
            plan,
            head_outputs,
            min_confidence,
        })
    }
}

impl Detector for YuNetDetector {
    fn detect(
        &mut self,
        frame: &RgbFrame,
        _frame_index: u64,
    ) -> Result<Vec<Detection>, RedactError> {
        let (input, letterbox) = preprocess(frame)?;
        let tensor = Tensor::from_shape(&[1, 3, INPUT_SIZE, INPUT_SIZE], &input)
            .map_err(|error| backend(format!("failed to build model input: {error}")))?;
        let outputs = self
            .plan
            .run(tvec![tensor.into()])
            .map_err(|error| backend(format!("YuNet inference failed: {error}")))?;
        let heads = extract_heads(&outputs, &self.head_outputs)?;
        let candidates = decode_heads(&heads, &letterbox, self.min_confidence)?;
        Ok(finalize_candidates(candidates, letterbox))
    }
}

fn backend(message: String) -> RedactError {
    RedactError::Backend(message)
}

fn validate_input(model: &InferenceModel) -> Result<(), RedactError> {
    let inputs = model
        .input_outlets()
        .map_err(|error| backend(format!("failed to inspect model inputs: {error}")))?;
    if inputs.len() != 1 {
        return Err(backend(format!(
            "expected a single model input, found {}",
            inputs.len()
        )));
    }
    let fact = model
        .input_fact(0)
        .map_err(|error| backend(format!("failed to inspect model input: {error}")))?;
    if fact.datum_type.concretize() != Some(DatumType::F32) {
        return Err(backend(format!(
            "unsupported model input type {:?}; expected f32",
            fact.datum_type
        )));
    }
    let concrete = fact
        .shape
        .as_concrete_finite()
        .map_err(|error| backend(format!("failed to read model input shape: {error}")))?;
    if concrete.as_deref() != Some(&[1, 3, INPUT_SIZE, INPUT_SIZE]) {
        return Err(backend(format!(
            "unsupported model input shape {:?}; expected [1, 3, {INPUT_SIZE}, {INPUT_SIZE}]",
            fact.shape
        )));
    }
    Ok(())
}

fn resolve_head_outputs(model: &InferenceModel) -> Result<Vec<usize>, RedactError> {
    let outputs = model
        .output_outlets()
        .map_err(|error| backend(format!("failed to inspect model outputs: {error}")))?;
    let labels: Vec<Option<String>> = outputs
        .iter()
        .map(|outlet| model.outlet_label(*outlet).map(String::from))
        .collect();
    map_head_outputs(&labels)
}

/// Maps output labels to positions in `cls`/`obj`/`bbox`-per-stride order, so
/// heads are found by name rather than by the model's output order.
fn map_head_outputs(labels: &[Option<String>]) -> Result<Vec<usize>, RedactError> {
    let mut positions = Vec::with_capacity(STRIDES.len() * HEAD_KINDS.len());
    for stride in STRIDES {
        for kind in HEAD_KINDS {
            let name = format!("{kind}_{stride}");
            let index = labels
                .iter()
                .position(|label| label.as_deref() == Some(name.as_str()))
                .ok_or_else(|| backend(format!("model is missing required output `{name}`")))?;
            positions.push(index);
        }
    }
    Ok(positions)
}

/// Checks the resolved outputs against the expected f32 head shapes on the
/// optimized model, so a structurally similar but incompatible graph fails at
/// load time instead of at the first frame.
fn validate_outputs(model: &TypedModel, positions: &[usize]) -> Result<(), RedactError> {
    let outputs = model
        .output_outlets()
        .map_err(|error| backend(format!("failed to inspect model outputs: {error}")))?;
    for (index, stride) in STRIDES.iter().enumerate() {
        let anchors = (INPUT_SIZE as u32 / stride).pow(2) as usize;
        for (kind_index, (kind, expected)) in [
            ("cls", vec![1, anchors, 1]),
            ("obj", vec![1, anchors, 1]),
            ("bbox", vec![1, anchors, 4]),
        ]
        .iter()
        .enumerate()
        {
            let name = format!("{kind}_{stride}");
            let position = positions[index * HEAD_KINDS.len() + kind_index];
            let outlet = outputs.get(position).ok_or_else(|| {
                backend(format!(
                    "optimized model has no output at position {position}"
                ))
            })?;
            let fact = model
                .outlet_fact(*outlet)
                .map_err(|error| backend(format!("failed to read `{name}` fact: {error}")))?;
            let dims: Option<Vec<usize>> =
                fact.shape.iter().map(|dim| dim.to_usize().ok()).collect();
            check_head_fact(&name, fact.datum_type, dims.as_deref(), expected)?;
        }
    }
    Ok(())
}

fn check_head_fact(
    name: &str,
    datum_type: DatumType,
    dims: Option<&[usize]>,
    expected: &[usize],
) -> Result<(), RedactError> {
    if datum_type != DatumType::F32 {
        return Err(backend(format!(
            "model output `{name}` has type {datum_type:?}; expected f32"
        )));
    }
    if dims != Some(expected) {
        return Err(backend(format!(
            "model output `{name}` has shape {dims:?}; expected {expected:?}"
        )));
    }
    Ok(())
}

/// Letterboxes `frame` into a BGR NCHW `1x3x640x640` f32 tensor. Channels keep
/// raw 0..=255 values; letterbox padding stays black.
fn preprocess(frame: &RgbFrame) -> Result<(Vec<f32>, Letterbox), RedactError> {
    let width = frame.width();
    let height = frame.height();
    if width == 0 || height == 0 {
        return Err(RedactError::InvalidDimensions);
    }
    let scale = (INPUT_SIZE as f64 / f64::from(width)).min(INPUT_SIZE as f64 / f64::from(height));
    let new_width = (f64::from(width) * scale)
        .round()
        .clamp(1.0, INPUT_SIZE as f64) as u32;
    let new_height = (f64::from(height) * scale)
        .round()
        .clamp(1.0, INPUT_SIZE as f64) as u32;
    let letterbox = Letterbox {
        offset_x: (INPUT_SIZE as u32 - new_width) / 2,
        offset_y: (INPUT_SIZE as u32 - new_height) / 2,
        scale_x: new_width as f32 / width as f32,
        scale_y: new_height as f32 / height as f32,
        frame_width: width,
        frame_height: height,
    };

    let plane = INPUT_SIZE * INPUT_SIZE;
    let mut tensor = vec![0.0_f32; 3 * plane];
    let data = frame.data();
    for dst_y in 0..new_height {
        let src_y = (dst_y as f32 + 0.5) * (height as f32 / new_height as f32) - 0.5;
        for dst_x in 0..new_width {
            let src_x = (dst_x as f32 + 0.5) * (width as f32 / new_width as f32) - 0.5;
            let (red, green, blue) = bilinear(data, width, height, src_x, src_y);
            let out_x = (letterbox.offset_x + dst_x) as usize;
            let out_y = (letterbox.offset_y + dst_y) as usize;
            tensor[out_y * INPUT_SIZE + out_x] = blue;
            tensor[plane + out_y * INPUT_SIZE + out_x] = green;
            tensor[2 * plane + out_y * INPUT_SIZE + out_x] = red;
        }
    }
    Ok((tensor, letterbox))
}

/// Clamped bilinear taps along one axis: the two source coordinates and the
/// blend factor toward the second tap. `i64` keeps the clamp safe for any
/// `u32` image size.
fn axis_taps(src: f32, len: u32) -> (u32, u32, f32) {
    let floor = src.floor();
    let frac = src - floor;
    let last = i64::from(len) - 1;
    let first = (floor as i64).clamp(0, last) as u32;
    let second = (floor as i64 + 1).clamp(0, last) as u32;
    (first, second, frac)
}

/// Byte offset of pixel `(x, y)` in a tightly packed RGB24 image.
fn pixel_offset(width: u32, x: u32, y: u32) -> usize {
    (y as usize * width as usize + x as usize) * 3
}

/// Bilinear half-pixel sample of a tightly packed RGB24 image, edge-clamped.
fn bilinear(data: &[u8], width: u32, height: u32, src_x: f32, src_y: f32) -> (f32, f32, f32) {
    let (x0, x1, fx) = axis_taps(src_x, width);
    let (y0, y1, fy) = axis_taps(src_y, height);

    let mut channels = [0.0_f32; 3];
    for (c, channel) in channels.iter_mut().enumerate() {
        let p00 = f32::from(data[pixel_offset(width, x0, y0) + c]);
        let p10 = f32::from(data[pixel_offset(width, x1, y0) + c]);
        let p01 = f32::from(data[pixel_offset(width, x0, y1) + c]);
        let p11 = f32::from(data[pixel_offset(width, x1, y1) + c]);
        *channel = p00 * (1.0 - fx) * (1.0 - fy)
            + p10 * fx * (1.0 - fy)
            + p01 * (1.0 - fx) * fy
            + p11 * fx * fy;
    }
    (channels[0], channels[1], channels[2])
}

/// Borrows each head's values out of the raw model outputs, checking the
/// expected `1x(640/stride)^2x1`/`1xNx4` f32 shapes.
fn extract_heads<'a>(
    outputs: &'a [TValue],
    positions: &[usize],
) -> Result<Vec<Head<'a>>, RedactError> {
    if positions.len() != STRIDES.len() * HEAD_KINDS.len() {
        return Err(backend(format!(
            "expected {} output positions, found {}",
            STRIDES.len() * HEAD_KINDS.len(),
            positions.len()
        )));
    }
    let mut heads = Vec::with_capacity(STRIDES.len());
    for (index, stride) in STRIDES.iter().enumerate() {
        let anchors = (INPUT_SIZE as u32 / stride).pow(2) as usize;
        let base = index * HEAD_KINDS.len();
        let output_at = |position: usize| -> Result<&'a TValue, RedactError> {
            outputs
                .get(position)
                .ok_or_else(|| backend(format!("model produced no output at position {position}")))
        };
        heads.push(Head {
            stride: *stride,
            cls: head_slice(
                output_at(positions[base])?,
                &format!("cls_{stride}"),
                &[1, anchors, 1],
            )?,
            obj: head_slice(
                output_at(positions[base + 1])?,
                &format!("obj_{stride}"),
                &[1, anchors, 1],
            )?,
            bbox: head_slice(
                output_at(positions[base + 2])?,
                &format!("bbox_{stride}"),
                &[1, anchors, 4],
            )?,
        });
    }
    Ok(heads)
}

fn head_slice<'a>(
    value: &'a TValue,
    name: &str,
    expected: &[usize],
) -> Result<&'a [f32], RedactError> {
    if value.datum_type() != DatumType::F32 {
        return Err(backend(format!(
            "model output `{name}` has type {:?}; expected f32",
            value.datum_type()
        )));
    }
    if value.shape() != expected {
        return Err(backend(format!(
            "model output `{name}` has shape {:?}; expected {expected:?}",
            value.shape()
        )));
    }
    value
        .as_slice::<f32>()
        .map_err(|error| backend(format!("model output `{name}` is not readable: {error}")))
}

/// Decodes every anchor into clipped float candidates. Non-finite outputs are
/// reported as backend errors rather than silently producing no detections.
fn decode_heads(
    heads: &[Head],
    letterbox: &Letterbox,
    min_confidence: f32,
) -> Result<Vec<Candidate>, RedactError> {
    let mut candidates = Vec::new();
    for head in heads {
        let columns = (INPUT_SIZE as u32 / head.stride) as usize;
        let anchors = columns * columns;
        if head.cls.len() != anchors || head.obj.len() != anchors {
            return Err(backend(format!(
                "cls_{0}/obj_{0} length mismatch: expected {anchors} values",
                head.stride
            )));
        }
        if head.bbox.len() != anchors * 4 {
            return Err(backend(format!(
                "bbox_{} length mismatch: expected {} values",
                head.stride,
                anchors * 4
            )));
        }
        let stride = head.stride as f32;
        let offset_x = letterbox.offset_x as f32;
        let offset_y = letterbox.offset_y as f32;
        let frame_width = letterbox.frame_width as f32;
        let frame_height = letterbox.frame_height as f32;
        for index in 0..anchors {
            let cls_score = head.cls[index];
            let obj_score = head.obj[index];
            if !cls_score.is_finite() || !obj_score.is_finite() {
                return Err(backend(format!(
                    "non-finite cls/obj output at stride {} anchor {index}",
                    head.stride
                )));
            }
            let score = (cls_score.clamp(0.0, 1.0) * obj_score.clamp(0.0, 1.0)).sqrt();
            if score < min_confidence {
                continue;
            }
            let col = index % columns;
            let row = index / columns;
            let deltas = &head.bbox[index * 4..index * 4 + 4];
            if !deltas.iter().all(|value| value.is_finite()) {
                return Err(backend(format!(
                    "non-finite bbox output at stride {} anchor {index}",
                    head.stride
                )));
            }
            let center_x = (col as f32 + deltas[0]) * stride;
            let center_y = (row as f32 + deltas[1]) * stride;
            let box_width = deltas[2].exp() * stride;
            let box_height = deltas[3].exp() * stride;
            if !center_x.is_finite() || !center_y.is_finite() {
                return Err(backend(format!(
                    "non-finite bbox center at stride {} anchor {index}",
                    head.stride
                )));
            }
            if !box_width.is_finite()
                || !box_height.is_finite()
                || box_width <= 0.0
                || box_height <= 0.0
            {
                return Err(backend(format!(
                    "invalid bbox extent at stride {} anchor {index}",
                    head.stride
                )));
            }
            let left = (center_x - box_width / 2.0 - offset_x) / letterbox.scale_x;
            let top = (center_y - box_height / 2.0 - offset_y) / letterbox.scale_y;
            let right = (center_x + box_width / 2.0 - offset_x) / letterbox.scale_x;
            let bottom = (center_y + box_height / 2.0 - offset_y) / letterbox.scale_y;
            if !left.is_finite() || !top.is_finite() || !right.is_finite() || !bottom.is_finite() {
                return Err(backend(format!(
                    "non-finite decoded box at stride {} anchor {index}",
                    head.stride
                )));
            }
            let left = left.clamp(0.0, frame_width);
            let top = top.clamp(0.0, frame_height);
            let right = right.clamp(0.0, frame_width);
            let bottom = bottom.clamp(0.0, frame_height);
            if right <= left || bottom <= top {
                continue;
            }
            candidates.push(Candidate {
                left,
                top,
                right,
                bottom,
                score,
            });
        }
    }
    Ok(candidates)
}

/// Stable-sorts candidates by descending score, keeps the top
/// [`NMS_TOP_K`], greedily suppresses `IoU` > [`NMS_IOU_THRESHOLD`] and emits
/// integer [`Detection`]s clipped to the frame.
fn finalize_candidates(mut candidates: Vec<Candidate>, letterbox: Letterbox) -> Vec<Detection> {
    candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
    candidates.truncate(NMS_TOP_K);
    let mut kept: Vec<Candidate> = Vec::new();
    'candidates: for candidate in candidates {
        for winner in &kept {
            if iou(&candidate, winner) > NMS_IOU_THRESHOLD {
                continue 'candidates;
            }
        }
        kept.push(candidate);
    }
    kept.iter()
        .filter_map(|candidate| {
            to_rect(candidate, letterbox).map(|rect| Detection {
                rect,
                confidence: candidate.score,
                label: DetectionLabel::Face,
            })
        })
        .collect()
}

fn to_rect(candidate: &Candidate, letterbox: Letterbox) -> Option<Rect> {
    let left = candidate.left.floor().max(0.0) as u32;
    let top = candidate.top.floor().max(0.0) as u32;
    let right = candidate.right.ceil().max(0.0) as u32;
    let bottom = candidate.bottom.ceil().max(0.0) as u32;
    Rect::new(left, top, right, bottom).clipped(letterbox.frame_width, letterbox.frame_height)
}

fn iou(a: &Candidate, b: &Candidate) -> f32 {
    let width = (a.right.min(b.right) - a.left.max(b.left)).max(0.0);
    let height = (a.bottom.min(b.bottom) - a.top.max(b.top)).max(0.0);
    let intersection = width * height;
    let area_a = (a.right - a.left) * (a.bottom - a.top);
    let area_b = (b.right - b.left) * (b.bottom - b.top);
    let union = area_a + area_b - intersection;
    if union <= 0.0 {
        0.0
    } else {
        intersection / union
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use video_redact_core::{DetectionLabel, Rect};

    struct HeadBuffers {
        cls: Vec<f32>,
        obj: Vec<f32>,
        bbox: Vec<f32>,
    }

    fn zero_head(stride: u32) -> HeadBuffers {
        let anchors = (INPUT_SIZE as u32 / stride).pow(2) as usize;
        HeadBuffers {
            cls: vec![0.0; anchors],
            obj: vec![0.0; anchors],
            bbox: vec![0.0; anchors * 4],
        }
    }

    fn letterbox_for(width: u32, height: u32) -> Letterbox {
        let scale =
            (INPUT_SIZE as f64 / f64::from(width)).min(INPUT_SIZE as f64 / f64::from(height));
        let new_width = (f64::from(width) * scale)
            .round()
            .clamp(1.0, INPUT_SIZE as f64) as u32;
        let new_height = (f64::from(height) * scale)
            .round()
            .clamp(1.0, INPUT_SIZE as f64) as u32;
        Letterbox {
            offset_x: (INPUT_SIZE as u32 - new_width) / 2,
            offset_y: (INPUT_SIZE as u32 - new_height) / 2,
            scale_x: new_width as f32 / width as f32,
            scale_y: new_height as f32 / height as f32,
            frame_width: width,
            frame_height: height,
        }
    }

    fn head(buffers: &HeadBuffers, stride: u32) -> Head<'_> {
        Head {
            stride,
            cls: &buffers.cls,
            obj: &buffers.obj,
            bbox: &buffers.bbox,
        }
    }

    fn run_decode(
        buffers: &[HeadBuffers; 3],
        letterbox: &Letterbox,
        min_confidence: f32,
    ) -> Result<Vec<Detection>, RedactError> {
        let heads = [
            head(&buffers[0], 8),
            head(&buffers[1], 16),
            head(&buffers[2], 32),
        ];
        let candidates = decode_heads(&heads, letterbox, min_confidence)?;
        Ok(finalize_candidates(candidates, *letterbox))
    }

    #[test]
    fn letterboxes_single_pixel_as_bgr() {
        let frame = RgbFrame::new(1, 1, vec![10, 20, 30]).unwrap();
        let (tensor, letterbox) = preprocess(&frame).unwrap();

        assert_eq!((letterbox.offset_x, letterbox.offset_y), (0, 0));
        assert_eq!(letterbox.scale_x, 640.0);
        let plane = INPUT_SIZE * INPUT_SIZE;
        for index in [0, 640 * 320 + 320, plane - 1] {
            assert!((tensor[index] - 30.0).abs() < 1e-4, "B plane at {index}");
            assert!(
                (tensor[plane + index] - 20.0).abs() < 1e-4,
                "G plane at {index}"
            );
            assert!(
                (tensor[2 * plane + index] - 10.0).abs() < 1e-4,
                "R plane at {index}"
            );
        }
    }

    #[test]
    fn letterboxes_two_by_one_with_black_padding() {
        let frame = RgbFrame::new(2, 1, vec![10, 20, 30, 10, 20, 30]).unwrap();
        let (tensor, letterbox) = preprocess(&frame).unwrap();

        assert_eq!((letterbox.offset_x, letterbox.offset_y), (0, 160));
        assert_eq!(letterbox.scale_x, 320.0);
        assert_eq!(letterbox.scale_y, 320.0);

        let plane = INPUT_SIZE * INPUT_SIZE;
        let at = |x: usize, y: usize| {
            [
                tensor[y * INPUT_SIZE + x],
                tensor[plane + y * INPUT_SIZE + x],
                tensor[2 * plane + y * INPUT_SIZE + x],
            ]
        };
        assert_eq!(at(320, 0), [0.0, 0.0, 0.0], "top padding row");
        assert_eq!(at(0, 159), [0.0, 0.0, 0.0], "last padding row");
        for (channel, expected) in [(0, 30.0), (1, 20.0), (2, 10.0)] {
            let bgr = at(0, 160);
            assert!(
                (bgr[channel] - expected).abs() < 1e-4,
                "first image row channel {channel}: {}",
                bgr[channel]
            );
            let bgr = at(639, 479);
            assert!(
                (bgr[channel] - expected).abs() < 1e-4,
                "last image row channel {channel}: {}",
                bgr[channel]
            );
        }
        assert_eq!(at(0, 480), [0.0, 0.0, 0.0], "bottom padding row");
    }

    #[test]
    fn bilinear_interpolates_between_source_pixels() {
        // First row ramps 0,10,20,30; grayscale keeps B == G == R.
        let mut data = vec![0_u8; 4 * 4 * 3];
        for x in 0..4 {
            for channel in 0..3 {
                data[x * 3 + channel] = (x * 10) as u8;
            }
        }
        let frame = RgbFrame::new(4, 4, data).unwrap();
        let (tensor, _) = preprocess(&frame).unwrap();

        // Top row: src_y = -0.5 clamps to row 0, so only horizontal taps apply.
        // dst x=319: src_x = 319.5 * 4 / 640 - 0.5 = 1.496875 -> p1/p2 mix.
        let fx = 0.496_875_f32;
        let expected = 10.0 * (1.0 - fx) + 20.0 * fx;
        let actual = tensor[319];
        assert!((actual - expected).abs() < 0.001, "{actual} vs {expected}");
        // dst x=320: src_x = 1.503125.
        let fx = 0.503_125_f32;
        let expected = 10.0 * (1.0 - fx) + 20.0 * fx;
        assert!((tensor[320] - expected).abs() < 0.001, "{}", tensor[320]);
        // Left edge clamps onto the first source pixel.
        assert_eq!(tensor[0], 0.0);
        // Right edge clamps onto the last source pixel.
        assert!((tensor[639] - 30.0).abs() < 1e-4, "{}", tensor[639]);
    }

    #[test]
    fn axis_taps_clamp_wide_coordinates() {
        assert_eq!(axis_taps(-0.5, 640), (0, 0, 0.5));
        // A u32::MAX axis clamps without i32 overflow or wraparound.
        assert_eq!(
            axis_taps(5.0e9, u32::MAX),
            (u32::MAX - 1, u32::MAX - 1, 0.0)
        );
        assert_eq!(
            axis_taps(4.0e9, u32::MAX),
            (4_000_000_000, 4_000_000_001, 0.0)
        );
        // A single-pixel axis degenerates both taps to pixel 0.
        assert_eq!(axis_taps(0.3, 1), (0, 0, 0.3));
    }

    #[test]
    fn pixel_offset_uses_wide_index_math() {
        // The last pixel of a 50000x50000 RGB24 frame exceeds i32::MAX; the
        // index math is usize only, no giant allocation needed.
        assert_eq!(pixel_offset(50000, 49_999, 49_999), 7_499_999_997);
        assert_eq!(pixel_offset(1, 0, 0), 0);
    }

    #[test]
    fn letterboxes_three_by_two_with_rounded_height() {
        let frame = RgbFrame::new(3, 2, vec![200; 3 * 2 * 3]).unwrap();
        let (tensor, letterbox) = preprocess(&frame).unwrap();

        assert_eq!((letterbox.offset_x, letterbox.offset_y), (0, 106));
        assert!((letterbox.scale_x - 640.0 / 3.0).abs() < 1e-6);
        // Post-rounding height: scale_y is 427/2, not the raw 640/3 ratio.
        assert!((letterbox.scale_y - 427.0 / 2.0).abs() < 1e-6);
        assert_eq!(tensor[105 * INPUT_SIZE + 320], 0.0, "padding row");
        assert!(
            (tensor[106 * INPUT_SIZE + 320] - 200.0).abs() < 0.5,
            "first image row: {}",
            tensor[106 * INPUT_SIZE + 320]
        );
    }

    #[test]
    fn identity_640_copies_channels() {
        let mut data = vec![0_u8; 640 * 640 * 3];
        data[0] = 1;
        data[1] = 2;
        data[2] = 3;
        let frame = RgbFrame::new(640, 640, data).unwrap();
        let (tensor, letterbox) = preprocess(&frame).unwrap();

        assert_eq!(letterbox.scale_x, 1.0);
        let plane = INPUT_SIZE * INPUT_SIZE;
        assert_eq!(tensor[0], 3.0);
        assert_eq!(tensor[plane], 2.0);
        assert_eq!(tensor[2 * plane], 1.0);
    }

    #[test]
    fn rejects_zero_sized_frame() {
        let frame = RgbFrame::new(0, 0, Vec::new()).unwrap();
        assert_eq!(
            preprocess(&frame).unwrap_err(),
            RedactError::InvalidDimensions
        );
    }

    #[test]
    fn decodes_anchor_on_square_frame() {
        let mut stride8 = zero_head(8);
        let index = 10 * 80 + 20;
        stride8.cls[index] = 0.9;
        stride8.obj[index] = 0.81;
        stride8.bbox[index * 4..index * 4 + 4].copy_from_slice(&[
            0.5,
            0.5,
            4.0_f32.ln(),
            4.0_f32.ln(),
        ]);
        let buffers = [stride8, zero_head(16), zero_head(32)];

        let detections = run_decode(&buffers, &letterbox_for(640, 640), 0.5).unwrap();

        assert_eq!(detections.len(), 1);
        assert_eq!(detections[0].rect, Rect::new(148, 68, 180, 100));
        assert!((detections[0].confidence - 0.729_f32.sqrt()).abs() < 1e-6);
        assert_eq!(detections[0].label, DetectionLabel::Face);
    }

    #[test]
    fn decodes_anchor_through_letterbox() {
        let mut stride8 = zero_head(8);
        let index = 30 * 80 + 40;
        stride8.cls[index] = 0.9;
        stride8.obj[index] = 0.81;
        stride8.bbox[index * 4..index * 4 + 4].copy_from_slice(&[
            0.5,
            0.5,
            4.0_f32.ln(),
            2.0_f32.ln(),
        ]);
        let buffers = [stride8, zero_head(16), zero_head(32)];

        let detections = run_decode(&buffers, &letterbox_for(320, 160), 0.5).unwrap();

        assert_eq!(detections.len(), 1);
        assert_eq!(detections[0].rect, Rect::new(154, 38, 170, 46));
    }

    #[test]
    fn keeps_scores_at_the_threshold() {
        let mut stride8 = zero_head(8);
        // Anchor (0,0): score exactly 0.5 -> kept.
        stride8.cls[0] = 0.25;
        stride8.obj[0] = 1.0;
        stride8.bbox[0..4].copy_from_slice(&[0.5, 0.5, 1.0_f32.ln(), 1.0_f32.ln()]);
        // Anchor (79,79): score just below -> dropped.
        let far = 79 * 80 + 79;
        stride8.cls[far] = 0.249_9;
        stride8.obj[far] = 1.0;
        stride8.bbox[far * 4..far * 4 + 4].copy_from_slice(&[0.5, 0.5, 1.0_f32.ln(), 1.0_f32.ln()]);
        let buffers = [stride8, zero_head(16), zero_head(32)];

        let detections = run_decode(&buffers, &letterbox_for(640, 640), 0.5).unwrap();

        assert_eq!(detections.len(), 1);
        assert_eq!(detections[0].confidence, 0.5);
    }

    #[test]
    fn clips_boxes_to_the_frame() {
        let mut stride8 = zero_head(8);
        // Right-edge anchor: box overflows x=640 and gets clipped.
        let edge = 79;
        stride8.cls[edge] = 0.9;
        stride8.obj[edge] = 0.9;
        stride8.bbox[edge * 4..edge * 4 + 4].copy_from_slice(&[
            2.0,
            0.5,
            8.0_f32.ln(),
            8.0_f32.ln(),
        ]);
        let buffers = [stride8, zero_head(16), zero_head(32)];

        let detections = run_decode(&buffers, &letterbox_for(640, 640), 0.5).unwrap();

        assert_eq!(detections.len(), 1);
        let rect = detections[0].rect;
        assert_eq!(rect.right, 640);
        assert!(rect.left < rect.right);
    }

    #[test]
    fn drops_candidates_outside_the_frame() {
        let mut stride8 = zero_head(8);
        // Anchor (0,79) pushed fully past the right edge -> empty after clip.
        let index = 79;
        stride8.cls[index] = 0.9;
        stride8.obj[index] = 0.9;
        stride8.bbox[index * 4..index * 4 + 4].copy_from_slice(&[
            20.0,
            0.5,
            4.0_f32.ln(),
            4.0_f32.ln(),
        ]);
        let buffers = [stride8, zero_head(16), zero_head(32)];

        let detections = run_decode(&buffers, &letterbox_for(640, 640), 0.5).unwrap();

        assert!(detections.is_empty());
    }

    #[test]
    fn nms_suppresses_the_weaker_duplicate() {
        let mut stride8 = zero_head(8);
        let box_delta = [0.5, 0.5, 8.0_f32.ln(), 8.0_f32.ln()];
        stride8.cls[0] = 0.9;
        stride8.obj[0] = 1.0;
        stride8.bbox[0..4].copy_from_slice(&box_delta);
        stride8.cls[1] = 0.8;
        stride8.obj[1] = 1.0;
        stride8.bbox[4..8].copy_from_slice(&box_delta);
        let buffers = [stride8, zero_head(16), zero_head(32)];

        let detections = run_decode(&buffers, &letterbox_for(640, 640), 0.5).unwrap();

        assert_eq!(detections.len(), 1);
        assert!((detections[0].confidence - 0.9_f32.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn nms_keeps_disjoint_boxes() {
        let mut stride8 = zero_head(8);
        let delta = [0.5, 0.5, 1.0_f32.ln(), 1.0_f32.ln()];
        stride8.cls[0] = 0.9;
        stride8.obj[0] = 1.0;
        stride8.bbox[0..4].copy_from_slice(&delta);
        let far = 79 * 80 + 79;
        stride8.cls[far] = 0.8;
        stride8.obj[far] = 1.0;
        stride8.bbox[far * 4..far * 4 + 4].copy_from_slice(&delta);
        let buffers = [stride8, zero_head(16), zero_head(32)];

        let detections = run_decode(&buffers, &letterbox_for(640, 640), 0.5).unwrap();

        assert_eq!(detections.len(), 2);
    }

    #[test]
    fn nms_truncates_to_top_k() {
        let letterbox = Letterbox {
            offset_x: 0,
            offset_y: 0,
            scale_x: 1.0,
            scale_y: 1.0,
            frame_width: 200_000,
            frame_height: 10,
        };
        let candidates: Vec<Candidate> = (0..NMS_TOP_K + 1)
            .map(|index| Candidate {
                left: index as f32 * 2.0,
                top: 0.0,
                right: index as f32 * 2.0 + 1.0,
                bottom: 1.0,
                score: 0.5 + index as f32 * 0.5 / (NMS_TOP_K + 1) as f32,
            })
            .collect();

        let detections = finalize_candidates(candidates, letterbox);

        assert_eq!(detections.len(), NMS_TOP_K);
        // The lowest score was truncated away.
        assert!(detections.iter().all(|d| d.confidence > 0.5));
    }

    #[test]
    fn rejects_bbox_center_overflow() {
        let mut stride8 = zero_head(8);
        stride8.cls[0] = 0.9;
        stride8.obj[0] = 0.9;
        // dx saturates the center coordinate to infinity once scaled.
        stride8.bbox[0..4].copy_from_slice(&[f32::MAX, 0.5, 0.0, 0.0]);
        let buffers = [stride8, zero_head(16), zero_head(32)];

        let error = run_decode(&buffers, &letterbox_for(640, 640), 0.5)
            .err()
            .unwrap();
        assert!(matches!(error, RedactError::Backend(_)), "{error}");
    }

    #[test]
    fn rejects_bbox_exponent_underflow() {
        let mut stride8 = zero_head(8);
        stride8.cls[0] = 0.9;
        stride8.obj[0] = 0.9;
        // dw = -1000 -> exp underflows to a zero-size box: an error, not a skip.
        stride8.bbox[0..4].copy_from_slice(&[0.5, 0.5, -1000.0, 0.0]);
        let buffers = [stride8, zero_head(16), zero_head(32)];

        let error = run_decode(&buffers, &letterbox_for(640, 640), 0.5)
            .err()
            .unwrap();
        assert!(matches!(error, RedactError::Backend(_)), "{error}");
    }

    #[test]
    fn rejects_inverse_transform_overflow() {
        let mut stride8 = zero_head(8);
        stride8.cls[0] = 0.9;
        stride8.obj[0] = 0.9;
        // Huge-but-finite center on a strongly downscaled frame overflows the
        // inverse transform before clipping.
        stride8.bbox[0..4].copy_from_slice(&[1.0e37, 0.5, 4.0_f32.ln(), 4.0_f32.ln()]);
        let buffers = [stride8, zero_head(16), zero_head(32)];

        let error = run_decode(&buffers, &letterbox_for(u32::MAX, u32::MAX), 0.5)
            .err()
            .unwrap();
        assert!(matches!(error, RedactError::Backend(_)), "{error}");
    }

    #[test]
    fn maps_head_outputs_by_name_in_any_order() {
        // Heads arrive per stride in reverse order, kps interleaved.
        let mut labels = Vec::new();
        for stride in [32, 16, 8] {
            for kind in ["bbox", "obj", "cls", "kps"] {
                labels.push(Some(format!("{kind}_{stride}")));
            }
        }
        let positions = map_head_outputs(&labels).unwrap();

        assert_eq!(positions, vec![10, 9, 8, 6, 5, 4, 2, 1, 0]);
    }

    #[test]
    fn reports_missing_head_outputs() {
        let mut labels = Vec::new();
        for stride in [8, 16, 32] {
            for kind in ["cls", "obj", "bbox", "kps"] {
                labels.push(Some(format!("{kind}_{stride}")));
            }
        }
        labels.retain(|label| label.as_deref() != Some("bbox_8"));

        let error = map_head_outputs(&labels).unwrap_err();
        assert!(error.to_string().contains("bbox_8"), "{error}");
    }

    #[test]
    fn validates_head_facts() {
        let expected = [1, 6400, 1];
        assert!(check_head_fact("cls_8", DatumType::F32, Some(&expected), &expected).is_ok());
        assert!(check_head_fact("cls_8", DatumType::I32, Some(&expected), &expected).is_err());
        assert!(check_head_fact("cls_8", DatumType::F32, Some(&[1, 6400]), &expected).is_err());
        assert!(check_head_fact("cls_8", DatumType::F32, None, &expected).is_err());
    }

    #[test]
    fn extract_heads_checks_output_positions() {
        let outputs = || -> Vec<TValue> {
            let mut values = Vec::new();
            for stride in STRIDES {
                let anchors = (INPUT_SIZE as u32 / stride).pow(2) as usize;
                values.push(
                    Tensor::from_shape(&[1, anchors, 1], &vec![0.0_f32; anchors])
                        .unwrap()
                        .into(),
                );
                values.push(
                    Tensor::from_shape(&[1, anchors, 1], &vec![0.0_f32; anchors])
                        .unwrap()
                        .into(),
                );
                values.push(
                    Tensor::from_shape(&[1, anchors, 4], &vec![0.0_f32; anchors * 4])
                        .unwrap()
                        .into(),
                );
            }
            values
        };
        let positions: Vec<usize> = (0..9).collect();
        assert!(extract_heads(&outputs(), &positions).is_ok());

        // A position past the output list is a backend error, not a panic.
        let mut out_of_range = positions.clone();
        out_of_range[8] = 99;
        let error = extract_heads(&outputs(), &out_of_range).err().unwrap();
        assert!(matches!(error, RedactError::Backend(_)), "{error}");

        // Wrong dtype and wrong shape are reported by name.
        let mut wrong_type = outputs();
        wrong_type[0] = Tensor::from_shape(&[1, 6400, 1], &vec![0_i32; 6400])
            .unwrap()
            .into();
        let error = extract_heads(&wrong_type, &positions).err().unwrap();
        assert!(error.to_string().contains("cls_8"), "{error}");

        let mut wrong_shape = outputs();
        wrong_shape[2] = Tensor::from_shape(&[1, 6400, 5], &vec![0.0_f32; 6400 * 5])
            .unwrap()
            .into();
        let error = extract_heads(&wrong_shape, &positions).err().unwrap();
        assert!(error.to_string().contains("bbox_8"), "{error}");
    }

    #[test]
    fn rejects_wrong_head_lengths() {
        let mut stride8 = zero_head(8);
        stride8.cls.pop();
        let buffers = [stride8, zero_head(16), zero_head(32)];
        let error = run_decode(&buffers, &letterbox_for(640, 640), 0.5).unwrap_err();
        assert!(error.to_string().contains("length"), "{error}");

        let mut stride8 = zero_head(8);
        stride8.bbox.pop();
        let buffers = [stride8, zero_head(16), zero_head(32)];
        assert!(run_decode(&buffers, &letterbox_for(640, 640), 0.5).is_err());
    }

    #[test]
    fn rejects_non_finite_scores() {
        let mut stride8 = zero_head(8);
        stride8.cls[0] = f32::NAN;
        let buffers = [stride8, zero_head(16), zero_head(32)];
        let error = run_decode(&buffers, &letterbox_for(640, 640), 0.5).unwrap_err();
        assert!(matches!(error, RedactError::Backend(_)), "{error}");
    }

    #[test]
    fn rejects_non_finite_bbox_when_accepted() {
        let mut stride8 = zero_head(8);
        stride8.cls[0] = 0.9;
        stride8.obj[0] = 0.9;
        stride8.bbox[0] = f32::NAN;
        // Below-threshold anchors with NaN boxes are ignored, not errors.
        stride8.bbox[100 * 4] = f32::NAN;
        let buffers = [stride8, zero_head(16), zero_head(32)];

        let error = run_decode(&buffers, &letterbox_for(640, 640), 0.5).unwrap_err();
        assert!(matches!(error, RedactError::Backend(_)), "{error}");
    }

    #[test]
    fn rejects_bbox_exponent_overflow() {
        let mut stride8 = zero_head(8);
        stride8.cls[0] = 0.9;
        stride8.obj[0] = 0.9;
        stride8.bbox[0..4].copy_from_slice(&[0.5, 0.5, 800.0, 0.0]);
        let buffers = [stride8, zero_head(16), zero_head(32)];

        let error = run_decode(&buffers, &letterbox_for(640, 640), 0.5).unwrap_err();
        assert!(matches!(error, RedactError::Backend(_)), "{error}");
    }

    #[test]
    fn rejects_invalid_confidence() {
        for confidence in [f32::NAN, 1.5, -0.1, f32::INFINITY] {
            assert_eq!(
                YuNetDetector::new("model.onnx", confidence).unwrap_err(),
                RedactError::InvalidConfidence
            );
        }
    }

    #[test]
    fn rejects_missing_model_file() {
        let error = YuNetDetector::new("/nonexistent/yunet.onnx", 0.5).unwrap_err();
        assert!(matches!(error, RedactError::Backend(_)), "{error}");
    }

    #[test]
    fn rejects_invalid_model_bytes() {
        let path = std::env::temp_dir().join(format!(
            "video-redact-invalid-model-{}-{:?}.onnx",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        // Exclusive creation so a stale file can never be silently overwritten.
        let mut file = std::fs::File::create_new(&path)
            .unwrap_or_else(|error| panic!("cannot create {path:?} exclusively: {error}"));
        std::io::Write::write_all(&mut file, b"not an onnx model").unwrap();
        drop(file);
        let error = YuNetDetector::new(&path, 0.5).unwrap_err();
        let _ = std::fs::remove_file(&path);
        assert!(matches!(error, RedactError::Backend(_)), "{error}");
    }
}
