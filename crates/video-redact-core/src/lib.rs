//! Domain types and backend-independent redaction logic.

use std::error::Error;
use std::fmt::{Display, Formatter};

/// A half-open rectangle: `[left, right) x [top, bottom)`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Rect {
    pub left: u32,
    pub top: u32,
    pub right: u32,
    pub bottom: u32,
}

impl Rect {
    #[must_use]
    pub const fn new(left: u32, top: u32, right: u32, bottom: u32) -> Self {
        Self {
            left,
            top,
            right,
            bottom,
        }
    }

    #[must_use]
    pub fn clipped(self, width: u32, height: u32) -> Option<Self> {
        let clipped = Self {
            left: self.left.min(width),
            top: self.top.min(height),
            right: self.right.min(width),
            bottom: self.bottom.min(height),
        };
        (clipped.left < clipped.right && clipped.top < clipped.bottom).then_some(clipped)
    }

    /// Returns the rect expanded by `padding` pixels on every side.
    #[must_use]
    pub const fn padded(self, padding: u32) -> Self {
        Self {
            left: self.left.saturating_sub(padding),
            top: self.top.saturating_sub(padding),
            right: self.right.saturating_add(padding),
            bottom: self.bottom.saturating_add(padding),
        }
    }
}

#[derive(Debug)]
pub struct RgbFrame {
    width: u32,
    height: u32,
    stride: usize,
    data: Vec<u8>,
}

impl RgbFrame {
    /// Creates a tightly packed RGB24 frame.
    ///
    /// # Errors
    ///
    /// Returns [`RedactError::InvalidDimensions`] when the dimensions overflow,
    /// or [`RedactError::InvalidBufferLength`] when `data` is not exactly
    /// `width * height * 3` bytes.
    pub fn new(width: u32, height: u32, data: Vec<u8>) -> Result<Self, RedactError> {
        let stride = usize::try_from(width)
            .ok()
            .and_then(|value| value.checked_mul(3))
            .ok_or(RedactError::InvalidDimensions)?;
        let expected = stride
            .checked_mul(usize::try_from(height).map_err(|_| RedactError::InvalidDimensions)?)
            .ok_or(RedactError::InvalidDimensions)?;

        if data.len() != expected {
            return Err(RedactError::InvalidBufferLength {
                expected,
                actual: data.len(),
            });
        }

        Ok(Self {
            width,
            height,
            stride,
            data,
        })
    }

    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    #[must_use]
    pub const fn stride(&self) -> usize {
        self.stride
    }

    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    #[must_use]
    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Returns the owned tightly packed RGB24 buffer.
    #[must_use]
    pub fn into_data(self) -> Vec<u8> {
        self.data
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RedactionEffect {
    Mosaic { block_size: u32 },
}

pub trait Redactor {
    /// Applies `effect` to every supplied region.
    ///
    /// # Errors
    ///
    /// Returns a validation error for an invalid effect, or a backend error when
    /// processing cannot be completed.
    fn redact(
        &mut self,
        frame: &mut RgbFrame,
        regions: &[Rect],
        effect: RedactionEffect,
    ) -> Result<(), RedactError>;
}

#[derive(Debug, Default)]
pub struct CpuRedactor;

impl Redactor for CpuRedactor {
    fn redact(
        &mut self,
        frame: &mut RgbFrame,
        regions: &[Rect],
        effect: RedactionEffect,
    ) -> Result<(), RedactError> {
        match effect {
            RedactionEffect::Mosaic { block_size: 0 } => Err(RedactError::InvalidBlockSize),
            RedactionEffect::Mosaic { block_size } => {
                let width = frame.width;
                let height = frame.height;
                for region in regions
                    .iter()
                    .filter_map(|region| region.clipped(width, height))
                {
                    mosaic_region(frame, region, block_size);
                }
                Ok(())
            }
        }
    }
}

fn mosaic_region(frame: &mut RgbFrame, region: Rect, block_size: u32) {
    for block_top in (region.top..region.bottom).step_by(block_size as usize) {
        for block_left in (region.left..region.right).step_by(block_size as usize) {
            let block_right = block_left.saturating_add(block_size).min(region.right);
            let block_bottom = block_top.saturating_add(block_size).min(region.bottom);
            let mut sum = [0_u64; 3];
            let mut count = 0_u64;

            for y in block_top..block_bottom {
                for x in block_left..block_right {
                    let offset = y as usize * frame.stride + x as usize * 3;
                    for (channel, value) in sum.iter_mut().zip(&frame.data[offset..offset + 3]) {
                        *channel += u64::from(*value);
                    }
                    count += 1;
                }
            }

            let color = sum.map(|value| u8::try_from(value / count).unwrap_or(u8::MAX));
            for y in block_top..block_bottom {
                for x in block_left..block_right {
                    let offset = y as usize * frame.stride + x as usize * 3;
                    frame.data[offset..offset + 3].copy_from_slice(&color);
                }
            }
        }
    }
}

/// A region proposed for redaction, with the detector's confidence.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Detection {
    pub rect: Rect,
    /// Detector confidence in `[0.0, 1.0]`. Values outside the range,
    /// including `NaN`, are treated as below every threshold.
    pub confidence: f32,
    pub label: DetectionLabel,
}

/// What a detected region is believed to contain.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DetectionLabel {
    Face,
    LicensePlate,
    /// A region supplied manually, e.g. via `--roi`.
    Manual,
    /// A detector class without a dedicated variant yet.
    Other,
}

/// Produces detections for a single decoded frame.
///
/// Detectors are fed frames in order; `frame_index` is the zero-based
/// position of `frame` in the stream.
pub trait Detector {
    /// Returns the detections found in `frame`.
    ///
    /// # Errors
    ///
    /// Returns [`RedactError::Backend`] when detection cannot be completed.
    fn detect(&mut self, frame: &RgbFrame, frame_index: u64)
    -> Result<Vec<Detection>, RedactError>;
}

/// A [`Detector`] that reports the same regions on every frame.
///
/// Backs the CLI `--roi` flag until learned detectors land.
#[derive(Clone, Debug)]
pub struct StaticDetector {
    detections: Vec<Detection>,
}

impl StaticDetector {
    /// Wraps `regions` as full-confidence [`DetectionLabel::Manual`] detections.
    #[must_use]
    pub fn new(regions: Vec<Rect>) -> Self {
        Self {
            detections: regions
                .into_iter()
                .map(|rect| Detection {
                    rect,
                    confidence: 1.0,
                    label: DetectionLabel::Manual,
                })
                .collect(),
        }
    }
}

impl Detector for StaticDetector {
    fn detect(
        &mut self,
        _frame: &RgbFrame,
        _frame_index: u64,
    ) -> Result<Vec<Detection>, RedactError> {
        Ok(self.detections.clone())
    }
}

/// A detection that fell below the redaction threshold.
///
/// Records are kept so a human can review whether the region should have
/// been redacted after all.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReviewRecord {
    /// Zero-based index of the frame the detection came from.
    pub frame_index: u64,
    pub detection: Detection,
}

/// Turns detector output into regions to redact.
///
/// Detections with confidence at or above the configured threshold are
/// padded and redacted; weaker detections are appended to the review log
/// instead of being silently dropped.
#[derive(Clone, Debug)]
pub struct RedactionPolicy {
    min_confidence: f32,
    padding: u32,
    review_log: Vec<ReviewRecord>,
}

impl RedactionPolicy {
    /// Creates a policy.
    ///
    /// `min_confidence` is the confidence required to redact a detection, and
    /// `padding` expands each accepted region by that many pixels on every
    /// side before it is clipped to the frame.
    ///
    /// # Errors
    ///
    /// Returns [`RedactError::InvalidConfidence`] when `min_confidence` is
    /// outside `[0.0, 1.0]`.
    pub fn new(min_confidence: f32, padding: u32) -> Result<Self, RedactError> {
        if !(0.0..=1.0).contains(&min_confidence) {
            return Err(RedactError::InvalidConfidence);
        }
        Ok(Self {
            min_confidence,
            padding,
            review_log: Vec::new(),
        })
    }

    /// Resolves `detections` into the regions to redact on frame
    /// `frame_index` of a `frame_width` × `frame_height` video.
    pub fn resolve(
        &mut self,
        detections: Vec<Detection>,
        frame_width: u32,
        frame_height: u32,
        frame_index: u64,
    ) -> Vec<Rect> {
        let mut regions = Vec::new();
        for detection in detections {
            if detection.confidence >= self.min_confidence {
                if let Some(region) = detection
                    .rect
                    .padded(self.padding)
                    .clipped(frame_width, frame_height)
                {
                    regions.push(region);
                }
            } else {
                self.review_log.push(ReviewRecord {
                    frame_index,
                    detection,
                });
            }
        }
        regions
    }

    /// Detections withheld from redaction for low confidence, oldest first.
    #[must_use]
    pub fn review_log(&self) -> &[ReviewRecord] {
        &self.review_log
    }
}

#[derive(Debug, Eq, PartialEq)]
pub enum RedactError {
    InvalidDimensions,
    InvalidBufferLength { expected: usize, actual: usize },
    InvalidBlockSize,
    InvalidConfidence,
    Backend(String),
}

impl Display for RedactError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidDimensions => formatter.write_str("invalid frame dimensions"),
            Self::InvalidBufferLength { expected, actual } => {
                write!(
                    formatter,
                    "invalid buffer length: expected {expected}, got {actual}"
                )
            }
            Self::InvalidBlockSize => {
                formatter.write_str("mosaic block size must be greater than zero")
            }
            Self::InvalidConfidence => {
                formatter.write_str("confidence threshold must be within [0.0, 1.0]")
            }
            Self::Backend(message) => write!(formatter, "backend error: {message}"),
        }
    }
}

impl Error for RedactError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_mismatched_buffer() {
        let error = RgbFrame::new(2, 2, vec![0; 11]).unwrap_err();
        assert_eq!(
            error,
            RedactError::InvalidBufferLength {
                expected: 12,
                actual: 11
            }
        );
    }

    #[test]
    fn mosaic_changes_only_the_region() {
        let original: Vec<u8> = (0..48).collect();
        let mut frame = RgbFrame::new(4, 4, original.clone()).unwrap();
        CpuRedactor
            .redact(
                &mut frame,
                &[Rect::new(1, 1, 3, 3)],
                RedactionEffect::Mosaic { block_size: 2 },
            )
            .unwrap();

        assert_eq!(&frame.data()[0..15], &original[0..15]);
        assert_eq!(&frame.data()[15..18], &[22, 23, 24]);
        assert_eq!(&frame.data()[18..21], &[22, 23, 24]);
        assert_eq!(&frame.data()[27..30], &[22, 23, 24]);
        assert_eq!(&frame.data()[30..33], &[22, 23, 24]);
        assert_eq!(&frame.data()[33..], &original[33..]);
    }

    #[test]
    fn clips_regions_to_the_frame() {
        assert_eq!(
            Rect::new(2, 2, 10, 10).clipped(4, 5),
            Some(Rect::new(2, 2, 4, 5))
        );
        assert_eq!(Rect::new(8, 8, 10, 10).clipped(4, 5), None);
    }

    #[test]
    fn padded_rect_saturates_at_the_edges() {
        assert_eq!(Rect::new(2, 3, 10, 11).padded(5), Rect::new(0, 0, 15, 16));
        assert_eq!(Rect::new(0, 0, 4, 4).padded(u32::MAX).right, u32::MAX);
    }

    #[test]
    fn policy_redacts_confident_detections_with_padding() {
        let mut policy = RedactionPolicy::new(0.5, 2).unwrap();
        let regions = policy.resolve(
            vec![Detection {
                rect: Rect::new(4, 4, 9, 9),
                confidence: 0.9,
                label: DetectionLabel::Face,
            }],
            10,
            10,
            0,
        );

        assert_eq!(regions, vec![Rect::new(2, 2, 10, 10)]);
        assert!(policy.review_log().is_empty());
    }

    #[test]
    fn policy_records_low_confidence_detections_for_review() {
        let mut policy = RedactionPolicy::new(0.5, 0).unwrap();
        let detection = Detection {
            rect: Rect::new(1, 1, 3, 3),
            confidence: 0.4,
            label: DetectionLabel::LicensePlate,
        };
        let regions = policy.resolve(vec![detection], 10, 10, 7);

        assert!(regions.is_empty());
        assert_eq!(
            policy.review_log(),
            &[ReviewRecord {
                frame_index: 7,
                detection,
            }]
        );
    }

    #[test]
    fn policy_rejects_out_of_range_confidence() {
        assert_eq!(
            RedactionPolicy::new(1.5, 0).unwrap_err(),
            RedactError::InvalidConfidence
        );
        assert_eq!(
            RedactionPolicy::new(f32::NAN, 0).unwrap_err(),
            RedactError::InvalidConfidence
        );
    }

    #[test]
    fn static_detector_repeats_its_regions() {
        let frame = RgbFrame::new(2, 2, vec![0; 12]).unwrap();
        let mut detector = StaticDetector::new(vec![Rect::new(0, 0, 1, 1)]);
        let detections = detector.detect(&frame, 0).unwrap();

        assert_eq!(
            detections,
            vec![Detection {
                rect: Rect::new(0, 0, 1, 1),
                confidence: 1.0,
                label: DetectionLabel::Manual,
            }]
        );
    }
}
