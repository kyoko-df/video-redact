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

#[derive(Debug, Eq, PartialEq)]
pub enum RedactError {
    InvalidDimensions,
    InvalidBufferLength { expected: usize, actual: usize },
    InvalidBlockSize,
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
}
