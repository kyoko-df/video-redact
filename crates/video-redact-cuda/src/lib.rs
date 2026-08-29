//! CUDA implementation of the redaction backend.

#[cfg(feature = "cuda")]
mod enabled {
    use std::sync::Arc;

    use cudarc::driver::{CudaContext, CudaFunction, LaunchConfig, PushKernelArg};
    use cudarc::nvrtc::compile_ptx;
    use video_redact_core::{Rect, RedactError, RedactionEffect, Redactor, RgbFrame};

    const MOSAIC_KERNEL: &str = include_str!("mosaic.cu");

    #[derive(Debug)]
    pub struct CudaRedactor {
        context: Arc<CudaContext>,
        function: CudaFunction,
    }

    impl CudaRedactor {
        /// Creates a redactor on the selected CUDA device.
        ///
        /// # Errors
        ///
        /// Returns a backend error when the CUDA driver, NVRTC compiler, device,
        /// module, or kernel cannot be initialized.
        pub fn new(device_ordinal: usize) -> Result<Self, RedactError> {
            let context = CudaContext::new(device_ordinal).map_err(backend_error)?;
            let ptx = compile_ptx(MOSAIC_KERNEL).map_err(backend_error)?;
            let module = context.load_module(ptx).map_err(backend_error)?;
            let function = module
                .load_function("mosaic_rgb24")
                .map_err(backend_error)?;
            Ok(Self { context, function })
        }
    }

    impl Redactor for CudaRedactor {
        fn redact(
            &mut self,
            frame: &mut RgbFrame,
            regions: &[Rect],
            effect: RedactionEffect,
        ) -> Result<(), RedactError> {
            let RedactionEffect::Mosaic { block_size } = effect;
            if block_size == 0 {
                return Err(RedactError::InvalidBlockSize);
            }

            let width = frame.width();
            let height = frame.height();
            let stride = u32::try_from(frame.stride())
                .map_err(|_| RedactError::Backend("frame stride exceeds CUDA ABI".into()))?;
            let stream = self.context.default_stream();
            let input = stream.clone_htod(frame.data()).map_err(backend_error)?;
            let mut output = stream.clone_htod(frame.data()).map_err(backend_error)?;

            for region in regions
                .iter()
                .filter_map(|region| region.clipped(width, height))
            {
                let pixel_count = region
                    .right
                    .saturating_sub(region.left)
                    .saturating_mul(region.bottom.saturating_sub(region.top));
                let mut launch = stream.launch_builder(&self.function);
                launch
                    .arg(&input)
                    .arg(&mut output)
                    .arg(&width)
                    .arg(&height)
                    .arg(&stride)
                    .arg(&region.left)
                    .arg(&region.top)
                    .arg(&region.right)
                    .arg(&region.bottom)
                    .arg(&block_size);

                // SAFETY: the kernel argument order and scalar widths match mosaic.cu;
                // input/output each cover `height * stride` bytes, and the region is clipped.
                unsafe { launch.launch(LaunchConfig::for_num_elems(pixel_count)) }
                    .map_err(backend_error)?;
            }

            let redacted = stream.clone_dtoh(&output).map_err(backend_error)?;
            frame.data_mut().copy_from_slice(&redacted);
            Ok(())
        }
    }

    fn backend_error(error: impl std::fmt::Display) -> RedactError {
        RedactError::Backend(error.to_string())
    }
}

#[cfg(feature = "cuda")]
pub use enabled::CudaRedactor;

#[cfg(not(feature = "cuda"))]
#[derive(Debug)]
pub struct CudaRedactor;

#[cfg(not(feature = "cuda"))]
impl CudaRedactor {
    /// Reports that this crate was built without its `cuda` feature.
    ///
    /// # Errors
    ///
    /// Always returns an error because no CUDA implementation is present.
    pub fn new(_device_ordinal: usize) -> Result<Self, video_redact_core::RedactError> {
        Err(video_redact_core::RedactError::Backend(
            "CUDA support is disabled; rebuild with `--features cuda`".into(),
        ))
    }
}
