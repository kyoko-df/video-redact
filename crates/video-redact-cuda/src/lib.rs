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
            ensure_dynamic_library("cuda")?;
            ensure_dynamic_library("nvrtc")?;
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
            let mut device_frame = stream.clone_htod(frame.data()).map_err(backend_error)?;

            for region in regions
                .iter()
                .filter_map(|region| region.clipped(width, height))
            {
                let region_width = region.right - region.left;
                let region_height = region.bottom - region.top;
                let block_columns = (region_width - 1) / block_size + 1;
                let block_rows = (region_height - 1) / block_size + 1;
                let block_count = block_columns.saturating_mul(block_rows);
                let mut launch = stream.launch_builder(&self.function);
                launch
                    .arg(&mut device_frame)
                    .arg(&stride)
                    .arg(&region.left)
                    .arg(&region.top)
                    .arg(&region.right)
                    .arg(&region.bottom)
                    .arg(&block_size);

                // SAFETY: the kernel argument order and scalar widths match mosaic.cu;
                // the frame covers `height * stride` bytes, and the region is clipped.
                unsafe { launch.launch(LaunchConfig::for_num_elems(block_count)) }
                    .map_err(backend_error)?;
            }

            let redacted = stream.clone_dtoh(&device_frame).map_err(backend_error)?;
            frame.data_mut().copy_from_slice(&redacted);
            Ok(())
        }
    }

    fn ensure_dynamic_library(name: &str) -> Result<(), RedactError> {
        let candidates = cudarc::get_lib_name_candidates(name);
        let found = candidates.iter().any(|candidate| {
            // SAFETY: this only asks the platform loader to open a CUDA library using
            // the exact candidate names that cudarc will try immediately afterwards.
            unsafe { libloading::Library::new(candidate) }.is_ok()
        });
        if found {
            Ok(())
        } else {
            let requirement = match name {
                "cuda" => "install a compatible NVIDIA driver",
                "nvrtc" => "install the CUDA toolkit with NVRTC",
                _ => "install the required CUDA component",
            };
            Err(RedactError::Backend(format!(
                "CUDA {name} shared library was not found; {requirement}"
            )))
        }
    }

    fn backend_error(error: impl std::fmt::Display) -> RedactError {
        RedactError::Backend(error.to_string())
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use video_redact_core::CpuRedactor;

        #[test]
        fn cuda_mosaic_matches_cpu_reference() {
            let mut cuda = match CudaRedactor::new(0) {
                Ok(cuda) => cuda,
                Err(error) if std::env::var_os("VIDEO_REDACT_REQUIRE_CUDA").is_none() => {
                    eprintln!("skipping CUDA parity test: {error}");
                    return;
                }
                Err(error) => panic!("CUDA is required for this test: {error}"),
            };
            let pixels = (0_u8..=u8::MAX).cycle().take(9 * 7 * 3).collect();
            let mut expected = RgbFrame::new(9, 7, pixels).unwrap();
            let mut actual = RgbFrame::new(9, 7, expected.data().to_vec()).unwrap();
            let regions = [Rect::new(1, 1, 8, 6), Rect::new(5, 3, 12, 9)];
            let effect = RedactionEffect::Mosaic { block_size: 3 };

            CpuRedactor.redact(&mut expected, &regions, effect).unwrap();
            cuda.redact(&mut actual, &regions, effect).unwrap();

            assert_eq!(actual.data(), expected.data());
        }
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
