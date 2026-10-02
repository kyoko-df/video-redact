//! ONNX Runtime + `CoreML` execution-provider engine for `YuNet` (aarch64
//! macOS only, behind the `coreml` feature).
//!
//! The session uses the frozen `CoreML` configuration: `MLProgram` format,
//! `CPUAndGPU` compute units, static input shapes, no low-precision GPU
//! accumulation, registration `error_on_failure`, graph optimization level 3
//! and ORT CPU fallback disabled, so a graph `CoreML` cannot take fails the
//! session instead of silently running on CPU. Letterboxing, decoding, scoring
//! and `NMS` stay in [`crate`] and are shared with the `tract` CPU engine.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use ort::ep::CoreML;
use ort::ep::coreml::{ComputeUnits, ModelFormat};
use ort::logging::LogLevel;
use ort::session::{Session, SessionOutputs, builder::GraphOptimizationLevel};
use ort::value::{Tensor, TensorElementType, ValueType};
use video_redact_core::RedactError;

use crate::{CoreMlOptions, Head, INPUT_SIZE, STRIDES, backend};

/// ONNX Runtime is initialized process-globally and a second `init_from` with
/// a different library is silently ignored by `ort`; the registry records the
/// first successfully loaded runtime so a different path is a hard error.
static ORT_RUNTIME: Mutex<Option<PathBuf>> = Mutex::new(None);

/// CoreML-backed inference engine. Wraps the prepared ORT session.
pub(crate) struct CoreMlEngine {
    session: Session,
    /// `true` until ORT profiling has been ended exactly once.
    profiling: bool,
}

impl CoreMlEngine {
    /// Runs one inference over the letterboxed BGR NCHW `[1,3,640,640]` input,
    /// taking ownership so the shared preprocessing buffer is not re-copied.
    pub(crate) fn infer(&mut self, input: Vec<f32>) -> Result<SessionOutputs<'_>, RedactError> {
        let tensor = Tensor::from_array(([1usize, 3, INPUT_SIZE, INPUT_SIZE], input))
            .map_err(|error| backend(format!("failed to build model input: {error}")))?;
        self.session
            .run(ort::inputs![tensor])
            .map_err(|error| backend(format!("YuNet CoreML inference failed: {error}")))
    }

    /// Ends ORT profiling once and returns the written profile path.
    /// Returns `Ok(None)` when profiling was not enabled or already ended.
    pub(crate) fn end_profiling(&mut self) -> Result<Option<PathBuf>, RedactError> {
        if !self.profiling {
            return Ok(None);
        }
        let file = self
            .session
            .end_profiling()
            .map_err(|error| backend(format!("failed to end ORT profiling: {error}")))?;
        self.profiling = false;
        Ok(Some(PathBuf::from(file)))
    }
}

/// Builds the `CoreML` engine: checks files, loads the process-global ONNX
/// Runtime, configures the session and validates the `YuNet` I/O contract.
pub(crate) fn build_engine(
    model_path: &Path,
    options: &CoreMlOptions,
) -> Result<CoreMlEngine, RedactError> {
    let model = canonical_file(model_path, "YuNet model")?;
    let runtime = canonical_file(&options.runtime_library, "ONNX Runtime library")?;
    ensure_runtime_loaded(&runtime)?;

    if options.verbose {
        // Global verbose is required for the CoreML compute-plan log.
        ort::environment::Environment::current()
            .map_err(|error| backend(format!("ORT environment unavailable: {error}")))?
            .set_log_level(LogLevel::Verbose);
    }
    let log_level = if options.verbose {
        LogLevel::Verbose
    } else {
        LogLevel::Warning
    };

    let builder = Session::builder()
        .map_err(|error| backend(format!("failed to create ORT session builder: {error}")))?;
    let builder = builder
        .with_log_level(log_level)
        .map_err(|error| backend(format!("failed to set log level: {error}")))?;
    let builder = builder
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|error| backend(format!("failed to set optimization level: {error}")))?;
    let builder = builder
        .with_execution_providers([CoreML::default()
            .with_model_format(ModelFormat::MLProgram)
            .with_compute_units(ComputeUnits::CPUAndGPU)
            .with_static_input_shapes(true)
            .with_low_precision_accumulation_on_gpu(false)
            .with_profile_compute_plan(options.verbose)
            .build()
            .error_on_failure()])
        .map_err(|error| backend(format!("failed to register CoreML EP: {error}")))?;
    let builder = builder
        .with_disable_cpu_fallback()
        .map_err(|error| backend(format!("failed to disable CPU fallback: {error}")))?;
    let profiling = options.profile_prefix.is_some();
    let mut builder = match &options.profile_prefix {
        Some(prefix) => builder
            .with_profiling(prefix)
            .map_err(|error| backend(format!("failed to enable profiling: {error}")))?,
        None => builder,
    };
    let session = builder.commit_from_file(&model).map_err(|error| {
        backend(format!(
            "failed to create CoreML session for {}: {error}",
            model.display()
        ))
    })?;
    validate_coreml_session(&session)?;
    Ok(CoreMlEngine { session, profiling })
}

/// Canonicalizes `path` and requires it to be a regular file, so missing
/// models/runtimes fail before any ORT call.
fn canonical_file(path: &Path, what: &str) -> Result<PathBuf, RedactError> {
    let canonical = std::fs::canonicalize(path).map_err(|error| {
        backend(format!(
            "{what} `{}` is not accessible: {error}",
            path.display()
        ))
    })?;
    if !canonical.is_file() {
        return Err(backend(format!(
            "{what} `{}` is not a file",
            canonical.display()
        )));
    }
    Ok(canonical)
}

/// Checks the process-global runtime registry without loading anything.
/// Returns `true` when the same canonical path is already loaded, `false`
/// when nothing is loaded, and an error for a different path.
fn check_runtime_claim(loaded: Option<&Path>, requested: &Path) -> Result<bool, RedactError> {
    match loaded {
        Some(path) if path == requested => Ok(true),
        Some(path) => Err(backend(format!(
            "ONNX Runtime is already loaded from `{}`; cannot switch to `{}`",
            path.display(),
            requested.display()
        ))),
        None => Ok(false),
    }
}

/// Pre-validates the runtime with `libloading` before `ort::init_from`:
/// ort rc.12's load-error path constructs `Error::new` inside its global
/// `G_ORT_LIB` `OnceLock`, which re-enters the same lock through
/// `crate::api()` and deadlocks — so an unloadable or incompatible library
/// must be rejected here, where the error is safe.
fn validate_runtime_library(runtime: &Path) -> Result<(), RedactError> {
    // SAFETY: this loads a native library the user explicitly pointed at via
    // `--ort-library`/`VIDEO_REDACT_ORT_DYLIB`; only a trusted official ONNX
    // Runtime build may ever be passed. `library` stays alive for the rest of
    // the function, so every pointer obtained from it remains valid to use.
    let library = unsafe { libloading::Library::new(runtime) }.map_err(|error| {
        backend(format!(
            "ONNX Runtime `{}` is not loadable: {error}",
            runtime.display()
        ))
    })?;
    // SAFETY: `OrtGetApiBase` is the stable entry point of ONNX Runtime; the
    // returned `OrtApiBase` pointer is backed by `library`, which outlives the
    // validation call below.
    let getter: libloading::Symbol<unsafe extern "C" fn() -> *const ort::sys::OrtApiBase> =
        unsafe { library.get(b"OrtGetApiBase") }.map_err(|error| {
            backend(format!(
                "ONNX Runtime `{}` does not export `OrtGetApiBase`: {error}",
                runtime.display()
            ))
        })?;
    // SAFETY: `base` is backed by `library`, still loaded in this scope.
    unsafe { validate_api_base(getter(), runtime) }
}

/// Checks the `OrtApiBase` a runtime returned before `ort::init_from` runs:
/// non-null version string and `GetApi(ORT_API_VERSION)` result, plus a
/// supported `1.x` version.
///
/// # Safety
/// `base` must be a valid `OrtApiBase` pointer (or null) whose backing
/// library stays loaded for the duration of this call. Tests pass a pointer
/// to a local struct, which satisfies that.
unsafe fn validate_api_base(
    base: *const ort::sys::OrtApiBase,
    runtime: &Path,
) -> Result<(), RedactError> {
    if base.is_null() {
        return Err(backend(format!(
            "ONNX Runtime `{}` returned a null API base",
            runtime.display()
        )));
    }
    // SAFETY: `base` is valid per the caller; `GetVersionString` follows the
    // ORT C contract of returning a null-terminated UTF-8 string or null.
    let raw_version = unsafe { ((*base).GetVersionString)() };
    if raw_version.is_null() {
        return Err(backend(format!(
            "ONNX Runtime `{}` returned a null version string",
            runtime.display()
        )));
    }
    // SAFETY: `raw_version` is non-null and the runtime contract guarantees a
    // null-terminated string valid for the loaded library's lifetime.
    let version = unsafe { core::ffi::CStr::from_ptr(raw_version) }
        .to_string_lossy()
        .into_owned();
    check_runtime_version(&version).map_err(|error| {
        backend(format!(
            "ONNX Runtime `{}` is version '{version}': {error}",
            runtime.display()
        ))
    })?;
    // SAFETY: `base` is valid per the caller; `GetApi` follows the ORT C
    // contract of returning the api table or null for unsupported versions.
    if unsafe { ((*base).GetApi)(ort::sys::ORT_API_VERSION) }.is_null() {
        return Err(backend(format!(
            "ONNX Runtime `{}` does not provide API version {}",
            runtime.display(),
            ort::sys::ORT_API_VERSION
        )));
    }
    Ok(())
}

/// Pure check of an ORT version string `major.minor[.patch]`: major must be
/// `1` and minor at least the `ORT_API_VERSION` `ort` was built against.
fn check_runtime_version(version: &str) -> Result<(), RedactError> {
    let mut parts = version.split('.');
    let parsed = (
        parts.next().and_then(|part| part.parse::<u32>().ok()),
        parts.next().and_then(|part| part.parse::<u32>().ok()),
    );
    let (Some(major), Some(minor)) = parsed else {
        return Err(backend("unparsable version string".to_owned()));
    };
    if major != 1 || minor < ort::sys::ORT_API_VERSION {
        return Err(backend(format!(
            "expected >= '1.{}.x'",
            ort::sys::ORT_API_VERSION
        )));
    }
    Ok(())
}

/// Loads ONNX Runtime once for the process; repeat sessions over the same
/// canonical path are fine, a different path is an error, and a failed load
/// is not cached so a later retry can succeed.
fn ensure_runtime_loaded(runtime: &Path) -> Result<(), RedactError> {
    let mut guard = ORT_RUNTIME
        .lock()
        .map_err(|_| backend("ONNX Runtime registry lock is poisoned".to_owned()))?;
    if check_runtime_claim(guard.as_deref(), runtime)? {
        return Ok(());
    }
    validate_runtime_library(runtime)?;
    ort::init_from(runtime)
        .map_err(|error| {
            backend(format!(
                "failed to load ONNX Runtime `{}`: {error}",
                runtime.display()
            ))
        })?
        .with_telemetry(false)
        .commit();
    *guard = Some(runtime.to_path_buf());
    Ok(())
}

/// Validates the ORT session metadata against the `YuNet` contract: a single
/// f32 `[1,3,640,640]` input and all nine `cls`/`obj`/`bbox` heads found by
/// name with concrete expected shapes.
fn validate_coreml_session(session: &Session) -> Result<(), RedactError> {
    let inputs = session.inputs();
    if inputs.len() != 1 {
        return Err(backend(format!(
            "expected a single model input, found {}",
            inputs.len()
        )));
    }
    let (ty, dims) = tensor_meta(inputs[0].dtype()).ok_or_else(|| {
        backend(format!(
            "model input `{}` is not a tensor",
            inputs[0].name()
        ))
    })?;
    check_tensor_meta(
        inputs[0].name(),
        ty,
        dims,
        &[
            1,
            3,
            i64::try_from(INPUT_SIZE).unwrap_or(i64::MAX),
            i64::try_from(INPUT_SIZE).unwrap_or(i64::MAX),
        ],
    )?;

    for stride in STRIDES {
        let anchors = (INPUT_SIZE as u32 / stride).pow(2);
        for (kind, width) in [("cls", 1), ("obj", 1), ("bbox", 4)] {
            let name = format!("{kind}_{stride}");
            let outlet = session
                .outputs()
                .iter()
                .find(|outlet| outlet.name() == name)
                .ok_or_else(|| backend(format!("model is missing required output `{name}`")))?;
            let (ty, dims) = tensor_meta(outlet.dtype())
                .ok_or_else(|| backend(format!("model output `{name}` is not a tensor")))?;
            check_tensor_meta(&name, ty, dims, &[1, i64::from(anchors), width])?;
        }
    }
    Ok(())
}

/// Extracts the element type and shape of a tensor-typed [`ValueType`].
fn tensor_meta(dtype: &ValueType) -> Option<(TensorElementType, &[i64])> {
    match dtype {
        ValueType::Tensor { ty, shape, .. } => Some((*ty, &shape[..])),
        _ => None,
    }
}

/// Checks one tensor's element type and exact shape; pure so unit tests can
/// exercise it without a session.
fn check_tensor_meta(
    name: &str,
    ty: TensorElementType,
    dims: &[i64],
    expected: &[i64],
) -> Result<(), RedactError> {
    if ty != TensorElementType::Float32 {
        return Err(backend(format!(
            "model tensor `{name}` has type {ty:?}; expected f32"
        )));
    }
    if dims != expected {
        return Err(backend(format!(
            "model tensor `{name}` has shape {dims:?}; expected {expected:?}"
        )));
    }
    Ok(())
}

/// Borrows the nine `cls`/`obj`/`bbox` head slices out of a run's outputs,
/// checked by name and shape — same contract as the `tract` extraction.
pub(crate) fn extract_heads<'o>(
    outputs: &'o SessionOutputs<'_>,
) -> Result<Vec<Head<'o>>, RedactError> {
    let mut heads = Vec::with_capacity(STRIDES.len());
    for stride in STRIDES {
        let anchors = i64::from((INPUT_SIZE as u32 / stride).pow(2));
        heads.push(Head {
            stride,
            cls: head_slice(outputs, &format!("cls_{stride}"), &[1, anchors, 1])?,
            obj: head_slice(outputs, &format!("obj_{stride}"), &[1, anchors, 1])?,
            bbox: head_slice(outputs, &format!("bbox_{stride}"), &[1, anchors, 4])?,
        });
    }
    Ok(heads)
}

fn head_slice<'o>(
    outputs: &'o SessionOutputs<'_>,
    name: &str,
    expected: &[i64],
) -> Result<&'o [f32], RedactError> {
    let value = outputs
        .get(name)
        .ok_or_else(|| backend(format!("model produced no output `{name}`")))?;
    let (shape, data) = value.try_extract_tensor::<f32>().map_err(|error| {
        backend(format!(
            "model output `{name}` is not an f32 tensor: {error}"
        ))
    })?;
    if &shape[..] != expected {
        return Err(backend(format!(
            "model output `{name}` has shape {shape}; expected {expected:?}"
        )));
    }
    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ort::value::Shape;

    unsafe extern "system" fn null_version() -> *const core::ffi::c_char {
        core::ptr::null()
    }

    unsafe extern "system" fn fake_version() -> *const core::ffi::c_char {
        c"1.24.3".as_ptr()
    }

    unsafe extern "system" fn null_api(_version: u32) -> *const ort::sys::OrtApi {
        core::ptr::null()
    }

    #[test]
    fn runtime_version_requires_compatible_1x_minor() {
        assert!(check_runtime_version("1.24.3").is_ok());
        assert!(check_runtime_version("1.25.0").is_ok());
        for bad in ["1.23.0", "2.24.0", "invalid", "1.x.0", ""] {
            assert!(check_runtime_version(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn api_base_rejects_null_members_without_panicking() {
        let runtime = Path::new("/fake/libonnxruntime.dylib");

        // A null base, a null version string and a null `GetApi` result must
        // each surface as a `Backend` error instead of a crash — the mock
        // struct lets us test this without loading a foreign dylib or calling
        // into ort's API.
        let error = unsafe { validate_api_base(core::ptr::null(), runtime) }.unwrap_err();
        assert!(error.to_string().contains("null API base"), "{error}");

        let base = ort::sys::OrtApiBase {
            GetApi: null_api,
            GetVersionString: null_version,
        };
        let error = unsafe { validate_api_base(&base, runtime) }.unwrap_err();
        assert!(error.to_string().contains("null version string"), "{error}");

        let base = ort::sys::OrtApiBase {
            GetApi: null_api,
            GetVersionString: fake_version,
        };
        let error = unsafe { validate_api_base(&base, runtime) }.unwrap_err();
        assert!(error.to_string().contains("API version"), "{error}");
    }

    #[test]
    fn runtime_claim_allows_same_and_rejects_different_library() {
        let a = Path::new("/runtime/a/libonnxruntime.dylib");
        let b = Path::new("/runtime/b/libonnxruntime.dylib");

        // Nothing loaded -> must load; same canonical path -> reuse.
        assert!(!check_runtime_claim(None, a).unwrap());
        assert!(check_runtime_claim(Some(a), a).unwrap());
        // A different path is a hard error, never silently ignored.
        let error = check_runtime_claim(Some(a), b).unwrap_err();
        assert!(error.to_string().contains("already loaded"), "{error}");
    }

    #[test]
    fn tensor_meta_validates_type_and_shape() {
        let expected = [1, 6400, 1];
        assert!(
            check_tensor_meta("cls_8", TensorElementType::Float32, &expected, &expected).is_ok()
        );
        assert!(
            check_tensor_meta("cls_8", TensorElementType::Int64, &expected, &expected).is_err()
        );
        assert!(
            check_tensor_meta("cls_8", TensorElementType::Float32, &[1, 6400], &expected).is_err()
        );

        // Non-tensor outputs yield no metadata to validate against.
        let map = ValueType::Map {
            key: TensorElementType::Int64,
            value: TensorElementType::Float32,
        };
        assert!(tensor_meta(&map).is_none());
        let tensor = ValueType::Tensor {
            ty: TensorElementType::Float32,
            shape: Shape::new([1, 6400, 1]),
            dimension_symbols: ort::value::SymbolicDimensions::empty(3),
        };
        assert_eq!(
            tensor_meta(&tensor).map(|(ty, dims)| (ty, dims.to_vec())),
            Some((TensorElementType::Float32, vec![1, 6400, 1]))
        );
    }
}
