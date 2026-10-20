//! Opt-in source compiler for native application qualification.
use super::{
    CompiledCoremlModel, CoremlComputeUnits, CoremlLoadRoute, CoremlModelBacking,
    CoremlNativeSession, LoadTrace, ReleaseOnDrop, nsurl_from_path, shim_error_to_string,
};
use crate::{converters, error::GraphError, protos::coreml::specification::Model};
use objc::rc::autoreleasepool;
use objc::runtime::Object;
use objc::{class, msg_send, sel, sel_impl};
use prost::Message;
use std::fs;
use std::path::{Path, PathBuf};
use std::ptr;

impl CoremlNativeSession {
    /// Parse a local WebNN JSON/text bundle, lower its static graph to native
    /// MIL, emit a compiled bundle locally, and load native pipeline children.
    ///
    /// This explicitly selects the experimental block-local numerical profile;
    /// it does not change the portable WebNN backend's default lowering. Models
    /// and policies must be qualified against their original accuracy gates.
    /// Compressed affine leaves and bounded expansion are enabled only on watchOS.
    /// No network access, helper process, or host inference stage is used.
    pub fn compile_webnn_source(
        path: impl AsRef<Path>,
        policy: CoremlComputeUnits,
    ) -> Result<Self, GraphError> {
        Self::compile_webnn_source_with_progress(path, policy, |_| {})
    }
    /// The same local compiler with coarse setup-only progress notifications.
    /// Notifications never expose tensor values and are not used by prediction.
    pub fn compile_webnn_source_with_progress(
        path: impl AsRef<Path>,
        policy: CoremlComputeUnits,
        mut progress: impl FnMut(&str),
    ) -> Result<Self, GraphError> {
        let (data, weights) = Self::lower_webnn_source(path.as_ref(), policy, &mut progress)?;
        progress("bundle/start");
        let result = Self::compile_file_backed_with_progress(data, weights, policy, &mut progress)?;
        progress("bundle/done");
        Ok(result)
    }
    /// [`Self::compile_webnn_source_with_progress`] with a persistent compiled-bundle
    /// cache owned by the caller.
    ///
    /// `cache_dir/<key>/` holds the exported bundle (`bundle.mlmodelc`), its native
    /// child routing plan (`plan.json`) and a completion marker. When the entry is
    /// complete the WebNN source is not read at all: the children are loaded from the
    /// cached bundle after a `cache/hit` notification. Otherwise (`cache/miss`) the
    /// source is lowered as usual, exported into a staging directory beside the entry,
    /// published with an atomic rename, and loaded from its final location. An entry
    /// that fails to load is removed and rebuilt once; a bundle that fails to load
    /// right after it was built is removed and the error returned.
    ///
    /// `key` is the caller's identity for the source bundle, the native library and
    /// the compute policy (all three change the exported bundle); it must be 1 to 128
    /// ASCII letters, digits, `-`, `_` or `.` and may not start with `.`. The cache
    /// never inspects the key, never evicts other keys and never deletes a complete
    /// entry on its own. Each entry is about the size of the source weights, so the
    /// caller decides when to remove entries whose source bundle is gone.
    pub fn compile_webnn_source_cached_with_progress(
        path: impl AsRef<Path>,
        policy: CoremlComputeUnits,
        cache_dir: impl AsRef<Path>,
        key: &str,
        mut progress: impl FnMut(&str),
    ) -> Result<Self, GraphError> {
        let entry = CacheEntry::new(cache_dir.as_ref(), key)?;
        if entry.is_complete() {
            progress("cache/hit");
            match entry.plan().and_then(|plan| {
                load_persistent_bundle(&entry.bundle(), plan.as_deref(), policy, &mut progress)
            }) {
                Ok(model) => {
                    progress("bundle/done");
                    return Ok(Self { model });
                }
                Err(error) => {
                    log::warn!(
                        "cached CoreML bundle {} failed to load and is rebuilt: {error}",
                        entry.dir.display()
                    );
                    progress("cache/evict");
                    let _ = fs::remove_dir_all(&entry.dir);
                }
            }
        }
        progress("cache/miss");
        let (data, weights) = Self::lower_webnn_source(path.as_ref(), policy, &mut progress)?;
        weights
            .flush()
            .map_err(|e| GraphError::CoremlRuntimeFailed {
                reason: e.to_string(),
            })?;
        progress("bundle/start");
        entry.publish(&data, weights.as_ref())?;
        drop(data);
        drop(weights);
        let model = entry
            .plan()
            .and_then(|plan| {
                load_persistent_bundle(&entry.bundle(), plan.as_deref(), policy, &mut progress)
            })
            .inspect_err(|_| {
                let _ = fs::remove_dir_all(&entry.dir);
            })?;
        progress("bundle/done");
        Ok(Self { model })
    }
    /// Parse, lower and partition the source into one encoded pipeline model plus
    /// its file-backed weights. Shared by the direct and the cached compilers.
    fn lower_webnn_source(
        path: &Path,
        policy: CoremlComputeUnits,
        progress: &mut impl FnMut(&str),
    ) -> Result<(Vec<u8>, converters::CoremlWeightStorage), GraphError> {
        let lowering_error = |reason: String| GraphError::ConversionFailed {
            format: "coreml".into(),
            reason,
        };
        progress("parse/start");
        let (data, mut weights) =
            converters::CoremlMlProgramConverter.convert_source_file_backed(path)?;
        progress("convert/done");
        let mut model =
            Model::decode(data.as_slice()).map_err(|e| lowering_error(e.to_string()))?;
        drop(data);
        converters::coreml_projection::apply_file_backed(
            &mut model,
            &mut weights,
            cfg!(target_os = "watchos"),
        )
        .map_err(lowering_error)?;
        #[cfg(not(target_os = "watchos"))]
        if policy == CoremlComputeUnits::CpuOnly {
            converters::coreml_projection::compress_cpu_constants(&mut model);
        }
        #[cfg(target_os = "watchos")]
        let _ = policy;
        progress("projection/done");
        converters::coreml_numerics::apply(&mut model);
        progress("numerics/done");
        converters::coreml_assets::compact_file_backed(&mut model, &mut weights)
            .map_err(lowering_error)?;
        progress("compact/done");
        converters::coreml_partition::split(&mut model, 1024)?;
        progress("partition/done");
        #[cfg(target_os = "watchos")]
        {
            let expansions =
                converters::coreml_packed::bound_packed_matmuls(&mut model, 2 * 1024 * 1024);
            converters::coreml_local_producers::clone_for_packed_children(&mut model, &expansions);
        }
        Ok((model.encode_to_vec(), weights))
    }
}

/// One entry of the persistent compiled-bundle cache: `<cache_dir>/<key>/`.
pub(super) struct CacheEntry {
    key: String,
    dir: PathBuf,
}

impl CacheEntry {
    const BUNDLE: &'static str = "bundle.mlmodelc";
    const PLAN: &'static str = "plan.json";
    const COMPLETE: &'static str = "complete";

    pub(super) fn new(cache_dir: &Path, key: &str) -> Result<Self, GraphError> {
        validate_cache_key(key)?;
        Ok(Self {
            key: key.to_owned(),
            dir: cache_dir.join(key),
        })
    }
    pub(super) fn bundle(&self) -> PathBuf {
        self.dir.join(Self::BUNDLE)
    }
    /// A complete entry has its marker and its bundle directory.
    pub(super) fn is_complete(&self) -> bool {
        self.dir.join(Self::COMPLETE).is_file() && self.bundle().is_dir()
    }
    /// The serialized child routing plan; `None` for a single-model bundle.
    fn plan(&self) -> Result<Option<Vec<u8>>, GraphError> {
        let path = self.dir.join(Self::PLAN);
        match fs::read(&path) {
            Ok(plan) => Ok(Some(plan)),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(GraphError::export(path, error)),
        }
    }
    /// Export the encoded model and its weights into a staging directory beside
    /// the entry, then publish the entry with one atomic rename. Stale staging
    /// directories of this key from earlier processes are removed first. When
    /// another process published the same key meanwhile, its complete entry wins.
    pub(super) fn publish(&self, model_bytes: &[u8], weights: &[u8]) -> Result<(), GraphError> {
        let parent = self
            .dir
            .parent()
            .ok_or_else(|| GraphError::CoremlRuntimeFailed {
                reason: "compiled bundle cache directory has no parent".into(),
            })?;
        fs::create_dir_all(parent).map_err(|e| GraphError::export(parent, e))?;
        let prefix = format!("{}.partial-", self.key);
        if let Ok(entries) = fs::read_dir(parent) {
            for stale in entries.flatten() {
                if stale.file_name().to_string_lossy().starts_with(&prefix) {
                    let _ = fs::remove_dir_all(stale.path());
                }
            }
        }
        let staging = parent.join(format!("{prefix}{}", std::process::id()));
        let staged = CacheEntry {
            key: self.key.clone(),
            dir: staging.clone(),
        };
        fs::create_dir_all(&staging).map_err(|e| GraphError::export(&staging, e))?;
        let exported =
            export_bundle_to(model_bytes, Some(weights), &staged.bundle()).and_then(|plan| {
                if let Some(plan) = plan {
                    let path = staging.join(Self::PLAN);
                    fs::write(&path, plan).map_err(|e| GraphError::export(path, e))?;
                }
                let marker = staging.join(Self::COMPLETE);
                fs::write(&marker, b"complete\n").map_err(|e| GraphError::export(marker, e))
            });
        if let Err(error) = exported {
            let _ = fs::remove_dir_all(&staging);
            return Err(error);
        }
        match fs::rename(&staging, &self.dir) {
            Ok(()) => Ok(()),
            Err(error) => {
                let _ = fs::remove_dir_all(&staging);
                if self.is_complete() {
                    Ok(())
                } else {
                    Err(GraphError::export(&self.dir, error))
                }
            }
        }
    }
}

/// Keys are path components chosen by the caller: 1 to 128 ASCII letters, digits,
/// `-`, `_` or `.`, not starting with `.`.
pub(super) fn validate_cache_key(key: &str) -> Result<(), GraphError> {
    let valid = !key.is_empty()
        && key.len() <= 128
        && !key.starts_with('.')
        && key
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'));
    if valid {
        Ok(())
    } else {
        Err(GraphError::ConversionFailed {
            format: "coreml".into(),
            reason: format!("invalid compiled bundle cache key {key:?}"),
        })
    }
}

/// Emit the bundle with `mlmodelc-export` and return the serialized routing plan
/// (`None` when the model is not a native pipeline).
fn export_bundle_to(
    model_bytes: &[u8],
    weights: Option<&[u8]>,
    bundle_dir: &Path,
) -> Result<Option<Vec<u8>>, GraphError> {
    let plan = super::pipeline::from_bytes(model_bytes)
        .and_then(|plan| {
            plan.map(|p| serde_json::to_vec(&p).map_err(|e| e.to_string()))
                .transpose()
        })
        .map_err(|reason| GraphError::ConversionFailed {
            format: "coreml".into(),
            reason,
        })?;
    mlmodelc_export::compile_to_dir(model_bytes, weights, bundle_dir).map_err(|e| {
        GraphError::CoremlRuntimeFailed {
            reason: format!("local CoreML bundle export failed: {e}"),
        }
    })?;
    Ok(plan)
}

/// Load a published bundle without CPU fallback. The directory stays in place when
/// the model is released; the cache owner decides its lifetime.
fn load_persistent_bundle(
    bundle_dir: &Path,
    plan: Option<&[u8]>,
    policy: CoremlComputeUnits,
    progress: &mut impl FnMut(&str),
) -> Result<CompiledCoremlModel, GraphError> {
    let mut trace = LoadTrace::strict(policy);
    let route = CoremlLoadRoute::RustExport;
    let callback: &mut dyn FnMut(&str) = progress;
    let mut observer = super::pipeline::LoadObserver::new(Some(callback));
    autoreleasepool(|| unsafe {
        let url = trace.prepare(route, || {
            let url = nsurl_from_path(bundle_dir)?;
            let _: *mut Object = msg_send![url, retain];
            Ok(url)
        })?;
        // Owned (+1) by us; released when this closure returns on any path.
        let _url_guard = ReleaseOnDrop(url);
        let (model, name) = trace.policies(route, |code| {
            let config: *mut Object = msg_send![class!(MLModelConfiguration), new];
            let _config_guard = ReleaseOnDrop(config);
            let () = msg_send![config, setComputeUnits: code];
            let mut model: *mut Object = ptr::null_mut();
            let mut error = [0u8; 1024];
            let status = if let Some(plan) = plan {
                super::rustnn_coreml_load_pipeline(
                    url,
                    config,
                    plan.as_ptr(),
                    plan.len(),
                    observer.callback(),
                    observer.context(),
                    &mut model,
                    error.as_mut_ptr().cast(),
                    error.len(),
                )
            } else {
                super::rustnn_coreml_load(
                    url,
                    config,
                    &mut model,
                    error.as_mut_ptr().cast(),
                    error.len(),
                )
            };
            if status != 0 || model.is_null() {
                return Err(format!(
                    "MLModel load failed: {}",
                    shim_error_to_string(&error)
                ));
            }
            Ok(model)
        })?;
        // The shim returns an owned (+1) model; `CompiledCoremlModel`'s Drop releases it.
        Ok(CompiledCoremlModel {
            model,
            compute_unit: name,
            diagnostics: trace.finish(route, name),
            backing: CoremlModelBacking::Persistent {
                compiled_dir: bundle_dir.to_path_buf(),
            },
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::converters::{CoremlMlProgramConverter, GraphConverter};
    use crate::mlcontext::{MLGraphBuilder, MLNamedOperands, MLOperandDescriptor};
    use crate::operator_enums::MLOperandDataType;

    fn tiny_model() -> Vec<u8> {
        let mut b = MLGraphBuilder::new_uncompiled();
        let desc = MLOperandDescriptor::new(MLOperandDataType::Float32, vec![4]);
        let x = b.input("x", &desc).unwrap();
        let y = b.add(x, x).unwrap();
        let g = b
            .finish_graph_info(&MLNamedOperands::from([("y", y)]))
            .unwrap();
        CoremlMlProgramConverter.convert(&g).unwrap().data
    }

    #[test]
    fn keys_are_single_safe_path_components() {
        for key in ["a", "sha256-0f3a", "model_1.cpu", &"k".repeat(128)] {
            validate_cache_key(key).unwrap();
        }
        for key in [
            "",
            ".",
            "..",
            ".hidden",
            "a/b",
            "a\\b",
            "a b",
            "ключ",
            &"k".repeat(129),
        ] {
            assert!(validate_cache_key(key).is_err(), "{key:?}");
        }
        let cache = tempfile::tempdir().unwrap();
        assert!(CacheEntry::new(cache.path(), "../escape").is_err());
        let entry = CacheEntry::new(cache.path(), "abc").unwrap();
        assert_eq!(entry.dir, cache.path().join("abc"));
        assert_eq!(
            entry.bundle(),
            cache.path().join("abc").join("bundle.mlmodelc")
        );
    }

    #[test]
    fn entries_are_complete_only_with_marker_and_bundle() {
        let cache = tempfile::tempdir().unwrap();
        let entry = CacheEntry::new(cache.path(), "k").unwrap();
        assert!(!entry.is_complete());
        fs::create_dir_all(entry.bundle()).unwrap();
        assert!(!entry.is_complete(), "bundle without marker");
        fs::write(entry.dir.join("complete"), b"complete\n").unwrap();
        assert!(entry.is_complete());
        fs::remove_dir_all(entry.bundle()).unwrap();
        assert!(!entry.is_complete(), "marker without bundle");
        assert_eq!(entry.plan().unwrap(), None);
    }

    #[test]
    fn publish_exports_atomically_and_removes_stale_staging() {
        let cache = tempfile::tempdir().unwrap();
        let root = cache.path().join("compiled");
        let entry = CacheEntry::new(&root, "tiny").unwrap();
        let stale = root.join("tiny.partial-1");
        fs::create_dir_all(stale.join("bundle.mlmodelc")).unwrap();
        let other = root.join("other.partial-1");
        fs::create_dir_all(&other).unwrap();
        entry.publish(&tiny_model(), &[]).unwrap();
        assert!(entry.is_complete());
        assert!(
            entry.bundle().join("model.mil").is_file(),
            "exported MIL program"
        );
        assert_eq!(
            entry.plan().unwrap(),
            None,
            "a single program has no routing plan"
        );
        assert!(!stale.exists(), "stale staging of this key removed");
        assert!(other.exists(), "other keys untouched");
        let names: Vec<_> = fs::read_dir(&root)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert!(
            !names.iter().any(|n| n.starts_with("tiny.partial-")),
            "{names:?}"
        );
        // Publishing again onto a complete entry keeps the existing entry.
        entry.publish(&tiny_model(), &[]).unwrap();
        assert!(entry.is_complete());
    }

    #[test]
    fn publish_failure_leaves_no_entry() {
        let cache = tempfile::tempdir().unwrap();
        let entry = CacheEntry::new(cache.path(), "bad").unwrap();
        assert!(entry.publish(b"invalid protobuf", &[]).is_err());
        assert!(!entry.dir.exists());
        assert!(!cache.path().join("bad.partial-1").exists());
        let leftovers: Vec<_> = fs::read_dir(cache.path()).unwrap().collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");
    }

    #[cfg(target_vendor = "apple")]
    #[test]
    fn cached_bundles_load_and_outlive_the_session() {
        let cache = tempfile::tempdir().unwrap();
        let entry = CacheEntry::new(cache.path(), "tiny").unwrap();
        entry.publish(&tiny_model(), &[]).unwrap();
        let mut phases = Vec::new();
        let model = load_persistent_bundle(
            &entry.bundle(),
            None,
            CoremlComputeUnits::CpuOnly,
            &mut |phase: &str| phases.push(phase.to_owned()),
        )
        .unwrap();
        assert_eq!(model.load_diagnostics().loaded_compute_units, "CPU_ONLY");
        drop(model);
        assert!(
            entry.is_complete(),
            "the session must not delete a cached bundle"
        );
        assert!(
            load_persistent_bundle(
                &cache.path().join("missing.mlmodelc"),
                None,
                CoremlComputeUnits::CpuOnly,
                &mut |_: &str| {}
            )
            .is_err()
        );
    }
}
