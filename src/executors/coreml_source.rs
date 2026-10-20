//! Opt-in source compiler for native application qualification.
use super::{CoremlComputeUnits, CoremlNativeSession};
use crate::{converters, error::GraphError, protos::coreml::specification::Model};
use prost::Message;
use std::path::Path;

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
        let data = model.encode_to_vec();
        drop(model);
        progress("bundle/start");
        let result = Self::compile_file_backed_with_progress(data, weights, policy, &mut progress)?;
        progress("bundle/done");
        Ok(result)
    }
}
