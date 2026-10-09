//! Bounded execution of documented CoreML Pipeline child specifications.
//!
//! A Pipeline is an ordered feature map, not a chain requiring passthrough
//! copies. Keep each produced feature until its last consumer and load only
//! one child MLModel at a time. Compiler-returned URLs are cached separately.

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::ptr;
use std::rc::Rc;

use objc::runtime::{BOOL, NO, Object};
use objc::{class, msg_send, sel, sel_impl};
use prost::Message;

use super::*;
use crate::backend_selection::DeviceType;
use crate::protos::coreml::mil_spec::{self as mil, argument, value};
use crate::protos::coreml::specification::{
    ArrayFeatureType, FeatureDescription, Model, ModelDescription, array_feature_type,
    feature_type, model,
};

#[path = "coreml_pipeline_weights.rs"]
mod source_weights;

#[derive(Debug)]
struct Stage {
    /// Original child protobuf bytes, including unknown fields.
    source: Vec<u8>,
    inputs: Vec<FeatureDescription>,
    outputs: Vec<FeatureDescription>,
    release_after: Vec<String>,
    has_weights: bool,
    execution: StageExecution,
}

#[derive(Clone, Debug, PartialEq)]
enum StageExecution {
    Native,
    TypedUnary(typed_unary::Kind),
    ConstantGelu(Box<typed_unary::ConstantUnary>),
}

fn execution(model: &Model, wire: &[u8]) -> StageExecution {
    if let Some(kind) = typed_unary::classify(model, wire) {
        StageExecution::TypedUnary(kind)
    } else if let Some(constant) = typed_unary::classify_constant(model, wire) {
        StageExecution::ConstantGelu(Box::new(constant))
    } else {
        StageExecution::Native
    }
}

pub(super) struct SourcePlan {
    inputs: HashMap<String, FeatureDescription>,
    outputs: HashMap<String, FeatureDescription>,
    aliases: CoremlFeatureAliases<'static>,
    // Resolve references only after the one shared source mapping is available.
    constant_copy_metadata: Option<String>,
    stages: Vec<Stage>,
}

fn arrays(
    features: &[FeatureDescription],
) -> Result<HashMap<String, FeatureDescription>, GraphError> {
    let mut result = HashMap::new();
    for feature in features {
        if feature.name.is_empty() || result.contains_key(&feature.name) {
            return Err(boundary_error(
                "Pipeline features must have unique nonempty names",
            ));
        }
        let array = array_type(feature)?;
        NativeType::from_code(i64::from(array.data_type))?;
        validate_feature_shape(array, &array.shape)?;
        result.insert(feature.name.clone(), feature.clone());
    }
    Ok(result)
}

fn array_type(feature: &FeatureDescription) -> Result<&ArrayFeatureType, GraphError> {
    match feature
        .r#type
        .as_ref()
        .and_then(|kind| kind.r#type.as_ref())
    {
        Some(feature_type::Type::MultiArrayType(array)) => Ok(array),
        _ => Err(boundary_error(format!(
            "Pipeline feature `{}` is not a multi-array",
            feature.name
        ))),
    }
}

fn validate_feature_shape(array: &ArrayFeatureType, actual: &[i64]) -> Result<(), GraphError> {
    if actual.is_empty() || actual.iter().any(|&extent| extent <= 0) {
        return Err(boundary_error(
            "Pipeline native feature shape must be positive and nonempty",
        ));
    }
    let allowed = match &array.shape_flexibility {
        None => actual == array.shape,
        Some(array_feature_type::ShapeFlexibility::EnumeratedShapes(shapes)) => {
            !shapes.shapes.is_empty() && shapes.shapes.iter().any(|shape| shape.shape == actual)
        }
        Some(array_feature_type::ShapeFlexibility::ShapeRange(range)) => {
            range.size_ranges.len() == actual.len()
                && range
                    .size_ranges
                    .iter()
                    .zip(actual)
                    .all(|(bound, &extent)| {
                        bound.upper_bound >= i64::try_from(bound.lower_bound).unwrap_or(i64::MAX)
                            && u64::try_from(extent).is_ok_and(|extent| extent >= bound.lower_bound)
                            && extent <= bound.upper_bound
                    })
        }
    };
    if !allowed {
        return Err(boundary_error(format!(
            "Pipeline actual shape {actual:?} violates its source feature constraint"
        )));
    }
    Ok(())
}

/// Read length-delimited messages without reconstructing their contents. Prost
/// remains responsible for decoding the schema; this preserves source bytes
/// when extracting the public Model.pipeline.models field.
fn message_fields(bytes: &[u8], wanted: u64) -> Result<Vec<&[u8]>, GraphError> {
    fn varint(bytes: &[u8], cursor: &mut usize) -> Result<u64, GraphError> {
        let mut value = 0u64;
        for shift in (0..70).step_by(7) {
            let byte = *bytes
                .get(*cursor)
                .ok_or_else(|| boundary_error("truncated Pipeline protobuf"))?;
            *cursor += 1;
            if shift == 63 && byte > 1 {
                return Err(boundary_error("Pipeline protobuf varint overflow"));
            }
            value |= u64::from(byte & 127) << shift;
            if byte & 128 == 0 {
                return Ok(value);
            }
        }
        Err(boundary_error("invalid Pipeline protobuf varint"))
    }
    let mut cursor = 0;
    let mut result = Vec::new();
    while cursor < bytes.len() {
        let key = varint(bytes, &mut cursor)?;
        if key >> 3 == 0 {
            return Err(boundary_error("invalid Pipeline protobuf field"));
        }
        let length = match key & 7 {
            0 => {
                varint(bytes, &mut cursor)?;
                continue;
            }
            1 => 8,
            2 => usize::try_from(varint(bytes, &mut cursor)?)
                .map_err(|_| boundary_error("Pipeline protobuf length overflow"))?,
            5 => 4,
            _ => return Err(boundary_error("unsupported Pipeline protobuf wire type")),
        };
        let end = cursor
            .checked_add(length)
            .filter(|&end| end <= bytes.len())
            .ok_or_else(|| boundary_error("truncated Pipeline protobuf field"))?;
        if key >> 3 == wanted {
            if key & 7 != 2 {
                return Err(boundary_error("Pipeline message has the wrong wire type"));
            }
            result.push(&bytes[cursor..end]);
        }
        cursor = end;
    }
    Ok(result)
}

fn weight_value(value: &mil::Value) -> Result<bool, GraphError> {
    match &value.value {
        Some(value::Value::BlobFileValue(blob)) => {
            if blob.file_name != "@model_path/weights/weights.bin" {
                return Err(boundary_error(format!(
                    "unsupported Pipeline weight reference `{}`",
                    blob.file_name
                )));
            }
            Ok(true)
        }
        Some(value::Value::ImmediateValue(immediate)) => {
            let values: Vec<&mil::Value> = match &immediate.value {
                Some(value::immediate_value::Value::Tuple(tuple)) => tuple.values.iter().collect(),
                Some(value::immediate_value::Value::List(list)) => list.values.iter().collect(),
                Some(value::immediate_value::Value::Dictionary(dictionary)) => dictionary
                    .values
                    .iter()
                    .flat_map(|pair| pair.key.iter().chain(pair.value.iter()))
                    .collect(),
                _ => vec![],
            };
            let mut any = false;
            for value in values {
                any |= weight_value(value)?;
            }
            Ok(any)
        }
        _ => Ok(false),
    }
}

fn block_weights(block: &mil::Block) -> Result<bool, GraphError> {
    let mut any = false;
    for value in block.attributes.values() {
        any |= weight_value(value)?;
    }
    for operation in &block.operations {
        for value in operation.attributes.values() {
            any |= weight_value(value)?;
        }
        for argument in operation.inputs.values() {
            for binding in &argument.arguments {
                if let Some(argument::binding::Binding::Value(value)) = &binding.binding {
                    any |= weight_value(value)?;
                }
            }
        }
        for child in &operation.blocks {
            any |= block_weights(child)?;
        }
    }
    Ok(any)
}

fn program_weights(program: &mil::Program) -> Result<bool, GraphError> {
    let mut any = false;
    for value in program.attributes.values() {
        any |= weight_value(value)?;
    }
    for function in program.functions.values() {
        for value in function.attributes.values() {
            any |= weight_value(value)?;
        }
        for block in function.block_specializations.values() {
            any |= block_weights(block)?;
        }
    }
    Ok(any)
}

fn source_aliases(
    description: &ModelDescription,
) -> Result<CoremlFeatureAliases<'static>, GraphError> {
    let empty = HashMap::new();
    let metadata = description
        .metadata
        .as_ref()
        .map(|metadata| &metadata.user_defined)
        .unwrap_or(&empty);
    let inputs: Vec<_> = description
        .input
        .iter()
        .map(|feature| feature.name.clone())
        .collect();
    let outputs: Vec<_> = description
        .output
        .iter()
        .map(|feature| feature.name.clone())
        .collect();
    let aliases = |key: &str, declared: &[String], input: bool| -> Result<_, GraphError> {
        if let Some(json) = metadata.get(key) {
            parse_feature_aliases(
                json,
                declared,
                if input { "input" } else { "output" },
                !input,
            )
        } else {
            Ok(HashMap::new())
        }
    };
    let input_aliases = aliases(INPUT_ALIASES_METADATA_KEY, &inputs, true)?;
    let output_aliases = aliases(OUTPUT_ALIASES_METADATA_KEY, &outputs, false)?;
    let passthroughs = metadata
        .get(OUTPUT_PASSTHROUGHS_METADATA_KEY)
        .map(|json| parse_passthroughs(json, &input_aliases, &output_aliases, &inputs, &outputs))
        .transpose()?
        .unwrap_or_default();
    let compact_input_views = metadata
        .get(input_views::METADATA_KEY)
        .map(|json| input_views::parse(json, &inputs))
        .transpose()?
        .unwrap_or_default();
    Ok(CoremlFeatureAliases {
        inputs: input_aliases,
        outputs: output_aliases,
        passthroughs,
        compact_input_views,
        constant_copies: HashMap::new(),
    })
}

impl SourcePlan {
    fn constant_copy_metadata(description: &ModelDescription) -> Option<String> {
        description
            .metadata
            .as_ref()?
            .user_defined
            .get(OUTPUT_CONSTANT_COPIES_METADATA_KEY)
            .cloned()
    }

    #[cfg(test)]
    fn native_control(mut self) -> Self {
        for stage in &mut self.stages {
            stage.execution = StageExecution::Native;
        }
        self
    }

    pub(super) fn parse(bytes: &[u8]) -> Result<Option<Self>, GraphError> {
        let model = Model::decode(bytes)
            .map_err(|error| boundary_error(format!("invalid CoreML source: {error}")))?;
        let stage_execution = execution(&model, bytes);
        if matches!(model.r#type, Some(model::Type::MlProgram(_)))
            && stage_execution != StageExecution::Native
        {
            let description = model
                .description
                .as_ref()
                .expect("classifier checks description");
            let inputs = arrays(&description.input)?;
            let outputs = arrays(&description.output)?;
            return Ok(Some(Self {
                aliases: source_aliases(description)?,
                constant_copy_metadata: Self::constant_copy_metadata(description),
                stages: vec![Stage {
                    source: bytes.to_vec(),
                    inputs: description.input.clone(),
                    outputs: description.output.clone(),
                    release_after: inputs
                        .keys()
                        .filter(|name| !outputs.contains_key(*name))
                        .cloned()
                        .collect(),
                    has_weights: match model.r#type.as_ref().unwrap() {
                        model::Type::MlProgram(program) => program_weights(program)?,
                        _ => unreachable!(),
                    },
                    execution: stage_execution,
                }],
                inputs,
                outputs,
            }));
        }
        let Some(model::Type::Pipeline(pipeline)) = &model.r#type else {
            return Ok(None);
        };
        let description = model
            .description
            .as_ref()
            .ok_or_else(|| boundary_error("Pipeline has no description"))?;
        let inputs = arrays(&description.input)?;
        let outputs = arrays(&description.output)?;
        if pipeline.models.is_empty() || outputs.is_empty() {
            return Err(boundary_error("Pipeline requires children and outputs"));
        }
        let aliases = source_aliases(description)?;
        for binding in &aliases.compact_input_views {
            let source = array_type(&inputs[&binding.source])?;
            let view = array_type(&inputs[&binding.view])?;
            if source.data_type != NativeType::Float16.code()
                || view.data_type != source.data_type
                || view.shape.len() != 1
            {
                return Err(boundary_error(
                    "Pipeline compact inputs require a Half source and rank-one Half view",
                ));
            }
        }
        let containers = message_fields(bytes, 202)?;
        if containers.len() != 1 {
            return Err(boundary_error("Pipeline must have one source container"));
        }
        let children = message_fields(containers[0], 1)?;
        if children.len() != pipeline.models.len() {
            return Err(boundary_error("Pipeline child wire count mismatch"));
        }
        let mut available = inputs.clone();
        let mut last_use: HashMap<_, _> = inputs.keys().map(|name| (name.clone(), 0)).collect();
        let mut stages = Vec::new();
        for (index, (child, wire)) in pipeline.models.iter().zip(children).enumerate() {
            let Some(model::Type::MlProgram(program)) = &child.r#type else {
                return Err(boundary_error(
                    "bounded Pipeline children must be MLProgram specifications",
                ));
            };
            let child_description = child
                .description
                .as_ref()
                .ok_or_else(|| boundary_error("Pipeline child has no description"))?;
            let child_inputs = arrays(&child_description.input)?;
            let child_outputs = arrays(&child_description.output)?;
            for (name, feature) in &child_inputs {
                let produced = available.get(name).ok_or_else(|| {
                    boundary_error(format!(
                        "Pipeline child {index} consumes unavailable feature `{name}`"
                    ))
                })?;
                if array_type(feature)?.data_type != array_type(produced)?.data_type {
                    return Err(boundary_error(format!(
                        "Pipeline child {index} changes wire dtype `{name}`"
                    )));
                }
                last_use.insert(name.clone(), index);
            }
            for (name, feature) in child_outputs {
                if available.insert(name.clone(), feature).is_some() {
                    return Err(boundary_error(format!(
                        "Pipeline child {index} redefines feature `{name}`"
                    )));
                }
                last_use.entry(name).or_insert(index);
            }
            stages.push(Stage {
                source: wire.to_vec(),
                inputs: child_description.input.clone(),
                outputs: child_description.output.clone(),
                release_after: vec![],
                has_weights: program_weights(program)?,
                execution: execution(child, wire),
            });
        }
        for (name, feature) in &outputs {
            let produced = available.get(name).ok_or_else(|| {
                boundary_error(format!("Pipeline output `{name}` has no producer"))
            })?;
            if array_type(feature)?.data_type != array_type(produced)?.data_type {
                return Err(boundary_error(format!(
                    "Pipeline output `{name}` changes wire dtype"
                )));
            }
        }
        for (name, index) in last_use {
            if !outputs.contains_key(&name) {
                stages[index].release_after.push(name);
            }
        }
        for stage in &mut stages {
            stage.release_after.sort();
        }
        Ok(Some(Self {
            inputs,
            outputs,
            aliases,
            constant_copy_metadata: Self::constant_copy_metadata(description),
            stages,
        }))
    }
}

struct CompiledChild {
    path: PathBuf,
    /// Remember a successful load policy; do not retry a known failed
    /// accelerator load on every token or grow identical fallback diagnostics.
    policy: Option<(i64, &'static str)>,
}

/// Materialize only derived, documented source assets. All borrowed/allocated
/// weight spans are dropped before invoking the SDK compiler below.
fn prepare_child_source(
    stage: &Stage,
    directory: &std::path::Path,
    shared_weights: &std::path::Path,
    weights: Option<&[u8]>,
) -> Result<Option<CoremlLoadFailure>, GraphError> {
    let source = directory.join("model.mlmodel");
    if !stage.has_weights {
        std::fs::write(&source, &stage.source)
            .map_err(|error| boundary_error(format!("Pipeline child source: {error}")))?;
        return Ok(None);
    }
    let weights =
        weights.ok_or_else(|| boundary_error("Pipeline shared weight mapping missing"))?;
    let derived = source_weights::repack_child(&stage.source, weights);
    let weight_directory = directory.join("weights");
    std::fs::create_dir(&weight_directory)
        .map_err(|error| boundary_error(format!("Pipeline child weights: {error}")))?;
    let child_weights = weight_directory.join("weights.bin");
    match derived {
        Ok(Some(compact)) => {
            if compact.mappings.is_empty() {
                return Err(boundary_error(
                    "weighted child has no inventoried weight entries",
                ));
            }
            std::fs::write(&source, &compact.source)
                .and_then(|()| std::fs::write(&child_weights, &compact.weights))
                .map_err(|error| {
                    boundary_error(format!("Pipeline derived child assets: {error}"))
                })?;
            Ok(None)
        }
        Ok(None) => Err(boundary_error(
            "weighted child has no source BlobFile references",
        )),
        Err(source_weights::RepackError::Invalid(reason)) => Err(boundary_error(format!(
            "Pipeline malformed source weights: {reason}"
        ))),
        Err(source_weights::RepackError::Unsupported(reason)) => {
            std::fs::write(&source, &stage.source)
                .and_then(|()| std::fs::hard_link(shared_weights, &child_weights))
                .map_err(|error| {
                    boundary_error(format!("Pipeline original child assets: {error}"))
                })?;
            Ok(Some(CoremlLoadFailure {
                route: CoremlLoadRoute::CompiledUrl,
                compute_units: None,
                reason: format!(
                    "source weight compaction unavailable; retained original source: {reason}"
                ),
            }))
        }
    }
}

impl Drop for CompiledChild {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

struct LoadedChild {
    index: usize,
    model: ReleaseOnDrop,
}

struct PipelineState {
    loaded: Option<LoadedChild>,
    compiled: Vec<Option<CompiledChild>>,
    diagnostics: CoremlLoadDiagnostics,
}

pub(super) struct MappedWeights {
    owner: ReleaseOnDrop,
    pointer: *const u8,
    length: usize,
}

impl MappedWeights {
    unsafe fn open(path: &std::path::Path) -> Result<Self, GraphError> {
        let filename = unsafe {
            nsstring_from_str(
                path.to_str()
                    .ok_or_else(|| boundary_error("weight path is not UTF-8"))?,
            )?
        };
        let mut error: *mut Object = ptr::null_mut();
        // NSDataReadingMappedAlways: immutable original payload, not a decoded
        // second tensor allocation. The graph owns this one shared mapping.
        let data: *mut Object = msg_send![class!(NSData), dataWithContentsOfFile: filename options: 8usize error: &mut error];
        if data.is_null() {
            return Err(boundary_error(unsafe {
                ns_error_to_string(error, "mapping source weights")
            }));
        }
        let owned: *mut Object = msg_send![data, retain];
        let owner = ReleaseOnDrop(owned);
        let length: usize = msg_send![data, length];
        let pointer: *const u8 = msg_send![data, bytes];
        if pointer.is_null() || length > isize::MAX as usize {
            return Err(boundary_error("source weight mapping span differs"));
        }
        Ok(Self {
            owner,
            pointer,
            length,
        })
    }

    pub(super) fn bytes(&self) -> &[u8] {
        let _ = &self.owner;
        // SAFETY: immutable NSData mapping is retained for the graph lifetime.
        unsafe { std::slice::from_raw_parts(self.pointer, self.length) }
    }
}

pub(crate) struct PipelineModel {
    plan: SourcePlan,
    /// One model at most; returned features own their native storage separately.
    state: RefCell<PipelineState>,
    source_root: tempfile::TempDir,
    device: DeviceType,
    weights: Option<Rc<MappedWeights>>,
}

impl std::fmt::Debug for PipelineModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (compiled, loaded) = self.cache_counts();
        let (native, host) = self.stage_counts();
        f.debug_struct("BoundedCoremlPipeline")
            .field("children", &self.plan.stages.len())
            .field("native_stage_count", &native)
            .field("host_typed_stages", &host)
            .field("loaded_model_capacity", &1)
            .field("compiled_children", &compiled)
            .field("loaded_children", &loaded)
            .field("diagnostics", &self.diagnostics())
            .finish()
    }
}

impl Drop for PipelineModel {
    fn drop(&mut self) {
        autoreleasepool(|| {
            let state = self.state.get_mut();
            state.loaded = None;
            state.compiled.clear();
        });
    }
}

impl PipelineModel {
    pub(super) fn feature_aliases(&self) -> &CoremlFeatureAliases<'static> {
        &self.plan.aliases
    }

    pub(super) fn compile(
        mut plan: SourcePlan,
        weights: Option<&[u8]>,
        device: DeviceType,
    ) -> Result<Self, GraphError> {
        let root = tempfile::Builder::new()
            .prefix("rustnn_coreml_children_")
            .tempdir()
            .map_err(|error| boundary_error(format!("Pipeline source directory: {error}")))?;
        let needs_weights = plan.stages.iter().any(|stage| stage.has_weights)
            || plan.constant_copy_metadata.is_some();
        if needs_weights {
            let weights = weights
                .ok_or_else(|| boundary_error("Pipeline source references missing weights"))?;
            std::fs::write(root.path().join("weights.bin"), weights)
                .map_err(|error| boundary_error(format!("Pipeline shared weights: {error}")))?;
        }
        let first_native = plan
            .stages
            .iter()
            .position(|stage| matches!(stage.execution, StageExecution::Native));
        let mapped = if needs_weights {
            Some(Rc::new(autoreleasepool(|| unsafe {
                MappedWeights::open(&root.path().join("weights.bin"))
            })?))
        } else {
            None
        };
        if let Some(metadata) = plan.constant_copy_metadata.take() {
            let storage = mapped
                .as_ref()
                .map(|mapping| CoremlWeightStorage::Mapped(Rc::clone(mapping)));
            let outputs = plan.outputs.keys().cloned().collect::<Vec<_>>();
            plan.aliases.constant_copies = parse_constant_copies(
                &metadata,
                &plan.aliases.outputs,
                &outputs,
                &plan.aliases.passthroughs,
                storage.as_ref(),
            )?;
        }
        for stage in &mut plan.stages {
            if let StageExecution::ConstantGelu(constant) = &mut stage.execution {
                constant.resolve(mapped.as_deref().map(MappedWeights::bytes))?;
            }
        }
        let diagnostics = LoadTrace::new(device).finish(
            if first_native.is_some() {
                CoremlLoadRoute::CompiledUrl
            } else {
                CoremlLoadRoute::TypedHost
            },
            if first_native.is_some() {
                load::compute_unit_for_device(device).1
            } else {
                "NOT_APPLICABLE"
            },
        );
        let model = Self {
            state: RefCell::new(PipelineState {
                loaded: None,
                compiled: (0..plan.stages.len()).map(|_| None).collect(),
                diagnostics,
            }),
            plan,
            source_root: root,
            device,
            weights: mapped,
        };
        // Establish an actual successful load, without compiling the opaque
        // Pipeline or retaining all children. Later children remain lazy.
        if let Some(index) = first_native {
            autoreleasepool(|| unsafe { model.load_child(index) })?;
        }
        Ok(model)
    }

    pub(super) fn diagnostics(&self) -> CoremlLoadDiagnostics {
        self.state.borrow().diagnostics.clone()
    }

    pub(super) fn cache_counts(&self) -> (usize, usize) {
        let state = self.state.borrow();
        (
            state.compiled.iter().flatten().count(),
            usize::from(state.loaded.is_some()),
        )
    }

    /// Exact source-classified execution census, distinct from lazy cache
    /// counts and requested accelerator permissions.
    pub(super) fn stage_counts(&self) -> (usize, usize) {
        let host = self
            .plan
            .stages
            .iter()
            .filter(|stage| !matches!(stage.execution, StageExecution::Native))
            .count();
        (self.plan.stages.len() - host, host)
    }

    unsafe fn load_child(&self, index: usize) -> Result<*mut Object, GraphError> {
        let mut state = self
            .state
            .try_borrow_mut()
            .map_err(|_| boundary_error("reentrant CoreML Pipeline dispatch"))?;
        if let Some(child) = &state.loaded
            && child.index == index
        {
            return Ok(child.model.0);
        }
        // Release before compiling/loading the next model: capacity is one even
        // at the transition, while feature values retain returned tensor owners.
        state.loaded = None;
        let mut trace = LoadTrace::new(self.device);
        let route = CoremlLoadRoute::CompiledUrl;
        if state.compiled[index].is_none() {
            let stage = &self.plan.stages[index];
            let temporary = tempfile::Builder::new()
                .prefix("child_")
                .tempdir_in(self.source_root.path())
                .map_err(|error| {
                    boundary_error(format!("Pipeline child {index} source: {error}"))
                })?;
            let source = temporary.path().join("model.mlmodel");
            if let Some(diagnostic) = prepare_child_source(
                stage,
                temporary.path(),
                &self.source_root.path().join("weights.bin"),
                self.weights.as_deref().map(MappedWeights::bytes),
            )? && !state.diagnostics.failures.contains(&diagnostic)
            {
                state.diagnostics.failures.push(diagnostic);
            }
            let compiled = trace.prepare(route, || unsafe {
                let url = nsurl_from_path(&source)?;
                let mut compiled: *mut Object = ptr::null_mut();
                let mut error = [0u8; 1024];
                let status = rustnn_coreml_compile(
                    url,
                    &mut compiled,
                    error.as_mut_ptr().cast(),
                    error.len(),
                );
                if status != 0 || compiled.is_null() {
                    return Err(boundary_error(format!(
                        "Pipeline child {index} compilation: {}",
                        shim_error_to_string(&error)
                    )));
                }
                let guard = ReleaseOnDrop(compiled);
                let path: *mut Object = msg_send![guard.0, path];
                Ok(CompiledChild {
                    path: PathBuf::from(nsstring_to_string(path)),
                    policy: None,
                })
            })?;
            state.compiled[index] = Some(compiled);
        }
        let compiled = state.compiled[index]
            .as_ref()
            .expect("compiled child was inserted");
        let url = unsafe { nsurl_from_path(&compiled.path)? };
        let cached_policy = compiled.policy;
        let mut load = |code| unsafe {
            let configuration: *mut Object = msg_send![class!(MLModelConfiguration), new];
            let configuration = ReleaseOnDrop(configuration);
            let () = msg_send![configuration.0, setComputeUnits: code];
            let mut child: *mut Object = ptr::null_mut();
            let mut error = [0u8; 1024];
            let status = rustnn_coreml_load(
                url,
                configuration.0,
                &mut child,
                error.as_mut_ptr().cast(),
                error.len(),
            );
            if status != 0 || child.is_null() {
                return Err(format!(
                    "Pipeline child {index} load: {}",
                    shim_error_to_string(&error)
                ));
            }
            Ok(ReleaseOnDrop(child))
        };
        let loaded = if let Some((code, name)) = cached_policy {
            match load(code) {
                Ok(model) => Ok((model, name)),
                Err(reason) => {
                    let failure = CoremlLoadFailure {
                        route,
                        compute_units: Some(name),
                        reason,
                    };
                    if !state.diagnostics.failures.contains(&failure) {
                        state.diagnostics.failures.push(failure);
                    }
                    trace.policies(route, &mut load)
                }
            }
        } else {
            trace.policies(route, &mut load)
        };
        let (child, policy) = loaded?;
        let diagnostics = trace.finish(route, policy);
        for failure in diagnostics.failures {
            if !state.diagnostics.failures.contains(&failure) {
                state.diagnostics.failures.push(failure);
            }
        }
        let code = if policy == "CPU_ONLY" {
            0
        } else {
            load::compute_unit_for_device(self.device).0
        };
        state.compiled[index]
            .as_mut()
            .expect("compiled child was inserted")
            .policy = Some((code, policy));
        if state
            .compiled
            .iter()
            .filter(|child| child.is_some())
            .count()
            == 1
        {
            state.diagnostics.loaded_compute_units = policy;
        } else if policy != state.diagnostics.loaded_compute_units {
            state.diagnostics.loaded_compute_units = "PER_STAGE";
        }
        let pointer = child.0;
        state.loaded = Some(LoadedChild {
            index,
            model: child,
        });
        Ok(pointer)
    }

    pub(super) fn predict(
        &self,
        inputs: &HashMap<String, CoremlByteInput<'_>>,
        outputs: &HashMap<String, OperandDescriptor>,
    ) -> Result<HashMap<String, Vec<u8>>, GraphError> {
        let mut passthroughs =
            snapshot_byte_passthroughs(&self.plan.aliases.passthroughs, inputs, outputs)?;
        passthroughs.extend(snapshot_byte_constant_copies(
            &self.plan.aliases.constant_copies,
            outputs,
        )?);
        let mut values = HashMap::<String, ReleaseOnDrop>::new();
        autoreleasepool(|| unsafe {
            for (name, input) in inputs {
                let physical = self.plan.aliases.inputs.get(name).unwrap_or(name);
                let feature = self.plan.inputs.get(physical).ok_or_else(|| {
                    boundary_error(format!("Pipeline input `{name}` is not declared"))
                })?;
                let mut shape: Vec<_> = input
                    .descriptor
                    .static_shape()
                    .ok_or_else(|| {
                        boundary_error("Pipeline input requires its actual bound shape")
                    })?
                    .into_iter()
                    .map(i64::from)
                    .collect();
                if shape.is_empty() {
                    shape.push(1);
                }
                let constraint = array_type(feature)?;
                validate_feature_shape(constraint, &shape)?;
                let array = create_multi_array(&shape, constraint.data_type)?;
                fill_multiarray_from_bytes(
                    array,
                    input.data,
                    input.descriptor.data_type,
                    constraint.data_type,
                )?;
                let value: *mut Object =
                    msg_send![class!(MLFeatureValue), featureValueWithMultiArray: array];
                let owned: *mut Object = msg_send![value, retain];
                if values
                    .insert(physical.clone(), ReleaseOnDrop(owned))
                    .is_some()
                {
                    return Err(boundary_error("duplicate bound Pipeline input"));
                }
            }
            Ok(())
        })?;
        let values = self.predict_features(values)?;
        self.extract_outputs(&values, outputs, passthroughs)
    }

    pub(super) fn input_dtype(
        &self,
        name: &str,
        descriptor: &OperandDescriptor,
    ) -> Result<i32, GraphError> {
        let physical = self
            .plan
            .aliases
            .inputs
            .get(name)
            .map_or(name, String::as_str);
        let feature =
            self.plan.inputs.get(physical).ok_or_else(|| {
                boundary_error(format!("Pipeline input `{name}` is not declared"))
            })?;
        let mut shape = descriptor
            .static_shape()
            .ok_or_else(|| boundary_error("Pipeline input requires its actual bound shape"))?
            .into_iter()
            .map(i64::from)
            .collect::<Vec<_>>();
        if shape.is_empty() {
            shape.push(1);
        }
        let constraint = array_type(feature)?;
        validate_feature_shape(constraint, &shape)?;
        Ok(constraint.data_type)
    }

    /// Retained input features remain owned by the caller throughout prediction.
    /// Only live stage values are kept; no public output backings enter children.
    pub(super) fn predict_features(
        &self,
        mut values: HashMap<String, ReleaseOnDrop>,
    ) -> Result<HashMap<String, ReleaseOnDrop>, GraphError> {
        autoreleasepool(|| unsafe {
            for binding in &self.plan.aliases.compact_input_views {
                let source = values
                    .get(&binding.source)
                    .ok_or_else(|| boundary_error("Pipeline compact source is not bound"))?;
                let array: *mut Object = msg_send![source.0, multiArrayValue];
                let view = input_views::flat_view(array)?;
                let (kind, layout, _) = multiarray_storage(view.array.0)?;
                let feature = &self.plan.inputs[&binding.view];
                validate_feature_shape(
                    array_type(feature)?,
                    &layout
                        .shape
                        .iter()
                        .map(|&size| size as i64)
                        .collect::<Vec<_>>(),
                )?;
                if kind != NativeType::Float16 {
                    return Err(boundary_error(
                        "Pipeline compact view lost its Half storage",
                    ));
                }
                let value: *mut Object =
                    msg_send![class!(MLFeatureValue), featureValueWithMultiArray: view.array.0];
                let owned: *mut Object = msg_send![value, retain];
                if values
                    .insert(binding.view.clone(), ReleaseOnDrop(owned))
                    .is_some()
                {
                    return Err(boundary_error(
                        "private Pipeline view must not be externally bound",
                    ));
                }
            }
            for required in self.plan.inputs.keys() {
                if !values.contains_key(required) {
                    return Err(boundary_error(format!(
                        "Pipeline input `{required}` is not bound"
                    )));
                }
            }
            Ok(())
        })?;
        for (index, stage) in self.plan.stages.iter().enumerate() {
            autoreleasepool(|| unsafe {
                if let StageExecution::TypedUnary(kind) = &stage.execution {
                    self.predict_typed_unary(stage, *kind, &mut values)?;
                    for name in &stage.release_after {
                        values.remove(name);
                    }
                    return Ok(());
                }
                if matches!(stage.execution, StageExecution::ConstantGelu(_)) {
                    self.predict_typed_unary(stage, typed_unary::Kind::Gelu, &mut values)?;
                    for name in &stage.release_after {
                        values.remove(name);
                    }
                    return Ok(());
                }
                let model = self.load_child(index)?;
                let dictionary: *mut Object = msg_send![class!(NSMutableDictionary), dictionary];
                let description: *mut Object = msg_send![model, modelDescription];
                let input_descriptions: *mut Object =
                    msg_send![description, inputDescriptionsByName];
                for input in &stage.inputs {
                    let feature = values.get(&input.name).ok_or_else(|| {
                        boundary_error(format!(
                            "Pipeline child {index} lacks input `{}`",
                            input.name
                        ))
                    })?;
                    validate_native_feature(input_descriptions, input, feature.0)?;
                    let key = nsstring_from_str(&input.name)?;
                    let () = msg_send![dictionary, setObject: feature.0 forKey: key];
                }
                let mut error: *mut Object = ptr::null_mut();
                let provider: *mut Object = msg_send![class!(MLDictionaryFeatureProvider), alloc];
                let provider: *mut Object =
                    msg_send![provider, initWithDictionary: dictionary error: &mut error];
                if provider.is_null() {
                    return Err(boundary_error(ns_error_to_string(
                        error,
                        "Pipeline provider init",
                    )));
                }
                let provider = ReleaseOnDrop(provider);
                let mut returned: *mut Object = ptr::null_mut();
                let mut error = [0u8; 1024];
                let status = rustnn_coreml_predict(
                    model,
                    provider.0,
                    &mut returned,
                    error.as_mut_ptr().cast(),
                    error.len(),
                );
                if status != 0 || returned.is_null() {
                    return Err(boundary_error(format!(
                        "Pipeline child {index} prediction: {}",
                        shim_error_to_string(&error)
                    )));
                }
                let returned = ReleaseOnDrop(returned);
                let output_descriptions: *mut Object =
                    msg_send![description, outputDescriptionsByName];
                for output in &stage.outputs {
                    let key = nsstring_from_str(&output.name)?;
                    let value: *mut Object = msg_send![returned.0, featureValueForName: key];
                    if value.is_null() {
                        return Err(boundary_error(format!(
                            "Pipeline child {index} missing output `{}`",
                            output.name
                        )));
                    }
                    validate_native_feature(output_descriptions, output, value)?;
                    let owned: *mut Object = msg_send![value, retain];
                    values.insert(output.name.clone(), ReleaseOnDrop(owned));
                }
                for name in &stage.release_after {
                    values.remove(name);
                }
                Ok(())
            })?;
        }
        Ok(values)
    }

    pub(super) unsafe fn output_array(
        &self,
        name: &str,
        descriptor: &OperandDescriptor,
        values: &HashMap<String, ReleaseOnDrop>,
    ) -> Result<*mut Object, GraphError> {
        let physical = self
            .plan
            .aliases
            .outputs
            .get(name)
            .map_or(name, String::as_str);
        let declared =
            self.plan.outputs.get(physical).ok_or_else(|| {
                boundary_error(format!("Pipeline output `{name}` is not declared"))
            })?;
        let value = values
            .get(physical)
            .ok_or_else(|| boundary_error(format!("Pipeline output `{name}` is absent")))?;
        let array: *mut Object = msg_send![value.0, multiArrayValue];
        unsafe { validate_source_feature(declared, array)? };
        let (_, layout, _) = unsafe { multiarray_storage(array)? };
        let mut actual = layout.shape.clone();
        if descriptor.shape.is_empty() && actual == [1] {
            actual.clear();
        }
        RuntimeShapeState::new().validate_shape(name, &actual, descriptor, TensorKind::Output)?;
        Ok(array)
    }

    fn extract_outputs(
        &self,
        values: &HashMap<String, ReleaseOnDrop>,
        outputs: &HashMap<String, OperandDescriptor>,
        passthroughs: HashMap<String, Vec<u8>>,
    ) -> Result<HashMap<String, Vec<u8>>, GraphError> {
        autoreleasepool(|| unsafe {
            let mut result = passthroughs;
            for (name, descriptor) in outputs {
                if result.contains_key(name) {
                    continue;
                }
                let array = self.output_array(name, descriptor, values)?;
                result.insert(name.clone(), extract_multiarray_bytes(array, descriptor)?);
            }
            Ok(result)
        })
    }

    unsafe fn predict_typed_unary(
        &self,
        stage: &Stage,
        operation: typed_unary::Kind,
        values: &mut HashMap<String, ReleaseOnDrop>,
    ) -> Result<(), GraphError> {
        let target = &stage.outputs[0];
        let (actual_shape, bytes) = if let StageExecution::ConstantGelu(constant) = &stage.execution
        {
            (
                constant.shape.clone(),
                constant.bytes(self.weights.as_deref().map(MappedWeights::bytes))?,
            )
        } else {
            let source = &stage.inputs[0];
            let feature = values.get(&source.name).ok_or_else(|| {
                boundary_error(format!("typed unary stage lacks source `{}`", source.name))
            })?;
            let array: *mut Object = msg_send![feature.0, multiArrayValue];
            unsafe { validate_source_feature(source, array)? };
            let (kind, layout, pointer) = unsafe { multiarray_storage(array)? };
            let shape = layout.shape.iter().map(|&size| size as i64).collect();
            let bytes = unsafe { read_array_storage(pointer, &layout, kind.element_size()) };
            (shape, std::borrow::Cow::Owned(bytes))
        };
        validate_feature_shape(array_type(target)?, &actual_shape)?;
        let result = typed_unary::evaluate(&bytes, operation).map_err(boundary_error)?;
        let output = unsafe { create_multi_array(&actual_shape, array_type(target)?.data_type)? };
        let (kind, layout, pointer) = unsafe { multiarray_storage(output)? };
        unsafe { write_array_storage(pointer, &layout, kind.element_size(), &result)? };
        unsafe { validate_source_feature(target, output)? };
        let value: *mut Object =
            msg_send![class!(MLFeatureValue), featureValueWithMultiArray: output];
        let owned: *mut Object = msg_send![value, retain];
        if owned.is_null()
            || values
                .insert(target.name.clone(), ReleaseOnDrop(owned))
                .is_some()
        {
            return Err(boundary_error(
                "typed unary stage did not produce a unique owned feature",
            ));
        }
        Ok(())
    }
}

unsafe fn validate_source_feature(
    feature: &FeatureDescription,
    array: *mut Object,
) -> Result<(), GraphError> {
    if array.is_null() {
        return Err(boundary_error(format!(
            "Pipeline feature `{}` has no native array",
            feature.name
        )));
    }
    let (kind, layout, _) = unsafe { multiarray_storage(array)? };
    let source = array_type(feature)?;
    if kind.code() != source.data_type {
        return Err(boundary_error(format!(
            "Pipeline feature `{}` changed native dtype",
            feature.name
        )));
    }
    validate_feature_shape(
        source,
        &layout
            .shape
            .iter()
            .map(|&size| size as i64)
            .collect::<Vec<_>>(),
    )
}

unsafe fn validate_native_feature(
    descriptions: *mut Object,
    source: &FeatureDescription,
    value: *mut Object,
) -> Result<(), GraphError> {
    let key = unsafe { nsstring_from_str(&source.name)? };
    let description: *mut Object = msg_send![descriptions, objectForKey: key];
    if description.is_null() {
        return Err(boundary_error(format!(
            "compiled child does not declare `{}`",
            source.name
        )));
    }
    let allowed: BOOL = msg_send![description, isAllowedValue: value];
    if allowed == NO {
        return Err(boundary_error(format!(
            "Pipeline feature `{}` violates native child constraints",
            source.name
        )));
    }
    let array: *mut Object = msg_send![value, multiArrayValue];
    unsafe { validate_source_feature(source, array) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protos::coreml::specification::{FeatureType, Pipeline};

    fn feature(name: &str) -> FeatureDescription {
        FeatureDescription {
            name: name.into(),
            r#type: Some(FeatureType {
                r#type: Some(feature_type::Type::MultiArrayType(ArrayFeatureType {
                    shape: vec![4],
                    data_type: NativeType::Float16.code(),
                    ..Default::default()
                })),
                ..Default::default()
            }),
            ..Default::default()
        }
    }
    fn child(inputs: &[&str], outputs: &[&str]) -> Model {
        Model {
            description: Some(ModelDescription {
                input: inputs.iter().map(|name| feature(name)).collect(),
                output: outputs.iter().map(|name| feature(name)).collect(),
                ..Default::default()
            }),
            r#type: Some(model::Type::MlProgram(Default::default())),
            ..Default::default()
        }
    }
    fn parent(children: Vec<Model>) -> Model {
        Model {
            description: Some(ModelDescription {
                input: vec![feature("x"), feature("unused")],
                output: vec![feature("result"), feature("early")],
                ..Default::default()
            }),
            r#type: Some(model::Type::Pipeline(Pipeline {
                models: children,
                names: vec![],
            })),
            ..Default::default()
        }
    }
    #[test]
    fn source_plan_preserves_children_and_fanout_last_use() {
        let children = vec![
            child(&["x"], &["early", "middle"]),
            child(&["middle"], &["later"]),
            child(&["early", "later"], &["result"]),
        ];
        let bytes = parent(children.clone()).encode_to_vec();
        let plan = SourcePlan::parse(&bytes).unwrap().unwrap();
        assert_eq!(plan.stages.len(), 3);
        for (stage, original) in plan.stages.iter().zip(children) {
            assert_eq!(stage.source, original.encode_to_vec());
        }
        assert_eq!(plan.stages[0].release_after, ["unused", "x"]);
        assert_eq!(plan.stages[1].release_after, ["middle"]);
        assert_eq!(plan.stages[2].release_after, ["later"]);
        assert!(
            plan.stages
                .iter()
                .all(|stage| !stage.release_after.contains(&"early".into()))
        );
    }

    #[test]
    fn source_plan_requires_explicit_aliases_and_rejects_duplicate_proofs() {
        use crate::converters::coreml_names;
        let input = coreml_names::encode("source input").into_owned();
        let output = coreml_names::encode("copy output").into_owned();
        let mut source = parent(vec![child(&[&input], &[&output])]);
        let description = source.description.as_mut().unwrap();
        description.input = vec![feature(&input)];
        description.output = vec![feature(&output)];
        let metadata = description.metadata.get_or_insert_default();
        metadata.user_defined.insert(
            coreml_names::METADATA_KEY.into(),
            coreml_names::METADATA_VALUE.into(),
        );
        let plan = SourcePlan::parse(&source.encode_to_vec()).unwrap().unwrap();
        assert!(plan.aliases.inputs.is_empty() && plan.aliases.outputs.is_empty());
        assert!(plan.inputs.contains_key(&input) && plan.outputs.contains_key(&output));

        let proof = CoremlPassthrough {
            input: "source input".into(),
            descriptor: OperandDescriptor {
                data_type: DataType::Float16,
                shape: vec![Dimension::Static(4)],
                pending_permutation: vec![],
            },
        };
        let metadata = &mut source
            .description
            .as_mut()
            .unwrap()
            .metadata
            .as_mut()
            .unwrap()
            .user_defined;
        metadata.insert(
            INPUT_ALIASES_METADATA_KEY.into(),
            serde_json::json!({"source input": input}).to_string(),
        );
        metadata.insert(
            OUTPUT_ALIASES_METADATA_KEY.into(),
            serde_json::json!({"copy output": output}).to_string(),
        );
        metadata.insert(
            OUTPUT_PASSTHROUGHS_METADATA_KEY.into(),
            serde_json::json!({"copy output": proof}).to_string(),
        );
        let plan = SourcePlan::parse(&source.encode_to_vec()).unwrap().unwrap();
        assert_eq!(plan.aliases.inputs["source input"], input);
        assert_eq!(plan.aliases.outputs["copy output"], output);
        assert_eq!(plan.aliases.passthroughs.len(), 1);

        let encoded_input = serde_json::to_string(&input).unwrap();
        let encoded_output = serde_json::to_string(&output).unwrap();
        let encoded_proof = serde_json::to_string(&proof).unwrap();
        for (key, duplicate) in [
            (
                INPUT_ALIASES_METADATA_KEY,
                format!(r#"{{"source input":{encoded_input},"source input":{encoded_input}}}"#),
            ),
            (
                OUTPUT_ALIASES_METADATA_KEY,
                format!(r#"{{"copy output":{encoded_output},"copy output":{encoded_output}}}"#),
            ),
            (
                OUTPUT_PASSTHROUGHS_METADATA_KEY,
                format!(r#"{{"copy output":{encoded_proof},"copy output":{encoded_proof}}}"#),
            ),
        ] {
            let mut malformed = source.clone();
            malformed
                .description
                .as_mut()
                .unwrap()
                .metadata
                .as_mut()
                .unwrap()
                .user_defined
                .insert(key.into(), duplicate);
            assert!(
                SourcePlan::parse(&malformed.encode_to_vec()).is_err(),
                "{key}"
            );
        }
    }

    #[test]
    fn source_plan_rejects_missing_redefined_or_mistyped_features() {
        for children in [
            vec![child(&["missing"], &["early", "result"])],
            vec![child(&["x"], &["x", "early", "result"])],
            vec![child(&["x"], &["early"])],
        ] {
            assert!(SourcePlan::parse(&parent(children).encode_to_vec()).is_err());
        }
        let mut second = child(&["early"], &["result"]);
        if let Some(feature_type::Type::MultiArrayType(array)) =
            second.description.as_mut().unwrap().input[0]
                .r#type
                .as_mut()
                .unwrap()
                .r#type
                .as_mut()
        {
            array.data_type = NativeType::Float32.code();
        }
        assert!(
            SourcePlan::parse(&parent(vec![child(&["x"], &["early"]), second]).encode_to_vec())
                .is_err()
        );
    }
    #[test]
    fn source_plan_preserves_unknown_child_fields() {
        let mut model = parent(vec![]);
        model.r#type = None;
        let mut wire = model.encode_to_vec();
        let mut original = child(&["x"], &["early", "result"]).encode_to_vec();
        original.extend([0xf8, 0x3f, 0x07]);
        let mut pipeline = vec![0x0a];
        prost::encoding::encode_varint(original.len() as u64, &mut pipeline);
        pipeline.extend_from_slice(&original);
        prost::encoding::encode_varint((202 << 3) | 2, &mut wire);
        prost::encoding::encode_varint(pipeline.len() as u64, &mut wire);
        wire.extend_from_slice(&pipeline);
        let plan = SourcePlan::parse(&wire).unwrap().unwrap();
        assert_eq!(plan.stages[0].source, original);
        assert!(message_fields(&[0x0a, 0x08, 0x01], 1).is_err());
        assert!(message_fields(&[0x0a, 0xff], 1).is_err());
    }
    #[test]
    fn actual_shapes_are_checked_without_freezing_to_maxima() {
        let array = ArrayFeatureType {
            shape: vec![3, 4],
            shape_flexibility: Some(array_feature_type::ShapeFlexibility::ShapeRange(
                array_feature_type::ShapeRange {
                    size_ranges: vec![
                        crate::protos::coreml::specification::SizeRange {
                            lower_bound: 1,
                            upper_bound: 3,
                        },
                        crate::protos::coreml::specification::SizeRange {
                            lower_bound: 4,
                            upper_bound: 4,
                        },
                    ],
                },
            )),
            ..Default::default()
        };
        for actual in [vec![1, 4], vec![3, 4], vec![2, 4]] {
            validate_feature_shape(&array, &actual).unwrap();
        }
        for actual in [vec![4, 4], vec![0, 4], vec![3], vec![3, 3]] {
            assert!(validate_feature_shape(&array, &actual).is_err());
        }
    }

    #[cfg(target_vendor = "apple")]
    fn cast_graph(shape: Vec<Dimension>) -> crate::graph::GraphInfo {
        use crate::graph::{Operand, OperandKind};
        use crate::operator_enums::MLOperandDataType;
        use crate::operators::Operation;
        let operand = |name: &str, kind, dtype| Operand {
            name: Some(name.into()),
            kind,
            descriptor: OperandDescriptor {
                data_type: dtype,
                shape: shape.clone(),
                pending_permutation: vec![],
            },
        };
        crate::graph::GraphInfo {
            operands: vec![
                operand("source input", OperandKind::Input, DataType::Float32),
                operand("represented half", OperandKind::Output, DataType::Float16),
                operand("wide result", OperandKind::Output, DataType::Float32),
            ],
            input_operands: vec![0],
            output_operands: vec![1, 2],
            operations: vec![
                Operation::Cast {
                    input: 0,
                    data_type: MLOperandDataType::Float16,
                    options: None,
                    outputs: vec![1],
                },
                Operation::Cast {
                    input: 1,
                    data_type: MLOperandDataType::Float32,
                    options: None,
                    outputs: vec![2],
                },
            ],
            ..Default::default()
        }
    }

    #[cfg(target_vendor = "apple")]
    fn check_cast_dispatch(
        model: &CompiledCoremlModel,
        graph: &crate::graph::GraphInfo,
        length: u32,
    ) {
        let special = [
            1.0003f32,
            1.0007,
            -0.0,
            -2f32.powi(-25),
            65520.0,
            -65520.0,
            f32::NAN,
            2f32.powi(-24),
        ];
        let data: Vec<_> = (0..length as usize)
            .map(|index| special[index % special.len()])
            .collect();
        let bytes: Vec<_> = data.iter().flat_map(|value| value.to_ne_bytes()).collect();
        let mut descriptor = graph.operands[0].descriptor.clone();
        descriptor.shape = vec![Dimension::Static(length)];
        let inputs = [(
            "source input".into(),
            CoremlByteInput {
                data: &bytes,
                descriptor: &descriptor,
            },
        )]
        .into();
        let outputs = [
            (
                "represented half".into(),
                graph.operands[1].descriptor.clone(),
            ),
            ("wide result".into(), graph.operands[2].descriptor.clone()),
        ]
        .into();
        let result = run_coreml_bytes(model, &inputs, &outputs).unwrap();
        assert_eq!(result["represented half"].len(), length as usize * 2);
        assert_eq!(result["wide result"].len(), length as usize * 4);
        for (index, source) in data.into_iter().enumerate() {
            let expected = half::f16::from_f32(source);
            let half = half::f16::from_bits(u16::from_ne_bytes(
                result["represented half"][index * 2..index * 2 + 2]
                    .try_into()
                    .unwrap(),
            ));
            let wide = f32::from_ne_bytes(
                result["wide result"][index * 4..index * 4 + 4]
                    .try_into()
                    .unwrap(),
            );
            if expected.is_nan() {
                assert!(half.is_nan() && wide.is_nan());
            } else {
                assert_eq!(half.to_bits(), expected.to_bits());
                assert_eq!(wide.to_bits(), expected.to_f32().to_bits());
            }
        }
    }

    #[cfg(target_vendor = "apple")]
    #[test]
    fn constant_copy_pipeline_shares_one_mapping_and_releases_it_after_dispatch() {
        use crate::converters::{CoremlMlProgramConverter, GraphConverter};
        use crate::graph::{ConstantData, Operand, OperandKind};
        use crate::operators::Operation;

        let mut graph = cast_graph(vec![Dimension::Static(8)]);
        let expected = [1_u32, 0x8000_0000, 0x3f80_0001, 0x7fc1_2345]
            .repeat(327_680)
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        assert_eq!(expected.len(), 5 * 1024 * 1024);
        let descriptor = OperandDescriptor {
            data_type: DataType::Float32,
            shape: vec![Dimension::Static((expected.len() / 4) as u32)],
            pending_permutation: vec![],
        };
        for (name, kind) in [
            ("original constant", OperandKind::Constant),
            ("first copy", OperandKind::Output),
            ("second copy", OperandKind::Output),
        ] {
            graph.operands.push(Operand {
                name: Some(name.into()),
                kind,
                descriptor: descriptor.clone(),
            });
        }
        graph.constant_operand_ids_to_handles.insert(
            3,
            ConstantData {
                data: expected.clone(),
                label: None,
            },
        );
        for (input, output) in [(3, 4), (4, 5)] {
            graph.operations.push(Operation::Identity {
                input,
                outputs: vec![output],
                options: None,
            });
            graph.output_operands.push(output);
        }
        let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
        let input_descriptor = graph.operands[0].descriptor.clone();
        let outputs = graph
            .output_operands
            .iter()
            .map(|&id| {
                let operand = &graph.operands[id as usize];
                (operand.name.clone().unwrap(), operand.descriptor.clone())
            })
            .collect::<HashMap<_, _>>();
        drop(graph);
        let input_bytes = [0_u8; 32];
        let inputs = HashMap::from([(
            "source input".into(),
            CoremlByteInput {
                data: &input_bytes,
                descriptor: &input_descriptor,
            },
        )]);
        assert!(compile_model(converted.data.clone(), None, DeviceType::Cpu, false).is_err());
        let mut corrupted = converted.weights_data.clone().unwrap();
        corrupted[128] ^= 1;
        assert!(
            compile_model(
                converted.data.clone(),
                Some(corrupted),
                DeviceType::Cpu,
                false
            )
            .is_err()
        );
        let weak = autoreleasepool(|| {
            let model = compile_model(
                converted.data,
                converted.weights_data,
                DeviceType::Cpu,
                false,
            )
            .unwrap();
            let CompiledCoremlModel::Pipeline(pipeline) = &model else {
                panic!("typed Cast plus constant output must remain a Pipeline")
            };
            let mapping = pipeline.weights.as_ref().unwrap();
            let weak = Rc::downgrade(mapping);
            for copy in pipeline.plan.aliases.constant_copies.values() {
                let CoremlWeightStorage::Mapped(owner) = &copy.bytes.storage else {
                    panic!("constant views must reuse the source mapping")
                };
                assert!(Rc::ptr_eq(mapping, owner));
                assert_eq!(&*copy.bytes, expected);
            }
            assert_eq!(pipeline.plan.aliases.constant_copies.len(), 2);
            for _ in 0..2 {
                let mut result = run_coreml_bytes(&model, &inputs, &outputs).unwrap();
                assert_eq!(result["first copy"], expected);
                assert_eq!(result["second copy"], expected);
                result.get_mut("first copy").unwrap()[0] ^= 0xff;
                assert_eq!(result["second copy"], expected);
            }
            let mut invalid = outputs.clone();
            invalid.get_mut("second copy").unwrap().data_type = DataType::Int32;
            assert!(run_coreml_bytes(&model, &inputs, &invalid).is_err());
            assert_eq!(
                run_coreml_bytes(&model, &inputs, &outputs).unwrap()["second copy"],
                expected
            );
            weak
        });
        assert!(
            weak.upgrade().is_none(),
            "mapping must not outlive the model"
        );
    }

    #[cfg(target_vendor = "apple")]
    #[test]
    fn bounded_native_cast_pipeline_keeps_two_outputs_and_reuses_compiled_urls() {
        use crate::converters::{CoremlMlProgramConverter, GraphConverter};
        let graph = cast_graph(vec![Dimension::Static(8)]);
        let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
        let plan = SourcePlan::parse(&converted.data)
            .unwrap()
            .unwrap()
            .native_control();
        let model = CompiledCoremlModel::Pipeline(Box::new(
            PipelineModel::compile(plan, converted.weights_data.as_deref(), DeviceType::Cpu)
                .unwrap(),
        ));
        let CompiledCoremlModel::Pipeline(pipeline) = &model else {
            panic!("native bounded Pipeline expected");
        };
        assert_eq!(pipeline.state.borrow().compiled.iter().flatten().count(), 1);
        check_cast_dispatch(&model, &graph, 8);
        let urls: Vec<_> = pipeline
            .state
            .borrow()
            .compiled
            .iter()
            .map(|child| child.as_ref().unwrap().path.clone())
            .collect();
        check_cast_dispatch(&model, &graph, 8);
        let reused: Vec<_> = pipeline
            .state
            .borrow()
            .compiled
            .iter()
            .map(|child| child.as_ref().unwrap().path.clone())
            .collect();
        assert_eq!(urls, reused);
        assert!(pipeline.state.borrow().loaded.is_some());
        drop(model);
        assert!(urls.iter().all(|path| !path.exists()));
    }

    #[cfg(all(target_vendor = "apple", feature = "dynamic-inputs"))]
    #[test]
    fn bounded_native_pipeline_uses_actual_grow_and_shrink_shapes() {
        use crate::converters::{CoremlMlProgramConverter, GraphConverter};
        let graph = cast_graph(vec![Dimension::Dynamic(crate::graph::DynamicDimension {
            name: "length".into(),
            max_size: 17,
        })]);
        let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
        let plan = SourcePlan::parse(&converted.data)
            .unwrap()
            .unwrap()
            .native_control();
        let model = CompiledCoremlModel::Pipeline(Box::new(
            PipelineModel::compile(plan, converted.weights_data.as_deref(), DeviceType::Cpu)
                .unwrap(),
        ));
        for length in [1, 17, 3, 8, 1] {
            check_cast_dispatch(&model, &graph, length);
        }
    }

    #[cfg(all(target_vendor = "apple", feature = "dynamic-inputs"))]
    #[test]
    fn typed_cast_pipeline_uses_proven_actual_dynamic_shapes() {
        use crate::converters::{CoremlMlProgramConverter, GraphConverter};
        use crate::protos::coreml::mil_spec::{dimension, value_type};
        let mut graph = cast_graph(vec![Dimension::Static(8)]);
        let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
        let mut source = Model::decode(converted.data.as_slice()).unwrap();
        fn bound(source: &mut Model) {
            let description = source.description.as_mut().unwrap();
            for feature in description.input.iter_mut().chain(&mut description.output) {
                let feature_type::Type::MultiArrayType(array) =
                    feature.r#type.as_mut().unwrap().r#type.as_mut().unwrap()
                else {
                    panic!()
                };
                array.shape = vec![17];
                array.shape_flexibility = Some(array_feature_type::ShapeFlexibility::ShapeRange(
                    array_feature_type::ShapeRange {
                        size_ranges: vec![crate::protos::coreml::specification::SizeRange {
                            lower_bound: 1,
                            upper_bound: 17,
                        }],
                    },
                ));
            }
            match source.r#type.as_mut().unwrap() {
                model::Type::Pipeline(pipeline) => pipeline.models.iter_mut().for_each(bound),
                model::Type::MlProgram(program) => {
                    for function in program.functions.values_mut() {
                        for named in function.inputs.iter_mut().chain(
                            function
                                .block_specializations
                                .values_mut()
                                .flat_map(|block| {
                                    block
                                        .operations
                                        .iter_mut()
                                        .flat_map(|operation| &mut operation.outputs)
                                }),
                        ) {
                            let value_type::Type::TensorType(tensor) =
                                named.r#type.as_mut().unwrap().r#type.as_mut().unwrap()
                            else {
                                panic!()
                            };
                            tensor.dimensions = vec![crate::protos::coreml::mil_spec::Dimension {
                                dimension: Some(dimension::Dimension::Unknown(
                                    dimension::UnknownDimension { variadic: false },
                                )),
                            }];
                        }
                    }
                }
                _ => panic!(),
            }
        }
        bound(&mut source);
        for operand in &mut graph.operands {
            operand.descriptor.shape = vec![Dimension::Dynamic(crate::graph::DynamicDimension {
                name: "length".into(),
                max_size: 17,
            })];
        }
        let model = compile_model(source.encode_to_vec(), None, DeviceType::Npu, false).unwrap();
        for length in [1, 17, 3, 8, 1] {
            check_cast_dispatch(&model, &graph, length);
        }
        let CompiledCoremlModel::Pipeline(pipeline) = &model else {
            panic!("typed Cast source plan expected");
        };
        assert_eq!(pipeline.cache_counts(), (0, 0));
        assert_eq!(pipeline.stage_counts(), (0, 2));
        assert_eq!(pipeline.diagnostics().route, CoremlLoadRoute::TypedHost);
        assert_eq!(
            pipeline.diagnostics().loaded_compute_units,
            "NOT_APPLICABLE"
        );
    }

    #[cfg(target_vendor = "apple")]
    #[test]
    fn typed_gelu_classifies_scalar_constant_and_dynamic_source_features() {
        use crate::converters::{CoremlMlProgramConverter, GraphConverter};
        let mut cases = Vec::new();
        cases.push((vec![], true));
        #[cfg(feature = "dynamic-inputs")]
        cases.push((
            vec![Dimension::Dynamic(crate::graph::DynamicDimension {
                name: "length".into(),
                max_size: 8,
            })],
            false,
        ));
        for (shape, constant) in cases {
            let mut graph = cast_graph(shape);
            graph.operands.truncate(2);
            graph.operands[1].descriptor.data_type = DataType::Float32;
            graph.output_operands = vec![1];
            graph.operations = vec![crate::operators::Operation::Gelu {
                input: 0,
                options: None,
                outputs: vec![1],
            }];
            if constant {
                graph.input_operands.clear();
                graph.operands[0].kind = crate::graph::OperandKind::Constant;
                graph.constant_operand_ids_to_handles.insert(
                    0,
                    crate::graph::ConstantData {
                        data: (-10.0_f32).to_le_bytes().to_vec(),
                        label: None,
                    },
                );
            }
            let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
            let model = Model::decode(converted.data.as_slice()).unwrap();
            let plan = SourcePlan::parse(&converted.data).unwrap();
            assert!(
                plan.as_ref().is_some_and(|plan| plan
                    .stages
                    .iter()
                    .all(|stage| stage.execution != StageExecution::Native)),
                "{model:#?}"
            );
        }
    }

    #[cfg(target_vendor = "apple")]
    #[test]
    fn typed_gelu_native_control_accuracy_and_warm_timings() {
        use crate::converters::{CoremlMlProgramConverter, GraphConverter};
        for length in [4_u32, 128, 4096] {
            let mut graph = cast_graph(vec![Dimension::Static(length)]);
            graph.operands.truncate(2);
            graph.operands[1].descriptor.data_type = DataType::Float32;
            graph.output_operands = vec![1];
            graph.operations = vec![crate::operators::Operation::Gelu {
                input: 0,
                options: None,
                outputs: vec![1],
            }];
            let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
            let input_bits = [0xc1200000_u32, 0xc0a00000, 0xbf800000, 0x3f800000];
            let expected = [0x9ab83c9b_u32, 0xb5c05e5d, 0xbe227686, 0x3f57625f];
            let bytes: Vec<_> = (0..length as usize)
                .flat_map(|index| input_bits[index % 4].to_ne_bytes())
                .collect();
            let inputs = [(
                "source input".into(),
                CoremlByteInput {
                    data: &bytes,
                    descriptor: &graph.operands[0].descriptor,
                },
            )]
            .into();
            let outputs = [(
                "represented half".into(),
                graph.operands[1].descriptor.clone(),
            )]
            .into();
            for device in [DeviceType::Cpu, DeviceType::Gpu, DeviceType::Npu] {
                for repaired in [false, true] {
                    let plan = SourcePlan::parse(&converted.data).unwrap().unwrap();
                    let plan = if repaired {
                        plan
                    } else {
                        plan.native_control()
                    };
                    let model = PipelineModel::compile(plan, None, device).unwrap();
                    let actual = model.predict(&inputs, &outputs).unwrap();
                    let max_ulp = actual["represented half"]
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .enumerate()
                        .map(|(index, bytes)| {
                            u32::from_ne_bytes(*bytes).abs_diff(expected[index % 4])
                        })
                        .max()
                        .unwrap();
                    if repaired {
                        assert_eq!(max_ulp, 0);
                    }
                    for _ in 0..5 {
                        std::hint::black_box(model.predict(&inputs, &outputs).unwrap());
                    }
                    let start = std::time::Instant::now();
                    for _ in 0..25 {
                        std::hint::black_box(model.predict(&inputs, &outputs).unwrap());
                    }
                    eprintln!(
                        "GELU length={length} permission={device:?} repaired={repaired} max_ulp={max_ulp} warm_us={:.3} route={:?}",
                        start.elapsed().as_secs_f64() * 1e6 / 25.0,
                        model.diagnostics().route
                    );
                }
            }
        }
    }

    #[cfg(target_vendor = "apple")]
    #[test]
    fn typed_cast_pipeline_preserves_outputs_without_compilation_or_policy() {
        use crate::converters::{CoremlMlProgramConverter, GraphConverter};
        let graph = cast_graph(vec![Dimension::Static(65_536)]);
        let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
        let model = compile_model(
            converted.data,
            converted.weights_data,
            DeviceType::Npu,
            false,
        )
        .unwrap();
        for _ in 0..2 {
            check_cast_dispatch(&model, &graph, 65_536);
        }
        let CompiledCoremlModel::Pipeline(pipeline) = &model else {
            panic!("typed Cast source plan expected");
        };
        assert_eq!(pipeline.cache_counts(), (0, 0));
        assert_eq!(pipeline.stage_counts(), (0, 2));
        assert_eq!(pipeline.diagnostics().route, CoremlLoadRoute::TypedHost);
        assert_eq!(pipeline.diagnostics().requested_compute_units, "CPU_AND_NE");
        assert_eq!(
            pipeline.diagnostics().loaded_compute_units,
            "NOT_APPLICABLE"
        );
        assert!(pipeline.diagnostics().failures.is_empty());
    }

    #[cfg(target_vendor = "apple")]
    #[test]
    fn typed_unary_reads_actual_padded_storage_without_padding_or_reinterpretation() {
        use crate::converters::{CoremlMlProgramConverter, GraphConverter};
        for operation in [
            typed_unary::Kind::FloatCast(typed_unary::Direction::Narrow),
            typed_unary::Kind::Gelu,
        ] {
            let mut graph = cast_graph(vec![Dimension::Static(6)]);
            if operation == typed_unary::Kind::Gelu {
                graph.operands.truncate(2);
                graph.operands[1].descriptor.data_type = DataType::Float32;
                graph.output_operands = vec![1];
                graph.operations = vec![crate::operators::Operation::Gelu {
                    input: 0,
                    options: None,
                    outputs: vec![1],
                }];
            }
            let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
            let plan = SourcePlan::parse(&converted.data).unwrap().unwrap();
            let model = PipelineModel::compile(plan, None, DeviceType::Cpu).unwrap();
            assert_eq!(
                model.plan.stages[0].execution,
                StageExecution::TypedUnary(operation)
            );
            let data: Vec<_> = [
                1.0003_f32,
                -0.0,
                -2f32.powi(-25),
                f32::NAN,
                f32::INFINITY,
                2f32.powi(-24),
            ]
            .into_iter()
            .flat_map(f32::to_ne_bytes)
            .collect();
            let mut backing = vec![0xdd_u8; 12 * 4];
            let result = autoreleasepool(|| unsafe {
                let mut error: *mut Object = ptr::null_mut();
                let numbers = |values: &[i64]| {
                    let objects: Vec<*mut Object> = values
                        .iter()
                        .map(|&value| {
                            let object: *mut Object =
                                msg_send![class!(NSNumber), numberWithLongLong: value];
                            object
                        })
                        .collect();
                    let array: *mut Object = msg_send![class!(NSArray), arrayWithObjects: objects.as_ptr() count: objects.len()];
                    array
                };
                let alloc: *mut Object = msg_send![class!(MLMultiArray), alloc];
                let source: *mut Object = msg_send![alloc,
                initWithDataPointer: backing.as_mut_ptr().cast::<std::ffi::c_void>()
                shape: numbers(&[6])
                dataType: NativeType::Float32.code()
                strides: numbers(&[2])
                deallocator: ptr::null_mut::<Object>()
                error: &mut error];
                assert!(
                    !source.is_null(),
                    "{}",
                    ns_error_to_string(error, "padded Cast source")
                );
                let source = ReleaseOnDrop(source);
                let (kind, layout, pointer) = multiarray_storage(source.0).unwrap();
                write_array_storage(pointer, &layout, kind.element_size(), &data).unwrap();
                let feature: *mut Object =
                    msg_send![class!(MLFeatureValue), featureValueWithMultiArray: source.0];
                let feature: *mut Object = msg_send![feature, retain];
                let mut values = HashMap::from([(
                    model.plan.stages[0].inputs[0].name.clone(),
                    ReleaseOnDrop(feature),
                )]);
                model
                    .predict_typed_unary(&model.plan.stages[0], operation, &mut values)
                    .unwrap();
                let returned: *mut Object = msg_send![
                    values[&model.plan.stages[0].outputs[0].name].0,
                    multiArrayValue
                ];
                let (kind, layout, pointer) = multiarray_storage(returned).unwrap();
                read_array_storage(pointer, &layout, kind.element_size())
            });
            assert_eq!(result, typed_unary::evaluate(&data, operation).unwrap());
            assert!(
                backing
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .enumerate()
                    .all(|(index, value)| index % 2 == 0 || *value == [0xdd; 4])
            );
            assert_eq!(model.cache_counts(), (0, 0));
        }
    }
}
