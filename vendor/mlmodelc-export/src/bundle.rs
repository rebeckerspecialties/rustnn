//! Generate the contents of a `.mlmodelc` bundle from a [`MILProgram`].
//!
//! A bundle is a directory containing four files:
//!
//! ```text
//! example.mlmodelc/
//!   model.mil          UTF-8 MIL text
//!   coremldata.bin     binary container with the source ModelDescription
//!   metadata.json      I/O schema, op histogram, availability matrix
//!   analytics/
//!     coremldata.bin   minimal stub satisfying CoreML's analytics check
//! ```
//!
//! The wire layout of `coremldata.bin` matches Apple `coremlc` 3520.x
//! byte-for-byte for the small models we've validated.

use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::Path;

use crate::description::{FeatureDescription, ModelDescription, ShapeFlexibility, shape_text};
use crate::types::*;

/// In-memory representation of a `.mlmodelc` bundle, ready to write to disk.
#[derive(Debug, Clone)]
pub struct MlmodelcBundle {
    pub model_mil: Vec<u8>,
    pub coremldata_bin: Vec<u8>,
    pub metadata_json: Vec<u8>,
    pub analytics_coremldata_bin: Vec<u8>,
    /// Mirror of the input `weights_data`, written to `weights/weights.bin`
    /// when present. None for graphs without external weights.
    pub weights_bin: Option<Vec<u8>>,
    /// Additional external weights keyed by bundle-relative filename. The
    /// legacy `weights/weights.bin` entry stays in `weights_bin`.
    pub weight_files: BTreeMap<String, Vec<u8>>,
    /// Present for a Pipeline wrapper, whose MIL programs live in child bundles.
    pub pipeline: Option<PipelineBundle>,
}

/// Children and parameter names of an exported CoreML Pipeline.
#[derive(Debug, Clone)]
pub struct PipelineBundle {
    pub model_names_bin: Vec<u8>,
    /// Each entry contains the child bundle and its references to shared assets.
    pub models: Vec<(MlmodelcBundle, Vec<String>)>,
}

impl MlmodelcBundle {
    /// Write the bundle to `directory`. Creates the directory and any
    /// missing parents. Always overwrites if files already exist.
    pub fn write_to_dir(&self, directory: impl AsRef<Path>) -> io::Result<()> {
        let dir = directory.as_ref();
        for path in self.weight_files.keys() {
            crate::weights::validate_path(path).map_err(io::Error::other)?;
            if path == "weights/weights.bin" && self.weights_bin.is_some() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "duplicate weights/weights.bin",
                ));
            }
        }
        let legacy_path = "weights/weights.bin".to_owned();
        crate::weights::prepare_directory(
            dir,
            self.weight_files
                .keys()
                .chain(self.weights_bin.as_ref().map(|_| &legacy_path)),
        )?;
        if let Some(pipeline) = &self.pipeline {
            crate::pipeline::prepare_children(dir, pipeline)?;
        } else {
            fs::write(dir.join("model.mil"), &self.model_mil)?;
        }
        fs::write(dir.join("coremldata.bin"), &self.coremldata_bin)?;
        fs::write(dir.join("metadata.json"), &self.metadata_json)?;

        let analytics_dir = dir.join("analytics");
        fs::create_dir_all(&analytics_dir)?;
        fs::write(
            analytics_dir.join("coremldata.bin"),
            &self.analytics_coremldata_bin,
        )?;

        if let Some(weights) = &self.weights_bin {
            let weights_dir = dir.join("weights");
            fs::create_dir_all(&weights_dir)?;
            fs::write(weights_dir.join("weights.bin"), weights)?;
        }
        crate::weights::write_files(dir, &self.weight_files)?;
        if let Some(pipeline) = &self.pipeline {
            let names = dir.join("modelNames");
            fs::create_dir_all(&names)?;
            fs::write(names.join("coremldata.bin"), &pipeline.model_names_bin)?;
            for (index, (model, paths)) in pipeline.models.iter().enumerate() {
                let child = dir.join(format!("model{index}"));
                model.write_to_dir(&child)?;
                crate::pipeline::remove_child_weights(&child, paths)?;
            }
        }
        Ok(())
    }
}

/// Build the four-file bundle from a decoded `MILProgram` and the MIL text
/// produced by [`crate::emitter::emit`].
///
/// This low-level constructor does not resolve assets. Prefer
/// [`crate::compile_to_bundle_with_weight_files`] for validated assembly.
pub fn build_bundle(
    program: &MILProgram,
    model_mil: Vec<u8>,
    weights_bin: Option<Vec<u8>>,
) -> MlmodelcBundle {
    MlmodelcBundle {
        model_mil,
        coremldata_bin: generate_coremldata_bin(program),
        metadata_json: generate_metadata_json(program),
        analytics_coremldata_bin: generate_analytics_bin(),
        weights_bin,
        weight_files: BTreeMap::new(),
        pipeline: None,
    }
}

// === coremldata.bin ===

/// Generate `coremldata.bin` from a decoded `MILProgram`.
///
/// Preserve the source ModelDescription, including feature types, flexible
/// shapes and single-/multi-function form. Synthesizing it from MIL types loses
/// defaults and constraints and can make older loaders reject valid models.
///
/// Wire layout (matches `coremlc` 3520.x byte-equivalent):
///
/// ```text
/// 0x00  4    uint32 LE 502 (model type = mlProgram)
/// 0x04  4    uint32 LE spec_version
/// 0x08  20   zeros (reserved)
/// 0x1C  4    uint32 LE 7 (target string length)
/// 0x20  4    zeros
/// 0x24  7    "generic"
/// 0x2B  1    spec_version as uint8
/// 0x2C  31   zeros
/// 0x4B  8    uint64 LE payload size
/// 0x53  var  ModelDescription protobuf (includes metadata field 100)
/// var   16*N per-function trailer: uint32 LE 502 followed by 12 zeros
/// ```
pub fn generate_coremldata_bin(program: &MILProgram) -> Vec<u8> {
    let mut trailer = program.description_data.clone();
    if trailer.is_empty() {
        // Retain support for callers constructing a static MILProgram directly.
        for (name, function) in &program.functions {
            let desc = encode_function_description(name, function);
            append_proto_length_delimited(&mut trailer, 20, &desc);
        }
        if let Some((name, _)) = program.functions.first() {
            append_proto_string(&mut trailer, 21, name);
        }
    }
    if !ModelDescription::decode(&trailer).has_metadata {
        append_proto_length_delimited(&mut trailer, 100, &[]);
    }

    let desc_len = trailer.len();
    let mut bin = Vec::with_capacity(desc_len + 102);

    append_u32_le(&mut bin, 502);
    append_u32_le(&mut bin, program.spec_version as u32);
    bin.extend(std::iter::repeat_n(0u8, 20));
    append_u32_le(&mut bin, 7);
    append_u32_le(&mut bin, 0);
    bin.extend_from_slice(b"generic");
    bin.push((program.spec_version & 0xFF) as u8);
    bin.extend(std::iter::repeat_n(0u8, 31));
    append_u64_le(&mut bin, desc_len as u64);

    bin.extend_from_slice(&trailer);

    for _ in 0..program.functions.len().max(1) {
        append_u32_le(&mut bin, 502);
        bin.extend(std::iter::repeat_n(0u8, 12));
    }
    bin
}

/// Generate a minimal `analytics/coremldata.bin` — CoreML expects the file
/// to exist; the contents are tolerated as long as the header matches.
pub fn generate_analytics_bin() -> Vec<u8> {
    let mut bin = Vec::new();
    let header = "NeuralNetworkModelDetails";
    append_u64_le(&mut bin, header.len() as u64);
    bin.extend_from_slice(header.as_bytes());
    append_u64_le(&mut bin, 2);
    append_analytics_entry(&mut bin, "containsCustomLayer", "0");
    append_analytics_entry(&mut bin, "modelDimension", "0");
    bin
}

fn append_analytics_entry(bin: &mut Vec<u8>, key: &str, value: &str) {
    append_u64_le(bin, key.len() as u64);
    bin.extend_from_slice(key.as_bytes());
    append_u64_le(bin, value.len() as u64);
    bin.extend_from_slice(value.as_bytes());
}

// === metadata.json ===

/// Generate `metadata.json`.
///
/// I/O schemas use the source defaults and constraints. Operation statistics
/// describe the source MIL graph rather than Apple's optimized executable.
pub fn generate_metadata_json(program: &MILProgram) -> Vec<u8> {
    let description = ModelDescription::decode(&program.description_data);
    let multifunction = !description.functions.is_empty() || program.description_data.is_empty();
    let main_func = program
        .functions
        .iter()
        .find(|(name, _)| name == &description.default_function)
        .or_else(|| program.functions.first());
    let main_name: String = main_func
        .map(|(n, _)| n.clone())
        .unwrap_or_else(|| "main".to_string());
    let main_description = description.function(&main_name);

    let mut hist: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    if let Some((_, f)) = main_func {
        let prefix = opset_prefix(&f.opset);
        for op in f.block.all_operations() {
            *hist.entry(format!("{prefix}.{}", op.r#type)).or_insert(0) += 1;
        }
    }

    let inputs: Vec<(String, MILType)> = main_func
        .map(|(_, f)| {
            f.inputs
                .iter()
                .map(|i| (i.name.clone(), i.r#type.clone()))
                .collect()
        })
        .unwrap_or_default();

    let output_types: Vec<(String, MILType)> = main_func
        .map(|(_, f)| {
            f.block
                .outputs
                .iter()
                .map(|name| (name.clone(), find_output_type(name, &f.block)))
                .collect()
        })
        .unwrap_or_default();

    // Dominant storage/compute precision (matches `coremlc` shape).
    let mut counts: std::collections::HashMap<MILDataType, usize> =
        std::collections::HashMap::new();
    if let Some((_, f)) = main_func {
        for op in f.block.all_operations() {
            for o in &op.outputs {
                *counts.entry(o.r#type.data_type).or_insert(0) += 1;
            }
        }
    }
    let dominant = counts
        .iter()
        .max_by_key(|(dtype, count)| (**count, std::cmp::Reverse(**dtype as u64)))
        .map(|(k, _)| *k)
        .unwrap_or(MILDataType::Float32);
    let precision = format_data_type_for_meta(dominant);
    let avail = availability_for_spec(program.spec_version);

    let mut json = String::new();
    json.push_str("[\n  {\n");
    json.push_str("    \"metadataOutputVersion\" : \"3.0\",\n");
    json.push_str(&format!(
        "    \"outputSchema\" : {},\n",
        schema_list(
            &output_types,
            main_description.map(|d| d.outputs.as_slice()),
            "    "
        )
    ));
    json.push_str("    \"modelParameters\" : [\n\n    ],\n");
    json.push_str(&format!(
        "    \"specificationVersion\" : {},\n",
        program.spec_version
    ));

    // Functions array only when the source uses multi-function descriptions.
    // Field order matches Apple's
    // coremlc 3520.x output: computePrecision, outputSchema, stateSchema,
    // name, mlProgramOperationTypeHistogram, inputSchema. coremlc does NOT
    // emit `storagePrecision` here — keeping our output aligned.
    if multifunction {
        json.push_str("    \"functions\" : [\n");
        for (index, (function_name, function)) in program.functions.iter().enumerate() {
            let function_description = description.function(function_name);
            let function_inputs: Vec<_> = function
                .inputs
                .iter()
                .map(|input| (input.name.clone(), input.r#type.clone()))
                .collect();
            let function_outputs: Vec<_> = function
                .block
                .outputs
                .iter()
                .map(|name| (name.clone(), find_output_type(name, &function.block)))
                .collect();
            json.push_str("      {\n");
            json.push_str(&format!(
                "        \"computePrecision\" : \"{precision}\",\n"
            ));
            json.push_str(&format!(
                "        \"outputSchema\" : {},\n",
                schema_list(
                    &function_outputs,
                    function_description.map(|d| d.outputs.as_slice()),
                    "        "
                )
            ));
            json.push_str("        \"stateSchema\" : [\n\n        ],\n");
            json.push_str(&format!(
                "        \"name\" : {},\n",
                json_string(function_name)
            ));
            json.push_str("        \"mlProgramOperationTypeHistogram\" : {\n");
            let mut function_hist = std::collections::BTreeMap::<String, usize>::new();
            for op in function.block.all_operations() {
                *function_hist
                    .entry(format!("{}.{}", opset_prefix(&function.opset), op.r#type))
                    .or_default() += 1;
            }
            for (i, (k, v)) in function_hist.iter().enumerate() {
                json.push_str(&format!("          \"{k}\" : {v}"));
                if i < function_hist.len() - 1 {
                    json.push(',');
                }
                json.push('\n');
            }
            json.push_str("        },\n");
            json.push_str(&format!(
                "        \"inputSchema\" : {}\n",
                schema_list(
                    &function_inputs,
                    function_description.map(|d| d.inputs.as_slice()),
                    "        "
                )
            ));
            json.push_str("      }");
            if index + 1 < program.functions.len() {
                json.push(',');
            }
            json.push('\n');
        }
        json.push_str("    ],\n");
    }

    json.push_str("    \"mlProgramOperationTypeHistogram\" : {\n");
    let hist_entries: Vec<(&String, &usize)> = hist.iter().collect();
    for (i, (k, v)) in hist_entries.iter().enumerate() {
        json.push_str(&format!("      \"{k}\" : {v}"));
        if i < hist_entries.len() - 1 {
            json.push(',');
        }
        json.push('\n');
    }
    json.push_str("    },\n");

    json.push_str("    \"isUpdatable\" : \"0\",\n");
    json.push_str("    \"stateSchema\" : [\n\n    ],\n");

    json.push_str("    \"availability\" : {\n");
    for (i, (k, v)) in avail.iter().enumerate() {
        json.push_str(&format!("      \"{k}\" : \"{v}\""));
        if i < avail.len() - 1 {
            json.push(',');
        }
        json.push('\n');
    }
    json.push_str("    },\n");

    json.push_str(&format!("    \"computePrecision\" : \"{precision}\",\n"));
    json.push_str("    \"modelType\" : {\n");
    json.push_str("      \"name\" : \"MLModelType_mlProgram\"\n");
    json.push_str("    },\n");
    json.push_str(&format!(
        "    \"inputSchema\" : {},\n",
        schema_list(
            &inputs,
            main_description.map(|d| d.inputs.as_slice()),
            "    "
        )
    ));
    if multifunction {
        json.push_str(&format!(
            "    \"defaultFunctionName\" : {},\n",
            json_string(&main_name)
        ));
    }
    json.push_str("    \"generatedClassName\" : \"model\",\n");
    json.push_str("    \"userDefinedMetadata\" : {\n");
    if description.user_defined_metadata.is_empty() {
        json.push('\n');
    } else {
        for (index, (key, value)) in description.user_defined_metadata.iter().enumerate() {
            if index != 0 {
                json.push_str(",\n");
            }
            json.push_str(&format!(
                "      {} : {}",
                json_string(key),
                json_string(value)
            ));
        }
        json.push('\n');
    }
    json.push_str("    },\n");
    json.push_str("    \"method\" : \"predict\"\n");
    json.push_str("  }\n");
    json.push(']');

    json.into_bytes()
}

pub(crate) fn schema_list(
    entries: &[(String, MILType)],
    features: Option<&[FeatureDescription]>,
    indent: &str,
) -> String {
    if entries.is_empty() {
        return format!("[\n\n{indent}]");
    }
    let mut s = String::from("[\n");
    for (i, (name, ty)) in entries.iter().enumerate() {
        let feature =
            features.and_then(|features| features.iter().find(|feature| &feature.name == name));
        s.push_str(&schema_entry(name, ty, feature, &format!("{indent}  ")));
        if i < entries.len() - 1 {
            s.push(',');
        }
        s.push('\n');
    }
    s.push_str(indent);
    s.push(']');
    s
}

fn schema_entry(
    name: &str,
    r#type: &MILType,
    feature: Option<&FeatureDescription>,
    indent: &str,
) -> String {
    let dt_str = match feature.map(|f| f.data_type) {
        Some(65568) => "Float32",
        Some(65552) => "Float16",
        Some(65600) => "Double",
        Some(131104) => "Int32",
        Some(131080) => "Int8",
        _ => format_data_type_for_meta(r#type.data_type),
    };
    let dimensions: Vec<_> = feature
        .map(|f| f.default_shape().iter().map(ToString::to_string).collect())
        .unwrap_or_else(|| r#type.shape.iter().map(ToString::to_string).collect());
    let shape_str = dimensions.join(", ");
    let formatted_shape = dimensions.join(" \u{00d7} ");
    let mut s = format!("{indent}{{\n");
    let flexibility = feature.and_then(|f| f.flexibility.as_ref());
    s.push_str(&format!(
        "{indent}  \"hasShapeFlexibility\" : \"{}\",\n",
        u8::from(flexibility.is_some())
    ));
    s.push_str(&format!(
        "{indent}  \"isOptional\" : \"{}\",\n",
        u8::from(feature.is_some_and(|f| f.optional))
    ));
    if let Some(flexibility) = flexibility {
        let (key, value, formatted) = match flexibility {
            ShapeFlexibility::Ranges(ranges) => (
                "shapeRange",
                format!(
                    "[{}]",
                    ranges
                        .iter()
                        .map(|(lo, hi)| format!("[{lo}, {hi}]"))
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
                ranges
                    .iter()
                    .map(|(lo, hi)| {
                        if *lo as i64 == *hi {
                            lo.to_string()
                        } else {
                            format!("{lo}...{hi}")
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(" × "),
            ),
            ShapeFlexibility::Enumerated(shapes) => (
                "enumeratedShapes",
                format!(
                    "[{}]",
                    shapes
                        .iter()
                        .map(|shape| shape_text(shape))
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
                shapes
                    .iter()
                    .map(|shape| {
                        shape
                            .iter()
                            .map(ToString::to_string)
                            .collect::<Vec<_>>()
                            .join(" × ")
                    })
                    .collect::<Vec<_>>()
                    .join(", "),
            ),
        };
        s.push_str(&format!("{indent}  \"{key}\" : {},\n", json_string(&value)));
        s.push_str(&format!(
            "{indent}  \"shapeFlexibility\" : {},\n",
            json_string(&formatted)
        ));
    }
    s.push_str(&format!("{indent}  \"dataType\" : \"{dt_str}\",\n"));
    s.push_str(&format!(
        "{indent}  \"formattedType\" : \"MultiArray ({dt_str} {formatted_shape})\",\n"
    ));
    s.push_str(&format!(
        "{indent}  \"shortDescription\" : {},\n",
        json_string(feature.map(|f| f.short_description.as_str()).unwrap_or(""))
    ));
    s.push_str(&format!("{indent}  \"shape\" : \"[{shape_str}]\",\n"));
    s.push_str(&format!("{indent}  \"name\" : {},\n", json_string(name)));
    s.push_str(&format!("{indent}  \"type\" : \"MultiArray\"\n"));
    s.push_str(&format!("{indent}}}"));
    s
}

pub(crate) fn json_string(text: &str) -> String {
    let mut result = String::from("\"");
    for ch in text.chars() {
        match ch {
            '"' => result.push_str("\\\""),
            '\\' => result.push_str("\\\\"),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            ch if ch.is_control() => result.push_str(&format!("\\u{:04x}", ch as u32)),
            ch => result.push(ch),
        }
    }
    result.push('"');
    result
}

pub(crate) fn format_data_type_for_meta(dt: MILDataType) -> &'static str {
    match dt {
        MILDataType::Float32 => "Float32",
        MILDataType::Float16 => "Float16",
        MILDataType::Int32 => "Int32",
        MILDataType::Int64 => "Int64",
        MILDataType::Bool => "Bool",
        MILDataType::String => "String",
        _ => dt.text_name(),
    }
}

pub(crate) fn opset_prefix(opset: &str) -> &'static str {
    match opset {
        "CoreML8" => "Ios18",
        "CoreML7" => "Ios17",
        "CoreML6" => "Ios16",
        "CoreML5" => "Ios15",
        _ => "Generic",
    }
}

pub(crate) fn availability_for_spec(spec: i64) -> Vec<(&'static str, &'static str)> {
    // Key order matches Apple's coremlc 3520.x output exactly: macOS, tvOS,
    // visionOS, watchOS, iOS, macCatalyst. JSON wouldn't care about order
    // semantically, but byte-exact tooling comparisons (and our own golden
    // tests) do.
    match spec {
        9 => vec![
            ("macOS", "15.0"),
            ("tvOS", "18.0"),
            ("visionOS", "2.0"),
            ("watchOS", "11.0"),
            ("iOS", "18.0"),
            ("macCatalyst", "18.0"),
        ],
        8 => vec![
            ("macOS", "14.0"),
            ("tvOS", "17.0"),
            ("visionOS", "1.0"),
            ("watchOS", "10.0"),
            ("iOS", "17.0"),
            ("macCatalyst", "17.0"),
        ],
        _ => vec![
            ("macOS", "14.0"),
            ("tvOS", "17.0"),
            ("watchOS", "10.0"),
            ("iOS", "17.0"),
        ],
    }
}

fn find_output_type(name: &str, block: &MILBlock) -> MILType {
    for op in block.operations.iter().rev() {
        for output in &op.outputs {
            if output.name == name {
                return output.r#type.clone();
            }
        }
    }
    MILType::new(MILDataType::Float32, Vec::new())
}

fn encode_function_description(name: &str, function: &MILFunction) -> Vec<u8> {
    let mut bin = Vec::new();
    append_proto_string(&mut bin, 1, name);
    for input in &function.inputs {
        let desc = encode_feature_description(&input.name, &input.r#type);
        append_proto_length_delimited(&mut bin, 2, &desc);
    }
    for output_name in &function.block.outputs {
        let ty = find_output_type(output_name, &function.block);
        let desc = encode_feature_description(output_name, &ty);
        append_proto_length_delimited(&mut bin, 3, &desc);
    }
    bin
}

fn encode_feature_description(name: &str, ty: &MILType) -> Vec<u8> {
    let mut bin = Vec::new();
    append_proto_string(&mut bin, 1, name);
    let feature_type = encode_feature_type(
        &ty.concrete_shape()
            .expect("a synthesized feature description needs a concrete shape"),
        ty.data_type,
    );
    append_proto_length_delimited(&mut bin, 3, &feature_type);
    bin
}

fn encode_feature_type(shape: &[usize], dt: MILDataType) -> Vec<u8> {
    let mut bin = Vec::new();
    let arr = encode_array_feature_type(shape, dt);
    append_proto_length_delimited(&mut bin, 5, &arr);
    bin
}

fn encode_array_feature_type(shape: &[usize], dt: MILDataType) -> Vec<u8> {
    let mut bin = Vec::new();
    let mut shape_bytes = Vec::new();
    for d in shape {
        append_proto_varint(&mut shape_bytes, *d as u64);
    }
    append_proto_length_delimited(&mut bin, 1, &shape_bytes);
    let dt_value: u64 = match dt {
        MILDataType::Float32 => 65568,
        MILDataType::Float16 => 65552,
        MILDataType::Float64 => 65600,
        MILDataType::Int32 => 131072,
        MILDataType::Int64 => 131104,
        MILDataType::Int16 => 131040,
        MILDataType::Int8 => 131024,
        MILDataType::Uint8 => 131136,
        MILDataType::Uint16 => 131152,
        MILDataType::Bool => 131168,
        MILDataType::String => 65696,
    };
    append_proto_varint_field(&mut bin, 2, dt_value);
    bin
}

// === minimal protobuf writer ===

fn append_proto_varint(bin: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        bin.push(((v & 0x7F) | 0x80) as u8);
        v >>= 7;
    }
    bin.push((v & 0x7F) as u8);
}

fn append_proto_tag(bin: &mut Vec<u8>, field: u32, wire_type: u64) {
    append_proto_varint(bin, ((field as u64) << 3) | wire_type);
}

fn append_proto_length_delimited(bin: &mut Vec<u8>, field: u32, value: &[u8]) {
    append_proto_tag(bin, field, 2);
    append_proto_varint(bin, value.len() as u64);
    bin.extend_from_slice(value);
}

fn append_proto_string(bin: &mut Vec<u8>, field: u32, value: &str) {
    append_proto_length_delimited(bin, field, value.as_bytes());
}

fn append_proto_varint_field(bin: &mut Vec<u8>, field: u32, value: u64) {
    append_proto_tag(bin, field, 0);
    append_proto_varint(bin, value);
}

fn append_u32_le(bin: &mut Vec<u8>, value: u32) {
    bin.extend_from_slice(&value.to_le_bytes());
}

fn append_u64_le(bin: &mut Vec<u8>, value: u64) {
    bin.extend_from_slice(&value.to_le_bytes());
}
