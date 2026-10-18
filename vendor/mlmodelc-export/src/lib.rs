//! Compile a CoreML MLProgram protobuf into a runnable `.mlmodelc` bundle in
//! pure Rust, without invoking Apple's `coremlc`.
//!
//! `coremlc` is the Apple tool that turns a `.mlmodel` (or `.mlpackage`)
//! protobuf into the directory layout that `MLModel(contentsOfURL:)` actually
//! loads at runtime. It exists only on macOS, only inside Xcode's toolchain,
//! and is not callable from sandboxed environments — watchOS apps, iOS App
//! Extensions, App Clips, tvOS sandbox, or any non-Apple host.
//!
//! This crate replaces `coremlc` for the MLProgram path with a pure-Rust
//! emitter. It accepts the protobuf bytes that you'd otherwise hand to
//! `coremlc` and produces the same four files Apple's compiler emits:
//!
//! ```text
//! example.mlmodelc/
//!   model.mil
//!   coremldata.bin
//!   metadata.json
//!   analytics/coremldata.bin
//!   weights/weights.bin   (optional, when the input has external weights)
//! ```
//!
//! ## Quick start
//!
//! ```no_run
//! use mlmodelc_export::compile_to_bundle;
//!
//! let mlmodel_bytes = std::fs::read("model.mlmodel")?;
//! let bundle = compile_to_bundle(&mlmodel_bytes, None)?;
//! bundle.write_to_dir("model.mlmodelc")?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Provenance
//!
//! Originated as a Swift package (`MILTextCompiler`) inside the
//! `metal-info-app` watchOS validation harness for [rustnn]. Ported to Rust
//! to align with rustnn's Rust-first architecture and so the output of
//! `CoremlMlProgramConverter` can be consumed end-to-end without leaving
//! the Cargo ecosystem.
//!
//! [rustnn]: https://github.com/rustnn/rustnn

mod bundle;
mod decoder;
mod description;
mod emitter;
mod hex_float;
mod pb_reader;
mod pipeline;
mod sink;
mod types;
mod weights;

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

pub use bundle::{
    MlmodelcBundle, PipelineBundle, build_bundle, generate_analytics_bin, generate_coremldata_bin,
    generate_metadata_json,
};
pub use decoder::{MILDecoderError, decode};
pub use emitter::{emit, emit_to_string};
pub use hex_float::{hex_float16, hex_float32, hex_float32_bytes};
pub use sink::MILOutputSink;
pub use types::{
    MILBinding, MILBlobRef, MILBlock, MILDataType, MILDimension, MILFunction, MILNamedType,
    MILOperation, MILProgram, MILTensorData, MILType, MILValue,
};

/// Statistics computed from a single compile pass.
#[derive(Debug, Clone)]
pub struct CompileStats {
    pub input_bytes: usize,
    pub output_bytes: usize,
    pub operation_count: usize,
    pub const_count: usize,
    pub largest_const_elements: usize,
}

/// Result of an in-memory compile.
#[derive(Debug, Clone)]
pub struct CompileResult {
    pub mil_text: String,
    pub stats: CompileStats,
}

/// Errors that can be returned by the public API.
#[derive(Debug)]
pub enum Error {
    Decode(MILDecoderError),
    Io(std::io::Error),
}

impl From<MILDecoderError> for Error {
    fn from(e: MILDecoderError) -> Self {
        Error::Decode(e)
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Decode(e) => write!(f, "decode error: {e}"),
            Self::Io(e) => write!(f, "io error: {e}"),
        }
    }
}

impl std::error::Error for Error {}

/// Compile a CoreML protobuf into MIL text only (no bundle assembly).
///
/// Useful for debugging or for callers that want to handle the bundle layout
/// themselves. Most users want [`compile_to_bundle`] instead.
pub fn compile_to_text(protobuf: &[u8]) -> Result<CompileResult, Error> {
    let program = decode(protobuf)?;
    let mil_text = emit_to_string(&program);
    let stats = compute_stats(protobuf, &mil_text, &program);
    Ok(CompileResult { mil_text, stats })
}

/// Compile an MLProgram or a flat Pipeline of MLPrograms into a `.mlmodelc` bundle.
///
/// `weights` supplies the model's single referenced external file, preserving
/// its filename. Pass `None` for inline-constant models. Missing assets fail
/// explicitly. Use [`compile_to_bundle_with_weight_files`] for multiple files.
pub fn compile_to_bundle(protobuf: &[u8], weights: Option<&[u8]>) -> Result<MlmodelcBundle, Error> {
    if let Some(pipeline) = pipeline::Pipeline::decode(protobuf)? {
        let files = pipeline.single_file(weights)?;
        return pipeline.bundle(&files);
    }
    let program = decode(protobuf)?;
    let files = weights::single_file(&program, weights)?;
    Ok(bundle_with_files(&program, &files))
}

/// The sorted, unique external weight paths referenced by a model, relative
/// to `@model_path/`. Invalid or nonportable paths are rejected.
pub fn referenced_weight_paths(protobuf: &[u8]) -> Result<Vec<String>, Error> {
    if let Some(pipeline) = pipeline::Pipeline::decode(protobuf)? {
        return Ok(pipeline.references()?.into_keys().collect());
    }
    Ok(weights::references(&decode(protobuf)?)?
        .into_keys()
        .collect())
}

/// Compile with named external files. Keys are bundle-relative paths returned
/// by [`referenced_weight_paths`], such as `weights/weight.bin`. File bytes,
/// names and reference offsets are preserved; all references must resolve.
pub fn compile_to_bundle_with_weight_files(
    protobuf: &[u8],
    files: &BTreeMap<String, Vec<u8>>,
) -> Result<MlmodelcBundle, Error> {
    if let Some(pipeline) = pipeline::Pipeline::decode(protobuf)? {
        return pipeline.bundle(files);
    }
    let program = decode(protobuf)?;
    weights::validate_files(&program, files)?;
    Ok(bundle_with_files(&program, files))
}

fn bundle_with_files<T: AsRef<[u8]>>(
    program: &MILProgram,
    files: &BTreeMap<String, T>,
) -> MlmodelcBundle {
    let mut bundle = build_bundle(program, emit_to_string(program).into_bytes(), None);
    for (path, bytes) in files {
        if path == "weights/weights.bin" {
            bundle.weights_bin = Some(bytes.as_ref().to_vec());
        } else {
            bundle
                .weight_files
                .insert(path.clone(), bytes.as_ref().to_vec());
        }
    }
    bundle
}

/// Streaming variant of [`compile_to_bundle`]: write the bundle directly to
/// `directory` without materialising the full MIL text in memory. Recommended
/// for large models on memory-constrained devices.
/// Pipeline stages reference one shared root asset through generated internal
/// relative paths, so directory copies do not duplicate weights per child.
pub fn compile_to_dir(
    protobuf: &[u8],
    weights: Option<&[u8]>,
    directory: impl AsRef<Path>,
) -> Result<CompileStats, Error> {
    if let Some(pipeline) = pipeline::Pipeline::decode(protobuf)? {
        let files = pipeline.single_file(weights)?;
        return pipeline.write(protobuf.len(), &files, directory.as_ref());
    }
    let program = decode(protobuf)?;
    let files = weights::single_file(&program, weights)?;
    compile_program_to_dir(protobuf, &program, &files, directory.as_ref(), false)
}

/// Streaming MIL emission with named external files. Like [`compile_to_dir`],
/// weight data is borrowed, not cloned. References are validated before writing.
/// Values may be slices, memory mappings, or owned buffers; the exporter does
/// not require a model-sized `Vec<u8>` allocation for mapped assets.
pub fn compile_to_dir_with_weight_files<T: AsRef<[u8]>>(
    protobuf: &[u8],
    files: &BTreeMap<String, T>,
    directory: impl AsRef<Path>,
) -> Result<CompileStats, Error> {
    if let Some(pipeline) = pipeline::Pipeline::decode(protobuf)? {
        return pipeline.write(protobuf.len(), files, directory.as_ref());
    }
    let program = decode(protobuf)?;
    weights::validate_files(&program, files)?;
    compile_program_to_dir(protobuf, &program, files, directory.as_ref(), false)
}

fn compile_program_to_dir<T: AsRef<[u8]>>(
    protobuf: &[u8],
    program: &MILProgram,
    files: &BTreeMap<String, T>,
    dir: &Path,
    pipeline_child: bool,
) -> Result<CompileStats, Error> {
    weights::prepare_directory(dir, files.keys())?;

    let mil_path = dir.join("model.mil");
    if mil_path.exists() {
        fs::remove_file(&mil_path)?;
    }
    let file = fs::File::create(&mil_path)?;
    let mut sink = MILOutputSink::streaming(file);
    if pipeline_child {
        emitter::emit_pipeline_child(program, &mut sink);
    } else {
        emit(program, &mut sink);
    }
    let bytes_written = sink.bytes_written();
    let _ = sink.finalize()?;

    fs::write(dir.join("coremldata.bin"), generate_coremldata_bin(program))?;
    fs::write(dir.join("metadata.json"), generate_metadata_json(program))?;

    let analytics_dir = dir.join("analytics");
    fs::create_dir_all(&analytics_dir)?;
    fs::write(
        analytics_dir.join("coremldata.bin"),
        generate_analytics_bin(),
    )?;

    weights::write_files(dir, files)?;

    Ok(CompileStats {
        input_bytes: protobuf.len(),
        output_bytes: bytes_written,
        operation_count: count_operations(program),
        const_count: count_consts(program),
        largest_const_elements: largest_const(program),
    })
}

fn compute_stats(protobuf: &[u8], mil_text: &str, program: &MILProgram) -> CompileStats {
    CompileStats {
        input_bytes: protobuf.len(),
        output_bytes: mil_text.len(),
        operation_count: count_operations(program),
        const_count: count_consts(program),
        largest_const_elements: largest_const(program),
    }
}

fn count_operations(program: &MILProgram) -> usize {
    program
        .functions
        .iter()
        .map(|(_, f)| f.block.all_operations().count())
        .sum()
}

fn count_consts(program: &MILProgram) -> usize {
    program
        .functions
        .iter()
        .flat_map(|(_, f)| f.block.all_operations())
        .filter(|op| op.r#type == "const")
        .count()
}

fn largest_const(program: &MILProgram) -> usize {
    let mut largest = 0;
    for (_, f) in &program.functions {
        for op in f.block.all_operations() {
            if op.r#type == "const" {
                for (k, v) in &op.attributes {
                    if k == "val" {
                        largest = largest.max(v.element_count());
                    }
                }
            }
        }
    }
    largest
}
