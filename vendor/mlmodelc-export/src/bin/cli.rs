//! `mlmodelc-export` CLI — compile a `.mlmodel` (or `.mlpackage`) into a
//! `.mlmodelc` bundle without Apple's `coremlc`.

use std::{collections::BTreeMap, fs, io, path::PathBuf};

use clap::Parser;
use mlmodelc_export::{compile_to_dir, compile_to_dir_with_weight_files, referenced_weight_paths};

#[derive(Parser, Debug)]
#[command(name = "mlmodelc-export")]
#[command(about = "Compile a CoreML MLProgram into a .mlmodelc bundle in pure Rust", long_about = None)]
#[command(version)]
struct Args {
    /// Input `.mlmodel` (raw protobuf) or `.mlpackage` directory containing
    /// `Data/com.apple.CoreML/model.mlmodel`.
    input: PathBuf,

    /// Output `.mlmodelc` directory. Created if missing; existing files
    /// inside are overwritten.
    output: PathBuf,

    /// Supply the model's single external weight file explicitly. Its referenced
    /// filename is preserved. Otherwise all referenced files are read relative
    /// to the model inside the package, or alongside a raw `.mlmodel`.
    #[arg(long)]
    weights: Option<PathBuf>,
}

fn main() {
    let args = Args::parse();
    if let Err(e) = run(args) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    let model = if args.input.is_dir() {
        args.input.join("Data/com.apple.CoreML/model.mlmodel")
    } else {
        args.input
    };
    let model_bytes = fs::read(&model)?;
    let stats = if let Some(path) = args.weights {
        compile_to_dir(&model_bytes, Some(&fs::read(path)?), &args.output)?
    } else {
        let root = model
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or_else(|| std::path::Path::new("."))
            .canonicalize()?;
        let mut files = BTreeMap::new();
        for relative in referenced_weight_paths(&model_bytes)? {
            let source = root.join(&relative).canonicalize().map_err(|error| {
                io::Error::new(error.kind(), format!("external weight {relative}: {error}"))
            })?;
            if !source.starts_with(&root) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("external weight escapes model directory: {relative}"),
                )
                .into());
            }
            files.insert(relative, fs::read(source)?);
        }
        compile_to_dir_with_weight_files(&model_bytes, &files, &args.output)?
    };
    eprintln!(
        "wrote {} ({} ops, {} consts, MIL {} bytes)",
        args.output.display(),
        stats.operation_count,
        stats.const_count,
        stats.output_bytes
    );
    Ok(())
}
