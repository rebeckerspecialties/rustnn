//! Referenced external assets, shared by buffered and streaming compilation.

use std::{collections::BTreeMap, fs, io, path::Path};

use crate::{Error, MILBinding, MILProgram};

pub(crate) fn invalid(message: impl Into<String>) -> Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into()).into()
}

fn relative_path(filename: &str) -> Result<&str, Error> {
    let path = filename.strip_prefix("@model_path/").ok_or_else(|| {
        invalid(format!(
            "external weight path must start with @model_path/: {filename:?}"
        ))
    })?;
    validate_path(path)?;
    Ok(path)
}

pub(crate) fn validate_path(path: &str) -> Result<(), Error> {
    // Use portable path rules even on Unix: bundles can be produced on one OS
    // and consumed on another. Reject characters unsafe in MIL string literals.
    if path
        .split('/')
        .any(|part| part.is_empty() || part == "." || part == "..")
        || path
            .chars()
            .any(|ch| ch.is_control() || matches!(ch, '\\' | ':' | '"'))
        || matches!(
            path.split('/').next(),
            Some("model.mil" | "metadata.json" | "coremldata.bin" | "analytics")
        )
    {
        return Err(invalid(format!("invalid external weight path: {path:?}")));
    }
    Ok(())
}

pub(crate) fn references(program: &MILProgram) -> Result<BTreeMap<String, u64>, Error> {
    let mut paths = BTreeMap::<String, u64>::new();
    for (_, function) in &program.functions {
        for operation in function.block.all_operations() {
            let inputs = operation
                .inputs
                .iter()
                .flat_map(|(_, bindings)| bindings)
                .filter_map(|binding| match binding {
                    MILBinding::Immediate(value) => Some(value),
                    _ => None,
                });
            for value in operation
                .attributes
                .iter()
                .map(|(_, value)| value)
                .chain(inputs)
            {
                if let Some(blob) = &value.blob {
                    let path = relative_path(&blob.filename)?;
                    let offset = paths.entry(path.to_owned()).or_default();
                    *offset = (*offset).max(blob.offset);
                }
            }
        }
    }
    Ok(paths)
}

pub(crate) fn validate_files<T: AsRef<[u8]>>(
    program: &MILProgram,
    files: &BTreeMap<String, T>,
) -> Result<(), Error> {
    for path in files.keys() {
        validate_path(path)?;
        for (index, _) in path.match_indices('/') {
            if files.contains_key(&path[..index]) {
                return Err(invalid(format!(
                    "external weight path is both a file and directory: {}",
                    &path[..index]
                )));
            }
        }
    }
    for (path, offset) in references(program)? {
        let bytes = files
            .get(&path)
            .ok_or_else(|| invalid(format!("missing external weight file: {path}")))?
            .as_ref();
        if offset >= bytes.len() as u64 {
            return Err(invalid(format!(
                "external weight offset {offset} is outside {path} ({} bytes)",
                bytes.len()
            )));
        }
    }
    Ok(())
}

pub(crate) fn single_file<'a>(
    program: &MILProgram,
    weights: Option<&'a [u8]>,
) -> Result<BTreeMap<String, &'a [u8]>, Error> {
    let paths = references(program)?;
    if paths.len() > 1 && weights.is_some() {
        return Err(invalid(
            "model references multiple external weight files; use the named weight-files API",
        ));
    }
    let mut files = BTreeMap::new();
    if let Some(bytes) = weights {
        let path = paths
            .into_keys()
            .next()
            .unwrap_or_else(|| "weights/weights.bin".into());
        files.insert(path, bytes);
    }
    validate_files(program, &files)?;
    Ok(files)
}

pub(crate) fn prepare_directory<'a>(
    directory: &Path,
    weights: impl Iterator<Item = &'a String>,
) -> io::Result<()> {
    let mut paths = vec![
        "model.mil",
        "coremldata.bin",
        "metadata.json",
        "analytics/coremldata.bin",
    ];
    paths.extend(weights.map(String::as_str));
    // Do not follow existing output symlinks when overwriting a bundle. Check
    // every destination before writing any bundle data.
    for relative in paths {
        let mut current = directory.to_path_buf();
        reject_symlink(&current)?;
        for component in relative.split('/') {
            current.push(component);
            reject_symlink(&current)?;
        }
    }
    fs::create_dir_all(directory)
}

fn reject_symlink(path: &Path) -> io::Result<()> {
    match fs::symlink_metadata(path) {
        Ok(meta) if meta.file_type().is_symlink() => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("bundle output contains a symlink: {}", path.display()),
        )),
        Ok(_) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

pub(crate) fn write_files<T: AsRef<[u8]>>(
    directory: &Path,
    files: &BTreeMap<String, T>,
) -> io::Result<()> {
    for (path, bytes) in files {
        let destination = directory.join(path);
        fs::create_dir_all(destination.parent().expect("weight file has a parent"))?;
        fs::write(destination, bytes)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collects_attributes_and_immediate_bindings_in_every_function() {
        let mut program = crate::decode(include_bytes!(
            "../tests/fixtures/flexible-weighted-legacy/input.mlmodel"
        ))
        .unwrap();
        let mut second = program.functions[0].1.clone();
        let mut value = second.block.operations[0].attributes.remove(0).1;
        value.blob.as_mut().unwrap().filename = "@model_path/weights/second.bin".into();
        second.block.operations[0]
            .inputs
            .push(("value".into(), vec![MILBinding::Immediate(value)]));
        program.functions.push(("second".into(), second));
        assert_eq!(
            references(&program).unwrap(),
            BTreeMap::from([
                ("weights/weights.bin".into(), 64),
                ("weights/second.bin".into(), 64)
            ])
        );
    }
}
