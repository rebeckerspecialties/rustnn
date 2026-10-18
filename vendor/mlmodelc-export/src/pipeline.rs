//! CoreML Pipeline containers retain separate MLPrograms and native tensor boundaries.

use std::{collections::BTreeMap, fs, io, path::Path};

use crate::bundle::{
    PipelineBundle, availability_for_spec, format_data_type_for_meta, json_string, opset_prefix,
    schema_list,
};
use crate::description::{FeatureDescription, ModelDescription};
use crate::pb_reader::PBReader;
use crate::{
    CompileStats, Error, MILBinding, MILDataType, MILProgram, MILType, MlmodelcBundle, decode,
    weights,
};

pub(crate) struct Pipeline {
    spec_version: i64,
    description: Vec<u8>,
    names: Vec<String>,
    programs: Vec<MILProgram>,
}

impl Pipeline {
    pub(crate) fn decode(data: &[u8]) -> Result<Option<Self>, Error> {
        let mut reader = PBReader::new(data);
        let mut spec_version = 0;
        let mut description = Vec::new();
        let mut body = None;
        while let Some((field, wire)) = reader.read_tag() {
            match field {
                1 => spec_version = reader.read_varint() as i64,
                2 => description = reader.read_length_delimited().to_vec(),
                202 => body = Some(reader.read_length_delimited()),
                _ => reader.skip(wire),
            }
        }
        let Some(body) = body else { return Ok(None) };
        let mut reader = PBReader::new(body);
        let mut programs = Vec::new();
        let mut names = Vec::new();
        while let Some((field, wire)) = reader.read_tag() {
            match field {
                1 => {
                    let child = reader.read_length_delimited();
                    if Self::has_pipeline(child) {
                        return Err(weights::invalid("nested Pipeline children are unsupported"));
                    }
                    let program = decode(child).map_err(|error| {
                        weights::invalid(format!(
                            "Pipeline child {} must be an MLProgram: {error}",
                            programs.len()
                        ))
                    })?;
                    programs.push(program);
                }
                2 => names.push(reader.read_string()),
                _ => reader.skip(wire),
            }
        }
        if programs.is_empty() {
            return Err(weights::invalid("empty Pipeline"));
        }
        if names.is_empty() {
            names = (0..programs.len()).map(|i| format!("model{i}")).collect();
        }
        if names.len() != programs.len() {
            return Err(weights::invalid(
                "Pipeline names must match its child count",
            ));
        }
        if names.iter().any(String::is_empty)
            || names
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                != names.len()
        {
            return Err(weights::invalid(
                "Pipeline child names must be nonempty and unique",
            ));
        }
        Ok(Some(Self {
            spec_version,
            description,
            names,
            programs,
        }))
    }

    fn has_pipeline(data: &[u8]) -> bool {
        let mut reader = PBReader::new(data);
        while let Some((field, wire)) = reader.read_tag() {
            if matches!(field, 200..=202) {
                return true;
            }
            reader.skip(wire);
        }
        false
    }

    pub(crate) fn references(&self) -> Result<BTreeMap<String, u64>, Error> {
        let mut result = BTreeMap::<String, u64>::new();
        for program in &self.programs {
            for (path, offset) in weights::references(program)? {
                let item = result.entry(path).or_default();
                *item = (*item).max(offset);
            }
        }
        Ok(result)
    }

    pub(crate) fn single_file<'a>(
        &self,
        data: Option<&'a [u8]>,
    ) -> Result<BTreeMap<String, &'a [u8]>, Error> {
        let paths = self.references()?;
        if paths.len() > 1 && data.is_some() {
            return Err(weights::invalid(
                "Pipeline references multiple external weight files; use the named weight-files API",
            ));
        }
        let mut files = BTreeMap::new();
        if let Some(bytes) = data {
            files.insert(
                paths
                    .into_keys()
                    .next()
                    .unwrap_or_else(|| "weights/weights.bin".into()),
                bytes,
            );
        }
        self.validate_files(&files)?;
        Ok(files)
    }

    pub(crate) fn validate_files<T: AsRef<[u8]>>(
        &self,
        files: &BTreeMap<String, T>,
    ) -> Result<(), Error> {
        for path in files.keys() {
            let component = path.split('/').next().unwrap_or_default();
            if component == "modelNames"
                || (0..self.programs.len()).any(|i| component == format!("model{i}"))
            {
                return Err(weights::invalid(format!(
                    "weight path collides with Pipeline layout: {path}"
                )));
            }
        }
        for program in &self.programs {
            weights::validate_files(program, files)?;
        }
        Ok(())
    }

    pub(crate) fn bundle<T: AsRef<[u8]>>(
        &self,
        files: &BTreeMap<String, T>,
    ) -> Result<MlmodelcBundle, Error> {
        self.validate_files(files)?;
        let models = self
            .programs
            .iter()
            .map(|program| {
                let mut sink = crate::MILOutputSink::in_memory(4096);
                crate::emitter::emit_pipeline_child(program, &mut sink);
                Ok((
                    crate::build_bundle(program, sink.finalize()?, None),
                    weights::references(program)?.into_keys().collect(),
                ))
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let mut result = MlmodelcBundle {
            model_mil: Vec::new(),
            coremldata_bin: self.coremldata(),
            metadata_json: self.metadata(),
            analytics_coremldata_bin: self.analytics(),
            weights_bin: None,
            weight_files: BTreeMap::new(),
            pipeline: Some(PipelineBundle {
                model_names_bin: self.model_names(),
                models,
            }),
        };
        for (path, bytes) in files {
            if path == "weights/weights.bin" {
                result.weights_bin = Some(bytes.as_ref().to_vec());
            } else {
                result
                    .weight_files
                    .insert(path.clone(), bytes.as_ref().to_vec());
            }
        }
        Ok(result)
    }

    pub(crate) fn write<T: AsRef<[u8]>>(
        &self,
        input_bytes: usize,
        files: &BTreeMap<String, T>,
        dir: &Path,
    ) -> Result<CompileStats, Error> {
        self.validate_files(files)?;
        let empty = BTreeMap::<String, &[u8]>::new();
        let layout = PipelineBundle {
            model_names_bin: self.model_names(),
            models: self
                .programs
                .iter()
                .map(|p| {
                    Ok((
                        crate::build_bundle(p, Vec::new(), None),
                        weights::references(p)?.into_keys().collect(),
                    ))
                })
                .collect::<Result<_, Error>>()?,
        };
        weights::prepare_directory(dir, files.keys())?;
        prepare_children(dir, &layout)?;
        fs::write(dir.join("coremldata.bin"), self.coremldata())?;
        fs::write(dir.join("metadata.json"), self.metadata())?;
        fs::create_dir_all(dir.join("analytics"))?;
        fs::write(dir.join("analytics/coremldata.bin"), self.analytics())?;
        fs::create_dir_all(dir.join("modelNames"))?;
        fs::write(dir.join("modelNames/coremldata.bin"), self.model_names())?;
        weights::write_files(dir, files)?;
        let mut result = CompileStats {
            input_bytes,
            output_bytes: 0,
            operation_count: 0,
            const_count: 0,
            largest_const_elements: 0,
        };
        for (index, program) in self.programs.iter().enumerate() {
            let child = dir.join(format!("model{index}"));
            let stats = crate::compile_program_to_dir(&[], program, &empty, &child, true)?;
            result.output_bytes += stats.output_bytes;
            result.operation_count += stats.operation_count;
            result.const_count += stats.const_count;
            result.largest_const_elements = result
                .largest_const_elements
                .max(stats.largest_const_elements);
            remove_child_weights(&child, &layout.models[index].1)?;
        }
        Ok(result)
    }

    fn coremldata(&self) -> Vec<u8> {
        let mut trailer = self.description.clone();
        if !ModelDescription::decode(&trailer).has_metadata {
            trailer.extend([0xa2, 0x06, 0]);
        }
        let mut target = String::new();
        for (i, program) in self.programs.iter().enumerate() {
            target.push_str(&format!("generic.model{i}v{}.0.0", program.spec_version));
        }
        target.push_str("generic");
        let mut bin = Vec::new();
        bin.extend(202u32.to_le_bytes());
        bin.extend(1u32.to_le_bytes());
        bin.extend([0u8; 12]);
        bin.extend(1u32.to_le_bytes());
        bin.extend(0u32.to_le_bytes());
        bin.extend((target.len() as u64).to_le_bytes());
        bin.extend(target.as_bytes());
        bin.push(self.spec_version as u8);
        bin.extend([0u8; 31]);
        bin.extend((trailer.len() as u64).to_le_bytes());
        bin.extend(trailer);
        bin.extend((self.programs.len() as u64).to_le_bytes());
        bin.extend(202u32.to_le_bytes());
        bin.extend(0u32.to_le_bytes());
        bin
    }

    fn model_names(&self) -> Vec<u8> {
        let mut bin = (self.names.len() as u64).to_le_bytes().to_vec();
        for name in &self.names {
            bin.extend((name.len() as u64).to_le_bytes());
            bin.extend(name.as_bytes());
        }
        bin
    }

    fn analytics(&self) -> Vec<u8> {
        let mut bin = 20u64.to_le_bytes().to_vec();
        bin.extend(b"PipelineModelDetails");
        bin.extend(1u64.to_le_bytes());
        bin.extend(14u64.to_le_bytes());
        bin.extend(b"modelDimension");
        let count = self.programs.len().to_string();
        bin.extend((count.len() as u64).to_le_bytes());
        bin.extend(count.as_bytes());
        bin
    }

    fn metadata(&self) -> Vec<u8> {
        let description = ModelDescription::decode(&self.description);
        let types = |features: &[FeatureDescription]| {
            features
                .iter()
                .map(|feature| {
                    let ty = self
                        .programs
                        .iter()
                        .flat_map(|p| &p.functions)
                        .flat_map(|(_, f)| {
                            f.inputs
                                .iter()
                                .chain(f.block.operations.iter().flat_map(|op| &op.outputs))
                        })
                        .find(|ty| ty.name == feature.name)
                        .map(|ty| ty.r#type.clone())
                        .unwrap_or_else(|| MILType::new(MILDataType::Float32, Vec::new()));
                    (feature.name.clone(), ty)
                })
                .collect::<Vec<_>>()
        };
        let mut hist = BTreeMap::<String, usize>::new();
        let mut precisions = std::collections::BTreeSet::new();
        let mut storage = std::collections::BTreeSet::new();
        for program in &self.programs {
            for (_, function) in &program.functions {
                for op in function.block.all_operations() {
                    if op.r#type != "const" {
                        let name = match op.r#type.as_str() {
                            "select" => "Select".to_owned(),
                            "reduce_argmax" => {
                                format!("{}.reduceArgmax", opset_prefix(&function.opset))
                            }
                            "reduce_argmin" => {
                                format!("{}.reduceArgmin", opset_prefix(&function.opset))
                            }
                            other => format!("{}.{}", opset_prefix(&function.opset), other),
                        };
                        *hist.entry(name).or_default() += 1;
                    }
                    let inputs = op
                        .inputs
                        .iter()
                        .flat_map(|(_, bindings)| bindings)
                        .filter_map(|binding| match binding {
                            MILBinding::Immediate(value) => Some(value),
                            _ => None,
                        });
                    for value in op.attributes.iter().map(|(_, value)| value).chain(inputs) {
                        if value.blob.is_some() {
                            storage.insert(format_data_type_for_meta(value.r#type.data_type));
                        }
                    }
                    for output in &op.outputs {
                        if op.r#type != "const"
                            && let Some(precision) = numeric_precision(output.r#type.data_type)
                        {
                            precisions.insert(precision);
                        }
                    }
                }
            }
        }
        let precision = match precisions.len() {
            0 => "Float32".into(),
            1 => precisions.into_iter().next().unwrap().into(),
            _ => format!(
                "Mixed ({})",
                precisions.into_iter().collect::<Vec<_>>().join(", ")
            ),
        };
        let hist = hist
            .iter()
            .map(|(name, count)| format!("\"{name}\":{count}"))
            .collect::<Vec<_>>()
            .join(",");
        let storage = match storage.len() {
            0 => String::new(),
            1 => format!(
                "\"storagePrecision\":\"{}\",",
                storage.into_iter().next().unwrap()
            ),
            _ => format!(
                "\"storagePrecision\":\"Mixed ({})\",",
                storage.into_iter().collect::<Vec<_>>().join(", ")
            ),
        };
        let avail = availability_for_spec(self.spec_version)
            .iter()
            .map(|(key, value)| format!("\"{key}\":\"{value}\""))
            .collect::<Vec<_>>()
            .join(",");
        let models =
            std::iter::repeat_n("{\"name\":\"MLModelType_mlProgram\"}", self.programs.len())
                .collect::<Vec<_>>()
                .join(",");
        let user_metadata = description
            .user_defined_metadata
            .iter()
            .map(|(key, value)| format!("{}:{}", json_string(key), json_string(value)))
            .collect::<Vec<_>>()
            .join(",");
        format!("[{{{storage}\"metadataOutputVersion\":\"3.0\",\"outputSchema\":{},\"modelParameters\":[],\"specificationVersion\":{},\"mlProgramOperationTypeHistogram\":{{{hist}}},\"computePrecision\":\"{precision}\",\"isUpdatable\":\"0\",\"stateSchema\":[],\"availability\":{{{avail}}},\"modelType\":{{\"name\":\"MLModelType_pipeline\",\"structure\":[{models}]}},\"userDefinedMetadata\":{{{user_metadata}}},\"inputSchema\":{},\"generatedClassName\":\"model\",\"method\":\"predict\"}}]",schema_list(&types(&description.main.outputs),Some(&description.main.outputs),""),self.spec_version,schema_list(&types(&description.main.inputs),Some(&description.main.inputs),"")).into_bytes()
    }
}

fn numeric_precision(dtype: MILDataType) -> Option<&'static str> {
    Some(match dtype {
        MILDataType::Float16 => "Float16",
        MILDataType::Float32 => "Float32",
        MILDataType::Float64 => "Double",
        MILDataType::Int8 => "Int8",
        MILDataType::Int16 => "Int16",
        MILDataType::Int32 => "Int32",
        MILDataType::Int64 => "Int64",
        MILDataType::Uint8 => "UInt8",
        MILDataType::Uint16 => "UInt16",
        _ => return None,
    })
}

pub(crate) fn prepare_children(dir: &Path, pipeline: &PipelineBundle) -> io::Result<()> {
    let mut paths = vec!["modelNames/coremldata.bin".to_owned()];
    for (i, (_, weights)) in pipeline.models.iter().enumerate() {
        for file in [
            "model.mil",
            "metadata.json",
            "coremldata.bin",
            "analytics/coremldata.bin",
        ] {
            paths.push(format!("model{i}/{file}"));
        }
        for file in weights {
            crate::weights::validate_path(file).map_err(io::Error::other)?;
            paths.push(format!("model{i}/{file}"));
        }
    }
    weights::prepare_directory(dir, paths.iter())
}

pub(crate) fn remove_child_weights(child: &Path, paths: &[String]) -> io::Result<()> {
    // A re-export may overwrite an older bundle that linked assets into each
    // child. These validated derived paths are now obsolete; retain only root
    // assets so ordinary directory copies cannot multiply the weight storage.
    for path in paths {
        let target = child.join(path);
        if target.exists() {
            fs::remove_file(&target)?;
        }
    }
    Ok(())
}
