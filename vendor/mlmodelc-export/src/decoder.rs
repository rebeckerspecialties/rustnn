//! Decode CoreML protobuf wire-format into a [`MILProgram`].
//!
//! Field numbers follow `MIL.proto` from apple/coremltools. We only walk the
//! message graph needed to reconstruct the MIL text and the function metadata
//! that goes into `coremldata.bin` / `metadata.json`.

use crate::description::{ModelDescription, ShapeFlexibility};
use crate::pb_reader::{PBReader, read_packed_floats, read_packed_signed_varints};
use crate::types::*;

/// Errors that can be produced by the decoder.
#[derive(Debug, Clone)]
pub enum MILDecoderError {
    MissingProgram,
    MissingFunction,
    MissingBlock,
    UnknownDataType(u64),
    UnsupportedFormat(String),
}

impl std::fmt::Display for MILDecoderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingProgram => f.write_str("Model.mlProgram (field 502) not present"),
            Self::MissingFunction => f.write_str("Program is missing functions"),
            Self::MissingBlock => f.write_str("Function is missing block_specializations"),
            Self::UnknownDataType(raw) => write!(f, "unknown MIL DataType raw value: {raw}"),
            Self::UnsupportedFormat(s) => write!(f, "unsupported format: {s}"),
        }
    }
}

impl std::error::Error for MILDecoderError {}

/// Decode a full CoreML `Model` protobuf into a [`MILProgram`].
pub fn decode(data: &[u8]) -> Result<MILProgram, MILDecoderError> {
    let mut reader = PBReader::new(data);
    let mut spec_version: i64 = 0;
    let mut program_data: Option<Vec<u8>> = None;
    let mut description_data: Vec<u8> = Vec::new();

    while let Some((field, wire)) = reader.read_tag() {
        match field {
            1 => spec_version = reader.read_varint() as i64,
            2 => description_data = reader.read_length_delimited().to_vec(),
            502 => program_data = Some(reader.read_length_delimited().to_vec()),
            _ => reader.skip(wire),
        }
    }

    let program_data = program_data.ok_or(MILDecoderError::MissingProgram)?;
    let functions = decode_program(&program_data)?;
    let description = ModelDescription::decode(&description_data);
    for function in std::iter::once(&description.main).chain(&description.functions) {
        if function
            .inputs
            .iter()
            .any(|feature| matches!(feature.flexibility, Some(ShapeFlexibility::Enumerated(_))))
        {
            return Err(MILDecoderError::UnsupportedFormat(
                "enumerated input shapes; only ranged flexibility is supported".into(),
            ));
        }
    }
    Ok(MILProgram {
        version: 1,
        functions,
        spec_version,
        description_data,
    })
}

fn decode_program(data: &[u8]) -> Result<Vec<(String, MILFunction)>, MILDecoderError> {
    let mut reader = PBReader::new(data);
    let mut functions = Vec::new();
    while let Some((field, wire)) = reader.read_tag() {
        match field {
            1 => {
                // Program.version (uint64, always 1).
                let _ = reader.read_varint();
            }
            2 => {
                let entry = reader.read_length_delimited();
                let (key, value) = decode_map_entry(entry)?;
                functions.push((key, decode_function(&value)?));
            }
            _ => reader.skip(wire),
        }
    }
    Ok(functions)
}

fn decode_function(data: &[u8]) -> Result<MILFunction, MILDecoderError> {
    let mut reader = PBReader::new(data);
    let mut inputs = Vec::new();
    let mut opset = String::new();
    let mut blocks: Vec<(String, MILBlock)> = Vec::new();

    while let Some((field, wire)) = reader.read_tag() {
        match field {
            1 => inputs.push(decode_named_value_type(reader.read_length_delimited())?),
            2 => opset = reader.read_string(),
            3 => {
                let entry = reader.read_length_delimited();
                let (key, value) = decode_map_entry(entry)?;
                blocks.push((key, decode_block(&value)?));
            }
            _ => reader.skip(wire),
        }
    }

    let block = blocks
        .iter()
        .find(|(k, _)| *k == opset)
        .map(|(_, b)| b.clone())
        .or_else(|| blocks.first().map(|(_, b)| b.clone()))
        .ok_or(MILDecoderError::MissingBlock)?;

    Ok(MILFunction {
        inputs,
        opset,
        block,
    })
}

fn decode_block(data: &[u8]) -> Result<MILBlock, MILDecoderError> {
    decode_block_at(data, 0)
}

fn decode_block_at(data: &[u8], depth: usize) -> Result<MILBlock, MILDecoderError> {
    if depth > 64 {
        return Err(MILDecoderError::UnsupportedFormat(
            "MIL control flow exceeds 64 nested blocks".into(),
        ));
    }
    let mut reader = PBReader::new(data);
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    let mut operations = Vec::new();
    while let Some((field, wire)) = reader.read_tag() {
        match field {
            1 => inputs.push(decode_named_value_type(reader.read_length_delimited())?),
            2 => outputs.push(reader.read_string()),
            3 => operations.push(decode_operation(reader.read_length_delimited(), depth)?),
            _ => reader.skip(wire),
        }
    }
    Ok(MILBlock {
        inputs,
        outputs,
        operations,
    })
}

fn decode_operation(data: &[u8], depth: usize) -> Result<MILOperation, MILDecoderError> {
    let mut reader = PBReader::new(data);
    let mut r#type = String::new();
    let mut inputs: Vec<(String, Vec<MILBinding>)> = Vec::new();
    let mut outputs: Vec<MILNamedType> = Vec::new();
    let mut attributes: Vec<(String, MILValue)> = Vec::new();
    let mut blocks = Vec::new();

    while let Some((field, wire)) = reader.read_tag() {
        match field {
            1 => r#type = reader.read_string(),
            2 => {
                let entry = reader.read_length_delimited();
                let (key, value) = decode_map_entry(entry)?;
                inputs.push((key, decode_argument(&value)?));
            }
            3 => outputs.push(decode_named_value_type(reader.read_length_delimited())?),
            4 => blocks.push(decode_block_at(reader.read_length_delimited(), depth + 1)?),
            5 => {
                let entry = reader.read_length_delimited();
                let (key, value) = decode_map_entry(entry)?;
                attributes.push((key, decode_value(&value)?));
            }
            _ => reader.skip(wire),
        }
    }

    Ok(MILOperation {
        r#type,
        blocks,
        inputs,
        outputs,
        attributes,
    })
}

fn decode_argument(data: &[u8]) -> Result<Vec<MILBinding>, MILDecoderError> {
    let mut reader = PBReader::new(data);
    let mut bindings = Vec::new();
    while let Some((field, wire)) = reader.read_tag() {
        match field {
            1 => bindings.push(decode_binding(reader.read_length_delimited())?),
            _ => reader.skip(wire),
        }
    }
    Ok(bindings)
}

fn decode_binding(data: &[u8]) -> Result<MILBinding, MILDecoderError> {
    let mut reader = PBReader::new(data);
    let mut name: Option<String> = None;
    let mut value: Option<MILValue> = None;
    while let Some((field, wire)) = reader.read_tag() {
        match field {
            1 => name = Some(reader.read_string()),
            2 => value = Some(decode_value(reader.read_length_delimited())?),
            _ => reader.skip(wire),
        }
    }
    if let Some(n) = name {
        return Ok(MILBinding::Reference(n));
    }
    if let Some(v) = value {
        return Ok(MILBinding::Immediate(v));
    }
    Ok(MILBinding::Reference(String::new()))
}

fn decode_value(data: &[u8]) -> Result<MILValue, MILDecoderError> {
    let mut reader = PBReader::new(data);
    let mut r#type = MILType::new(MILDataType::Float32, Vec::new());
    let mut tensor = MILTensorData::default();
    let mut blob: Option<MILBlobRef> = None;

    while let Some((field, wire)) = reader.read_tag() {
        match field {
            2 => r#type = decode_value_type(reader.read_length_delimited())?,
            3 => tensor = decode_immediate_value(reader.read_length_delimited())?,
            5 => blob = Some(decode_blob_file_value(reader.read_length_delimited())?),
            _ => reader.skip(wire),
        }
    }

    let Some(shape) = r#type.concrete_shape() else {
        return Err(MILDecoderError::UnsupportedFormat(
            "immediate/blob values must have concrete dimensions".into(),
        ));
    };
    // TensorValue.bytes is shared by FP16 and 8-bit integers. Its storage
    // width and interpretation come from Value.type, not from the wire field.
    if blob.is_none()
        && let MILTensorData::Fp16Bytes(bytes) = tensor
    {
        let width: usize = match r#type.data_type {
            MILDataType::Float16 => 2,
            MILDataType::Int8 | MILDataType::Uint8 => 1,
            dtype => {
                return Err(MILDecoderError::UnsupportedFormat(format!(
                    "byte tensor storage for {}",
                    dtype.text_name()
                )));
            }
        };
        let expected_bytes = shape
            .iter()
            .try_fold(width, |length, &size| length.checked_mul(size))
            .ok_or_else(|| {
                MILDecoderError::UnsupportedFormat("byte tensor size exceeds platform size".into())
            })?;
        if bytes.len() != expected_bytes {
            return Err(MILDecoderError::UnsupportedFormat(format!(
                "{} byte tensor needs {expected_bytes} bytes, got {}",
                r#type.data_type.text_name(),
                bytes.len()
            )));
        }
        tensor = match r#type.data_type {
            MILDataType::Int8 => {
                MILTensorData::Ints(bytes.into_iter().map(|byte| byte as i8 as i32).collect())
            }
            MILDataType::Uint8 => MILTensorData::Ints(bytes.into_iter().map(i32::from).collect()),
            _ => MILTensorData::Fp16Bytes(bytes),
        };
    }
    Ok(MILValue {
        r#type,
        tensor,
        blob,
    })
}

fn decode_blob_file_value(data: &[u8]) -> Result<MILBlobRef, MILDecoderError> {
    let mut reader = PBReader::new(data);
    let mut filename = String::new();
    let mut offset: u64 = 0;
    while let Some((field, wire)) = reader.read_tag() {
        match field {
            1 => filename = reader.read_string(),
            2 => offset = reader.read_varint(),
            _ => reader.skip(wire),
        }
    }
    Ok(MILBlobRef { filename, offset })
}

fn decode_immediate_value(data: &[u8]) -> Result<MILTensorData, MILDecoderError> {
    let mut reader = PBReader::new(data);
    let mut tensor = MILTensorData::default();
    while let Some((field, wire)) = reader.read_tag() {
        match field {
            1 => tensor = decode_tensor_value(reader.read_length_delimited())?,
            _ => reader.skip(wire),
        }
    }
    Ok(tensor)
}

fn decode_tensor_value(data: &[u8]) -> Result<MILTensorData, MILDecoderError> {
    let mut reader = PBReader::new(data);
    while let Some((field, wire)) = reader.read_tag() {
        match field {
            1 => {
                return Ok(MILTensorData::Floats(decode_repeated_floats(
                    reader.read_length_delimited(),
                )));
            }
            2 => {
                return Ok(MILTensorData::Ints(decode_repeated_ints(
                    reader.read_length_delimited(),
                )));
            }
            3 => {
                return Ok(MILTensorData::Bools(decode_repeated_bools(
                    reader.read_length_delimited(),
                )));
            }
            4 => {
                return Ok(MILTensorData::Strings(decode_repeated_strings(
                    reader.read_length_delimited(),
                )));
            }
            7 => {
                return Ok(MILTensorData::Fp16Bytes(decode_repeated_bytes(
                    reader.read_length_delimited(),
                )));
            }
            _ => reader.skip(wire),
        }
    }
    Ok(MILTensorData::Floats(Vec::new()))
}

/// `RepeatedFloats { repeated float values = 1; }` — packed varints inside a
/// length-delimited sub-message.
fn decode_repeated_floats(data: &[u8]) -> Vec<f32> {
    let mut reader = PBReader::new(data);
    while let Some((field, wire)) = reader.read_tag() {
        if field == 1 {
            return read_packed_floats(reader.read_length_delimited());
        }
        reader.skip(wire);
    }
    Vec::new()
}

fn decode_repeated_ints(data: &[u8]) -> Vec<i32> {
    let mut reader = PBReader::new(data);
    while let Some((field, wire)) = reader.read_tag() {
        if field == 1 {
            return read_packed_signed_varints(reader.read_length_delimited());
        }
        reader.skip(wire);
    }
    Vec::new()
}

fn decode_repeated_bools(data: &[u8]) -> Vec<bool> {
    let mut reader = PBReader::new(data);
    while let Some((field, wire)) = reader.read_tag() {
        if field == 1 {
            return reader
                .read_length_delimited()
                .iter()
                .map(|&b| b != 0)
                .collect();
        }
        reader.skip(wire);
    }
    Vec::new()
}

fn decode_repeated_bytes(data: &[u8]) -> Vec<u8> {
    let mut reader = PBReader::new(data);
    while let Some((field, wire)) = reader.read_tag() {
        if field == 1 {
            return reader.read_length_delimited().to_vec();
        }
        reader.skip(wire);
    }
    Vec::new()
}

fn decode_repeated_strings(data: &[u8]) -> Vec<String> {
    let mut reader = PBReader::new(data);
    let mut result = Vec::new();
    while let Some((field, wire)) = reader.read_tag() {
        match field {
            1 => result.push(reader.read_string()),
            _ => reader.skip(wire),
        }
    }
    result
}

fn decode_value_type(data: &[u8]) -> Result<MILType, MILDecoderError> {
    let mut reader = PBReader::new(data);
    while let Some((field, wire)) = reader.read_tag() {
        match field {
            1 => return decode_tensor_type(reader.read_length_delimited()),
            _ => reader.skip(wire),
        }
    }
    Ok(MILType::new(MILDataType::Float32, Vec::new()))
}

fn decode_tensor_type(data: &[u8]) -> Result<MILType, MILDecoderError> {
    let mut reader = PBReader::new(data);
    let mut data_type_raw: u64 = 11; // default Float32
    let mut shape = Vec::new();
    while let Some((field, wire)) = reader.read_tag() {
        match field {
            1 => data_type_raw = reader.read_varint(),
            2 => {
                let rank = reader.read_varint() as i64;
                if rank < 0 {
                    return Err(MILDecoderError::UnsupportedFormat(
                        "variable-rank tensors".into(),
                    ));
                }
            }
            3 => shape.push(decode_dimension(reader.read_length_delimited())?),
            _ => reader.skip(wire),
        }
    }
    let data_type = MILDataType::from_raw(data_type_raw)
        .ok_or(MILDecoderError::UnknownDataType(data_type_raw))?;
    Ok(MILType::with_dimensions(data_type, shape))
}

fn decode_dimension(data: &[u8]) -> Result<MILDimension, MILDecoderError> {
    let mut reader = PBReader::new(data);
    let mut dimension = None;
    while let Some((field, wire)) = reader.read_tag() {
        match field {
            1 => {
                let const_dim = reader.read_length_delimited();
                let mut sub = PBReader::new(const_dim);
                let mut size = 0;
                while let Some((sub_field, sub_wire)) = sub.read_tag() {
                    match sub_field {
                        1 => {
                            size = usize::try_from(sub.read_varint()).map_err(|_| {
                                MILDecoderError::UnsupportedFormat(
                                    "dimension exceeds platform size".into(),
                                )
                            })?
                        }
                        _ => sub.skip(sub_wire),
                    }
                }
                dimension = Some(MILDimension::Constant(size));
            }
            2 => {
                let mut sub = PBReader::new(reader.read_length_delimited());
                let mut variadic = false;
                while let Some((field, wire)) = sub.read_tag() {
                    if field == 1 {
                        variadic = sub.read_varint() != 0;
                    } else {
                        sub.skip(wire);
                    }
                }
                if variadic {
                    return Err(MILDecoderError::UnsupportedFormat(
                        "variadic dimensions".into(),
                    ));
                }
                dimension = Some(MILDimension::Unknown);
            }
            _ => reader.skip(wire),
        }
    }
    dimension.ok_or_else(|| {
        MILDecoderError::UnsupportedFormat(
            "dimension has neither constant nor unknown extent".into(),
        )
    })
}

fn decode_named_value_type(data: &[u8]) -> Result<MILNamedType, MILDecoderError> {
    let mut reader = PBReader::new(data);
    let mut name = String::new();
    let mut r#type = MILType::new(MILDataType::Float32, Vec::new());
    while let Some((field, wire)) = reader.read_tag() {
        match field {
            1 => name = reader.read_string(),
            2 => r#type = decode_value_type(reader.read_length_delimited())?,
            _ => reader.skip(wire),
        }
    }
    Ok(MILNamedType { name, r#type })
}

/// Decode a protobuf map entry (`key = field 1` string, `value = field 2`
/// message). Returns the key string and the raw value bytes.
fn decode_map_entry(data: &[u8]) -> Result<(String, Vec<u8>), MILDecoderError> {
    let mut reader = PBReader::new(data);
    let mut key = String::new();
    let mut value = Vec::new();
    while let Some((field, wire)) = reader.read_tag() {
        match field {
            1 => key = reader.read_string(),
            2 => value = reader.read_length_delimited().to_vec(),
            _ => reader.skip(wire),
        }
    }
    Ok((key, value))
}

#[cfg(test)]
mod shape_tests {
    use super::*;

    #[test]
    fn unknown_extent_is_not_constant_zero() {
        assert_eq!(decode_dimension(&[0x12, 0]).unwrap(), MILDimension::Unknown);
        assert_eq!(
            decode_dimension(&[0x0a, 0]).unwrap(),
            MILDimension::Constant(0)
        );
        assert_eq!(
            decode_dimension(&[0x0a, 2, 8, 4]).unwrap(),
            MILDimension::Constant(4)
        );
        let ty = MILType::with_dimensions(
            MILDataType::Float32,
            vec![MILDimension::Constant(0), MILDimension::Unknown],
        );
        assert_eq!(ty.text_representation(), "tensor<fp32, [0, ?]>");
        assert_eq!(ty.concrete_shape(), None);
        assert_eq!(
            MILType::new(MILDataType::Float32, vec![0]).concrete_shape(),
            Some(vec![0])
        );
    }

    #[test]
    fn unsupported_dimensions_are_not_silently_zero() {
        assert!(decode_dimension(&[]).is_err());
        assert!(decode_dimension(&[0x12, 2, 8, 1]).is_err());
        // TensorType.rank = -1.
        assert!(
            decode_tensor_type(&[
                0x10, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 1
            ])
            .is_err()
        );
    }

    #[test]
    fn values_require_concrete_dimensions() {
        // Value.type.tensorType = { dataType: FLOAT32, rank: 1, dimensions: unknown }.
        let ty = [0x12, 10, 0x0a, 8, 8, 11, 16, 1, 26, 2, 18, 0];
        assert!(decode_value(&ty).is_err());
        let mut blob = ty.to_vec();
        blob.extend_from_slice(&[0x2a, 0]);
        assert!(decode_value(&blob).is_err());
    }
}
