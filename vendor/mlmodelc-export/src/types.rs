//! Intermediate types for MIL protobuf ↔ text translation.

/// MIL DataType enum values (from MIL.proto).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u64)]
pub enum MILDataType {
    Bool = 1,
    String = 2,
    Float16 = 10,
    Float32 = 11,
    Float64 = 12,
    Int8 = 21,
    Int16 = 22,
    Int32 = 23,
    Int64 = 24,
    Uint8 = 31,
    Uint16 = 32,
}

impl MILDataType {
    /// Decode from the raw varint value used on the protobuf wire.
    pub fn from_raw(raw: u64) -> Option<Self> {
        Some(match raw {
            1 => Self::Bool,
            2 => Self::String,
            10 => Self::Float16,
            11 => Self::Float32,
            12 => Self::Float64,
            21 => Self::Int8,
            22 => Self::Int16,
            23 => Self::Int32,
            24 => Self::Int64,
            31 => Self::Uint8,
            32 => Self::Uint16,
            _ => return None,
        })
    }

    /// The MIL text name for this data type (`fp32`, `int32`, etc.).
    pub fn text_name(self) -> &'static str {
        match self {
            Self::Bool => "bool",
            Self::String => "string",
            Self::Float16 => "fp16",
            Self::Float32 => "fp32",
            Self::Float64 => "fp64",
            Self::Int8 => "int8",
            Self::Int16 => "int16",
            Self::Int32 => "int32",
            Self::Int64 => "int64",
            Self::Uint8 => "uint8",
            Self::Uint16 => "uint16",
        }
    }
}

/// A fixed extent or an unknown runtime extent. Zero is a fixed, empty extent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MILDimension {
    Constant(usize),
    Unknown,
}

impl std::fmt::Display for MILDimension {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Constant(size) => write!(f, "{size}"),
            Self::Unknown => f.write_str("?"),
        }
    }
}

/// A MIL tensor type: data type + shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MILType {
    pub data_type: MILDataType,
    pub shape: Vec<MILDimension>,
}

impl MILType {
    pub fn new(data_type: MILDataType, shape: Vec<usize>) -> Self {
        Self::with_dimensions(
            data_type,
            shape.into_iter().map(MILDimension::Constant).collect(),
        )
    }

    pub fn with_dimensions(data_type: MILDataType, shape: Vec<MILDimension>) -> Self {
        Self { data_type, shape }
    }

    /// Returns `None` if any extent is determined at runtime.
    pub fn concrete_shape(&self) -> Option<Vec<usize>> {
        self.shape
            .iter()
            .map(|dimension| match dimension {
                MILDimension::Constant(size) => Some(*size),
                MILDimension::Unknown => None,
            })
            .collect()
    }

    /// Whether this is a scalar (rank-0 tensor).
    pub fn is_scalar(&self) -> bool {
        self.shape.is_empty()
    }

    /// Format as MIL text: `fp32` for scalars, `tensor<fp32, [1, 32]>` for tensors.
    pub fn text_representation(&self) -> String {
        if self.is_scalar() {
            return self.data_type.text_name().to_string();
        }
        let dims: Vec<String> = self.shape.iter().map(|d| d.to_string()).collect();
        format!(
            "tensor<{}, [{}]>",
            self.data_type.text_name(),
            dims.join(", ")
        )
    }
}

/// A named+typed value (used for function inputs and operation outputs).
#[derive(Debug, Clone)]
pub struct MILNamedType {
    pub name: String,
    pub r#type: MILType,
}

/// Raw tensor data in one of the supported storage formats.
#[derive(Debug, Clone)]
pub enum MILTensorData {
    Floats(Vec<f32>),
    Ints(Vec<i32>),
    Bools(Vec<bool>),
    Strings(Vec<String>),
    /// Raw IEEE 754 half-precision bytes, 2 bytes per element.
    Fp16Bytes(Vec<u8>),
    LongInts(Vec<i64>),
    Doubles(Vec<f64>),
}

impl Default for MILTensorData {
    fn default() -> Self {
        MILTensorData::Floats(Vec::new())
    }
}

/// An external-weights reference. CoreML MLProgram uses this for non-scalar
/// constants: `filename` is relative to `@model_path/` (for example,
/// `weights/weight.bin`), and `offset` is the byte offset into that file.
#[derive(Debug, Clone)]
pub struct MILBlobRef {
    pub filename: String,
    pub offset: u64,
}

/// A typed immediate value (type + data). When `blob` is `Some`, the value is
/// resolved from the external weights file at load time; `tensor` is ignored
/// by the MIL text emitter in that case and should be set to an empty payload.
#[derive(Debug, Clone)]
pub struct MILValue {
    pub r#type: MILType,
    pub tensor: MILTensorData,
    pub blob: Option<MILBlobRef>,
}

impl MILValue {
    pub fn new(r#type: MILType, tensor: MILTensorData) -> Self {
        Self {
            r#type,
            tensor,
            blob: None,
        }
    }

    /// Number of elements in the tensor data. For blob-backed values the
    /// element count is derived from `type.shape` since the tensor payload
    /// itself is empty.
    pub fn element_count(&self) -> usize {
        if self.blob.is_some() {
            return self
                .r#type
                .concrete_shape()
                .map(|shape| shape.iter().product())
                .unwrap_or(0);
        }
        match &self.tensor {
            MILTensorData::Floats(v) => v.len(),
            MILTensorData::Ints(v) => v.len(),
            MILTensorData::Bools(v) => v.len(),
            MILTensorData::Strings(v) => v.len(),
            MILTensorData::Fp16Bytes(d) => d.len() / 2,
            MILTensorData::LongInts(v) => v.len(),
            MILTensorData::Doubles(v) => v.len(),
        }
    }
}

/// A binding in an operation's argument list — either a reference to another
/// variable or an inline immediate value.
#[derive(Debug, Clone)]
pub enum MILBinding {
    Reference(String),
    Immediate(MILValue),
}

/// A single MIL operation in SSA form.
#[derive(Debug, Clone)]
pub struct MILOperation {
    pub r#type: String,
    pub blocks: Vec<MILBlock>,
    /// Inputs map: each key maps to one or more bindings (usually one; multi
    /// for concat/repeated values).
    pub inputs: Vec<(String, Vec<MILBinding>)>,
    pub outputs: Vec<MILNamedType>,
    /// Attributes map: each key maps to a raw `MILValue` (used by const ops
    /// for the `val` attribute).
    pub attributes: Vec<(String, MILValue)>,
}

/// A MIL basic block containing operations and output names.
#[derive(Debug, Clone, Default)]
pub struct MILBlock {
    pub inputs: Vec<MILNamedType>,
    pub outputs: Vec<String>,
    pub operations: Vec<MILOperation>,
}

impl MILBlock {
    /// Preorder traversal, including operations in nested control-flow blocks.
    pub fn all_operations(&self) -> impl Iterator<Item = &MILOperation> {
        // Keep only pending block iterators, rather than allocating a second
        // operation-sized collection for each metadata/weight traversal.
        let mut pending = vec![self.operations.iter()];
        std::iter::from_fn(move || {
            loop {
                let current = pending.last_mut()?;
                if let Some(op) = current.next() {
                    pending.extend(op.blocks.iter().rev().map(|block| block.operations.iter()));
                    return Some(op);
                }
                pending.pop();
            }
        })
    }
}

/// A MIL function with typed inputs, an opset identifier, and a main block.
#[derive(Debug, Clone)]
pub struct MILFunction {
    pub inputs: Vec<MILNamedType>,
    pub opset: String,
    pub block: MILBlock,
}

/// Top-level MIL program.
#[derive(Debug, Clone)]
pub struct MILProgram {
    pub version: u64,
    pub functions: Vec<(String, MILFunction)>,
    /// The spec version from the enclosing `Model` (e.g. 9 for CoreML8/iOS18).
    pub spec_version: i64,
    /// Raw protobuf bytes of `Model.description`. Retained for tools that want
    /// to round-trip the description, including flexible shapes and function selection.
    pub description_data: Vec<u8>,
}
