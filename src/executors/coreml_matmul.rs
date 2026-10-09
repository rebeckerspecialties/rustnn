//! Source-proven Float32 matrix products with bounded exact Rust arithmetic.

use super::*;
use crate::protos::coreml::mil_spec::{
    self as mil, argument, dimension, tensor_value, value, value_type,
};
use crate::protos::coreml::specification::{Model, feature_type, model};
use prost::Message;

#[path = "coreml_matmul/kernel.rs"]
pub(super) mod kernel;

#[derive(Clone, Debug, PartialEq)]
pub(super) enum Source {
    Input(String),
    Blob {
        metadata_offset: usize,
        elements: usize,
    },
    Immediate(Vec<f32>),
}

#[derive(Clone, Debug, PartialEq)]
enum View {
    Transpose(Vec<usize>),
    Reshape(Vec<usize>),
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct Operand {
    pub(super) source: Source,
    base_shape: Vec<usize>,
    pub(super) shape: Vec<usize>,
    views: Vec<View>,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct Plan {
    pub(super) left: Operand,
    pub(super) right: Operand,
    pub(super) output: String,
    pub(super) output_shape: Vec<usize>,
    pub(super) transpose_left: bool,
    pub(super) transpose_right: bool,
}

type BoundConstant<'a> = (&'a [f32], Vec<usize>, Vec<usize>);

fn invalid(reason: &str) -> GraphError {
    boundary_error(format!("unsupported exact Float32 matmul source: {reason}"))
}

fn shape(ty: &mil::ValueType) -> Result<(i32, Vec<usize>), GraphError> {
    let Some(value_type::Type::TensorType(t)) = &ty.r#type else {
        return Err(invalid("non-tensor type"));
    };
    if !t.attributes.is_empty() || t.rank < 0 || t.rank as usize != t.dimensions.len() {
        return Err(invalid("tensor rank/attributes"));
    }
    let shape = t
        .dimensions
        .iter()
        .map(|d| match &d.dimension {
            Some(dimension::Dimension::Constant(c)) => usize::try_from(c.size)
                .ok()
                .filter(|&x| x > 0)
                .ok_or_else(|| invalid("zero/overflow dimension")),
            _ => Err(invalid("unresolved shape")),
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((t.data_type, shape))
}

fn named(op: &mil::Operation, key: &str) -> Result<String, GraphError> {
    let a = op
        .inputs
        .get(key)
        .ok_or_else(|| invalid("missing operand"))?;
    if a.arguments.len() != 1 {
        return Err(invalid("multiple operand bindings"));
    }
    match &a.arguments[0].binding {
        Some(argument::binding::Binding::Name(name)) => Ok(name.clone()),
        _ => Err(invalid("operand is not a named value")),
    }
}

fn literal<'a>(op: &'a mil::Operation, key: &str) -> Result<&'a mil::Value, GraphError> {
    let a = op
        .inputs
        .get(key)
        .ok_or_else(|| invalid("missing literal"))?;
    if a.arguments.len() != 1 {
        return Err(invalid("multiple literal bindings"));
    }
    match &a.arguments[0].binding {
        Some(argument::binding::Binding::Value(v)) => Ok(v),
        _ => Err(invalid("literal is not immediate")),
    }
}

fn boolean(op: &mil::Operation, key: &str) -> Result<bool, GraphError> {
    let v = literal(op, key)?;
    if shape(
        v.r#type
            .as_ref()
            .ok_or_else(|| invalid("missing Boolean type"))?,
    )? != (mil::DataType::Bool as i32, vec![])
    {
        return Err(invalid("Boolean type differs"));
    }
    match &v.value {
        Some(value::Value::ImmediateValue(i)) => match &i.value {
            Some(value::immediate_value::Value::Tensor(t)) => match &t.value {
                Some(tensor_value::Value::Bools(b)) if b.values.len() == 1 => Ok(b.values[0]),
                _ => Err(invalid("Boolean payload differs")),
            },
            _ => Err(invalid("Boolean container differs")),
        },
        _ => Err(invalid("Boolean storage differs")),
    }
}

fn integers(op: &mil::Operation, key: &str) -> Result<Vec<usize>, GraphError> {
    let v = literal(op, key)?;
    let (dtype, sh) = shape(
        v.r#type
            .as_ref()
            .ok_or_else(|| invalid("missing index type"))?,
    )?;
    let Some(value::Value::ImmediateValue(i)) = &v.value else {
        return Err(invalid("view indices are not immediate"));
    };
    let Some(value::immediate_value::Value::Tensor(t)) = &i.value else {
        return Err(invalid("index container differs"));
    };
    let Some(tensor_value::Value::Ints(values)) = &t.value else {
        return Err(invalid("index payload differs"));
    };
    if dtype != mil::DataType::Int32 as i32 || sh != [values.values.len()] {
        return Err(invalid("index type/length differs"));
    }
    values
        .values
        .iter()
        .map(|&x| usize::try_from(x).map_err(|_| invalid("negative view index")))
        .collect()
}

fn product(shape: &[usize]) -> Result<usize, GraphError> {
    shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .ok_or_else(|| invalid("element count overflows"))
}

fn contiguous(shape: &[usize]) -> Result<Vec<usize>, GraphError> {
    let mut strides = vec![0; shape.len()];
    let mut s = 1usize;
    for i in (0..shape.len()).rev() {
        strides[i] = s;
        s = s
            .checked_mul(shape[i])
            .ok_or_else(|| invalid("stride overflows"))?;
    }
    Ok(strides)
}

/// A complete straight-line child may contain only original Float32 constants,
/// lossless shape/transpose views, and one Float32 matmul. No optimizer inference
/// or feature name establishes the operation's semantics.
pub(super) fn classify(model: &Model, source: &[u8]) -> Result<Option<Plan>, GraphError> {
    let marker = model
        .description
        .as_ref()
        .and_then(|d| d.metadata.as_ref())
        .and_then(|m| m.user_defined.get("rustnn.webnn.exact_f32_matmul"));
    if marker.is_none() {
        return Ok(None);
    }
    if marker.map(String::as_str) != Some("1") {
        return Err(invalid("unknown source proof version"));
    }
    let Some(model::Type::MlProgram(program)) = &model.r#type else {
        return Err(invalid("marked source is not an MLProgram"));
    };
    let Some(function) = program.functions.get("main") else {
        return Err(invalid("marked source has no main function"));
    };
    let Some(block) = function.block_specializations.get(&function.opset) else {
        return Err(invalid("marked source has no main specialization"));
    };
    let matrices:Vec<_>=block.operations.iter().filter(|o|o.r#type=="matmul" && o.outputs.iter().any(|t|t.r#type.as_ref().is_some_and(|ty|matches!(&ty.r#type,Some(value_type::Type::TensorType(t)) if t.data_type==mil::DataType::Float32 as i32)))).collect();
    if matrices.is_empty() {
        return Err(invalid("marked source has no Float32 matmul"));
    }
    if matrices.len() != 1
        || program.version != 1
        || program.functions.len() != 1
        || !program.attributes.is_empty()
        || function.block_specializations.len() != 1
        || !function.attributes.is_empty()
        || !block.inputs.is_empty()
        || !block.attributes.is_empty()
        || block.outputs.len() != 1
        || model.encoded_len() != source.len()
        || !float_cast::known_matmul_wire(source)
    {
        return Err(invalid("not a complete known-wire one-matmul child"));
    }
    let description = model
        .description
        .as_ref()
        .ok_or_else(|| invalid("missing model description"))?;
    if description.output.len() != 1 {
        return Err(invalid("multiple output features"));
    }
    let mut operands = HashMap::<String, Operand>::new();
    for input in &function.inputs {
        let (dtype, sh) = shape(
            input
                .r#type
                .as_ref()
                .ok_or_else(|| invalid("missing input type"))?,
        )?;
        if dtype != mil::DataType::Float32 as i32 {
            return Err(invalid("input is not Float32"));
        }
        let feature = description
            .input
            .iter()
            .find(|v| v.name == input.name)
            .ok_or_else(|| invalid("input feature absent"))?;
        let Some(feature_type::Type::MultiArrayType(a)) =
            feature.r#type.as_ref().and_then(|x| x.r#type.as_ref())
        else {
            return Err(invalid("input feature is not an array"));
        };
        if a.data_type != NativeType::Float32.code()
            || a.shape_flexibility.is_some()
            || a.shape
                .iter()
                .map(|&x| usize::try_from(x))
                .collect::<Result<Vec<_>, _>>()
                .ok()
                != Some(sh.clone())
        {
            return Err(invalid("input descriptor differs"));
        }
        if operands
            .insert(
                input.name.clone(),
                Operand {
                    source: Source::Input(input.name.clone()),
                    base_shape: sh.clone(),
                    shape: sh,
                    views: vec![],
                },
            )
            .is_some()
        {
            return Err(invalid("redefined input"));
        }
    }
    if description.input.len() != function.inputs.len() {
        return Err(invalid("extra model inputs"));
    }
    let mut result = None;
    for op in &block.operations {
        if op.outputs.len() != 1 || !op.blocks.is_empty() {
            return Err(invalid("multiple/nested operation outputs"));
        }
        let output = &op.outputs[0];
        let (dtype, sh) = shape(
            output
                .r#type
                .as_ref()
                .ok_or_else(|| invalid("missing output type"))?,
        )?;
        if dtype != mil::DataType::Float32 as i32 {
            return Err(invalid("non-Float32 value in matrix child"));
        }
        if operands.contains_key(&output.name) {
            return Err(invalid("redefined SSA value"));
        }
        let binding = match op.r#type.as_str() {
            "const" => {
                if !op.inputs.is_empty() || op.attributes.len() != 1 {
                    return Err(invalid("constant attributes/inputs"));
                }
                let v = op
                    .attributes
                    .get("val")
                    .ok_or_else(|| invalid("constant value absent"))?;
                if shape(
                    v.r#type
                        .as_ref()
                        .ok_or_else(|| invalid("constant type absent"))?,
                )? != (dtype, sh.clone())
                {
                    return Err(invalid("constant type differs"));
                }
                let elements = product(&sh)?;
                let source = match &v.value {
                    Some(value::Value::BlobFileValue(b))
                        if b.file_name == "@model_path/weights/weights.bin" =>
                    {
                        Source::Blob {
                            metadata_offset: usize::try_from(b.offset)
                                .map_err(|_| invalid("blob offset overflows"))?,
                            elements,
                        }
                    }
                    Some(value::Value::ImmediateValue(i)) => match &i.value {
                        Some(value::immediate_value::Value::Tensor(t)) => match &t.value {
                            Some(tensor_value::Value::Floats(f)) if f.values.len() == elements => {
                                Source::Immediate(f.values.clone())
                            }
                            _ => return Err(invalid("constant Float32 payload differs")),
                        },
                        _ => return Err(invalid("constant immediate container differs")),
                    },
                    _ => return Err(invalid("unsupported constant storage")),
                };
                Operand {
                    source,
                    base_shape: sh.clone(),
                    shape: sh.clone(),
                    views: vec![],
                }
            }
            "transpose" | "reshape" | "identity" => {
                if !op.attributes.is_empty() {
                    return Err(invalid("view attributes"));
                }
                let mut input = operands
                    .get(&named(op, "x")?)
                    .cloned()
                    .ok_or_else(|| invalid("unavailable view source"))?;
                match op.r#type.as_str() {
                    "transpose" => {
                        if op.inputs.len() != 2 {
                            return Err(invalid("transpose inputs"));
                        }
                        let axes = integers(op, "perm")?;
                        let mut sorted = axes.clone();
                        sorted.sort_unstable();
                        if sorted != (0..input.shape.len()).collect::<Vec<_>>()
                            || axes.iter().map(|&i| input.shape[i]).collect::<Vec<_>>() != sh
                        {
                            return Err(invalid("transpose shape/permutation"));
                        }
                        input.views.push(View::Transpose(axes));
                    }
                    "reshape" => {
                        if op.inputs.len() != 2 || product(&input.shape)? != product(&sh)? {
                            return Err(invalid("reshape element count"));
                        }
                        let requested = integers(op, "shape")?;
                        if requested != sh {
                            return Err(invalid("reshape parameters differ"));
                        }
                        input.views.push(View::Reshape(sh.clone()));
                    }
                    _ => {
                        if op.inputs.len() != 1 || input.shape != sh {
                            return Err(invalid("identity shape"));
                        }
                    }
                }
                input.shape = sh.clone();
                input
            }
            "matmul" => {
                if !op.attributes.is_empty() || op.inputs.len() != 4 || result.is_some() {
                    return Err(invalid("matmul attributes/inputs"));
                }
                let left = operands
                    .get(&named(op, "x")?)
                    .cloned()
                    .ok_or_else(|| invalid("left operand absent"))?;
                let right = operands
                    .get(&named(op, "y")?)
                    .cloned()
                    .ok_or_else(|| invalid("right operand absent"))?;
                let tx = boolean(op, "transpose_x")?;
                let ty = boolean(op, "transpose_y")?;
                if left.shape.len() < 2 || right.shape.len() < 2 {
                    return Err(invalid("matrix rank below two"));
                }
                let (m, k) = if tx {
                    (
                        left.shape[left.shape.len() - 1],
                        left.shape[left.shape.len() - 2],
                    )
                } else {
                    (
                        left.shape[left.shape.len() - 2],
                        left.shape[left.shape.len() - 1],
                    )
                };
                let (bk, n) = if ty {
                    (
                        right.shape[right.shape.len() - 1],
                        right.shape[right.shape.len() - 2],
                    )
                } else {
                    (
                        right.shape[right.shape.len() - 2],
                        right.shape[right.shape.len() - 1],
                    )
                };
                if left.shape.len() < 2 || right.shape.len() < 2 || k != bk || k > i32::MAX as usize
                {
                    return Err(invalid("matrix rank/reduction"));
                }
                let rank = left.shape.len().max(right.shape.len()) - 2;
                let mut expected = vec![1; rank];
                for (i, d) in expected.iter_mut().enumerate() {
                    let axis = rank - i;
                    let a = left
                        .shape
                        .len()
                        .checked_sub(axis + 2)
                        .map_or(1, |j| left.shape[j]);
                    let b = right
                        .shape
                        .len()
                        .checked_sub(axis + 2)
                        .map_or(1, |j| right.shape[j]);
                    if a != b && a != 1 && b != 1 {
                        return Err(invalid("batch broadcast mismatch"));
                    }
                    *d = a.max(b);
                }
                expected.extend([m, n]);
                if expected != sh {
                    return Err(invalid("matrix output shape differs"));
                }
                result = Some(Plan {
                    left,
                    right,
                    output: output.name.clone(),
                    output_shape: sh.clone(),
                    transpose_left: tx,
                    transpose_right: ty,
                });
                Operand {
                    source: Source::Input(output.name.clone()),
                    base_shape: sh.clone(),
                    shape: sh.clone(),
                    views: vec![],
                }
            }
            _ => return Err(invalid("extra arithmetic in matrix child")),
        };
        operands.insert(output.name.clone(), binding);
    }
    let plan = result.ok_or_else(|| invalid("matrix operation absent"))?;
    if block.outputs != [plan.output.clone()] || description.output[0].name != plan.output {
        return Err(invalid("post-matmul view/output requires a separate child"));
    }
    let Some(feature_type::Type::MultiArrayType(output)) = description.output[0]
        .r#type
        .as_ref()
        .and_then(|ty| ty.r#type.as_ref())
    else {
        return Err(invalid("output feature is not an array"));
    };
    if output.data_type != NativeType::Float32.code()
        || output.shape_flexibility.is_some()
        || output
            .shape
            .iter()
            .map(|&x| usize::try_from(x))
            .collect::<Result<Vec<_>, _>>()
            .ok()
            != Some(plan.output_shape.clone())
    {
        return Err(invalid("output descriptor differs"));
    }
    Ok(Some(plan))
}

impl Operand {
    pub(super) fn geometry(
        &self,
        actual_shape: &[usize],
        actual_strides: &[usize],
    ) -> Result<(Vec<usize>, Vec<usize>), GraphError> {
        if actual_shape != self.base_shape || actual_shape.len() != actual_strides.len() {
            return Err(invalid("bound operand shape/stride differs"));
        }
        let mut shape = actual_shape.to_vec();
        let mut strides = actual_strides.to_vec();
        for view in &self.views {
            match view {
                View::Transpose(axes) => {
                    shape = axes.iter().map(|&i| shape[i]).collect();
                    strides = axes.iter().map(|&i| strides[i]).collect();
                }
                View::Reshape(next) => {
                    let expected = contiguous(&shape)?;
                    if shape
                        .iter()
                        .zip(&strides)
                        .zip(&expected)
                        .any(|((&d, &s), &e)| d > 1 && s != e)
                    {
                        return Err(invalid(
                            "noncontiguous reshape would materialize an immutable operand",
                        ));
                    }
                    shape = next.clone();
                    strides = contiguous(&shape)?;
                }
            }
        }
        Ok((shape, strides))
    }

    pub(super) fn constant<'a>(
        &'a self,
        weights: Option<&'a [u8]>,
    ) -> Result<BoundConstant<'a>, GraphError> {
        if let Source::Immediate(data) = &self.source {
            let (shape, strides) =
                self.geometry(&self.base_shape, &contiguous(&self.base_shape)?)?;
            return Ok((data.as_slice(), shape, strides));
        }
        let Source::Blob {
            metadata_offset,
            elements,
        } = &self.source
        else {
            return Err(invalid("constant is not a source blob"));
        };
        let weights = weights.ok_or_else(|| invalid("weight file missing"))?;
        let end = metadata_offset
            .checked_add(24)
            .ok_or_else(|| invalid("blob header overflows"))?;
        let h = weights
            .get(*metadata_offset..end)
            .ok_or_else(|| invalid("truncated blob header"))?;
        let sentinel = u32::from_le_bytes(h[0..4].try_into().unwrap());
        let dtype = u32::from_le_bytes(h[4..8].try_into().unwrap());
        let length = usize::try_from(u64::from_le_bytes(h[8..16].try_into().unwrap()))
            .map_err(|_| invalid("blob length overflows"))?;
        let offset = usize::try_from(u64::from_le_bytes(h[16..24].try_into().unwrap()))
            .map_err(|_| invalid("blob payload offset overflows"))?;
        if sentinel != 0xDEADBEEF
            || dtype != 2
            || length
                != elements
                    .checked_mul(4)
                    .ok_or_else(|| invalid("blob byte count overflows"))?
            || offset < end
        {
            return Err(invalid("blob type/length/offset differs"));
        }
        let bytes = weights
            .get(
                offset
                    ..offset
                        .checked_add(length)
                        .ok_or_else(|| invalid("blob span overflows"))?,
            )
            .ok_or_else(|| invalid("truncated blob payload"))?;
        if !(bytes.as_ptr() as usize).is_multiple_of(4) || length > isize::MAX as usize {
            return Err(invalid("unaligned/overflow Float32 blob"));
        }
        // SAFETY: immutable validated aligned Float32 byte span lives with its graph-owned weight owner.
        let data = unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<f32>(), *elements) };
        let (shape, strides) = self.geometry(&self.base_shape, &contiguous(&self.base_shape)?)?;
        Ok((data, shape, strides))
    }
}
