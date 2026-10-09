//! Prove exact signed integer index reductions without a floating-point boundary.

use super::int32_binary::{self, Input, immediate, ints, named, output, storage_fits};
use super::typed_unary::{array, compatible_shape, known_source, tensor};
use crate::protos::coreml::mil_spec::{self as mil, dimension, tensor_value, value};
use crate::protos::coreml::specification::{Model, array_feature_type, model};

#[path = "coreml_int32_arg/kernels.rs"]
mod kernels;
pub(super) use kernels::evaluate;

#[derive(Clone, Debug, PartialEq)]
pub(super) struct Reduction {
    pub input: Input,
    pub axis: usize,
    pub keep_dimensions: bool,
    pub maximum: bool,
}

fn scalar_int(value: &mil::Value) -> Option<i32> {
    let ty = tensor(value.r#type.as_ref()?)?;
    let [value] = ints(value)? else { return None };
    (ty.rank == 0 && ty.dimensions.is_empty()).then_some(*value)
}

fn scalar_bool(value: &mil::Value) -> Option<bool> {
    let ty = tensor(value.r#type.as_ref()?)?;
    if ty.data_type != mil::DataType::Bool as i32
        || ty.rank != 0
        || !ty.dimensions.is_empty()
        || !ty.attributes.is_empty()
    {
        return None;
    }
    let value::Value::ImmediateValue(value) = value.value.as_ref()? else {
        return None;
    };
    let value::immediate_value::Value::Tensor(value) = value.value.as_ref()? else {
        return None;
    };
    let tensor_value::Value::Bools(value) = value.value.as_ref()? else {
        return None;
    };
    let [value] = value.values.as_slice() else {
        return None;
    };
    Some(*value)
}

/// Only a whole known-wire Int32 argument reduction and exact constant views qualify.
pub(super) fn classify(model: &Model, source: &[u8]) -> Option<Reduction> {
    if !known_source(model, source) {
        return None;
    }
    let model::Type::MlProgram(program) = model.r#type.as_ref()? else {
        return None;
    };
    if program.version != 1 || program.functions.len() != 1 || !program.attributes.is_empty() {
        return None;
    }
    let function = program.functions.get("main")?;
    if function.opset != "CoreML7"
        || function.block_specializations.len() != 1
        || !function.attributes.is_empty()
        || function.inputs.len() > 1
    {
        return None;
    }
    let block = function.block_specializations.get(&function.opset)?;
    if !block.inputs.is_empty() || !block.attributes.is_empty() || block.outputs.len() != 1 {
        return None;
    }
    let (operation, prefix) = block.operations.split_last()?;
    let maximum = match operation.r#type.as_str() {
        "reduce_argmin" => false,
        "reduce_argmax" => true,
        _ => return None,
    };
    let keys = if operation.inputs.contains_key("keep_dims") {
        &["x", "axis", "keep_dims"][..]
    } else {
        &["x", "axis"][..]
    };
    let result = output(operation, &operation.r#type, keys)?;
    let axis = usize::try_from(scalar_int(immediate(operation, "axis")?)?).ok()?;
    let keep_dimensions = if operation.inputs.contains_key("keep_dims") {
        scalar_bool(immediate(operation, "keep_dims")?)?
    } else {
        false
    };
    let description = model.description.as_ref()?;
    if description.output.len() != 1 {
        return None;
    }
    let values = int32_binary::proven_inputs(function, description, prefix)?;
    let name = named(operation, "x")?;
    let (source_type, input) = values.get(name)?;
    let source_type = tensor(source_type.r#type.as_ref()?)?;
    if axis >= source_type.dimensions.len() || values.contains_key(&result.name) {
        return None;
    }
    // Native argument-reduction outputs are represented as Int32 even when the
    // public WebNN descriptor requests an Int64 proxy. Never truncate an index.
    let maximum_axis = match input {
        Input::Constant { shape, .. } => *shape.get(axis)?,
        Input::Runtime(name) => {
            let feature = array(description.input.iter().find(|value| value.name == *name)?)?;
            let maximum = match &feature.shape_flexibility {
                None => *feature.shape.get(axis)?,
                Some(array_feature_type::ShapeFlexibility::ShapeRange(range)) => {
                    range.size_ranges.get(axis)?.upper_bound
                }
                Some(array_feature_type::ShapeFlexibility::EnumeratedShapes(shapes)) => shapes
                    .shapes
                    .iter()
                    .map(|shape| shape.shape.get(axis).copied())
                    .collect::<Option<Vec<_>>>()?
                    .into_iter()
                    .max()?,
            };
            usize::try_from(maximum).ok()?
        }
    };
    if maximum_axis == 0 || maximum_axis > i32::MAX as usize {
        return None;
    }
    let mut dimensions = source_type.dimensions.clone();
    if keep_dimensions {
        dimensions[axis] = mil::Dimension {
            dimension: Some(dimension::Dimension::Constant(
                dimension::ConstantDimension { size: 1 },
            )),
        };
    } else {
        dimensions.remove(axis);
    }
    // RustNN's native scalar feature adapter is [1]; the public descriptor stays [].
    if dimensions.is_empty() {
        dimensions.push(mil::Dimension {
            dimension: Some(dimension::Dimension::Constant(
                dimension::ConstantDimension { size: 1 },
            )),
        });
    }
    let result_type = tensor(result.r#type.as_ref()?)?;
    let feature = &description.output[0];
    if result_type.data_type != mil::DataType::Int32 as i32
        || result_type.rank as usize != dimensions.len()
        || result_type.dimensions != dimensions
        || !result_type.attributes.is_empty()
        || result.name != block.outputs[0]
        || result.name != feature.name
        || feature.r#type.as_ref()?.is_optional
        || array(feature)?.data_type != 131104
        || !compatible_shape(result_type, array(feature)?)
        || !storage_fits(array(feature)?, usize::MAX as u64)
        || function
            .inputs
            .iter()
            .any(|value| !matches!(input, Input::Runtime(name) if *name == value.name))
    {
        return None;
    }
    Some(Reduction {
        input: input.clone(),
        axis,
        keep_dimensions,
        maximum,
    })
}

#[cfg(test)]
#[path = "coreml_int32_arg/tests.rs"]
mod tests;
