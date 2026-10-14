//! Replace full-axis even/odd selection with rank-bounded, unit-stride layout operations.
//!
//! For an even axis length N, reshape [..., N, ...] into [..., N/2, 2, ...],
//! select parity 0 or 1 on the new axis, and restore [..., N/2, ...].
//! No arithmetic, casts, weights or public tensor interfaces change. This avoids
//! costly native stride-two lowering seen while qualifying TinyStories on A12/A13.
//! Reject every unsupported shape or parameter without modifying the original op.

use super::{
    Argument, Binding, Block, CoremlMlProgramConverter, Dimension, MilOperation, NamedValueType,
    TensorType, ValueType, dimension,
};
use crate::protos::coreml::mil_spec::{DataType as MilType, tensor_value, value, value_type};
use crate::protos::coreml::specification::{Model, model};
use std::collections::{HashMap, HashSet};

fn tensor(value: &ValueType) -> Option<&TensorType> {
    match &value.r#type {
        Some(value_type::Type::TensorType(tensor)) => Some(tensor),
        _ => None,
    }
}

fn shape(value: &ValueType) -> Option<Vec<u32>> {
    let tensor = tensor(value)?;
    if usize::try_from(tensor.rank).ok()? != tensor.dimensions.len() {
        return None;
    }
    tensor
        .dimensions
        .iter()
        .map(|dim| match &dim.dimension {
            Some(dimension::Dimension::Constant(d)) if d.size > 0 && d.size <= i32::MAX as u64 => {
                Some(d.size as u32)
            }
            _ => None,
        })
        .collect()
}

fn immediate_tensor(
    argument: &Argument,
    length: usize,
    dtype: MilType,
) -> Option<&crate::protos::coreml::mil_spec::TensorValue> {
    let [binding] = argument.arguments.as_slice() else {
        return None;
    };
    let Some(Binding::Value(value)) = &binding.binding else {
        return None;
    };
    let value_type = value.r#type.as_ref()?;
    if tensor(value_type)?.data_type != dtype as i32 || shape(value_type)? != [length as u32] {
        return None;
    }
    let Some(value::Value::ImmediateValue(value)) = &value.value else {
        return None;
    };
    let Some(value::immediate_value::Value::Tensor(tensor)) = &value.value else {
        return None;
    };
    Some(tensor)
}

fn integers(argument: &Argument, length: usize) -> Option<Vec<i32>> {
    let Some(tensor_value::Value::Ints(values)) =
        &immediate_tensor(argument, length, MilType::Int32)?.value
    else {
        return None;
    };
    (values.values.len() == length).then(|| values.values.clone())
}

fn shaped(original: &ValueType, shape: &[u32]) -> ValueType {
    let mut result = original.clone();
    let Some(value_type::Type::TensorType(tensor)) = &mut result.r#type else {
        unreachable!()
    };
    tensor.rank = shape.len() as i64;
    tensor.dimensions = shape
        .iter()
        .map(|&size| Dimension {
            dimension: Some(dimension::Dimension::Constant(
                dimension::ConstantDimension { size: size.into() },
            )),
        })
        .collect();
    result
}

fn unique(base: String, used: &mut HashSet<String>) -> String {
    let mut name = base.clone();
    let mut serial = 0;
    while !used.insert(name.clone()) {
        serial += 1;
        name = format!("{base}_{serial}");
    }
    name
}

fn replacement(
    operation: &MilOperation,
    input_type: &ValueType,
    used: &mut HashSet<String>,
) -> Option<Vec<MilOperation>> {
    if operation.r#type != "slice_by_index"
        || !operation.blocks.is_empty()
        || !operation.attributes.is_empty()
    {
        return None;
    }
    let shape = shape(input_type)?;
    let rank = shape.len();
    // Keep integer tensors on their existing path: native CoreML layout
    // compositions can round wide int32 values through floating-point storage.
    // That separate boundary problem needs its own exact-value qualification.
    if !(1..=4).contains(&rank)
        || ![MilType::Float16 as i32, MilType::Float32 as i32]
            .contains(&tensor(input_type)?.data_type)
    {
        return None;
    }
    if !["x", "begin", "end", "stride"]
        .iter()
        .all(|key| operation.inputs.contains_key(*key))
        || operation.inputs.keys().any(|key| {
            ![
                "x",
                "begin",
                "end",
                "stride",
                "begin_mask",
                "end_mask",
                "squeeze_mask",
            ]
            .contains(&key.as_str())
        })
    {
        return None;
    }
    let [binding] = operation.inputs["x"].arguments.as_slice() else {
        return None;
    };
    let Some(Binding::Name(input_name)) = &binding.binding else {
        return None;
    };
    let begin = integers(&operation.inputs["begin"], rank)?;
    let end = integers(&operation.inputs["end"], rank)?;
    let stride = integers(&operation.inputs["stride"], rank)?;
    for name in ["begin_mask", "end_mask", "squeeze_mask"] {
        if let Some(argument) = operation.inputs.get(name) {
            let Some(tensor_value::Value::Bools(values)) =
                &immediate_tensor(argument, rank, MilType::Bool)?.value
            else {
                return None;
            };
            if values.values.len() != rank || values.values.iter().any(|&v| v) {
                return None;
            }
        }
    }
    let axes = stride
        .iter()
        .enumerate()
        .filter(|(_, stride)| **stride != 1)
        .map(|(axis, _)| axis)
        .collect::<Vec<_>>();
    let [axis] = axes.as_slice() else { return None };
    let axis = *axis;
    if stride[axis] != 2
        || !shape[axis].is_multiple_of(2)
        || ![0, 1].contains(&begin[axis])
        || end.iter().zip(&shape).any(|(&a, &b)| a != b as i32)
        || begin.iter().enumerate().any(|(i, &v)| i != axis && v != 0)
    {
        return None;
    }
    let mut output_shape = shape.clone();
    output_shape[axis] /= 2;
    let [output] = operation.outputs.as_slice() else {
        return None;
    };
    if output.r#type.as_ref()? != &shaped(input_type, &output_shape) {
        return None;
    }
    let mut split = shape;
    split[axis] /= 2;
    split.insert(axis + 1, 2);
    let mut selected = split.clone();
    selected[axis + 1] = 1;
    let mut offset = vec![0; rank + 1];
    offset[axis + 1] = begin[axis] as u32;
    let view = unique(format!("{}_stride2_view", output.name), used);
    let unit = unique(format!("{}_stride2_unit", output.name), used);
    let layout = |kind: &str,
                  input: String,
                  output_name: String,
                  out_shape: &[u32],
                  parameters: Vec<(&str, &[u32])>| {
        let mut inputs = HashMap::from([(
            "x".into(),
            CoremlMlProgramConverter::create_name_argument(input),
        )]);
        for (key, values) in parameters {
            inputs.insert(
                key.into(),
                CoremlMlProgramConverter::create_immediate_int_array(values),
            );
        }
        CoremlMlProgramConverter::create_mil_operation(
            kind,
            inputs,
            vec![NamedValueType {
                name: output_name,
                r#type: Some(shaped(input_type, out_shape)),
            }],
        )
    };
    Some(vec![
        layout(
            "reshape",
            input_name.clone(),
            view.clone(),
            &split,
            vec![("shape", &split)],
        ),
        layout(
            "slice_by_size",
            view,
            unit.clone(),
            &selected,
            vec![("begin", &offset), ("size", &selected)],
        ),
        layout(
            "reshape",
            unit,
            output.name.clone(),
            &output_shape,
            vec![("shape", &output_shape)],
        ),
    ])
}

fn rewrite(inputs: &[NamedValueType], block: &mut Block) {
    // Nested blocks have their own capture and shadowing scopes. Leave those
    // graphs alone until rewriting their symbol environments is supported.
    if !block.inputs.is_empty() || block.operations.iter().any(|op| !op.blocks.is_empty()) {
        return;
    }
    let mut types = HashMap::new();
    let mut used = HashSet::new();
    for input in inputs {
        if !used.insert(input.name.clone()) {
            return;
        }
        if let Some(value_type) = &input.r#type {
            types.insert(input.name.clone(), value_type.clone());
        }
    }
    for operation in &block.operations {
        for output in &operation.outputs {
            if !used.insert(output.name.clone()) {
                return;
            }
        }
    }
    used.extend(block.outputs.iter().cloned());
    for operation in std::mem::take(&mut block.operations) {
        let generated = operation.inputs.get("x").and_then(|argument| {
            let [binding] = argument.arguments.as_slice() else {
                return None;
            };
            let Some(Binding::Name(name)) = &binding.binding else {
                return None;
            };
            replacement(&operation, types.get(name)?, &mut used)
        });
        for output in &operation.outputs {
            if let Some(value_type) = &output.r#type {
                types.insert(output.name.clone(), value_type.clone());
            }
        }
        match generated {
            Some(operations) => block.operations.extend(operations),
            None => block.operations.push(operation),
        }
    }
}

/// Run last, after all arithmetic and partition lowering. A pipeline's model
/// boundaries, feature interfaces and metadata remain unchanged; only eligible
/// layout operations inside each leaf MLProgram are replaced.
pub(super) fn rewrite_model(model: &mut Model) {
    match &mut model.r#type {
        Some(model::Type::MlProgram(program)) => {
            for function in program.functions.values_mut() {
                for block in function.block_specializations.values_mut() {
                    rewrite(&function.inputs, block);
                }
            }
        }
        Some(model::Type::Pipeline(pipeline)) => {
            for child in &mut pipeline.models {
                rewrite_model(child);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
#[path = "stride_two_tests.rs"]
mod tests;
