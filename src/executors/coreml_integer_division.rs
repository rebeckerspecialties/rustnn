//! Recognize the complete signed truncation expression, never a bare floor_div.

use std::collections::HashMap;
use std::sync::Arc;

use super::typed_unary::{array, compatible_shape, known_source, tensor};
use crate::protos::coreml::mil_spec::{self as mil, argument, dimension, tensor_value, value};
use crate::protos::coreml::specification::{ArrayFeatureType, Model, array_feature_type, model};

#[path = "coreml_integer_division/kernels.rs"]
mod kernels;
pub(super) use kernels::evaluate;

#[derive(Clone, Debug, PartialEq)]
pub(super) enum Input {
    Runtime(String),
    Constant { bytes: Arc<[u8]>, shape: Vec<usize> },
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct Division {
    pub left: Input,
    pub right: Input,
}

fn named<'a>(operation: &'a mil::Operation, key: &str) -> Option<&'a str> {
    let [binding] = operation.inputs.get(key)?.arguments.as_slice() else {
        return None;
    };
    match binding.binding.as_ref()? {
        argument::binding::Binding::Name(name) => Some(name),
        _ => None,
    }
}

fn immediate<'a>(operation: &'a mil::Operation, key: &str) -> Option<&'a mil::Value> {
    let [binding] = operation.inputs.get(key)?.arguments.as_slice() else {
        return None;
    };
    match binding.binding.as_ref()? {
        argument::binding::Binding::Value(value) => Some(value),
        _ => None,
    }
}

fn ints(value: &mil::Value) -> Option<&[i32]> {
    let ty = tensor(value.r#type.as_ref()?)?;
    let shape = static_shape(ty)?;
    let value::Value::ImmediateValue(value) = value.value.as_ref()? else {
        return None;
    };
    let value::immediate_value::Value::Tensor(value) = value.value.as_ref()? else {
        return None;
    };
    let tensor_value::Value::Ints(values) = value.value.as_ref()? else {
        return None;
    };
    (ty.data_type == mil::DataType::Int32 as i32 && elements(&shape) == Some(values.values.len()))
        .then_some(values.values.as_slice())
}

fn int_vector(value: &mil::Value) -> Option<&[i32]> {
    let values = ints(value)?;
    (tensor(value.r#type.as_ref()?)?.rank == 1).then_some(values)
}

fn static_shape(ty: &mil::TensorType) -> Option<Vec<usize>> {
    if ty.rank < 0
        || usize::try_from(ty.rank).ok()? != ty.dimensions.len()
        || !ty.attributes.is_empty()
    {
        return None;
    }
    ty.dimensions
        .iter()
        .map(|dimension| {
            let dimension::Dimension::Constant(size) = dimension.dimension.as_ref()? else {
                return None;
            };
            usize::try_from(size.size).ok().filter(|&size| size > 0)
        })
        .collect()
}

fn elements(shape: &[usize]) -> Option<usize> {
    shape
        .iter()
        .try_fold(1usize, |n, &size| n.checked_mul(size))
        .filter(|&n| n <= usize::MAX / 4)
}

fn storage_fits(array: &ArrayFeatureType, max_bytes: u64) -> bool {
    let fits = |shape: &[i64]| {
        !shape.is_empty()
            && shape
                .iter()
                .try_fold(4u64, |bytes, &size| {
                    let size = u64::try_from(size).ok().filter(|&size| size > 0)?;
                    bytes.checked_mul(size).filter(|&bytes| bytes <= max_bytes)
                })
                .is_some()
    };
    fits(&array.shape)
        && match &array.shape_flexibility {
            None => true,
            Some(array_feature_type::ShapeFlexibility::EnumeratedShapes(shapes)) => {
                !shapes.shapes.is_empty() && shapes.shapes.iter().all(|shape| fits(&shape.shape))
            }
            Some(array_feature_type::ShapeFlexibility::ShapeRange(range)) => {
                range.size_ranges.len() == array.shape.len()
                    && range.size_ranges.iter().all(|bound| {
                        bound.lower_bound > 0
                            && u64::try_from(bound.upper_bound)
                                .is_ok_and(|upper| upper >= bound.lower_bound)
                    })
                    && fits(
                        &range
                            .size_ranges
                            .iter()
                            .map(|bound| bound.upper_bound)
                            .collect::<Vec<_>>(),
                    )
            }
        }
}

fn transpose(
    bytes: &[u8],
    before: &[usize],
    shape: &[usize],
    permutation: &[i32],
) -> Option<Vec<u8>> {
    let permutation = permutation
        .iter()
        .map(|&axis| usize::try_from(axis).ok())
        .collect::<Option<Vec<_>>>()?;
    if permutation.len() != before.len()
        || shape.len() != before.len()
        || permutation
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>()
            .len()
            != before.len()
        || permutation
            .iter()
            .enumerate()
            .any(|(axis, &source)| before.get(source) != shape.get(axis))
        || elements(before)?.checked_mul(4)? != bytes.len()
    {
        return None;
    }
    let mut strides = vec![1usize; before.len()];
    for axis in (0..before.len().saturating_sub(1)).rev() {
        strides[axis] = strides[axis + 1].checked_mul(before[axis + 1])?;
    }
    let mut result = Vec::new();
    result.try_reserve_exact(bytes.len()).ok()?;
    for index in 0..elements(shape)? {
        let mut remaining = index;
        let mut source = 0;
        for axis in (0..shape.len()).rev() {
            source += (remaining % shape[axis]) * strides[permutation[axis]];
            remaining /= shape[axis];
        }
        result.extend_from_slice(&bytes[source * 4..source * 4 + 4]);
    }
    Some(result)
}

fn output(operation: &mil::Operation, kind: &str, keys: &[&str]) -> Option<mil::NamedValueType> {
    if operation.r#type != kind
        || !operation.blocks.is_empty()
        || !operation.attributes.is_empty()
        || operation.outputs.len() != 1
        || operation.inputs.len() != keys.len()
        || !keys.iter().all(|key| operation.inputs.contains_key(*key))
    {
        return None;
    }
    let result = operation.outputs[0].clone();
    (!result.name.is_empty()).then_some(result)
}

fn zero(operation: &mil::Operation, key: &str) -> bool {
    immediate(operation, key).is_some_and(|value| {
        ints(value) == Some(&[0][..]) && tensor(value.r#type.as_ref().unwrap()).unwrap().rank == 0
    })
}

fn target_int32(operation: &mil::Operation) -> Option<()> {
    let value = immediate(operation, "dtype")?;
    let ty = tensor(value.r#type.as_ref()?)?;
    if ty.data_type != mil::DataType::String as i32
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
    let tensor_value::Value::Strings(values) = value.value.as_ref()? else {
        return None;
    };
    (values.values.as_slice() == ["int32"]).then_some(())
}

fn broadcast(left: &mil::TensorType, right: &mil::TensorType) -> Option<Vec<mil::Dimension>> {
    let rank = left.dimensions.len().max(right.dimensions.len());
    let unit = mil::Dimension {
        dimension: Some(dimension::Dimension::Constant(
            dimension::ConstantDimension { size: 1 },
        )),
    };
    let mut result = vec![unit.clone(); rank];
    for source in [left, right] {
        if source.rank < 0
            || source.rank as usize != source.dimensions.len()
            || !source.attributes.is_empty()
        {
            return None;
        }
        for (axis, size) in source.dimensions.iter().enumerate() {
            let target = &mut result[rank - source.dimensions.len() + axis];
            use dimension::Dimension::{Constant, Unknown};
            if !matches!(size.dimension.as_ref()?, Constant(n) if n.size > 0)
                && !matches!(size.dimension.as_ref()?, Unknown(n) if !n.variadic)
            {
                return None;
            }
            *target = match (target.dimension.as_ref()?, size.dimension.as_ref()?) {
                (_, Constant(n)) if n.size == 1 => target.clone(),
                (Constant(n), _) if n.size == 1 => size.clone(),
                (Constant(a), Constant(b)) if a.size == b.size && a.size > 0 => target.clone(),
                (Unknown(a), Constant(b)) if !a.variadic && b.size > 0 => size.clone(),
                (Constant(a), Unknown(b)) if a.size > 0 && !b.variadic => target.clone(),
                (Unknown(a), Unknown(b)) if !a.variadic && !b.variadic => target.clone(),
                _ => return None,
            };
        }
    }
    Some(result)
}

/// Accept only a complete known-wire program with the proven quotient/remainder
/// correction and its exact Int32 constant/view closure. No name, marker or
/// individual opcode is sufficient evidence of WebNN truncation semantics.
pub(super) fn classify(model: &Model, source: &[u8]) -> Option<Division> {
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
        || function.inputs.len() > 2
    {
        return None;
    }
    let block = function.block_specializations.get(&function.opset)?;
    if !block.inputs.is_empty()
        || !block.attributes.is_empty()
        || block.outputs.len() != 1
        || block.operations.len() < 7
    {
        return None;
    }
    let description = model.description.as_ref()?;
    if description.input.len() != function.inputs.len() || description.output.len() != 1 {
        return None;
    }
    let mut values = HashMap::new();
    for input in &function.inputs {
        let ty = tensor(input.r#type.as_ref()?)?;
        let feature = description
            .input
            .iter()
            .find(|value| value.name == input.name)?;
        if ty.data_type != mil::DataType::Int32 as i32
            || feature.r#type.as_ref()?.is_optional
            || array(feature)?.data_type != 131104
            || !compatible_shape(ty, array(feature)?)
            || !storage_fits(array(feature)?, usize::MAX as u64)
            || values
                .insert(
                    input.name.clone(),
                    (input.clone(), Input::Runtime(input.name.clone())),
                )
                .is_some()
        {
            return None;
        }
    }
    let (prefix, tail) = block.operations.split_at(block.operations.len() - 7);
    for operation in prefix {
        if !operation.blocks.is_empty() || operation.outputs.len() != 1 {
            return None;
        }
        let result = operation.outputs[0].clone();
        let ty = tensor(result.r#type.as_ref()?)?;
        if ty.data_type != mil::DataType::Int32 as i32 {
            return None;
        }
        let shape = static_shape(ty)?;
        let input = if operation.r#type == "const"
            && operation.inputs.is_empty()
            && operation.attributes.len() == 1
        {
            let value = operation.attributes.get("val")?;
            if value.r#type != result.r#type {
                return None;
            }
            let bytes: Arc<[u8]> = ints(value)?
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>()
                .into();
            Input::Constant { bytes, shape }
        } else if matches!(
            operation.r#type.as_str(),
            "reshape" | "identity" | "transpose" | "mul"
        ) && operation.attributes.is_empty()
            && operation.inputs.len() == if operation.r#type == "identity" { 1 } else { 2 }
        {
            let (
                _,
                Input::Constant {
                    bytes,
                    shape: before,
                },
            ) = values.get(named(operation, "x")?)?
            else {
                return None;
            };
            if elements(&shape)? != elements(before)? {
                return None;
            }
            let bytes = match operation.r#type.as_str() {
                "reshape" => {
                    let target = int_vector(immediate(operation, "shape")?)?
                        .iter()
                        .map(|&n| usize::try_from(n).ok().filter(|&n| n > 0))
                        .collect::<Option<Vec<_>>>()?;
                    if shape != target {
                        return None;
                    }
                    bytes.clone()
                }
                "identity" if shape == *before => bytes.clone(),
                "mul"
                    if shape == *before
                        && immediate(operation, "y").is_some_and(|value| {
                            ints(value) == Some(&[1][..])
                                && tensor(value.r#type.as_ref().unwrap()).unwrap().rank == 0
                        }) =>
                {
                    bytes.clone()
                }
                "transpose" => transpose(
                    bytes,
                    before,
                    &shape,
                    int_vector(immediate(operation, "perm")?)?,
                )?
                .into(),
                _ => return None,
            };
            Input::Constant { bytes, shape }
        } else {
            return None;
        };
        if result.name.is_empty()
            || values
                .insert(result.name.clone(), (result, input))
                .is_some()
        {
            return None;
        }
    }
    let q = output(&tail[0], "floor_div", &["x", "y"])?;
    let r = output(&tail[1], "mod", &["x", "y"])?;
    let negative = output(&tail[2], "less", &["x", "y"])?;
    let nonzero = output(&tail[3], "not_equal", &["x", "y"])?;
    let adjust = output(&tail[4], "logical_and", &["x", "y"])?;
    let integer = output(&tail[5], "cast", &["x", "dtype"])?;
    let result = output(&tail[6], "add", &["x", "y"])?;
    let left_name = named(&tail[0], "x")?;
    let right_name = named(&tail[0], "y")?;
    if named(&tail[1], "x")? != left_name
        || named(&tail[1], "y")? != right_name
        || named(&tail[2], "x")? != q.name
        || !zero(&tail[2], "y")
        || named(&tail[3], "x")? != r.name
        || !zero(&tail[3], "y")
        || named(&tail[4], "x")? != negative.name
        || named(&tail[4], "y")? != nonzero.name
        || named(&tail[5], "x")? != adjust.name
        || target_int32(&tail[5]).is_none()
        || named(&tail[6], "x")? != q.name
        || named(&tail[6], "y")? != integer.name
        || block.outputs[0] != result.name
        || description.output[0].name != result.name
    {
        return None;
    }
    let (left_type, left) = values.get(left_name)?;
    let (right_type, right) = values.get(right_name)?;
    let dimensions = broadcast(
        tensor(left_type.r#type.as_ref()?)?,
        tensor(right_type.r#type.as_ref()?)?,
    )?;
    let mut names = values
        .keys()
        .cloned()
        .collect::<std::collections::HashSet<_>>();
    for (index, value) in [&q, &r, &negative, &nonzero, &adjust, &integer, &result]
        .into_iter()
        .enumerate()
    {
        let ty = tensor(value.r#type.as_ref()?)?;
        if ty.rank as usize != dimensions.len()
            || ty.dimensions != dimensions
            || !ty.attributes.is_empty()
            || ty.data_type
                != if [2, 3, 4].contains(&index) {
                    mil::DataType::Bool
                } else {
                    mil::DataType::Int32
                } as i32
            || !names.insert(value.name.clone())
        {
            return None;
        }
    }
    let feature = &description.output[0];
    if feature.r#type.as_ref()?.is_optional
        || array(feature)?.data_type != 131104
        || !compatible_shape(tensor(result.r#type.as_ref()?)?, array(feature)?)
        || !storage_fits(array(feature)?, usize::MAX as u64)
    {
        return None;
    }
    // Extra inputs are not part of the proved closed expression.
    if function.inputs.iter().any(|input| {
        ![left, right]
            .iter()
            .any(|source| matches!(source,Input::Runtime(name) if *name == input.name))
    }) {
        return None;
    }
    Some(Division {
        left: left.clone(),
        right: right.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::converters::{CoremlMlProgramConverter, GraphConverter};
    use crate::graph::{
        DataType, GraphInfo, Operand, OperandDescriptor, OperandKind, to_dimension_vector,
    };
    use crate::operators::Operation;
    use prost::Message;

    fn fixture() -> Model {
        let operand = |name: &str, kind| Operand {
            name: Some(name.into()),
            kind,
            descriptor: OperandDescriptor {
                data_type: DataType::Int32,
                shape: to_dimension_vector(&[2]),
                pending_permutation: vec![],
            },
        };
        let graph = GraphInfo {
            operands: vec![
                operand("left", OperandKind::Input),
                operand("right", OperandKind::Input),
                operand("result", OperandKind::Output),
            ],
            input_operands: vec![0, 1],
            output_operands: vec![2],
            operations: vec![Operation::Div {
                a: 0,
                b: 1,
                outputs: vec![2],
                options: None,
            }],
            ..Default::default()
        };
        Model::decode(
            CoremlMlProgramConverter
                .convert(&graph)
                .unwrap()
                .data
                .as_slice(),
        )
        .unwrap()
    }

    fn block(model: &mut Model) -> &mut mil::Block {
        let model::Type::MlProgram(program) = model.r#type.as_mut().unwrap() else {
            panic!()
        };
        program
            .functions
            .get_mut("main")
            .unwrap()
            .block_specializations
            .get_mut("CoreML7")
            .unwrap()
    }

    fn accepted(model: &Model) -> bool {
        classify(model, &model.encode_to_vec()).is_some()
    }

    fn constant_view_fixture(kind: &str) -> Model {
        let mut source = fixture();
        source.description.as_mut().unwrap().input.remove(0);
        let model::Type::MlProgram(program) = source.r#type.as_mut().unwrap() else {
            panic!()
        };
        let function = program.functions.get_mut("main").unwrap();
        let left = function.inputs.remove(0);
        let b = function.block_specializations.get_mut("CoreML7").unwrap();
        let mut data = immediate(&b.operations[2], "y").unwrap().clone();
        data.r#type = left.r#type.clone();
        let value::Value::ImmediateValue(value) = data.value.as_mut().unwrap() else {
            panic!()
        };
        let value::immediate_value::Value::Tensor(value) = value.value.as_mut().unwrap() else {
            panic!()
        };
        let tensor_value::Value::Ints(ints) = value.value.as_mut().unwrap() else {
            panic!()
        };
        ints.values = vec![-7, 7];
        let mut constant = left.clone();
        constant.name = "constant".into();
        let mut parameter = immediate(&b.operations[2], "y").unwrap().clone();
        if kind != "mul" {
            let mil::value_type::Type::TensorType(ty) =
                parameter.r#type.as_mut().unwrap().r#type.as_mut().unwrap()
            else {
                panic!()
            };
            ty.rank = 1;
            ty.dimensions = vec![mil::Dimension {
                dimension: Some(dimension::Dimension::Constant(
                    dimension::ConstantDimension { size: 1 },
                )),
            }];
        }
        let value::Value::ImmediateValue(value) = parameter.value.as_mut().unwrap() else {
            panic!()
        };
        let value::immediate_value::Value::Tensor(value) = value.value.as_mut().unwrap() else {
            panic!()
        };
        let tensor_value::Value::Ints(ints) = value.value.as_mut().unwrap() else {
            panic!()
        };
        ints.values = vec![match kind {
            "reshape" => 2,
            "mul" => 1,
            _ => 0,
        }];
        let argument = |binding| mil::Argument {
            arguments: vec![argument::Binding {
                binding: Some(binding),
            }],
        };
        b.operations.splice(
            0..0,
            [
                mil::Operation {
                    r#type: "const".into(),
                    outputs: vec![constant],
                    attributes: HashMap::from([("val".into(), data)]),
                    ..Default::default()
                },
                mil::Operation {
                    r#type: kind.into(),
                    outputs: vec![left],
                    inputs: HashMap::from([
                        (
                            "x".into(),
                            argument(argument::binding::Binding::Name("constant".into())),
                        ),
                        (
                            match kind {
                                "reshape" => "shape",
                                "mul" => "y",
                                _ => "perm",
                            }
                            .into(),
                            argument(argument::binding::Binding::Value(parameter)),
                        ),
                    ]),
                    ..Default::default()
                },
            ],
        );
        source
    }

    #[test]
    fn integer_division_constant_views_require_parameter_rank_and_exact_unit_literal() {
        for kind in ["reshape", "transpose", "mul"] {
            let source = constant_view_fixture(kind);
            assert!(accepted(&source), "{kind}");
            for rank in [0, 1, 2] {
                if (kind == "mul" && rank == 0) || (kind != "mul" && rank == 1) {
                    continue;
                }
                let mut changed = source.clone();
                let operation = &mut block(&mut changed).operations[1];
                let key = match kind {
                    "reshape" => "shape",
                    "mul" => "y",
                    _ => "perm",
                };
                let argument::binding::Binding::Value(value) =
                    operation.inputs.get_mut(key).unwrap().arguments[0]
                        .binding
                        .as_mut()
                        .unwrap()
                else {
                    panic!()
                };
                let mil::value_type::Type::TensorType(ty) =
                    value.r#type.as_mut().unwrap().r#type.as_mut().unwrap()
                else {
                    panic!()
                };
                ty.rank = rank;
                ty.dimensions = vec![
                    mil::Dimension {
                        dimension: Some(dimension::Dimension::Constant(
                            dimension::ConstantDimension { size: 1 }
                        ))
                    };
                    rank as usize
                ];
                assert!(!accepted(&changed), "{kind} rank={rank}");
            }
        }
        let mut changed = constant_view_fixture("mul");
        let argument::binding::Binding::Value(value) = block(&mut changed).operations[1]
            .inputs
            .get_mut("y")
            .unwrap()
            .arguments[0]
            .binding
            .as_mut()
            .unwrap()
        else {
            panic!()
        };
        let value::Value::ImmediateValue(value) = value.value.as_mut().unwrap() else {
            panic!()
        };
        let value::immediate_value::Value::Tensor(value) = value.value.as_mut().unwrap() else {
            panic!()
        };
        let tensor_value::Value::Ints(ints) = value.value.as_mut().unwrap() else {
            panic!()
        };
        ints.values[0] = 2;
        assert!(!accepted(&changed));
    }

    #[test]
    fn integer_division_requires_complete_closed_expression_not_a_marker_or_floor_div() {
        let source = fixture();
        assert!(accepted(&source));
        for index in 0..7 {
            let mut changed = source.clone();
            block(&mut changed).operations[index].r#type = "identity".into();
            assert!(!accepted(&changed), "operation {index}");
        }
        let mut bare = source.clone();
        let block = block(&mut bare);
        block.operations.truncate(1);
        block.outputs[0] = block.operations[0].outputs[0].name.clone();
        bare.description.as_mut().unwrap().output[0].name = block.outputs[0].clone();
        assert!(!accepted(&bare));
        let mut wire = source.encode_to_vec();
        wire.extend_from_slice(&[0xf8, 0x7f, 1]);
        assert!(classify(&Model::decode(wire.as_slice()).unwrap(), &wire).is_none());
    }

    #[test]
    fn integer_division_rejects_changed_bindings_shapes_dtypes_and_nested_programs() {
        let source = fixture();
        for mutation in 0..8 {
            let mut changed = source.clone();
            let b = block(&mut changed);
            match mutation {
                0 => {
                    b.operations[1].inputs.get_mut("y").unwrap().arguments =
                        b.operations[0].inputs["x"].arguments.clone()
                }
                1 => {
                    b.operations[4].inputs.get_mut("x").unwrap().arguments =
                        b.operations[0].inputs["x"].arguments.clone()
                }
                2 => {
                    let duplicate = b.operations[0].inputs["x"].arguments[0].clone();
                    b.operations[0]
                        .inputs
                        .get_mut("x")
                        .unwrap()
                        .arguments
                        .push(duplicate);
                }
                3 => b.operations[0].blocks.push(mil::Block::default()),
                4 => b.operations[0].outputs[0].name = "left".into(),
                5 | 6 => {
                    let mil::value_type::Type::TensorType(ty) = b.operations[2].outputs[0]
                        .r#type
                        .as_mut()
                        .unwrap()
                        .r#type
                        .as_mut()
                        .unwrap()
                    else {
                        panic!()
                    };
                    if mutation == 5 {
                        ty.data_type = mil::DataType::Int32 as i32;
                    } else {
                        ty.rank = 0;
                        ty.dimensions.clear();
                    }
                }
                7 => {
                    changed.description.as_mut().unwrap().input[0]
                        .r#type
                        .as_mut()
                        .unwrap()
                        .is_optional = true
                }
                _ => unreachable!(),
            }
            assert!(!accepted(&changed), "mutation {mutation}");
        }
    }

    #[test]
    fn integer_division_shape_limits_and_constant_permutations_are_checked() {
        let mut feature = array(&fixture().description.unwrap().input[0])
            .unwrap()
            .clone();
        assert!(storage_fits(&feature, u32::MAX as u64));
        feature.shape = vec![1i64 << 30];
        assert!(!storage_fits(&feature, u32::MAX as u64));
        feature.shape = vec![i64::MAX, i64::MAX];
        assert!(!storage_fits(&feature, u64::MAX));
        let bytes: Vec<_> = [16_777_217_i32, -7, 3, 4, 5, 6]
            .iter()
            .flat_map(|n| n.to_le_bytes())
            .collect();
        let expected: Vec<_> = [16_777_217_i32, 4, -7, 5, 3, 6]
            .iter()
            .flat_map(|n| n.to_le_bytes())
            .collect();
        assert_eq!(transpose(&bytes, &[2, 3], &[3, 2], &[1, 0]), Some(expected));
        assert!(transpose(&bytes, &[2, 3], &[3, 2], &[1, 1]).is_none());
        assert!(transpose(&bytes, &[2, 3], &[3, 2], &[-1, 0]).is_none());
        assert!(transpose(&bytes, &[2, 3], &[2, 3], &[1, 0]).is_none());
    }
}
