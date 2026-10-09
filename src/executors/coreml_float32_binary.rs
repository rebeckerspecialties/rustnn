//! Source-proven Float32 multiplication/division with exact bit interchange.

use std::collections::HashMap;

use super::typed_unary::{array, compatible_shape, constant::ConstantInput, known_source, tensor};
use crate::error::GraphError;
use crate::protos::coreml::mil_spec::{self as mil, argument, dimension};
use crate::protos::coreml::specification::{ArrayFeatureType, Model, array_feature_type, model};

#[path = "coreml_float32_binary/kernels.rs"]
mod kernels;
pub(super) use kernels::evaluate;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    Multiply,
    Divide,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum Input {
    Runtime(String),
    Constant(ConstantInput),
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct Binary {
    pub kind: Kind,
    pub left: Input,
    pub right: Input,
}

impl Binary {
    pub fn resolve(&mut self, weights: Option<&[u8]>) -> Result<(), GraphError> {
        for input in [&mut self.left, &mut self.right] {
            if let Input::Constant(input) = input {
                input.resolve(weights)?;
            }
        }
        Ok(())
    }
}

fn named<'a>(op: &'a mil::Operation, key: &str) -> Option<&'a str> {
    let [binding] = op.inputs.get(key)?.arguments.as_slice() else {
        return None;
    };
    match binding.binding.as_ref()? {
        argument::binding::Binding::Name(name) => Some(name),
        _ => None,
    }
}

fn storage_fits(array: &ArrayFeatureType, max_bytes: u64) -> bool {
    let fits = |shape: &[i64]| {
        !shape.is_empty()
            && shape
                .iter()
                .try_fold(4_u64, |bytes, &size| {
                    let size = u64::try_from(size).ok().filter(|&n| n > 0)?;
                    bytes.checked_mul(size).filter(|&n| n <= max_bytes)
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
                        u64::try_from(bound.upper_bound)
                            .is_ok_and(|upper| upper > 0 && upper >= bound.lower_bound)
                    })
                    && fits(
                        &range
                            .size_ranges
                            .iter()
                            .map(|n| n.upper_bound)
                            .collect::<Vec<_>>(),
                    )
            }
        }
}

fn broadcast(left: &mil::TensorType, right: &mil::TensorType) -> Option<Vec<mil::Dimension>> {
    let rank = left.dimensions.len().max(right.dimensions.len());
    let mut result = vec![
        mil::Dimension {
            dimension: Some(dimension::Dimension::Constant(
                dimension::ConstantDimension { size: 1 }
            ))
        };
        rank
    ];
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

/// Prove the complete known-wire program, including only checked floating
/// constants/views before one binary operation. Unknown operations remain
/// native; the converter materializes unsupported constant producers first.
pub(super) fn classify(model: &Model, source: &[u8]) -> Option<Binary> {
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
        || function.inputs.len() > 2
        || function.block_specializations.len() != 1
        || !function.attributes.is_empty()
    {
        return None;
    }
    let block = function.block_specializations.get(&function.opset)?;
    let (op, prefix) = block.operations.split_last()?;
    let kind = match op.r#type.as_str() {
        "mul" => Kind::Multiply,
        "real_div" => Kind::Divide,
        _ => return None,
    };
    if !block.inputs.is_empty()
        || !block.attributes.is_empty()
        || block.outputs.len() != 1
        || !op.attributes.is_empty()
        || !op.blocks.is_empty()
        || op.inputs.len() != 2
        || op.outputs.len() != 1
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
        if input.name.is_empty()
            || ty.data_type != 11
            || feature.r#type.as_ref()?.is_optional
            || array(feature)?.data_type != 65568
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
    for operation in prefix {
        if operation.outputs.len() != 1 {
            return None;
        }
        let output = &operation.outputs[0];
        let input = if operation.r#type == "const" {
            ConstantInput::parse(operation)?
        } else {
            let (source, Input::Constant(input)) = values.get(named(operation, "x")?)? else {
                return None;
            };
            input.view(model, source, operation)?
        };
        if output.name.is_empty()
            || values
                .insert(
                    output.name.clone(),
                    (output.clone(), Input::Constant(input)),
                )
                .is_some()
        {
            return None;
        }
    }
    let left_name = named(op, "x")?;
    let right_name = named(op, "y")?;
    if function
        .inputs
        .iter()
        .any(|input| input.name != left_name && input.name != right_name)
    {
        return None;
    }
    let (left_type, left) = values.get(left_name)?;
    let (right_type, right) = values.get(right_name)?;
    let left_type = tensor(left_type.r#type.as_ref()?)?;
    let right_type = tensor(right_type.r#type.as_ref()?)?;
    let output = &op.outputs[0];
    let target = tensor(output.r#type.as_ref()?)?;
    let feature = &description.output[0];
    if output.name.is_empty()
        || values.contains_key(&output.name)
        || block.outputs[0] != output.name
        || feature.name != output.name
        || feature.r#type.as_ref()?.is_optional
        || array(feature)?.data_type != 65568
        || [left_type, right_type, target]
            .iter()
            .any(|ty| ty.data_type != 11 || !ty.attributes.is_empty())
        || target.rank < 0
        || target.rank as usize != target.dimensions.len()
        || target.dimensions != broadcast(left_type, right_type)?
        || !compatible_shape(target, array(feature)?)
        || !storage_fits(array(feature)?, usize::MAX as u64)
    {
        return None;
    }
    Some(Binary {
        kind,
        left: left.clone(),
        right: right.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::converters::{CoremlMlProgramConverter, GraphConverter};
    use crate::graph::{DataType, Dimension, GraphInfo, Operand, OperandDescriptor, OperandKind};
    use crate::operators::Operation;
    use crate::protos::coreml::specification::feature_type;
    use prost::Message;

    fn model() -> Model {
        let descriptor = OperandDescriptor {
            data_type: DataType::Float32,
            shape: vec![Dimension::Static(2)],
            pending_permutation: vec![],
        };
        let graph = GraphInfo {
            operands: ["a", "b", "result"]
                .iter()
                .enumerate()
                .map(|(id, name)| Operand {
                    name: Some((*name).into()),
                    kind: if id < 2 {
                        OperandKind::Input
                    } else {
                        OperandKind::Output
                    },
                    descriptor: descriptor.clone(),
                })
                .collect(),
            operations: vec![Operation::Mul {
                a: 0,
                b: 1,
                options: None,
                outputs: vec![2],
            }],
            input_operands: vec![0, 1],
            output_operands: vec![2],
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

    fn function(model: &mut Model) -> &mut mil::Function {
        let Some(model::Type::MlProgram(program)) = &mut model.r#type else {
            panic!("not MIL")
        };
        program.functions.get_mut("main").unwrap()
    }

    fn feature_array(
        feature: &mut crate::protos::coreml::specification::FeatureDescription,
    ) -> &mut ArrayFeatureType {
        let Some(feature_type::Type::MultiArrayType(array)) =
            &mut feature.r#type.as_mut().unwrap().r#type
        else {
            panic!("not array")
        };
        array
    }

    #[test]
    fn float32_binary_source_proof_rejects_unproven_programs() {
        let original = model();
        assert!(classify(&original, &original.encode_to_vec()).is_some());
        for mutation in 0..10 {
            let mut changed = original.clone();
            match mutation {
                0 => {
                    function(&mut changed).opset = "CoreML8".into();
                }
                1 => {
                    function(&mut changed)
                        .attributes
                        .insert("unexpected".into(), Default::default());
                }
                2 => {
                    let op = &mut function(&mut changed)
                        .block_specializations
                        .get_mut("CoreML7")
                        .unwrap()
                        .operations[0];
                    op.outputs[0].name = "a".into();
                }
                3 => {
                    let op = &mut function(&mut changed)
                        .block_specializations
                        .get_mut("CoreML7")
                        .unwrap()
                        .operations[0];
                    op.inputs
                        .get_mut("x")
                        .unwrap()
                        .arguments
                        .push(Default::default());
                }
                4 => {
                    changed.description.as_mut().unwrap().input[0]
                        .r#type
                        .as_mut()
                        .unwrap()
                        .is_optional = true;
                }
                5 => {
                    feature_array(&mut changed.description.as_mut().unwrap().input[0]).data_type =
                        65552;
                }
                6 => {
                    feature_array(&mut changed.description.as_mut().unwrap().output[0]).shape =
                        vec![3];
                }
                7 => {
                    let array = feature_array(&mut changed.description.as_mut().unwrap().input[0]);
                    array.shape_flexibility =
                        Some(array_feature_type::ShapeFlexibility::ShapeRange(
                            array_feature_type::ShapeRange {
                                size_ranges: vec![
                                    crate::protos::coreml::specification::SizeRange {
                                        lower_bound: 1,
                                        upper_bound: 4,
                                    },
                                ],
                            },
                        ));
                }
                8 => {
                    let op = &mut function(&mut changed)
                        .block_specializations
                        .get_mut("CoreML7")
                        .unwrap()
                        .operations[0];
                    op.r#type = "add".into();
                }
                9 => {
                    let value = function(&mut changed).inputs[0].clone();
                    function(&mut changed).inputs.push(value);
                }
                _ => unreachable!(),
            }
            assert!(
                classify(&changed, &changed.encode_to_vec()).is_none(),
                "mutation {mutation}"
            );
        }
        let mut unknown = original.encode_to_vec();
        unknown.extend([0xa0, 0x06, 0x01]);
        assert!(classify(&Model::decode(unknown.as_slice()).unwrap(), &unknown).is_none());
    }

    #[test]
    fn float32_binary_feature_extents_respect_32_bit_storage() {
        let limit = u64::from(u32::MAX);
        for (size, expected) in [
            (1, true),
            (1 << 28, true),
            (1 << 30, false),
            (1_i64 << 32, false),
        ] {
            let mut array = ArrayFeatureType {
                shape: vec![size],
                data_type: 65568,
                ..Default::default()
            };
            assert_eq!(storage_fits(&array, limit), expected);
            array.shape = vec![1];
            array.shape_flexibility = Some(array_feature_type::ShapeFlexibility::ShapeRange(
                array_feature_type::ShapeRange {
                    size_ranges: vec![crate::protos::coreml::specification::SizeRange {
                        lower_bound: 0,
                        upper_bound: size,
                    }],
                },
            ));
            assert_eq!(storage_fits(&array, limit), expected);
        }
    }
}
