//! Adapt native scalar values at tensor-valued elementwise consumers.

use super::*;
use crate::protos::coreml::mil_spec::value_type;

fn tensor(value: &NamedValueType) -> Option<&TensorType> {
    match &value.r#type.as_ref()?.r#type {
        Some(value_type::Type::TensorType(value)) => Some(value),
        _ => None,
    }
}

impl CoremlMlProgramConverter {
    pub(super) fn adapt_scalar_tensor_inputs(inputs: &[NamedValueType], block: &mut Block) {
        let types: HashMap<_, _> = inputs
            .iter()
            .chain(
                block
                    .operations
                    .iter()
                    .flat_map(|operation| &operation.outputs),
            )
            .filter_map(|value| tensor(value).map(|ty| (value.name.clone(), ty.clone())))
            .collect();
        let mut names: HashSet<_> = types.keys().cloned().collect();
        let mut views = HashMap::new();
        let operations = std::mem::take(&mut block.operations);
        for mut operation in operations {
            // Scalar constants must remain scalars: other consumers use them as
            // scalar parameters. Only adapt named tensor operands where the
            // converter declares the result using its [1] scalar convention.
            let tensor_inputs: &[&str] = match operation.r#type.as_str() {
                "add" | "sub" | "mul" | "real_div" | "floor_div" | "pow" | "maximum"
                | "minimum" | "equal" | "not_equal" | "greater" | "greater_equal" | "less"
                | "less_equal" | "logical_and" | "logical_or" | "logical_xor" => &["x", "y"],
                // Elementwise unary operators preserve their tensor input's
                // shape. Scalar parameters (clip bounds, activation options,
                // GELU mode, etc.) must retain their original rank and values.
                "abs" | "ceil" | "floor" | "exp" | "log" | "sqrt" | "sign" | "sin" | "cos"
                | "tan" | "erf" | "identity" | "logical_not" | "relu" | "sigmoid" | "tanh"
                | "gelu" | "elu" | "leaky_relu" | "softplus" | "softsign" | "sigmoid_hard"
                | "clip" => &["x"],
                _ => &[],
            };
            let all_scalar = tensor_inputs.iter().all(|key| {
                let Some(argument) = operation.inputs.get(*key) else {
                    return false;
                };
                let [binding] = argument.arguments.as_slice() else {
                    return false;
                };
                let ty = match &binding.binding {
                    Some(Binding::Name(name)) => types.get(name),
                    Some(Binding::Value(value)) => {
                        match value.r#type.as_ref().and_then(|ty| ty.r#type.as_ref()) {
                            Some(value_type::Type::TensorType(ty)) => Some(ty),
                            _ => None,
                        }
                    }
                    None => None,
                };
                ty.is_some_and(|ty| ty.rank == 0 && ty.dimensions.is_empty())
            });
            if !tensor_inputs.is_empty()
                && all_scalar
                && operation.outputs.len() == 1
                && operation.outputs.iter().all(|value| {
                    tensor(value).is_some_and(|ty| {
                        ty.rank == 1
                            && matches!(ty.dimensions.as_slice(), [Dimension {
                            dimension: Some(dimension::Dimension::Constant(size)),
                        }] if size.size == 1)
                    })
                })
            {
                for &key in tensor_inputs {
                    let Some(Binding::Name(name)) = operation
                        .inputs
                        .get_mut(key)
                        .and_then(|argument| argument.arguments.first_mut())
                        .and_then(|binding| binding.binding.as_mut())
                    else {
                        continue;
                    };
                    let Some(ty) = types.get(name).filter(|ty| ty.rank == 0) else {
                        continue;
                    };
                    let view = views.entry(name.clone()).or_insert_with(|| {
                        let stem = format!("{name}_binary_scalar");
                        let mut candidate = stem.clone();
                        let mut suffix = 0;
                        while !names.insert(candidate.clone()) {
                            suffix += 1;
                            candidate = format!("{stem}_{suffix}");
                        }
                        Self::rnn_reshape(block, name, &[1], candidate, ty.data_type)
                    });
                    *name = view.clone();
                }
            }
            block.operations.push(operation);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protos::coreml::mil_spec::DataType;

    #[test]
    fn scalar_unary_views_are_reused_without_changing_scalar_parameters() {
        let scalar = CoremlMlProgramConverter::value_type_for_static_shape(
            "shared".into(),
            DataType::Float32 as i32,
            &[],
        );
        let vector = CoremlMlProgramConverter::value_type_for_static_shape(
            "vector".into(),
            DataType::Float32 as i32,
            &[2],
        );
        let unary = |kind, name: &str| {
            CoremlMlProgramConverter::create_mil_operation(
                kind,
                HashMap::from([(
                    "x".into(),
                    CoremlMlProgramConverter::create_name_argument("shared".into()),
                )]),
                vec![CoremlMlProgramConverter::value_type_for_static_shape(
                    name.into(),
                    DataType::Float32 as i32,
                    &[1],
                )],
            )
        };
        let parameter = CoremlMlProgramConverter::create_mil_operation(
            "clip",
            HashMap::from([
                (
                    "x".into(),
                    CoremlMlProgramConverter::create_name_argument("vector".into()),
                ),
                (
                    "alpha".into(),
                    CoremlMlProgramConverter::create_name_argument("shared".into()),
                ),
                (
                    "beta".into(),
                    CoremlMlProgramConverter::create_immediate_float(8.),
                ),
            ]),
            vec![CoremlMlProgramConverter::value_type_for_static_shape(
                "clipped".into(),
                DataType::Float32 as i32,
                &[2],
            )],
        );
        let mut block = Block {
            operations: vec![
                unary("sqrt", "root"),
                unary("abs", "absolute"),
                parameter.clone(),
            ],
            ..Default::default()
        };
        CoremlMlProgramConverter::adapt_scalar_tensor_inputs(&[scalar.clone(), vector], &mut block);
        assert_eq!(
            block
                .operations
                .iter()
                .filter(|op| op.r#type == "reshape")
                .count(),
            1
        );
        assert_eq!(
            block.operations[1].inputs["x"],
            block.operations[2].inputs["x"]
        );
        assert_ne!(block.operations[1].inputs["x"], parameter.inputs["alpha"]);
        assert_eq!(block.operations[3], parameter);
        assert_eq!(tensor(&scalar).unwrap().rank, 0);
    }

    #[test]
    fn scalar_views_require_a_proven_elementwise_tensor_shape() {
        for (kind, input_shape, output_shape) in [
            ("sqrt", &[][..], &[][..]),
            ("sqrt", &[][..], &[2][..]),
            ("sqrt", &[1][..], &[1][..]),
            ("reduce_sum", &[][..], &[1][..]),
            ("unrecognized", &[][..], &[1][..]),
        ] {
            let input = CoremlMlProgramConverter::value_type_for_static_shape(
                "source".into(),
                DataType::Float32 as i32,
                input_shape,
            );
            let operation = CoremlMlProgramConverter::create_mil_operation(
                kind,
                HashMap::from([(
                    "x".into(),
                    CoremlMlProgramConverter::create_name_argument("source".into()),
                )]),
                vec![CoremlMlProgramConverter::value_type_for_static_shape(
                    "result".into(),
                    DataType::Float32 as i32,
                    output_shape,
                )],
            );
            let mut block = Block {
                operations: vec![operation.clone()],
                ..Default::default()
            };
            CoremlMlProgramConverter::adapt_scalar_tensor_inputs(&[input], &mut block);
            assert_eq!(block.operations, [operation], "{kind}");
        }
    }

    #[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
    #[test]
    fn exported_scalar_unary_models_compile_without_typed_execution() {
        use crate::converters::GraphConverter;
        use crate::graph::{
            ConstantData, DataType as GraphDataType, GraphInfo, Operand, OperandDescriptor,
            OperandKind,
        };
        use crate::operators::Operation;
        use objc::{msg_send, sel, sel_impl};
        for dtype in [GraphDataType::Float32, GraphDataType::Float16] {
            for constant in [false, true] {
                for name in [
                    "sqrt",
                    "exp",
                    "log",
                    "abs",
                    "neg",
                    "tanh",
                    "gelu",
                    "relu",
                    "ceil",
                    "floor",
                    "roundEven",
                    "reciprocal",
                    "sign",
                    "sin",
                    "cos",
                    "tan",
                    "erf",
                    "sigmoid",
                    "elu",
                    "leakyRelu",
                    "softplus",
                    "softsign",
                    "hardSigmoid",
                    "hardSwish",
                    "clamp",
                    "identity",
                    "cast",
                    "castSame",
                    "isNaN",
                    "isInfinite",
                ] {
                    let descriptor = OperandDescriptor {
                        data_type: dtype,
                        shape: vec![],
                        pending_permutation: vec![],
                    };
                    let output_dtype = match name {
                        "cast" if dtype == GraphDataType::Float32 => GraphDataType::Float16,
                        "cast" => GraphDataType::Float32,
                        "isNaN" | "isInfinite" => GraphDataType::Uint8,
                        _ => dtype,
                    };
                    let operation = if matches!(name, "cast" | "castSame") {
                        Operation::Cast {
                            input: 0,
                            data_type: output_dtype.try_into().unwrap(),
                            outputs: vec![1],
                            options: None,
                        }
                    } else {
                        Operation::from_json_attributes(name, &[0], &[1], &serde_json::json!({}))
                            .unwrap()
                    };
                    let mut graph = GraphInfo {
                        operands: vec![
                            Operand {
                                name: Some("input".into()),
                                kind: if constant {
                                    OperandKind::Constant
                                } else {
                                    OperandKind::Input
                                },
                                descriptor: descriptor.clone(),
                            },
                            Operand {
                                name: Some("result".into()),
                                kind: OperandKind::Output,
                                descriptor: OperandDescriptor {
                                    data_type: output_dtype,
                                    ..descriptor
                                },
                            },
                        ],
                        operations: vec![operation],
                        input_operands: if constant { vec![] } else { vec![0] },
                        output_operands: vec![1],
                        ..Default::default()
                    };
                    if constant {
                        graph.constant_operand_ids_to_handles.insert(
                            0,
                            ConstantData {
                                data: if dtype == GraphDataType::Float32 {
                                    0.25_f32.to_le_bytes().to_vec()
                                } else {
                                    0x3400_u16.to_le_bytes().to_vec()
                                },
                                label: None,
                            },
                        );
                    }
                    let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
                    let cache = tempfile::tempdir().unwrap();
                    let path = cache.path().join("scalar.mlmodelc");
                    objc::rc::autoreleasepool(|| unsafe {
                        // Compile the emitted model directly. MLContext's typed
                        // stages must not be able to hide an invalid export.
                        let (url, _, source) =
                            crate::executors::coreml::prepare_compiled_model_with_weights(
                                &converted.data,
                                converted.weights_data.as_deref(),
                                Some(&path),
                            )
                            .unwrap_or_else(|error| {
                                panic!("{name}/{dtype:?}/constant={constant}: {error}")
                            });
                        let _: () = msg_send![url, release];
                        drop(source);
                    });
                    assert!(path.is_dir());
                }
            }
        }
    }

    #[test]
    fn scalar_views_are_reused_without_changing_attributes_or_other_consumers() {
        let scalar = CoremlMlProgramConverter::value_type_for_static_shape(
            "value".into(),
            DataType::Float32 as i32,
            &[],
        );
        let occupied = CoremlMlProgramConverter::value_type_for_static_shape(
            "value_binary_scalar".into(),
            DataType::Float32 as i32,
            &[1],
        );
        let binary = |name: &str| {
            CoremlMlProgramConverter::create_mil_operation(
                "add",
                HashMap::from([
                    (
                        "x".into(),
                        CoremlMlProgramConverter::create_name_argument("value".into()),
                    ),
                    (
                        "y".into(),
                        CoremlMlProgramConverter::create_immediate_float(2.),
                    ),
                ]),
                vec![CoremlMlProgramConverter::value_type_for_static_shape(
                    name.into(),
                    DataType::Float32 as i32,
                    &[1],
                )],
            )
        };
        let other = CoremlMlProgramConverter::create_mil_operation(
            "clip",
            HashMap::from([
                (
                    "x".into(),
                    CoremlMlProgramConverter::create_name_argument("value_binary_scalar".into()),
                ),
                (
                    "alpha".into(),
                    CoremlMlProgramConverter::create_name_argument("value".into()),
                ),
                (
                    "beta".into(),
                    CoremlMlProgramConverter::create_immediate_float(8.),
                ),
            ]),
            vec![CoremlMlProgramConverter::value_type_for_static_shape(
                "clipped".into(),
                DataType::Float32 as i32,
                &[1],
            )],
        );
        let first = binary("sum1");
        let immediate = first.inputs["y"].clone();
        let mut block = Block {
            operations: vec![first, binary("sum2"), other.clone()],
            ..Default::default()
        };
        CoremlMlProgramConverter::adapt_scalar_tensor_inputs(
            &[scalar.clone(), occupied],
            &mut block,
        );
        assert_eq!(
            block
                .operations
                .iter()
                .filter(|op| op.r#type == "reshape")
                .count(),
            1
        );
        assert_eq!(block.operations[0].outputs[0].name, "value_binary_scalar_1");
        assert_eq!(
            block.operations[1].inputs["x"],
            block.operations[2].inputs["x"]
        );
        assert_eq!(block.operations[1].inputs["y"], immediate);
        assert_eq!(block.operations[3], other);
        assert_eq!(tensor(&scalar).unwrap().rank, 0);
    }

    #[test]
    fn mixed_scalar_and_rank_one_binary_keeps_native_broadcasting() {
        let scalar = CoremlMlProgramConverter::value_type_for_static_shape(
            "scalar".into(),
            DataType::Int32 as i32,
            &[],
        );
        let vector = CoremlMlProgramConverter::value_type_for_static_shape(
            "shape".into(),
            DataType::Int32 as i32,
            &[1],
        );
        for reverse in [false, true] {
            let (x, y) = if reverse {
                ("scalar", "shape")
            } else {
                ("shape", "scalar")
            };
            let operation = CoremlMlProgramConverter::create_mil_operation(
                "sub",
                HashMap::from([
                    (
                        "x".into(),
                        CoremlMlProgramConverter::create_name_argument(x.into()),
                    ),
                    (
                        "y".into(),
                        CoremlMlProgramConverter::create_name_argument(y.into()),
                    ),
                ]),
                vec![CoremlMlProgramConverter::value_type_for_static_shape(
                    "result".into(),
                    DataType::Int32 as i32,
                    &[1],
                )],
            );
            let mut block = Block {
                operations: vec![operation.clone()],
                ..Default::default()
            };
            CoremlMlProgramConverter::adapt_scalar_tensor_inputs(
                &[scalar.clone(), vector.clone()],
                &mut block,
            );
            assert_eq!(block.operations, [operation]);
        }
    }
}
