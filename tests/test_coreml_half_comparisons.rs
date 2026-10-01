//! Half comparisons retain subnormal ordering without changing WebNN types.
use prost::Message;
use rustnn::converters::{CoremlMlProgramConverter, GraphConverter};
use rustnn::graph::{DataType, Dimension, GraphInfo, Operand, OperandDescriptor, OperandKind};
use rustnn::operator_enums::MLOperandDataType;
use rustnn::operators::Operation;
use rustnn::protos::coreml::{
    mil_spec as mil,
    specification::{Model, model},
};
use std::collections::HashMap;

fn operand(name: &str, kind: OperandKind, dtype: DataType, shape: &[Dimension]) -> Operand {
    Operand {
        name: Some(name.into()),
        kind,
        descriptor: OperandDescriptor {
            data_type: dtype,
            shape: shape.to_vec(),
            pending_permutation: vec![],
        },
    }
}

fn comparison(index: usize, a: u32, b: u32, output: u32) -> Operation {
    let outputs = vec![output];
    match index {
        0 => Operation::Equal {
            a,
            b,
            outputs,
            options: None,
        },
        1 => Operation::NotEqual {
            a,
            b,
            outputs,
            options: None,
        },
        2 => Operation::Greater {
            a,
            b,
            outputs,
            options: None,
        },
        3 => Operation::GreaterOrEqual {
            a,
            b,
            outputs,
            options: None,
        },
        4 => Operation::Lesser {
            a,
            b,
            outputs,
            options: None,
        },
        5 => Operation::LesserOrEqual {
            a,
            b,
            outputs,
            options: None,
        },
        _ => unreachable!(),
    }
}

fn graph(index: usize, shape: &[Dimension]) -> GraphInfo {
    GraphInfo {
        operands: vec![
            operand("x", OperandKind::Input, DataType::Float16, shape),
            operand("threshold", OperandKind::Input, DataType::Float16, &[]),
            operand(
                "condition",
                OperandKind::Intermediate,
                DataType::Uint8,
                shape,
            ),
            operand("one", OperandKind::Input, DataType::Float32, &[]),
            operand("zero", OperandKind::Input, DataType::Float32, &[]),
            operand("selected", OperandKind::Output, DataType::Float32, shape),
        ],
        input_operands: vec![0, 1, 3, 4],
        output_operands: vec![5],
        operations: vec![
            comparison(index, 0, 1, 2),
            Operation::Where {
                condition: 2,
                true_value: 3,
                false_value: 4,
                options: None,
                outputs: vec![5],
            },
        ],
        ..Default::default()
    }
}

fn tensor(value: &mil::NamedValueType) -> &mil::TensorType {
    let Some(mil::value_type::Type::TensorType(tensor)) =
        value.r#type.as_ref().unwrap().r#type.as_ref()
    else {
        panic!("tensor")
    };
    tensor
}

fn visit(model: &Model, callback: &mut impl FnMut(&mil::Function, &mil::Block)) {
    match model.r#type.as_ref().unwrap() {
        model::Type::MlProgram(program) => {
            for function in program.functions.values() {
                for block in function.block_specializations.values() {
                    callback(function, block);
                }
            }
        }
        model::Type::Pipeline(pipeline) => {
            for child in &pipeline.models {
                visit(child, callback)
            }
        }
        _ => panic!("native program"),
    }
}

fn binding(operation: &mil::Operation, key: &str) -> String {
    let Some(mil::argument::binding::Binding::Name(name)) =
        operation.inputs[key].arguments[0].binding.as_ref()
    else {
        panic!("named input")
    };
    name.clone()
}

fn assert_constant_has_materialized_pure_widening(model: &Model, kernel: &str) {
    let mut stages = Vec::new();
    visit(model, &mut |function, block| {
        stages.push((function.clone(), block.clone()))
    });
    let mut materialized = Vec::new();
    for (index, (_, block)) in stages.iter().enumerate() {
        if !block.operations.iter().any(|operation| {
            operation.r#type == "const"
                && operation.outputs.iter().any(|value| {
                    tensor(value).data_type == mil::DataType::Float16 as i32
                        && tensor(value).rank == 1
                })
        }) {
            continue;
        }
        for value in block
            .operations
            .iter()
            .flat_map(|operation| &operation.outputs)
        {
            if block.outputs.contains(&value.name)
                && tensor(value).data_type == mil::DataType::Float16 as i32
            {
                materialized.push((index, value.name.clone()));
            }
        }
    }
    assert!(
        !materialized.is_empty(),
        "stored Half constant must be a real feature before widening"
    );
    let mut pure = Vec::new();
    for (index, (function, block)) in stages.iter().enumerate() {
        if block.operations.len() != 1 || block.operations[0].r#type != "cast" {
            continue;
        }
        let cast = &block.operations[0];
        let input = binding(cast, "x");
        if materialized
            .iter()
            .any(|(producer, name)| *producer < index && *name == input)
            && function.inputs.iter().any(|value| {
                value.name == input && tensor(value).data_type == mil::DataType::Float16 as i32
            })
            && tensor(&cast.outputs[0]).data_type == mil::DataType::Float32 as i32
        {
            pure.push((index, cast.outputs[0].name.clone()));
        }
    }
    assert!(
        !pure.is_empty(),
        "materialized constant must feed a Cast-only native child"
    );
    assert!(
        stages
            .iter()
            .enumerate()
            .any(|(index, (function, block))| block
                .operations
                .iter()
                .any(|operation| operation.r#type == kernel)
                && function.inputs.iter().any(|value| pure
                    .iter()
                    .any(|(producer, name)| *producer < index && *name == value.name))),
        "{kernel} must consume the real Float32 feature rather than clone/fuse the Half constant cast"
    );
}

#[test]
fn half_constant_predicates_materialize_ingress_without_widening_storage() {
    for index in 0..6 {
        for explicit_wide_control in [false, true] {
            let mut graph = graph(index, &[Dimension::Static(4)]);
            graph.operands[0].kind = OperandKind::Constant;
            graph.input_operands.remove(0);
            graph.constant_operand_ids_to_handles.insert(
                0,
                rustnn::graph::ConstantData {
                    data: [1u16, 0x8001, 0, 0x7e00]
                        .into_iter()
                        .flat_map(u16::to_le_bytes)
                        .collect(),
                    label: None,
                },
            );
            if explicit_wide_control {
                graph.operands.push(operand(
                    "wide_x",
                    OperandKind::Intermediate,
                    DataType::Float32,
                    &[Dimension::Static(4)],
                ));
                graph.operands.push(operand(
                    "wide_threshold",
                    OperandKind::Intermediate,
                    DataType::Float32,
                    &[],
                ));
                graph.operations[0] = comparison(index, 6, 7, 2);
                graph.operations.splice(
                    0..0,
                    [
                        Operation::Cast {
                            input: 0,
                            data_type: MLOperandDataType::Float32,
                            outputs: vec![6],
                            options: None,
                        },
                        Operation::Cast {
                            input: 1,
                            data_type: MLOperandDataType::Float32,
                            outputs: vec![7],
                            options: None,
                        },
                    ],
                );
            }
            let original = serde_json::to_value(&graph).unwrap();
            let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
            assert_eq!(serde_json::to_value(&graph).unwrap(), original);
            let model = Model::decode(converted.data.as_slice()).unwrap();
            let kernel = match index {
                0 | 1 => "equal",
                2 => "greater",
                3 => "greater_equal",
                4 => "less",
                5 => "less_equal",
                _ => unreachable!(),
            };
            assert_constant_has_materialized_pure_widening(&model, kernel);
            visit(&model, &mut |_, block| {
                for constant in block
                    .operations
                    .iter()
                    .filter(|operation| operation.r#type == "const")
                {
                    for value in &constant.outputs {
                        if value.name == "x" {
                            assert_eq!(tensor(value).data_type, mil::DataType::Float16 as i32);
                        }
                    }
                }
            });
        }
    }
}

#[test]
fn half_constant_arg_extrema_materialize_ingress_and_keep_integer_result() {
    for maximum in [false, true] {
        for explicit_wide_control in [false, true] {
            let mut graph = GraphInfo {
                operands: vec![
                    operand(
                        "x",
                        OperandKind::Constant,
                        DataType::Float16,
                        &[Dimension::Static(4)],
                    ),
                    operand("result", OperandKind::Output, DataType::Int32, &[]),
                ],
                output_operands: vec![1],
                operations: vec![
                    Operation::from_json_attributes(
                        if maximum { "argMax" } else { "argMin" },
                        &[0],
                        &[1],
                        &serde_json::json!({"axis":0}),
                    )
                    .unwrap(),
                ],
                ..Default::default()
            };
            graph.constant_operand_ids_to_handles.insert(
                0,
                rustnn::graph::ConstantData {
                    data: [1u16, 2, 0x8001, 0x8002]
                        .into_iter()
                        .flat_map(u16::to_le_bytes)
                        .collect(),
                    label: None,
                },
            );
            if explicit_wide_control {
                graph.operands.push(operand(
                    "wide_x",
                    OperandKind::Intermediate,
                    DataType::Float32,
                    &[Dimension::Static(4)],
                ));
                graph.operations[0] = Operation::from_json_attributes(
                    if maximum { "argMax" } else { "argMin" },
                    &[2],
                    &[1],
                    &serde_json::json!({"axis":0}),
                )
                .unwrap();
                graph.operations.insert(
                    0,
                    Operation::Cast {
                        input: 0,
                        data_type: MLOperandDataType::Float32,
                        outputs: vec![2],
                        options: None,
                    },
                );
            }
            let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
            let model = Model::decode(converted.data.as_slice()).unwrap();
            assert_constant_has_materialized_pure_widening(
                &model,
                if maximum {
                    "reduce_argmax"
                } else {
                    "reduce_argmin"
                },
            );
            visit(&model, &mut |_, block| {
                for operation in &block.operations {
                    if matches!(operation.r#type.as_str(), "reduce_argmin" | "reduce_argmax") {
                        assert_eq!(
                            tensor(&operation.outputs[0]).data_type,
                            mil::DataType::Int32 as i32
                        );
                    }
                }
            });
        }
    }
}

#[test]
fn half_predicates_use_fp32_sources_and_retain_original_descriptors() {
    for index in 0..6 {
        for shape in [
            vec![],
            vec![Dimension::Static(11)],
            vec![Dimension::Static(1), Dimension::Static(11)],
        ] {
            let graph = graph(index, &shape);
            let unchanged = serde_json::to_value(&graph).unwrap();
            let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
            assert_eq!(serde_json::to_value(&graph).unwrap(), unchanged);
            let model = Model::decode(converted.data.as_slice()).unwrap();
            assert!(matches!(model.r#type, Some(model::Type::Pipeline(_))));
            let mut comparisons = 0;
            visit(&model, &mut |function, block| {
                let mut types: HashMap<_, _> = function
                    .inputs
                    .iter()
                    .map(|v| (v.name.clone(), tensor(v).data_type))
                    .collect();
                for operation in &block.operations {
                    if matches!(
                        operation.r#type.as_str(),
                        "equal" | "greater" | "greater_equal" | "less" | "less_equal"
                    ) {
                        for key in ["x", "y"] {
                            assert_eq!(
                                types[&binding(operation, key)],
                                mil::DataType::Float32 as i32
                            );
                        }
                        comparisons += 1;
                    }
                    for output in &operation.outputs {
                        types.insert(output.name.clone(), tensor(output).data_type);
                    }
                }
            });
            assert_eq!(comparisons, 1);
        }
    }
}

#[test]
fn where_uses_native_boolean_producer_without_uint8_round_trip() {
    let model = Model::decode(
        CoremlMlProgramConverter
            .convert(&graph(0, &[Dimension::Static(11)]))
            .unwrap()
            .data
            .as_slice(),
    )
    .unwrap();
    let mut selected = 0;
    visit(&model, &mut |_, block| {
        let producers: HashMap<_, _> = block
            .operations
            .iter()
            .flat_map(|operation| {
                operation
                    .outputs
                    .iter()
                    .map(move |v| (v.name.clone(), operation))
            })
            .collect();
        for operation in &block.operations {
            if operation.r#type == "select" {
                let condition = binding(operation, "cond");
                assert_eq!(producers[&condition].r#type, "equal");
                selected += 1;
            }
        }
    });
    assert_eq!(selected, 1);
}

#[test]
fn floating_not_equal_materializes_equality_before_negation() {
    for dtype in [DataType::Float16, DataType::Float32] {
        let mut graph = graph(1, &[Dimension::Static(11)]);
        graph.operands[0].descriptor.data_type = dtype;
        graph.operands[1].descriptor.data_type = dtype;
        let model = Model::decode(
            CoremlMlProgramConverter
                .convert(&graph)
                .unwrap()
                .data
                .as_slice(),
        )
        .unwrap();
        let mut equal_stages = Vec::new();
        let mut not_stages = Vec::new();
        let mut stage = 0;
        visit(&model, &mut |_, block| {
            if block.operations.iter().any(|op| op.r#type == "equal") {
                equal_stages.push(stage);
                assert!(block.outputs.iter().any(|name| {
                    block.operations.iter().any(|op| {
                        op.r#type == "cast"
                            && op.outputs.iter().any(|value| {
                                value.name == *name
                                    && tensor(value).data_type == mil::DataType::Int32 as i32
                            })
                    })
                }));
            }
            if block.operations.iter().any(|op| op.r#type == "logical_not") {
                not_stages.push(stage);
            }
            stage += 1;
        });
        assert_eq!(equal_stages.len(), 1);
        assert_eq!(not_stages.len(), 1);
        assert!(equal_stages[0] < not_stages[0]);
    }
}

#[test]
fn explicit_condition_dtype_casts_are_not_bypassed() {
    let mut graph = graph(0, &[Dimension::Static(11)]);
    graph.operands.push(operand(
        "float_condition",
        OperandKind::Intermediate,
        DataType::Float32,
        &[Dimension::Static(11)],
    ));
    graph.operands.push(operand(
        "byte_condition",
        OperandKind::Intermediate,
        DataType::Uint8,
        &[Dimension::Static(11)],
    ));
    graph.operations.insert(
        1,
        Operation::Cast {
            input: 2,
            data_type: MLOperandDataType::Float32,
            options: None,
            outputs: vec![6],
        },
    );
    graph.operations.insert(
        2,
        Operation::Cast {
            input: 6,
            data_type: MLOperandDataType::Uint8,
            options: None,
            outputs: vec![7],
        },
    );
    if let Operation::Where { condition, .. } = &mut graph.operations[3] {
        *condition = 7;
    } else {
        unreachable!()
    };
    let model = Model::decode(
        CoremlMlProgramConverter
            .convert(&graph)
            .unwrap()
            .data
            .as_slice(),
    )
    .unwrap();
    let mut retained = 0;
    visit(&model, &mut |_, block| {
        for operation in &block.operations {
            if operation.r#type == "cast"
                && operation
                    .outputs
                    .iter()
                    .any(|v| v.name.contains("byte_condition"))
            {
                retained += 1;
            }
        }
    });
    assert!(retained > 0, "explicit source cast retained");
}

#[test]
fn half_arg_extrema_compute_on_fp32_without_changing_index_type() {
    for maximum in [false, true] {
        let shape = [Dimension::Static(2), Dimension::Static(11)];
        let graph = GraphInfo {
            operands: vec![
                operand("x", OperandKind::Input, DataType::Float16, &shape),
                operand(
                    "result",
                    OperandKind::Output,
                    DataType::Int32,
                    &[Dimension::Static(2)],
                ),
            ],
            input_operands: vec![0],
            output_operands: vec![1],
            operations: vec![if maximum {
                Operation::ArgMax {
                    input: 0,
                    axis: 1,
                    options: None,
                    outputs: vec![1],
                }
            } else {
                Operation::ArgMin {
                    input: 0,
                    axis: 1,
                    options: None,
                    outputs: vec![1],
                }
            }],
            ..Default::default()
        };
        let model = Model::decode(
            CoremlMlProgramConverter
                .convert(&graph)
                .unwrap()
                .data
                .as_slice(),
        )
        .unwrap();
        let mut seen = 0;
        visit(&model, &mut |function, block| {
            let mut types: HashMap<_, _> = function
                .inputs
                .iter()
                .map(|v| (v.name.clone(), tensor(v).data_type))
                .collect();
            for operation in &block.operations {
                if operation.r#type == "cast" {
                    let source = types[&binding(operation, "x")];
                    let target = tensor(&operation.outputs[0]).data_type;
                    assert_ne!(
                        (source, target),
                        (mil::DataType::Int32 as i32, mil::DataType::Int32 as i32),
                        "arg extrema must name the native Int32 result directly"
                    );
                }
                if matches!(operation.r#type.as_str(), "reduce_argmax" | "reduce_argmin") {
                    assert_eq!(
                        types[&binding(operation, "x")],
                        mil::DataType::Float32 as i32
                    );
                    assert_eq!(
                        tensor(&operation.outputs[0]).data_type,
                        mil::DataType::Int32 as i32
                    );
                    seen += 1;
                }
                for output in &operation.outputs {
                    types.insert(output.name.clone(), tensor(output).data_type);
                }
            }
        });
        assert_eq!(seen, 1);
    }
}
