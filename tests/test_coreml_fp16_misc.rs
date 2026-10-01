//! Half PReLU/GEMM/triangular compute must retain represented input values.

use prost::Message;
use rustnn::converters::{CoremlMlProgramConverter, GraphConverter};
use rustnn::graph::{
    ConstantData, DataType, Dimension, GraphInfo, Operand, OperandDescriptor, OperandKind,
};
use rustnn::operators::Operation;
use rustnn::protos::coreml::{mil_spec, specification};
use serde_json::json;

fn operand(name: &str, shape: &[u32], kind: OperandKind) -> Operand {
    Operand {
        name: Some(name.into()),
        kind,
        descriptor: OperandDescriptor {
            data_type: DataType::Float16,
            shape: shape.iter().copied().map(Dimension::Static).collect(),
            pending_permutation: vec![],
        },
    }
}

fn operations(graph: &GraphInfo) -> Vec<mil_spec::Operation> {
    let model = specification::Model::decode(
        CoremlMlProgramConverter
            .convert(graph)
            .unwrap()
            .data
            .as_slice(),
    )
    .unwrap();
    fn collect(model: specification::Model, result: &mut Vec<mil_spec::Operation>) {
        match model.r#type.unwrap() {
            specification::model::Type::MlProgram(program) => {
                let function = &program.functions["main"];
                let mut names: std::collections::HashSet<_> =
                    function.inputs.iter().map(|value| &value.name).collect();
                for output in function.block_specializations[&function.opset]
                    .operations
                    .iter()
                    .flat_map(|operation| &operation.outputs)
                {
                    assert!(
                        names.insert(&output.name),
                        "each native child has unique SSA names"
                    );
                }
                result.extend(
                    function.block_specializations[&function.opset]
                        .operations
                        .clone(),
                );
            }
            specification::model::Type::Pipeline(pipeline) => {
                for child in pipeline.models {
                    collect(child, result);
                }
            }
            _ => panic!("expected MLProgram or Pipeline"),
        }
    }
    let mut result = vec![];
    collect(model, &mut result);
    result
}

fn tensor_dtype(value: &mil_spec::NamedValueType) -> i32 {
    let Some(mil_spec::value_type::Type::TensorType(tensor)) =
        &value.r#type.as_ref().unwrap().r#type
    else {
        panic!("expected tensor")
    };
    tensor.data_type
}

#[test]
fn half_prelu_widens_the_multiply_and_select_as_one_operation() {
    for (shape, alpha, output) in [
        (vec![2, 3], vec![], vec![2, 3]),
        (vec![1, 3], vec![2, 1], vec![2, 3]),
        (vec![], vec![], vec![]),
    ] {
        let graph = GraphInfo {
            operands: vec![
                operand("input", &shape, OperandKind::Input),
                operand("slope", &alpha, OperandKind::Input),
                operand("result", &output, OperandKind::Output),
            ],
            input_operands: vec![0, 1],
            output_operands: vec![2],
            operations: vec![
                Operation::from_json_attributes("prelu", &[0, 1], &[2], &json!({})).unwrap(),
            ],
            ..Default::default()
        };
        let operations = operations(&graph);
        for operation in operations
            .iter()
            .filter(|op| matches!(op.r#type.as_str(), "mul" | "select"))
        {
            assert!(
                operation
                    .outputs
                    .iter()
                    .all(|value| tensor_dtype(value) == mil_spec::DataType::Float32 as i32),
                "{}",
                operation.r#type
            );
        }
        assert!(operations.iter().any(|op| op.r#type == "cast"
            && op.outputs.iter().any(|value| value.name == "result"
                && tensor_dtype(value) == mil_spec::DataType::Float16 as i32)));
    }
}

#[test]
fn half_gemm_does_not_round_the_dot_product_before_alpha_and_bias() {
    let graph = GraphInfo {
        operands: vec![
            operand("a", &[1, 2], OperandKind::Input),
            operand("b", &[2, 1], OperandKind::Input),
            operand("bias", &[], OperandKind::Input),
            operand("result", &[1, 1], OperandKind::Output),
        ],
        input_operands: vec![0, 1, 2],
        output_operands: vec![3],
        operations: vec![
            Operation::from_json_attributes(
                "gemm",
                &[0, 1],
                &[3],
                &json!({"alpha":0.5,"beta":0.33333,"c":2}),
            )
            .unwrap(),
        ],
        ..Default::default()
    };
    let operations = operations(&graph);
    for operation in operations
        .iter()
        .filter(|op| matches!(op.r#type.as_str(), "matmul" | "mul" | "add"))
    {
        assert!(
            operation
                .outputs
                .iter()
                .all(|value| tensor_dtype(value) == mil_spec::DataType::Float32 as i32),
            "{}",
            operation.r#type
        );
    }
    let half_results: Vec<_> = operations
        .iter()
        .flat_map(|op| &op.outputs)
        .filter(|value| tensor_dtype(value) == mil_spec::DataType::Float16 as i32)
        .collect();
    assert_eq!(half_results.len(), 1, "GEMM rounds only its logical result");
    assert_eq!(half_results[0].name, "result");
}

#[test]
fn gemm_keeps_representable_parameters_adjacent_to_one() {
    for alpha in [
        f32::from_bits(1.0f32.to_bits() - 1),
        f32::from_bits(1.0f32.to_bits() + 1),
    ] {
        let mut graph = GraphInfo {
            operands: vec![
                operand("a", &[1, 1], OperandKind::Input),
                operand("b", &[1, 1], OperandKind::Input),
                operand("result", &[1, 1], OperandKind::Output),
            ],
            input_operands: vec![0, 1],
            output_operands: vec![2],
            operations: vec![
                Operation::from_json_attributes("gemm", &[0, 1], &[2], &json!({"alpha":alpha}))
                    .unwrap(),
            ],
            ..Default::default()
        };
        for operand in &mut graph.operands {
            operand.descriptor.data_type = DataType::Float32;
        }
        assert!(
            operations(&graph).iter().any(|op| op.r#type == "mul"),
            "alpha={alpha}"
        );
    }
}

#[test]
fn half_gemm_keeps_weight_blob_half_and_rounds_source_options_before_widening() {
    let mut graph = GraphInfo {
        operands: vec![
            operand("a", &[1, 2], OperandKind::Input),
            operand("b", &[2, 1], OperandKind::Constant),
            operand("result", &[1, 1], OperandKind::Output),
        ],
        input_operands: vec![0],
        output_operands: vec![2],
        operations: vec![
            Operation::from_json_attributes("gemm", &[0, 1], &[2], &json!({"alpha":0.33333}))
                .unwrap(),
        ],
        ..Default::default()
    };
    let bytes: Vec<u8> = [0x3c00u16, 1]
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    graph.constant_operand_ids_to_handles.insert(
        1,
        ConstantData {
            data: bytes.clone(),
            label: None,
        },
    );
    let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
    assert!(
        converted
            .weights_data
            .as_ref()
            .unwrap()
            .windows(bytes.len())
            .any(|window| window == bytes)
    );
    let ops = operations(&graph);
    for operation in ops.iter().filter(|op| op.r#type == "const") {
        assert!(
            operation
                .outputs
                .iter()
                .all(|value| tensor_dtype(value) == mil_spec::DataType::Float16 as i32)
        );
    }
    let mul = ops.iter().find(|op| op.r#type == "mul").unwrap();
    let Some(mil_spec::argument::binding::Binding::Value(value)) =
        &mul.inputs["y"].arguments[0].binding
    else {
        panic!("expected coefficient");
    };
    let Some(mil_spec::value::Value::ImmediateValue(value)) = &value.value else {
        panic!("expected immediate");
    };
    let Some(mil_spec::value::immediate_value::Value::Tensor(value)) = &value.value else {
        panic!("expected tensor");
    };
    let Some(mil_spec::tensor_value::Value::Floats(value)) = &value.value else {
        panic!("expected Float32 represented parameter");
    };
    assert_eq!(value.values, [half::f16::from_f32(0.33333).to_f32()]);
}

#[test]
fn half_triangular_native_band_and_masked_select_preserve_kept_values() {
    for upper in [false, true] {
        for diagonal in [-1, 0, 1] {
            let graph = GraphInfo {
                operands: vec![
                    operand("input", &[2, 3, 4], OperandKind::Input),
                    operand("result", &[2, 3, 4], OperandKind::Output),
                ],
                input_operands: vec![0],
                output_operands: vec![1],
                operations: vec![
                    Operation::from_json_attributes(
                        "triangular",
                        &[0],
                        &[1],
                        &json!({"upper":upper,"diagonal":diagonal}),
                    )
                    .unwrap(),
                ],
                ..Default::default()
            };
            for operation in operations(&graph)
                .iter()
                .filter(|op| matches!(op.r#type.as_str(), "band_part" | "select"))
            {
                assert!(
                    operation
                        .outputs
                        .iter()
                        .all(|value| tensor_dtype(value) == mil_spec::DataType::Float32 as i32),
                    "upper={upper}, diagonal={diagonal}"
                );
            }
        }
    }
}
