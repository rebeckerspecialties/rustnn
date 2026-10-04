//! Exact copies are selected by public graph provenance, not native output values.

use std::collections::HashMap;

use prost::Message;
use rustnn::converters::{CoremlMlProgramConverter, GraphConverter};
use rustnn::graph::{
    ConstantData, DataType, Dimension, GraphInfo, Operand, OperandDescriptor, OperandKind,
};
use rustnn::operator_enums::MLOperandDataType;
use rustnn::operator_options::{MLDimension, MLSliceOptions, MLTransposeOptions};
use rustnn::operators::Operation;
use rustnn::protos::coreml::specification;
use serde_json::Value;

const INPUT_COPIES: &str = "rustnn.webnn.output_passthroughs";
const CONSTANT_COPIES: &str = "rustnn.webnn.output_constant_copies";

#[derive(Clone, Copy)]
enum CopyOp {
    Identity,
    Cast,
    Transpose,
    Slice,
    Reshape,
}

fn descriptor() -> OperandDescriptor {
    OperandDescriptor {
        data_type: DataType::Int32,
        shape: vec![Dimension::Static(6)],
        pending_permutation: vec![],
    }
}

fn copy_operation(kind: CopyOp, input: u32, output: u32) -> Operation {
    match kind {
        CopyOp::Identity => Operation::Identity {
            input,
            options: None,
            outputs: vec![output],
        },
        CopyOp::Cast => Operation::Cast {
            input,
            data_type: MLOperandDataType::Int32,
            options: None,
            outputs: vec![output],
        },
        CopyOp::Transpose => Operation::Transpose {
            input,
            options: None,
            outputs: vec![output],
        },
        CopyOp::Slice => Operation::Slice {
            input,
            starts: vec![0],
            sizes: vec![MLDimension::Static(6)],
            options: None,
            outputs: vec![output],
        },
        CopyOp::Reshape => Operation::Reshape {
            input,
            new_shape: vec![MLDimension::Static(6)],
            options: None,
            outputs: vec![output],
        },
    }
}

fn exact_bytes() -> Vec<u8> {
    [0i32, -1, 16_777_217, -16_777_217, i32::MIN, i32::MAX]
        .into_iter()
        .flat_map(i32::to_le_bytes)
        .collect()
}

fn copy_graph(constant: bool, operations: &[CopyOp]) -> GraphInfo {
    let source = Operand {
        kind: if constant {
            OperandKind::Constant
        } else {
            OperandKind::Input
        },
        descriptor: descriptor(),
        name: Some("source".into()),
    };
    let mut graph = GraphInfo {
        operands: vec![source],
        input_operands: if constant { vec![] } else { vec![0] },
        ..Default::default()
    };
    if constant {
        graph.constant_operand_ids_to_handles.insert(
            0,
            ConstantData {
                data: exact_bytes(),
                label: None,
            },
        );
    }
    for (index, &kind) in operations.iter().enumerate() {
        let output = (index + 1) as u32;
        graph.operands.push(Operand {
            kind: OperandKind::Output,
            descriptor: descriptor(),
            name: Some(format!("copy{output}")),
        });
        graph
            .operations
            .push(copy_operation(kind, output - 1, output));
        graph.output_operands.push(output);
    }
    graph
}

fn metadata(graph: &GraphInfo) -> HashMap<String, String> {
    let converted = CoremlMlProgramConverter.convert(graph).unwrap();
    let model = specification::Model::decode(converted.data.as_slice()).unwrap();
    let specification::model::Type::MlProgram(program) = model.r#type.unwrap() else {
        panic!("expected MLProgram")
    };
    assert!(
        !program.functions["main"].block_specializations["CoreML7"]
            .operations
            .is_empty()
    );
    model.description.unwrap().metadata.unwrap().user_defined
}

#[test]
fn input_no_op_chains_keep_the_original_binding_proof() {
    let graph = copy_graph(
        false,
        &[
            CopyOp::Transpose,
            CopyOp::Slice,
            CopyOp::Reshape,
            CopyOp::Identity,
            CopyOp::Cast,
        ],
    );
    let metadata = metadata(&graph);
    let proofs: Value = serde_json::from_str(&metadata[INPUT_COPIES]).unwrap();
    assert_eq!(proofs.as_object().unwrap().len(), 5);
    for id in 1..=5 {
        assert_eq!(proofs[format!("copy{id}")]["input"], "source");
    }
    assert!(!metadata.contains_key(CONSTANT_COPIES));
}

#[test]
fn constant_no_op_fanout_serializes_one_original_exact_payload() {
    let graph = copy_graph(
        true,
        &[
            CopyOp::Identity,
            CopyOp::Transpose,
            CopyOp::Slice,
            CopyOp::Reshape,
            CopyOp::Cast,
        ],
    );
    let before = serde_json::to_vec(&graph).unwrap();
    let metadata = metadata(&graph);
    let copies: Value = serde_json::from_str(&metadata[CONSTANT_COPIES]).unwrap();
    assert_eq!(copies["version"], 1);
    assert_eq!(copies["sources"].as_object().unwrap().len(), 1);
    assert_eq!(copies["outputs"].as_object().unwrap().len(), 5);
    assert_eq!(copies["sources"]["0"]["descriptor"]["data_type"], "int32");
    use base64::Engine;
    assert_eq!(
        base64::engine::general_purpose::STANDARD
            .decode(copies["sources"]["0"]["data"].as_str().unwrap())
            .unwrap(),
        exact_bytes()
    );
    assert!(!metadata.contains_key(INPUT_COPIES));
    assert_eq!(serde_json::to_vec(&graph).unwrap(), before);
}

#[test]
fn equal_shape_does_not_prove_a_transpose_or_strided_slice_is_a_copy() {
    let mut graph = copy_graph(false, &[CopyOp::Transpose]);
    for operand in &mut graph.operands {
        operand.descriptor.shape = vec![Dimension::Static(2), Dimension::Static(2)];
    }
    graph.operations[0] = Operation::Transpose {
        input: 0,
        options: Some(MLTransposeOptions {
            permutation: vec![1, 0],
            ..Default::default()
        }),
        outputs: vec![1],
    };
    assert!(!metadata(&graph).contains_key(INPUT_COPIES));

    let mut graph = copy_graph(false, &[CopyOp::Slice]);
    graph.operands[0].descriptor.shape = vec![Dimension::Static(12)];
    graph.operations[0] = Operation::Slice {
        input: 0,
        starts: vec![0],
        sizes: vec![MLDimension::Static(12)],
        options: Some(MLSliceOptions {
            strides: vec![2],
            ..Default::default()
        }),
        outputs: vec![1],
    };
    assert!(!metadata(&graph).contains_key(INPUT_COPIES));
}

#[test]
fn arithmetic_intermediates_are_not_original_source_copies() {
    let mut graph = copy_graph(false, &[CopyOp::Identity, CopyOp::Reshape]);
    graph.operations[0] = Operation::Neg {
        input: 0,
        options: None,
        outputs: vec![1],
    };
    assert!(!metadata(&graph).contains_key(INPUT_COPIES));
    assert!(!metadata(&graph).contains_key(CONSTANT_COPIES));
}

#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
mod runtime {
    use super::*;
    use rustnn::mlcontext::{
        Backend, MLContext, MLContextOptions, MLGraphBuilder, MLNamedOperands, MLNamedTensors,
        MLOperandDescriptor, MLPowerPreference, MLTensorDescriptor,
    };

    #[test]
    fn public_constant_bytes_and_readable_outputs_preserve_float_payloads() {
        for (dtype, expected) in [
            (
                MLOperandDataType::Float32,
                [1u32, 0x80000000, 0x3f800001, 0x7fc12345]
                    .into_iter()
                    .flat_map(u32::to_le_bytes)
                    .collect::<Vec<_>>(),
            ),
            (
                MLOperandDataType::Float16,
                [1u16, 0x8000, 0x3c01, 0x7e15]
                    .into_iter()
                    .flat_map(u16::to_le_bytes)
                    .collect(),
            ),
        ] {
            let mut context = MLContext::create(
                &MLContextOptions::new(MLPowerPreference::Default, false)
                    .with_rustnn_backend_hint(Backend::Coreml),
            )
            .unwrap();
            let mut builder = MLGraphBuilder::new(&mut context).unwrap();
            let constant = builder
                .constant_from_bytes(&MLOperandDescriptor::new(dtype, vec![4]), expected.clone())
                .unwrap();
            let identity = builder.identity(constant).unwrap();
            let sliced = builder
                .slice(constant, &[0], &[MLDimension::Static(4)])
                .unwrap();
            let reshaped = builder
                .reshape(constant, vec![MLDimension::Static(4)])
                .unwrap();
            let mut graph = builder
                .build(&MLNamedOperands::from([
                    ("identity", identity),
                    ("sliced", sliced),
                    ("reshaped", reshaped),
                ]))
                .unwrap();
            let descriptor = MLTensorDescriptor::new(dtype, vec![4])
                .to_readable()
                .to_writable();
            let identity = context.create_tensor(&descriptor).unwrap();
            let sliced = context.create_tensor(&descriptor).unwrap();
            let reshaped = context.create_tensor(&descriptor).unwrap();
            context
                .dispatch(
                    &mut graph,
                    &MLNamedTensors::new(),
                    &MLNamedTensors::from([
                        ("identity", &identity),
                        ("sliced", &sliced),
                        ("reshaped", &reshaped),
                    ]),
                )
                .unwrap();
            context
                .write_tensor(&identity, &vec![0u8; expected.len()])
                .unwrap();
            for output in [&sliced, &reshaped] {
                let mut actual = vec![0u8; expected.len()];
                context.read_tensor(output, &mut actual).unwrap();
                assert_eq!(actual, expected);
            }
        }
    }

    #[test]
    fn typed_rank_one_copies_preserve_large_int32_values_and_independent_outputs() {
        for constant in [false, true] {
            let mut context = MLContext::create(
                &MLContextOptions::new(MLPowerPreference::Default, false)
                    .with_rustnn_backend_hint(Backend::Coreml),
            )
            .unwrap();
            let mut graph = context
                .rustnn_build_graph(copy_graph(
                    constant,
                    &[
                        CopyOp::Identity,
                        CopyOp::Transpose,
                        CopyOp::Slice,
                        CopyOp::Reshape,
                        CopyOp::Cast,
                    ],
                ))
                .unwrap();
            let descriptor = MLTensorDescriptor::new(MLOperandDataType::Int32, vec![6])
                .to_readable()
                .to_writable();
            let input = context.create_tensor(&descriptor).unwrap();
            let expected = exact_bytes();
            context.write_tensor(&input, &expected).unwrap();
            let outputs: Vec<_> = (0..5)
                .map(|_| context.create_tensor(&descriptor).unwrap())
                .collect();
            let names: Vec<_> = (1..=5).map(|index| format!("copy{index}")).collect();
            context
                .dispatch(
                    &mut graph,
                    &if constant {
                        MLNamedTensors::new()
                    } else {
                        MLNamedTensors::from([("source", &input)])
                    },
                    &names
                        .iter()
                        .zip(&outputs)
                        .map(|(name, tensor)| (name.as_str(), tensor))
                        .collect(),
                )
                .unwrap();
            context.write_tensor(&input, &[0u8; 24]).unwrap();
            for output in &outputs {
                let mut actual = vec![0u8; expected.len()];
                context.read_tensor(output, &mut actual).unwrap();
                assert_eq!(actual, expected);
            }
            context.write_tensor(&outputs[0], &[0u8; 24]).unwrap();
            let mut unaffected = vec![0u8; expected.len()];
            context.read_tensor(&outputs[1], &mut unaffected).unwrap();
            assert_eq!(unaffected, expected);
        }
    }
}
