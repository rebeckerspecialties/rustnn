//! WebNN scalars stay rank zero even when the native feature interface is [1].
#![cfg(all(target_os = "macos", feature = "coreml-runtime"))]

use rustnn::graph::{
    ConstantData, DataType, GraphInfo, Operand, OperandDescriptor, OperandKind, to_dimension_vector,
};
use rustnn::mlcontext::{
    Backend, MLContext, MLContextOptions, MLNamedTensors, MLPowerPreference, MLTensorDescriptor,
};
use rustnn::operator_enums::MLOperandDataType;
use rustnn::operators::Operation;

fn constant_binary(name: &str, shape: &[u32]) -> GraphInfo {
    let operand = |name: &str, kind| Operand {
        kind,
        name: Some(name.into()),
        descriptor: OperandDescriptor {
            data_type: DataType::Float32,
            shape: to_dimension_vector(shape),
            pending_permutation: vec![],
        },
    };
    GraphInfo {
        operands: vec![
            operand("left", OperandKind::Constant),
            operand("right", OperandKind::Constant),
            operand("result", OperandKind::Output),
        ],
        constant_operand_ids_to_handles: [(0, 4f32), (1, 2f32)]
            .into_iter()
            .map(|(id, value)| {
                (
                    id,
                    ConstantData {
                        data: value.to_le_bytes().to_vec(),
                        label: None,
                    },
                )
            })
            .collect(),
        operations: vec![
            Operation::from_json_attributes(name, &[0, 1], &[2], &serde_json::json!({})).unwrap(),
        ],
        output_operands: vec![2],
        ..Default::default()
    }
}

fn dispatch_scalar(graph_info: GraphInfo, expected: f32) {
    let shape = graph_info.operands[2].descriptor.static_shape().unwrap();
    let mut context = MLContext::create(
        &MLContextOptions::new(MLPowerPreference::Default, false)
            .with_rustnn_backend_hint(Backend::Coreml),
    )
    .unwrap();
    let mut graph = context.rustnn_build_graph(graph_info).unwrap();
    assert_eq!(
        graph.output_descriptors["result"].shape,
        to_dimension_vector(&shape)
    );
    let output = context
        .create_tensor(
            &MLTensorDescriptor::new(
                MLOperandDataType::Float32,
                shape.into_iter().map(u64::from).collect(),
            )
            .to_readable(),
        )
        .unwrap();
    context
        .dispatch(
            &mut graph,
            &MLNamedTensors::new(),
            &MLNamedTensors::from([("result", &output)]),
        )
        .unwrap();
    let mut actual = [0f32];
    context.read_tensor(&output, &mut actual).unwrap();
    assert_eq!(actual[0].to_bits(), expected.to_bits());
}

#[test]
fn scalar_constant_add_compiles_and_dispatches() {
    dispatch_scalar(constant_binary("add", &[]), 6.);
}

#[test]
fn rank_one_constant_add_control() {
    dispatch_scalar(constant_binary("add", &[1]), 6.);
}

#[test]
fn scalar_constant_and_runtime_inputs_keep_logical_rank() {
    for runtime_id in [0, 1] {
        for shape in [&[][..], &[1][..]] {
            let mut info = constant_binary("sub", shape);
            info.operands[runtime_id].kind = OperandKind::Input;
            info.constant_operand_ids_to_handles
                .remove(&(runtime_id as u32));
            info.input_operands = vec![runtime_id as u32];
            let name = info.operands[runtime_id].name.clone().unwrap();
            let mut context = MLContext::create(
                &MLContextOptions::new(MLPowerPreference::Default, false)
                    .with_rustnn_backend_hint(Backend::Coreml),
            )
            .unwrap();
            let mut graph = context.rustnn_build_graph(info).unwrap();
            let descriptor = MLTensorDescriptor::new(
                MLOperandDataType::Float32,
                shape.iter().map(|&n| u64::from(n)).collect(),
            );
            let input = context
                .create_tensor(&descriptor.clone().to_writable())
                .unwrap();
            let output = context.create_tensor(&descriptor.to_readable()).unwrap();
            context
                .write_tensor(&input, &[if runtime_id == 0 { 4f32 } else { 2. }])
                .unwrap();
            context
                .dispatch(
                    &mut graph,
                    &MLNamedTensors::from([(name.as_str(), &input)]),
                    &MLNamedTensors::from([("result", &output)]),
                )
                .unwrap();
            let mut actual = [0f32];
            context.read_tensor(&output, &mut actual).unwrap();
            assert_eq!(actual, [2.]);
            assert_eq!(
                graph.output_descriptors["result"].shape,
                to_dimension_vector(shape)
            );
        }
    }
}

#[test]
#[cfg(feature = "dynamic-inputs")]
fn scalar_constant_broadcasts_to_growing_and_shrinking_runtime_vector() {
    use rustnn::graph::{Dimension, DynamicDimension};
    let mut info = constant_binary("add", &[]);
    info.operands[0].kind = OperandKind::Input;
    info.input_operands = vec![0];
    info.constant_operand_ids_to_handles.remove(&0);
    let dynamic = vec![Dimension::Dynamic(DynamicDimension {
        name: "length".into(),
        max_size: 8,
    })];
    info.operands[0].descriptor.shape = dynamic.clone();
    info.operands[2].descriptor.shape = dynamic;
    let mut context = MLContext::create(
        &MLContextOptions::new(MLPowerPreference::Default, false)
            .with_rustnn_backend_hint(Backend::Coreml),
    )
    .unwrap();
    let mut graph = context.rustnn_build_graph(info).unwrap();
    for count in [1, 8, 3, 1] {
        let descriptor = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![count as u64]);
        let input = context
            .create_tensor(&descriptor.clone().to_writable())
            .unwrap();
        let output = context.create_tensor(&descriptor.to_readable()).unwrap();
        let values: Vec<f32> = (0..count).map(|n| n as f32).collect();
        context.write_tensor(&input, &values).unwrap();
        context
            .dispatch(
                &mut graph,
                &MLNamedTensors::from([("left", &input)]),
                &MLNamedTensors::from([("result", &output)]),
            )
            .unwrap();
        let mut actual = vec![0f32; count];
        context.read_tensor(&output, &mut actual).unwrap();
        assert_eq!(
            actual,
            values.iter().map(|value| value + 2.).collect::<Vec<_>>()
        );
    }
}

#[test]
fn scalar_integer_constant_cast_control() {
    let mut info = constant_binary("add", &[]);
    info.operands[0].descriptor.data_type = DataType::Int32;
    info.constant_operand_ids_to_handles
        .get_mut(&0)
        .unwrap()
        .data = 4i32.to_le_bytes().to_vec();
    info.operands[1].kind = OperandKind::Intermediate;
    info.constant_operand_ids_to_handles.remove(&1);
    info.operations = vec![
        Operation::from_json_attributes("cast", &[0], &[1], &serde_json::json!({"to": "float32"}))
            .unwrap(),
        Operation::from_json_attributes("neg", &[1], &[2], &serde_json::json!({})).unwrap(),
    ];
    dispatch_scalar(info, -4.);
}

#[test]
fn scalar_constant_binary_arithmetic() {
    for (name, expected) in [
        ("sub", 2.),
        ("mul", 8.),
        ("div", 2.),
        ("pow", 16.),
        ("max", 4.),
        ("min", 2.),
    ] {
        dispatch_scalar(constant_binary(name, &[]), expected);
    }
}

#[test]
fn scalar_constant_binary_comparisons() {
    for (name, expected) in [
        ("equal", 0u8),
        ("notEqual", 1),
        ("greater", 1),
        ("greaterOrEqual", 1),
        ("lesser", 0),
        ("lesserOrEqual", 0),
    ] {
        let mut info = constant_binary(name, &[]);
        info.operands[2].descriptor.data_type = DataType::Uint8;
        let mut context = MLContext::create(
            &MLContextOptions::new(MLPowerPreference::Default, false)
                .with_rustnn_backend_hint(Backend::Coreml),
        )
        .unwrap();
        let mut graph = context.rustnn_build_graph(info).unwrap();
        assert!(graph.output_descriptors["result"].shape.is_empty());
        let output = context
            .create_tensor(&MLTensorDescriptor::new(MLOperandDataType::Uint8, vec![]).to_readable())
            .unwrap();
        context
            .dispatch(
                &mut graph,
                &MLNamedTensors::new(),
                &MLNamedTensors::from([("result", &output)]),
            )
            .unwrap();
        let mut actual = [255u8];
        context.read_tensor(&output, &mut actual).unwrap();
        assert_eq!(actual, [expected], "{name}");
    }
}

#[test]
fn scalar_half_and_int32_add() {
    for (dtype, tensor_dtype, left, right, expected) in [
        (
            DataType::Float16,
            MLOperandDataType::Float16,
            half::f16::from_f32(4.).to_le_bytes().to_vec(),
            half::f16::from_f32(2.).to_le_bytes().to_vec(),
            half::f16::from_f32(6.).to_le_bytes().to_vec(),
        ),
        (
            DataType::Int32,
            MLOperandDataType::Int32,
            4i32.to_le_bytes().to_vec(),
            2i32.to_le_bytes().to_vec(),
            6i32.to_le_bytes().to_vec(),
        ),
    ] {
        let mut info = constant_binary("add", &[]);
        for operand in &mut info.operands {
            operand.descriptor.data_type = dtype;
        }
        info.constant_operand_ids_to_handles
            .get_mut(&0)
            .unwrap()
            .data = left;
        info.constant_operand_ids_to_handles
            .get_mut(&1)
            .unwrap()
            .data = right;
        let mut context = MLContext::create(
            &MLContextOptions::new(MLPowerPreference::Default, false)
                .with_rustnn_backend_hint(Backend::Coreml),
        )
        .unwrap();
        let mut graph = context.rustnn_build_graph(info).unwrap();
        let output = context
            .create_tensor(&MLTensorDescriptor::new(tensor_dtype, vec![]).to_readable())
            .unwrap();
        context
            .dispatch(
                &mut graph,
                &MLNamedTensors::new(),
                &MLNamedTensors::from([("result", &output)]),
            )
            .unwrap();
        let mut actual = vec![0u8; expected.len()];
        context.read_tensor(&output, &mut actual).unwrap();
        assert_eq!(actual, expected, "{dtype:?}");
    }
}

#[test]
fn scalar_integer_division() {
    let mut info = constant_binary("div", &[]);
    for operand in &mut info.operands {
        operand.descriptor.data_type = DataType::Int32;
    }
    info.constant_operand_ids_to_handles
        .get_mut(&0)
        .unwrap()
        .data = 4i32.to_le_bytes().to_vec();
    info.constant_operand_ids_to_handles
        .get_mut(&1)
        .unwrap()
        .data = 2i32.to_le_bytes().to_vec();
    let mut context = MLContext::create(
        &MLContextOptions::new(MLPowerPreference::Default, false)
            .with_rustnn_backend_hint(Backend::Coreml),
    )
    .unwrap();
    let mut graph = context.rustnn_build_graph(info).unwrap();
    let output = context
        .create_tensor(&MLTensorDescriptor::new(MLOperandDataType::Int32, vec![]).to_readable())
        .unwrap();
    context
        .dispatch(
            &mut graph,
            &MLNamedTensors::new(),
            &MLNamedTensors::from([("result", &output)]),
        )
        .unwrap();
    let mut actual = [0i32];
    context.read_tensor(&output, &mut actual).unwrap();
    assert_eq!(actual, [2]);
}

#[test]
fn scalar_logical_binary_operations() {
    for (name, expected) in [("logicalAnd", 0u8), ("logicalOr", 1), ("logicalXor", 1)] {
        let mut info = constant_binary(name, &[]);
        for operand in &mut info.operands {
            operand.descriptor.data_type = DataType::Uint8;
        }
        info.constant_operand_ids_to_handles
            .get_mut(&0)
            .unwrap()
            .data = vec![1];
        info.constant_operand_ids_to_handles
            .get_mut(&1)
            .unwrap()
            .data = vec![0];
        let mut context = MLContext::create(
            &MLContextOptions::new(MLPowerPreference::Default, false)
                .with_rustnn_backend_hint(Backend::Coreml),
        )
        .unwrap();
        let mut graph = context.rustnn_build_graph(info).unwrap();
        let output = context
            .create_tensor(&MLTensorDescriptor::new(MLOperandDataType::Uint8, vec![]).to_readable())
            .unwrap();
        context
            .dispatch(
                &mut graph,
                &MLNamedTensors::new(),
                &MLNamedTensors::from([("result", &output)]),
            )
            .unwrap();
        let mut actual = [255u8];
        context.read_tensor(&output, &mut actual).unwrap();
        assert_eq!(actual, [expected], "{name}");
    }
}
