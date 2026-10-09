//! Logical scalar tensors and scalar-valued parameters must remain distinct.
#![cfg(all(target_os = "macos", feature = "coreml-runtime"))]

use rustnn::graph::{
    ConstantData, DataType, GraphInfo, Operand, OperandDescriptor, OperandKind, to_dimension_vector,
};
use rustnn::mlcontext::{
    Backend, MLContext, MLContextOptions, MLNamedTensors, MLPowerPreference, MLTensorDescriptor,
    RustNNOptions,
};
use rustnn::operator_enums::MLOperandDataType;
use rustnn::operators::Operation;

fn operand(name: &str, kind: OperandKind, dtype: DataType, shape: &[u32]) -> Operand {
    Operand {
        name: Some(name.into()),
        kind,
        descriptor: OperandDescriptor {
            data_type: dtype,
            shape: to_dimension_vector(shape),
            pending_permutation: vec![],
        },
    }
}
fn constant(graph: &mut GraphInfo, id: u32, bytes: Vec<u8>) {
    graph.operands[id as usize].kind = OperandKind::Constant;
    graph.input_operands.retain(|&input| input != id);
    graph.constant_operand_ids_to_handles.insert(
        id,
        ConstantData {
            data: bytes,
            label: None,
        },
    );
}
fn value_bytes(dtype: DataType, f32_bits: u32, f16_bits: u16) -> Vec<u8> {
    if dtype == DataType::Float32 {
        f32_bits.to_le_bytes().to_vec()
    } else {
        f16_bits.to_le_bytes().to_vec()
    }
}
fn context(mode: usize) -> MLContext<'static> {
    let mut tuning = RustNNOptions::default();
    tuning.coreml.reuse_tensor_storage = mode != 0;
    tuning.coreml.output_backings = mode == 2;
    MLContext::create(
        &MLContextOptions::new(MLPowerPreference::Default, false)
            .with_rustnn_backend_hint(Backend::Coreml)
            .with_rustnn_options(tuning),
    )
    .unwrap()
}

#[test]
fn scalar_unary_constants_and_runtime_controls_keep_shape_and_values() {
    // Independently rounded at x=1/4; allowances are the unchanged per-op
    // gates in WPT 784112f0. Native exported-model compilation is separately
    // tested in scalar_inputs so a typed executor cannot hide invalid MIL.
    let references: [(&str, u32, u16, u32, u32); 8] = [
        ("sqrt", 1056964608, 14336, 1, 1),
        ("exp", 1067735794, 15651, 32, 1),
        ("log", 3216077336, 48524, 8, 8),
        ("abs", 1048576000, 13312, 0, 0),
        ("neg", 3196059648, 46080, 0, 0),
        ("tanh", 1048234997, 13270, 16, 16),
        ("gelu", 1041843409, 12490, 18, 18),
        ("relu", 1048576000, 13312, 0, 0),
    ];
    for dtype in [DataType::Float32, DataType::Float16] {
        for is_constant in [false, true] {
            for (name, f32_bits, f16_bits, f32_allowance, f16_allowance) in references {
                let mut info = GraphInfo {
                    operands: vec![
                        operand("input", OperandKind::Input, dtype, &[]),
                        operand("result", OperandKind::Output, dtype, &[]),
                    ],
                    operations: vec![
                        Operation::from_json_attributes(name, &[0], &[1], &serde_json::json!({}))
                            .unwrap(),
                    ],
                    input_operands: vec![0],
                    output_operands: vec![1],
                    ..Default::default()
                };
                let bytes = value_bytes(dtype, 0.25_f32.to_bits(), 0x3400);
                if is_constant {
                    constant(&mut info, 0, bytes.clone());
                }
                let mut context = context(0);
                let mut graph = context.rustnn_build_graph(info).unwrap();
                assert!(graph.output_descriptors["result"].shape.is_empty());
                let descriptor = MLTensorDescriptor::new(dtype.try_into().unwrap(), vec![]);
                let input = context
                    .create_tensor(&descriptor.clone().to_writable())
                    .unwrap();
                let output = context.create_tensor(&descriptor.to_readable()).unwrap();
                context.write_tensor(&input, &bytes).unwrap();
                let inputs = if is_constant {
                    MLNamedTensors::new()
                } else {
                    MLNamedTensors::from([("input", &input)])
                };
                context
                    .dispatch(
                        &mut graph,
                        &inputs,
                        &MLNamedTensors::from([("result", &output)]),
                    )
                    .unwrap();
                let mut actual = vec![0_u8; bytes.len()];
                context.read_tensor(&output, &mut actual).unwrap();
                let (actual, expected, allowance) = if dtype == DataType::Float32 {
                    (
                        u32::from_le_bytes(actual.try_into().unwrap()),
                        f32_bits,
                        f32_allowance,
                    )
                } else {
                    (
                        u32::from(u16::from_le_bytes(actual.try_into().unwrap())),
                        u32::from(f16_bits),
                        f16_allowance,
                    )
                };
                assert!(
                    actual.abs_diff(expected) <= allowance,
                    "{name}/{dtype:?}/constant={is_constant}: {actual:08x} vs {expected:08x}"
                );
            }
        }
    }
}

#[test]
fn scalar_constant_can_feed_unary_tensor_and_quantization_parameter() {
    for dtype in [DataType::Float32, DataType::Float16] {
        for mode in 0..3 {
            let mut info = GraphInfo {
                operands: vec![
                    operand("scale", OperandKind::Constant, dtype, &[]),
                    operand("input", OperandKind::Input, dtype, &[2]),
                    operand("zero", OperandKind::Constant, DataType::Int8, &[]),
                    operand("root", OperandKind::Output, dtype, &[]),
                    operand("quantized", OperandKind::Output, DataType::Int8, &[2]),
                ],
                operations: vec![
                    Operation::Sqrt {
                        input: 0,
                        outputs: vec![3],
                        options: None,
                    },
                    Operation::QuantizeLinear {
                        input: 1,
                        scale: 0,
                        zero_point: Some(2),
                        outputs: vec![4],
                        options: None,
                    },
                ],
                input_operands: vec![1],
                output_operands: vec![3, 4],
                ..Default::default()
            };
            constant(&mut info, 0, value_bytes(dtype, 0.25_f32.to_bits(), 0x3400));
            constant(&mut info, 2, vec![0]);
            let mut context = context(mode);
            let mut graph = context.rustnn_build_graph(info).unwrap();
            let input = context
                .create_tensor(
                    &MLTensorDescriptor::new(dtype.try_into().unwrap(), vec![2]).to_writable(),
                )
                .unwrap();
            let root = context
                .create_tensor(
                    &MLTensorDescriptor::new(dtype.try_into().unwrap(), vec![])
                        .to_readable()
                        .to_writable(),
                )
                .unwrap();
            let quantized = context
                .create_tensor(
                    &MLTensorDescriptor::new(MLOperandDataType::Int8, vec![2]).to_readable(),
                )
                .unwrap();
            let input_bytes = [
                value_bytes(dtype, 0.5_f32.to_bits(), 0x3800),
                value_bytes(dtype, 0.75_f32.to_bits(), 0x3a00),
            ]
            .concat();
            for _ in 0..2 {
                context.write_tensor(&input, &input_bytes).unwrap();
                context
                    .dispatch(
                        &mut graph,
                        &MLNamedTensors::from([("input", &input)]),
                        &MLNamedTensors::from([("root", &root), ("quantized", &quantized)]),
                    )
                    .unwrap();
                let mut root_bytes = vec![0_u8; root.rustnn_required_bytes()];
                context.read_tensor(&root, &mut root_bytes).unwrap();
                assert_eq!(root_bytes, value_bytes(dtype, 0.5_f32.to_bits(), 0x3800));
                context
                    .write_tensor(&root, &vec![0_u8; root_bytes.len()])
                    .unwrap();
                let mut values = [0_i8; 2];
                context.read_tensor(&quantized, &mut values).unwrap();
                assert_eq!(values, [2, 3]);
            }
            assert!(graph.output_descriptors["root"].shape.is_empty());
        }
    }
}
