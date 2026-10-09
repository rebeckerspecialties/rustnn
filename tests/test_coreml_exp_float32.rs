//! Float32 Exp accuracy through public tensors, including representable tails.

#![cfg(all(target_os = "macos", feature = "coreml-runtime"))]

use rustnn::backend_selection::{BackendDevice, DeviceType};
use rustnn::executors::coreml::CoremlLoadRoute;
use rustnn::graph::{
    ConstantData, DataType, Dimension, GraphInfo, Operand, OperandDescriptor, OperandKind,
};
use rustnn::mlcontext::{
    LoadDiagnostics, MLContext, MLContextOptions, MLNamedTensors, MLPowerPreference,
    MLTensorDescriptor, RustNNOptions,
};
use rustnn::operator_enums::MLOperandDataType;
use rustnn::operators::Operation;

const INPUT: [u32; 9] = [
    0xc2d0_0000,
    0xc2ce_0000,
    0xc2c8_0000,
    0xc2be_0000,
    0xc2b4_0000,
    0xc2b0_0000,
    0xbf80_0000,
    0,
    0x3f80_0000,
];
const EXPECTED: [u32; 9] = [
    0,
    1,
    27,
    3940,
    584744,
    4320708,
    0x3ebc_5ab2,
    0x3f80_0000,
    0x402d_f854,
];

fn context(mode: usize, policy: DeviceType) -> MLContext<'static> {
    let mut options = RustNNOptions::default();
    options.coreml.reuse_tensor_storage = mode != 0;
    options.coreml.output_backings = mode == 2;
    MLContext::create(
        &MLContextOptions::new(MLPowerPreference::Default, policy != DeviceType::Cpu)
            .with_rustnn_device_hint(BackendDevice::Coreml {
                device_type: policy,
            })
            .with_rustnn_options(options),
    )
    .unwrap()
}

fn exp_graph(shape: Vec<Dimension>) -> GraphInfo {
    let descriptor = OperandDescriptor {
        data_type: DataType::Float32,
        shape,
        pending_permutation: vec![],
    };
    GraphInfo {
        operands: vec![
            Operand {
                name: Some("input".into()),
                kind: OperandKind::Input,
                descriptor: descriptor.clone(),
            },
            Operand {
                name: Some("result".into()),
                kind: OperandKind::Output,
                descriptor,
            },
        ],
        operations: vec![Operation::Exp {
            input: 0,
            options: None,
            outputs: vec![1],
        }],
        input_operands: vec![0],
        output_operands: vec![1],
        ..Default::default()
    }
}

#[test]
fn exp_float32_representable_tails_respect_existing_ulp_budget() {
    // The frozen nine-input packet also used by the native device controls.
    // exp(-95), exp(-90), exp(-88), independently rounded from Decimal140/180.
    // These positive results are respectively 3940, 584744 and 4320708 ULP
    // from zero, exceeding the existing Exp Float32 allowance of 32 ULP.
    // This is an operator-accuracy regression, not a stricter model profile.
    for policy in [DeviceType::Cpu, DeviceType::Gpu, DeviceType::Npu] {
        let mut context = context(0, policy);
        let mut graph = context
            .rustnn_build_graph(exp_graph(vec![Dimension::Static(9)]))
            .unwrap();
        let descriptor = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![9]);
        let x = context
            .create_tensor(&descriptor.clone().to_writable())
            .unwrap();
        let result = context.create_tensor(&descriptor.to_readable()).unwrap();
        context.write_tensor(&x, &INPUT).unwrap();
        context
            .dispatch(
                &mut graph,
                &MLNamedTensors::from([("input", &x)]),
                &MLNamedTensors::from([("result", &result)]),
            )
            .unwrap();
        let mut actual = [0_u32; 9];
        context.read_tensor(&result, &mut actual).unwrap();
        for (index, (&actual, &expected)) in actual.iter().zip(&EXPECTED).enumerate() {
            assert!(
                actual.abs_diff(expected) <= 32,
                "policy={policy:?}, index={index}, actual={actual:08x}, expected={expected:08x}"
            );
        }
    }
}

fn constant(source: &mut GraphInfo, id: u32, bits: &[u32]) {
    source.operands[id as usize].kind = OperandKind::Constant;
    source.input_operands.retain(|&input| input != id);
    source.constant_operand_ids_to_handles.insert(
        id,
        ConstantData {
            data: bits.iter().flat_map(|value| value.to_le_bytes()).collect(),
            label: None,
        },
    );
}

#[test]
fn exp_float32_runtime_constant_scalar_and_owned_outputs() {
    for mode in 0..3 {
        for policy in [DeviceType::Cpu, DeviceType::Gpu, DeviceType::Npu] {
            for is_constant in [false, true] {
                for scalar in [false, true] {
                    let shape = if scalar { vec![] } else { vec![9] };
                    let input = if scalar { &INPUT[4..5] } else { &INPUT[..] };
                    let expected = if scalar {
                        &EXPECTED[4..5]
                    } else {
                        &EXPECTED[..]
                    };
                    let mut source =
                        exp_graph(shape.iter().map(|&size| Dimension::Static(size)).collect());
                    if is_constant {
                        constant(&mut source, 0, input);
                    }
                    let mut context = context(mode, policy);
                    let mut graph = context.rustnn_build_graph(source).unwrap();
                    let Some(LoadDiagnostics::Coreml(load)) = graph.rustnn_load_diagnostics()
                    else {
                        panic!("missing load diagnostics")
                    };
                    assert_eq!(load.route, CoremlLoadRoute::TypedHost);
                    assert_eq!(load.loaded_compute_units, "NOT_APPLICABLE");
                    assert!(load.failures.is_empty());
                    let descriptor = MLTensorDescriptor::new(
                        MLOperandDataType::Float32,
                        shape.into_iter().map(u64::from).collect(),
                    );
                    let x = context
                        .create_tensor(&descriptor.clone().to_writable())
                        .unwrap();
                    let y = context.create_tensor(&descriptor.to_readable()).unwrap();
                    for _ in 0..2 {
                        context.write_tensor(&x, input).unwrap();
                        context
                            .dispatch(
                                &mut graph,
                                &if is_constant {
                                    MLNamedTensors::new()
                                } else {
                                    MLNamedTensors::from([("input", &x)])
                                },
                                &MLNamedTensors::from([("result", &y)]),
                            )
                            .unwrap();
                        context.write_tensor(&x, &vec![0_u32; input.len()]).unwrap();
                        let mut actual = vec![0_u32; input.len()];
                        context.read_tensor(&y, &mut actual).unwrap();
                        assert_eq!(actual, expected, "{mode} {policy:?} {is_constant} {scalar}");
                    }
                    drop(graph);
                    let mut actual = vec![0_u32; input.len()];
                    context.read_tensor(&y, &mut actual).unwrap();
                    assert_eq!(actual, expected);
                }
            }
        }
    }
}

#[test]
fn exp_float32_native_constant_arithmetic_materializes_before_exp() {
    for mode in 0..3 {
        for policy in [DeviceType::Cpu, DeviceType::Gpu, DeviceType::Npu] {
            for scalar in [false, true] {
                let shape = if scalar {
                    vec![]
                } else {
                    vec![Dimension::Static(2)]
                };
                let count = if scalar { 1 } else { 2 };
                let mut source = exp_graph(shape);
                source.operands[1].name = Some("offset".into());
                for name in ["sum", "result"] {
                    let mut operand = source.operands[1].clone();
                    operand.name = Some(name.into());
                    operand.kind = if name == "sum" {
                        OperandKind::Intermediate
                    } else {
                        OperandKind::Output
                    };
                    source.operands.push(operand);
                }
                constant(
                    &mut source,
                    0,
                    &[(-91_f32).to_bits(), (-96_f32).to_bits()][..count],
                );
                constant(&mut source, 1, &[1_f32.to_bits(); 2][..count]);
                source.operations = vec![
                    Operation::Add {
                        a: 0,
                        b: 1,
                        options: None,
                        outputs: vec![2],
                    },
                    Operation::Exp {
                        input: 2,
                        options: None,
                        outputs: vec![3],
                    },
                ];
                source.output_operands = vec![3];
                let mut context = context(mode, policy);
                let mut graph = context.rustnn_build_graph(source).unwrap();
                let y = context
                    .create_tensor(
                        &MLTensorDescriptor::new(
                            MLOperandDataType::Float32,
                            if scalar { vec![] } else { vec![2] },
                        )
                        .to_readable(),
                    )
                    .unwrap();
                context
                    .dispatch(
                        &mut graph,
                        &MLNamedTensors::new(),
                        &MLNamedTensors::from([("result", &y)]),
                    )
                    .unwrap();
                let mut actual = vec![0_u32; count];
                context.read_tensor(&y, &mut actual).unwrap();
                assert_eq!(
                    actual,
                    [584744, 3940][..count],
                    "{mode} {policy:?} {scalar}"
                );
            }
        }
    }
}

#[test]
fn exp_float32_constant_transpose_reads_original_bits() {
    for mode in 0..3 {
        let mut source = exp_graph(vec![Dimension::Static(3), Dimension::Static(3)]);
        source.operands[1].name = Some("transposed".into());
        source.operands[1].kind = OperandKind::Intermediate;
        let mut result = source.operands[1].clone();
        result.name = Some("result".into());
        result.kind = OperandKind::Output;
        source.operands.push(result);
        constant(&mut source, 0, &INPUT);
        source.operations = vec![
            Operation::from_json_attributes(
                "transpose",
                &[0],
                &[1],
                &serde_json::json!({"permutation": [1, 0]}),
            )
            .unwrap(),
            Operation::Exp {
                input: 1,
                options: None,
                outputs: vec![2],
            },
        ];
        source.output_operands = vec![2];
        let mut context = context(mode, DeviceType::Cpu);
        let mut graph = context.rustnn_build_graph(source).unwrap();
        let y = context
            .create_tensor(
                &MLTensorDescriptor::new(MLOperandDataType::Float32, vec![3, 3]).to_readable(),
            )
            .unwrap();
        context
            .dispatch(
                &mut graph,
                &MLNamedTensors::new(),
                &MLNamedTensors::from([("result", &y)]),
            )
            .unwrap();
        let mut actual = [0_u32; 9];
        context.read_tensor(&y, &mut actual).unwrap();
        assert_eq!(actual, [0, 3, 6, 1, 4, 7, 2, 5, 8].map(|i| EXPECTED[i]));
    }
}

#[test]
fn exp_float32_native_neighbors_and_public_fanout_keep_normal_values() {
    // Subnormal native consumers are qualified separately: repairing Exp does
    // not redefine another operator's accuracy contract or arithmetic kernel.
    for mode in 0..3 {
        for policy in [DeviceType::Cpu, DeviceType::Gpu, DeviceType::Npu] {
            let mut source = exp_graph(vec![Dimension::Static(3)]);
            source.operands[1].name = Some("negated".into());
            source.operands[1].kind = OperandKind::Intermediate;
            for name in ["exponential", "result"] {
                let mut operand = source.operands[1].clone();
                operand.name = Some(name.into());
                operand.kind = OperandKind::Output;
                source.operands.push(operand);
            }
            source.operations = vec![
                Operation::Neg {
                    input: 0,
                    options: None,
                    outputs: vec![1],
                },
                Operation::Exp {
                    input: 1,
                    options: None,
                    outputs: vec![2],
                },
                Operation::Neg {
                    input: 2,
                    options: None,
                    outputs: vec![3],
                },
            ];
            source.output_operands = vec![2, 3];
            let mut context = context(mode, policy);
            let mut graph = context.rustnn_build_graph(source).unwrap();
            let descriptor = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![3]);
            let x = context
                .create_tensor(&descriptor.clone().to_writable())
                .unwrap();
            let exponential = context
                .create_tensor(&descriptor.clone().to_readable())
                .unwrap();
            let result = context.create_tensor(&descriptor.to_readable()).unwrap();
            context.write_tensor(&x, &[1_f32, 0., -1.]).unwrap();
            context
                .dispatch(
                    &mut graph,
                    &MLNamedTensors::from([("input", &x)]),
                    &MLNamedTensors::from([("exponential", &exponential), ("result", &result)]),
                )
                .unwrap();
            drop(graph);
            let expected = [0x3ebc_5ab2, 0x3f80_0000, 0x402d_f854];
            let mut actual = [0_u32; 3];
            context.read_tensor(&exponential, &mut actual).unwrap();
            assert_eq!(actual, expected);
            context.read_tensor(&result, &mut actual).unwrap();
            assert_eq!(actual, expected.map(|bits| bits ^ 0x8000_0000));
        }
    }
}

#[cfg(feature = "dynamic-inputs")]
#[test]
fn exp_float32_bounded_actual_shapes_grow_then_shrink() {
    for mode in 0..3 {
        let mut context = context(mode, DeviceType::Cpu);
        let mut graph = context
            .rustnn_build_graph(exp_graph(vec![Dimension::Dynamic(
                rustnn::graph::DynamicDimension {
                    name: "length".into(),
                    max_size: 9,
                },
            )]))
            .unwrap();
        for count in [1, 9, 3] {
            let input = if count == 1 {
                &INPUT[4..5]
            } else {
                &INPUT[..count]
            };
            let expected = if count == 1 {
                &EXPECTED[4..5]
            } else {
                &EXPECTED[..count]
            };
            let descriptor =
                MLTensorDescriptor::new(MLOperandDataType::Float32, vec![count as u64]);
            let x = context
                .create_tensor(&descriptor.clone().to_writable())
                .unwrap();
            let y = context.create_tensor(&descriptor.to_readable()).unwrap();
            context.write_tensor(&x, input).unwrap();
            context
                .dispatch(
                    &mut graph,
                    &MLNamedTensors::from([("input", &x)]),
                    &MLNamedTensors::from([("result", &y)]),
                )
                .unwrap();
            let mut actual = vec![0_u32; count];
            context.read_tensor(&y, &mut actual).unwrap();
            assert_eq!(actual, expected);
            assert_eq!(y.shape(), [count as u64]);
        }
    }
}
