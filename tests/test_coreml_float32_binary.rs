//! Float32 Mul/Div preserve tiny inputs whose scaled results are normal.

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

fn graph(divide: bool, shape: Vec<Dimension>) -> GraphInfo {
    let descriptor = OperandDescriptor {
        data_type: DataType::Float32,
        shape,
        pending_permutation: vec![],
    };
    GraphInfo {
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
        operations: vec![if divide {
            Operation::Div {
                a: 0,
                b: 1,
                options: None,
                outputs: vec![2],
            }
        } else {
            Operation::Mul {
                a: 0,
                b: 1,
                options: None,
                outputs: vec![2],
            }
        }],
        input_operands: vec![0, 1],
        output_operands: vec![2],
        ..Default::default()
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

fn shape(sizes: &[u32]) -> Vec<Dimension> {
    sizes.iter().copied().map(Dimension::Static).collect()
}

#[test]
fn float32_binary_scalar_broadcast_constants_and_independent_outputs() {
    for divide in [false, true] {
        for mode in 0..3 {
            for constants in 0..4 {
                for scalar in [false, true] {
                    let dims = if scalar { vec![] } else { vec![2, 3] };
                    let count = if scalar { 1 } else { 6 };
                    let mut source = graph(divide, shape(&dims));
                    source.operands[1].descriptor.shape = vec![];
                    let input = vec![1_u32; count];
                    let scale = [if divide { 0x0080_0000_u32 } else { 0x7e80_0000 }];
                    if constants & 1 != 0 {
                        constant(&mut source, 0, &input);
                    }
                    if constants & 2 != 0 {
                        constant(&mut source, 1, &scale);
                    }
                    let mut context = context(mode, DeviceType::Cpu);
                    let mut graph = context.rustnn_build_graph(source).unwrap();
                    let descriptor = MLTensorDescriptor::new(
                        MLOperandDataType::Float32,
                        dims.into_iter().map(u64::from).collect(),
                    );
                    let a = context
                        .create_tensor(&descriptor.clone().to_writable())
                        .unwrap();
                    let b = context
                        .create_tensor(
                            &MLTensorDescriptor::new(MLOperandDataType::Float32, vec![])
                                .to_writable(),
                        )
                        .unwrap();
                    let first = context
                        .create_tensor(&descriptor.clone().to_readable())
                        .unwrap();
                    let second = context.create_tensor(&descriptor.to_readable()).unwrap();
                    context.write_tensor(&a, &input).unwrap();
                    context.write_tensor(&b, &scale).unwrap();
                    let mut bindings = MLNamedTensors::new();
                    if constants & 1 == 0 {
                        bindings.insert("a", &a);
                    }
                    if constants & 2 == 0 {
                        bindings.insert("b", &b);
                    }
                    context
                        .dispatch(
                            &mut graph,
                            &bindings,
                            &MLNamedTensors::from([("result", &first)]),
                        )
                        .unwrap();
                    if constants & 1 == 0 {
                        context.write_tensor(&a, &vec![2_u32; count]).unwrap();
                    }
                    context
                        .dispatch(
                            &mut graph,
                            &bindings,
                            &MLNamedTensors::from([("result", &second)]),
                        )
                        .unwrap();
                    drop(graph);
                    let mut actual = vec![0_u32; count];
                    context.read_tensor(&first, &mut actual).unwrap();
                    assert_eq!(
                        actual,
                        vec![0x3400_0000; count],
                        "first {divide} {mode} {constants} {scalar}"
                    );
                    context.read_tensor(&second, &mut actual).unwrap();
                    assert_eq!(
                        actual,
                        vec![
                            if constants & 1 == 0 {
                                0x3480_0000
                            } else {
                                0x3400_0000
                            };
                            count
                        ]
                    );
                }
            }
        }
    }
}

#[test]
fn float32_binary_constant_transpose_and_runtime_broadcast_are_exact() {
    for divide in [false, true] {
        for mode in 0..3 {
            let mut source = graph(divide, shape(&[2, 3]));
            source.operands[0].descriptor.shape = shape(&[3, 2]);
            source.operands[1].descriptor.shape = shape(&[3]);
            let mut transposed = source.operands[2].clone();
            transposed.kind = OperandKind::Intermediate;
            transposed.name = Some("transposed".into());
            source.operands.push(transposed);
            source.operations.insert(
                0,
                Operation::from_json_attributes(
                    "transpose",
                    &[0],
                    &[3],
                    &serde_json::json!({"permutation":[1,0]}),
                )
                .unwrap(),
            );
            source.operations[1] = if divide {
                Operation::Div {
                    a: 3,
                    b: 1,
                    options: None,
                    outputs: vec![2],
                }
            } else {
                Operation::Mul {
                    a: 3,
                    b: 1,
                    options: None,
                    outputs: vec![2],
                }
            };
            constant(&mut source, 0, &[1, 2, 4, 8, 16, 32]);
            let mut context = context(mode, DeviceType::Cpu);
            let mut graph = context.rustnn_build_graph(source).unwrap();
            let b = context
                .create_tensor(
                    &MLTensorDescriptor::new(MLOperandDataType::Float32, vec![3]).to_writable(),
                )
                .unwrap();
            let result = context
                .create_tensor(
                    &MLTensorDescriptor::new(MLOperandDataType::Float32, vec![2, 3]).to_readable(),
                )
                .unwrap();
            context
                .write_tensor(&b, &[if divide { 0x0080_0000_u32 } else { 0x7e80_0000 }; 3])
                .unwrap();
            context
                .dispatch(
                    &mut graph,
                    &MLNamedTensors::from([("b", &b)]),
                    &MLNamedTensors::from([("result", &result)]),
                )
                .unwrap();
            let mut actual = [0_u32; 6];
            context.read_tensor(&result, &mut actual).unwrap();
            assert_eq!(
                actual,
                [
                    0x34000000, 0x35000000, 0x36000000, 0x34800000, 0x35800000, 0x36800000
                ]
            );
        }
    }
}

#[test]
fn float32_binary_shares_constant_cast_identity_and_reshape_sources() {
    for half in [false, true] {
        for divide in [false, true] {
            for mode in 0..3 {
                let mut source = graph(divide, shape(&[1, 2]));
                source.operands[0].descriptor.shape = shape(&[2]);
                source.operands[1].descriptor.shape = vec![];
                constant(&mut source, 0, &[1, 2]);
                if half {
                    source.operands[0].descriptor.data_type = DataType::Float16;
                    source
                        .constant_operand_ids_to_handles
                        .get_mut(&0)
                        .unwrap()
                        .data = [1_u16, 2].into_iter().flat_map(u16::to_le_bytes).collect();
                }
                constant(
                    &mut source,
                    1,
                    &[if divide { 0x40000000 } else { 0x3f000000 }],
                );
                for (name, dims) in [
                    ("converted", vec![2]),
                    ("copied", vec![2]),
                    ("reshaped", vec![1, 2]),
                ] {
                    let mut value = source.operands[2].clone();
                    value.name = Some(name.into());
                    value.kind = OperandKind::Intermediate;
                    value.descriptor.shape = shape(&dims);
                    source.operands.push(value);
                }
                source.operations = vec![
                    Operation::Cast {
                        input: 0,
                        data_type: MLOperandDataType::Float32,
                        options: None,
                        outputs: vec![3],
                    },
                    Operation::Identity {
                        input: 3,
                        options: None,
                        outputs: vec![4],
                    },
                    Operation::from_json_attributes(
                        "reshape",
                        &[4],
                        &[5],
                        &serde_json::json!({"newShape":[1,2]}),
                    )
                    .unwrap(),
                    if divide {
                        Operation::Div {
                            a: 5,
                            b: 1,
                            options: None,
                            outputs: vec![2],
                        }
                    } else {
                        Operation::Mul {
                            a: 5,
                            b: 1,
                            options: None,
                            outputs: vec![2],
                        }
                    },
                ];
                let mut context = context(mode, DeviceType::Cpu);
                let mut graph = context.rustnn_build_graph(source).unwrap();
                let Some(LoadDiagnostics::Coreml(load)) = graph.rustnn_load_diagnostics() else {
                    panic!("diagnostics")
                };
                assert_eq!(load.route, CoremlLoadRoute::TypedHost);
                let output = context
                    .create_tensor(
                        &MLTensorDescriptor::new(MLOperandDataType::Float32, vec![1, 2])
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
                let mut actual = [0_u32; 2];
                context.read_tensor(&output, &mut actual).unwrap();
                assert_eq!(
                    actual,
                    if half {
                        [0x33000000, 0x33800000]
                    } else {
                        [0, 1]
                    }
                );
            }
        }
    }
}

#[cfg(feature = "dynamic-inputs")]
#[test]
fn float32_binary_dynamic_broadcast_grows_shrinks_and_rejects_before_publication() {
    for divide in [false, true] {
        for mode in 0..3 {
            let dynamic = Dimension::Dynamic(rustnn::graph::DynamicDimension {
                name: "length".into(),
                max_size: 6,
            });
            let mut source = graph(divide, vec![dynamic.clone(), Dimension::Static(3)]);
            source.operands[0].descriptor.shape = vec![dynamic.clone(), Dimension::Static(1)];
            source.operands[1].descriptor.shape = shape(&[3]);
            let mut context = context(mode, DeviceType::Cpu);
            let mut graph = context.rustnn_build_graph(source).unwrap();
            for count in [1, 6, 2] {
                let a = context
                    .create_tensor(
                        &MLTensorDescriptor::new(MLOperandDataType::Float32, vec![count, 1])
                            .to_writable(),
                    )
                    .unwrap();
                let b = context
                    .create_tensor(
                        &MLTensorDescriptor::new(MLOperandDataType::Float32, vec![3]).to_writable(),
                    )
                    .unwrap();
                let result = context
                    .create_tensor(
                        &MLTensorDescriptor::new(MLOperandDataType::Float32, vec![count, 3])
                            .to_readable(),
                    )
                    .unwrap();
                context
                    .write_tensor(&a, &vec![1_u32; count as usize])
                    .unwrap();
                context
                    .write_tensor(&b, &[if divide { 0x0080_0000_u32 } else { 0x7e80_0000 }; 3])
                    .unwrap();
                context
                    .dispatch(
                        &mut graph,
                        &MLNamedTensors::from([("a", &a), ("b", &b)]),
                        &MLNamedTensors::from([("result", &result)]),
                    )
                    .unwrap();
                let mut actual = vec![0_u32; count as usize * 3];
                context.read_tensor(&result, &mut actual).unwrap();
                assert_eq!(actual, vec![0x3400_0000; actual.len()]);
                let invalid = context
                    .create_tensor(
                        &MLTensorDescriptor::new(MLOperandDataType::Float32, vec![4]).to_writable(),
                    )
                    .unwrap();
                assert!(
                    context
                        .dispatch(
                            &mut graph,
                            &MLNamedTensors::from([("a", &a), ("b", &invalid)]),
                            &MLNamedTensors::from([("result", &result)])
                        )
                        .is_err()
                );
                context.read_tensor(&result, &mut actual).unwrap();
                assert_eq!(actual, vec![0x3400_0000; actual.len()]);
                context
                    .dispatch(
                        &mut graph,
                        &MLNamedTensors::from([("a", &a), ("b", &b)]),
                        &MLNamedTensors::from([("result", &result)]),
                    )
                    .unwrap();
            }
        }
    }
}

#[test]
fn float32_binary_tiny_operands_have_normal_results_and_a_proven_typed_route() {
    // Exact powers of two give an independent integer reference: the first
    // three values are subnormal; multiplying by 2^126, or dividing by
    // 2^-126, yields 2^-23, 2^-22, and 1-2^-23. Zero is far outside the
    // existing Mul 1 ULP / Div 2 ULP allowance. The native M4 result can pass
    // this packet; the typed route also guards the reproduced A12 GPU loss.
    let input = [1_u32, 2, 0x007f_ffff, 0x8000_0001, 0, 0x8000_0000];
    let expected = [
        0x3400_0000_u32,
        0x3480_0000,
        0x3f7f_fffe,
        0xb400_0000,
        0,
        0x8000_0000,
    ];
    for divide in [false, true] {
        for mode in 0..3 {
            for policy in [DeviceType::Cpu, DeviceType::Gpu, DeviceType::Npu] {
                let mut context = context(mode, policy);
                let mut graph = context
                    .rustnn_build_graph(graph(divide, vec![Dimension::Static(6)]))
                    .unwrap();
                let descriptor = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![6]);
                let a = context
                    .create_tensor(&descriptor.clone().to_writable())
                    .unwrap();
                let b = context
                    .create_tensor(&descriptor.clone().to_writable())
                    .unwrap();
                let result = context.create_tensor(&descriptor.to_readable()).unwrap();
                context.write_tensor(&a, &input).unwrap();
                context
                    .write_tensor(&b, &[if divide { 0x0080_0000_u32 } else { 0x7e80_0000 }; 6])
                    .unwrap();
                context
                    .dispatch(
                        &mut graph,
                        &MLNamedTensors::from([("a", &a), ("b", &b)]),
                        &MLNamedTensors::from([("result", &result)]),
                    )
                    .unwrap();
                let mut actual = [0_u32; 6];
                context.read_tensor(&result, &mut actual).unwrap();
                assert_eq!(
                    actual, expected,
                    "divide={divide}, mode={mode}, policy={policy:?}"
                );
                let Some(LoadDiagnostics::Coreml(load)) = graph.rustnn_load_diagnostics() else {
                    panic!("missing diagnostics")
                };
                assert_eq!(load.route, CoremlLoadRoute::TypedHost);
                assert_eq!(load.loaded_compute_units, "NOT_APPLICABLE");
                assert!(load.failures.is_empty());
            }
        }
    }
}

#[test]
fn float32_exp_binary_chain_and_native_consumer_preserve_each_boundary() {
    // These exact local composition checks preserve the independently rounded
    // Exp boundary. They are not a new universal subgraph WPT tolerance: a
    // permitted Exp error can be amplified by subsequent scaling.
    let exponents = [
        0xc2ce0000_u32,
        0xc2c80000,
        0xc2be0000,
        0xc2b40000,
        0xc2b00000,
    ];
    let tails = [1_u32, 27, 3940, 584744, 4320708];
    let scaled = [
        0x34000000_u32,
        0x36580000,
        0x39f64000,
        0x3d8ec280,
        0x3f03db88,
    ];
    for divide in [false, true] {
        for mode in 0..3 {
            for policy in [DeviceType::Cpu, DeviceType::Gpu, DeviceType::Npu] {
                for constant_input in [false, true] {
                    for fanout in [false, true] {
                        for native_consumer in [false, true] {
                            let mut source = graph(divide, shape(&[5]));
                            source.operands[1].descriptor.shape = vec![];
                            source.operands[2].name = Some("scaled".into());
                            for name in ["exponential", "result"] {
                                let mut output = source.operands[2].clone();
                                output.name = Some(name.into());
                                source.operands.push(output);
                            }
                            source.operations = vec![
                                Operation::Exp {
                                    input: 0,
                                    options: None,
                                    outputs: vec![3],
                                },
                                if divide {
                                    Operation::Div {
                                        a: 3,
                                        b: 1,
                                        options: None,
                                        outputs: vec![2],
                                    }
                                } else {
                                    Operation::Mul {
                                        a: 3,
                                        b: 1,
                                        options: None,
                                        outputs: vec![2],
                                    }
                                },
                                Operation::Neg {
                                    input: 2,
                                    options: None,
                                    outputs: vec![4],
                                },
                            ];
                            let final_id = if native_consumer { 4 } else { 2 };
                            if !native_consumer {
                                source.operations.pop();
                                source.operands.pop();
                                source.operands[2].name = Some("result".into());
                            }
                            source.output_operands = if fanout {
                                if native_consumer {
                                    vec![2, 3, 4]
                                } else {
                                    vec![2, 3]
                                }
                            } else {
                                vec![final_id]
                            };
                            if !fanout {
                                if native_consumer {
                                    source.operands[2].kind = OperandKind::Intermediate;
                                }
                                source.operands[3].kind = OperandKind::Intermediate;
                            }
                            if constant_input {
                                constant(&mut source, 0, &exponents);
                            }
                            let mut context = context(mode, policy);
                            let mut graph = context.rustnn_build_graph(source).unwrap();
                            let descriptor =
                                MLTensorDescriptor::new(MLOperandDataType::Float32, vec![5]);
                            let a = context
                                .create_tensor(&descriptor.clone().to_writable())
                                .unwrap();
                            let b = context
                                .create_tensor(
                                    &MLTensorDescriptor::new(MLOperandDataType::Float32, vec![])
                                        .to_writable(),
                                )
                                .unwrap();
                            let x = context
                                .create_tensor(&descriptor.clone().to_readable())
                                .unwrap();
                            let y = context
                                .create_tensor(&descriptor.clone().to_readable())
                                .unwrap();
                            let z = context.create_tensor(&descriptor.to_readable()).unwrap();
                            context.write_tensor(&a, &exponents).unwrap();
                            context
                                .write_tensor(
                                    &b,
                                    &[if divide { 0x00800000_u32 } else { 0x7e800000 }],
                                )
                                .unwrap();
                            let mut inputs = MLNamedTensors::from([("b", &b)]);
                            if !constant_input {
                                inputs.insert("a", &a);
                            }
                            let mut outputs = MLNamedTensors::from([("result", &z)]);
                            if fanout {
                                if native_consumer {
                                    outputs.insert("scaled", &x);
                                }
                                outputs.insert("exponential", &y);
                            }
                            context.dispatch(&mut graph, &inputs, &outputs).unwrap();
                            drop(graph);
                            let mut checks = vec![(
                                &z,
                                if native_consumer {
                                    scaled.map(|n| n ^ 0x80000000)
                                } else {
                                    scaled
                                },
                            )];
                            if fanout {
                                if native_consumer {
                                    checks.push((&x, scaled));
                                }
                                checks.push((&y, tails));
                            }
                            for (output, expected) in checks {
                                let mut actual = [0_u32; 5];
                                context.read_tensor(output, &mut actual).unwrap();
                                assert_eq!(
                                    actual, expected,
                                    "{divide} {mode} {policy:?} constant={constant_input} fanout={fanout} native={native_consumer}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn float32_binary_materializes_constant_arithmetic_before_its_typed_boundary() {
    for divide in [false, true] {
        for mode in 0..3 {
            for scalar in [false, true] {
                let dims = if scalar { vec![] } else { vec![2] };
                let count = if scalar { 1 } else { 2 };
                let mut source = graph(divide, shape(&dims));
                let mut zero = source.operands[0].clone();
                zero.name = Some("zero".into());
                source.operands.push(zero);
                let mut sum = source.operands[0].clone();
                sum.kind = OperandKind::Intermediate;
                sum.name = Some("sum".into());
                source.operands.push(sum);
                constant(&mut source, 0, &vec![0x00800000; count]);
                constant(&mut source, 3, &vec![0; count]);
                constant(
                    &mut source,
                    1,
                    &vec![if divide { 0x40000000 } else { 0x3f000000 }; count],
                );
                source.operations = vec![
                    Operation::Add {
                        a: 0,
                        b: 3,
                        options: None,
                        outputs: vec![4],
                    },
                    if divide {
                        Operation::Div {
                            a: 4,
                            b: 1,
                            options: None,
                            outputs: vec![2],
                        }
                    } else {
                        Operation::Mul {
                            a: 4,
                            b: 1,
                            options: None,
                            outputs: vec![2],
                        }
                    },
                ];
                let mut context = context(mode, DeviceType::Cpu);
                let mut graph = context.rustnn_build_graph(source).unwrap();
                let output = context
                    .create_tensor(
                        &MLTensorDescriptor::new(
                            MLOperandDataType::Float32,
                            dims.into_iter().map(u64::from).collect(),
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
                let mut actual = vec![0_u32; count];
                context.read_tensor(&output, &mut actual).unwrap();
                assert_eq!(actual, vec![0x00400000; count]);
            }
        }
    }
}

#[test]
fn float32_binary_public_tensors_match_the_independent_special_and_finite_corpus() {
    let corpus = include_bytes!("fixtures/coreml_float32_binary/cases.bin");
    for divide in [false, true] {
        let rows: Vec<_> = corpus
            .as_chunks::<16>()
            .0
            .iter()
            .map(|row| {
                row.as_chunks::<4>()
                    .0
                    .iter()
                    .map(|n| u32::from_le_bytes(*n))
                    .collect::<Vec<_>>()
            })
            .filter(|row| row[0] == u32::from(divide))
            .collect();
        let left: Vec<_> = rows.iter().map(|row| row[1]).collect();
        let right: Vec<_> = rows.iter().map(|row| row[2]).collect();
        for mode in 0..3 {
            let mut context = context(mode, DeviceType::Cpu);
            let mut graph = context
                .rustnn_build_graph(graph(divide, shape(&[rows.len() as u32])))
                .unwrap();
            let descriptor =
                MLTensorDescriptor::new(MLOperandDataType::Float32, vec![rows.len() as u64]);
            let a = context
                .create_tensor(&descriptor.clone().to_writable())
                .unwrap();
            let b = context
                .create_tensor(&descriptor.clone().to_writable())
                .unwrap();
            let result = context.create_tensor(&descriptor.to_readable()).unwrap();
            context.write_tensor(&a, &left).unwrap();
            context.write_tensor(&b, &right).unwrap();
            context
                .dispatch(
                    &mut graph,
                    &MLNamedTensors::from([("a", &a), ("b", &b)]),
                    &MLNamedTensors::from([("result", &result)]),
                )
                .unwrap();
            let mut actual = vec![0_u32; rows.len()];
            context.read_tensor(&result, &mut actual).unwrap();
            for (actual, row) in actual.into_iter().zip(&rows) {
                if row[3] & 0x7fffffff > 0x7f800000 {
                    assert!(actual & 0x7fffffff > 0x7f800000);
                } else {
                    assert_eq!(
                        actual, row[3],
                        "{divide} {mode} {:08x} {:08x}",
                        row[1], row[2]
                    );
                }
            }
        }
    }
}
