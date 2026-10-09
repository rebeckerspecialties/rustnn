//! Integer division truncates toward zero; MIL floor_div alone does not.
#![cfg(all(target_os = "macos", feature = "coreml-runtime"))]

use rustnn::mlcontext::{
    Backend, MLContext, MLContextOptions, MLGraphBuilder, MLNamedOperands, MLNamedTensors,
    MLOperandDescriptor, MLPowerPreference, MLTensorDescriptor,
};
use rustnn::operator_enums::MLOperandDataType;

fn storage_context(mode: usize, accelerated: bool, power: MLPowerPreference) -> MLContext<'static> {
    let mut options = rustnn::mlcontext::RustNNOptions::default();
    options.coreml.reuse_tensor_storage = mode != 0;
    options.coreml.output_backings = mode == 2;
    MLContext::create(
        &MLContextOptions::new(power, accelerated)
            .with_rustnn_backend_hint(Backend::Coreml)
            .with_rustnn_options(options),
    )
    .unwrap()
}

fn check_division(left: &[i32], right: &[i32], constants: bool) {
    assert_eq!(left.len(), right.len());
    let shape = vec![left.len() as u64];
    let expected: Vec<_> = left
        .iter()
        .zip(right)
        .map(|(&a, &b)| i32::try_from(i64::from(a) / i64::from(b)).unwrap())
        .collect();
    check_case(
        (left, &shape),
        (right, &shape),
        (&shape, &expected),
        [constants; 2],
    );
}

fn check_case(
    left: (&[i32], &[u64]),
    right: (&[i32], &[u64]),
    output: (&[u64], &[i32]),
    constants: [bool; 2],
) {
    let (left, left_shape) = left;
    let (right, right_shape) = right;
    let (shape, expected) = output;
    for (accelerated, power) in [
        (false, MLPowerPreference::Default),
        (true, MLPowerPreference::Default),
        (true, MLPowerPreference::LowPower),
    ] {
        let mut context = MLContext::create(
            &MLContextOptions::new(power, accelerated).with_rustnn_backend_hint(Backend::Coreml),
        )
        .unwrap();
        let mut builder = MLGraphBuilder::new(&mut context).unwrap();
        let mut operand = |name, values: &[i32], shape: &[u64], constant| {
            let descriptor = MLOperandDescriptor::new(MLOperandDataType::Int32, shape.to_vec());
            if constant {
                builder
                    .constant_from_bytes(
                        &descriptor,
                        values
                            .iter()
                            .flat_map(|value| value.to_le_bytes())
                            .collect(),
                    )
                    .unwrap()
            } else {
                builder.input(name, &descriptor).unwrap()
            }
        };
        let a = operand("left", left, left_shape, constants[0]);
        let b = operand("right", right, right_shape, constants[1]);
        let result = builder.div(a, b).unwrap();
        let mut graph = builder
            .build(&MLNamedOperands::from([("result", result)]))
            .unwrap();
        let tensor_descriptor = MLTensorDescriptor::new(MLOperandDataType::Int32, shape.to_vec());
        let a = context
            .create_tensor(
                &MLTensorDescriptor::new(MLOperandDataType::Int32, left_shape.to_vec())
                    .to_writable(),
            )
            .unwrap();
        let b = context
            .create_tensor(
                &MLTensorDescriptor::new(MLOperandDataType::Int32, right_shape.to_vec())
                    .to_writable(),
            )
            .unwrap();
        let output = context
            .create_tensor(&tensor_descriptor.to_readable())
            .unwrap();
        let mut inputs = MLNamedTensors::new();
        if !constants[0] {
            context.write_tensor(&a, left).unwrap();
            inputs.insert("left", &a);
        }
        if !constants[1] {
            context.write_tensor(&b, right).unwrap();
            inputs.insert("right", &b);
        }
        context
            .dispatch(
                &mut graph,
                &inputs,
                &MLNamedTensors::from([("result", &output)]),
            )
            .unwrap();
        let mut actual = vec![0i32; expected.len()];
        context.read_tensor(&output, &mut actual).unwrap();
        let Some(rustnn::mlcontext::LoadDiagnostics::Coreml(diagnostics)) =
            graph.rustnn_load_diagnostics()
        else {
            panic!("CoreML diagnostics required")
        };
        assert_eq!(
            diagnostics.route,
            rustnn::executors::coreml::CoremlLoadRoute::TypedHost,
            "standalone Int32 Div must use the proved exact stage"
        );
        eprintln!(
            "Int32 Div: constants={constants:?}, left_shape={left_shape:?}, right_shape={right_shape:?}, output_shape={shape:?}, diagnostics={:?}",
            graph.rustnn_load_diagnostics()
        );
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(
                actual,
                expected,
                "index={index}, constants={constants:?}, accelerated={accelerated}, power={power:?}, diagnostics={:?}",
                graph.rustnn_load_diagnostics()
            );
        }
        assert_eq!(
            graph.output_descriptors["result"]
                .static_shape()
                .unwrap()
                .iter()
                .map(|&n| u64::from(n))
                .collect::<Vec<_>>(),
            shape
        );
    }
}

#[test]
fn int32_nondivisible_quotients_truncate_toward_zero() {
    check_division(
        &[-7, 7, -7, 7, -1, 1, -6, 6],
        &[3, -3, -3, 3, 3, -3, 3, -3],
        false,
    );
}

#[test]
fn int32_constant_nondivisible_quotients_truncate_toward_zero() {
    check_division(
        &[-7, 7, -7, 7, -1, 1, -6, 6],
        &[3, -3, -3, 3, 3, -3, 3, -3],
        true,
    );
}

#[test]
fn int32_division_keeps_values_beyond_float32_precision() {
    for constants in [false, true] {
        check_division(
            &[
                i32::MIN,
                i32::MIN,
                i32::MAX,
                i32::MAX,
                16_777_217,
                -16_777_217,
                -2_147_483_647,
                2_147_483_647,
            ],
            &[3, i32::MAX, -3, i32::MIN, 1, 1, 2, -2],
            constants,
        );
    }
}

#[test]
fn int32_division_signed_extremes_and_deterministic_full_range() {
    let cases = [
        i32::MIN,
        i32::MIN + 1,
        -16_777_217,
        -7,
        -3,
        -1,
        0,
        1,
        2,
        3,
        7,
        16_777_217,
        i32::MAX - 1,
        i32::MAX,
    ];
    let mut left = Vec::new();
    let mut right = Vec::new();
    for a in cases {
        for b in cases {
            if b != 0 && !(a == i32::MIN && b == -1) {
                left.push(a);
                right.push(b);
            }
        }
    }
    let mut state = 0x8320_59a3u32;
    for _ in 0..20_000 {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let a = state as i32;
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let b = state as i32;
        if b != 0 && !(a == i32::MIN && b == -1) {
            left.push(a);
            right.push(b);
        }
    }
    for constants in [false, true] {
        check_division(&left, &right, constants);
    }
}

#[test]
fn int32_division_scalar_and_multidirectional_broadcast() {
    for constants in [[false, false], [true, false], [false, true], [true, true]] {
        check_case((&[-7], &[]), (&[3], &[]), (&[], &[-2]), constants);
        check_case(
            (&[16_777_217, -16_777_217], &[2, 1]),
            (&[1, -1, 3], &[1, 3]),
            (
                &[2, 3],
                &[
                    16_777_217,
                    -16_777_217,
                    5_592_405,
                    -16_777_217,
                    16_777_217,
                    -5_592_405,
                ],
            ),
            constants,
        );
    }
}

#[test]
#[cfg(feature = "dynamic-inputs")]
fn int32_division_dynamic_bindings_keep_public_output_aliases_exact() {
    use rustnn::graph::{
        ConstantData, DataType, Dimension, DynamicDimension, GraphInfo, Operand, OperandDescriptor,
        OperandKind,
    };
    use rustnn::operators::Operation;
    let dynamic = vec![Dimension::Dynamic(DynamicDimension {
        name: "length".into(),
        max_size: 8,
    })];
    let operand = |name: &str, kind, shape| Operand {
        name: Some(name.into()),
        kind,
        descriptor: OperandDescriptor {
            data_type: DataType::Int32,
            shape,
            pending_permutation: vec![],
        },
    };
    // Both quotient and its copy are public outputs. This exercises output
    // coalescing and dynamic rebinding, not an intermediate native Identity.
    let info = GraphInfo {
        operands: vec![
            operand("left", OperandKind::Input, dynamic.clone()),
            operand("right", OperandKind::Constant, vec![]),
            operand("quotient", OperandKind::Output, dynamic.clone()),
            operand("result", OperandKind::Output, dynamic),
        ],
        input_operands: vec![0],
        output_operands: vec![2, 3],
        constant_operand_ids_to_handles: [(
            1,
            ConstantData {
                data: 3i32.to_le_bytes().to_vec(),
                label: None,
            },
        )]
        .into(),
        operations: vec![
            Operation::Div {
                a: 0,
                b: 1,
                options: None,
                outputs: vec![2],
            },
            Operation::Identity {
                input: 2,
                options: None,
                outputs: vec![3],
            },
        ],
        ..Default::default()
    };
    for (accelerated, power) in [
        (false, MLPowerPreference::Default),
        (true, MLPowerPreference::Default),
        (true, MLPowerPreference::LowPower),
    ] {
        let mut context = MLContext::create(
            &MLContextOptions::new(power, accelerated).with_rustnn_backend_hint(Backend::Coreml),
        )
        .unwrap();
        let mut graph = context.rustnn_build_graph(info.clone()).unwrap();
        for count in [1, 8, 3, 1] {
            let descriptor = MLTensorDescriptor::new(MLOperandDataType::Int32, vec![count]);
            let input = context
                .create_tensor(&descriptor.clone().to_writable())
                .unwrap();
            let quotient = context
                .create_tensor(&descriptor.clone().to_readable())
                .unwrap();
            let result = context.create_tensor(&descriptor.to_readable()).unwrap();
            let mut source = [i32::MIN, -16_777_217, -7, -1, 0, 7, 16_777_217, i32::MAX]
                [..count as usize]
                .to_vec();
            for _ in 0..2 {
                context.write_tensor(&input, &source).unwrap();
                context
                    .dispatch(
                        &mut graph,
                        &MLNamedTensors::from([("left", &input)]),
                        &MLNamedTensors::from([("quotient", &quotient), ("result", &result)]),
                    )
                    .unwrap();
                let expected = source
                    .iter()
                    .map(|&n| (i64::from(n) / 3) as i32)
                    .collect::<Vec<_>>();
                for output in [&quotient, &result] {
                    let mut actual = vec![0; count as usize];
                    context.read_tensor(output, &mut actual).unwrap();
                    assert_eq!(
                        actual, expected,
                        "count={count}, accelerated={accelerated}, power={power:?}"
                    );
                }
                source.reverse();
            }
        }
    }
}

#[test]
fn int32_division_errors_do_not_publish_partial_outputs_in_any_storage_mode() {
    for mode in 0..3 {
        for (accelerated, power) in [
            (false, MLPowerPreference::Default),
            (true, MLPowerPreference::Default),
            (true, MLPowerPreference::LowPower),
        ] {
            let mut context = storage_context(mode, accelerated, power);
            let descriptor = MLOperandDescriptor::new(MLOperandDataType::Int32, vec![4]);
            let mut builder = MLGraphBuilder::new(&mut context).unwrap();
            let a = builder.input("left", &descriptor).unwrap();
            let b = builder.input("right", &descriptor).unwrap();
            let quotient = builder.div(a, b).unwrap();
            let copy = builder.identity(a).unwrap();
            let mut graph = builder
                .build(&MLNamedOperands::from([
                    ("quotient", quotient),
                    ("source", copy),
                ]))
                .unwrap();
            let descriptor = MLTensorDescriptor::new(MLOperandDataType::Int32, vec![4])
                .to_readable()
                .to_writable();
            let left = context.create_tensor(&descriptor).unwrap();
            let right = context.create_tensor(&descriptor).unwrap();
            let output = context.create_tensor(&descriptor).unwrap();
            let source = context.create_tensor(&descriptor).unwrap();
            context.write_tensor(&left, &[-7, 7, 17, i32::MIN]).unwrap();
            for (divisor, message) in [
                ([3, 3, 3, 0], "division by zero"),
                ([3, 3, 3, -1], "outside the represented range"),
            ] {
                context.write_tensor(&right, &divisor).unwrap();
                for tensor in [&output, &source] {
                    context.write_tensor(tensor, &[123_456_789; 4]).unwrap();
                }
                let error = context
                    .dispatch(
                        &mut graph,
                        &MLNamedTensors::from([("left", &left), ("right", &right)]),
                        &MLNamedTensors::from([("quotient", &output), ("source", &source)]),
                    )
                    .unwrap_err();
                assert!(error.to_string().contains(message), "{error}");
                for tensor in [&output, &source] {
                    let mut actual = [0; 4];
                    context.read_tensor(tensor, &mut actual).unwrap();
                    assert_eq!(
                        actual, [123_456_789; 4],
                        "mode={mode}, policy={power:?}, accelerated={accelerated}"
                    );
                }
            }
            context.write_tensor(&right, &[3, 3, 3, 2]).unwrap();
            context
                .dispatch(
                    &mut graph,
                    &MLNamedTensors::from([("left", &left), ("right", &right)]),
                    &MLNamedTensors::from([("quotient", &output), ("source", &source)]),
                )
                .unwrap();
            let mut actual = [0; 4];
            context.read_tensor(&output, &mut actual).unwrap();
            assert_eq!(actual, [-2, 2, 5, i32::MIN / 2]);
        }
    }
}

#[test]
fn int32_division_constant_views_and_native_cast_producers_keep_stage_boundaries() {
    use rustnn::graph::{
        ConstantData, DataType, GraphInfo, Operand, OperandDescriptor, OperandKind,
        to_dimension_vector,
    };
    use rustnn::operator_options::MLDimension;
    use rustnn::operators::Operation;
    let operand = |name: &str, kind, data_type, shape: &[u32]| Operand {
        name: Some(name.into()),
        kind,
        descriptor: OperandDescriptor {
            data_type,
            shape: to_dimension_vector(shape),
            pending_permutation: vec![],
        },
    };
    for producer in [4, 5, 0, 1, 2, 3] {
        let native_cast = producer == 1;
        let scalar = producer >= 4;
        let input_shape: &[u32] = if producer == 4 {
            &[1]
        } else if producer == 5 {
            &[]
        } else {
            &[2, 2]
        };
        let view_shape: &[u32] = if scalar { &[] } else { &[2, 2] };
        let count = if scalar { 1 } else { 4 };
        let bytes: Vec<u8> = if native_cast {
            [-7.0_f32, 7.0, 17.0, 16_777_216.0]
                .iter()
                .flat_map(|n| n.to_le_bytes())
                .collect()
        } else if scalar {
            (-16_777_217_i32).to_le_bytes().to_vec()
        } else {
            [16_777_217_i32, -16_777_217, -7, 7]
                .iter()
                .flat_map(|n| n.to_le_bytes())
                .collect()
        };
        let info = GraphInfo {
            operands: vec![
                operand(
                    "constant",
                    OperandKind::Constant,
                    if native_cast {
                        DataType::Float32
                    } else {
                        DataType::Int32
                    },
                    input_shape,
                ),
                operand(
                    "view",
                    OperandKind::Intermediate,
                    DataType::Int32,
                    view_shape,
                ),
                operand(
                    "identity",
                    OperandKind::Intermediate,
                    DataType::Int32,
                    view_shape,
                ),
                operand("flat", OperandKind::Intermediate, DataType::Int32, &[count]),
                operand("divisor", OperandKind::Constant, DataType::Int32, &[]),
                operand("result", OperandKind::Output, DataType::Int32, &[count]),
            ],
            output_operands: vec![5],
            constant_operand_ids_to_handles: [
                (
                    0,
                    ConstantData {
                        data: bytes,
                        label: None,
                    },
                ),
                (
                    4,
                    ConstantData {
                        data: 3_i32.to_le_bytes().to_vec(),
                        label: None,
                    },
                ),
            ]
            .into(),
            operations: vec![
                if native_cast {
                    Operation::Cast {
                        input: 0,
                        data_type: MLOperandDataType::Int32,
                        outputs: vec![1],
                        options: None,
                    }
                } else if producer == 2 {
                    Operation::Identity {
                        input: 0,
                        outputs: vec![1],
                        options: None,
                    }
                } else if producer == 3 {
                    Operation::Cast {
                        input: 0,
                        data_type: MLOperandDataType::Int32,
                        outputs: vec![1],
                        options: None,
                    }
                } else if producer == 4 {
                    Operation::Reshape {
                        input: 0,
                        new_shape: vec![],
                        outputs: vec![1],
                        options: None,
                    }
                } else {
                    Operation::Transpose {
                        input: 0,
                        outputs: vec![1],
                        options: None,
                    }
                },
                Operation::Identity {
                    input: 1,
                    outputs: vec![2],
                    options: None,
                },
                Operation::Reshape {
                    input: 2,
                    new_shape: vec![MLDimension::Static(count)],
                    outputs: vec![3],
                    options: None,
                },
                Operation::Div {
                    a: 3,
                    b: 4,
                    outputs: vec![5],
                    options: None,
                },
            ],
            ..Default::default()
        };
        for (accelerated, power) in [
            (false, MLPowerPreference::Default),
            (true, MLPowerPreference::Default),
            (true, MLPowerPreference::LowPower),
        ] {
            let mut context = storage_context(2, accelerated, power);
            let mut graph = context.rustnn_build_graph(info.clone()).unwrap();
            let output = context
                .create_tensor(
                    &MLTensorDescriptor::new(MLOperandDataType::Int32, vec![u64::from(count)])
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
            let mut actual = vec![0; count as usize];
            context.read_tensor(&output, &mut actual).unwrap();
            assert_eq!(
                actual,
                if native_cast {
                    vec![-2, 2, 5, 5_592_405]
                } else if scalar {
                    vec![-5_592_405]
                } else if producer > 1 {
                    vec![5_592_405, -5_592_405, -2, 2]
                } else {
                    vec![5_592_405, -2, -5_592_405, 2]
                },
                "producer={producer}"
            );
            if !native_cast {
                let Some(rustnn::mlcontext::LoadDiagnostics::Coreml(diagnostic)) =
                    graph.rustnn_load_diagnostics()
                else {
                    panic!()
                };
                assert_eq!(
                    diagnostic.route,
                    rustnn::executors::coreml::CoremlLoadRoute::TypedHost,
                    "producer={producer}"
                );
            }
        }
    }
}
