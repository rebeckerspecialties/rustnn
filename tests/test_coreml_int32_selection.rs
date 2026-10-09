//! Preserve the selected Int32 operand, including values not representable as Float32.
#![cfg(all(target_os = "macos", feature = "coreml-runtime"))]

use rustnn::mlcontext::{
    Backend, MLContext, MLContextOptions, MLGraphBuilder, MLNamedOperands, MLNamedTensors,
    MLOperandDescriptor, MLPowerPreference, MLTensorDescriptor, RustNNOptions,
};
use rustnn::operator_enums::MLOperandDataType;

fn context(mode: usize, accelerated: bool, power: MLPowerPreference) -> MLContext<'static> {
    let mut options = RustNNOptions::default();
    options.coreml.reuse_tensor_storage = mode != 0;
    options.coreml.output_backings = mode == 2;
    MLContext::create(
        &MLContextOptions::new(power, accelerated)
            .with_rustnn_backend_hint(Backend::Coreml)
            .with_rustnn_options(options),
    )
    .unwrap()
}

fn check(
    maximum: bool,
    constants: [bool; 2],
    left: (&[i32], &[u64]),
    right: (&[i32], &[u64]),
    expected: (&[i32], &[u64]),
) {
    for mode in 0..3 {
        for (accelerated, power) in [
            (false, MLPowerPreference::Default),
            (true, MLPowerPreference::Default),
            (true, MLPowerPreference::LowPower),
        ] {
            let mut context = context(mode, accelerated, power);
            let mut builder = MLGraphBuilder::new(&mut context).unwrap();
            let mut operand = |name, (values, shape): (&[i32], &[u64]), constant| {
                let descriptor = MLOperandDescriptor::new(MLOperandDataType::Int32, shape.to_vec());
                if constant {
                    builder
                        .constant_from_bytes(
                            &descriptor,
                            values.iter().flat_map(|n| n.to_le_bytes()).collect(),
                        )
                        .unwrap()
                } else {
                    builder.input(name, &descriptor).unwrap()
                }
            };
            let a = operand("left", left, constants[0]);
            let b = operand("right", right, constants[1]);
            let selection = if maximum {
                builder.max(a, b)
            } else {
                builder.min(a, b)
            }
            .unwrap();
            let mut graph = builder
                .build(&MLNamedOperands::from([("result", selection)]))
                .unwrap();
            let make_tensor = |context: &mut MLContext, shape: &[u64]| {
                context
                    .create_tensor(
                        &MLTensorDescriptor::new(MLOperandDataType::Int32, shape.to_vec())
                            .to_readable()
                            .to_writable(),
                    )
                    .unwrap()
            };
            let a = make_tensor(&mut context, left.1);
            let b = make_tensor(&mut context, right.1);
            let output = make_tensor(&mut context, expected.1);
            let mut inputs = MLNamedTensors::new();
            for (name, tensor, values, constant) in [
                ("left", &a, left.0, constants[0]),
                ("right", &b, right.0, constants[1]),
            ] {
                if !constant {
                    context.write_tensor(tensor, values).unwrap();
                    inputs.insert(name, tensor);
                }
            }
            for _ in 0..2 {
                context
                    .write_tensor(&output, &vec![123_i32; expected.0.len()])
                    .unwrap();
                context
                    .dispatch(
                        &mut graph,
                        &inputs,
                        &MLNamedTensors::from([("result", &output)]),
                    )
                    .unwrap();
                let mut actual = vec![0_i32; expected.0.len()];
                context.read_tensor(&output, &mut actual).unwrap();
                assert_eq!(
                    actual, expected.0,
                    "max={maximum}, constants={constants:?}, mode={mode}, accelerated={accelerated}, power={power:?}"
                );
            }
        }
    }
}

#[test]
fn int32_selection_preserves_full_range_selected_operand_bits() {
    let left = [16_777_217, -16_777_217, i32::MAX, i32::MIN, 0, -1];
    let right = [16_777_218, -16_777_218, i32::MAX - 1, i32::MIN + 1, -1, 0];
    for maximum in [false, true] {
        let expected: Vec<_> = left
            .iter()
            .zip(right)
            .map(|(&a, b)| if maximum { a.max(b) } else { a.min(b) })
            .collect();
        for constants in [[false, false], [true, false], [false, true], [true, true]] {
            check(
                maximum,
                constants,
                (&left, &[6]),
                (&right, &[6]),
                (&expected, &[6]),
            );
        }
    }
}

#[test]
fn int32_selection_scalar_and_multidirectional_broadcast_are_exact() {
    for maximum in [false, true] {
        for constants in [[false, false], [true, false], [false, true], [true, true]] {
            let expected = if maximum { 16_777_218 } else { 16_777_217 };
            check(
                maximum,
                constants,
                (&[16_777_217], &[]),
                (&[16_777_218], &[]),
                (&[expected], &[]),
            );
            let expected = if maximum {
                [
                    16_777_217,
                    16_777_218,
                    16_777_217,
                    -16_777_217,
                    16_777_218,
                    0,
                ]
            } else {
                [
                    -16_777_218,
                    16_777_217,
                    0,
                    -16_777_218,
                    -16_777_217,
                    -16_777_217,
                ]
            };
            check(
                maximum,
                constants,
                (&[16_777_217, -16_777_217], &[2, 1]),
                (&[-16_777_218, 16_777_218, 0], &[1, 3]),
                (&expected, &[2, 3]),
            );
        }
    }
}

#[test]
fn int32_selection_native_neighbors_and_shared_constant_remain_independent() {
    for mode in 0..3 {
        for maximum in [false, true] {
            let mut context = context(mode, true, MLPowerPreference::Default);
            let mut builder = MLGraphBuilder::new(&mut context).unwrap();
            let source = builder
                .input(
                    "source",
                    &MLOperandDescriptor::new(MLOperandDataType::Float32, vec![2]),
                )
                .unwrap();
            let cast = builder.cast(source, MLOperandDataType::Int32).unwrap();
            let original = [16_777_217_i32, -16_777_217];
            let constant = builder
                .constant_from_bytes(
                    &MLOperandDescriptor::new(MLOperandDataType::Int32, vec![2]),
                    original.iter().flat_map(|n| n.to_le_bytes()).collect(),
                )
                .unwrap();
            let selected = if maximum {
                builder.max(cast, constant)
            } else {
                builder.min(cast, constant)
            }
            .unwrap();
            let converted = builder.cast(selected, MLOperandDataType::Float32).unwrap();
            let copy = builder.identity(constant).unwrap();
            let mut graph = builder
                .build(&MLNamedOperands::from([
                    ("selected", selected),
                    ("float", converted),
                    ("constant", copy),
                ]))
                .unwrap();
            let float = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![2])
                .to_readable()
                .to_writable();
            let integer = MLTensorDescriptor::new(MLOperandDataType::Int32, vec![2])
                .to_readable()
                .to_writable();
            let source = context.create_tensor(&float).unwrap();
            let float_output = context.create_tensor(&float).unwrap();
            let selected = context.create_tensor(&integer).unwrap();
            let copied = context.create_tensor(&integer).unwrap();
            context
                .write_tensor(&source, &[-3.0_f32, 16_777_216.0])
                .unwrap();
            let expected = if maximum {
                [16_777_217, 16_777_216]
            } else {
                [-3, -16_777_217]
            };
            for _ in 0..2 {
                // Public output mutation must not alter an owned constant or a later selection.
                context.write_tensor(&copied, &[123_i32, 456]).unwrap();
                context
                    .dispatch(
                        &mut graph,
                        &MLNamedTensors::from([("source", &source)]),
                        &MLNamedTensors::from([
                            ("selected", &selected),
                            ("float", &float_output),
                            ("constant", &copied),
                        ]),
                    )
                    .unwrap();
                let mut actual = [0_i32; 2];
                context.read_tensor(&selected, &mut actual).unwrap();
                assert_eq!(actual, expected, "mode={mode}, maximum={maximum}");
                context.read_tensor(&copied, &mut actual).unwrap();
                assert_eq!(actual, original);
                let mut actual_float = [0_f32; 2];
                context
                    .read_tensor(&float_output, &mut actual_float)
                    .unwrap();
                assert_eq!(actual_float, expected.map(|n| n as f32));
            }
        }
    }
}

#[test]
fn int32_selection_bad_binding_is_atomic_and_recoverable() {
    for mode in 0..3 {
        let mut context = context(mode, false, MLPowerPreference::Default);
        let mut builder = MLGraphBuilder::new(&mut context).unwrap();
        let descriptor = MLOperandDescriptor::new(MLOperandDataType::Int32, vec![2]);
        let a = builder.input("a", &descriptor).unwrap();
        let b = builder.input("b", &descriptor).unwrap();
        let minimum = builder.min(a, b).unwrap();
        let maximum = builder.max(a, b).unwrap();
        let mut graph = builder
            .build(&MLNamedOperands::from([("min", minimum), ("max", maximum)]))
            .unwrap();
        let descriptor = MLTensorDescriptor::new(MLOperandDataType::Int32, vec![2])
            .to_readable()
            .to_writable();
        let a = context.create_tensor(&descriptor).unwrap();
        let b = context.create_tensor(&descriptor).unwrap();
        let minimum = context.create_tensor(&descriptor).unwrap();
        let maximum = context.create_tensor(&descriptor).unwrap();
        let wrong = context
            .create_tensor(
                &MLTensorDescriptor::new(MLOperandDataType::Float32, vec![2])
                    .to_readable()
                    .to_writable(),
            )
            .unwrap();
        context
            .write_tensor(&a, &[16_777_217_i32, -16_777_217])
            .unwrap();
        context
            .write_tensor(&b, &[16_777_218_i32, -16_777_218])
            .unwrap();
        for _ in 0..2 {
            context.write_tensor(&minimum, &[123_i32, 456]).unwrap();
            context.write_tensor(&wrong, &[123.0_f32, 456.0]).unwrap();
            let error = context
                .dispatch(
                    &mut graph,
                    &MLNamedTensors::from([("a", &a), ("b", &b)]),
                    &MLNamedTensors::from([("min", &minimum), ("max", &wrong)]),
                )
                .unwrap_err();
            assert!(error.to_string().to_lowercase().contains("type"), "{error}");
            let mut actual = [0_i32; 2];
            context.read_tensor(&minimum, &mut actual).unwrap();
            assert_eq!(actual, [123, 456]);
            let mut actual_wrong = [0_f32; 2];
            context.read_tensor(&wrong, &mut actual_wrong).unwrap();
            assert_eq!(actual_wrong, [123.0, 456.0]);
            context
                .dispatch(
                    &mut graph,
                    &MLNamedTensors::from([("a", &a), ("b", &b)]),
                    &MLNamedTensors::from([("min", &minimum), ("max", &maximum)]),
                )
                .unwrap();
            context.read_tensor(&minimum, &mut actual).unwrap();
            assert_eq!(actual, [16_777_217, -16_777_218]);
            context.read_tensor(&maximum, &mut actual).unwrap();
            assert_eq!(actual, [16_777_218, -16_777_217]);
        }
    }
}

#[test]
#[cfg(feature = "dynamic-inputs")]
fn int32_selection_dynamic_shapes_grow_shrink_and_rebind() {
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
    let info = GraphInfo {
        operands: vec![
            operand("source", OperandKind::Input, dynamic.clone()),
            operand("limit", OperandKind::Constant, vec![]),
            operand("min", OperandKind::Output, dynamic.clone()),
            operand("max", OperandKind::Output, dynamic),
        ],
        input_operands: vec![0],
        output_operands: vec![2, 3],
        constant_operand_ids_to_handles: [(
            1,
            ConstantData {
                data: 16_777_217_i32.to_le_bytes().to_vec(),
                label: None,
            },
        )]
        .into(),
        operations: vec![
            Operation::Min {
                a: 0,
                b: 1,
                outputs: vec![2],
                options: None,
            },
            Operation::Max {
                a: 0,
                b: 1,
                outputs: vec![3],
                options: None,
            },
        ],
        ..Default::default()
    };
    for mode in 0..3 {
        let mut context = context(mode, true, MLPowerPreference::LowPower);
        let mut graph = context.rustnn_build_graph(info.clone()).unwrap();
        // The bounded descriptor may start at zero, but native feature storage
        // still requires positive actual extents. Rejection must be atomic and
        // must not poison the following nonempty calls on the same graph.
        let zero = MLTensorDescriptor::new(MLOperandDataType::Int32, vec![0])
            .to_readable()
            .to_writable();
        let empty_source = context.create_tensor(&zero).unwrap();
        let empty_minimum = context.create_tensor(&zero).unwrap();
        let empty_maximum = context.create_tensor(&zero).unwrap();
        for output in [&empty_minimum, &empty_maximum] {
            context.write_tensor(output, &[123_i32]).unwrap();
        }
        let error = context
            .dispatch(
                &mut graph,
                &MLNamedTensors::from([("source", &empty_source)]),
                &MLNamedTensors::from([("min", &empty_minimum), ("max", &empty_maximum)]),
            )
            .unwrap_err();
        assert!(
            error.to_string().contains("positive") || error.to_string().contains("zero"),
            "{error}"
        );
        for output in [&empty_minimum, &empty_maximum] {
            let mut actual = [0_i32];
            context.read_tensor(output, &mut actual).unwrap();
            assert_eq!(actual, [123]);
        }
        for length in [1, 8, 3, 1] {
            let descriptor = MLTensorDescriptor::new(MLOperandDataType::Int32, vec![length])
                .to_readable()
                .to_writable();
            let source = context.create_tensor(&descriptor).unwrap();
            let minimum = context.create_tensor(&descriptor).unwrap();
            let maximum = context.create_tensor(&descriptor).unwrap();
            let values = [
                i32::MIN,
                -16_777_217,
                -1,
                0,
                16_777_216,
                16_777_217,
                16_777_218,
                i32::MAX,
            ];
            context
                .write_tensor(&source, &values[..length as usize])
                .unwrap();
            context
                .dispatch(
                    &mut graph,
                    &MLNamedTensors::from([("source", &source)]),
                    &MLNamedTensors::from([("min", &minimum), ("max", &maximum)]),
                )
                .unwrap();
            for (maximum, tensor) in [(false, &minimum), (true, &maximum)] {
                let mut actual = vec![0_i32; length as usize];
                context.read_tensor(tensor, &mut actual).unwrap();
                assert_eq!(
                    actual,
                    values[..length as usize]
                        .iter()
                        .map(|&n| if maximum {
                            n.max(16_777_217)
                        } else {
                            n.min(16_777_217)
                        })
                        .collect::<Vec<_>>()
                );
            }
        }
    }
}
