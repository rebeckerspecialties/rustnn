//! Selecting an integer index must not introduce ties by first rounding to Float32.

use prost::Message;
use rustnn::converters::{CoremlMlProgramConverter, GraphConverter};
use rustnn::graph::{
    ConstantData, DataType, GraphInfo, Operand, OperandDescriptor, OperandKind, to_dimension_vector,
};
use rustnn::operator_enums::MLOperandDataType;
use rustnn::operator_options::MLArgMinMaxOptions;
use rustnn::operators::Operation;
use rustnn::protos::coreml::{mil_spec, specification};

fn graph(
    values: &[i32],
    shape: &[u32],
    axis: u32,
    maximum: bool,
    constant: bool,
    options: MLArgMinMaxOptions,
) -> GraphInfo {
    let mut output_shape = shape.to_vec();
    if options.keep_dimensions {
        output_shape[axis as usize] = 1;
    } else {
        output_shape.remove(axis as usize);
    }
    let operand = |name: &str, kind, data_type, shape: &[u32]| Operand {
        name: Some(name.into()),
        kind,
        descriptor: OperandDescriptor {
            data_type,
            shape: to_dimension_vector(shape),
            pending_permutation: vec![],
        },
    };
    GraphInfo {
        operands: vec![
            operand(
                "source",
                if constant {
                    OperandKind::Constant
                } else {
                    OperandKind::Input
                },
                DataType::Int32,
                shape,
            ),
            operand(
                "result",
                OperandKind::Output,
                options.output_data_type.into(),
                &output_shape,
            ),
        ],
        input_operands: if constant { vec![] } else { vec![0] },
        output_operands: vec![1],
        constant_operand_ids_to_handles: if constant {
            [(
                0,
                ConstantData {
                    data: values
                        .iter()
                        .flat_map(|value| value.to_le_bytes())
                        .collect(),
                    label: None,
                },
            )]
            .into_iter()
            .collect()
        } else {
            Default::default()
        },
        operations: vec![if maximum {
            Operation::ArgMax {
                input: 0,
                axis,
                options: Some(options),
                outputs: vec![1],
            }
        } else {
            Operation::ArgMin {
                input: 0,
                axis,
                options: Some(options),
                outputs: vec![1],
            }
        }],
        ..Default::default()
    }
}

fn inspect_integer_reduction(model: &specification::Model) -> usize {
    match model.r#type.as_ref().unwrap() {
        specification::model::Type::Pipeline(pipeline) => {
            pipeline.models.iter().map(inspect_integer_reduction).sum()
        }
        specification::model::Type::MlProgram(program) => {
            let function = &program.functions["main"];
            let block = &function.block_specializations[&function.opset];
            let mut types: std::collections::HashMap<_, _> = function
                .inputs
                .iter()
                .map(|input| (&input.name, input.r#type.as_ref().unwrap()))
                .collect();
            let mut count = 0;
            for operation in &block.operations {
                if matches!(operation.r#type.as_str(), "reduce_argmin" | "reduce_argmax") {
                    let Some(mil_spec::argument::binding::Binding::Name(name)) =
                        &operation.inputs["x"].arguments[0].binding
                    else {
                        panic!("named integer reduction input required")
                    };
                    let Some(mil_spec::value_type::Type::TensorType(tensor)) = &types[name].r#type
                    else {
                        panic!("tensor reduction input required")
                    };
                    assert_eq!(
                        tensor.data_type,
                        mil_spec::DataType::Int32 as i32,
                        "integer selection must not round its input through Float32"
                    );
                    count += 1;
                }
                for output in &operation.outputs {
                    types.insert(&output.name, output.r#type.as_ref().unwrap());
                }
            }
            count
        }
        _ => panic!("MLProgram or precision pipeline required"),
    }
}

#[test]
fn int32_arg_selection_exports_native_integer_input() {
    for maximum in [false, true] {
        for constant in [false, true] {
            for output_data_type in [MLOperandDataType::Int32, MLOperandDataType::Int64] {
                let source = graph(
                    &[16_777_216, 16_777_217],
                    &[2],
                    0,
                    maximum,
                    constant,
                    MLArgMinMaxOptions {
                        output_data_type,
                        ..Default::default()
                    },
                );
                let converted = CoremlMlProgramConverter.convert(&source).unwrap();
                let model = specification::Model::decode(converted.data.as_slice()).unwrap();
                assert_eq!(inspect_integer_reduction(&model), 1);
            }
        }
    }
}

#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
mod runtime {
    use super::*;
    use rustnn::mlcontext::{
        Backend, MLContext, MLContextOptions, MLNamedTensors, MLPowerPreference,
        MLTensorDescriptor, RustNNOptions,
    };

    // Return all valid winners: WebNN intentionally leaves true ties implementation-dependent.
    fn winners(values: &[i32], shape: &[u32], axis: usize, maximum: bool) -> Vec<Vec<i64>> {
        let inner = shape[axis + 1..]
            .iter()
            .map(|&n| n as usize)
            .product::<usize>();
        let width = shape[axis] as usize;
        (0..values.len() / width)
            .map(|cell| {
                let start = cell / inner * width * inner + cell % inner;
                let row: Vec<_> = (0..width)
                    .map(|index| values[start + index * inner])
                    .collect();
                let selected = if maximum {
                    row.iter().max()
                } else {
                    row.iter().min()
                }
                .unwrap();
                row.iter()
                    .enumerate()
                    .filter_map(|(index, value)| (value == selected).then_some(index as i64))
                    .collect()
            })
            .collect()
    }

    fn check(source: GraphInfo, values: &[i32], mode: usize) {
        let operation = &source.operations[0];
        let (axis, maximum) = match operation {
            Operation::ArgMax { axis, .. } => (*axis as usize, true),
            Operation::ArgMin { axis, .. } => (*axis as usize, false),
            _ => unreachable!(),
        };
        let shape = source.operands[0].descriptor.static_shape().unwrap();
        let expected = winners(values, &shape, axis, maximum);
        let output_descriptor = &source.operands[1].descriptor;
        let output_shape: Vec<_> = output_descriptor
            .static_shape()
            .unwrap()
            .into_iter()
            .map(u64::from)
            .collect();
        let output_dtype = match output_descriptor.data_type {
            DataType::Int32 => MLOperandDataType::Int32,
            DataType::Int64 => MLOperandDataType::Int64,
            _ => unreachable!(),
        };
        for (accelerated, power) in [
            (false, MLPowerPreference::Default),
            (true, MLPowerPreference::Default),
            (true, MLPowerPreference::LowPower),
        ] {
            let mut options = RustNNOptions::default();
            options.coreml.reuse_tensor_storage = mode != 0;
            options.coreml.output_backings = mode == 2;
            let mut context = MLContext::create(
                &MLContextOptions::new(power, accelerated)
                    .with_rustnn_backend_hint(Backend::Coreml)
                    .with_rustnn_options(options),
            )
            .unwrap();
            let mut compiled = context.rustnn_build_graph(source.clone()).unwrap();
            let input = context
                .create_tensor(
                    &MLTensorDescriptor::new(
                        MLOperandDataType::Int32,
                        shape.iter().map(|&n| u64::from(n)).collect(),
                    )
                    .to_writable(),
                )
                .unwrap();
            context.write_tensor(&input, values).unwrap();
            let output = context
                .create_tensor(
                    &MLTensorDescriptor::new(output_dtype, output_shape.clone()).to_readable(),
                )
                .unwrap();
            context
                .dispatch(
                    &mut compiled,
                    &if source.input_operands.is_empty() {
                        MLNamedTensors::new()
                    } else {
                        MLNamedTensors::from([("source", &input)])
                    },
                    &MLNamedTensors::from([("result", &output)]),
                )
                .unwrap();
            let Some(rustnn::mlcontext::LoadDiagnostics::Coreml(diagnostics)) =
                compiled.rustnn_load_diagnostics()
            else {
                panic!("CoreML load diagnostics required")
            };
            assert_eq!(
                diagnostics.route,
                rustnn::executors::coreml::CoremlLoadRoute::TypedHost,
                "the complete exact integer reduction must not execute natively"
            );
            let actual = if output_dtype == MLOperandDataType::Int64 {
                let mut actual = vec![0i64; expected.len()];
                context.read_tensor(&output, &mut actual).unwrap();
                actual
            } else {
                let mut actual = vec![0i32; expected.len()];
                context.read_tensor(&output, &mut actual).unwrap();
                actual.into_iter().map(i64::from).collect()
            };
            assert_eq!(
                compiled.output_descriptors["result"].data_type,
                output_descriptor.data_type
            );
            assert_eq!(
                compiled.output_descriptors["result"].shape,
                output_descriptor.shape
            );
            for (index, (actual, allowed)) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    allowed.contains(actual),
                    "cell={index}, actual={actual}, valid={allowed:?}, shape={shape:?}, axis={axis}, maximum={maximum}, constants={}, mode={mode}, accelerated={accelerated}, power={power:?}, diagnostics={:?}",
                    source.input_operands.is_empty(),
                    compiled.rustnn_load_diagnostics()
                );
            }
        }
    }

    #[test]
    fn int32_arg_selection_distinguishes_adjacent_large_integers_and_true_ties() {
        let values = [
            16_777_216,
            16_777_217,
            16_777_217,
            16_777_216,
            -16_777_217,
            -16_777_216,
            -16_777_216,
            -16_777_217,
            i32::MAX - 1,
            i32::MAX,
            i32::MIN,
            i32::MIN + 1,
            16_777_217,
            16_777_217,
            -16_777_217,
            -16_777_217,
            3,
            4,
            3,
            3,
        ];
        for maximum in [false, true] {
            for constant in [false, true] {
                check(
                    graph(&values, &[10, 2], 1, maximum, constant, Default::default()),
                    &values,
                    0,
                );
            }
        }
    }

    #[test]
    fn int32_arg_selection_keeps_axes_output_types_and_storage_modes() {
        let values = [
            16_777_216,
            16_777_217,
            -16_777_216,
            -16_777_217,
            i32::MAX,
            i32::MIN,
            16_777_217,
            16_777_216,
            -16_777_217,
            -16_777_216,
            i32::MIN,
            i32::MAX,
        ];
        for axis in 0..3 {
            for maximum in [false, true] {
                for keep_dimensions in [false, true] {
                    let output_data_type = if keep_dimensions {
                        MLOperandDataType::Int64
                    } else {
                        MLOperandDataType::Int32
                    };
                    let options = MLArgMinMaxOptions {
                        keep_dimensions,
                        output_data_type,
                        ..Default::default()
                    };
                    check(
                        graph(&values, &[2, 3, 2], axis, maximum, false, options),
                        &values,
                        axis as usize,
                    );
                }
            }
        }
    }

    #[test]
    fn int32_arg_selection_preserves_logical_scalar_output() {
        for maximum in [false, true] {
            for constant in [false, true] {
                for output_data_type in [MLOperandDataType::Int32, MLOperandDataType::Int64] {
                    let options = MLArgMinMaxOptions {
                        output_data_type,
                        ..Default::default()
                    };
                    check(
                        graph(
                            &[16_777_216, 16_777_217],
                            &[2],
                            0,
                            maximum,
                            constant,
                            options,
                        ),
                        &[16_777_216, 16_777_217],
                        2,
                    );
                }
            }
        }
    }

    #[test]
    #[cfg(feature = "dynamic-inputs")]
    fn int32_arg_selection_rebinds_dynamic_shapes_and_feeds_a_native_consumer() {
        use rustnn::graph::{Dimension, DynamicDimension};
        let pairs = [
            16_777_216,
            16_777_217,
            i32::MAX,
            i32::MAX - 1,
            -16_777_216,
            -16_777_217,
            i32::MIN,
            i32::MIN + 1,
        ];
        let mut source = graph(&pairs, &[4, 2], 1, true, false, Default::default());
        let rows = Dimension::Dynamic(DynamicDimension {
            name: "rows".into(),
            max_size: 4,
        });
        source.operands[0].descriptor.shape[0] = rows.clone();
        source.operands[1].descriptor.shape[0] = rows;
        let mut consumer = source.operands[1].clone();
        consumer.name = Some("as_float".into());
        consumer.descriptor.data_type = DataType::Float32;
        source.operands.push(consumer);
        source.output_operands.push(2);
        source.operations.push(Operation::Cast {
            input: 1,
            data_type: MLOperandDataType::Float32,
            options: None,
            outputs: vec![2],
        });
        for mode in 0..3 {
            let mut options = RustNNOptions::default();
            options.coreml.reuse_tensor_storage = mode != 0;
            options.coreml.output_backings = mode == 2;
            let mut context = MLContext::create(
                &MLContextOptions::new(MLPowerPreference::Default, false)
                    .with_rustnn_backend_hint(Backend::Coreml)
                    .with_rustnn_options(options),
            )
            .unwrap();
            let mut compiled = context.rustnn_build_graph(source.clone()).unwrap();
            for count in [1, 4, 2, 1] {
                let input = context
                    .create_tensor(
                        &MLTensorDescriptor::new(MLOperandDataType::Int32, vec![count, 2])
                            .to_writable(),
                    )
                    .unwrap();
                let output = context
                    .create_tensor(
                        &MLTensorDescriptor::new(MLOperandDataType::Int32, vec![count])
                            .to_readable()
                            .to_writable(),
                    )
                    .unwrap();
                let as_float = context
                    .create_tensor(
                        &MLTensorDescriptor::new(MLOperandDataType::Float32, vec![count])
                            .to_readable()
                            .to_writable(),
                    )
                    .unwrap();
                context
                    .write_tensor(&input, &pairs[..count as usize * 2])
                    .unwrap();
                context
                    .dispatch(
                        &mut compiled,
                        &MLNamedTensors::from([("source", &input)]),
                        &MLNamedTensors::from([("result", &output), ("as_float", &as_float)]),
                    )
                    .unwrap();
                let mut actual = vec![0i32; count as usize];
                let mut floats = vec![0f32; count as usize];
                context.read_tensor(&output, &mut actual).unwrap();
                context.read_tensor(&as_float, &mut floats).unwrap();
                assert_eq!(actual, [1, 0, 0, 1][..count as usize]);
                assert_eq!(
                    floats,
                    actual.iter().map(|&value| value as f32).collect::<Vec<_>>()
                );
                // An invalid output shape must fail before publishing either destination.
                let wrong = context
                    .create_tensor(
                        &MLTensorDescriptor::new(MLOperandDataType::Float32, vec![0])
                            .to_readable()
                            .to_writable(),
                    )
                    .unwrap();
                context
                    .write_tensor(&output, &vec![99i32; count as usize])
                    .unwrap();
                assert!(
                    context
                        .dispatch(
                            &mut compiled,
                            &MLNamedTensors::from([("source", &input)]),
                            &MLNamedTensors::from([("result", &output), ("as_float", &wrong)])
                        )
                        .is_err()
                );
                context.read_tensor(&output, &mut actual).unwrap();
                assert_eq!(actual, vec![99; count as usize]);
            }
        }
    }
}
