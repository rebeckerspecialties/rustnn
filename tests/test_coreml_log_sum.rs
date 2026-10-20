//! LogSum must not add an implementation-specific stabilizing epsilon.
use prost::Message;
use rustnn::converters::{CoremlMlProgramConverter, GraphConverter};
use rustnn::graph::{DataType, Dimension, GraphInfo, Operand, OperandDescriptor, OperandKind};
use rustnn::operators::Operation;
use rustnn::protos::coreml::specification::{Model, model};
use serde_json::json;

fn graph(
    dtype: DataType,
    input: Vec<Dimension>,
    output: Vec<Dimension>,
    options: serde_json::Value,
) -> GraphInfo {
    let operand = |name: &str, kind, shape| Operand {
        name: Some(name.into()),
        kind,
        descriptor: OperandDescriptor {
            data_type: dtype,
            shape,
            pending_permutation: vec![],
        },
    };
    GraphInfo {
        operands: vec![
            operand("input", OperandKind::Input, input),
            operand("result", OperandKind::Output, output),
        ],
        input_operands: vec![0],
        output_operands: vec![1],
        operations: vec![
            Operation::from_json_attributes("reduceLogSum", &[0], &[1], &options).unwrap(),
        ],
        ..Default::default()
    }
}

fn shape(values: &[u32]) -> Vec<Dimension> {
    values.iter().copied().map(Dimension::Static).collect()
}

#[test]
fn log_sum_uses_sum_and_zero_epsilon_log() {
    for dtype in [DataType::Float16, DataType::Float32] {
        for (input, output, options, has_sum) in [
            (vec![2, 4], vec![2], json!({"axes":[1]}), true),
            (
                vec![2, 4],
                vec![2, 1],
                json!({"axes":[1],"keepDimensions":true}),
                true,
            ),
            (vec![2, 4], vec![2, 4], json!({"axes":[]}), false),
            (vec![2, 4], vec![], json!({}), true),
            (vec![], vec![], json!({}), false),
        ] {
            let converted = CoremlMlProgramConverter
                .convert(&graph(dtype, shape(&input), shape(&output), options))
                .unwrap();
            let model = Model::decode(converted.data.as_slice()).unwrap();
            let model::Type::MlProgram(program) = model.r#type.unwrap() else {
                panic!("MLProgram")
            };
            let ops = &program.functions["main"]
                .block_specializations
                .values()
                .next()
                .unwrap()
                .operations;
            assert!(!ops.iter().any(|op| op.r#type == "reduce_log_sum"));
            assert_eq!(
                ops.iter().filter(|op| op.r#type == "reduce_sum").count(),
                usize::from(has_sum)
            );
            let log = ops
                .iter()
                .find(|op| op.r#type == "log")
                .expect("explicit log");
            use rustnn::protos::coreml::mil_spec::{
                argument::binding::Binding, tensor_value, value,
            };
            let Binding::Value(value) =
                log.inputs["epsilon"].arguments[0].binding.as_ref().unwrap()
            else {
                panic!("epsilon")
            };
            let value::Value::ImmediateValue(immediate) = value.value.as_ref().unwrap() else {
                panic!("epsilon")
            };
            let value::immediate_value::Value::Tensor(tensor) = immediate.value.as_ref().unwrap()
            else {
                panic!("epsilon")
            };
            let tensor_value::Value::Floats(values) = tensor.value.as_ref().unwrap() else {
                panic!("Float32 epsilon")
            };
            assert_eq!(values.values, [0.]);
        }
    }
}

#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
mod runtime {
    use super::*;
    use rustnn::mlcontext::{
        Backend, MLContext, MLContextOptions, MLGraphBuilder, MLNamedTensors, MLPowerPreference,
        MLTensorDescriptor,
    };
    use rustnn::operator_enums::MLOperandDataType;

    fn check(
        info: GraphInfo,
        values: &[f32],
        input_shape: &[u64],
        output_shape: &[u64],
        expected: &[f32],
    ) {
        let mut context = MLContext::create(
            &MLContextOptions::new(MLPowerPreference::Default, false)
                .with_rustnn_backend_hint(Backend::Coreml),
        )
        .unwrap();
        let mut compiled = MLGraphBuilder::new(&mut context)
            .unwrap()
            .build_graph_info(info)
            .unwrap();
        let input = context
            .create_tensor(
                &MLTensorDescriptor::new(MLOperandDataType::Float32, input_shape.to_vec())
                    .to_writable(),
            )
            .unwrap();
        context.write_tensor(&input, values).unwrap();
        let output = context
            .create_tensor(
                &MLTensorDescriptor::new(MLOperandDataType::Float32, output_shape.to_vec())
                    .to_readable(),
            )
            .unwrap();
        context
            .dispatch(
                &mut compiled,
                &MLNamedTensors::from([("input", &input)]),
                &MLNamedTensors::from([("result", &output)]),
            )
            .unwrap();
        let mut actual = vec![f32::NAN; expected.len()];
        context.read_tensor(&output, &mut actual).unwrap();
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            if expected.is_nan() {
                assert!(actual.is_nan(), "index={index}: {actual}");
            } else if expected.is_infinite() {
                assert_eq!(actual, expected, "index={index}");
            } else {
                let ordered = |value: f32| {
                    let bits = value.to_bits();
                    if bits & 0x8000_0000 == 0 {
                        bits | 0x8000_0000
                    } else {
                        !bits
                    }
                };
                assert!(
                    actual.is_finite() && ordered(actual).abs_diff(ordered(expected)) <= 22,
                    "index={index}: {actual} != {expected}"
                );
            }
        }
    }

    #[test]
    fn log_sum_tiny_positive_zero_cancelled_and_negative_values() {
        let tiny = 2f32.powi(-24);
        check(
            graph(
                DataType::Float32,
                shape(&[5, 4]),
                shape(&[5]),
                json!({"axes":[1]}),
            ),
            &[
                tiny,
                3. * tiny,
                5. * tiny,
                7. * tiny,
                0.,
                0.,
                0.,
                0.,
                1.,
                -1.,
                0.,
                0.,
                -1.,
                -2.,
                -3.,
                -4.,
                1.,
                2.,
                3.,
                4.,
            ],
            &[5, 4],
            &[5],
            &[
                2f32.powi(-20).ln(),
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NAN,
                10f32.ln(),
            ],
        );
    }

    #[test]
    fn log_sum_ordinary_cancellation_does_not_rescale_represented_values() {
        let a = 3f32;
        let b = f32::from_bits(a.to_bits() - 1);
        check(
            graph(
                DataType::Float32,
                shape(&[1, 4]),
                shape(&[1]),
                json!({"axes":[1]}),
            ),
            &[a, -b, 0., 0.],
            &[1, 4],
            &[1],
            &[(f64::from(a) - f64::from(b)).ln() as f32],
        );
    }

    #[test]
    fn log_sum_nonfinite_inputs_preserve_result_classes() {
        check(
            graph(
                DataType::Float32,
                shape(&[4, 4]),
                shape(&[4]),
                json!({"axes":[1]}),
            ),
            &[
                f32::INFINITY,
                1.,
                2.,
                3.,
                f32::NEG_INFINITY,
                1.,
                2.,
                3.,
                f32::INFINITY,
                f32::NEG_INFINITY,
                0.,
                0.,
                f32::NAN,
                1.,
                2.,
                3.,
            ],
            &[4, 4],
            &[4],
            &[f32::INFINITY, f32::NAN, f32::NAN, f32::NAN],
        );
    }

    #[test]
    fn log_sum_empty_axes_and_scalar_keep_log_semantics() {
        check(
            graph(
                DataType::Float32,
                shape(&[4]),
                shape(&[4]),
                json!({"axes":[]}),
            ),
            &[2f32.powi(-20), 0., -1., 2.],
            &[4],
            &[4],
            &[2f32.powi(-20).ln(), f32::NEG_INFINITY, f32::NAN, 2f32.ln()],
        );
        check(
            graph(DataType::Float32, vec![], vec![], json!({})),
            &[0.],
            &[],
            &[],
            &[f32::NEG_INFINITY],
        );
    }

    #[test]
    fn log_sum_half_accumulates_before_rounding_its_result() {
        let tiny = 2f32.powi(-24);
        let info = graph(
            DataType::Float16,
            shape(&[4, 4]),
            shape(&[4]),
            json!({"axes":[1]}),
        );
        let mut context = MLContext::create(
            &MLContextOptions::new(MLPowerPreference::Default, false)
                .with_rustnn_backend_hint(Backend::Coreml),
        )
        .unwrap();
        let mut compiled = MLGraphBuilder::new(&mut context)
            .unwrap()
            .build_graph_info(info)
            .unwrap();
        let input = context
            .create_tensor(
                &MLTensorDescriptor::new(MLOperandDataType::Float16, vec![4, 4]).to_writable(),
            )
            .unwrap();
        let values: Vec<_> = [
            tiny,
            3. * tiny,
            5. * tiny,
            7. * tiny,
            65504.,
            65504.,
            65504.,
            65504.,
            0.,
            0.,
            0.,
            0.,
            -1.,
            -2.,
            -3.,
            -4.,
        ]
        .into_iter()
        .map(|value| half::f16::from_f32(value).to_bits())
        .collect();
        context.write_tensor(&input, &values).unwrap();
        let output = context
            .create_tensor(
                &MLTensorDescriptor::new(MLOperandDataType::Float16, vec![4]).to_readable(),
            )
            .unwrap();
        context
            .dispatch(
                &mut compiled,
                &MLNamedTensors::from([("input", &input)]),
                &MLNamedTensors::from([("result", &output)]),
            )
            .unwrap();
        let mut actual = [0u16; 4];
        context.read_tensor(&output, &mut actual).unwrap();
        let expected = [
            2f32.powi(-20).ln(),
            (4. * 65504f32).ln(),
            f32::NEG_INFINITY,
            f32::NAN,
        ];
        for (index, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
            let actual = half::f16::from_bits(actual);
            let expected = half::f16::from_f32(expected);
            if expected.is_nan() {
                assert!(actual.is_nan(), "index={index}: {actual}");
            } else {
                assert_eq!(actual, expected, "index={index}");
            }
        }
    }

    #[cfg(feature = "dynamic-inputs")]
    #[test]
    fn log_sum_dynamic_batch_reuses_actual_shapes() {
        use rustnn::graph::DynamicDimension;
        let batch = Dimension::Dynamic(DynamicDimension {
            name: "batch".into(),
            max_size: 3,
        });
        let info = graph(
            DataType::Float32,
            vec![batch.clone(), Dimension::Static(4)],
            vec![batch],
            json!({"axes":[1]}),
        );
        let mut context = MLContext::create(
            &MLContextOptions::new(MLPowerPreference::Default, false)
                .with_rustnn_backend_hint(Backend::Coreml),
        )
        .unwrap();
        let mut compiled = MLGraphBuilder::new(&mut context)
            .unwrap()
            .build_graph_info(info)
            .unwrap();
        for batch in [1, 3, 1] {
            let input = context
                .create_tensor(
                    &MLTensorDescriptor::new(MLOperandDataType::Float32, vec![batch, 4])
                        .to_writable(),
                )
                .unwrap();
            context
                .write_tensor(&input, &[0.25f32, 0.5, 1., 2.].repeat(batch as usize))
                .unwrap();
            let output = context
                .create_tensor(
                    &MLTensorDescriptor::new(MLOperandDataType::Float32, vec![batch]).to_readable(),
                )
                .unwrap();
            context
                .dispatch(
                    &mut compiled,
                    &MLNamedTensors::from([("input", &input)]),
                    &MLNamedTensors::from([("result", &output)]),
                )
                .unwrap();
            let mut actual = vec![f32::NAN; batch as usize];
            context.read_tensor(&output, &mut actual).unwrap();
            assert!(
                actual
                    .iter()
                    .all(|value| (value - 3.75f32.ln()).abs() < 1e-6),
                "{actual:?}"
            );
        }
    }
}
