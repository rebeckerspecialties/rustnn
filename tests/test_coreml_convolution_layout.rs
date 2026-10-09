//! Layout adaptation belongs to a convolution argument, not its source operand.
use prost::Message;
use rustnn::converters::{CoremlMlProgramConverter, GraphConverter};
use rustnn::graph::{DataType, Dimension, GraphInfo, Operand, OperandDescriptor, OperandKind};
use rustnn::operators::Operation;
use rustnn::protos::coreml::specification::{Model, model};
use serde_json::json;

fn shared_argument_graph(transposed: bool) -> GraphInfo {
    let operand = |name: &str, kind, shape: &[u32]| Operand {
        name: Some(name.into()),
        kind,
        descriptor: OperandDescriptor {
            data_type: DataType::Float32,
            shape: shape
                .iter()
                .map(|&value| Dimension::Static(value))
                .collect(),
            pending_permutation: vec![],
        },
    };
    GraphInfo {
        operands: vec![
            operand("source", OperandKind::Input, &[2, 2, 2, 2]),
            operand("result", OperandKind::Output, if transposed { &[2, 3, 3, 2] } else { &[2, 1, 1, 2] }),
            // Deliberately collide with the first proposed internal view name.
            operand("source_conv_layout", OperandKind::Output, &[2, 2, 2, 2]),
        ],
        input_operands: vec![0],
        output_operands: vec![1, 2],
        operations: vec![
            Operation::from_json_attributes(
                if transposed { "convTranspose2d" } else { "conv2d" },
                &[0, 0],
                &[1],
                &json!({"inputLayout":"nhwc", "filterLayout": if transposed { "ohwi" } else { "ihwo" }}),
            )
            .unwrap(),
            Operation::Neg {
                input: 0,
                outputs: vec![2],
                options: None,
            },
        ],
        ..Default::default()
    }
}

#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
fn shared_computed_filter_graph(dynamic: bool) -> GraphInfo {
    use rustnn::graph::ConstantData;
    let shape: Vec<_> = [1, 2, 2, 2].map(Dimension::Static).to_vec();
    let operand = |name: &str, kind, shape: Vec<Dimension>, data_type| Operand {
        name: Some(name.into()),
        kind,
        descriptor: OperandDescriptor {
            data_type,
            shape,
            pending_permutation: vec![],
        },
    };
    let mut graph = GraphInfo {
        operands: vec![
            operand(
                "source",
                OperandKind::Input,
                shape.clone(),
                DataType::Float32,
            ),
            operand(
                "activation",
                OperandKind::Intermediate,
                shape.clone(),
                DataType::Float32,
            ),
            operand(
                "stored",
                OperandKind::Constant,
                [2, 1, 1, 2].map(Dimension::Static).to_vec(),
                DataType::Float16,
            ),
            operand(
                "filter",
                OperandKind::Intermediate,
                [2, 1, 1, 2].map(Dimension::Static).to_vec(),
                DataType::Float32,
            ),
            operand(
                "first",
                OperandKind::Output,
                shape.clone(),
                DataType::Float32,
            ),
            operand(
                "negated",
                OperandKind::Output,
                shape.clone(),
                DataType::Float32,
            ),
            operand("second", OperandKind::Output, shape, DataType::Float32),
        ],
        input_operands: vec![0],
        output_operands: vec![4, 5, 6],
        constant_operand_ids_to_handles: [(
            2,
            ConstantData {
                data: [1f32, 2., 3., 4.]
                    .into_iter()
                    .flat_map(|v| half::f16::from_f32(v).to_bits().to_le_bytes())
                    .collect(),
                label: None,
            },
        )]
        .into_iter()
        .collect(),
        operations: vec![
            Operation::from_json_attributes("identity", &[0], &[1], &json!({})).unwrap(),
            Operation::from_json_attributes("cast", &[2], &[3], &json!({"to":"float32"})).unwrap(),
            Operation::from_json_attributes(
                "conv2d",
                &[1, 3],
                &[4],
                &json!({"inputLayout":"nhwc","filterLayout":"ohwi"}),
            )
            .unwrap(),
            Operation::Neg {
                input: 1,
                outputs: vec![5],
                options: None,
            },
            Operation::from_json_attributes(
                "conv2d",
                &[1, 3],
                &[6],
                &json!({"inputLayout":"nhwc","filterLayout":"ihwo"}),
            )
            .unwrap(),
        ],
        ..Default::default()
    };
    if dynamic {
        for id in [0, 1, 4, 5, 6] {
            graph.operands[id].descriptor.shape[0] =
                Dimension::Dynamic(rustnn::graph::DynamicDimension {
                    name: "batch".into(),
                    max_size: 3,
                });
        }
    }
    graph
}

#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
#[test]
fn convolution_shared_computed_filter_keeps_per_consumer_layouts() {
    check_shared_filter(false, &[1]);
}

#[cfg(all(
    target_os = "macos",
    feature = "coreml-runtime",
    feature = "dynamic-inputs"
))]
#[test]
fn convolution_shared_computed_filter_retains_active_batch() {
    check_shared_filter(true, &[1, 3, 2]);
}

#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
fn check_shared_filter(dynamic: bool, batches: &[u64]) {
    use rustnn::mlcontext::{
        Backend, MLContext, MLContextOptions, MLNamedTensors, MLPowerPreference, MLTensorDescriptor,
    };
    use rustnn::operator_enums::MLOperandDataType;
    let mut context = MLContext::create(
        &MLContextOptions::new(MLPowerPreference::Default, false)
            .with_rustnn_backend_hint(Backend::Coreml),
    )
    .unwrap();
    let mut graph = context
        .rustnn_build_graph(shared_computed_filter_graph(dynamic))
        .unwrap();
    for &batch in batches {
        let descriptor = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![batch, 2, 2, 2]);
        let source = context
            .create_tensor(&descriptor.clone().to_writable())
            .unwrap();
        let outputs: Vec<_> = ["first", "negated", "second"]
            .into_iter()
            .map(|name| {
                (
                    name,
                    context
                        .create_tensor(&descriptor.clone().to_readable())
                        .unwrap(),
                )
            })
            .collect();
        let values: Vec<f32> = (1..=batch * 8).map(|v| v as f32).collect();
        context.write_tensor(&source, &values).unwrap();
        context
            .dispatch(
                &mut graph,
                &MLNamedTensors::from([("source", &source)]),
                &MLNamedTensors::from_iter(outputs.iter().map(|(name, tensor)| (*name, tensor))),
            )
            .unwrap();
        for (name, tensor) in &outputs {
            let expected: Vec<f32> = match *name {
                "first" => values
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .flat_map(|[a, b]| [a + 2. * b, 3. * a + 4. * b])
                    .collect(),
                "second" => values
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .flat_map(|[a, b]| [a + 3. * b, 2. * a + 4. * b])
                    .collect(),
                _ => values.iter().map(|v| -v).collect(),
            };
            let mut actual = vec![0f32; values.len()];
            context.read_tensor(tensor, &mut actual).unwrap();
            assert_eq!(actual, expected, "output={name},batch={batch}");
        }
    }
}

#[test]
fn convolution_same_operand_has_distinct_role_views_and_unique_names() {
    fn visit(model: &Model) {
        match model.r#type.as_ref().unwrap() {
            model::Type::Pipeline(pipeline) => pipeline.models.iter().for_each(visit),
            model::Type::MlProgram(program) => {
                for function in program.functions.values() {
                    for block in function.block_specializations.values() {
                        let mut names = std::collections::HashSet::new();
                        for value in function
                            .inputs
                            .iter()
                            .chain(block.operations.iter().flat_map(|op| &op.outputs))
                        {
                            assert!(names.insert(&value.name), "duplicate {}", value.name);
                        }
                        for op in &block.operations {
                            if op.r#type == "conv" || op.r#type == "conv_transpose" {
                                assert_ne!(op.inputs["x"], op.inputs["weight"]);
                            }
                        }
                    }
                }
            }
            other => panic!("unexpected model {other:?}"),
        }
    }
    for transposed in [false, true] {
        let converted = CoremlMlProgramConverter
            .convert(&shared_argument_graph(transposed))
            .unwrap();
        visit(&Model::decode(converted.data.as_slice()).unwrap());
    }
}

#[test]
fn convolution_layout_rejects_malformed_direct_converter_rank() {
    let mut graph = shared_argument_graph(false);
    graph.operands[0].descriptor.shape.pop();
    let error = CoremlMlProgramConverter.convert(&graph).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("convolution layout requires rank-4 operand 0"),
        "{error}"
    );
}

#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
#[test]
fn convolution_same_operand_roles_preserve_logical_fanout_values() {
    for transposed in [false, true] {
        check_shared_argument_values(transposed);
    }
}

#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
fn check_shared_argument_values(transposed: bool) {
    use rustnn::mlcontext::{
        Backend, MLContext, MLContextOptions, MLNamedTensors, MLPowerPreference, MLTensorDescriptor,
    };
    use rustnn::operator_enums::MLOperandDataType;
    let mut context = MLContext::create(
        &MLContextOptions::new(MLPowerPreference::Default, false)
            .with_rustnn_backend_hint(Backend::Coreml),
    )
    .unwrap();
    let mut graph = context
        .rustnn_build_graph(shared_argument_graph(transposed))
        .unwrap();
    let input = context
        .create_tensor(
            &MLTensorDescriptor::new(MLOperandDataType::Float32, vec![2, 2, 2, 2]).to_writable(),
        )
        .unwrap();
    let result = context
        .create_tensor(
            &MLTensorDescriptor::new(
                MLOperandDataType::Float32,
                if transposed {
                    vec![2, 3, 3, 2]
                } else {
                    vec![2, 1, 1, 2]
                },
            )
            .to_readable(),
        )
        .unwrap();
    let negated = context
        .create_tensor(
            &MLTensorDescriptor::new(MLOperandDataType::Float32, vec![2, 2, 2, 2]).to_readable(),
        )
        .unwrap();
    let values: Vec<f32> = (1..=16).map(|value| value as f32).collect();
    context.write_tensor(&input, &values).unwrap();
    context
        .dispatch(
            &mut graph,
            &MLNamedTensors::from([("source", &input)]),
            &MLNamedTensors::from([("result", &result), ("source_conv_layout", &negated)]),
        )
        .unwrap();
    let mut actual = vec![0f32; if transposed { 36 } else { 4 }];
    context.read_tensor(&result, &mut actual).unwrap();
    let mut expected = vec![];
    for batch in 0..2 {
        for output_channel in 0..2 {
            let mut sum = 0.;
            for height in 0..2 {
                for width in 0..2 {
                    for input_channel in 0..2 {
                        sum += values[((batch * 2 + height) * 2 + width) * 2 + input_channel]
                            * values
                                [((input_channel * 2 + height) * 2 + width) * 2 + output_channel];
                    }
                }
            }
            expected.push(sum);
        }
    }
    if transposed {
        expected = vec![0.; 36];
        for batch in 0..2 {
            for input_h in 0..2 {
                for input_w in 0..2 {
                    for input_channel in 0..2 {
                        for kernel_h in 0..2 {
                            for kernel_w in 0..2 {
                                for output_channel in 0..2 {
                                    expected[((batch * 3 + input_h + kernel_h) * 3
                                        + input_w
                                        + kernel_w)
                                        * 2
                                        + output_channel] += values
                                        [((batch * 2 + input_h) * 2 + input_w) * 2 + input_channel]
                                        * values[((output_channel * 2 + kernel_h) * 2 + kernel_w)
                                            * 2
                                            + input_channel];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    assert_eq!(actual, expected, "transposed={transposed}");
    let mut actual_negated = vec![0f32; 16];
    context.read_tensor(&negated, &mut actual_negated).unwrap();
    assert_eq!(
        actual_negated,
        values.iter().map(|value| -value).collect::<Vec<_>>()
    );
}
