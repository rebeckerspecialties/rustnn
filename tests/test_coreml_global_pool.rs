//! Global pooling aliases preserve the caller's channel/spatial layout.
//!
//! These RustNN aliases are not additional WebNN standard operators.

use prost::Message;
use rustnn::converters::{CoremlMlProgramConverter, GraphConverter};
use rustnn::graph::GraphInfo;
use rustnn::mlcontext::{MLNamedOperands, MLOperandDescriptor};
use rustnn::mlgraphbuilder::MLGraphBuilder;
use rustnn::operator_enums::MLOperandDataType;
use rustnn::operator_options::MLPool2dOptions;
use rustnn::protos::coreml::specification::mil_spec::{Value, tensor_value, value};
use rustnn::protos::coreml::specification::{Model, model};

fn graph(layout: Option<&str>, maximum: bool, dtype: MLOperandDataType) -> GraphInfo {
    let nhwc = layout.is_some_and(|layout| layout.eq_ignore_ascii_case("nhwc"));
    let mut builder = MLGraphBuilder::new_uncompiled();
    let input = builder
        .input(
            "input",
            &MLOperandDescriptor::new(
                dtype,
                if nhwc {
                    vec![2, 2, 3, 2]
                } else {
                    vec![2, 2, 2, 3]
                },
            ),
        )
        .unwrap();
    let options = layout.map(|layout| MLPool2dOptions {
        layout: layout.into(),
        ..Default::default()
    });
    let output = match (maximum, options) {
        (true, Some(options)) => builder.global_max_pool_with_options(input, options),
        (true, None) => builder.global_max_pool(input),
        (false, Some(options)) => builder.global_average_pool_with_options(input, options),
        (false, None) => builder.global_average_pool(input),
    }
    .unwrap();
    builder
        .finish_graph_info(&MLNamedOperands::from([("result", output)]))
        .unwrap()
}

fn ints(value: &Value) -> &[i32] {
    let Some(value::Value::ImmediateValue(immediate)) = &value.value else {
        panic!("axes must be an immediate value")
    };
    let Some(value::immediate_value::Value::Tensor(tensor)) = &immediate.value else {
        panic!("axes must be a tensor")
    };
    let Some(tensor_value::Value::Ints(values)) = &tensor.value else {
        panic!("axes must be integers")
    };
    &values.values
}

#[test]
fn global_pool_aliases_reduce_only_the_layout_spatial_axes() {
    for layout in [None, Some("nchw"), Some("nhwc"), Some("NHWC")] {
        for maximum in [false, true] {
            for dtype in [MLOperandDataType::Float16, MLOperandDataType::Float32] {
                let graph = graph(layout, maximum, dtype);
                let before = serde_json::to_value(&graph).unwrap();
                let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
                assert_eq!(serde_json::to_value(&graph).unwrap(), before);
                let model = Model::decode(converted.data.as_slice()).unwrap();
                let Some(model::Type::MlProgram(program)) = model.r#type else {
                    panic!("global pooling is an MLProgram")
                };
                let function = &program.functions["main"];
                let operation = function.block_specializations[&function.opset]
                    .operations
                    .iter()
                    .find(|operation| {
                        operation.r#type == if maximum { "reduce_max" } else { "reduce_mean" }
                    })
                    .unwrap();
                let binding = &operation.inputs["axes"].arguments[0].binding;
                let Some(
                    rustnn::protos::coreml::specification::mil_spec::argument::binding::Binding::Value(
                        value,
                    ),
                ) = binding
                else {
                    panic!("axes must be a value binding")
                };
                assert_eq!(
                    ints(value),
                    if layout.is_some_and(|layout| layout.eq_ignore_ascii_case("nhwc")) {
                        &[1, 2]
                    } else {
                        &[2, 3]
                    },
                    "layout={layout:?}, maximum={maximum}, dtype={dtype:?}"
                );
            }
        }
    }
}

#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
#[test]
fn global_pool_aliases_keep_each_batch_and_channel_numerically_distinct() {
    use rustnn::executors::coreml::{CoremlInput, run_coreml_with_inputs_checked};
    use std::collections::HashMap;

    for layout in ["nchw", "nhwc"] {
        for maximum in [false, true] {
            let graph = graph(Some(layout), maximum, MLOperandDataType::Float32);
            let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
            let shape = if layout == "nhwc" {
                vec![2, 2, 3, 2]
            } else {
                vec![2, 2, 2, 3]
            };
            let mut data = Vec::new();
            if layout == "nhwc" {
                for batch in 0..2 {
                    for spatial in 0..6 {
                        for channel in 0..2 {
                            data.push((100 * batch + 10 * channel + spatial) as f32);
                        }
                    }
                }
            } else {
                for batch in 0..2 {
                    for channel in 0..2 {
                        for spatial in 0..6 {
                            data.push((100 * batch + 10 * channel + spatial) as f32);
                        }
                    }
                }
            }
            let inputs = vec![CoremlInput {
                name: "input".into(),
                shape,
                data,
            }];
            let attempts = run_coreml_with_inputs_checked(
                &converted.data,
                inputs,
                &HashMap::from([("input".into(), graph.operands[0].descriptor.clone())]),
                &HashMap::from([(
                    "result".into(),
                    graph.operands[graph.output_operands[0] as usize]
                        .descriptor
                        .clone(),
                )]),
            )
            .unwrap();
            let expected = if maximum {
                [5.0, 15.0, 105.0, 115.0]
            } else {
                [2.5, 12.5, 102.5, 112.5]
            };
            for attempt in attempts {
                let outputs = attempt.result.unwrap_or_else(|error| {
                    panic!(
                        "{layout}, maximum={maximum}, {}: {error}",
                        attempt.compute_unit
                    )
                });
                assert_eq!(outputs.len(), 1);
                assert_eq!(
                    outputs[0].shape,
                    if layout == "nhwc" {
                        vec![2, 1, 1, 2]
                    } else {
                        vec![2, 2, 1, 1]
                    }
                );
                assert_eq!(
                    outputs[0].data, expected,
                    "{layout}, maximum={maximum}, {}",
                    attempt.compute_unit
                );
            }
        }
    }
}
