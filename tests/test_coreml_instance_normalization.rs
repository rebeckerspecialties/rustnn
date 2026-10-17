//! NHWC normalization must preserve layout when affine operands are computed.
use prost::Message;
use rustnn::converters::{CoremlMlProgramConverter, GraphConverter};
use rustnn::graph::{
    ConstantData, DataType, Dimension, GraphInfo, Operand, OperandDescriptor, OperandKind,
};
use rustnn::operators::Operation;
use rustnn::protos::coreml::specification::{Model, model};
use serde_json::json;

fn graph(batch: Dimension, scale: bool, bias: bool) -> GraphInfo {
    let operand = |name: &str, kind, shape| Operand {
        name: Some(name.into()),
        kind,
        descriptor: OperandDescriptor {
            data_type: DataType::Float32,
            shape,
            pending_permutation: vec![],
        },
    };
    let shape = vec![
        batch,
        Dimension::Static(2),
        Dimension::Static(2),
        Dimension::Static(2),
    ];
    let mut graph = GraphInfo {
        operands: vec![operand("input", OperandKind::Input, shape.clone())],
        input_operands: vec![0],
        ..Default::default()
    };
    let mut options = json!({"layout":"nhwc","epsilon":4});
    for (enabled, name) in [(scale, "scale"), (bias, "bias")] {
        if enabled {
            let id = graph.operands.len() as u32;
            graph.operands.push(operand(
                name,
                OperandKind::Input,
                vec![Dimension::Static(2)],
            ));
            graph.input_operands.push(id);
            options[name] = json!(id);
        }
    }
    let output = graph.operands.len() as u32;
    graph
        .operands
        .push(operand("result", OperandKind::Output, shape));
    graph.output_operands.push(output);
    graph.operations.push(
        Operation::from_json_attributes("instanceNormalization", &[0], &[output], &options)
            .unwrap(),
    );
    graph
}

fn computed_affine_graph() -> GraphInfo {
    let mut result = graph(Dimension::Static(1), true, true);
    result.input_operands = vec![0];
    for (id, values) in [(1, [3f32, 6.]), (2, [1f32, 2.])] {
        result.operands[id].kind = OperandKind::Constant;
        result.operands[id].descriptor.data_type = DataType::Float16;
        result.constant_operand_ids_to_handles.insert(
            id as u32,
            ConstantData {
                data: values
                    .into_iter()
                    .flat_map(|value| half::f16::from_f32(value).to_bits().to_le_bytes())
                    .collect(),
                label: None,
            },
        );
        let widened = result.operands.len() as u32;
        let mut operand = result.operands[id].clone();
        operand.name = Some(format!("wide_parameter_{id}"));
        operand.kind = OperandKind::Intermediate;
        operand.descriptor.data_type = DataType::Float32;
        result.operands.push(operand);
        result.operations.insert(
            0,
            Operation::from_json_attributes(
                "cast",
                &[id as u32],
                &[widened],
                &json!({"to":"float32"}),
            )
            .unwrap(),
        );
        let Operation::InstanceNormalization {
            options: Some(options),
            ..
        } = result.operations.last_mut().unwrap()
        else {
            unreachable!()
        };
        if id == 1 {
            options.scale = Some(widened);
        } else {
            options.bias = Some(widened);
        }
    }
    result
}

#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
fn following_convolution_graph() -> GraphInfo {
    let mut result = graph(Dimension::Static(1), true, true);
    result.operands[3].kind = OperandKind::Intermediate;
    result.operands[3].name = Some("normalized".into());
    let filter = result.operands.len() as u32;
    result.operands.push(Operand {
        name: Some("filter".into()),
        kind: OperandKind::Constant,
        descriptor: OperandDescriptor {
            data_type: DataType::Float32,
            shape: [2, 1, 1, 2].map(Dimension::Static).to_vec(),
            pending_permutation: vec![],
        },
    });
    result.constant_operand_ids_to_handles.insert(
        filter,
        ConstantData {
            data: [1f32, 0., 0., 1.]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect(),
            label: None,
        },
    );
    let output = result.operands.len() as u32;
    let mut operand = result.operands[3].clone();
    operand.kind = OperandKind::Output;
    operand.name = Some("result".into());
    result.operands.push(operand);
    result.output_operands = vec![output];
    result.operations.push(
        Operation::from_json_attributes(
            "conv2d",
            &[3, filter],
            &[output],
            &json!({"inputLayout":"nhwc","filterLayout":"ohwi"}),
        )
        .unwrap(),
    );
    result
}

#[test]
fn instance_normalization_runtime_affine_nhwc_has_spatial_transposes() {
    for graph in [
        graph(Dimension::Static(1), true, true),
        graph(Dimension::Static(1), true, false),
        graph(Dimension::Static(1), false, true),
        computed_affine_graph(),
    ] {
        let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
        let decoded = Model::decode(converted.data.as_slice()).unwrap();
        let model::Type::MlProgram(program) = decoded.r#type.unwrap() else {
            panic!("MLProgram")
        };
        let operations = &program.functions["main"]
            .block_specializations
            .values()
            .next()
            .unwrap()
            .operations;
        assert_eq!(
            operations
                .iter()
                .filter(|op| op.r#type == "transpose")
                .count(),
            2,
            "runtime affine must not bypass NHWC layout conversion"
        );
        let norm = operations
            .iter()
            .position(|op| op.r#type == "instance_norm")
            .unwrap();
        let transposes: Vec<_> = operations
            .iter()
            .enumerate()
            .filter(|(_, op)| op.r#type == "transpose")
            .map(|(index, _)| index)
            .collect();
        assert!(transposes[0] < norm && norm < transposes[1]);
        assert!(
            operations[transposes[1] + 1..]
                .iter()
                .any(|op| op.r#type == "mul" || op.r#type == "add")
        );
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

    fn check(info: GraphInfo, scale: bool, bias: bool, batches: &[u64]) {
        let runtime_scale = info
            .input_operands
            .iter()
            .any(|&id| info.operands[id as usize].name.as_deref() == Some("scale"));
        let runtime_bias = info
            .input_operands
            .iter()
            .any(|&id| info.operands[id as usize].name.as_deref() == Some("bias"));
        let mut context = MLContext::create(
            &MLContextOptions::new(MLPowerPreference::Default, false)
                .with_rustnn_backend_hint(Backend::Coreml),
        )
        .unwrap();
        let mut compiled = MLGraphBuilder::new(&mut context)
            .unwrap()
            .build_graph_info(info)
            .unwrap();
        for &batch in batches {
            let input = context
                .create_tensor(
                    &MLTensorDescriptor::new(MLOperandDataType::Float32, vec![batch, 2, 2, 2])
                        .to_writable(),
                )
                .unwrap();
            let data: Vec<_> = (0..batch)
                .flat_map(|n| {
                    [0f32, 0., 2., 2., 4., 4., 6., 6.].map(|value| value + 32. * n as f32)
                })
                .collect();
            context.write_tensor(&input, &data).unwrap();
            let scale_tensor = context
                .create_tensor(
                    &MLTensorDescriptor::new(MLOperandDataType::Float32, vec![2]).to_writable(),
                )
                .unwrap();
            context.write_tensor(&scale_tensor, &[3f32, 6.]).unwrap();
            let bias_tensor = context
                .create_tensor(
                    &MLTensorDescriptor::new(MLOperandDataType::Float32, vec![2]).to_writable(),
                )
                .unwrap();
            context.write_tensor(&bias_tensor, &[1f32, 2.]).unwrap();
            let output = context
                .create_tensor(
                    &MLTensorDescriptor::new(MLOperandDataType::Float32, vec![batch, 2, 2, 2])
                        .to_readable(),
                )
                .unwrap();
            let mut inputs = MLNamedTensors::from([("input", &input)]);
            if runtime_scale {
                inputs.insert("scale", &scale_tensor);
            }
            if runtime_bias {
                inputs.insert("bias", &bias_tensor);
            }
            context
                .dispatch(
                    &mut compiled,
                    &inputs,
                    &MLNamedTensors::from([("result", &output)]),
                )
                .unwrap();
            let mut actual = vec![f32::NAN; (batch * 8) as usize];
            context.read_tensor(&output, &mut actual).unwrap();
            let expected: Vec<_> = [-1., -1., -1. / 3., -1. / 3., 1. / 3., 1. / 3., 1., 1.]
                .iter()
                .enumerate()
                .map(|(index, &value)| {
                    value * if scale { [3., 6.][index % 2] } else { 1. }
                        + if bias { [1., 2.][index % 2] } else { 0. }
                })
                .collect();
            for (index, (a, e)) in actual.iter().zip(expected.iter().cycle()).enumerate() {
                assert!(
                    (a - e).abs() <= 1e-6,
                    "batch={batch},scale={scale},bias={bias},index={index}: {a} != {e}"
                );
            }
        }
    }

    #[test]
    fn instance_normalization_runtime_affine_nhwc_values() {
        for (scale, bias) in [(true, true), (true, false), (false, true)] {
            check(graph(Dimension::Static(1), scale, bias), scale, bias, &[1]);
        }
        check(computed_affine_graph(), true, true, &[1]);
    }

    #[test]
    fn instance_normalization_runtime_affine_nhwc_feeds_convolution() {
        check(following_convolution_graph(), true, true, &[1]);
    }

    #[cfg(feature = "dynamic-inputs")]
    #[test]
    fn instance_normalization_runtime_affine_nhwc_dynamic_batch() {
        use rustnn::graph::DynamicDimension;
        check(
            graph(
                Dimension::Dynamic(DynamicDimension {
                    name: "batch".into(),
                    max_size: 3,
                }),
                true,
                true,
            ),
            true,
            true,
            &[1, 3, 1],
        );
    }
}
