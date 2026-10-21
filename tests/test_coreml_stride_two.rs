//! Parity-slice lowering and exact native results for every eligible rank/axis.
use prost::Message;
use rustnn::converters::{CoremlMlProgramConverter, GraphConverter};
use rustnn::graph::{
    DataType, GraphInfo, Operand, OperandDescriptor, OperandKind, to_dimension_vector,
};
use rustnn::operators::Operation;
use rustnn::protos::coreml::{
    mil_spec::Block,
    specification::{Model, model},
};

fn graph(shape: &[u32], axis: usize, parity: u32, dtype: DataType) -> GraphInfo {
    let mut begin = vec![0; shape.len()];
    begin[axis] = parity;
    let mut sizes = shape.to_vec();
    sizes[axis] -= parity;
    let mut strides = vec![1; shape.len()];
    strides[axis] = 2;
    let mut output_shape = shape.to_vec();
    output_shape[axis] /= 2;
    let operand = |name: &str, kind, shape: &[u32]| Operand {
        name: Some(name.into()),
        kind,
        descriptor: OperandDescriptor {
            data_type: dtype,
            shape: to_dimension_vector(shape),
            pending_permutation: vec![],
        },
    };
    GraphInfo {
        input_operands: vec![0],
        output_operands: vec![1],
        operands: vec![
            operand("input", OperandKind::Input, shape),
            operand("result", OperandKind::Output, &output_shape),
        ],
        operations: vec![
            Operation::from_json_attributes(
                "slice",
                &[0],
                &[1],
                &serde_json::json!({
                    "starts": begin, "sizes": sizes, "strides": strides,
                }),
            )
            .unwrap(),
        ],
        ..Default::default()
    }
}

fn blocks(data: &[u8]) -> Vec<Block> {
    fn collect(model: Model, blocks: &mut Vec<Block>) {
        match model.r#type {
            Some(model::Type::MlProgram(program)) => {
                let function = &program.functions["main"];
                blocks.push(function.block_specializations[&function.opset].clone());
            }
            Some(model::Type::Pipeline(pipeline)) => {
                for child in pipeline.models {
                    collect(child, blocks);
                }
            }
            _ => panic!("expected MLProgram or native pipeline"),
        }
    }
    let mut result = Vec::new();
    collect(Model::decode(data).unwrap(), &mut result);
    assert!(!result.is_empty());
    result
}

#[test]
fn coreml_stride_two_conversion_uses_only_unit_stride_layouts() {
    for shape in [vec![8], vec![2, 8], vec![2, 4, 6], vec![2, 4, 2, 6]] {
        for axis in 0..shape.len() {
            for parity in 0..2 {
                for dtype in [DataType::Float16, DataType::Float32] {
                    let converted = CoremlMlProgramConverter
                        .convert(&graph(&shape, axis, parity, dtype))
                        .unwrap();
                    let blocks = blocks(&converted.data);
                    let kinds = blocks
                        .iter()
                        .flat_map(|block| &block.operations)
                        .map(|op| op.r#type.as_str())
                        .collect::<Vec<_>>();
                    assert!(
                        !kinds.contains(&"slice_by_index"),
                        "{shape:?}, {axis}, {parity}, {dtype:?}"
                    );
                    assert!(blocks.iter().any(|block| {
                        block.operations.windows(3).any(|ops| {
                            [
                                ops[0].r#type.as_str(),
                                ops[1].r#type.as_str(),
                                ops[2].r#type.as_str(),
                            ] == ["reshape", "slice_by_size", "reshape"]
                        })
                    }));
                    assert!(blocks.iter().any(|block| block.outputs == ["result"]));
                    assert!(converted.weights_data.is_none());
                }
            }
        }
    }
}

#[test]
fn coreml_stride_two_integer_and_partial_slices_keep_original_lowering() {
    for (dtype, partial) in [(DataType::Int32, false), (DataType::Float32, true)] {
        let mut info = graph(&[8], 0, 0, dtype);
        if partial {
            let Operation::Slice { sizes, .. } = &mut info.operations[0] else {
                unreachable!()
            };
            // [0:7:2] and [0:8:2] select the same four elements, but partial
            // slices deliberately remain outside this conservative matcher.
            sizes[0] = rustnn::operator_options::MLDimension::Static(7);
        }
        let converted = CoremlMlProgramConverter.convert(&info).unwrap();
        assert!(
            blocks(&converted.data)
                .iter()
                .flat_map(|block| &block.operations)
                .any(|op| op.r#type == "slice_by_index")
        );
    }
}

#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
#[test]
fn coreml_stride_two_native_results_are_exact() {
    use rustnn::executors::coreml::{CoremlInput, run_coreml_with_inputs_checked};
    use std::collections::HashMap;
    for shape in [vec![8], vec![2, 8], vec![2, 4, 6], vec![2, 4, 2, 6]] {
        for axis in 0..shape.len() {
            for parity in 0..2 {
                for dtype in [DataType::Float16, DataType::Float32] {
                    let graph = graph(&shape, axis, parity, dtype);
                    let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
                    // All values are exactly representable in each dtype; include
                    // alternating signs and zeros without adding arithmetic.
                    let data = (0..shape.iter().product::<u32>())
                        .map(|i| (i as i32 - 39) as f32)
                        .collect::<Vec<_>>();
                    let trailing = shape[axis + 1..].iter().product::<u32>();
                    let expected = data
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| (*i as u32 / trailing) % shape[axis] % 2 == parity)
                        .map(|(_, value)| value.to_bits())
                        .collect::<Vec<_>>();
                    let attempts = run_coreml_with_inputs_checked(
                        &converted.data,
                        vec![CoremlInput {
                            name: "input".into(),
                            shape: shape.iter().map(|&d| d as usize).collect(),
                            data,
                        }],
                        &HashMap::from([("input".into(), graph.operands[0].descriptor.clone())]),
                        &HashMap::from([("result".into(), graph.operands[1].descriptor.clone())]),
                    )
                    .unwrap();
                    assert_eq!(
                        attempts.iter().map(|a| a.compute_unit).collect::<Vec<_>>(),
                        ["CPU_AND_NE", "ALL", "CPU_ONLY"]
                    );
                    for attempt in attempts {
                        let outputs = attempt
                            .result
                            .unwrap_or_else(|e| panic!("{}: {e}", attempt.compute_unit));
                        let output = outputs.iter().find(|o| o.name == "result").unwrap();
                        assert_eq!(
                            output.data.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            expected,
                            "{shape:?}, axis={axis}, parity={parity}, {dtype:?}, {}",
                            attempt.compute_unit
                        );
                    }
                    check_typed_gpu(&graph, &shape, axis, parity);
                }
            }
        }
    }
}

#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
fn check_typed_gpu(info: &GraphInfo, shape: &[u32], axis: usize, parity: u32) {
    use rustnn::backend_selection::{BackendDevice, DeviceType};
    use rustnn::mlcontext::{
        LoadDiagnostics, MLContext, MLContextOptions, MLNamedTensors, MLPowerPreference,
        MLTensorDescriptor,
    };
    use rustnn::mlgraphbuilder::MLGraphBuilder;

    let mut context = MLContext::create(
        &MLContextOptions::new(MLPowerPreference::Default, true).with_rustnn_device_hint(
            BackendDevice::Coreml {
                device_type: DeviceType::Gpu,
            },
        ),
    )
    .unwrap();
    let mut graph = MLGraphBuilder::new(&mut context)
        .unwrap()
        .build_graph_info(info.clone())
        .unwrap();
    let Some(LoadDiagnostics::Coreml(diagnostics)) = graph.rustnn_load_diagnostics() else {
        panic!("missing load diagnostics")
    };
    assert_eq!(
        diagnostics.loaded_compute_units, "CPU_AND_GPU",
        "{diagnostics:?}"
    );
    assert!(diagnostics.failures.is_empty(), "{diagnostics:?}");
    let input_desc = &info.operands[0].descriptor;
    let output_desc = &info.operands[1].descriptor;
    let values = (0..shape.iter().product::<u32>())
        .map(|i| match input_desc.data_type {
            DataType::Float16 => half::f16::from_f32(i as f32 - 39.).to_le_bytes().to_vec(),
            DataType::Float32 => (i as f32 - 39.).to_le_bytes().to_vec(),
            _ => unreachable!(),
        })
        .collect::<Vec<_>>();
    let data = values.iter().flatten().copied().collect::<Vec<_>>();
    let trailing = shape[axis + 1..].iter().product::<u32>();
    let expected = values
        .iter()
        .enumerate()
        .filter(|(i, _)| (*i as u32 / trailing) % shape[axis] % 2 == parity)
        .flat_map(|(_, value)| value.iter().copied())
        .collect::<Vec<_>>();
    let tensor_descriptor = |descriptor: &OperandDescriptor| {
        MLTensorDescriptor::new(
            descriptor.data_type.try_into().unwrap(),
            descriptor
                .static_or_max_shape()
                .into_iter()
                .map(u64::from)
                .collect(),
        )
    };
    let input = context
        .create_tensor(&tensor_descriptor(input_desc).to_writable())
        .unwrap();
    let output = context
        .create_tensor(&tensor_descriptor(output_desc).to_readable())
        .unwrap();
    context.write_tensor(&input, &data).unwrap();
    context
        .dispatch(
            &mut graph,
            &MLNamedTensors::from([("input", &input)]),
            &MLNamedTensors::from([("result", &output)]),
        )
        .unwrap();
    let mut actual = vec![0u8; expected.len()];
    context.read_tensor(&output, &mut actual).unwrap();
    assert_eq!(
        actual, expected,
        "typed GPU {shape:?}, axis={axis}, parity={parity}, {:?}",
        input_desc.data_type
    );
}
