//! Narrow integer copies use supported native kernel types without changing WebNN storage.
use prost::Message;
use rustnn::converters::{CoremlMlProgramConverter, GraphConverter};
use rustnn::graph::{
    DataType, Dimension, DynamicDimension, GraphInfo, Operand, OperandDescriptor, OperandKind,
};
use rustnn::operators::Operation;
use rustnn::protos::coreml::{mil_spec, specification};

fn identity_graph(dtype: DataType, shape: Vec<Dimension>, constant: bool) -> GraphInfo {
    let descriptor = OperandDescriptor {
        data_type: dtype,
        shape: shape.clone(),
        pending_permutation: vec![],
    };
    let mut graph = GraphInfo {
        operands: vec![
            Operand {
                name: Some("input".into()),
                kind: if constant {
                    OperandKind::Constant
                } else {
                    OperandKind::Input
                },
                descriptor: descriptor.clone(),
            },
            Operand {
                name: Some("result".into()),
                kind: OperandKind::Output,
                descriptor,
            },
        ],
        input_operands: if constant { vec![] } else { vec![0] },
        output_operands: vec![1],
        operations: vec![Operation::Identity {
            input: 0,
            outputs: vec![1],
            options: None,
        }],
        ..Default::default()
    };
    if constant {
        let count = shape
            .iter()
            .map(|dim| match dim {
                Dimension::Static(size) => *size as usize,
                _ => unreachable!(),
            })
            .product::<usize>();
        graph.constant_operand_ids_to_handles.insert(
            0,
            rustnn::graph::ConstantData {
                data: (0..count.max(1))
                    .map(|index| if shape.is_empty() { 255 } else { index as u8 })
                    .collect(),
                label: None,
            },
        );
    }
    graph
}

#[test]
fn narrow_identity_helpers_do_not_shadow_graph_inputs() {
    let mut graph = identity_graph(DataType::Uint8, vec![Dimension::Static(256)], false);
    let descriptor = graph.operands[0].descriptor.clone();
    graph.operands.push(Operand {
        name: Some("result_graph_narrow_identity_int_input".into()),
        kind: OperandKind::Input,
        descriptor: descriptor.clone(),
    });
    graph.operands.push(Operand {
        name: Some("other".into()),
        kind: OperandKind::Output,
        descriptor,
    });
    graph.input_operands.push(2);
    graph.output_operands.push(3);
    graph.operations.push(Operation::Identity {
        input: 2,
        outputs: vec![3],
        options: None,
    });
    let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
    let model = specification::Model::decode(converted.data.as_slice()).unwrap();
    let specification::model::Type::MlProgram(program) = model.r#type.unwrap() else {
        panic!("expected MLProgram")
    };
    let function = &program.functions["main"];
    let block = &function.block_specializations["CoreML7"];
    let mut names = std::collections::HashSet::new();
    for value in function.inputs.iter().chain(
        block
            .operations
            .iter()
            .flat_map(|operation| &operation.outputs),
    ) {
        assert!(names.insert(&value.name), "duplicate SSA {}", value.name);
    }
}

#[test]
fn signed_constant_copy_and_cast_retain_byte_storage_with_exact_reconstruction() {
    use rustnn::operator_enums::MLOperandDataType;
    use rustnn::protos::coreml::mil_spec::{tensor_value, value, value_type};
    for scalar in [false, true] {
        for cast_dtype in [
            None,
            Some(DataType::Int8),
            Some(DataType::Int32),
            Some(DataType::Float32),
        ] {
            let mut graph = identity_graph(
                DataType::Int8,
                if scalar {
                    vec![]
                } else {
                    vec![Dimension::Static(256)]
                },
                true,
            );
            if let Some(dtype) = cast_dtype {
                graph.operands[1].descriptor.data_type = dtype;
                graph.operations[0] = Operation::Cast {
                    input: 0,
                    outputs: vec![1],
                    options: None,
                    data_type: match dtype {
                        DataType::Int8 => MLOperandDataType::Int8,
                        DataType::Int32 => MLOperandDataType::Int32,
                        DataType::Float32 => MLOperandDataType::Float32,
                        _ => unreachable!(),
                    },
                };
            }
            let original = serde_json::to_vec(&graph).unwrap();
            let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
            assert_eq!(serde_json::to_vec(&graph).unwrap(), original);
            let model = specification::Model::decode(converted.data.as_slice()).unwrap();
            let specification::model::Type::MlProgram(program) = model.r#type.unwrap() else {
                panic!("program")
            };
            let function = &program.functions["main"];
            let block = &function.block_specializations[&function.opset];
            let constant = block
                .operations
                .iter()
                .find(|op| op.r#type == "const")
                .unwrap();
            let Some(value_type::Type::TensorType(tensor)) =
                constant.outputs[0].r#type.as_ref().unwrap().r#type.as_ref()
            else {
                panic!("tensor")
            };
            assert_eq!(
                tensor.data_type,
                mil_spec::DataType::Uint8 as i32,
                "stored signed bytes use a private unsigned interpretation"
            );
            let expected = &graph.constant_operand_ids_to_handles[&0].data;
            let stored = match constant.attributes["val"].value.as_ref().unwrap() {
                value::Value::BlobFileValue(blob) => {
                    let weights = converted.weights_data.as_ref().unwrap();
                    let metadata = &weights[blob.offset as usize..blob.offset as usize + 64];
                    assert_eq!(u32::from_le_bytes(metadata[4..8].try_into().unwrap()), 3);
                    let count = u64::from_le_bytes(metadata[8..16].try_into().unwrap()) as usize;
                    let offset = u64::from_le_bytes(metadata[16..24].try_into().unwrap()) as usize;
                    assert_eq!(
                        u32::from_le_bytes(weights[..4].try_into().unwrap()),
                        1,
                        "no duplicated widened weights"
                    );
                    &weights[offset..offset + count]
                }
                value::Value::ImmediateValue(immediate) => {
                    let Some(value::immediate_value::Value::Tensor(tensor)) = &immediate.value
                    else {
                        panic!("immediate tensor")
                    };
                    let Some(tensor_value::Value::Bytes(bytes)) = &tensor.value else {
                        panic!("raw bytes")
                    };
                    bytes.values.as_ref()
                }
            };
            assert_eq!(stored, expected);
            let select = block
                .operations
                .iter()
                .find(|op| op.r#type == "select")
                .expect("signed values must be reconstructed before copying/casting");
            let name =
                |op: &mil_spec::Operation, key: &str| match &op.inputs[key].arguments[0].binding {
                    Some(mil_spec::argument::binding::Binding::Name(name)) => name.clone(),
                    _ => panic!("name"),
                };
            let signed = block
                .operations
                .iter()
                .find(|op| op.r#type == "sub")
                .unwrap();
            let high = block
                .operations
                .iter()
                .find(|op| op.r#type == "greater_equal")
                .unwrap();
            assert_eq!(name(select, "cond"), high.outputs[0].name);
            assert_eq!(name(select, "a"), signed.outputs[0].name);
            assert_eq!(name(select, "b"), name(signed, "x"));
        }
    }
}

#[test]
fn narrow_constant_same_type_cast_materializes_a_supported_int32_copy() {
    use rustnn::operator_enums::MLOperandDataType;
    for dtype in [DataType::Int8, DataType::Uint8] {
        for shape in [vec![], vec![Dimension::Static(256)]] {
            let mut graph = identity_graph(dtype, shape, true);
            graph.operations[0] = Operation::Cast {
                input: 0,
                outputs: vec![1],
                options: None,
                data_type: match dtype {
                    DataType::Int8 => MLOperandDataType::Int8,
                    DataType::Uint8 => MLOperandDataType::Uint8,
                    _ => unreachable!(),
                },
            };
            let original = serde_json::to_vec(&graph).unwrap();
            let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
            assert_eq!(serde_json::to_vec(&graph).unwrap(), original);
            let model = specification::Model::decode(converted.data.as_slice()).unwrap();
            let specification::model::Type::MlProgram(program) = model.r#type.unwrap() else {
                panic!("expected MLProgram")
            };
            let function = &program.functions["main"];
            let block = &function.block_specializations[&function.opset];
            let copy = block
                .operations
                .iter()
                .find(|operation| operation.r#type == "identity")
                .expect("same-type byte Cast must materialize its output, not alias a constant");
            let Some(mil_spec::value_type::Type::TensorType(tensor)) =
                copy.outputs[0].r#type.as_ref().unwrap().r#type.as_ref()
            else {
                panic!("copy tensor")
            };
            assert_eq!(tensor.data_type, mil_spec::DataType::Int32 as i32);
            assert!(block.operations.iter().any(|operation| {
                operation.r#type == "cast"
                    && matches!(
                        &operation.inputs["x"].arguments[0].binding,
                        Some(mil_spec::argument::binding::Binding::Name(name))
                            if name == &copy.outputs[0].name
                    )
                    && matches!(
                        operation.outputs[0].r#type.as_ref().unwrap().r#type.as_ref(),
                        Some(mil_spec::value_type::Type::TensorType(result))
                            if result.data_type == if dtype == DataType::Int8 {
                                mil_spec::DataType::Int8 as i32
                            } else {
                                mil_spec::DataType::Uint8 as i32
                            }
                    )
            }));
            assert_eq!(graph.operands[1].descriptor.data_type, dtype);
        }
    }
}

#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
#[test]
fn narrow_constant_identity_and_same_cast_keep_every_encoding_and_scalar() {
    use rustnn::backend_selection::{BackendDevice, DeviceType};
    use rustnn::mlcontext::{
        MLContext, MLContextOptions, MLNamedTensors, MLPowerPreference, MLTensorDescriptor,
    };
    use rustnn::operator_enums::MLOperandDataType;

    for dtype in [DataType::Uint8, DataType::Int8] {
        let api_dtype = if dtype == DataType::Uint8 {
            MLOperandDataType::Uint8
        } else {
            MLOperandDataType::Int8
        };
        for device in [DeviceType::Cpu, DeviceType::Gpu, DeviceType::Npu] {
            for scalar in [false, true] {
                let count = if scalar { 1 } else { 256 };
                let shape = if scalar {
                    vec![]
                } else {
                    vec![Dimension::Static(count)]
                };
                for same_cast in [false, true] {
                    let mut graph = identity_graph(dtype, shape.clone(), true);
                    if same_cast {
                        graph.operations[0] = Operation::Cast {
                            input: 0,
                            outputs: vec![1],
                            options: None,
                            data_type: api_dtype,
                        };
                    }
                    let expected = graph.constant_operand_ids_to_handles[&0].data.clone();
                    let mut context = MLContext::create(
                        &MLContextOptions::new(
                            MLPowerPreference::Default,
                            device != DeviceType::Cpu,
                        )
                        .with_rustnn_device_hint(BackendDevice::Coreml {
                            device_type: device,
                        }),
                    )
                    .unwrap();
                    let mut graph = context.rustnn_build_graph(graph).unwrap();
                    let shape = if scalar { vec![] } else { vec![count as u64] };
                    let output = context
                        .create_tensor(&MLTensorDescriptor::new(api_dtype, shape).to_readable())
                        .unwrap();
                    for _ in 0..2 {
                        context
                            .dispatch(
                                &mut graph,
                                &MLNamedTensors::new(),
                                &MLNamedTensors::from([("result", &output)]),
                            )
                            .unwrap();
                        let mut actual = vec![0; expected.len()];
                        context.read_tensor(&output, &mut actual).unwrap();
                        assert_eq!(
                            actual, expected,
                            "{dtype:?} {device:?} scalar={scalar} same_cast={same_cast}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn narrow_identity_uses_int32_kernel_and_preserves_public_dtype_and_shape() {
    for dtype in [DataType::Uint8, DataType::Int8] {
        for (shape, constant) in [
            (vec![Dimension::Static(256)], false),
            (vec![], false),
            (
                vec![Dimension::Dynamic(DynamicDimension {
                    name: "length".into(),
                    max_size: 256,
                })],
                false,
            ),
            (vec![Dimension::Static(256)], true),
            (vec![], true),
        ] {
            if !cfg!(feature = "dynamic-inputs")
                && shape
                    .iter()
                    .any(|dimension| matches!(dimension, Dimension::Dynamic(_)))
            {
                continue;
            }
            let graph = identity_graph(dtype, shape.clone(), constant);
            let source = serde_json::to_value(&graph).unwrap();
            let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
            assert_eq!(serde_json::to_value(&graph).unwrap(), source);
            let model = specification::Model::decode(converted.data.as_slice()).unwrap();
            let specification::model::Type::MlProgram(program) = model.r#type.unwrap() else {
                panic!("expected MLProgram")
            };
            let block = &program.functions["main"].block_specializations["CoreML7"];
            let kernel = block
                .operations
                .iter()
                .find(|operation| operation.r#type == "identity")
                .unwrap();
            let mil_spec::value_type::Type::TensorType(tensor) = kernel.outputs[0]
                .r#type
                .as_ref()
                .unwrap()
                .r#type
                .as_ref()
                .unwrap()
            else {
                panic!("tensor type")
            };
            assert_eq!(
                tensor.data_type,
                mil_spec::DataType::Int32 as i32,
                "{dtype:?} {shape:?} constant={constant}"
            );
            assert_eq!(tensor.rank, shape.len().max(1) as i64);
            let input = match &kernel.inputs["x"].arguments[0].binding {
                Some(mil_spec::argument::binding::Binding::Name(name)) => name,
                _ => panic!("identity input binding"),
            };
            assert!(
                block
                    .operations
                    .iter()
                    .any(|operation| operation.r#type == "cast"
                        && operation.outputs.iter().any(|output| &output.name == input))
            );
            assert_eq!(graph.operands[1].descriptor.data_type, dtype);
        }
    }
}

#[cfg(all(
    target_os = "macos",
    feature = "coreml-runtime",
    feature = "dynamic-inputs"
))]
#[test]
fn narrow_identity_keeps_every_encoding_scalar_and_resized_tensors() {
    use rustnn::backend_selection::{BackendDevice, DeviceType};
    use rustnn::mlcontext::{
        MLContext, MLContextOptions, MLNamedTensors, MLPowerPreference, MLTensorDescriptor,
    };
    use rustnn::operator_enums::MLOperandDataType;

    for dtype in [DataType::Uint8, DataType::Int8] {
        let api_dtype = if dtype == DataType::Uint8 {
            MLOperandDataType::Uint8
        } else {
            MLOperandDataType::Int8
        };
        for device in [DeviceType::Cpu, DeviceType::Gpu, DeviceType::Npu] {
            for scalar in [false, true] {
                let shape = if scalar {
                    vec![]
                } else {
                    vec![Dimension::Dynamic(DynamicDimension {
                        name: "length".into(),
                        max_size: 256,
                    })]
                };
                let mut context = MLContext::create(
                    &MLContextOptions::new(MLPowerPreference::Default, device != DeviceType::Cpu)
                        .with_rustnn_device_hint(BackendDevice::Coreml {
                            device_type: device,
                        }),
                )
                .unwrap();
                let mut graph = context
                    .rustnn_build_graph(identity_graph(dtype, shape, false))
                    .unwrap();
                for length in if scalar {
                    &[1][..]
                } else {
                    &[1, 256, 3, 1][..]
                } {
                    let shape = if scalar { vec![] } else { vec![*length as u64] };
                    let descriptor = MLTensorDescriptor::new(api_dtype, shape);
                    let input = context
                        .create_tensor(&descriptor.clone().to_writable())
                        .unwrap();
                    let output = context.create_tensor(&descriptor.to_readable()).unwrap();
                    let expected = (0..*length)
                        .map(|index| if scalar { 255 } else { index as u8 })
                        .collect::<Vec<_>>();
                    context.write_tensor(&input, &expected).unwrap();
                    for _ in 0..2 {
                        context
                            .dispatch(
                                &mut graph,
                                &MLNamedTensors::from([("input", &input)]),
                                &MLNamedTensors::from([("result", &output)]),
                            )
                            .unwrap();
                        let mut actual = vec![0; *length];
                        context.read_tensor(&output, &mut actual).unwrap();
                        assert_eq!(
                            actual, expected,
                            "{dtype:?} {device:?} scalar={scalar} count={length}"
                        );
                    }
                }
            }
        }
    }
}
