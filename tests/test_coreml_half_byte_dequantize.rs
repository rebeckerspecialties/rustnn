//! Byte dequantization keeps source Half scales but evaluates its formula wide.
#[path = "common/half_reference.rs"]
mod half_reference;

use prost::Message;
use rustnn::converters::{CoremlMlProgramConverter, GraphConverter};
use rustnn::graph::{
    ConstantData, DataType, Dimension, GraphInfo, Operand, OperandDescriptor, OperandKind,
};
use rustnn::operators::Operation;
use rustnn::protos::coreml::{mil_spec, specification};
use serde_json::json;

fn graph(dtype: DataType, constant_parameters: bool) -> GraphInfo {
    let operand = |name: &str, kind, dtype, shape: &[u32]| Operand {
        name: Some(name.into()),
        kind,
        descriptor: OperandDescriptor {
            data_type: dtype,
            shape: shape.iter().copied().map(Dimension::Static).collect(),
            pending_permutation: vec![],
        },
    };
    let parameter = if constant_parameters {
        OperandKind::Constant
    } else {
        OperandKind::Input
    };
    GraphInfo {
        operands: vec![
            operand("input", OperandKind::Input, dtype, &[4]),
            operand("scale", parameter, DataType::Float16, &[1]),
            operand("zero_point", parameter, dtype, &[1]),
            operand("result", OperandKind::Output, DataType::Float16, &[4]),
        ],
        input_operands: if constant_parameters {
            vec![0]
        } else {
            vec![0, 1, 2]
        },
        output_operands: vec![3],
        operations: vec![
            Operation::from_json_attributes("dequantizeLinear", &[0, 1, 2], &[3], &json!({}))
                .unwrap(),
        ],
        constant_operand_ids_to_handles: if constant_parameters {
            [
                (
                    1,
                    ConstantData {
                        data: vec![1, 0],
                        label: None,
                    },
                ),
                (
                    2,
                    ConstantData {
                        data: vec![0],
                        label: None,
                    },
                ),
            ]
            .into_iter()
            .collect()
        } else {
            Default::default()
        },
        ..Default::default()
    }
}

fn operations(model: &specification::Model) -> Vec<&mil_spec::Operation> {
    let children = match model.r#type.as_ref().unwrap() {
        specification::model::Type::MlProgram(_) => vec![model],
        specification::model::Type::Pipeline(p) => p.models.iter().collect(),
        _ => panic!("MLProgram/Pipeline"),
    };
    children
        .into_iter()
        .flat_map(|child| {
            let specification::model::Type::MlProgram(program) = child.r#type.as_ref().unwrap()
            else {
                panic!("MLProgram child")
            };
            program.functions.values().flat_map(|function| {
                function
                    .block_specializations
                    .values()
                    .flat_map(|block| &block.operations)
            })
        })
        .collect()
}

fn dtype(value: &mil_spec::NamedValueType) -> i32 {
    let mil_spec::value_type::Type::TensorType(t) =
        value.r#type.as_ref().unwrap().r#type.as_ref().unwrap()
    else {
        panic!("tensor")
    };
    t.data_type
}

#[test]
fn signed_constant_formula_ingress_uses_its_exact_int32_reconstruction() {
    for constant_input in [false, true] {
        let mut source = graph(DataType::Int8, true);
        // Reserve the helper's natural signed-value name. The consumer must
        // use recorded SSA provenance, not reconstruct a suffix convention.
        source.operands[1].name = Some("zero_point_signed_constant_value".into());
        if constant_input {
            source.operands[0].kind = OperandKind::Constant;
            source.input_operands.clear();
            source.constant_operand_ids_to_handles.insert(
                0,
                ConstantData {
                    data: vec![128, 255, 0, 127],
                    label: None,
                },
            );
        }
        let original = serde_json::to_vec(&source).unwrap();
        let converted = CoremlMlProgramConverter.convert(&source).unwrap();
        assert_eq!(serde_json::to_vec(&source).unwrap(), original);
        let model = specification::Model::decode(converted.data.as_slice()).unwrap();
        let operations = operations(&model);
        for role in if constant_input {
            vec!["zp", "in"]
        } else {
            vec!["zp"]
        } {
            let kernel = operations
                .iter()
                .find(|operation| {
                    operation.r#type == "cast"
                        && operation
                            .outputs
                            .iter()
                            .any(|output| output.name == format!("result_dq_{role}_f"))
                })
                .unwrap();
            let Some(mil_spec::argument::binding::Binding::Name(input)) =
                &kernel.inputs["x"].arguments[0].binding
            else {
                panic!("named signed ingress")
            };
            let value = operations
                .iter()
                .flat_map(|operation| &operation.outputs)
                .find(|value| &value.name == input)
                .unwrap();
            assert_eq!(
                dtype(value),
                mil_spec::DataType::Int32 as i32,
                "original signed constant must not cross a redundant private Int8 feature"
            );
        }
    }
}

#[test]
fn runtime_half_scale_widening_is_a_pure_native_program() {
    for input_type in [DataType::Int8, DataType::Uint8] {
        let result = CoremlMlProgramConverter
            .convert(&graph(input_type, false))
            .unwrap();
        let model = specification::Model::decode(result.data.as_slice()).unwrap();
        let Some(specification::model::Type::Pipeline(pipeline)) = &model.r#type else {
            panic!("real precision boundaries")
        };
        let child = pipeline.models.iter().find(|child| {
            operations(child).iter().any(|op| {
                op.r#type == "cast"
                    && op.inputs["x"].arguments.iter().any(|argument| {
                        matches!(&argument.binding,
                            Some(mil_spec::argument::binding::Binding::Name(name)) if name == "scale")
                    })
                    && op.outputs.iter().any(|value| dtype(value) == mil_spec::DataType::Float32 as i32)
            })
        }).unwrap();
        assert_eq!(
            operations(child)
                .iter()
                .filter(|op| op.r#type != "const")
                .count(),
            1,
            "source Half scale must not widen in a mixed byte-cast program"
        );
    }
}

#[test]
fn byte_dequantize_uses_one_float32_formula_and_a_real_half_result() {
    for input_type in [DataType::Int8, DataType::Uint8] {
        for constant_parameters in [false, true] {
            let graph = graph(input_type, constant_parameters);
            let original = serde_json::to_vec(&graph).unwrap();
            let result = CoremlMlProgramConverter.convert(&graph).unwrap();
            assert_eq!(serde_json::to_vec(&graph).unwrap(), original);
            let model = specification::Model::decode(result.data.as_slice()).unwrap();
            let operations = operations(&model);
            assert!(!operations.iter().any(|op| op.r#type == "dequantize"));
            for name in ["sub", "mul"] {
                let kernels: Vec<_> = operations.iter().filter(|op| op.r#type == name).collect();
                assert!(
                    !kernels.is_empty(),
                    "{input_type:?} constant={constant_parameters}:{name}"
                );
                // Signed ingress reconstruction can also have an Int32 Sub.
                assert!(kernels.iter().any(|op| {
                    op.outputs
                        .iter()
                        .all(|value| dtype(value) == mil_spec::DataType::Float32 as i32)
                }));
            }
            assert!(operations.iter().any(|op| op.r#type == "cast"
                && op.outputs.iter().any(|value| value.name == "result"
                    && dtype(value) == mil_spec::DataType::Float16 as i32)));
        }
    }
}

#[test]
fn high_bit_integer_dequantize_keeps_its_existing_typed_computation() {
    let mut graph = graph(DataType::Int32, false);
    let result = CoremlMlProgramConverter.convert(&graph).unwrap();
    let model = specification::Model::decode(result.data.as_slice()).unwrap();
    assert!(operations(&model).iter().any(|op| {
        op.r#type == "mul"
            && op
                .outputs
                .iter()
                .all(|value| dtype(value) == mil_spec::DataType::Float16 as i32)
    }));
    graph.operands[0].descriptor.data_type = DataType::Uint32;
    graph.operands[2].descriptor.data_type = DataType::Uint32;
    CoremlMlProgramConverter.convert(&graph).unwrap();
}

#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
#[test]
fn byte_dequantize_preserves_all_codes_at_scale_seams_rounding_ties_and_overflow() {
    use half::f16;
    use rustnn::backend_selection::{BackendDevice, DeviceType};
    use rustnn::mlcontext::{
        MLContext, MLContextOptions, MLNamedTensors, MLPowerPreference, MLTensorDescriptor,
    };
    use rustnn::operator_enums::MLOperandDataType;

    for dtype in [DataType::Int8, DataType::Uint8] {
        for scale_bits in [1u16, 0x03ff, 0x0400, 0x3c01, 0x7bff] {
            for constant_input in [false, true] {
                for constant_parameters in [false, true] {
                    let mut source = graph(dtype, constant_parameters);
                    source.operands[0].descriptor.shape = vec![Dimension::Static(256)];
                    source.operands[3].descriptor.shape = vec![Dimension::Static(256)];
                    let bytes: Vec<u8> = (0..=255).collect();
                    if constant_input {
                        source.operands[0].kind = OperandKind::Constant;
                        source.input_operands.retain(|&id| id != 0);
                        source.constant_operand_ids_to_handles.insert(
                            0,
                            ConstantData {
                                data: bytes.clone(),
                                label: None,
                            },
                        );
                    }
                    let zero = if dtype == DataType::Int8 {
                        128u8
                    } else {
                        255u8
                    };
                    if constant_parameters {
                        source
                            .constant_operand_ids_to_handles
                            .get_mut(&1)
                            .unwrap()
                            .data = scale_bits.to_le_bytes().to_vec();
                        source
                            .constant_operand_ids_to_handles
                            .get_mut(&2)
                            .unwrap()
                            .data = vec![zero];
                    }
                    let expected: Vec<u16> = bytes
                        .iter()
                        .map(|&value| {
                            let value = if dtype == DataType::Int8 {
                                f64::from(value as i8)
                            } else {
                                f64::from(value)
                            };
                            let zero = if dtype == DataType::Int8 {
                                f64::from(zero as i8)
                            } else {
                                f64::from(zero)
                            };
                            half_reference::reference_half_bits(
                                (value - zero) * f64::from(f16::from_bits(scale_bits)),
                            )
                        })
                        .collect();
                    for policy in [DeviceType::Cpu, DeviceType::Gpu, DeviceType::Npu] {
                        let mut context = MLContext::create(
                            &MLContextOptions::new(
                                MLPowerPreference::Default,
                                policy != DeviceType::Cpu,
                            )
                            .with_rustnn_device_hint(
                                BackendDevice::Coreml {
                                    device_type: policy,
                                },
                            ),
                        )
                        .unwrap();
                        let mut compiled = context.rustnn_build_graph(source.clone()).unwrap();
                        let public_type = if dtype == DataType::Int8 {
                            MLOperandDataType::Int8
                        } else {
                            MLOperandDataType::Uint8
                        };
                        let input = context
                            .create_tensor(
                                &MLTensorDescriptor::new(public_type, vec![256]).to_writable(),
                            )
                            .unwrap();
                        let scale = context
                            .create_tensor(
                                &MLTensorDescriptor::new(MLOperandDataType::Float16, vec![1])
                                    .to_writable(),
                            )
                            .unwrap();
                        let zero_point = context
                            .create_tensor(
                                &MLTensorDescriptor::new(public_type, vec![1]).to_writable(),
                            )
                            .unwrap();
                        let output = context
                            .create_tensor(
                                &MLTensorDescriptor::new(MLOperandDataType::Float16, vec![256])
                                    .to_readable(),
                            )
                            .unwrap();
                        context.write_tensor(&input, &bytes).unwrap();
                        context.write_tensor(&scale, &[scale_bits]).unwrap();
                        context.write_tensor(&zero_point, &[zero]).unwrap();
                        let mut inputs = vec![];
                        if !constant_input {
                            inputs.push(("input", &input));
                        }
                        if !constant_parameters {
                            inputs.extend([("scale", &scale), ("zero_point", &zero_point)]);
                        }
                        context
                            .dispatch(
                                &mut compiled,
                                &MLNamedTensors::from_iter(inputs),
                                &MLNamedTensors::from([("result", &output)]),
                            )
                            .unwrap();
                        let mut actual = vec![0u16; 256];
                        context.read_tensor(&output, &mut actual).unwrap();
                        assert_eq!(
                            actual, expected,
                            "{dtype:?}/{policy:?}/input const={constant_input}/parameters const={constant_parameters}/scale=0x{scale_bits:04x}"
                        );
                    }
                }
            }
        }
    }
}

#[cfg(all(
    target_os = "macos",
    feature = "coreml-runtime",
    feature = "dynamic-inputs"
))]
#[test]
fn reused_byte_dequantize_graph_preserves_small_block_scales_and_signed_zero_points() {
    use half::f16;
    use rustnn::backend_selection::{BackendDevice, DeviceType};
    use rustnn::graph::DynamicDimension;
    use rustnn::mlcontext::{
        MLContext, MLContextOptions, MLNamedTensors, MLPowerPreference, MLTensorDescriptor,
    };
    use rustnn::operator_enums::MLOperandDataType;

    for dtype in [DataType::Int8, DataType::Uint8] {
        let public_type = if dtype == DataType::Int8 {
            MLOperandDataType::Int8
        } else {
            MLOperandDataType::Uint8
        };
        let selected: [u8; 8] = if dtype == DataType::Int8 {
            [128, 129, 255, 0, 1, 126, 127, 192]
        } else {
            [0, 1, 127, 128, 129, 254, 255, 64]
        };
        let zero_bytes = if dtype == DataType::Int8 {
            [128, 0]
        } else {
            [255, 0]
        };
        for constant_parameters in [false, true] {
            let mut source = graph(dtype, constant_parameters);
            for id in [0, 3] {
                source.operands[id].descriptor.shape = vec![
                    Dimension::Dynamic(DynamicDimension {
                        name: "batch".into(),
                        max_size: 3,
                    }),
                    Dimension::Static(4),
                ];
            }
            for id in [1, 2] {
                source.operands[id].descriptor.shape =
                    vec![Dimension::Static(1), Dimension::Static(2)];
            }
            if constant_parameters {
                source
                    .constant_operand_ids_to_handles
                    .get_mut(&1)
                    .unwrap()
                    .data = [1u16, 2].into_iter().flat_map(u16::to_le_bytes).collect();
                source
                    .constant_operand_ids_to_handles
                    .get_mut(&2)
                    .unwrap()
                    .data = zero_bytes.to_vec();
            }
            for policy in [DeviceType::Cpu, DeviceType::Gpu, DeviceType::Npu] {
                let mut context = MLContext::create(
                    &MLContextOptions::new(MLPowerPreference::Default, policy != DeviceType::Cpu)
                        .with_rustnn_device_hint(BackendDevice::Coreml {
                            device_type: policy,
                        }),
                )
                .unwrap();
                let mut compiled = context.rustnn_build_graph(source.clone()).unwrap();
                let scale = context
                    .create_tensor(
                        &MLTensorDescriptor::new(MLOperandDataType::Float16, vec![1, 2])
                            .to_writable(),
                    )
                    .unwrap();
                let zero = context
                    .create_tensor(&MLTensorDescriptor::new(public_type, vec![1, 2]).to_writable())
                    .unwrap();
                context.write_tensor(&scale, &[1u16, 2]).unwrap();
                context.write_tensor(&zero, &zero_bytes).unwrap();
                for batch in [1u64, 3, 1] {
                    let bytes: Vec<_> = (0..batch as usize * 4)
                        .map(|index| selected[index % selected.len()])
                        .collect();
                    let input = context
                        .create_tensor(
                            &MLTensorDescriptor::new(public_type, vec![batch, 4]).to_writable(),
                        )
                        .unwrap();
                    let output = context
                        .create_tensor(
                            &MLTensorDescriptor::new(MLOperandDataType::Float16, vec![batch, 4])
                                .to_readable(),
                        )
                        .unwrap();
                    context.write_tensor(&input, &bytes).unwrap();
                    let mut inputs = vec![("input", &input)];
                    if !constant_parameters {
                        inputs.extend([("scale", &scale), ("zero_point", &zero)]);
                    }
                    context
                        .dispatch(
                            &mut compiled,
                            &MLNamedTensors::from_iter(inputs),
                            &MLNamedTensors::from([("result", &output)]),
                        )
                        .unwrap();
                    let expected: Vec<_> = bytes
                        .iter()
                        .enumerate()
                        .map(|(index, &value)| {
                            let parameter = (index % 4) / 2;
                            let value = if dtype == DataType::Int8 {
                                f64::from(value as i8)
                            } else {
                                f64::from(value)
                            };
                            let zero = if dtype == DataType::Int8 {
                                f64::from(zero_bytes[parameter] as i8)
                            } else {
                                f64::from(zero_bytes[parameter])
                            };
                            let scale = f64::from(f16::from_bits([1, 2][parameter]));
                            half_reference::reference_half_bits((value - zero) * scale)
                        })
                        .collect();
                    let mut actual = vec![0u16; bytes.len()];
                    context.read_tensor(&output, &mut actual).unwrap();
                    assert_eq!(
                        actual, expected,
                        "{dtype:?}/{policy:?}/parameters const={constant_parameters}/batch={batch}"
                    );
                }
            }
        }
    }
}

fn shared_signed_graph() -> GraphInfo {
    let mut source = graph(DataType::Int8, true);
    source.operands[0].kind = OperandKind::Constant;
    source.input_operands.clear();
    source.constant_operand_ids_to_handles.insert(
        0,
        ConstantData {
            data: vec![128, 129, 0, 127],
            label: None,
        },
    );
    source
        .constant_operand_ids_to_handles
        .get_mut(&2)
        .unwrap()
        .data = vec![128];
    let mut append = |name: &str, kind, data_type, shape: Vec<Dimension>| {
        let id = source.operands.len() as u32;
        source.operands.push(Operand {
            name: Some(name.into()),
            kind,
            descriptor: OperandDescriptor {
                data_type,
                shape,
                pending_permutation: vec![],
            },
        });
        id
    };
    let wide_scale = append(
        "wide_scale",
        OperandKind::Constant,
        DataType::Float32,
        vec![Dimension::Static(1)],
    );
    let wide_result = append(
        "wide_result",
        OperandKind::Output,
        DataType::Float32,
        vec![Dimension::Static(4)],
    );
    let copied = append(
        "copied",
        OperandKind::Output,
        DataType::Int8,
        vec![Dimension::Static(4)],
    );
    let cast = append(
        "cast_result",
        OperandKind::Output,
        DataType::Float32,
        vec![Dimension::Static(4)],
    );
    source.constant_operand_ids_to_handles.insert(
        wide_scale,
        ConstantData {
            data: 0.25f32.to_le_bytes().to_vec(),
            label: None,
        },
    );
    source.output_operands.extend([wide_result, copied, cast]);
    source.operations.extend([
        Operation::from_json_attributes(
            "dequantizeLinear",
            &[0, wide_scale, 2],
            &[wide_result],
            &json!({}),
        )
        .unwrap(),
        Operation::from_json_attributes("identity", &[0], &[copied], &json!({})).unwrap(),
        Operation::from_json_attributes("cast", &[0], &[cast], &json!({"to":"float32"})).unwrap(),
    ]);
    source
}

#[test]
fn reconstructed_signed_blobs_are_not_used_as_native_constexpr_attributes() {
    let source = shared_signed_graph();
    let original = serde_json::to_vec(&source).unwrap();
    let converted = CoremlMlProgramConverter.convert(&source).unwrap();
    assert_eq!(serde_json::to_vec(&source).unwrap(), original);
    let model = specification::Model::decode(converted.data.as_slice()).unwrap();
    assert!(!operations(&model).iter().any(|operation| matches!(
        operation.r#type.as_str(),
        "dequantize" | "constexpr_affine_dequantize"
    )));
}

#[test]
fn independent_float32_constant_dequantization_keeps_its_native_signed_blob() {
    let mut source = graph(DataType::Int8, true);
    source.operands[0].kind = OperandKind::Constant;
    source.operands[1].descriptor.data_type = DataType::Float32;
    source.operands[3].descriptor.data_type = DataType::Float32;
    source.input_operands.clear();
    source.constant_operand_ids_to_handles.insert(
        0,
        ConstantData {
            data: vec![128, 129, 0, 127],
            label: None,
        },
    );
    source
        .constant_operand_ids_to_handles
        .get_mut(&1)
        .unwrap()
        .data = 0.25f32.to_le_bytes().to_vec();
    source
        .constant_operand_ids_to_handles
        .get_mut(&2)
        .unwrap()
        .data = vec![128];
    let converted = CoremlMlProgramConverter.convert(&source).unwrap();
    let model = specification::Model::decode(converted.data.as_slice()).unwrap();
    let operations = operations(&model);
    let native = operations
        .iter()
        .find(|operation| operation.r#type == "constexpr_affine_dequantize")
        .expect("independent Float32 dequantization retains native constexpr lowering");
    let mil_spec::value::Value::BlobFileValue(blob) =
        native.attributes["quantized_data"].value.as_ref().unwrap()
    else {
        panic!("quantized data is a blob")
    };
    let offset = usize::try_from(blob.offset).unwrap();
    let weights = converted.weights_data.unwrap();
    assert_eq!(&weights[offset..offset + 4], &0xdeadbeefu32.to_le_bytes());
    assert_eq!(
        &weights[offset + 4..offset + 8],
        &4u32.to_le_bytes(),
        "Int8 constexpr attribute retains its matching Int8 physical header"
    );
}

#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
#[test]
fn signed_constants_remain_valid_for_other_consumers_of_half_dequantize() {
    use rustnn::backend_selection::{BackendDevice, DeviceType};
    use rustnn::mlcontext::{
        MLContext, MLContextOptions, MLNamedTensors, MLPowerPreference, MLTensorDescriptor,
    };
    use rustnn::operator_enums::MLOperandDataType;

    let source = shared_signed_graph();
    for policy in [DeviceType::Cpu, DeviceType::Gpu, DeviceType::Npu] {
        let mut context = MLContext::create(
            &MLContextOptions::new(MLPowerPreference::Default, policy != DeviceType::Cpu)
                .with_rustnn_device_hint(BackendDevice::Coreml {
                    device_type: policy,
                }),
        )
        .unwrap();
        let mut compiled = context.rustnn_build_graph(source.clone()).unwrap();
        let mut tensor = |data_type| {
            context
                .create_tensor(&MLTensorDescriptor::new(data_type, vec![4]).to_readable())
                .unwrap()
        };
        let half = tensor(MLOperandDataType::Float16);
        let wide = tensor(MLOperandDataType::Float32);
        let copied = tensor(MLOperandDataType::Int8);
        let cast = tensor(MLOperandDataType::Float32);
        context
            .dispatch(
                &mut compiled,
                &MLNamedTensors::new(),
                &MLNamedTensors::from([
                    ("result", &half),
                    ("wide_result", &wide),
                    ("copied", &copied),
                    ("cast_result", &cast),
                ]),
            )
            .unwrap();
        let mut actual_half = [0u16; 4];
        let mut actual_wide = [0f32; 4];
        let mut actual_copied = [0i8; 4];
        let mut actual_cast = [0f32; 4];
        context.read_tensor(&half, &mut actual_half).unwrap();
        context.read_tensor(&wide, &mut actual_wide).unwrap();
        context.read_tensor(&copied, &mut actual_copied).unwrap();
        context.read_tensor(&cast, &mut actual_cast).unwrap();
        assert_eq!(actual_half, [0, 1, 128, 255], "{policy:?}:Half");
        assert_eq!(actual_wide, [0., 0.25, 32., 63.75], "{policy:?}:Float32");
        assert_eq!(actual_copied, [-128, -127, 0, 127], "{policy:?}:Identity");
        assert_eq!(actual_cast, [-128., -127., 0., 127.], "{policy:?}:Cast");
    }
}
