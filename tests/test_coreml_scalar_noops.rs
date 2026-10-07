//! Rank-zero transpose/slice preserve values and WebNN descriptors.

use prost::Message;
use rustnn::converters::{CoremlMlProgramConverter, GraphConverter};
use rustnn::graph::{ConstantData, DataType, GraphInfo, Operand, OperandDescriptor, OperandKind};
use rustnn::operator_options::MLTransposeOptions;
use rustnn::operators::Operation;
use rustnn::protos::coreml::{
    mil_spec::{self, dimension, value_type},
    specification::{
        ArrayFeatureType, FeatureDescription, Model, array_feature_type, feature_type, model,
    },
};

#[derive(Clone, Copy, Debug)]
enum ScalarNoop {
    TransposeDefault,
    TransposeEmpty,
    Slice,
    TransposeThenSlice,
}

const NOOPS: [ScalarNoop; 4] = [
    ScalarNoop::TransposeDefault,
    ScalarNoop::TransposeEmpty,
    ScalarNoop::Slice,
    ScalarNoop::TransposeThenSlice,
];

fn graph(dtype: DataType, constant: bool, noop: ScalarNoop) -> GraphInfo {
    let descriptor = OperandDescriptor {
        data_type: dtype,
        shape: vec![],
        pending_permutation: vec![],
    };
    let operand = |name: &str, kind| Operand {
        name: Some(name.into()),
        kind,
        descriptor: descriptor.clone(),
    };
    let transpose = |options| Operation::Transpose {
        input: 0,
        options,
        outputs: vec![1],
    };
    let slice = |input, output| Operation::Slice {
        input,
        starts: vec![],
        sizes: vec![],
        options: None,
        outputs: vec![output],
    };
    let mut graph = GraphInfo {
        operands: vec![
            operand(
                "input",
                if constant {
                    OperandKind::Constant
                } else {
                    OperandKind::Input
                },
            ),
            operand("result", OperandKind::Output),
        ],
        input_operands: if constant { vec![] } else { vec![0] },
        output_operands: vec![1],
        operations: vec![match noop {
            ScalarNoop::TransposeDefault | ScalarNoop::TransposeThenSlice => transpose(None),
            ScalarNoop::TransposeEmpty => transpose(Some(MLTransposeOptions::default())),
            ScalarNoop::Slice => slice(0, 1),
        }],
        ..Default::default()
    };
    if matches!(noop, ScalarNoop::TransposeThenSlice) {
        graph.operands[1] = operand("intermediate", OperandKind::Intermediate);
        graph.operands.push(operand("result", OperandKind::Output));
        graph.operations.push(slice(1, 2));
        graph.output_operands = vec![2];
    }
    if constant {
        graph.constant_operand_ids_to_handles.insert(
            0,
            ConstantData {
                data: match dtype {
                    DataType::Int32 => 16777217i32.to_le_bytes().to_vec(),
                    DataType::Float32 => (-1.5f32).to_le_bytes().to_vec(),
                    DataType::Float16 => 0xbe00u16.to_le_bytes().to_vec(),
                    _ => unreachable!(),
                },
                label: None,
            },
        );
    }
    graph
}

fn feature_array_type(feature: &FeatureDescription) -> &ArrayFeatureType {
    let Some(feature_type::Type::MultiArrayType(array)) = feature
        .r#type
        .as_ref()
        .and_then(|feature_type| feature_type.r#type.as_ref())
    else {
        panic!("expected multi-array feature {}", feature.name);
    };
    array
}

fn check_lowering(dtype: DataType, expected_mil_type: mil_spec::DataType) {
    for constant in [false, true] {
        for noop in NOOPS {
            let graph = graph(dtype, constant, noop);
            let source = serde_json::to_vec(&graph).unwrap();
            let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
            assert_eq!(serde_json::to_vec(&graph).unwrap(), source);
            assert!(graph.operands.iter().all(|o| o.descriptor.shape.is_empty()));
            let model = Model::decode(converted.data.as_slice()).unwrap();
            let description = model.description.unwrap();
            let output_type = feature_array_type(&description.output[0]);
            assert_eq!(output_type.shape, [1]);
            if dtype == DataType::Int32 {
                assert_eq!(
                    output_type.data_type,
                    array_feature_type::ArrayDataType::Int32 as i32
                );
            }
            if !constant {
                assert_eq!(feature_array_type(&description.input[0]).shape, [1]);
            }
            let Some(model::Type::MlProgram(program)) = model.r#type else {
                panic!("expected MLProgram");
            };
            let block = &program.functions["main"].block_specializations["CoreML7"];
            assert!(
                block
                    .operations
                    .iter()
                    .all(|op| matches!(op.r#type.as_str(), "const" | "reshape")),
                "{dtype:?} {noop:?} constant={constant}: {:?}",
                block
                    .operations
                    .iter()
                    .map(|op| &op.r#type)
                    .collect::<Vec<_>>()
            );
            let reshapes: Vec<_> = block
                .operations
                .iter()
                .filter(|op| op.r#type == "reshape")
                .collect();
            assert_eq!(reshapes.len(), graph.operations.len());
            for reshape in reshapes {
                let Some(value_type::Type::TensorType(tensor)) = reshape.outputs[0]
                    .r#type
                    .as_ref()
                    .and_then(|ty| ty.r#type.as_ref())
                else {
                    panic!("expected tensor type");
                };
                assert_eq!(tensor.data_type, expected_mil_type as i32);
                assert_eq!(tensor.rank, 1);
                assert!(matches!(tensor.dimensions[0].dimension.as_ref(),
                    Some(dimension::Dimension::Constant(size)) if size.size == 1));
            }
        }
    }
}

#[test]
fn scalar_int32_transpose_and_slice_use_rank_one_noop_lowering() {
    check_lowering(DataType::Int32, mil_spec::DataType::Int32);
}

#[test]
fn scalar_float_transpose_and_slice_keep_existing_noop_lowering() {
    check_lowering(DataType::Float32, mil_spec::DataType::Float32);
    check_lowering(DataType::Float16, mil_spec::DataType::Float16);
}

#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
mod runtime {
    use super::{NOOPS, ScalarNoop};
    use rustnn::mlcontext::{
        Backend, MLContext, MLContextOptions, MLNamedOperands, MLNamedTensors, MLOperandDescriptor,
        MLPowerPreference, MLTensorDescriptor,
    };
    use rustnn::mlgraphbuilder::MLGraphBuilder;
    use rustnn::operator_enums::MLOperandDataType;
    use rustnn::operator_options::MLTransposeOptions;

    fn check<T: bytemuck::Pod>(dtype: MLOperandDataType, values: &[T]) {
        for noop in NOOPS {
            for constant in [false, true] {
                let mut context = MLContext::create(
                    &MLContextOptions::new(MLPowerPreference::Default, false)
                        .with_rustnn_backend_hint(Backend::Coreml),
                )
                .unwrap();
                let builds = if constant { values } else { &values[..1] };
                for &initial in builds {
                    let mut builder = MLGraphBuilder::new(&mut context).unwrap();
                    let descriptor = MLOperandDescriptor::new(dtype, vec![]);
                    let input = if constant {
                        builder
                            .constant_from_slice(&descriptor, &[initial])
                            .unwrap()
                    } else {
                        builder.input("input", &descriptor).unwrap()
                    };
                    let result = match noop {
                        ScalarNoop::TransposeDefault => builder.transpose(input).unwrap(),
                        ScalarNoop::TransposeEmpty => builder
                            .transpose_with_options(input, MLTransposeOptions::default())
                            .unwrap(),
                        ScalarNoop::Slice => builder.slice(input, &[], &[]).unwrap(),
                        ScalarNoop::TransposeThenSlice => {
                            let stage = builder.transpose(input).unwrap();
                            builder.slice(stage, &[], &[]).unwrap()
                        }
                    };
                    let mut graph = builder
                        .build(&MLNamedOperands::from([("result", result)]))
                        .unwrap_or_else(|error| {
                            panic!("{dtype:?} {noop:?} constant={constant}: {error:?}")
                        });
                    assert!(graph.output_descriptors["result"].shape.is_empty());
                    let input_tensor = (!constant).then(|| {
                        context
                            .create_tensor(
                                &MLTensorDescriptor::new(dtype, vec![])
                                    .to_writable()
                                    .to_readable(),
                            )
                            .unwrap()
                    });
                    let output = context
                        .create_tensor(&MLTensorDescriptor::new(dtype, vec![]).to_readable())
                        .unwrap();
                    let runs = if constant {
                        std::slice::from_ref(&initial)
                    } else {
                        values
                    };
                    for &value in runs {
                        let mut inputs = MLNamedTensors::new();
                        if let Some(input) = &input_tensor {
                            context.write_tensor(input, &[value]).unwrap();
                            let mut written = vec![0u8; std::mem::size_of::<T>()];
                            context.read_tensor(input, &mut written).unwrap();
                            assert_eq!(written, bytemuck::bytes_of(&value), "exact input storage");
                            inputs.insert("input", input);
                        }
                        context
                            .dispatch(
                                &mut graph,
                                &inputs,
                                &MLNamedTensors::from([("result", &output)]),
                            )
                            .unwrap();
                        let mut actual = vec![0u8; std::mem::size_of::<T>()];
                        context.read_tensor(&output, &mut actual).unwrap();
                        assert!(output.shape().is_empty());
                        assert_eq!(
                            actual,
                            bytemuck::bytes_of(&value),
                            "{dtype:?} {noop:?} constant={constant}; exact typed bytes"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn scalar_int32_noops_preserve_exact_typed_values() {
        // No value passes through f32: these detect accidental arithmetic/casts.
        check(
            MLOperandDataType::Int32,
            &[0i32, -1, 16777217, -16777217, i32::MIN, i32::MAX],
        );
    }

    #[test]
    fn scalar_float_noops_preserve_exact_typed_controls() {
        check(MLOperandDataType::Float32, &[0f32, -0., -1.5, 32.]);
        check(MLOperandDataType::Float16, &[0u16, 0x8000, 0xbe00, 0x5000]);
    }
}
