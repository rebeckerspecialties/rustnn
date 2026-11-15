//! Nearest resampling must use WebNN half-pixel coordinates and lower-index ties.
use prost::Message;
use rustnn::converters::{CoremlMlProgramConverter, GraphConverter};
use rustnn::graph::{DataType, Dimension, GraphInfo, Operand, OperandDescriptor, OperandKind};
use rustnn::operators::Operation;
use rustnn::protos::coreml::mil_spec::{Argument, argument::binding::Binding, tensor_value, value};
use rustnn::protos::coreml::specification::{Model, model};
use serde_json::json;

fn graph(
    dtype: DataType,
    input: Vec<Dimension>,
    output: Vec<Dimension>,
    axes: [u32; 2],
    sizes: [u32; 2],
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
            Operation::from_json_attributes(
                "resample2d",
                &[0],
                &[1],
                &json!({"axes":axes,"sizes":sizes,"mode":"nearest-neighbor"}),
            )
            .unwrap(),
        ],
        ..Default::default()
    }
}

fn static_shape(shape: &[u32]) -> Vec<Dimension> {
    shape.iter().copied().map(Dimension::Static).collect()
}

fn immediate_ints(argument: &Argument) -> &[i32] {
    let Binding::Value(value) = argument.arguments[0].binding.as_ref().unwrap() else {
        panic!("immediate binding")
    };
    let value::Value::ImmediateValue(immediate) = value.value.as_ref().unwrap() else {
        panic!("immediate value")
    };
    let value::immediate_value::Value::Tensor(tensor) = immediate.value.as_ref().unwrap() else {
        panic!("tensor")
    };
    let tensor_value::Value::Ints(values) = tensor.value.as_ref().unwrap() else {
        panic!("integer values")
    };
    &values.values
}

#[test]
fn nearest_resample_noninteger_scales_emit_exact_axis_indices() {
    for dtype in [DataType::Float16, DataType::Float32] {
        let graph = graph(
            dtype,
            static_shape(&[1, 2, 3, 2]),
            static_shape(&[1, 3, 4, 2]),
            [1, 2],
            [3, 4],
        );
        let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
        let model = Model::decode(converted.data.as_slice()).unwrap();
        let model::Type::MlProgram(program) = model.r#type.unwrap() else {
            panic!("MLProgram")
        };
        let operations = &program.functions["main"]
            .block_specializations
            .values()
            .next()
            .unwrap()
            .operations;
        let gathers: Vec<_> = operations
            .iter()
            .filter(|operation| operation.r#type == "gather")
            .collect();
        assert_eq!(
            gathers.len(),
            2,
            "native nearest sampling has a different coordinate rule"
        );
        assert_eq!(immediate_ints(&gathers[0].inputs["axis"]), [1]);
        assert_eq!(immediate_ints(&gathers[0].inputs["indices"]), [0, 0, 1]);
        assert_eq!(immediate_ints(&gathers[1].inputs["axis"]), [2]);
        assert_eq!(immediate_ints(&gathers[1].inputs["indices"]), [0, 1, 1, 2]);
        assert!(
            !operations
                .iter()
                .any(|operation| operation.r#type == "upsample_nearest_neighbor")
        );
    }
}

#[cfg(feature = "dynamic-inputs")]
#[test]
fn nearest_resample_rejects_unproven_dynamic_resized_axis() {
    use rustnn::graph::DynamicDimension;
    let graph = graph(
        DataType::Float32,
        vec![
            Dimension::Static(1),
            Dimension::Static(1),
            Dimension::Static(1),
            Dimension::Dynamic(DynamicDimension {
                name: "width".into(),
                max_size: 3,
            }),
        ],
        static_shape(&[1, 1, 1, 4]),
        [2, 3],
        [1, 4],
    );
    let error = CoremlMlProgramConverter
        .convert(&graph)
        .unwrap_err()
        .to_string();
    assert!(error.contains("dynamic resized axis"), "{error}");
}

#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
mod runtime {
    use super::*;
    use rustnn::mlcontext::{
        Backend, MLContext, MLContextOptions, MLGraphBuilder, MLNamedTensors, MLPowerPreference,
        MLTensorDescriptor,
    };
    use rustnn::operator_enums::MLOperandDataType;

    #[test]
    fn nearest_resample_upsample_downsample_and_singleton_values() {
        for (width, target, indices) in [
            (3, 4, vec![0, 1, 1, 2]),
            (4, 3, vec![0, 1, 3]),
            (5, 2, vec![1, 3]),
            (3, 1, vec![1]),
            (1, 4, vec![0, 0, 0, 0]),
        ] {
            let mut context = MLContext::create(
                &MLContextOptions::new(MLPowerPreference::Default, false)
                    .with_rustnn_backend_hint(Backend::Coreml),
            )
            .unwrap();
            let info = graph(
                DataType::Float32,
                static_shape(&[1, 1, 1, width]),
                static_shape(&[1, 1, 1, target]),
                [2, 3],
                [1, target],
            );
            let mut compiled = MLGraphBuilder::new(&mut context)
                .unwrap()
                .build_graph_info(info)
                .unwrap();
            let input = context
                .create_tensor(
                    &MLTensorDescriptor::new(
                        MLOperandDataType::Float32,
                        vec![1, 1, 1, width as u64],
                    )
                    .to_writable(),
                )
                .unwrap();
            let data: Vec<_> = (0..width).map(|index| 10. + index as f32).collect();
            context.write_tensor(&input, &data).unwrap();
            let output = context
                .create_tensor(
                    &MLTensorDescriptor::new(
                        MLOperandDataType::Float32,
                        vec![1, 1, 1, target as u64],
                    )
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
            let mut values = vec![f32::NAN; target as usize];
            context.read_tensor(&output, &mut values).unwrap();
            assert_eq!(
                values,
                indices.iter().map(|&index| data[index]).collect::<Vec<_>>(),
                "{width}->{target}"
            );
        }
    }

    #[cfg(feature = "dynamic-inputs")]
    #[test]
    fn nearest_resample_dynamic_batch_reuses_actual_shapes() {
        use rustnn::graph::DynamicDimension;
        let dimension = Dimension::Dynamic(DynamicDimension {
            name: "batch".into(),
            max_size: 3,
        });
        let mut context = MLContext::create(
            &MLContextOptions::new(MLPowerPreference::Default, false)
                .with_rustnn_backend_hint(Backend::Coreml),
        )
        .unwrap();
        let info = graph(
            DataType::Float32,
            vec![
                dimension.clone(),
                Dimension::Static(1),
                Dimension::Static(1),
                Dimension::Static(3),
            ],
            vec![
                dimension,
                Dimension::Static(1),
                Dimension::Static(1),
                Dimension::Static(4),
            ],
            [2, 3],
            [1, 4],
        );
        let mut compiled = MLGraphBuilder::new(&mut context)
            .unwrap()
            .build_graph_info(info)
            .unwrap();
        for batch in [1, 3, 1] {
            let input = context
                .create_tensor(
                    &MLTensorDescriptor::new(MLOperandDataType::Float32, vec![batch, 1, 1, 3])
                        .to_writable(),
                )
                .unwrap();
            let data: Vec<_> = (0..batch * 3).map(|index| 1. + index as f32).collect();
            context.write_tensor(&input, &data).unwrap();
            let output = context
                .create_tensor(
                    &MLTensorDescriptor::new(MLOperandDataType::Float32, vec![batch, 1, 1, 4])
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
            let mut actual = vec![f32::NAN; batch as usize * 4];
            context.read_tensor(&output, &mut actual).unwrap();
            let expected: Vec<_> = data
                .as_chunks::<3>()
                .0
                .iter()
                .flat_map(|row| [row[0], row[1], row[1], row[2]])
                .collect();
            assert_eq!(actual, expected, "batch={batch}");
        }
    }
}
