//! Compile once and check exact values through nonzero grow/shrink/reset cycles.
#![cfg(all(
    target_os = "macos",
    feature = "coreml-runtime",
    feature = "dynamic-inputs"
))]

use std::collections::HashMap;

use rustnn::converters::{CoremlMlProgramConverter, GraphConverter};
use rustnn::executors::coreml::{CoremlInput, run_coreml_with_inputs_checked};
use rustnn::graph::{
    DataType, Dimension, DynamicDimension, GraphInfo, Operand, OperandDescriptor, OperandKind,
};
use rustnn::mlcontext::{
    Backend, MLContext, MLContextOptions, MLGraphBuilder, MLNamedTensors, MLPowerPreference,
    MLTensorDescriptor,
};
use rustnn::operator_enums::MLOperandDataType;
use rustnn::operator_options::{MLDimension, MLDynamicDimension};
use rustnn::operators::Operation;

fn reshape_graph() -> GraphInfo {
    let sequence = Dimension::Dynamic(DynamicDimension {
        name: "sequence".into(),
        max_size: 64,
    });
    let input_shape = vec![
        Dimension::Static(1),
        sequence.clone(),
        Dimension::Static(1024),
    ];
    let output_shape = vec![
        Dimension::Static(1),
        sequence,
        Dimension::Static(16),
        Dimension::Static(64),
    ];
    let operand = |name: &str, kind, shape| Operand {
        name: Some(name.into()),
        kind,
        descriptor: OperandDescriptor {
            data_type: DataType::Float32,
            shape,
            pending_permutation: vec![],
        },
    };
    GraphInfo {
        operands: vec![
            operand("input", OperandKind::Input, input_shape),
            operand("output", OperandKind::Output, output_shape),
        ],
        input_operands: vec![0],
        output_operands: vec![1],
        operations: vec![Operation::Reshape {
            input: 0,
            new_shape: vec![
                MLDimension::Static(1),
                MLDimension::Dynamic(MLDynamicDimension {
                    name: "sequence".into(),
                    max_size: 64,
                }),
                MLDimension::Static(16),
                MLDimension::Static(64),
            ],
            options: None,
            outputs: vec![1],
        }],
        ..Default::default()
    }
}

fn input_values(length: usize) -> Vec<f32> {
    (0..length * 1024)
        .map(|index| ((index % 257) as f32 - 128.0) / 8.0)
        .collect()
}

#[test]
fn inferred_reshape_preserves_exact_values_across_repeated_resizes() {
    for accelerated in [false, true] {
        let mut context = MLContext::create(
            &MLContextOptions::new(MLPowerPreference::Default, accelerated)
                .with_rustnn_backend_hint(Backend::Coreml),
        )
        .expect("CoreML context");
        let mut graph = MLGraphBuilder::new(&mut context)
            .unwrap()
            .build_graph_info(reshape_graph())
            .unwrap();
        for length in [1u64, 2, 4, 64, 2, 1] {
            let values = input_values(length as usize);
            let input = context
                .create_tensor(
                    &MLTensorDescriptor::new(MLOperandDataType::Float32, vec![1, length, 1024])
                        .to_writable(),
                )
                .unwrap();
            let output = context
                .create_tensor(
                    &MLTensorDescriptor::new(MLOperandDataType::Float32, vec![1, length, 16, 64])
                        .to_readable(),
                )
                .unwrap();
            context.write_tensor(&input, &values).unwrap();
            context
                .dispatch(
                    &mut graph,
                    &MLNamedTensors::from([("input", &input)]),
                    &MLNamedTensors::from([("output", &output)]),
                )
                .unwrap();
            let mut actual = vec![f32::NAN; values.len()];
            context.read_tensor(&output, &mut actual).unwrap();
            assert_eq!(output.shape(), &[1, length, 16, 64]);
            assert_eq!(actual, values, "accelerated={accelerated}, length={length}");
        }
    }
}

#[test]
fn inferred_reshape_reports_actual_output_shape() {
    // This executor reads CoreML's returned shape rather than the public
    // tensor's preallocated descriptor. Each invocation compiles separately;
    // the dispatch test above exercises one model across repeated resizes.
    let graph = reshape_graph();
    let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
    assert!(converted.weights_data.is_none());
    let input_descriptors = HashMap::from([("input".into(), graph.operands[0].descriptor.clone())]);
    let output_descriptors =
        HashMap::from([("output".into(), graph.operands[1].descriptor.clone())]);
    for length in [1usize, 4, 64, 2, 1] {
        let values = input_values(length);
        let attempts = run_coreml_with_inputs_checked(
            &converted.data,
            vec![CoremlInput {
                name: "input".into(),
                shape: vec![1, length, 1024],
                data: values.clone(),
            }],
            &input_descriptors,
            &output_descriptors,
        )
        .unwrap();
        let mut cpu_succeeded = false;
        for attempt in attempts {
            if attempt.compute_unit == "CPU_ONLY" {
                assert!(
                    attempt.result.is_ok(),
                    "length={length}: {:?}",
                    attempt.result
                );
                cpu_succeeded = true;
            }
            match attempt.result {
                Ok(outputs) => {
                    assert_eq!(outputs.len(), 1);
                    assert_eq!(outputs[0].name, "output");
                    assert_eq!(outputs[0].shape, vec![1, length as i64, 16, 64]);
                    assert_eq!(
                        outputs[0].data, values,
                        "length={length}, {}",
                        attempt.compute_unit
                    );
                }
                Err(error) => eprintln!("length={length}, {}: {error}", attempt.compute_unit),
            }
        }
        assert!(cpu_succeeded, "CPU_ONLY must produce checked output");
    }
}
