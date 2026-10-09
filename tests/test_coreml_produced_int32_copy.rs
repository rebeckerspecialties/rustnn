//! Exact copy transport must also preserve values produced inside a graph.
#![cfg(all(target_os = "macos", feature = "coreml-runtime"))]

use rustnn::mlcontext::{
    Backend, MLContext, MLContextOptions, MLGraphBuilder, MLNamedOperands, MLNamedTensors,
    MLOperandDescriptor, MLPowerPreference, MLTensorDescriptor, RustNNOptions,
};
use rustnn::operator_enums::MLOperandDataType;

#[test]
#[cfg(feature = "dynamic-inputs")]
fn produced_int32_identity_chains_keep_native_arithmetic_and_actual_shapes() {
    use rustnn::graph::{
        DataType, Dimension, DynamicDimension, GraphInfo, Operand, OperandDescriptor, OperandKind,
    };
    use rustnn::operators::Operation;
    let shape = vec![Dimension::Dynamic(DynamicDimension {
        name: "length".into(),
        max_size: 4,
    })];
    let operand = |name: &str, kind| Operand {
        name: Some(name.into()),
        kind,
        descriptor: OperandDescriptor {
            data_type: DataType::Int32,
            shape: shape.clone(),
            pending_permutation: vec![],
        },
    };
    let graph = GraphInfo {
        operands: vec![
            operand("source", OperandKind::Input),
            operand("zero", OperandKind::Input),
            operand("sum", OperandKind::Intermediate),
            operand("copy", OperandKind::Intermediate),
            operand("copy2", OperandKind::Intermediate),
            operand("result", OperandKind::Output),
        ],
        input_operands: vec![0, 1],
        output_operands: vec![5],
        operations: vec![
            Operation::Add {
                a: 0,
                b: 1,
                outputs: vec![2],
                options: None,
            },
            Operation::Identity {
                input: 2,
                outputs: vec![3],
                options: None,
            },
            Operation::Identity {
                input: 3,
                outputs: vec![4],
                options: None,
            },
            Operation::Add {
                a: 4,
                b: 1,
                outputs: vec![5],
                options: None,
            },
        ],
        ..Default::default()
    };
    for (accelerated, power) in [
        (false, MLPowerPreference::Default),
        (true, MLPowerPreference::Default),
        (true, MLPowerPreference::LowPower),
    ] {
        let mut options = RustNNOptions::default();
        options.coreml.reuse_tensor_storage = true;
        options.coreml.output_backings = true;
        let mut context = MLContext::create(
            &MLContextOptions::new(power, accelerated)
                .with_rustnn_backend_hint(Backend::Coreml)
                .with_rustnn_options(options),
        )
        .unwrap();
        let mut graph = context.rustnn_build_graph(graph.clone()).unwrap();
        for count in [1, 4, 2, 1] {
            let descriptor = MLTensorDescriptor::new(MLOperandDataType::Int32, vec![count])
                .to_readable()
                .to_writable();
            let source = context.create_tensor(&descriptor).unwrap();
            let zero = context.create_tensor(&descriptor).unwrap();
            let result = context.create_tensor(&descriptor).unwrap();
            // This checks shape/lifetime transport through native arithmetic,
            // not its full-range accuracy. Physical-device large-Int32 Add/Mul
            // failures are audited separately; the Div-produced copy test below
            // retains its exact large-value gate.
            let expected = [17, -17, -101, 101][..count as usize].to_vec();
            context.write_tensor(&source, &expected).unwrap();
            context
                .write_tensor(&zero, &vec![0_i32; count as usize])
                .unwrap();
            context
                .dispatch(
                    &mut graph,
                    &MLNamedTensors::from([("source", &source), ("zero", &zero)]),
                    &MLNamedTensors::from([("result", &result)]),
                )
                .unwrap();
            let mut actual = vec![0_i32; count as usize];
            context.read_tensor(&result, &mut actual).unwrap();
            assert_eq!(actual, expected, "count={count}, policy={power:?}");
            let Some(rustnn::mlcontext::LoadDiagnostics::Coreml(diagnostics)) =
                graph.rustnn_load_diagnostics()
            else {
                panic!()
            };
            assert_ne!(
                diagnostics.loaded_compute_units, "NOT_APPLICABLE",
                "native arithmetic must remain present"
            );
        }
    }
}

#[test]
fn produced_int32_copy_plan_does_not_require_division_boundaries() {
    use prost::Message;
    use rustnn::converters::{CoremlMlProgramConverter, GraphConverter};
    use rustnn::graph::{
        DataType, GraphInfo, Operand, OperandDescriptor, OperandKind, to_dimension_vector,
    };
    use rustnn::operators::Operation;
    use rustnn::protos::coreml::specification::{Model, model};
    let operand = |name: &str, kind| Operand {
        name: Some(name.into()),
        kind,
        descriptor: OperandDescriptor {
            data_type: DataType::Int32,
            shape: to_dimension_vector(&[4]),
            pending_permutation: vec![],
        },
    };
    let graph = GraphInfo {
        operands: vec![
            operand("source", OperandKind::Input),
            operand("zero", OperandKind::Input),
            operand("sum", OperandKind::Intermediate),
            operand("copy", OperandKind::Intermediate),
            operand("result", OperandKind::Output),
        ],
        input_operands: vec![0, 1],
        output_operands: vec![4],
        operations: vec![
            Operation::Add {
                a: 0,
                b: 1,
                outputs: vec![2],
                options: None,
            },
            Operation::Identity {
                input: 2,
                outputs: vec![3],
                options: None,
            },
            Operation::Identity {
                input: 3,
                outputs: vec![4],
                options: None,
            },
        ],
        ..Default::default()
    };
    let bytes = CoremlMlProgramConverter.convert(&graph).unwrap().data;
    let model::Type::Pipeline(pipeline) = Model::decode(bytes.as_slice()).unwrap().r#type.unwrap()
    else {
        panic!("produced Int32 copy must be isolated without a Div cut");
    };
    assert_eq!(pipeline.models.len(), 3);
    for child in &pipeline.models[1..] {
        let model::Type::MlProgram(program) = child.r#type.as_ref().unwrap() else {
            panic!()
        };
        let function = &program.functions["main"];
        let block = &function.block_specializations[&function.opset];
        assert_eq!(block.operations.len(), 1);
        assert_eq!(block.operations[0].r#type, "identity");
    }
}

#[test]
fn produced_int32_identity_only_outputs_preserve_bits_and_independent_lifetimes() {
    let expected = [16_777_217, -16_777_217, i32::MAX, i32::MIN];
    for mode in 0..3 {
        for (accelerated, power) in [
            (false, MLPowerPreference::Default),
            (true, MLPowerPreference::Default),
            (true, MLPowerPreference::LowPower),
        ] {
            for (before, same_type_cast) in
                [(false, false), (false, true), (true, false), (true, true)]
            {
                let mut options = RustNNOptions::default();
                options.coreml.reuse_tensor_storage = mode != 0;
                options.coreml.output_backings = mode == 2;
                let mut context = MLContext::create(
                    &MLContextOptions::new(power, accelerated)
                        .with_rustnn_backend_hint(Backend::Coreml)
                        .with_rustnn_options(options),
                )
                .unwrap();
                let mut builder = MLGraphBuilder::new(&mut context).unwrap();
                let descriptor = MLOperandDescriptor::new(MLOperandDataType::Int32, vec![4]);
                let input = builder.input("input", &descriptor).unwrap();
                let divisor = builder.input("divisor", &descriptor).unwrap();
                let source = if before {
                    builder.identity(input).unwrap()
                } else {
                    input
                };
                let quotient = builder.div(source, divisor).unwrap();
                let first = builder.identity(quotient).unwrap();
                let second = if same_type_cast {
                    builder.cast(first, MLOperandDataType::Int32).unwrap()
                } else {
                    builder.identity(first).unwrap()
                };
                // Do not also return quotient: that would let public-output
                // coalescing hide an inaccurate intermediate Identity child.
                let mut graph = builder
                    .build(&MLNamedOperands::from([
                        ("first", first),
                        ("second", second),
                    ]))
                    .unwrap();
                let descriptor = MLTensorDescriptor::new(MLOperandDataType::Int32, vec![4])
                    .to_readable()
                    .to_writable();
                let source = context.create_tensor(&descriptor).unwrap();
                let divisor = context.create_tensor(&descriptor).unwrap();
                let first = context.create_tensor(&descriptor).unwrap();
                let second = context.create_tensor(&descriptor).unwrap();
                context.write_tensor(&divisor, &[1; 4]).unwrap();
                for values in [expected, [i32::MIN, i32::MAX, -7, 7], expected] {
                    context.write_tensor(&source, &values).unwrap();
                    context
                        .dispatch(
                            &mut graph,
                            &MLNamedTensors::from([("input", &source), ("divisor", &divisor)]),
                            &MLNamedTensors::from([("first", &first), ("second", &second)]),
                        )
                        .unwrap();
                    let Some(rustnn::mlcontext::LoadDiagnostics::Coreml(diagnostics)) =
                        graph.rustnn_load_diagnostics()
                    else {
                        panic!("missing CoreML load diagnostics")
                    };
                    assert_eq!(
                        diagnostics.loaded_compute_units, "NOT_APPLICABLE",
                        "Div and Identity must both use source-proven typed execution"
                    );
                    for output in [&first, &second] {
                        let mut actual = [0_i32; 4];
                        context.read_tensor(output, &mut actual).unwrap();
                        assert_eq!(
                            actual, values,
                            "storage={mode}, power={power:?}, before={before}, cast={same_type_cast}"
                        );
                    }
                }
                drop(graph);
                context.write_tensor(&source, &[0; 4]).unwrap();
                context.write_tensor(&first, &[123; 4]).unwrap();
                let mut retained = [0_i32; 4];
                context.read_tensor(&second, &mut retained).unwrap();
                assert_eq!(
                    retained, expected,
                    "returned tensors must own independent destinations"
                );
            }
        }
    }
}
