use super::*;
use crate::converters::{CoremlMlProgramConverter, GraphConverter};
use crate::graph::{
    ConstantData, DataType, Dimension, GraphInfo, Operand, OperandDescriptor, OperandKind,
};
use crate::operator_options::MLTransposeOptions;
use crate::operators::Operation;
use crate::{ContextProperties, GraphValidator};
use std::collections::HashSet;

struct Fixture {
    label: String,
    graph: GraphInfo,
    inputs: Vec<(String, Vec<u8>, OperandDescriptor)>,
    expected_bits: Vec<u32>,
    right_constant: Option<Vec<u8>>,
}

fn operand(name: &str, kind: OperandKind, shape: &[usize]) -> Operand {
    Operand {
        name: Some(name.into()),
        kind,
        descriptor: OperandDescriptor {
            data_type: DataType::Float32,
            shape: shape
                .iter()
                .map(|&size| Dimension::Static(size.try_into().unwrap()))
                .collect(),
            pending_permutation: vec![],
        },
    }
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn f32_output_bits(bytes: &[u8]) -> Vec<u32> {
    let (values, trailing) = bytes.as_chunks::<4>();
    assert!(trailing.is_empty(), "partial Float32 output element");
    values
        .iter()
        .map(|bytes| u32::from_le_bytes(*bytes))
        .collect()
}

fn permutations(inner: usize) -> Vec<Vec<usize>> {
    fn visit(prefix: &mut Vec<usize>, unused: &mut Vec<usize>, output: &mut Vec<Vec<usize>>) {
        if unused.is_empty() {
            output.push(prefix.clone());
            return;
        }
        for index in 0..unused.len() {
            let value = unused.remove(index);
            prefix.push(value);
            visit(prefix, unused, output);
            prefix.pop();
            unused.insert(index, value);
        }
    }
    let mut output = vec![];
    visit(&mut vec![], &mut (0..inner).collect(), &mut output);
    output
}

fn transpose_last_two(values: &[f32], shape: &[usize]) -> (Vec<f32>, Vec<usize>) {
    let rank = shape.len();
    let rows = shape[rank - 2];
    let columns = shape[rank - 1];
    let mut result = vec![0f32; values.len()];
    for batch in 0..values.len() / (rows * columns) {
        for row in 0..rows {
            for column in 0..columns {
                result[batch * rows * columns + column * rows + row] =
                    values[batch * rows * columns + row * columns + column];
            }
        }
    }
    let mut result_shape = shape.to_vec();
    result_shape.swap(rank - 2, rank - 1);
    (result, result_shape)
}

fn cancellation(
    inner: usize,
    transpose_left: bool,
    transpose_right: bool,
    broadcast: bool,
    constant_right: bool,
) -> Fixture {
    assert!(inner == 3 || inner == 5);
    let coefficients = if inner == 3 {
        vec![2f32.powi(24), 1., -2f32.powi(24)]
    } else {
        vec![2f32.powi(60), 1., 2f32.powi(-24), -2f32.powi(60), -1.]
    };
    let permutations = permutations(inner);
    let columns = permutations.len() * 2;
    let mut left = Vec::new();
    for batch in 0..if broadcast { 2 } else { 1 } {
        let sign = if batch == 0 { 1. } else { -1. };
        left.extend(std::iter::repeat_n(sign, inner));
        left.extend(std::iter::repeat_n(-sign, inner));
    }
    let mut right = Vec::new();
    for batch in 0..if broadcast { 3 } else { 1 } {
        let batch_sign = if batch == 1 { -1. } else { 1. };
        for k in 0..inner {
            for permutation in &permutations {
                for sign in [1., -1.] {
                    right.push(coefficients[permutation[k]] * sign * batch_sign);
                }
            }
        }
    }
    let left_shape = if broadcast {
        vec![2, 1, 2, inner]
    } else {
        vec![2, inner]
    };
    let right_shape = if broadcast {
        vec![3, inner, columns]
    } else {
        vec![inner, columns]
    };
    let output_shape = if broadcast {
        vec![2, 3, 2, columns]
    } else {
        vec![2, columns]
    };
    let (left, stored_left_shape) = if transpose_left {
        transpose_last_two(&left, &left_shape)
    } else {
        (left, left_shape.clone())
    };
    let (right, stored_right_shape) = if transpose_right {
        transpose_last_two(&right, &right_shape)
    } else {
        (right, right_shape.clone())
    };
    let left_bytes = f32_bytes(&left);
    let right_bytes = f32_bytes(&right);
    let mut graph = GraphInfo {
        operands: vec![
            operand("left input", OperandKind::Input, &stored_left_shape),
            operand(
                "right input",
                if constant_right {
                    OperandKind::Constant
                } else {
                    OperandKind::Input
                },
                &stored_right_shape,
            ),
        ],
        input_operands: if constant_right { vec![0] } else { vec![0, 1] },
        ..Default::default()
    };
    if constant_right {
        graph.constant_operand_ids_to_handles.insert(
            1,
            ConstantData {
                data: right_bytes.clone(),
                label: None,
            },
        );
    }
    let mut left_id = 0;
    let mut right_id = 1;
    for (transpose, id, shape, name) in [
        (transpose_left, &mut left_id, &left_shape, "left matrix"),
        (transpose_right, &mut right_id, &right_shape, "right matrix"),
    ] {
        if transpose {
            let result = graph.operands.len().try_into().unwrap();
            graph
                .operands
                .push(operand(name, OperandKind::Intermediate, shape));
            let mut permutation: Vec<u32> = (0..shape.len())
                .map(|index| index.try_into().unwrap())
                .collect();
            permutation.swap(shape.len() - 2, shape.len() - 1);
            graph.operations.push(Operation::Transpose {
                input: *id,
                options: Some(MLTransposeOptions {
                    permutation,
                    ..Default::default()
                }),
                outputs: vec![result],
            });
            *id = result;
        }
    }
    let output = graph.operands.len().try_into().unwrap();
    graph
        .operands
        .push(operand("exact result", OperandKind::Output, &output_shape));
    graph.output_operands = vec![output];
    graph.operations.push(Operation::Matmul {
        a: left_id,
        b: right_id,
        options: None,
        outputs: vec![output],
    });
    let value = if inner == 3 { 1f32 } else { 2f32.powi(-24) };
    let mut expected_bits = Vec::new();
    for left_batch in 0..if broadcast { 2 } else { 1 } {
        for right_batch in 0..if broadcast { 3 } else { 1 } {
            for row in 0..2 {
                for column in 0..columns {
                    let sign =
                        if (left_batch + usize::from(right_batch == 1) + row + column % 2) % 2 == 0
                        {
                            1.
                        } else {
                            -1.
                        };
                    expected_bits.push((value * sign).to_bits());
                }
            }
        }
    }
    let mut inputs = vec![(
        "left input".into(),
        left_bytes,
        graph.operands[0].descriptor.clone(),
    )];
    if !constant_right {
        inputs.push((
            "right input".into(),
            right_bytes.clone(),
            graph.operands[1].descriptor.clone(),
        ));
    }
    Fixture {
        label: format!(
            "k{inner}-tx{transpose_left}-ty{transpose_right}-broadcast{broadcast}-constant{constant_right}"
        ),
        graph,
        inputs,
        expected_bits,
        right_constant: constant_right.then_some(right_bytes),
    }
}

fn fixtures() -> Vec<Fixture> {
    let mut output = vec![];
    for inner in [3, 5] {
        for tx in [false, true] {
            for ty in [false, true] {
                for batch in [false, true] {
                    for constant in [false, true] {
                        output.push(cancellation(inner, tx, ty, batch, constant));
                    }
                }
            }
        }
    }
    output
}

fn blob_bytes(weights: &[u8], metadata_offset: usize, elements: usize) -> &[u8] {
    let metadata = &weights[metadata_offset..metadata_offset + 24];
    assert_eq!(
        u32::from_le_bytes(metadata[0..4].try_into().unwrap()),
        0xdeadbeef
    );
    assert_eq!(u32::from_le_bytes(metadata[4..8].try_into().unwrap()), 2);
    let length: usize = u64::from_le_bytes(metadata[8..16].try_into().unwrap())
        .try_into()
        .unwrap();
    let offset: usize = u64::from_le_bytes(metadata[16..24].try_into().unwrap())
        .try_into()
        .unwrap();
    assert_eq!(length, elements.checked_mul(4).unwrap());
    &weights[offset..offset + length]
}

fn assert_dependencies(plan: &SourcePlan, label: &str) {
    let mut available: HashSet<_> = plan.inputs.keys().cloned().collect();
    let mut releases = HashSet::new();
    for (index, stage) in plan.stages.iter().enumerate() {
        for input in &stage.inputs {
            assert!(
                available.contains(&input.name),
                "{label}: stage{index} input has no live producer"
            );
        }
        for output in &stage.outputs {
            assert!(
                available.insert(output.name.clone()),
                "{label}: feature redefinition"
            );
        }
        for name in &stage.release_after {
            assert!(
                !plan.outputs.contains_key(name),
                "{label}: public output released"
            );
            assert!(available.remove(name), "{label}: releasing absent feature");
            assert!(releases.insert(name.clone()), "{label}: repeated release");
            assert!(
                plan.stages[index + 1..]
                    .iter()
                    .all(|later| later.inputs.iter().all(|input| input.name != *name)),
                "{label}: feature released before final consumer"
            );
        }
    }
    assert!(
        plan.outputs.keys().all(|name| available.contains(name)),
        "{label}: final output lost"
    );
}

fn assert_live_source_closures(model: &Model) {
    match model.r#type.as_ref().unwrap() {
        model::Type::Pipeline(pipeline) => {
            for child in &pipeline.models {
                assert_live_source_closures(child);
            }
        }
        model::Type::MlProgram(program) => {
            let function = &program.functions["main"];
            let block = &function.block_specializations[&function.opset];
            let mut needed: HashSet<_> = block.outputs.iter().cloned().collect();
            for operation in block.operations.iter().rev() {
                let live = operation
                    .outputs
                    .iter()
                    .any(|output| needed.contains(&output.name));
                if !live {
                    assert!(
                        !matches!(
                            operation.r#type.as_str(),
                            "const" | "transpose" | "reshape" | "identity"
                        ),
                        "dead {} closure in stage: {:?}",
                        operation.r#type,
                        operation.outputs
                    );
                    continue;
                }
                for output in &operation.outputs {
                    needed.remove(&output.name);
                }
                for argument in operation.inputs.values().flat_map(|input| &input.arguments) {
                    if let Some(
                        crate::protos::coreml::mil_spec::argument::binding::Binding::Name(name),
                    ) = &argument.binding
                    {
                        needed.insert(name.clone());
                    }
                }
            }
            let inputs: HashSet<_> = function.inputs.iter().map(|v| v.name.clone()).collect();
            assert!(
                needed.is_subset(&inputs),
                "live closure has an unresolved source"
            );
        }
        _ => panic!("Expected ordinary MLProgram sources"),
    }
}

fn recognized<'a>(
    plan: &'a SourcePlan,
    fixture: &Fixture,
    weights: Option<&[u8]>,
) -> &'a matmul::Plan {
    let matrices: Vec<_> = plan
        .stages
        .iter()
        .filter_map(|stage| match &stage.execution {
            StageExecution::ExactMatmul(matrix) => Some((stage, matrix)),
            _ => None,
        })
        .collect();
    assert_eq!(
        matrices.len(),
        1,
        "{}: parser must recognize exactly one Rust matmul",
        fixture.label
    );
    assert_dependencies(plan, &fixture.label);
    let (stage, matrix) = matrices[0];
    let expected_shape: Vec<usize> = fixture.graph.operands
        [*fixture.graph.output_operands.first().unwrap() as usize]
        .descriptor
        .shape
        .iter()
        .map(|size| size.get_static_or_max_size() as usize)
        .collect();
    assert_eq!(
        matrix.output_shape, expected_shape,
        "{}: recognized geometry",
        fixture.label
    );
    assert_eq!(stage.outputs.len(), 1);
    assert_eq!(stage.outputs[0].name, matrix.output);
    let left_rank = matrix.left.shape.len();
    let right_rank = matrix.right.shape.len();
    let left_k = matrix.left.shape[left_rank - if matrix.transpose_left { 2 } else { 1 }];
    let right_k = matrix.right.shape[right_rank - if matrix.transpose_right { 1 } else { 2 }];
    assert_eq!(left_k, right_k, "{}: reduction dimension", fixture.label);
    assert!(left_k == 3 || left_k == 5);
    if let Some(original) = &fixture.right_constant {
        let actual = match &matrix.right.source {
            matmul::Source::Blob {
                metadata_offset,
                elements,
            } => blob_bytes(weights.unwrap(), *metadata_offset, *elements).to_vec(),
            matmul::Source::Immediate(values) => f32_bytes(values),
            matmul::Source::Input(_) => panic!(
                "{}: constant transpose became a per-token native feature",
                fixture.label
            ),
        };
        assert_eq!(
            actual, *original,
            "{}: original learned/constant bytes changed",
            fixture.label
        );
    } else {
        assert!(
            matches!(&matrix.right.source, matmul::Source::Input(_)),
            "{}: runtime RHS was captured as a constant",
            fixture.label
        );
    }
    matrix
}
#[test]
fn exact_matmul_source_plan_recognizes_generic_cancellation_views_and_constants() {
    for fixture in fixtures() {
        GraphValidator::new(&fixture.graph, ContextProperties::default())
            .validate()
            .unwrap();
        let original = serde_json::to_value(&fixture.graph).unwrap();
        let converted = CoremlMlProgramConverter.convert(&fixture.graph).unwrap();
        assert_live_source_closures(&Model::decode(converted.data.as_slice()).unwrap());
        assert_eq!(
            serde_json::to_value(&fixture.graph).unwrap(),
            original,
            "{}: conversion mutated GraphInfo",
            fixture.label
        );
        let plan = SourcePlan::parse(&converted.data)
            .unwrap()
            .expect("source plan must be recognized");
        recognized(&plan, &fixture, converted.weights_data.as_deref());
    }
}

fn exact_marker_count(source: &Model) -> usize {
    usize::from(
        source
            .description
            .as_ref()
            .and_then(|description| description.metadata.as_ref())
            .is_some_and(|metadata| {
                metadata
                    .user_defined
                    .contains_key("rustnn.webnn.exact_f32_matmul")
            }),
    ) + match &source.r#type {
        Some(model::Type::Pipeline(pipeline)) => {
            pipeline.models.iter().map(exact_marker_count).sum()
        }
        _ => 0,
    }
}

#[cfg(feature = "dynamic-inputs")]
#[test]
fn exact_matmul_marking_preserves_native_bounded_dynamic_route() {
    let mut graph = recurrent_cancellation_graph();
    for id in [0, 2, 3, 4] {
        graph.operands[id].descriptor.shape[0] =
            Dimension::Dynamic(crate::graph::DynamicDimension {
                name: "batch".into(),
                max_size: 4,
            });
    }
    GraphValidator::new(&graph, ContextProperties::default())
        .validate()
        .unwrap();
    let before = serde_json::to_value(&graph).unwrap();
    let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
    assert_eq!(serde_json::to_value(&graph).unwrap(), before);
    let source = Model::decode(converted.data.as_slice()).unwrap();
    assert_eq!(exact_marker_count(&source), 0);
    assert!(SourcePlan::parse(&converted.data).unwrap().is_none());
}

#[test]
fn exact_matmul_marking_preserves_native_output_layout_adapters() {
    let mut fixture = cancellation(3, false, false, false, false);
    let id = fixture.graph.output_operands[0] as usize;
    fixture.graph.operands[id].descriptor.pending_permutation = vec![1, 0];
    let before = serde_json::to_value(&fixture.graph).unwrap();
    let converted = CoremlMlProgramConverter.convert(&fixture.graph).unwrap();
    assert_eq!(serde_json::to_value(&fixture.graph).unwrap(), before);
    assert_eq!(
        exact_marker_count(&Model::decode(converted.data.as_slice()).unwrap()),
        0
    );
}

#[test]
fn exact_matmul_marking_preserves_native_noncontiguous_constant_reshape() {
    let graph = GraphInfo {
        operands: vec![
            operand("left", OperandKind::Input, &[1, 2]),
            operand("weight", OperandKind::Constant, &[2, 3]),
            operand("transposed", OperandKind::Intermediate, &[3, 2]),
            operand("reshaped", OperandKind::Intermediate, &[2, 3]),
            operand("result", OperandKind::Output, &[1, 3]),
        ],
        input_operands: vec![0],
        output_operands: vec![4],
        operations: vec![
            Operation::Transpose {
                input: 1,
                options: Some(MLTransposeOptions {
                    permutation: vec![1, 0],
                    ..Default::default()
                }),
                outputs: vec![2],
            },
            Operation::Reshape {
                input: 2,
                new_shape: vec![
                    crate::operator_options::MLDimension::Static(2),
                    crate::operator_options::MLDimension::Static(3),
                ],
                options: None,
                outputs: vec![3],
            },
            Operation::Matmul {
                a: 0,
                b: 3,
                options: None,
                outputs: vec![4],
            },
        ],
        constant_operand_ids_to_handles: [(
            1,
            ConstantData {
                data: f32_bytes(&[1., 2., 3., 4., 5., 6.]),
                label: None,
            },
        )]
        .into(),
        ..Default::default()
    };
    GraphValidator::new(&graph, ContextProperties::default())
        .validate()
        .unwrap();
    let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
    assert_eq!(
        exact_marker_count(&Model::decode(converted.data.as_slice()).unwrap()),
        0
    );
    assert!(SourcePlan::parse(&converted.data).unwrap().is_none());
}

#[test]
fn exact_matmul_marking_preserves_native_constexpr_parameter_closures() {
    let mut graph = GraphInfo {
        operands: vec![
            operand("left", OperandKind::Input, &[1, 2]),
            operand("quantized", OperandKind::Constant, &[2, 2]),
            operand("scale", OperandKind::Constant, &[]),
            operand("zero", OperandKind::Constant, &[]),
            operand("weight", OperandKind::Intermediate, &[2, 2]),
            operand("result", OperandKind::Output, &[1, 2]),
        ],
        input_operands: vec![0],
        output_operands: vec![5],
        operations: vec![
            Operation::DequantizeLinear {
                input: 1,
                scale: 2,
                zero_point: Some(3),
                options: None,
                outputs: vec![4],
            },
            Operation::Matmul {
                a: 0,
                b: 4,
                options: None,
                outputs: vec![5],
            },
        ],
        quantized: true,
        ..Default::default()
    };
    graph.operands[1].descriptor.data_type = DataType::Uint8;
    graph.operands[3].descriptor.data_type = DataType::Uint8;
    for (id, data) in [(1, vec![1, 2, 3, 4]), (2, f32_bytes(&[0.5])), (3, vec![0])] {
        graph
            .constant_operand_ids_to_handles
            .insert(id, ConstantData { data, label: None });
    }
    GraphValidator::new(&graph, ContextProperties::default())
        .validate()
        .unwrap();
    let before = serde_json::to_value(&graph).unwrap();
    let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
    assert_eq!(serde_json::to_value(&graph).unwrap(), before);
    let source = Model::decode(converted.data.as_slice()).unwrap();
    fn has_constexpr(model: &Model) -> bool {
        match model.r#type.as_ref().unwrap() {
            model::Type::Pipeline(pipeline) => pipeline.models.iter().any(has_constexpr),
            model::Type::MlProgram(program) => program.functions.values().any(|function| {
                function.block_specializations.values().any(|block| {
                    block
                        .operations
                        .iter()
                        .any(|operation| operation.r#type.starts_with("constexpr_"))
                })
            }),
            _ => false,
        }
    }
    assert!(
        has_constexpr(&source),
        "fixture must exercise a native constexpr closure"
    );
    assert_eq!(exact_marker_count(&source), 0);
    assert!(SourcePlan::parse(&converted.data).unwrap().is_none());
}

#[test]
fn exact_matmul_constant_transpose_is_not_retained_in_unrelated_native_stages() {
    let mut fixture = cancellation(3, false, true, false, true);
    let graph = &mut fixture.graph;
    let one = u32::try_from(graph.operands.len()).unwrap();
    graph
        .operands
        .push(operand("one", OperandKind::Constant, &[1]));
    graph.constant_operand_ids_to_handles.insert(
        one,
        ConstantData {
            data: f32_bytes(&[1.0]),
            label: None,
        },
    );
    let pre = u32::try_from(graph.operands.len()).unwrap();
    graph
        .operands
        .push(operand("native pre", OperandKind::Intermediate, &[2, 3]));
    for operation in &mut graph.operations {
        if let Operation::Matmul { a, .. } = operation {
            *a = pre;
        }
    }
    graph.operations.insert(
        0,
        Operation::Mul {
            a: 0,
            b: one,
            options: None,
            outputs: vec![pre],
        },
    );
    let matrix_output = graph.output_operands[0];
    graph.operands[matrix_output as usize].name = None;
    graph.operands[matrix_output as usize].kind = OperandKind::Intermediate;
    let final_output = u32::try_from(graph.operands.len()).unwrap();
    graph
        .operands
        .push(operand("exact result", OperandKind::Output, &[2, 12]));
    graph.output_operands = vec![final_output];
    graph.operations.push(Operation::Mul {
        a: matrix_output,
        b: one,
        options: None,
        outputs: vec![final_output],
    });
    GraphValidator::new(graph, ContextProperties::default())
        .validate()
        .unwrap();
    let converted = CoremlMlProgramConverter.convert(graph).unwrap();
    let source = Model::decode(converted.data.as_slice()).unwrap();
    assert_live_source_closures(&source);
    let plan = SourcePlan::parse(&converted.data).unwrap().unwrap();
    recognized(&plan, &fixture, converted.weights_data.as_deref());
    assert!(
        plan.stages
            .iter()
            .filter(|stage| matches!(stage.execution, StageExecution::Native))
            .count()
            >= 2,
        "the regression must exercise native stages around the exact stage"
    );
}

fn matrix_operation_mut(model: &mut Model) -> &mut mil::Operation {
    match model.r#type.as_mut().unwrap() {
        model::Type::MlProgram(program) => program
            .functions
            .get_mut("main")
            .unwrap()
            .block_specializations
            .values_mut()
            .next()
            .unwrap()
            .operations
            .iter_mut()
            .find(|operation| operation.r#type == "matmul")
            .unwrap(),
        model::Type::Pipeline(pipeline) => {
            let index = pipeline
                .models
                .iter()
                .position(|child| match &child.r#type {
                    Some(model::Type::MlProgram(program)) => {
                        program.functions.values().any(|function| {
                            function.block_specializations.values().any(|block| {
                                block
                                    .operations
                                    .iter()
                                    .any(|operation| operation.r#type == "matmul")
                            })
                        })
                    }
                    _ => false,
                })
                .unwrap();
            matrix_operation_mut(&mut pipeline.models[index])
        }
        _ => panic!("Expected a documented MLProgram matrix source"),
    }
}

#[test]
fn exact_matmul_source_plan_rejects_invalid_contracts_instead_of_native_fallback() {
    let fixture = cancellation(3, false, false, false, true);
    let converted = CoremlMlProgramConverter.convert(&fixture.graph).unwrap();
    let mut missing_flag = Model::decode(converted.data.as_slice()).unwrap();
    matrix_operation_mut(&mut missing_flag)
        .inputs
        .remove("transpose_y");
    assert!(SourcePlan::parse(&missing_flag.encode_to_vec()).is_err());

    let mut altered_output = Model::decode(converted.data.as_slice()).unwrap();
    let operation = matrix_operation_mut(&mut altered_output);
    let Some(crate::protos::coreml::mil_spec::value_type::Type::TensorType(tensor)) =
        &mut operation.outputs[0].r#type.as_mut().unwrap().r#type
    else {
        panic!("tensor")
    };
    let Some(crate::protos::coreml::mil_spec::dimension::Dimension::Constant(extent)) =
        &mut tensor.dimensions[0].dimension
    else {
        panic!("static dimension")
    };
    extent.size += 1;
    assert!(SourcePlan::parse(&altered_output.encode_to_vec()).is_err());

    let mut weights = converted
        .weights_data
        .clone()
        .expect("generic weight is a source blob");
    weights.clear();
    assert!(compile_model(converted.data, Some(weights), DeviceType::Cpu, true).is_err());
}

#[test]
#[cfg(target_vendor = "apple")]
fn exact_matmul_normal_rust_executor_preserves_cancelled_results_across_routes_and_hints() {
    // Cancellation-sensitive values strengthen this implementation's fidelity
    // contract, not the WPT-mandated accuracy budget. WPT deliberately removed
    // such Float32 matmul inputs: https://github.com/web-platform-tests/wpt/pull/38679.
    for fixture in fixtures() {
        let converted = CoremlMlProgramConverter.convert(&fixture.graph).unwrap();
        let outputs: HashMap<_, _> = fixture
            .graph
            .output_operands
            .iter()
            .map(|&id| {
                let operand = &fixture.graph.operands[id as usize];
                (operand.name.clone().unwrap(), operand.descriptor.clone())
            })
            .collect();
        let inputs: HashMap<_, _> = fixture
            .inputs
            .iter()
            .map(|(name, bytes, descriptor)| {
                (
                    name.clone(),
                    CoremlByteInput {
                        data: bytes,
                        descriptor,
                    },
                )
            })
            .collect();
        for device in [DeviceType::Cpu, DeviceType::Gpu, DeviceType::Npu] {
            for in_memory in [false, true] {
                let model = compile_model(
                    converted.data.clone(),
                    converted.weights_data.clone(),
                    device,
                    in_memory,
                )
                .unwrap();
                let CompiledCoremlModel::Pipeline(pipeline) = &model else {
                    panic!("{}: normal executor bypassed source plan", fixture.label);
                };
                recognized(&pipeline.plan, &fixture, converted.weights_data.as_deref());
                for _ in 0..2 {
                    let actual = run_coreml_bytes(&model, &inputs, &outputs).unwrap();
                    let bytes = &actual["exact result"];
                    assert_eq!(bytes.len(), fixture.expected_bits.len() * 4);
                    assert_eq!(
                        f32_output_bits(bytes),
                        fixture.expected_bits,
                        "{}: ordinary executor, device{device:?}, in-memory{in_memory}",
                        fixture.label
                    );
                }
            }
        }
    }
}

fn recurrent_cancellation_graph() -> GraphInfo {
    let large = 2f32.powi(24);
    GraphInfo {
        operands: vec![
            operand("state", OperandKind::Input, &[1, 3]),
            operand("transition", OperandKind::Constant, &[3, 3]),
            operand("projected", OperandKind::Output, &[1, 3]),
            operand("increment", OperandKind::Input, &[1, 3]),
            operand("carry", OperandKind::Output, &[1, 3]),
        ],
        input_operands: vec![0, 3],
        output_operands: vec![2, 4],
        operations: vec![
            Operation::Matmul {
                a: 0,
                b: 1,
                options: None,
                outputs: vec![2],
            },
            Operation::Add {
                a: 2,
                b: 3,
                options: None,
                outputs: vec![4],
            },
        ],
        constant_operand_ids_to_handles: [(
            1,
            ConstantData {
                // Each column sums to one, with three accumulation orders.
                data: f32_bytes(&[large, 1., -large, 1., -large, large, -large, large, 1.]),
                label: None,
            },
        )]
        .into(),
        ..Default::default()
    }
}

fn dyadic_carry(sign: f32, completed_steps: u32) -> [u32; 3] {
    // The oracle is a closed-form dyadic, independent of the matrix kernel.
    // All intermediate carries are exactly representable in Float32.
    [(sign * (1024 + completed_steps) as f32 / 1024.).to_bits(); 3]
}

#[test]
fn recurrent_cancellation_oracle_rejects_lossy_summation() {
    let large = 2f32.powi(24);
    for sign in [1f32, -1.] {
        let expected = dyadic_carry(sign, 1);
        let increment = sign / 1024.;
        // Ordinary left-to-right Float32 loses the unit between large terms.
        let lossy = ((sign * large + sign) + sign * -large) + increment;
        assert_ne!(lossy.to_bits(), expected[0]);
    }
}

#[test]
#[cfg(target_vendor = "apple")]
fn exact_matmul_normal_rust_executor_preserves_own_output_recurrent_state() {
    // This is a stronger implementation/model-fidelity regression, not a new
    // WebNN or WPT accuracy requirement. WPT intentionally avoids catastrophic
    // cancellation (https://github.com/web-platform-tests/wpt/pull/38679), while
    // accumulation and composed-output budgets remain under discussion:
    // https://github.com/webmachinelearning/webnn/issues/948 and
    // https://github.com/webmachinelearning/webnn/issues/950.
    // Checking every signed carry prevents a loose endpoint-only check from
    // hiding an arithmetic error compounded by subsequent predictions.
    let graph = recurrent_cancellation_graph();
    GraphValidator::new(&graph, ContextProperties::default())
        .validate()
        .unwrap();
    let before = serde_json::to_value(&graph).unwrap();
    let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
    assert_eq!(serde_json::to_value(&graph).unwrap(), before);
    let outputs: HashMap<_, _> = graph
        .output_operands
        .iter()
        .map(|&id| {
            let output = &graph.operands[id as usize];
            (output.name.clone().unwrap(), output.descriptor.clone())
        })
        .collect();
    for device in [DeviceType::Cpu, DeviceType::Gpu, DeviceType::Npu] {
        for in_memory in [false, true] {
            let model = compile_model(
                converted.data.clone(),
                converted.weights_data.clone(),
                device,
                in_memory,
            )
            .unwrap();
            let CompiledCoremlModel::Pipeline(pipeline) = &model else {
                panic!("ordinary executor bypassed the source plan");
            };
            assert_eq!(
                pipeline
                    .plan
                    .stages
                    .iter()
                    .filter(|stage| matches!(stage.execution, StageExecution::ExactMatmul(_)))
                    .count(),
                1
            );
            assert!(
                pipeline
                    .plan
                    .stages
                    .iter()
                    .any(|stage| matches!(stage.execution, StageExecution::Native)),
                "the update must exercise the native consumer of the exact matrix result"
            );
            for sign in [1f32, -1.] {
                let mut state = f32_bytes(&[sign; 3]);
                let increment = f32_bytes(&[sign / 1024.; 3]);
                for step in 1..=256 {
                    let inputs = HashMap::from([
                        (
                            "state".into(),
                            CoremlByteInput {
                                data: &state,
                                descriptor: &graph.operands[0].descriptor,
                            },
                        ),
                        (
                            "increment".into(),
                            CoremlByteInput {
                                data: &increment,
                                descriptor: &graph.operands[3].descriptor,
                            },
                        ),
                    ]);
                    let actual = run_coreml_bytes(&model, &inputs, &outputs).unwrap();
                    assert_eq!(
                        f32_output_bits(&actual["projected"]),
                        dyadic_carry(sign, step - 1),
                        "projected, step{step}, sign{sign}, device{device:?}, in-memory{in_memory}"
                    );
                    assert_eq!(
                        f32_output_bits(&actual["carry"]),
                        dyadic_carry(sign, step),
                        "carry, step{step}, sign{sign}, device{device:?}, in-memory{in_memory}"
                    );
                    // Feed the actual native output back, never the oracle.
                    state = actual["carry"].clone();
                }
                // Negative control: reloading the initial state returns an
                // accurate single step, but fails the accumulated-state gate.
                let reset = f32_bytes(&[sign; 3]);
                let reset_inputs = HashMap::from([
                    (
                        "state".into(),
                        CoremlByteInput {
                            data: &reset,
                            descriptor: &graph.operands[0].descriptor,
                        },
                    ),
                    (
                        "increment".into(),
                        CoremlByteInput {
                            data: &increment,
                            descriptor: &graph.operands[3].descriptor,
                        },
                    ),
                ]);
                let reset_actual = run_coreml_bytes(&model, &reset_inputs, &outputs).unwrap();
                let reset_carry = f32_output_bits(&reset_actual["carry"]);
                assert_eq!(reset_carry, dyadic_carry(sign, 1));
                assert_ne!(reset_carry, dyadic_carry(sign, 256));
            }
        }
    }
}

#[test]
#[cfg(any(target_os = "macos", target_os = "ios"))]
fn exact_matmul_retained_storage_preserves_signed_own_state_without_input_copies() {
    // This repeats the same closed-form implementation-fidelity oracle through
    // retained native storage. It does not add a composed-graph WPT tolerance.
    let graph = recurrent_cancellation_graph();
    let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
    let descriptor = &graph.operands[0].descriptor;
    for device in [DeviceType::Cpu, DeviceType::Gpu, DeviceType::Npu] {
        for backings in [false, true] {
            let model = compile_model(
                converted.data.clone(),
                converted.weights_data.clone(),
                device,
                false,
            )
            .unwrap();
            assert!(matches!(model, CompiledCoremlModel::Pipeline(_)));
            for sign in [1f32, -1.] {
                let mut state = CoremlTensorStorage::new(DataType::Float32, 12, true).unwrap();
                let mut carry = CoremlTensorStorage::new(DataType::Float32, 12, true).unwrap();
                let mut increment = CoremlTensorStorage::new(DataType::Float32, 12, true).unwrap();
                let projected = CoremlTensorStorage::new(DataType::Float32, 12, true).unwrap();
                state.write(&f32_bytes(&[sign; 3])).unwrap();
                increment.write(&f32_bytes(&[sign / 1024.; 3])).unwrap();
                let mut statistics = crate::mlcontextoptions::CoremlTensorStatistics::default();
                for step in 1..=256 {
                    let inputs = HashMap::from([
                        (
                            "state".into(),
                            CoremlTensorBinding {
                                storage: &state,
                                descriptor,
                            },
                        ),
                        (
                            "increment".into(),
                            CoremlTensorBinding {
                                storage: &increment,
                                descriptor,
                            },
                        ),
                    ]);
                    let outputs = HashMap::from([
                        (
                            "carry".into(),
                            CoremlTensorBinding {
                                storage: &carry,
                                descriptor,
                            },
                        ),
                        (
                            "projected".into(),
                            CoremlTensorBinding {
                                storage: &projected,
                                descriptor,
                            },
                        ),
                    ]);
                    assert!(
                        run_coreml_tensors(&model, &inputs, &outputs, backings, &mut statistics)
                            .unwrap()
                            .is_empty()
                    );
                    let mut returned = [0u8; 12];
                    projected.read(&mut returned).unwrap();
                    assert_eq!(f32_output_bits(&returned), dyadic_carry(sign, step - 1));
                    carry.read(&mut returned).unwrap();
                    assert_eq!(
                        f32_output_bits(&returned),
                        dyadic_carry(sign, step),
                        "{device:?}, backings={backings}, sign={sign}, step={step}"
                    );
                    state.read(&mut returned).unwrap();
                    assert_eq!(
                        f32_output_bits(&returned),
                        dyadic_carry(sign, step - 1),
                        "destination aliased retained input"
                    );
                    // Rebind the actual output allocation, not reference bytes.
                    std::mem::swap(&mut state, &mut carry);
                }
                assert_eq!(statistics.native_input_bindings, 512);
                assert_eq!(statistics.input_copy_bytes, 0);
                assert_eq!(statistics.output_copy_bytes, 24 * 256);
                assert_eq!(statistics.output_backings_requested, 0);
                assert_eq!(statistics.output_backings_accepted, 0);
            }
        }
    }
}
