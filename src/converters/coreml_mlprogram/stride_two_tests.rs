use super::*;
use crate::graph::Dimension as GraphDimension;
use prost::Message;

fn block_fixture(shape: &[u32]) -> (Vec<NamedValueType>, Block) {
    let (input, operation) = fixture(shape, 0, 0, MilType::Float32 as i32);
    (
        vec![NamedValueType {
            name: "x".into(),
            r#type: Some(input),
        }],
        Block {
            operations: vec![operation],
            outputs: vec!["y".into()],
            ..Default::default()
        },
    )
}

#[test]
fn rewrites_chained_slices_without_changing_public_names_or_arithmetic() {
    let (inputs, mut block) = block_fixture(&[8, 2]);
    let (_, mut second) = fixture(&[4, 2], 0, 1, MilType::Float32 as i32);
    second.inputs.insert(
        "x".into(),
        CoremlMlProgramConverter::create_name_argument("y".into()),
    );
    second.outputs[0].name = "z".into();
    let arithmetic = CoremlMlProgramConverter::create_mil_operation(
        "add",
        HashMap::from([
            (
                "x".into(),
                CoremlMlProgramConverter::create_name_argument("z".into()),
            ),
            (
                "y".into(),
                CoremlMlProgramConverter::create_name_argument("z".into()),
            ),
        ]),
        vec![NamedValueType {
            name: "z_stride2_view".into(),
            r#type: second.outputs[0].r#type.clone(),
        }],
    );
    block.operations.extend([second, arithmetic.clone()]);
    block.outputs = vec!["y".into(), "z".into(), "z_stride2_view".into()];
    let public_outputs = block.outputs.clone();
    rewrite(&inputs, &mut block);
    assert_eq!(block.operations.len(), 7);
    assert_eq!(block.outputs, public_outputs);
    assert_eq!(block.operations[2].outputs[0].name, "y");
    assert_eq!(block.operations[3].outputs[0].name, "z_stride2_view_1");
    assert_eq!(block.operations[5].outputs[0].name, "z");
    assert_eq!(block.operations[6], arithmetic);
    let revised = block.clone();
    rewrite(&inputs, &mut block);
    assert_eq!(block, revised, "rewrite must be idempotent");
}

#[test]
fn unsupported_scopes_and_shapes_remain_identical() {
    for shape in [vec![7, 2], vec![2; 5]] {
        let (inputs, mut block) = block_fixture(&shape);
        let original = block.clone();
        rewrite(&inputs, &mut block);
        assert_eq!(block, original);
    }
    let (inputs, original) = block_fixture(&[8, 2]);
    for variation in 0..5 {
        let mut block = original.clone();
        let mut inputs = inputs.clone();
        match variation {
            0 => block.operations[0].blocks.push(Block::default()),
            1 => inputs.push(inputs[0].clone()),
            2 => block.operations.push(block.operations[0].clone()),
            3 => {
                if let Some(value_type::Type::TensorType(tensor)) =
                    &mut inputs[0].r#type.as_mut().unwrap().r#type
                {
                    tensor.dimensions[0].dimension = None;
                }
            }
            4 => block.inputs.push(inputs[0].clone()),
            _ => unreachable!(),
        }
        let untouched = block.clone();
        rewrite(&inputs, &mut block);
        assert_eq!(block, untouched, "variation {variation}");
        // With one operation and no maps, protobuf encoding also remains exact.
        if variation == 0 || variation == 3 {
            assert_eq!(block.encode_to_vec(), untouched.encode_to_vec());
        }
    }
}

#[test]
fn pipeline_rewrite_preserves_children_interfaces_and_metadata() {
    use crate::protos::coreml::mil_spec::{Function, Program};
    use crate::protos::coreml::specification::{FeatureDescription, ModelDescription, Pipeline};
    let (inputs, block) = block_fixture(&[8, 2]);
    let description = ModelDescription {
        input: vec![FeatureDescription {
            name: "x".into(),
            ..Default::default()
        }],
        output: vec![FeatureDescription {
            name: "y".into(),
            ..Default::default()
        }],
        ..Default::default()
    };
    let leaf = Model {
        description: Some(description.clone()),
        r#type: Some(model::Type::MlProgram(Program {
            functions: HashMap::from([(
                "main".into(),
                Function {
                    inputs,
                    opset: "CoreML7".into(),
                    block_specializations: HashMap::from([("CoreML7".into(), block)]),
                    ..Default::default()
                },
            )]),
            ..Default::default()
        })),
        ..Default::default()
    };
    let mut pipeline = Model {
        description: Some(description),
        r#type: Some(model::Type::Pipeline(Pipeline {
            names: vec!["first".into(), "second".into()],
            models: vec![leaf.clone(), leaf],
        })),
        ..Default::default()
    };
    pipeline
        .description
        .as_mut()
        .unwrap()
        .metadata
        .get_or_insert_default()
        .user_defined
        .insert("untouched".into(), "partition inventory".into());
    let original_description = pipeline.description.clone();
    rewrite_model(&mut pipeline);
    assert_eq!(pipeline.description, original_description);
    let Some(model::Type::Pipeline(children)) = &pipeline.r#type else {
        unreachable!()
    };
    assert_eq!(children.names, ["first", "second"]);
    assert_eq!(children.models.len(), 2);
    for child in &children.models {
        assert_eq!(child.description.as_ref().unwrap().input[0].name, "x");
        assert_eq!(child.description.as_ref().unwrap().output[0].name, "y");
        let Some(model::Type::MlProgram(program)) = &child.r#type else {
            unreachable!()
        };
        let block = &program.functions["main"].block_specializations["CoreML7"];
        assert_eq!(block.outputs, ["y"]);
        assert_eq!(block.operations.len(), 3);
    }
}

#[test]
fn rejects_masks_unknown_parameters_dtypes_and_nonpositive_strides() {
    let (input, original) = fixture(&[8, 2], 0, 0, MilType::Float32 as i32);
    for key in ["begin_mask", "end_mask", "squeeze_mask", "unknown"] {
        let mut operation = original.clone();
        operation.inputs.insert(
            key.into(),
            CoremlMlProgramConverter::create_immediate_bool(true),
        );
        assert!(replacement(&operation, &input, &mut HashSet::new()).is_none());
    }
    for strides in [vec![0, 1], vec![u32::MAX, 1], vec![3, 1], vec![2, 2]] {
        let mut operation = original.clone();
        operation.inputs.insert(
            "stride".into(),
            CoremlMlProgramConverter::create_immediate_int_array(&strides),
        );
        assert!(replacement(&operation, &input, &mut HashSet::new()).is_none());
    }
    for dtype in [
        MilType::Bool,
        MilType::Int8,
        MilType::Int32,
        MilType::Uint32,
    ] {
        let (input, operation) = fixture(&[8, 2], 0, 0, dtype as i32);
        assert!(replacement(&operation, &input, &mut HashSet::new()).is_none());
    }
}

fn fixture(shape: &[u32], axis: usize, parity: u32, dtype: i32) -> (ValueType, MilOperation) {
    let input = CoremlMlProgramConverter::create_named_value_type(
        "x".into(),
        dtype,
        &shape
            .iter()
            .copied()
            .map(GraphDimension::Static)
            .collect::<Vec<_>>(),
        false,
    );
    let mut begin = vec![0; shape.len()];
    begin[axis] = parity;
    let mut stride = vec![1; shape.len()];
    stride[axis] = 2;
    let mut output_shape = shape.to_vec();
    output_shape[axis] = (shape[axis] - parity).div_ceil(2);
    let operation = CoremlMlProgramConverter::create_mil_operation(
        "slice_by_index",
        HashMap::from([
            (
                "x".into(),
                CoremlMlProgramConverter::create_name_argument("x".into()),
            ),
            (
                "begin".into(),
                CoremlMlProgramConverter::create_immediate_int_array(&begin),
            ),
            (
                "end".into(),
                CoremlMlProgramConverter::create_immediate_int_array(shape),
            ),
            (
                "stride".into(),
                CoremlMlProgramConverter::create_immediate_int_array(&stride),
            ),
        ]),
        vec![CoremlMlProgramConverter::create_named_value_type(
            "y".into(),
            dtype,
            &output_shape
                .iter()
                .copied()
                .map(GraphDimension::Static)
                .collect::<Vec<_>>(),
            false,
        )],
    );
    (input.r#type.unwrap(), operation)
}

fn ints(operation: &MilOperation, name: &str) -> Vec<usize> {
    let Some(Binding::Value(value)) = &operation.inputs[name].arguments[0].binding else {
        panic!("not literal")
    };
    let Some(value::Value::ImmediateValue(value)) = &value.value else {
        panic!("not immediate")
    };
    let Some(value::immediate_value::Value::Tensor(tensor)) = &value.value else {
        panic!("not tensor")
    };
    let Some(tensor_value::Value::Ints(values)) = &tensor.value else {
        panic!("not ints")
    };
    values
        .values
        .iter()
        .map(|&n| usize::try_from(n).unwrap())
        .collect()
}

fn interpret(operations: &[MilOperation], shape: &[u32], words: &[u32]) -> Vec<u32> {
    let mut shape = shape.iter().map(|&d| d as usize).collect::<Vec<_>>();
    let mut words = words.to_vec();
    for operation in operations {
        let new_shape = ints(
            operation,
            if operation.r#type == "reshape" {
                "shape"
            } else {
                "size"
            },
        );
        if operation.r#type == "reshape" {
            assert_eq!(
                shape.iter().product::<usize>(),
                new_shape.iter().product::<usize>()
            );
        } else {
            assert_eq!(operation.r#type, "slice_by_size");
            let begin = ints(operation, "begin");
            assert!(
                begin
                    .iter()
                    .zip(&new_shape)
                    .zip(&shape)
                    .all(|((&b, &s), &d)| b + s <= d)
            );
            words = (0..new_shape.iter().product())
                .map(|mut index| {
                    let mut coordinate = vec![0; new_shape.len()];
                    for (c, &d) in coordinate.iter_mut().zip(&new_shape).rev() {
                        *c = index % d;
                        index /= d;
                    }
                    let source = coordinate
                        .iter()
                        .zip(&begin)
                        .zip(&shape)
                        .fold(0, |position, ((c, b), d)| position * d + c + b);
                    words[source]
                })
                .collect();
        }
        shape = new_shape;
    }
    words
}

#[test]
fn preserves_opaque_payload_every_axis_dtype() {
    for shape in [vec![8], vec![2, 8], vec![2, 4, 6], vec![2, 4, 2, 6]] {
        for dtype in [MilType::Float16, MilType::Float32] {
            for axis in 0..shape.len() {
                for parity in 0..2 {
                    let (input, original) = fixture(&shape, axis, parity, dtype as i32);
                    let mut used = HashSet::from(["x".into(), "y".into(), "y_stride2_view".into()]);
                    let operations = replacement(&original, &input, &mut used).expect("eligible");
                    assert_eq!(operations.len(), 3);
                    assert_eq!(operations[0].outputs[0].name, "y_stride2_view_1");
                    assert_eq!(operations[2].outputs[0], original.outputs[0]);
                    // Treat these as opaque storage: signed zeros, subnormals,
                    // infinities and NaN payloads.
                    let patterns = match dtype {
                        MilType::Float16 => [0, 0x8000, 1, 0x8001, 0x7bff, 0x7c00, 0xfc00, 0x7e55],
                        MilType::Float32 => [
                            0, 0x80000000, 1, 0x80000001, 0x7f7fffff, 0x7f800000, 0xff800000,
                            0x7fc12345,
                        ],
                        _ => unreachable!(),
                    };
                    let words = (0..shape.iter().product::<u32>())
                        .map(|i| patterns[i as usize % patterns.len()])
                        .collect::<Vec<_>>();
                    let trailing = shape[axis + 1..].iter().product::<u32>();
                    let expected = words
                        .iter()
                        .enumerate()
                        .filter(|(index, _)| (*index as u32 / trailing) % shape[axis] % 2 == parity)
                        .map(|(_, &word)| word)
                        .collect::<Vec<_>>();
                    assert_eq!(interpret(&operations, &shape, &words), expected);
                }
            }
        }
    }
}

#[test]
fn rejects_odd_rank_bounds_and_dynamic_parameters() {
    for shape in [vec![7, 2], vec![2; 5], vec![2, 0]] {
        let (input, operation) = fixture(&shape, 0, 0, MilType::Float32 as i32);
        assert!(replacement(&operation, &input, &mut HashSet::new()).is_none());
    }
    let (input, mut operation) = fixture(&[8, 2], 0, 0, MilType::Float32 as i32);
    operation.inputs.insert(
        "end".into(),
        CoremlMlProgramConverter::create_immediate_int_array(&[6, 2]),
    );
    assert!(replacement(&operation, &input, &mut HashSet::new()).is_none());
    operation.inputs.insert(
        "end".into(),
        CoremlMlProgramConverter::create_immediate_int_array(&[8, 2]),
    );
    operation.inputs.insert(
        "stride".into(),
        CoremlMlProgramConverter::create_name_argument("dynamic".into()),
    );
    assert!(replacement(&operation, &input, &mut HashSet::new()).is_none());
}

#[test]
fn malformed_vectors_and_masks_are_not_rewritten() {
    let (input, mut operation) = fixture(&[8, 2], 0, 0, MilType::Float32 as i32);
    operation.inputs.insert(
        "stride".into(),
        CoremlMlProgramConverter::create_immediate_int_array(&[2]),
    );
    assert!(replacement(&operation, &input, &mut HashSet::new()).is_none());
    operation.inputs.insert(
        "stride".into(),
        CoremlMlProgramConverter::create_immediate_int_array(&[2, 2]),
    );
    assert!(replacement(&operation, &input, &mut HashSet::new()).is_none());
    operation.inputs.insert(
        "stride".into(),
        CoremlMlProgramConverter::create_immediate_int_array(&[2, 1]),
    );
    operation.inputs.insert(
        "squeeze_mask".into(),
        CoremlMlProgramConverter::create_immediate_bool(true),
    );
    assert!(replacement(&operation, &input, &mut HashSet::new()).is_none());
    operation.inputs.remove("squeeze_mask");
    if let Some(value_type::Type::TensorType(t)) =
        &mut operation.outputs[0].r#type.as_mut().unwrap().r#type
    {
        t.data_type = MilType::Float16 as i32;
    }
    assert!(replacement(&operation, &input, &mut HashSet::new()).is_none());
}

#[test]
fn correctly_shaped_false_masks_are_eligible_but_true_masks_are_unchanged() {
    let (input, original) = fixture(&[8, 2], 0, 0, MilType::Float32 as i32);
    for key in ["begin_mask", "end_mask", "squeeze_mask"] {
        for masked in [false, true] {
            let mut mask = CoremlMlProgramConverter::create_immediate_bool(masked);
            let Some(Binding::Value(value)) = &mut mask.arguments[0].binding else {
                unreachable!()
            };
            let Some(value_type::Type::TensorType(ty)) = &mut value.r#type.as_mut().unwrap().r#type
            else {
                unreachable!()
            };
            ty.rank = 1;
            ty.dimensions = vec![Dimension {
                dimension: Some(dimension::Dimension::Constant(
                    dimension::ConstantDimension { size: 2 },
                )),
            }];
            let Some(value::Value::ImmediateValue(immediate)) = &mut value.value else {
                unreachable!()
            };
            let Some(value::immediate_value::Value::Tensor(payload)) = &mut immediate.value else {
                unreachable!()
            };
            payload.value = Some(tensor_value::Value::Bools(tensor_value::RepeatedBools {
                values: vec![masked, false],
            }));
            let mut operation = original.clone();
            operation.inputs.insert(key.into(), mask);
            assert_eq!(
                replacement(&operation, &input, &mut HashSet::new()).is_some(),
                !masked,
                "{key}: {masked}"
            );
        }
    }
}
