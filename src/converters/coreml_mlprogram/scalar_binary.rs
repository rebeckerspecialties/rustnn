//! Adapt native scalar values at tensor-valued elementwise consumers.

use super::*;
use crate::protos::coreml::mil_spec::value_type;

fn tensor(value: &NamedValueType) -> Option<&TensorType> {
    match &value.r#type.as_ref()?.r#type {
        Some(value_type::Type::TensorType(value)) => Some(value),
        _ => None,
    }
}

impl CoremlMlProgramConverter {
    pub(super) fn adapt_scalar_binary_inputs(inputs: &[NamedValueType], block: &mut Block) {
        let types: HashMap<_, _> = inputs
            .iter()
            .chain(
                block
                    .operations
                    .iter()
                    .flat_map(|operation| &operation.outputs),
            )
            .filter_map(|value| tensor(value).map(|ty| (value.name.clone(), ty.clone())))
            .collect();
        let mut names: HashSet<_> = types.keys().cloned().collect();
        let mut views = HashMap::new();
        let operations = std::mem::take(&mut block.operations);
        for mut operation in operations {
            // Scalar constants must remain scalars: other consumers use them as
            // scalar parameters. Only adapt named tensor operands where the
            // converter declares the result using its [1] scalar convention.
            let binary = matches!(
                operation.r#type.as_str(),
                "add"
                    | "sub"
                    | "mul"
                    | "real_div"
                    | "floor_div"
                    | "pow"
                    | "maximum"
                    | "minimum"
                    | "equal"
                    | "not_equal"
                    | "greater"
                    | "greater_equal"
                    | "less"
                    | "less_equal"
                    | "logical_and"
                    | "logical_or"
                    | "logical_xor"
            );
            let all_scalar = ["x", "y"].iter().all(|key| {
                let Some(argument) = operation.inputs.get(*key) else {
                    return false;
                };
                let [binding] = argument.arguments.as_slice() else {
                    return false;
                };
                let ty = match &binding.binding {
                    Some(Binding::Name(name)) => types.get(name),
                    Some(Binding::Value(value)) => {
                        match value.r#type.as_ref().and_then(|ty| ty.r#type.as_ref()) {
                            Some(value_type::Type::TensorType(ty)) => Some(ty),
                            _ => None,
                        }
                    }
                    None => None,
                };
                ty.is_some_and(|ty| ty.rank == 0)
            });
            if binary
                && all_scalar
                && operation.outputs.iter().any(|value| {
                    tensor(value).is_some_and(|ty| {
                        ty.rank == 1
                            && matches!(ty.dimensions.as_slice(), [Dimension {
                            dimension: Some(dimension::Dimension::Constant(size)),
                        }] if size.size == 1)
                    })
                })
            {
                for key in ["x", "y"] {
                    let Some(Binding::Name(name)) = operation
                        .inputs
                        .get_mut(key)
                        .and_then(|argument| argument.arguments.first_mut())
                        .and_then(|binding| binding.binding.as_mut())
                    else {
                        continue;
                    };
                    let Some(ty) = types.get(name).filter(|ty| ty.rank == 0) else {
                        continue;
                    };
                    let view = views.entry(name.clone()).or_insert_with(|| {
                        let stem = format!("{name}_binary_scalar");
                        let mut candidate = stem.clone();
                        let mut suffix = 0;
                        while !names.insert(candidate.clone()) {
                            suffix += 1;
                            candidate = format!("{stem}_{suffix}");
                        }
                        Self::rnn_reshape(block, name, &[1], candidate, ty.data_type)
                    });
                    *name = view.clone();
                }
            }
            block.operations.push(operation);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protos::coreml::mil_spec::DataType;

    #[test]
    fn scalar_views_are_reused_without_changing_attributes_or_other_consumers() {
        let scalar = CoremlMlProgramConverter::value_type_for_static_shape(
            "value".into(),
            DataType::Float32 as i32,
            &[],
        );
        let occupied = CoremlMlProgramConverter::value_type_for_static_shape(
            "value_binary_scalar".into(),
            DataType::Float32 as i32,
            &[1],
        );
        let binary = |name: &str| {
            CoremlMlProgramConverter::create_mil_operation(
                "add",
                HashMap::from([
                    (
                        "x".into(),
                        CoremlMlProgramConverter::create_name_argument("value".into()),
                    ),
                    (
                        "y".into(),
                        CoremlMlProgramConverter::create_immediate_float(2.),
                    ),
                ]),
                vec![CoremlMlProgramConverter::value_type_for_static_shape(
                    name.into(),
                    DataType::Float32 as i32,
                    &[1],
                )],
            )
        };
        let other = CoremlMlProgramConverter::create_mil_operation(
            "clip",
            HashMap::from([
                (
                    "x".into(),
                    CoremlMlProgramConverter::create_name_argument("value_binary_scalar".into()),
                ),
                (
                    "alpha".into(),
                    CoremlMlProgramConverter::create_name_argument("value".into()),
                ),
                (
                    "beta".into(),
                    CoremlMlProgramConverter::create_immediate_float(8.),
                ),
            ]),
            vec![CoremlMlProgramConverter::value_type_for_static_shape(
                "clipped".into(),
                DataType::Float32 as i32,
                &[1],
            )],
        );
        let first = binary("sum1");
        let immediate = first.inputs["y"].clone();
        let mut block = Block {
            operations: vec![first, binary("sum2"), other.clone()],
            ..Default::default()
        };
        CoremlMlProgramConverter::adapt_scalar_binary_inputs(
            &[scalar.clone(), occupied],
            &mut block,
        );
        assert_eq!(
            block
                .operations
                .iter()
                .filter(|op| op.r#type == "reshape")
                .count(),
            1
        );
        assert_eq!(block.operations[0].outputs[0].name, "value_binary_scalar_1");
        assert_eq!(
            block.operations[1].inputs["x"],
            block.operations[2].inputs["x"]
        );
        assert_eq!(block.operations[1].inputs["y"], immediate);
        assert_eq!(block.operations[3], other);
        assert_eq!(tensor(&scalar).unwrap().rank, 0);
    }

    #[test]
    fn mixed_scalar_and_rank_one_binary_keeps_native_broadcasting() {
        let scalar = CoremlMlProgramConverter::value_type_for_static_shape(
            "scalar".into(),
            DataType::Int32 as i32,
            &[],
        );
        let vector = CoremlMlProgramConverter::value_type_for_static_shape(
            "shape".into(),
            DataType::Int32 as i32,
            &[1],
        );
        for reverse in [false, true] {
            let (x, y) = if reverse {
                ("scalar", "shape")
            } else {
                ("shape", "scalar")
            };
            let operation = CoremlMlProgramConverter::create_mil_operation(
                "sub",
                HashMap::from([
                    (
                        "x".into(),
                        CoremlMlProgramConverter::create_name_argument(x.into()),
                    ),
                    (
                        "y".into(),
                        CoremlMlProgramConverter::create_name_argument(y.into()),
                    ),
                ]),
                vec![CoremlMlProgramConverter::value_type_for_static_shape(
                    "result".into(),
                    DataType::Int32 as i32,
                    &[1],
                )],
            );
            let mut block = Block {
                operations: vec![operation.clone()],
                ..Default::default()
            };
            CoremlMlProgramConverter::adapt_scalar_binary_inputs(
                &[scalar.clone(), vector.clone()],
                &mut block,
            );
            assert_eq!(block.operations, [operation]);
        }
    }
}
