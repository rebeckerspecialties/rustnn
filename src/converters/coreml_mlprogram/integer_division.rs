//! A complete MIL expression for WebNN's signed, truncating Int32 division.

use super::*;
use crate::protos::coreml::mil_spec::{DataType as MilType, value_type};

impl CoremlMlProgramConverter {
    pub(super) fn lower_int32_division(
        graph: &LoweringGraph<'_>,
        names: &[String],
        output: NamedValueType,
        block: &mut Block,
    ) {
        let mut reserved: HashSet<_> = (0..graph.operands.len())
            .map(|id| operand_name(graph, id as u32))
            .chain(
                block
                    .operations
                    .iter()
                    .flat_map(|op| op.outputs.iter().map(|value| value.name.clone())),
            )
            .collect();
        let mut temporary = |role: &str, boolean: bool| {
            let stem = format!("{}_div_{role}", output.name);
            let mut name = stem.clone();
            let mut suffix = 0;
            while !reserved.insert(name.clone()) {
                suffix += 1;
                name = format!("{stem}_{suffix}");
            }
            let mut value = output.clone();
            value.name = name;
            if boolean
                && let Some(value_type::Type::TensorType(ty)) =
                    value.r#type.as_mut().and_then(|ty| ty.r#type.as_mut())
            {
                ty.data_type = MilType::Bool as i32;
            }
            value
        };
        let quotient = temporary("floor", false);
        let remainder = temporary("remainder", false);
        let negative = temporary("negative", true);
        let fractional = temporary("fractional", true);
        let adjustment = temporary("adjustment", true);
        let integer_adjustment = temporary("integer_adjustment", false);
        let named = |name: &str| Self::create_name_argument(name.into());
        let binary = |op: &str, x: Argument, y: Argument, output: NamedValueType| {
            Self::create_mil_operation(
                op,
                HashMap::from([("x".into(), x), ("y".into(), y)]),
                vec![output],
            )
        };
        // For nonzero divisors and representable quotients, floor(a/b) differs
        // from trunc(a/b) only when it is negative and the remainder is nonzero.
        // Keep the whole expression together: a source-proven typed stage may
        // evaluate it exactly, but unrelated MIL floor_div remains floor_div.
        block.operations.extend([
            binary(
                "floor_div",
                named(&names[0]),
                named(&names[1]),
                quotient.clone(),
            ),
            binary("mod", named(&names[0]), named(&names[1]), remainder.clone()),
            binary(
                "less",
                named(&quotient.name),
                Self::create_immediate_int(0),
                negative.clone(),
            ),
            binary(
                "not_equal",
                named(&remainder.name),
                Self::create_immediate_int(0),
                fractional.clone(),
            ),
            binary(
                "logical_and",
                named(&negative.name),
                named(&fractional.name),
                adjustment.clone(),
            ),
            Self::create_cast_operation(adjustment.name, integer_adjustment.clone(), "int32"),
            binary(
                "add",
                named(&quotient.name),
                named(&integer_adjustment.name),
                output,
            ),
        ]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::converters::GraphConverter;
    use crate::graph::{Operand, OperandDescriptor, to_dimension_vector};
    use crate::protos::coreml::specification::model;

    #[test]
    fn signed_division_has_complete_typed_expression_and_collision_safe_names() {
        let operand = |name: &str, kind| Operand {
            name: Some(name.into()),
            kind,
            descriptor: OperandDescriptor {
                data_type: DataType::Int32,
                shape: to_dimension_vector(&[2]),
                pending_permutation: vec![],
            },
        };
        let graph = GraphInfo {
            operands: vec![
                operand("result_div_floor", OperandKind::Input),
                operand("right", OperandKind::Input),
                operand("result", OperandKind::Output),
            ],
            input_operands: vec![0, 1],
            output_operands: vec![2],
            operations: vec![Operation::Div {
                a: 0,
                b: 1,
                outputs: vec![2],
                options: None,
            }],
            ..Default::default()
        };
        let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
        let model = Model::decode(converted.data.as_slice()).unwrap();
        let model::Type::MlProgram(program) = model.r#type.unwrap() else {
            panic!("standalone Int32 division must remain an MLProgram");
        };
        let function = &program.functions["main"];
        let block = &function.block_specializations[&function.opset];
        assert_eq!(
            block
                .operations
                .iter()
                .map(|op| op.r#type.as_str())
                .collect::<Vec<_>>(),
            [
                "floor_div",
                "mod",
                "less",
                "not_equal",
                "logical_and",
                "cast",
                "add"
            ]
        );
        let names = function
            .inputs
            .iter()
            .chain(block.operations.iter().flat_map(|op| &op.outputs))
            .map(|value| value.name.as_str())
            .collect::<HashSet<_>>();
        assert_eq!(names.len(), function.inputs.len() + block.operations.len());
        assert_ne!(block.operations[0].outputs[0].name, "result_div_floor");
        for (index, operation) in block.operations.iter().enumerate() {
            let value_type::Type::TensorType(ty) = operation.outputs[0]
                .r#type
                .as_ref()
                .unwrap()
                .r#type
                .as_ref()
                .unwrap()
            else {
                panic!("tensor required")
            };
            assert_eq!(
                ty.data_type,
                if [2, 3, 4].contains(&index) {
                    MilType::Bool
                } else {
                    MilType::Int32
                } as i32
            );
            assert_eq!(ty.rank, 1);
        }
    }
}
