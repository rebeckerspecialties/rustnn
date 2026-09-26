//! Shape-only bounds checks for slices whose extents vary between dispatches.
//!
//! Keep only input-axis bindings and slice metadata, not constants or model
//! weights. This deliberately does not evaluate expressions in dimension names
//! or read tensor contents; operand-driven shape evaluation is a separate API.

use std::collections::HashMap;

use crate::error::GraphError;
use crate::graph::{Dimension, GraphInfo, OperandDescriptor};
use crate::operators::Operation;
use crate::shape_inference::infer_slice_shape;

#[derive(Debug)]
enum Extent {
    Fixed(u32),
    InputAxis { name: String, axis: usize, max: u32 },
}

fn error(reason: impl Into<String>) -> GraphError {
    GraphError::ShapeInferenceFailed {
        reason: reason.into(),
    }
}

impl Extent {
    fn bind(
        dimension: &Dimension,
        bindings: &HashMap<String, (String, usize)>,
    ) -> Result<Self, GraphError> {
        match dimension {
            Dimension::Static(value) => Ok(Self::Fixed(*value)),
            Dimension::Dynamic(dynamic) => {
                let (name, axis) = bindings
                    .get(&dynamic.name)
                    .filter(|_| !dynamic.name.is_empty())
                    .ok_or_else(|| {
                        error(format!(
                            "CoreML slice dimension {:?} is not bound to a graph input shape",
                            dynamic.name
                        ))
                    })?;
                Ok(Self::InputAxis {
                    name: name.clone(),
                    axis: *axis,
                    max: dynamic.max_size,
                })
            }
        }
    }

    fn resolve(&self, inputs: &HashMap<String, OperandDescriptor>) -> Result<u32, GraphError> {
        match self {
            Self::Fixed(value) => Ok(*value),
            Self::InputAxis { name, axis, max } => {
                let Some(Dimension::Static(value)) =
                    inputs.get(name).and_then(|d| d.shape.get(*axis))
                else {
                    return Err(error(format!(
                        "CoreML slice needs concrete input {name:?} axis {axis}"
                    )));
                };
                if value > max {
                    return Err(error(format!(
                        "CoreML slice input {name:?} axis {axis} extent {value} exceeds bound {max}"
                    )));
                }
                Ok(*value)
            }
        }
    }
}

#[derive(Debug)]
struct SliceConstraint {
    operation: usize,
    input: Vec<Extent>,
    output: Vec<Extent>,
    starts: Vec<u32>,
    sizes: Vec<Extent>,
    strides: Vec<u32>,
}

#[derive(Debug, Default)]
pub(super) struct SliceConstraints(Vec<SliceConstraint>);

impl SliceConstraints {
    pub(super) fn new(graph: &GraphInfo) -> Result<Self, GraphError> {
        let mut bindings = HashMap::new();
        for &id in &graph.input_operands {
            let operand = graph
                .operand(id)
                .ok_or_else(|| error(format!("missing input operand {id}")))?;
            for (axis, dimension) in operand.descriptor.shape.iter().enumerate() {
                if let Dimension::Dynamic(dynamic) = dimension {
                    let name = operand
                        .name
                        .as_ref()
                        .ok_or(GraphError::MissingInputName { operand: id })?;
                    bindings
                        .entry(dynamic.name.clone())
                        .or_insert_with(|| (name.clone(), axis));
                }
            }
        }
        let bind_shape = |shape: &[Dimension]| -> Result<Vec<Extent>, GraphError> {
            shape
                .iter()
                .map(|dimension| Extent::bind(dimension, &bindings))
                .collect()
        };
        let mut constraints = Vec::new();
        for (operation, op) in graph.operations.iter().enumerate() {
            let Operation::Slice {
                input,
                starts,
                sizes,
                options,
                outputs,
            } = op
            else {
                continue;
            };
            let input = graph
                .operand(*input)
                .ok_or_else(|| error("CoreML slice input is missing"))?;
            let size_dimensions: Vec<Dimension> = sizes.iter().cloned().map(Into::into).collect();
            if !input
                .descriptor
                .shape
                .iter()
                .chain(&size_dimensions)
                .any(|d| matches!(d, Dimension::Dynamic(_)))
            {
                continue;
            }
            let output = outputs
                .first()
                .and_then(|id| graph.operand(*id))
                .ok_or_else(|| error("CoreML slice output is missing"))?;
            constraints.push(SliceConstraint {
                operation,
                input: bind_shape(&input.descriptor.shape)?,
                output: bind_shape(&output.descriptor.shape)?,
                starts: starts.clone(),
                sizes: bind_shape(&size_dimensions)?,
                strides: options
                    .as_ref()
                    .map(|options| options.strides.clone())
                    .unwrap_or_default(),
            });
        }
        Ok(Self(constraints))
    }

    pub(super) fn validate(
        &self,
        inputs: &HashMap<String, OperandDescriptor>,
    ) -> Result<(), GraphError> {
        let resolve = |shape: &[Extent]| -> Result<Vec<u32>, GraphError> {
            shape.iter().map(|extent| extent.resolve(inputs)).collect()
        };
        for constraint in &self.0 {
            let input = resolve(&constraint.input)?;
            let sizes = resolve(&constraint.sizes)?;
            let expected = resolve(&constraint.output)?;
            let actual = infer_slice_shape(
                &input,
                &constraint.starts,
                &sizes,
                Some(&constraint.strides),
            )
            .map_err(|source| {
                error(format!(
                    "CoreML slice operation {}: {source}",
                    constraint.operation
                ))
            })?;
            if actual != expected {
                return Err(error(format!(
                    "CoreML slice operation {} produces shape {actual:?}, descriptor expects {expected:?}",
                    constraint.operation
                )));
            }
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "dynamic-inputs"))]
mod tests {
    use super::*;
    use crate::graph::{DataType, DynamicDimension, Operand, OperandKind};
    use crate::operator_options::{MLDimension, MLDynamicDimension};

    fn dynamic() -> Dimension {
        Dimension::Dynamic(DynamicDimension {
            name: "sequence".to_string(),
            max_size: 8,
        })
    }

    fn descriptor(shape: Vec<Dimension>) -> OperandDescriptor {
        OperandDescriptor {
            data_type: DataType::Float32,
            shape,
            pending_permutation: vec![],
        }
    }

    fn graph(size: MLDimension) -> GraphInfo {
        GraphInfo {
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    name: Some("input".to_string()),
                    descriptor: descriptor(vec![dynamic()]),
                },
                Operand {
                    kind: OperandKind::Output,
                    name: Some("output".to_string()),
                    descriptor: descriptor(vec![size.clone().into()]),
                },
            ],
            input_operands: vec![0],
            output_operands: vec![1],
            operations: vec![Operation::Slice {
                input: 0,
                starts: vec![0],
                sizes: vec![size],
                options: None,
                outputs: vec![1],
            }],
            ..Default::default()
        }
    }

    fn inputs(size: u32) -> HashMap<String, OperandDescriptor> {
        HashMap::from([(
            "input".to_string(),
            descriptor(vec![Dimension::Static(size)]),
        )])
    }

    #[test]
    fn runtime_slice_constraints_distinguish_fixed_and_dynamic_sizes() {
        let full = SliceConstraints::new(&graph(MLDimension::Dynamic(MLDynamicDimension {
            name: "sequence".to_string(),
            max_size: 8,
        })))
        .unwrap();
        for size in [1, 4, 2, 1] {
            full.validate(&inputs(size)).unwrap();
        }
        let fixed_eight = SliceConstraints::new(&graph(MLDimension::Static(8))).unwrap();
        fixed_eight.validate(&inputs(8)).unwrap();
        assert!(
            fixed_eight
                .validate(&inputs(4))
                .unwrap_err()
                .to_string()
                .contains("CoreML slice operation 0")
        );
        let fixed_two = SliceConstraints::new(&graph(MLDimension::Static(2))).unwrap();
        fixed_two.validate(&inputs(4)).unwrap();
        fixed_two.validate(&inputs(2)).unwrap();
        assert!(fixed_two.validate(&inputs(1)).is_err());
    }

    #[test]
    fn runtime_slice_constraints_resolve_independent_size_inputs() {
        let mut info = graph(MLDimension::Dynamic(MLDynamicDimension {
            name: "window".to_string(),
            max_size: 8,
        }));
        info.operands.push(Operand {
            kind: OperandKind::Input,
            name: Some("size_source".to_string()),
            descriptor: descriptor(vec![Dimension::Dynamic(DynamicDimension {
                name: "window".to_string(),
                max_size: 8,
            })]),
        });
        info.input_operands.push(2);
        if let Operation::Slice { starts, .. } = &mut info.operations[0] {
            starts[0] = 1;
        }
        let constraints = SliceConstraints::new(&info).unwrap();
        let mut active = inputs(8);
        assert!(constraints.validate(&active).is_err());
        for (window, valid) in [(2, true), (7, true), (8, false), (9, false)] {
            active.insert(
                "size_source".to_string(),
                descriptor(vec![Dimension::Static(window)]),
            );
            assert_eq!(constraints.validate(&active).is_ok(), valid);
        }
    }

    #[test]
    fn runtime_slice_constraints_keep_fixed_strided_window_semantics() {
        let mut info = graph(MLDimension::Static(4));
        info.operands[1].descriptor.shape = vec![Dimension::Static(2)];
        if let Operation::Slice {
            starts, options, ..
        } = &mut info.operations[0]
        {
            starts[0] = 1;
            *options = Some(crate::operator_options::MLSliceOptions {
                strides: vec![2],
                ..Default::default()
            });
        }
        let constraints = SliceConstraints::new(&info).unwrap();
        constraints.validate(&inputs(5)).unwrap();
        assert!(constraints.validate(&inputs(4)).is_err());
    }

    #[test]
    fn runtime_slice_constraints_reject_unknown_labels_and_inconsistent_outputs() {
        let mut info = graph(MLDimension::Dynamic(MLDynamicDimension {
            name: "sequence + 1".to_string(),
            max_size: 8,
        }));
        assert!(
            SliceConstraints::new(&info)
                .unwrap_err()
                .to_string()
                .contains("not bound")
        );
        info = graph(MLDimension::Static(2));
        info.operands[1].descriptor.shape = vec![dynamic()];
        assert!(
            SliceConstraints::new(&info)
                .unwrap()
                .validate(&inputs(4))
                .unwrap_err()
                .to_string()
                .contains("descriptor expects")
        );
        if let Operation::Slice { starts, .. } = &mut info.operations[0] {
            starts[0] = u32::MAX;
        }
        assert!(
            SliceConstraints::new(&info)
                .unwrap()
                .validate(&inputs(4))
                .is_err()
        );
    }
}
