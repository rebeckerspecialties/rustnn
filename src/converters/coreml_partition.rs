//! Native MIL child partitioning at small live Float32 tensor frontiers.
//! Constants are rematerialized by reference; tensor values never visit Rust.
use super::{CoremlMlProgramConverter, coreml_packed::tensor_type};
use crate::protos::coreml::{
    mil_spec as m,
    specification::{FeatureDescription, Model, ModelDescription, Pipeline, model},
};
use crate::{
    error::GraphError,
    graph::{DataType, OperandDescriptor, to_dimension_vector},
};
use std::collections::{BTreeMap, BTreeSet};
fn error(message: impl Into<String>) -> GraphError {
    GraphError::ConversionFailed {
        format: "coreml".into(),
        reason: message.into(),
    }
}
fn names(op: &m::Operation) -> Vec<&str> {
    op.inputs
        .values()
        .flat_map(|a| &a.arguments)
        .filter_map(|a| match &a.binding {
            Some(m::argument::binding::Binding::Name(n)) => Some(n.as_str()),
            _ => None,
        })
        .collect()
}
fn feature(v: &m::NamedValueType) -> Result<FeatureDescription, GraphError> {
    let (dtype, shape) = v
        .r#type
        .as_ref()
        .and_then(tensor_type)
        .ok_or_else(|| error("dynamic/unsupported child boundary"))?;
    let dtype = match m::DataType::try_from(dtype) {
        Ok(m::DataType::Float32) => DataType::Float32,
        Ok(m::DataType::Float16) => DataType::Float16,
        Ok(m::DataType::Int32) => DataType::Int32,
        _ => return Err(error("unsupported child boundary type")),
    };
    if shape.is_empty() || shape.contains(&0) {
        return Err(error("scalar/empty child boundary"));
    }
    Ok(FeatureDescription {
        name: v.name.clone(),
        r#type: Some(CoremlMlProgramConverter::create_feature_type(
            &OperandDescriptor {
                data_type: dtype,
                shape: to_dimension_vector(&shape),
                pending_permutation: vec![],
            },
        )?),
        ..Default::default()
    })
}
fn attribute_storage_bytes(op: &m::Operation) -> Result<u64, GraphError> {
    op.attributes.values().try_fold(0u64, |total, value| {
        let Some((dtype, shape)) = value.r#type.as_ref().and_then(tensor_type) else {
            return Ok(total);
        };
        let bits = match m::DataType::try_from(dtype) {
            Ok(m::DataType::Uint1) => 1,
            Ok(m::DataType::Uint2) => 2,
            Ok(m::DataType::Uint3) => 3,
            Ok(m::DataType::Int4 | m::DataType::Uint4) => 4,
            Ok(m::DataType::Uint6) => 6,
            Ok(
                m::DataType::Bool
                | m::DataType::Int8
                | m::DataType::Uint8
                | m::DataType::Float8e4m3fn
                | m::DataType::Float8e5m2,
            ) => 8,
            Ok(
                m::DataType::Float16
                | m::DataType::Bfloat16
                | m::DataType::Int16
                | m::DataType::Uint16,
            ) => 16,
            Ok(m::DataType::Float32 | m::DataType::Int32 | m::DataType::Uint32) => 32,
            Ok(m::DataType::Float64 | m::DataType::Int64 | m::DataType::Uint64) => 64,
            // String attributes include operation names and dtype selectors.
            _ => return Ok(total),
        };
        let bits = shape
            .iter()
            .try_fold(bits, |n: u64, &d| n.checked_mul(d.into()))
            .ok_or_else(|| error("constant extent overflow"))?;
        total
            .checked_add(bits.div_ceil(8))
            .ok_or_else(|| error("constant extent overflow"))
    })
}
/// Split one static program into native children. `target_operations` counts
/// dynamic operations; it is a scheduling target, not a physical-memory bound.
/// Small programs are left unchanged. No cuts pass Bool or packed-weight views.
pub fn split(model: &mut Model, target_operations: usize) -> Result<usize, GraphError> {
    if target_operations < 16 {
        return Err(error("child operation target must be at least 16"));
    }
    let Some(model::Type::MlProgram(program)) = &model.r#type else {
        return Ok(0);
    };
    if program.functions.len() != 1 {
        return Err(error("partitioning requires one main function"));
    }
    let f = program
        .functions
        .get("main")
        .ok_or_else(|| error("missing main function"))?;
    if f.block_specializations.len() != 1 {
        return Err(error("partitioning requires one specialization"));
    }
    let block = f
        .block_specializations
        .get(&f.opset)
        .ok_or_else(|| error("missing active block"))?;
    if block.operations.iter().any(|o| !o.blocks.is_empty()) {
        return Err(error("partition before introducing native control flow"));
    }
    let mut types = f
        .inputs
        .iter()
        .map(|v| (v.name.clone(), v.clone()))
        .collect::<BTreeMap<_, _>>();
    let mut producers = BTreeMap::new();
    let mut constants = BTreeSet::new();
    let mut last = BTreeMap::new();
    for (index, op) in block.operations.iter().enumerate() {
        let deps = names(op);
        if op.r#type == "const"
            || ((op.r#type.starts_with("constexpr_")
                || matches!(
                    op.r#type.as_str(),
                    "cast"
                        | "abs"
                        | "relu"
                        | "reduce_max"
                        | "reduce_sum"
                        | "reshape"
                        | "transpose"
                        | "add"
                        | "sub"
                        | "mul"
                        | "identity"
                        | "concat"
                        | "slice_by_size"
                ))
                && deps
                    .iter()
                    .all(|n| producers.get(*n).is_some_and(|p| constants.contains(p))))
        {
            constants.insert(index);
        }
        for name in deps {
            if !types.contains_key(name) {
                return Err(error(format!("non-topological dependency {name}")));
            }
            last.insert(name.to_string(), index);
        }
        for v in &op.outputs {
            if types.insert(v.name.clone(), v.clone()).is_some() {
                return Err(error(format!("redefined MIL value {}", v.name)));
            }
            producers.insert(v.name.clone(), index);
        }
    }
    let dynamic = block.operations.len() - constants.len();
    if dynamic <= target_operations {
        return Ok(1);
    }
    let final_names = block.outputs.iter().cloned().collect::<BTreeSet<_>>();
    // Each live interval contributes two events, avoiding a quadratic scan
    // over the model while selecting every possible child boundary.
    let mut events = vec![(0i128, 0i64); block.operations.len() + 2];
    for (name, &producer) in &producers {
        if constants.contains(&producer) {
            continue;
        }
        let Some(&used) = last.get(name).filter(|&&used| used > producer) else {
            continue;
        };
        let bytes = (|| {
            feature(&types[name]).ok()?;
            let (_, shape) = tensor_type(types[name].r#type.as_ref()?)?;
            shape.iter().try_fold(4u64, |a, &b| a.checked_mul(b.into()))
        })();
        match bytes {
            Some(bytes) => {
                events[producer + 1].0 += i128::from(bytes);
                events[used + 1].0 -= i128::from(bytes);
            }
            None => {
                events[producer + 1].1 += 1;
                events[used + 1].1 -= 1;
            }
        }
    }
    let (mut bytes, mut invalid) = (0i128, 0i64);
    let mut costs = Vec::with_capacity(events.len());
    for (delta, bad) in events {
        bytes += delta;
        invalid += bad;
        costs.push(if invalid == 0 {
            u64::try_from(bytes).ok()
        } else {
            None
        });
    }
    let mut root_sets = Vec::<BTreeSet<usize>>::new();
    let mut root_bytes = BTreeMap::new();
    for (i, op) in block.operations.iter().enumerate() {
        let mut roots = BTreeSet::new();
        if op.r#type == "const" || op.r#type.starts_with("constexpr_") {
            // Older constexpr operations carry their packed leaves as typed
            // attributes, rather than named const inputs. Count those leaves
            // too: otherwise a compressed LM head bypasses the weight budget
            // and several large projections collapse into one native child.
            let bytes = attribute_storage_bytes(op)?;
            if bytes >= 64 {
                root_bytes.insert(i, bytes);
                roots.insert(i);
            }
        }
        for name in names(op) {
            if let Some(&p) = producers.get(name)
                && constants.contains(&p)
            {
                roots.extend(root_sets[p].iter().copied());
            }
        }
        root_sets.push(roots);
    }
    let mut ranges = Vec::new();
    let mut start = 0;
    while start < block.operations.len() {
        let mut count = 0;
        let mut weight_bytes = 0;
        let mut weights = BTreeSet::new();
        let mut best = None::<(u64, usize)>;
        let mut end = block.operations.len();
        for i in start..block.operations.len() {
            if !constants.contains(&i) {
                let added = root_sets[i]
                    .iter()
                    .filter(|p| !weights.contains(*p))
                    .map(|p| root_bytes[p])
                    .sum::<u64>();
                if weight_bytes + added > 6 * 1024 * 1024
                    && count > 0
                    && let Some((_, cut)) = best
                {
                    end = cut;
                    break;
                }
                weights.extend(root_sets[i].iter().copied());
                weight_bytes += added;
                count += 1;
            }
            if count > 0
                && let Some(cost) = costs[i + 1]
                && (count >= target_operations / 2 || weight_bytes >= 3 * 1024 * 1024)
                && best.is_none_or(|b| cost <= b.0)
            {
                best = Some((cost, i + 1));
            }
            if count >= target_operations
                && let Some((_, cut)) = best
            {
                end = cut;
                break;
            }
        }
        if end <= start {
            return Err(error("no legal child frontier"));
        }
        ranges.push((start, end));
        start = end;
    }
    let mut children = Vec::new();
    for (start, end) in ranges {
        let mut selected = (start..end)
            .filter(|i| !constants.contains(i))
            .collect::<BTreeSet<_>>();
        // Original graph outputs that are constant still need an owning child.
        for n in &final_names {
            if let Some(&i) = producers.get(n)
                && (start..end).contains(&i)
            {
                selected.insert(i);
            }
        }
        if selected.is_empty() {
            continue;
        }
        let mut stack = selected.iter().copied().collect::<Vec<_>>();
        let mut inputs = BTreeSet::new();
        while let Some(i) = stack.pop() {
            for n in names(&block.operations[i]) {
                match producers.get(n) {
                    Some(&p) if constants.contains(&p) => {
                        if selected.insert(p) {
                            stack.push(p)
                        }
                    }
                    Some(&p) if (start..end).contains(&p) => {}
                    _ => {
                        inputs.insert(n.to_string());
                    }
                }
            }
        }
        let outputs = selected
            .iter()
            .flat_map(|&i| &block.operations[i].outputs)
            .filter(|v| {
                final_names.contains(&v.name)
                    || (!constants.contains(&producers[&v.name])
                        && last.get(&v.name).is_some_and(|&i| i >= end))
            })
            .map(|v| v.name.clone())
            .collect::<BTreeSet<_>>();
        let input_values = inputs.iter().map(|n| types[n].clone()).collect::<Vec<_>>();
        let desc = ModelDescription {
            input: input_values
                .iter()
                .map(feature)
                .collect::<Result<Vec<_>, _>>()?,
            output: outputs
                .iter()
                .map(|n| feature(&types[n]))
                .collect::<Result<Vec<_>, _>>()?,
            ..Default::default()
        };
        let mut child_function = f.clone();
        child_function.inputs = input_values;
        child_function.block_specializations.insert(
            f.opset.clone(),
            m::Block {
                operations: selected
                    .into_iter()
                    .map(|i| block.operations[i].clone())
                    .collect(),
                outputs: outputs.into_iter().collect(),
                ..Default::default()
            },
        );
        let child_program = m::Program {
            functions: [("main".into(), child_function)].into(),
            version: program.version,
            ..Default::default()
        };
        children.push(Model {
            specification_version: model.specification_version,
            description: Some(desc),
            r#type: Some(model::Type::MlProgram(child_program)),
            ..Default::default()
        });
    }
    let count = children.len();
    model.r#type = Some(model::Type::Pipeline(Pipeline {
        models: children,
        names: vec![],
    }));
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::converters::coreml_packed::{immediate, named_type, reference};

    fn blob(dtype: m::DataType, shape: &[u32], offset: u64) -> m::Value {
        m::Value {
            r#type: named_type("weight", dtype, shape).r#type,
            value: Some(m::value::Value::BlobFileValue(m::value::BlobFileValue {
                file_name: "@model_path/weights/weight.bin".into(),
                offset,
            })),
            ..Default::default()
        }
    }
    fn scalar(dtype: m::DataType, payload: m::tensor_value::Value) -> m::Value {
        let arg = immediate(dtype, &[], payload);
        let Some(m::argument::binding::Binding::Value(value)) =
            arg.arguments.into_iter().next().unwrap().binding
        else {
            panic!()
        };
        value
    }
    fn chain(affine: bool) -> Model {
        let input = named_type("x", m::DataType::Float32, &[1, 2048]);
        let mut operations = vec![];
        let mut current = "x".to_string();
        for layer in 0..12 {
            let q = format!("q{layer}");
            let w = format!("w{layer}");
            let packed = blob(m::DataType::Int8, &[2048, 2048], layer * 4_194_368);
            if affine {
                operations.push(m::Operation {
                    r#type: "constexpr_affine_dequantize".into(),
                    attributes: [
                        ("quantized_data".into(), packed),
                        (
                            "scale".into(),
                            scalar(
                                m::DataType::Float32,
                                m::tensor_value::Value::Floats(m::tensor_value::RepeatedFloats {
                                    values: vec![1.],
                                }),
                            ),
                        ),
                        (
                            "zero_point".into(),
                            scalar(
                                m::DataType::Int8,
                                m::tensor_value::Value::Bytes(m::tensor_value::RepeatedBytes {
                                    values: vec![0].into(),
                                }),
                            ),
                        ),
                        (
                            "axis".into(),
                            scalar(
                                m::DataType::Int32,
                                m::tensor_value::Value::Ints(m::tensor_value::RepeatedInts {
                                    values: vec![0],
                                }),
                            ),
                        ),
                    ]
                    .into(),
                    outputs: vec![named_type(&w, m::DataType::Float32, &[2048, 2048])],
                    ..Default::default()
                });
            } else {
                operations.push(m::Operation {
                    r#type: "const".into(),
                    attributes: [("val".into(), packed)].into(),
                    outputs: vec![named_type(&q, m::DataType::Int8, &[2048, 2048])],
                    ..Default::default()
                });
                operations.push(m::Operation {
                    r#type: "cast".into(),
                    inputs: [
                        ("x".into(), reference(&q)),
                        (
                            "dtype".into(),
                            immediate(
                                m::DataType::String,
                                &[],
                                m::tensor_value::Value::Strings(m::tensor_value::RepeatedStrings {
                                    values: vec!["fp32".into()],
                                }),
                            ),
                        ),
                    ]
                    .into(),
                    outputs: vec![named_type(&w, m::DataType::Float32, &[2048, 2048])],
                    ..Default::default()
                });
            }
            let output = format!("product{layer}");
            operations.push(m::Operation {
                r#type: "matmul".into(),
                inputs: [
                    ("x".into(), reference(&current)),
                    ("y".into(), reference(&w)),
                ]
                .into(),
                outputs: vec![named_type(&output, m::DataType::Float32, &[1, 2048])],
                ..Default::default()
            });
            current = output;
            for step in 0..18 {
                let output = format!("relu{layer}_{step}");
                operations.push(m::Operation {
                    r#type: "relu".into(),
                    inputs: [("x".into(), reference(&current))].into(),
                    outputs: vec![named_type(&output, m::DataType::Float32, &[1, 2048])],
                    ..Default::default()
                });
                current = output;
            }
        }
        let output = named_type(&current, m::DataType::Float32, &[1, 2048]);
        let function = m::Function {
            inputs: vec![input.clone()],
            opset: "CoreML8".into(),
            block_specializations: [(
                "CoreML8".into(),
                m::Block {
                    operations,
                    outputs: vec![current],
                    ..Default::default()
                },
            )]
            .into(),
            ..Default::default()
        };
        Model {
            specification_version: 9,
            description: Some(ModelDescription {
                input: vec![feature(&input).unwrap()],
                output: vec![feature(&output).unwrap()],
                ..Default::default()
            }),
            r#type: Some(model::Type::MlProgram(m::Program {
                version: 1,
                functions: [("main".into(), function)].into(),
                ..Default::default()
            })),
            ..Default::default()
        }
    }
    #[test]
    fn packed_attribute_leaves_obey_the_same_child_budget_as_named_constants() {
        let mut named = chain(false);
        let mut attributes = chain(true);
        assert_eq!(split(&mut named, 128).unwrap(), 12);
        assert_eq!(split(&mut attributes, 128).unwrap(), 12);
        let children = |model: Model| match model.r#type.unwrap() {
            model::Type::Pipeline(p) => p.models,
            _ => panic!(),
        };
        for (left, right) in children(named).into_iter().zip(children(attributes)) {
            assert_eq!(left.description, right.description);
        }
    }
    #[test]
    fn stored_dtype_width_and_overflow_are_accounted_without_expanding_weights() {
        let operation = |dtype, shape: &[u32]| m::Operation {
            r#type: "const".into(),
            attributes: [("val".into(), blob(dtype, shape, 0))].into(),
            ..Default::default()
        };
        assert_eq!(
            attribute_storage_bytes(&operation(m::DataType::Int8, &[18, 32, 4096])).unwrap(),
            18 * 32 * 4096
        );
        assert_eq!(
            attribute_storage_bytes(&operation(m::DataType::Float16, &[128])).unwrap(),
            256
        );
        assert_eq!(
            attribute_storage_bytes(&operation(m::DataType::Int4, &[3])).unwrap(),
            2
        );
        assert_eq!(
            attribute_storage_bytes(&operation(m::DataType::Float32, &[])).unwrap(),
            4
        );
        assert!(
            attribute_storage_bytes(&operation(m::DataType::Float64, &[1 << 30, 1 << 30, 64]))
                .is_err()
        );
    }
}
