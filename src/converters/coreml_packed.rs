//! Bounded native expansion of unit-scale packed Int8 matrix operands.
//!
//! This pass changes column scheduling only. It keeps the original contraction
//! width, Float32 output, packed bytes, scales and downstream correction graph.
//! It does not add an arithmetic executor or read any runtime tensor. Callers
//! must separately qualify the surrounding lowering and target memory behavior.

use crate::protos::coreml::mil_spec as mil;
use crate::protos::coreml::specification::{Model, model};
use std::collections::{BTreeMap, BTreeSet};

/// A native matrix product whose packed operand was moved into a bounded loop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedMatmulExpansion {
    /// Original result name, unchanged by the rewrite.
    pub output: String,
    /// Number of independent width-32 contraction groups.
    pub groups: u32,
    /// Output columns evaluated in one iteration.
    pub tile_columns: u32,
    /// Number of iterations, with no partial final tile.
    pub iterations: u32,
    /// Logical Float32 weight expansion per iteration, not process memory.
    pub expanded_tile_bytes: u64,
}

/// Replace single-consumer, unit-scale Int8 affine leaves feeding rank-3
/// Float32 matmul with a native MIL loop that expands one column tile at a time.
///
/// Admission requires static positive shapes, K=32, matching batch dimensions,
/// no transposition, zero Int8 zero-point, and exactly unit Float32 scale.
/// Other operations are left unchanged. A budget too small for one column also
/// leaves that product unchanged; the returned records identify what changed.
/// This is an explicit lowering utility, not enabled by the default converter.
/// Changed functions use CoreML8 (`slice_update`), requiring iOS/tvOS 18,
/// watchOS 11, macOS 15, or visionOS 2 and newer.
pub fn bound_packed_matmuls(
    model: &mut Model,
    max_expanded_bytes: u64,
) -> Vec<PackedMatmulExpansion> {
    let mut changes = Vec::new();
    visit_model(model, max_expanded_bytes, &mut changes);
    changes
}

fn visit_model(model: &mut Model, cap: u64, changes: &mut Vec<PackedMatmulExpansion>) {
    let before = changes.len();
    match model.r#type.as_mut() {
        Some(model::Type::Pipeline(pipeline)) => {
            for child in &mut pipeline.models {
                visit_model(child, cap, changes);
            }
        }
        Some(model::Type::MlProgram(program)) => {
            for function in program.functions.values_mut() {
                // Do not silently rewrite just one branch of a multi-opset
                // function. slice_update needs CoreML8 even when the incoming
                // converter program used the earlier CoreML7 specialization.
                let Some(version) = function
                    .opset
                    .strip_prefix("CoreML")
                    .and_then(|v| v.parse::<u32>().ok())
                else {
                    continue;
                };
                if function.block_specializations.len() != 1 || !(5..=8).contains(&version) {
                    continue;
                }
                let Some(block) = function.block_specializations.get_mut(&function.opset) else {
                    continue;
                };
                let function_before = changes.len();
                rewrite_block(block, &function.inputs, cap, changes);
                if changes.len() != function_before && version < 8 {
                    let block = function
                        .block_specializations
                        .remove(&function.opset)
                        .unwrap();
                    function.opset = "CoreML8".into();
                    function
                        .block_specializations
                        .insert(function.opset.clone(), block);
                }
            }
        }
        _ => {}
    }
    if changes.len() != before {
        model.specification_version = model.specification_version.max(9);
    }
}

pub(super) fn tensor_type(value: &mil::ValueType) -> Option<(i32, Vec<u32>)> {
    let mil::value_type::Type::TensorType(tensor) = value.r#type.as_ref()? else {
        return None;
    };
    if tensor.rank < 0 || tensor.rank as usize != tensor.dimensions.len() {
        return None;
    }
    let shape = tensor
        .dimensions
        .iter()
        .map(|dim| match &dim.dimension {
            Some(mil::dimension::Dimension::Constant(n)) => {
                u32::try_from(n.size).ok().filter(|&n| n <= i32::MAX as u32)
            }
            _ => None,
        })
        .collect::<Option<Vec<_>>>()?;
    Some((tensor.data_type, shape))
}

pub(super) fn named_type(name: &str, dtype: mil::DataType, shape: &[u32]) -> mil::NamedValueType {
    mil::NamedValueType {
        name: name.into(),
        r#type: Some(mil::ValueType {
            r#type: Some(mil::value_type::Type::TensorType(mil::TensorType {
                data_type: dtype as i32,
                rank: shape.len() as i64,
                dimensions: shape
                    .iter()
                    .map(|&size| mil::Dimension {
                        dimension: Some(mil::dimension::Dimension::Constant(
                            mil::dimension::ConstantDimension { size: size.into() },
                        )),
                    })
                    .collect(),
                ..Default::default()
            })),
        }),
    }
}

pub(super) fn reference(name: &str) -> mil::Argument {
    mil::Argument {
        arguments: vec![mil::argument::Binding {
            binding: Some(mil::argument::binding::Binding::Name(name.into())),
        }],
    }
}

pub(super) fn immediate(
    dtype: mil::DataType,
    shape: &[u32],
    payload: mil::tensor_value::Value,
) -> mil::Argument {
    mil::Argument {
        arguments: vec![mil::argument::Binding {
            binding: Some(mil::argument::binding::Binding::Value(mil::Value {
                r#type: named_type("", dtype, shape).r#type,
                value: Some(mil::value::Value::ImmediateValue(
                    mil::value::ImmediateValue {
                        value: Some(mil::value::immediate_value::Value::Tensor(
                            mil::TensorValue {
                                value: Some(payload),
                            },
                        )),
                    },
                )),
                ..Default::default()
            })),
        }],
    }
}

fn ints(values: &[u32], scalar: bool) -> mil::Argument {
    let shape = if scalar {
        vec![]
    } else {
        vec![values.len() as u32]
    };
    immediate(
        mil::DataType::Int32,
        &shape,
        mil::tensor_value::Value::Ints(mil::tensor_value::RepeatedInts {
            values: values.iter().map(|&v| v as i32).collect(),
        }),
    )
}

fn boolean(value: bool) -> mil::Argument {
    immediate(
        mil::DataType::Bool,
        &[],
        mil::tensor_value::Value::Bools(mil::tensor_value::RepeatedBools {
            values: vec![value],
        }),
    )
}

fn name_arg<'a>(op: &'a mil::Operation, key: &str) -> Option<&'a str> {
    let [arg] = op.inputs.get(key)?.arguments.as_slice() else {
        return None;
    };
    match arg.binding.as_ref()? {
        mil::argument::binding::Binding::Name(name) => Some(name),
        _ => None,
    }
}

fn payload(value: &mil::Value) -> Option<&mil::tensor_value::Value> {
    let mil::value::Value::ImmediateValue(value) = value.value.as_ref()? else {
        return None;
    };
    let mil::value::immediate_value::Value::Tensor(value) = value.value.as_ref()? else {
        return None;
    };
    value.value.as_ref()
}

fn constant<'a>(
    op: &'a mil::Operation,
    key: &str,
    producers: &BTreeMap<&str, &'a mil::Operation>,
) -> Option<&'a mil::Value> {
    let [arg] = op.inputs.get(key)?.arguments.as_slice() else {
        return None;
    };
    match arg.binding.as_ref()? {
        mil::argument::binding::Binding::Value(value) => Some(value),
        mil::argument::binding::Binding::Name(name) => {
            let source = producers.get(name.as_str())?;
            (source.r#type == "const")
                .then(|| source.attributes.get("val"))
                .flatten()
        }
    }
}

fn counts(block: &mil::Block, uses: &mut BTreeMap<String, usize>, names: &mut BTreeSet<String>) {
    for input in &block.inputs {
        names.insert(input.name.clone());
    }
    for output in &block.outputs {
        *uses.entry(output.clone()).or_default() += 1;
    }
    for op in &block.operations {
        for out in &op.outputs {
            names.insert(out.name.clone());
        }
        for arg in op.inputs.values().flat_map(|arg| &arg.arguments) {
            if let Some(mil::argument::binding::Binding::Name(name)) = &arg.binding {
                *uses.entry(name.clone()).or_default() += 1;
            }
        }
        for child in &op.blocks {
            counts(child, uses, names);
        }
    }
}

fn tile_columns(groups: u32, columns: u32, cap: u64) -> Option<u32> {
    if groups == 0 || columns == 0 {
        return None;
    }
    let per_column = u64::from(groups).checked_mul(32 * 4)?;
    let upper = (cap / per_column).min(u64::from(columns)) as u32;
    if upper == 0 {
        return None;
    }
    if upper == columns {
        return Some(columns);
    }
    // Enumerate divisors in O(sqrt(N)), avoiding an unbounded reverse scan.
    let mut best = 1;
    let mut divisor = 1;
    while divisor <= columns / divisor {
        if columns.is_multiple_of(divisor) {
            for candidate in [divisor, columns / divisor] {
                if candidate <= upper {
                    best = best.max(candidate);
                }
            }
        }
        divisor += 1;
    }
    Some(best)
}

fn rewrite_block(
    block: &mut mil::Block,
    inputs: &[mil::NamedValueType],
    cap: u64,
    changes: &mut Vec<PackedMatmulExpansion>,
) {
    let producers: BTreeMap<_, _> = block
        .operations
        .iter()
        .flat_map(|op| op.outputs.iter().map(move |out| (out.name.as_str(), op)))
        .collect();
    let types: BTreeMap<_, _> = inputs
        .iter()
        .chain(&block.inputs)
        .chain(block.operations.iter().flat_map(|op| &op.outputs))
        .filter_map(|out| Some((out.name.as_str(), tensor_type(out.r#type.as_ref()?)?)))
        .collect();
    let mut uses = BTreeMap::new();
    let mut names = inputs.iter().map(|input| input.name.clone()).collect();
    counts(block, &mut uses, &mut names);
    let mut replacements = BTreeMap::new();
    for op in &block.operations {
        if op.r#type != "matmul"
            || !op.blocks.is_empty()
            || op.outputs.len() != 1
            || op
                .inputs
                .keys()
                .any(|key| !matches!(key.as_str(), "x" | "y" | "transpose_x" | "transpose_y"))
            || op.attributes.keys().any(|key| key != "name")
        {
            continue;
        }
        let (Some(x), Some(y)) = (name_arg(op, "x"), name_arg(op, "y")) else {
            continue;
        };
        let Some(weight) = producers.get(y) else {
            continue;
        };
        if weight.r#type != "constexpr_affine_dequantize"
            || weight.outputs.len() != 1
            || !weight.blocks.is_empty()
            || !weight.inputs.is_empty()
            || weight.attributes.keys().any(|key| {
                !matches!(
                    key.as_str(),
                    "quantized_data" | "scale" | "zero_point" | "axis" | "name"
                )
            })
            || uses.get(y) != Some(&1)
        {
            continue;
        }
        let Some(q) = weight.attributes.get("quantized_data") else {
            continue;
        };
        let Some((dtype, shape)) = q.r#type.as_ref().and_then(tensor_type) else {
            continue;
        };
        let [groups, k, n] = shape.as_slice() else {
            continue;
        };
        if dtype != mil::DataType::Int8 as i32 || *k != 32 || *groups == 0 || *n == 0 {
            continue;
        }
        let Some(packed_count) = u64::from(*groups)
            .checked_mul(32)
            .and_then(|count| count.checked_mul(u64::from(*n)))
        else {
            continue;
        };
        if !matches!(&q.value, Some(mil::value::Value::BlobFileValue(_)))
            && !matches!(payload(q), Some(mil::tensor_value::Value::Bytes(v)) if v.values.len() as u64 == packed_count)
        {
            continue;
        }
        let Some(scale) = weight.attributes.get("scale") else {
            continue;
        };
        let Some(zero) = weight.attributes.get("zero_point") else {
            continue;
        };
        if scale.r#type.as_ref().and_then(tensor_type)
            != Some((mil::DataType::Float32 as i32, vec![]))
            || zero.r#type.as_ref().and_then(tensor_type)
                != Some((mil::DataType::Int8 as i32, vec![]))
            || !matches!(payload(scale), Some(mil::tensor_value::Value::Floats(v)) if v.values == [1.0])
            || !matches!(payload(zero), Some(mil::tensor_value::Value::Bytes(v)) if v.values.as_ref() == [0])
        {
            continue;
        }
        if !["transpose_x", "transpose_y"].iter().all(|key| matches!(constant(op, key, &producers).and_then(payload), Some(mil::tensor_value::Value::Bools(v)) if v.values == [false])) { continue; }
        let Some((xt, xs)) = types.get(x) else {
            continue;
        };
        if *xt != mil::DataType::Float32 as i32
            || xs.len() != 3
            || xs[0] != *groups
            || xs[2] != 32
            || xs[1] == 0
        {
            continue;
        }
        let output = &op.outputs[0];
        if output.r#type.as_ref().and_then(tensor_type)
            != Some((mil::DataType::Float32 as i32, vec![*groups, xs[1], *n]))
            || types.get(y) != Some(&(mil::DataType::Float32 as i32, shape.clone()))
        {
            continue;
        }
        let Some(tile) = tile_columns(*groups, *n, cap) else {
            continue;
        };
        // Check addressable logical outputs on arm64_32 as well as arm64 hosts.
        if u64::from(*groups)
            .checked_mul(u64::from(xs[1]))
            .and_then(|v| v.checked_mul(u64::from(*n)))
            .and_then(|v| v.checked_mul(4))
            .is_none_or(|v| v > i32::MAX as u64)
        {
            continue;
        }
        let mut nonce = 0;
        let prefix = loop {
            let candidate = format!("{}_packed_{nonce}_", output.name);
            if !names.iter().any(|name| name.starts_with(&candidate)) {
                break candidate;
            }
            nonce += 1;
        };
        let packed = format!("{prefix}weights");
        names.insert(packed.clone());
        let leaf = mil::Operation {
            r#type: "const".into(),
            outputs: vec![mil::NamedValueType {
                name: packed.clone(),
                r#type: q.r#type.clone(),
            }],
            attributes: [("val".into(), q.clone())].into_iter().collect(),
            ..Default::default()
        };
        replacements.insert(y.to_owned(), vec![leaf]);
        replacements.insert(
            output.name.clone(),
            emit_loop(&prefix, x, &packed, [*groups, xs[1], *n], tile, output),
        );
        changes.push(PackedMatmulExpansion {
            output: output.name.clone(),
            groups: *groups,
            tile_columns: tile,
            iterations: *n / tile,
            expanded_tile_bytes: u64::from(*groups) * 32 * u64::from(tile) * 4,
        });
    }
    if replacements.is_empty() {
        return;
    }
    let original = std::mem::take(&mut block.operations);
    for op in original {
        if let Some(replacement) = op
            .outputs
            .first()
            .and_then(|out| replacements.remove(&out.name))
        {
            block.operations.extend(replacement);
        } else {
            block.operations.push(op);
        }
    }
}

fn operation(
    kind: &str,
    inputs: &[(&str, mil::Argument)],
    name: &str,
    dtype: mil::DataType,
    shape: &[u32],
) -> mil::Operation {
    mil::Operation {
        r#type: kind.into(),
        inputs: inputs
            .iter()
            .map(|(key, value)| ((*key).into(), value.clone()))
            .collect(),
        outputs: vec![named_type(name, dtype, shape)],
        ..Default::default()
    }
}

fn emit_loop(
    prefix: &str,
    x: &str,
    packed: &str,
    dimensions: [u32; 3],
    tile: u32,
    output: &mil::NamedValueType,
) -> Vec<mil::Operation> {
    use mil::DataType::{Bool, Float32, Int8, Int32, String as Str};
    let [g, rows, n] = dimensions;
    let id = |suffix: &str| format!("{prefix}{suffix}");
    let scalar_zero = immediate(
        Float32,
        &[],
        mil::tensor_value::Value::Floats(mil::tensor_value::RepeatedFloats { values: vec![0.0] }),
    );
    let initial = operation(
        "fill",
        &[
            ("shape", ints(&[g, rows, n], false)),
            ("value", scalar_zero),
        ],
        &id("initial"),
        Float32,
        &[g, rows, n],
    );
    let cond = mil::Block {
        inputs: vec![
            named_type(&id("ci"), Int32, &[]),
            named_type(&id("cv"), Float32, &[g, rows, n]),
        ],
        outputs: vec![id("continue")],
        operations: vec![operation(
            "less",
            &[("x", reference(&id("ci"))), ("y", ints(&[n], true))],
            &id("continue"),
            Bool,
            &[],
        )],
        ..Default::default()
    };
    let mut body = mil::Block {
        inputs: vec![
            named_type(&id("i"), Int32, &[]),
            named_type(&id("carried"), Float32, &[g, rows, n]),
        ],
        outputs: vec![id("next"), id("updated")],
        ..Default::default()
    };
    body.operations.push(operation(
        "expand_dims",
        &[("x", reference(&id("i"))), ("axes", ints(&[0], false))],
        &id("iv"),
        Int32,
        &[1],
    ));
    let tuple = |left: mil::Argument, right: mil::Argument| mil::Argument {
        arguments: [left.arguments, right.arguments].concat(),
    };
    body.operations.push(operation(
        "concat",
        &[
            ("values", tuple(ints(&[0, 0], false), reference(&id("iv")))),
            ("axis", ints(&[0], true)),
            ("interleave", boolean(false)),
        ],
        &id("begin"),
        Int32,
        &[3],
    ));
    body.operations.push(operation(
        "slice_by_size",
        &[
            ("x", reference(packed)),
            ("begin", reference(&id("begin"))),
            ("size", ints(&[g, 32, tile], false)),
        ],
        &id("tile"),
        Int8,
        &[g, 32, tile],
    ));
    let fp32 = immediate(
        Str,
        &[],
        mil::tensor_value::Value::Strings(mil::tensor_value::RepeatedStrings {
            values: vec!["fp32".into()],
        }),
    );
    body.operations.push(operation(
        "cast",
        &[("x", reference(&id("tile"))), ("dtype", fp32)],
        &id("expanded"),
        Float32,
        &[g, 32, tile],
    ));
    body.operations.push(operation(
        "matmul",
        &[
            ("x", reference(x)),
            ("y", reference(&id("expanded"))),
            ("transpose_x", boolean(false)),
            ("transpose_y", boolean(false)),
        ],
        &id("product"),
        Float32,
        &[g, rows, tile],
    ));
    body.operations.push(operation(
        "add",
        &[("x", reference(&id("i"))), ("y", ints(&[tile], true))],
        &id("next"),
        Int32,
        &[],
    ));
    body.operations.push(operation(
        "expand_dims",
        &[("x", reference(&id("next"))), ("axes", ints(&[0], false))],
        &id("nv"),
        Int32,
        &[1],
    ));
    body.operations.push(operation(
        "concat",
        &[
            (
                "values",
                tuple(ints(&[g, rows], false), reference(&id("nv"))),
            ),
            ("axis", ints(&[0], true)),
            ("interleave", boolean(false)),
        ],
        &id("end"),
        Int32,
        &[3],
    ));
    body.operations.push(operation(
        "slice_update",
        &[
            ("x", reference(&id("carried"))),
            ("update", reference(&id("product"))),
            ("begin", reference(&id("begin"))),
            ("end", reference(&id("end"))),
        ],
        &id("updated"),
        Float32,
        &[g, rows, n],
    ));
    let loop_op = mil::Operation {
        r#type: "while_loop".into(),
        inputs: [(
            "loop_vars".into(),
            tuple(ints(&[0], true), reference(&id("initial"))),
        )]
        .into_iter()
        .collect(),
        outputs: vec![
            named_type(&id("final_index"), Int32, &[]),
            named_type(&id("result"), Float32, &[g, rows, n]),
        ],
        blocks: vec![cond, body],
        ..Default::default()
    };
    let result = mil::Operation {
        r#type: "identity".into(),
        inputs: [("x".into(), reference(&id("result")))]
            .into_iter()
            .collect(),
        outputs: vec![output.clone()],
        ..Default::default()
    };
    let mut operations = vec![initial, loop_op, result];
    materialize_loop_constants(&mut operations, prefix, &mut 0);
    operations
}

// Keep the loop parameters as explicit SSA constants, matching the qualified
// native template rather than introducing an inline-value specialization.
fn materialize_loop_constants(operations: &mut Vec<mil::Operation>, prefix: &str, next: &mut u32) {
    let mut expanded = Vec::new();
    for mut op in std::mem::take(operations) {
        let mut keys: Vec<_> = op.inputs.keys().cloned().collect();
        keys.sort();
        for key in keys {
            for binding in &mut op.inputs.get_mut(&key).unwrap().arguments {
                if let Some(mil::argument::binding::Binding::Value(value)) = &binding.binding {
                    let name = format!("{prefix}constant_{next}");
                    *next += 1;
                    expanded.push(mil::Operation {
                        r#type: "const".into(),
                        outputs: vec![mil::NamedValueType {
                            name: name.clone(),
                            r#type: value.r#type.clone(),
                        }],
                        attributes: [("val".into(), value.clone())].into_iter().collect(),
                        ..Default::default()
                    });
                    binding.binding = Some(mil::argument::binding::Binding::Name(name));
                }
            }
        }
        for block in &mut op.blocks {
            materialize_loop_constants(&mut block.operations, prefix, next);
        }
        expanded.push(op);
    }
    *operations = expanded;
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tile_budget_handles_empty_small_prime_and_large_shapes() {
        assert_eq!(tile_columns(0, 576, u64::MAX), None);
        assert_eq!(tile_columns(18, 0, u64::MAX), None);
        assert_eq!(tile_columns(18, 576, 2303), None);
        assert_eq!(tile_columns(18, 576, 2304), Some(1));
        assert_eq!(tile_columns(18, 1536, 2 * 1024 * 1024), Some(768));
        assert_eq!(tile_columns(18, 4093, 2 * 1024 * 1024), Some(1));
        assert_eq!(tile_columns(u32::MAX, u32::MAX, 0), None);
        assert_eq!(tile_columns(18, 576, u64::MAX), Some(576));
    }

    fn fixture() -> Model {
        use mil::DataType::{Float32, Int8, Int32};
        let extract =
            |a: mil::Argument| match a.arguments.into_iter().next().unwrap().binding.unwrap() {
                mil::argument::binding::Binding::Value(v) => v,
                _ => unreachable!(),
            };
        let q = extract(immediate(
            Int8,
            &[2, 32, 8],
            mil::tensor_value::Value::Bytes(mil::tensor_value::RepeatedBytes {
                values: vec![1; 512].into(),
            }),
        ));
        let mut weight = operation(
            "constexpr_affine_dequantize",
            &[],
            "weights",
            Float32,
            &[2, 32, 8],
        );
        weight.attributes = [
            ("quantized_data".into(), q),
            (
                "zero_point".into(),
                extract(immediate(
                    Int8,
                    &[],
                    mil::tensor_value::Value::Bytes(mil::tensor_value::RepeatedBytes {
                        values: vec![0].into(),
                    }),
                )),
            ),
            (
                "scale".into(),
                extract(immediate(
                    Float32,
                    &[],
                    mil::tensor_value::Value::Floats(mil::tensor_value::RepeatedFloats {
                        values: vec![1.0],
                    }),
                )),
            ),
            (
                "axis".into(),
                extract(immediate(
                    Int32,
                    &[],
                    mil::tensor_value::Value::Ints(mil::tensor_value::RepeatedInts {
                        values: vec![0],
                    }),
                )),
            ),
        ]
        .into_iter()
        .collect();
        let product = operation(
            "matmul",
            &[
                ("x", reference("input")),
                ("y", reference("weights")),
                ("transpose_x", boolean(false)),
                ("transpose_y", boolean(false)),
            ],
            "output",
            Float32,
            &[2, 3, 8],
        );
        let function = mil::Function {
            inputs: vec![named_type("input", Float32, &[2, 3, 32])],
            opset: "CoreML8".into(),
            block_specializations: [(
                "CoreML8".into(),
                mil::Block {
                    outputs: vec!["output".into()],
                    operations: vec![weight, product],
                    ..Default::default()
                },
            )]
            .into_iter()
            .collect(),
            ..Default::default()
        };
        Model {
            specification_version: 9,
            r#type: Some(model::Type::MlProgram(mil::Program {
                version: 1,
                functions: [("main".into(), function)].into_iter().collect(),
                ..Default::default()
            })),
            ..Default::default()
        }
    }

    fn main_block(model: &mut Model) -> &mut mil::Block {
        let Some(model::Type::MlProgram(p)) = model.r#type.as_mut() else {
            panic!("program")
        };
        p.functions
            .get_mut("main")
            .unwrap()
            .block_specializations
            .get_mut("CoreML8")
            .unwrap()
    }

    #[test]
    fn packed_bytes_result_type_and_native_contraction_survive() {
        let mut model = fixture();
        let before = main_block(&mut model).clone();
        let changes = bound_packed_matmuls(&mut model, 512);
        assert_eq!(changes.len(), 1);
        assert_eq!(
            (
                changes[0].tile_columns,
                changes[0].iterations,
                changes[0].expanded_tile_bytes
            ),
            (2, 4, 512)
        );
        let after = main_block(&mut model);
        assert_eq!(after.outputs, before.outputs);
        assert_eq!(
            after.operations[0].attributes["val"],
            before.operations[0].attributes["quantized_data"]
        );
        assert_eq!(
            after.operations.last().unwrap().outputs,
            before.operations[1].outputs
        );
        let loop_op = after
            .operations
            .iter()
            .find(|o| o.r#type == "while_loop")
            .unwrap();
        assert_eq!(loop_op.blocks.len(), 2);
        let body = &loop_op.blocks[1];
        assert!(body.operations.iter().any(|o| o.r#type == "cast"));
        let product = body
            .operations
            .iter()
            .find(|o| o.r#type == "matmul")
            .unwrap();
        assert_eq!(
            tensor_type(product.outputs[0].r#type.as_ref().unwrap())
                .unwrap()
                .1,
            [2, 3, 2]
        );
        assert!(bound_packed_matmuls(&mut model, 512).is_empty());
    }

    #[cfg(all(
        target_os = "macos",
        feature = "coreml-runtime",
        feature = "coreml-export"
    ))]
    #[test]
    fn native_packed_loop_preserves_exact_dyadic_products() {
        use crate::executors::coreml::{CoremlInput, run_coreml_with_inputs};
        use crate::protos::coreml::specification::{
            ArrayFeatureType, FeatureDescription, FeatureType, ModelDescription, feature_type,
        };
        use prost::Message;
        let feature = |name: &str, shape: Vec<i64>| FeatureDescription {
            name: name.into(),
            r#type: Some(FeatureType {
                r#type: Some(feature_type::Type::MultiArrayType(ArrayFeatureType {
                    shape,
                    data_type: 65568,
                    ..Default::default()
                })),
                ..Default::default()
            }),
            ..Default::default()
        };
        let mut model = fixture();
        model.description = Some(ModelDescription {
            input: vec![feature("input", vec![2, 3, 32])],
            output: vec![feature("output", vec![2, 3, 8])],
            ..Default::default()
        });
        assert_eq!(bound_packed_matmuls(&mut model, 512).len(), 1);
        let data: Vec<f32> = (0..192).map(|i| ((i % 17) as f32 - 8.0) * 0.125).collect();
        // Every product/sum is exactly representable for these dyadic packets.
        let expected: Vec<f32> = data
            .chunks(32)
            .flat_map(|row| {
                let sum: f32 = row.iter().sum();
                [sum; 8]
            })
            .collect();
        let attempts = run_coreml_with_inputs(
            &model.encode_to_vec(),
            vec![CoremlInput {
                name: "input".into(),
                shape: vec![2, 3, 32],
                data,
            }],
        )
        .unwrap();
        assert!(!attempts.is_empty());
        for attempt in attempts {
            let outputs = attempt
                .result
                .unwrap_or_else(|e| panic!("{}: {e}", attempt.compute_unit));
            assert_eq!(outputs.len(), 1);
            assert_eq!(outputs[0].shape, [2, 3, 8]);
            assert_eq!(outputs[0].data, expected, "{}", attempt.compute_unit);
        }
    }

    #[test]
    fn no_rewrite_for_fanout_transposition_nonunit_scale_or_small_budget() {
        for case in 0..7 {
            let mut model = fixture();
            let b = main_block(&mut model);
            match case {
                0 => b.outputs.push("weights".into()),
                1 => {
                    b.operations[1]
                        .inputs
                        .insert("transpose_y".into(), boolean(true));
                }
                2 => {
                    if let Some(mil::value::Value::ImmediateValue(v)) =
                        &mut b.operations[0].attributes.get_mut("scale").unwrap().value
                        && let Some(mil::value::immediate_value::Value::Tensor(t)) = &mut v.value
                        && let Some(mil::tensor_value::Value::Floats(f)) = &mut t.value
                    {
                        f.values[0] = 0.5;
                    }
                }
                3 => {
                    b.operations[1].outputs[0] =
                        named_type("output", mil::DataType::Float16, &[2, 3, 8]);
                }
                5 => {
                    let q = b.operations[0]
                        .attributes
                        .get_mut("quantized_data")
                        .unwrap();
                    q.r#type = named_type(
                        "huge",
                        mil::DataType::Int8,
                        &[i32::MAX as u32, 32, i32::MAX as u32],
                    )
                    .r#type;
                }
                6 => {
                    let unknown = b.operations[0].attributes["axis"].clone();
                    b.operations[0]
                        .attributes
                        .insert("future_semantics".into(), unknown);
                }
                _ => {}
            }
            let before = model.clone();
            assert!(bound_packed_matmuls(&mut model, if case == 4 { 255 } else { 512 }).is_empty());
            assert_eq!(model, before);
        }
    }
}
