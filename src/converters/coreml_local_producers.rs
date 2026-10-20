//! Bounded duplication of pure producers into admitted packed native children.
//!
//! The experimental Watch source profile uses this placement candidate after
//! bounded packed matmul scheduling. Operations and constants are copied exactly;
//! no inference arithmetic or runtime tensor access is introduced in Rust.
use super::coreml_packed::PackedMatmulExpansion;
use crate::protos::coreml::{
    mil_spec as m,
    specification::{
        ArrayFeatureType, FeatureDescription, FeatureType, Model, feature_type, model,
    },
};
use prost::Message;
use std::collections::{BTreeMap, BTreeSet};

const MAX_ELEMENTS: u64 = 2048;
const MAX_DYNAMIC_OPS: usize = 256;
const MAX_CONSTANT_BYTES: u64 = 65536;
const MAX_LOGICAL_OUTPUT_BYTES: u64 = 256 * 1024;
const MAX_BATCH_LOGICAL_OUTPUT_BYTES: u64 = 2 * 1024 * 1024;

/// One admitted placement; byte budgets describe graph tensors, not process memory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProducerPlacement {
    /// Child receiving an unchanged producer closure.
    pub child: usize,
    /// Replaced external input.
    pub original_input: String,
    /// Earlier producer value forwarded to this child.
    pub root_input: String,
    /// Child that exports the new input.
    pub root_owner: Option<usize>,
    /// Number of copied operations, including constants.
    pub copied_operations: usize,
    /// Number of copied nonconstant operations.
    pub copied_dynamic_operations: usize,
    /// Sum of logical output bytes, including repeated temporaries.
    pub logical_output_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Value {
    Input(String),
    Output(usize, usize, String),
}
type ClosurePlan = (Value, BTreeSet<(usize, usize)>);
impl Value {
    fn producer(&self) -> Option<(usize, usize)> {
        match self {
            Self::Output(c, o, _) => Some((*c, *o)),
            _ => None,
        }
    }
    fn name(&self) -> &str {
        match self {
            Self::Input(n) | Self::Output(_, _, n) => n,
        }
    }
}
fn function(model: &Model) -> Result<&m::Function, String> {
    let Some(model::Type::MlProgram(p)) = &model.r#type else {
        return Err("closure probe requires MLProgram children".into());
    };
    if p.functions.len() != 1 {
        return Err("closure probe requires one function".into());
    }
    let f = p.functions.get("main").ok_or("missing main")?;
    if f.block_specializations.len() != 1 {
        return Err("multiple specializations".into());
    }
    Ok(f)
}
fn block(model: &Model) -> Result<&m::Block, String> {
    let f = function(model)?;
    f.block_specializations
        .get(&f.opset)
        .ok_or("missing active block".into())
}
fn shape(v: &m::NamedValueType) -> Option<(i32, Vec<u64>)> {
    let m::value_type::Type::TensorType(t) = v.r#type.as_ref()?.r#type.as_ref()? else {
        return None;
    };
    let dims = t
        .dimensions
        .iter()
        .map(|d| match &d.dimension {
            Some(m::dimension::Dimension::Constant(c)) if c.size > 0 => Some(c.size),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()?;
    (t.rank == dims.len() as i64).then_some((t.data_type, dims))
}
fn elements(v: &m::NamedValueType) -> Option<u64> {
    shape(v)?.1.iter().try_fold(1u64, |n, &d| n.checked_mul(d))
}
fn logical_bytes(v: &m::NamedValueType) -> Option<u64> {
    let bits: u64 = match m::DataType::try_from(shape(v)?.0).ok()? {
        m::DataType::Bool | m::DataType::Int8 | m::DataType::Uint8 => 8,
        m::DataType::Float16 | m::DataType::Int16 | m::DataType::Uint16 => 16,
        m::DataType::Float32 | m::DataType::Int32 | m::DataType::Uint32 => 32,
        m::DataType::Float64 | m::DataType::Int64 | m::DataType::Uint64 => 64,
        // Strings are accounted by their encoded representation below.
        m::DataType::String => return Some(0),
        _ => return None,
    };
    elements(v)?.checked_mul(bits).map(|n| n.div_ceil(8))
}
fn operation_output_bytes(op: &m::Operation) -> Result<u64, String> {
    op.outputs.iter().try_fold(0u64, |sum, v| {
        sum.checked_add(logical_bytes(v).ok_or("unsupported output extent")?)
            .ok_or_else(|| "output extent overflow".into())
    })
}
fn embedded_bytes(op: &m::Operation) -> Result<u64, String> {
    op.attributes
        .values()
        .chain(
            op.inputs
                .values()
                .flat_map(|a| &a.arguments)
                .filter_map(|b| match &b.binding {
                    Some(m::argument::binding::Binding::Value(v)) => Some(v),
                    _ => None,
                }),
        )
        .try_fold(0u64, |sum, v| {
            let typed = m::NamedValueType {
                name: String::new(),
                r#type: v.r#type.clone(),
            };
            let logical = logical_bytes(&typed).ok_or("unsupported embedded extent")?;
            sum.checked_add(logical.max(v.encoded_len() as u64))
                .ok_or_else(|| "embedded extent overflow".into())
        })
}
fn description_matches(desc: &[FeatureDescription], values: &[m::NamedValueType]) -> bool {
    desc.len() == values.len()
        && values
            .iter()
            .map(|v| &v.name)
            .collect::<BTreeSet<_>>()
            .len()
            == values.len()
        && values.iter().all(|v| {
            feature(v).is_ok_and(|expected| {
                desc.iter().filter(|d| d.name == v.name).count() == 1
                    && desc
                        .iter()
                        .any(|d| d.name == v.name && d.r#type == expected.r#type)
            })
        })
}
fn feature(v: &m::NamedValueType) -> Result<FeatureDescription, String> {
    let (dtype, dims) = shape(v).ok_or("nonstatic boundary")?;
    if dims.is_empty() {
        return Err("scalar boundary".into());
    }
    let data_type = match m::DataType::try_from(dtype) {
        Ok(m::DataType::Float32) => 65568,
        Ok(m::DataType::Float16) => 65552,
        Ok(m::DataType::Int32) => 131104,
        _ => return Err("unsupported boundary dtype".into()),
    };
    Ok(FeatureDescription {
        name: v.name.clone(),
        r#type: Some(FeatureType {
            r#type: Some(feature_type::Type::MultiArrayType(ArrayFeatureType {
                shape: dims
                    .into_iter()
                    .map(i64::try_from)
                    .collect::<Result<_, _>>()
                    .map_err(|_| "boundary dimension overflow")?,
                data_type,
                ..Default::default()
            })),
            ..Default::default()
        }),
        ..Default::default()
    })
}
fn references(op: &m::Operation) -> Vec<String> {
    let mut names = op
        .inputs
        .values()
        .flat_map(|a| &a.arguments)
        .filter_map(|b| match &b.binding {
            Some(m::argument::binding::Binding::Name(n)) => Some(n.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    // Capture-aware even though a copied closure itself cannot include blocks.
    for b in &op.blocks {
        let local = b
            .inputs
            .iter()
            .chain(b.operations.iter().flat_map(|o| &o.outputs))
            .map(|v| v.name.as_str())
            .collect::<BTreeSet<_>>();
        names.extend(
            b.operations
                .iter()
                .flat_map(references)
                .filter(|n| !local.contains(n.as_str())),
        );
        names.extend(
            b.outputs
                .iter()
                .filter(|n| !local.contains(n.as_str()))
                .cloned(),
        );
    }
    names
}
fn pure(kind: &str) -> bool {
    matches!(
        kind,
        "identity"
            | "reshape"
            | "transpose"
            | "abs"
            | "relu"
            | "add"
            | "sub"
            | "mul"
            | "real_div"
            | "sqrt"
            | "reduce_mean"
            | "reduce_sum"
            | "reduce_max"
            | "greater"
            | "greater_equal"
            | "less"
            | "less_equal"
            | "equal"
            | "logical_and"
            | "logical_or"
            | "select"
            | "cast"
            | "concat"
    )
}
struct Index<'a> {
    ops: Vec<Vec<&'a m::Operation>>,
    resolutions: Vec<BTreeMap<String, Value>>,
    types: BTreeMap<Value, m::NamedValueType>,
}
impl<'a> Index<'a> {
    fn build(children: &'a [Model]) -> Result<Self, String> {
        let mut index = Self {
            ops: vec![],
            resolutions: vec![],
            types: BTreeMap::new(),
        };
        let mut prior = BTreeMap::<String, Value>::new();
        for (ci, child) in children.iter().enumerate() {
            if child
                .description
                .as_ref()
                .is_some_and(|d| !d.state.is_empty())
            {
                return Err("stateful children are outside closure probe".into());
            }
            let f = function(child)?;
            let b = block(child)?;
            let description = child
                .description
                .as_ref()
                .ok_or("missing child description")?;
            if !description.functions.is_empty()
                || !description_matches(&description.input, &f.inputs)
            {
                return Err("inconsistent child input description".into());
            }
            let mut resolution = BTreeMap::new();
            for input in &f.inputs {
                let value = prior
                    .get(&input.name)
                    .cloned()
                    .unwrap_or_else(|| Value::Input(input.name.clone()));
                if index
                    .types
                    .get(&value)
                    .is_some_and(|v| v.r#type != input.r#type)
                {
                    return Err("inconsistent forwarded type".into());
                }
                index
                    .types
                    .entry(value.clone())
                    .or_insert_with(|| input.clone());
                resolution.insert(input.name.clone(), value);
            }
            for (oi, op) in b.operations.iter().enumerate() {
                if matches!(op.r#type.as_str(), "read_state" | "coreml_update_state") {
                    return Err("stateful operations are outside closure probe".into());
                }
                if references(op).iter().any(|n| !resolution.contains_key(n)) {
                    return Err("non-topological child reference".into());
                }
                for v in &op.outputs {
                    if resolution.contains_key(&v.name) {
                        return Err("redefined child SSA value".into());
                    }
                    let key = Value::Output(ci, oi, v.name.clone());
                    index.types.insert(key.clone(), v.clone());
                    resolution.insert(v.name.clone(), key);
                }
            }
            let outputs = b
                .outputs
                .iter()
                .map(|n| resolution.get(n).and_then(|v| index.types.get(v)).cloned())
                .collect::<Option<Vec<_>>>()
                .ok_or("unresolved child output")?;
            if !description_matches(&description.output, &outputs) {
                return Err("inconsistent child output description".into());
            }
            for name in &b.outputs {
                prior.insert(
                    name.clone(),
                    resolution
                        .get(name)
                        .ok_or("unresolved child output")?
                        .clone(),
                );
            }
            index.resolutions.push(resolution);
            index.ops.push(b.operations.iter().collect());
        }
        Ok(index)
    }
    fn op(&self, key: (usize, usize)) -> &m::Operation {
        self.ops[key.0][key.1]
    }
    fn constant(&self, value: &Value) -> bool {
        value
            .producer()
            .is_some_and(|k| self.op(k).r#type == "const")
    }
    fn deps(&self, key: (usize, usize)) -> Result<Vec<Value>, String> {
        references(self.op(key))
            .into_iter()
            .map(|n| {
                self.resolutions[key.0]
                    .get(&n)
                    .cloned()
                    .ok_or_else(|| format!("unresolved closure reference {n}"))
            })
            .collect()
    }
    fn plan(&self, value: Value) -> Result<Option<ClosurePlan>, String> {
        let mut frontier = BTreeSet::from([value]);
        let mut chosen = BTreeSet::new();
        let mut best = None;
        while !frontier.is_empty() && chosen.len() < MAX_DYNAMIC_OPS {
            let expand = frontier
                .iter()
                .max_by_key(|v| v.producer())
                .unwrap()
                .clone();
            let Some(key) = expand.producer() else { break };
            let op = self.op(key);
            let deps = self.deps(key)?;
            if !op.blocks.is_empty()
                || !pure(&op.r#type)
                || op
                    .outputs
                    .iter()
                    .any(|v| elements(v).is_none_or(|n| n > MAX_ELEMENTS))
                || deps
                    .iter()
                    .filter(|v| !self.constant(v))
                    .any(|v| elements(&self.types[v]).is_none_or(|n| n > MAX_ELEMENTS))
            {
                break;
            }
            chosen.insert(key);
            frontier.remove(&expand);
            frontier.extend(deps.into_iter().filter(|v| !self.constant(v)));
            frontier.retain(|v| v.producer().is_none_or(|p| !chosen.contains(&p)));
            if frontier.len() == 1 {
                let root = frontier.first().unwrap();
                if feature(&self.types[root]).is_ok() {
                    best = Some((root.clone(), chosen.clone()));
                }
            }
        }
        let Some((root, mut chosen)) = best else {
            return Ok(None);
        };
        let mut pending = chosen.iter().copied().collect::<Vec<_>>();
        while let Some(key) = pending.pop() {
            for dep in self.deps(key)? {
                if self.constant(&dep) {
                    let producer = dep.producer().unwrap();
                    if chosen.insert(producer) {
                        pending.push(producer);
                    }
                }
            }
        }
        let mut embedded = 0u64;
        let mut outputs = 0u64;
        for &key in &chosen {
            let op = self.op(key);
            embedded = embedded
                .checked_add(embedded_bytes(op)?)
                .ok_or("embedded budget overflow")?;
            outputs = outputs
                .checked_add(operation_output_bytes(op)?)
                .ok_or("output budget overflow")?;
        }
        if embedded > MAX_CONSTANT_BYTES || outputs > MAX_LOGICAL_OUTPUT_BYTES {
            return Ok(None);
        }
        Ok(Some((root, chosen)))
    }
}

/// Clone bounded pure producer closures into one selected child. Other children
/// retain all operations; only newly required root output descriptions are added.
/// No dead-code elimination is performed. Error leaves the pipeline unchanged.
fn clone_into_child(
    model: &mut Model,
    selected: usize,
    budget: u64,
) -> Result<Vec<ProducerPlacement>, String> {
    let Some(model::Type::Pipeline(p)) = &model.r#type else {
        return Err("expected full pipeline".into());
    };
    if selected >= p.models.len() {
        return Err("child index out of range".into());
    }
    let index = Index::build(&p.models)?;
    let old = &p.models[selected];
    let f = function(old)?;
    let b = block(old)?;
    let eligible = f
        .inputs
        .iter()
        .filter(|v| {
            elements(v).is_some_and(|n| n <= MAX_ELEMENTS)
                && shape(v).is_some_and(|(dt, _)| dt == m::DataType::Float32 as i32)
        })
        .collect::<Vec<_>>();
    if eligible.len() != 1
        || f.inputs
            .iter()
            .any(|v| v.name != eligible[0].name && elements(v).is_none_or(|n| n <= MAX_ELEMENTS))
    {
        return Ok(vec![]);
    }
    let eligible = eligible[0].name.as_str();
    let retained_names = f
        .inputs
        .iter()
        .filter(|v| v.name != eligible)
        .map(|v| v.name.as_str())
        .collect::<BTreeSet<_>>();
    let mut existing = b
        .operations
        .iter()
        .flat_map(|o| o.outputs.iter().map(move |v| (v.name.clone(), o)))
        .collect::<BTreeMap<_, _>>();
    let mut prefix = Vec::<m::Operation>::new();
    let mut input_values = Vec::<m::NamedValueType>::new();
    let mut roots = BTreeMap::<(usize, String), m::NamedValueType>::new();
    let mut audit = vec![];
    // Do not mutate until all references, types and name collisions validate.
    for input in &f.inputs {
        if input.name != eligible {
            input_values.push(input.clone());
            continue;
        }
        let value = index.resolutions[selected][&input.name].clone();
        let Some((root, chosen)) = index.plan(value)? else {
            input_values.push(input.clone());
            continue;
        };
        let root_type = index.types[&root].clone();
        feature(&root_type)?;
        if existing.contains_key(root.name()) || retained_names.contains(root.name()) {
            return Err("new root input collides with selected child SSA value".into());
        }
        if shape(&root_type).is_none_or(|(dt, _)| dt != m::DataType::Float32 as i32) {
            continue;
        }
        // Exporting an internal value must not shadow a different version of
        // that name consumed or produced by another pipeline child.
        let Some((owner, _)) = root.producer() else {
            continue;
        };
        if owner >= selected {
            continue;
        }
        if !block(&p.models[owner])?
            .outputs
            .iter()
            .any(|n| n == root.name())
            && model
                .description
                .as_ref()
                .is_some_and(|d| d.output.iter().any(|v| v.name == root.name()))
        {
            return Err("root export would change a pipeline output binding".into());
        }
        for (ci, child) in p.models.iter().enumerate().skip(owner + 1) {
            if block(child)?.outputs.iter().any(|n| n == root.name())
                || function(child)?.inputs.iter().any(|v| {
                    v.name == root.name() && index.resolutions[ci].get(&v.name) != Some(&root)
                })
            {
                return Err("root export would shadow a different value".into());
            }
        }
        let mut count = 0;
        let mut dynamic = 0;
        for key in chosen {
            let op = index.op(key);
            if op
                .outputs
                .iter()
                .any(|v| existing.contains_key(&v.name) || retained_names.contains(v.name.as_str()))
            {
                // Source graphs use globally unique dynamic names. Reject any
                // collision instead of guessing across child scopes/NaN payloads.
                return Err("copied closure name collides with selected child".into());
            }
            prefix.push(op.clone());
            count += 1;
            dynamic += usize::from(op.r#type != "const");
            for v in &op.outputs {
                existing.insert(v.name.clone(), op);
            }
        }
        if let Some((owner, _)) = root.producer() {
            if owner >= selected {
                return Err("closure root is not from an earlier child".into());
            }
            roots.insert((owner, root.name().to_owned()), root_type.clone());
        }
        input_values.push(root_type);
        audit.push(ProducerPlacement {
            child: selected,
            original_input: input.name.clone(),
            root_input: root.name().to_owned(),
            root_owner: root.producer().map(|p| p.0),
            copied_operations: count,
            copied_dynamic_operations: dynamic,
            logical_output_bytes: prefix.iter().try_fold(0u64, |sum, op| {
                sum.checked_add(operation_output_bytes(op)?)
                    .ok_or_else(|| "output budget overflow".to_string())
            })?,
        });
    }
    if audit.is_empty() {
        return Ok(audit);
    }
    if audit.iter().map(|a| a.logical_output_bytes).sum::<u64>() > budget {
        return Ok(vec![]);
    }
    let mut seen = BTreeSet::new();
    input_values.retain(|v| seen.insert(v.name.clone()));
    let descriptions = input_values
        .iter()
        .map(|v| {
            if retained_names.contains(v.name.as_str()) {
                old.description
                    .as_ref()
                    .unwrap()
                    .input
                    .iter()
                    .find(|d| d.name == v.name)
                    .cloned()
                    .ok_or_else(|| "missing retained description".into())
            } else {
                feature(v)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    // Clone only models whose interfaces change. The full pipeline/weight bytes
    // are never cloned, preserving the source parser's memory discipline.
    let mut changed = BTreeMap::<usize, Model>::new();
    changed.insert(selected, old.clone());
    for ((owner, name), ty) in roots {
        let child = changed
            .entry(owner)
            .or_insert_with(|| p.models[owner].clone());
        let Some(model::Type::MlProgram(program)) = &mut child.r#type else {
            unreachable!()
        };
        let f = program.functions.get_mut("main").unwrap();
        let b = f.block_specializations.get_mut(&f.opset).unwrap();
        if !b.outputs.contains(&name) {
            b.outputs.push(name);
            child
                .description
                .as_mut()
                .ok_or("missing child description")?
                .output
                .push(feature(&ty)?);
        }
    }
    let target = changed.get_mut(&selected).unwrap();
    let Some(model::Type::MlProgram(program)) = &mut target.r#type else {
        unreachable!()
    };
    let f = program.functions.get_mut("main").unwrap();
    let b = f.block_specializations.get_mut(&f.opset).unwrap();
    prefix.append(&mut b.operations);
    b.operations = prefix;
    f.inputs = input_values;
    target
        .description
        .as_mut()
        .ok_or("missing selected description")?
        .input = descriptions;
    drop(index);
    let Some(model::Type::Pipeline(p)) = &mut model.r#type else {
        unreachable!()
    };
    for (i, child) in changed {
        p.models[i] = child;
    }
    Ok(audit)
}

/// Copy an unchanged, bounded pure producer closure into each eligible native
/// child containing a result named by the immediately preceding packed-matmul
/// rewrite. A child must have exactly one small static Float32 input; all other
/// inputs must be larger static tensors and are retained unchanged. Admission
/// uses graph structure and byte budgets, never model names or child positions.
///
/// Inadmissible children are left unchanged, including on descriptor/name
/// ambiguity. Original producer operations and all declared graph outputs are
/// retained. New root outputs are forwarded by the existing native last-use
/// router. This is a placement candidate, not a numerical or memory guarantee;
/// callers must qualify the resulting CoreML compilation on the target OS.
pub fn clone_for_packed_children(
    model: &mut Model,
    expansions: &[PackedMatmulExpansion],
) -> Vec<ProducerPlacement> {
    let outputs = expansions
        .iter()
        .map(|e| e.output.as_str())
        .collect::<BTreeSet<_>>();
    let Some(model::Type::Pipeline(pipeline)) = &model.r#type else {
        return vec![];
    };
    // Expansion records currently carry result names rather than child scope.
    // Reject ambiguous names instead of admitting an unrelated same-named op.
    let mut occurrences = BTreeMap::<&str, usize>::new();
    for child in &pipeline.models {
        if let Ok(block) = block(child) {
            for value in block.operations.iter().flat_map(|op| &op.outputs) {
                if outputs.contains(value.name.as_str()) {
                    *occurrences.entry(&value.name).or_default() += 1;
                }
            }
        }
    }
    let candidates = pipeline
        .models
        .iter()
        .enumerate()
        .filter_map(|(i, child)| {
            let block = block(child).ok()?;
            block
                .operations
                .iter()
                .flat_map(|op| &op.outputs)
                .any(|value| occurrences.get(value.name.as_str()) == Some(&1))
                .then_some(i)
        })
        .collect::<Vec<_>>();
    let mut placements = Vec::new();
    let mut remaining = MAX_BATCH_LOGICAL_OUTPUT_BYTES;
    for child in candidates {
        if let Ok(admitted) = clone_into_child(model, child, remaining) {
            remaining -= admitted.iter().map(|a| a.logical_output_bytes).sum::<u64>();
            placements.extend(admitted);
        }
    }
    placements
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protos::coreml::specification::{ModelDescription, Pipeline};
    fn value(n: &str, shape: &[u64]) -> m::NamedValueType {
        m::NamedValueType {
            name: n.into(),
            r#type: Some(m::ValueType {
                r#type: Some(m::value_type::Type::TensorType(m::TensorType {
                    data_type: m::DataType::Float32 as i32,
                    rank: shape.len() as i64,
                    dimensions: shape
                        .iter()
                        .map(|&size| m::Dimension {
                            dimension: Some(m::dimension::Dimension::Constant(
                                m::dimension::ConstantDimension { size },
                            )),
                        })
                        .collect(),
                    ..Default::default()
                })),
            }),
        }
    }
    fn op(kind: &str, name: &str, inputs: &[&str], dims: &[u64]) -> m::Operation {
        m::Operation {
            r#type: kind.into(),
            inputs: inputs
                .iter()
                .enumerate()
                .map(|(i, n)| {
                    (
                        format!("a{i}"),
                        m::Argument {
                            arguments: vec![m::argument::Binding {
                                binding: Some(m::argument::binding::Binding::Name((*n).into())),
                            }],
                        },
                    )
                })
                .collect(),
            outputs: vec![value(name, dims)],
            ..Default::default()
        }
    }
    fn child(inputs: Vec<m::NamedValueType>, ops: Vec<m::Operation>, out: &str) -> Model {
        let output = ops
            .iter()
            .flat_map(|o| &o.outputs)
            .find(|x| x.name == out)
            .unwrap();
        Model {
            specification_version: 9,
            description: Some(ModelDescription {
                input: inputs.iter().map(|v| feature(v).unwrap()).collect(),
                output: vec![feature(output).unwrap()],
                ..Default::default()
            }),
            r#type: Some(model::Type::MlProgram(m::Program {
                version: 1,
                functions: [(
                    "main".into(),
                    m::Function {
                        inputs,
                        opset: "CoreML8".into(),
                        block_specializations: [(
                            "CoreML8".into(),
                            m::Block {
                                operations: ops,
                                outputs: vec![out.into()],
                                ..Default::default()
                            },
                        )]
                        .into(),
                        ..Default::default()
                    },
                )]
                .into(),
                ..Default::default()
            })),
            ..Default::default()
        }
    }
    fn fixture(two_roots: bool) -> Model {
        let mut ops = vec![
            op("matmul", "root", &["x", "x"], &[1, 4]),
            op("mul", "a", &["root", "root"], &[1, 4]),
            op("abs", "b", &["root"], &[1, 4]),
        ];
        ops.push(op(
            "add",
            "prepared",
            if two_roots { &["a", "y"] } else { &["a", "b"] },
            &[1, 4],
        ));
        let first = child(
            vec![value("x", &[1, 4]), value("y", &[1, 4])],
            ops,
            "prepared",
        );
        let second = child(
            vec![value("prepared", &[1, 4])],
            vec![op("matmul", "z", &["prepared", "prepared"], &[1, 4])],
            "z",
        );
        Model {
            r#type: Some(model::Type::Pipeline(Pipeline {
                models: vec![first, second],
                names: vec![],
            })),
            ..Default::default()
        }
    }
    fn expansion(output: &str) -> PackedMatmulExpansion {
        PackedMatmulExpansion {
            output: output.into(),
            groups: 1,
            tile_columns: 1,
            iterations: 4,
            expanded_tile_bytes: 128,
        }
    }
    fn pipeline_mut(model: &mut Model) -> &mut Pipeline {
        let Some(model::Type::Pipeline(p)) = &mut model.r#type else {
            panic!()
        };
        p
    }
    #[test]
    fn clones_diamond_without_changing_operations_and_keeps_producer() {
        let mut m = fixture(false);
        let before = m.clone();
        let audit = clone_into_child(&mut m, 1, MAX_LOGICAL_OUTPUT_BYTES).unwrap();
        assert_eq!(audit[0].root_input, "root");
        assert_eq!(audit[0].copied_dynamic_operations, 3);
        let model::Type::Pipeline(p) = m.r#type.unwrap() else {
            panic!()
        };
        let model::Type::Pipeline(old) = before.r#type.unwrap() else {
            panic!()
        };
        assert_eq!(
            block(&p.models[0]).unwrap().operations,
            block(&old.models[0]).unwrap().operations
        );
        assert!(
            block(&p.models[0])
                .unwrap()
                .outputs
                .contains(&"root".into())
        );
        assert_eq!(
            &block(&p.models[1]).unwrap().operations[..3],
            &block(&old.models[0]).unwrap().operations[1..]
        );
        assert_eq!(function(&p.models[1]).unwrap().inputs[0].name, "root");
    }
    #[test]
    fn rejects_two_root_frontier_without_partial_mutation() {
        let mut m = fixture(true);
        let old = m.clone();
        assert!(
            clone_into_child(&mut m, 1, MAX_LOGICAL_OUTPUT_BYTES)
                .unwrap()
                .is_empty()
        );
        assert_eq!(m, old);
    }
    #[test]
    fn rejects_large_input_reduction() {
        let mut m = fixture(false);
        let model::Type::Pipeline(p) = m.r#type.as_mut().unwrap() else {
            panic!()
        };
        let model::Type::MlProgram(prog) = p.models[0].r#type.as_mut().unwrap() else {
            panic!()
        };
        let f = prog.functions.get_mut("main").unwrap();
        let b = f.block_specializations.get_mut("CoreML8").unwrap();
        b.operations = vec![
            op("matmul", "large", &["x", "x"], &[1, 4096]),
            op("reduce_sum", "prepared", &["large"], &[1, 4]),
        ];
        let old = m.clone();
        assert!(
            clone_into_child(&mut m, 1, MAX_LOGICAL_OUTPUT_BYTES)
                .unwrap()
                .is_empty()
        );
        assert_eq!(m, old);
    }
    #[test]
    fn scalar_or_unknown_child_does_not_mutate() {
        let mut m = fixture(false);
        let old = m.clone();
        assert!(clone_into_child(&mut m, 99, MAX_LOGICAL_OUTPUT_BYTES).is_err());
        assert_eq!(m, old);
    }
    #[test]
    fn repeated_roots_preserve_descriptions_outputs_and_last_use_routes() {
        let mut m = fixture(false);
        pipeline_mut(&mut m).models.push(child(
            vec![value("prepared", &[1, 4])],
            vec![op("matmul", "z2", &["prepared", "prepared"], &[1, 4])],
            "z2",
        ));
        let original = m.clone();
        let records = [expansion("z"), expansion("z2")];
        let changed = clone_for_packed_children(&mut m, &records);
        assert_eq!(changed.len(), 2);
        assert!(
            changed
                .iter()
                .all(|a| a.root_input == "root" && a.copied_dynamic_operations == 3)
        );
        let p = pipeline_mut(&mut m);
        let Some(model::Type::Pipeline(old)) = &original.r#type else {
            panic!()
        };
        assert_eq!(
            p.models[0]
                .description
                .as_ref()
                .unwrap()
                .output
                .iter()
                .filter(|d| d.name == "root")
                .count(),
            1
        );
        for i in 1..=2 {
            assert_eq!(
                p.models[i].description.as_ref().unwrap().output,
                old.models[i].description.as_ref().unwrap().output
            );
            assert_eq!(
                block(&p.models[i]).unwrap().outputs,
                block(&old.models[i]).unwrap().outputs
            );
            assert!(description_matches(
                &p.models[i].description.as_ref().unwrap().input,
                &function(&p.models[i]).unwrap().inputs
            ));
        }
        // The existing name-based last-use router sees one exported root with
        // two downstream consumers; the original prepared output is retained.
        let mut owner = BTreeMap::new();
        let mut last = BTreeMap::new();
        for (ci, c) in p.models.iter().enumerate() {
            for input in &c.description.as_ref().unwrap().input {
                if input.name == "root" {
                    assert_eq!(owner.get("root"), Some(&0));
                    last.insert("root", ci);
                }
            }
            for output in &c.description.as_ref().unwrap().output {
                owner.insert(output.name.as_str(), ci);
            }
        }
        assert_eq!(last["root"], 2);
        assert_eq!(
            block(&p.models[0]).unwrap().operations,
            block(&old.models[0]).unwrap().operations
        );
        let once = m.clone();
        assert!(clone_for_packed_children(&mut m, &records).is_empty());
        assert_eq!(m, once);
    }
    #[test]
    fn only_unique_new_expansion_results_admit_children() {
        let mut m = fixture(false);
        let old = m.clone();
        assert!(clone_for_packed_children(&mut m, &[]).is_empty());
        assert_eq!(m, old);
        let duplicate = pipeline_mut(&mut m).models[1].clone();
        pipeline_mut(&mut m).models.push(duplicate);
        let old = m.clone();
        assert!(clone_for_packed_children(&mut m, &[expansion("z")]).is_empty());
        assert_eq!(m, old);
    }
    #[test]
    fn shape_or_collision_failure_is_transactional() {
        for kind in 0..3 {
            let mut m = fixture(false);
            let target = &mut pipeline_mut(&mut m).models[1];
            if kind == 0 {
                target.description.as_mut().unwrap().input[0] =
                    feature(&value("prepared", &[1, 8])).unwrap();
            } else {
                let Some(model::Type::MlProgram(p)) = &mut target.r#type else {
                    panic!()
                };
                let b = p
                    .functions
                    .get_mut("main")
                    .unwrap()
                    .block_specializations
                    .get_mut("CoreML8")
                    .unwrap();
                b.operations.insert(
                    0,
                    op(
                        "relu",
                        if kind == 1 { "a" } else { "root" },
                        &["prepared"],
                        &[1, 4],
                    ),
                );
            }
            let old = m.clone();
            assert!(clone_for_packed_children(&mut m, &[expansion("z")]).is_empty());
            assert_eq!(m, old);
        }
    }
    #[test]
    fn exported_internal_root_cannot_shadow_an_intervening_use_or_definition() {
        for define in [false, true] {
            let mut m = fixture(false);
            let middle = if define {
                child(
                    vec![value("prepared", &[1, 4])],
                    vec![op("relu", "root", &["prepared"], &[1, 4])],
                    "root",
                )
            } else {
                child(
                    vec![value("root", &[1, 4])],
                    vec![op("relu", "other", &["root"], &[1, 4])],
                    "other",
                )
            };
            pipeline_mut(&mut m).models.insert(1, middle);
            let old = m.clone();
            assert!(clone_for_packed_children(&mut m, &[expansion("z")]).is_empty());
            assert_eq!(m, old);
        }
    }
    #[test]
    fn exporting_an_internal_root_preserves_top_level_output_bindings() {
        let mut m = fixture(false);
        m.description = Some(ModelDescription {
            input: vec![feature(&value("root", &[1, 4])).unwrap()],
            output: vec![
                feature(&value("root", &[1, 4])).unwrap(),
                feature(&value("z", &[1, 4])).unwrap(),
            ],
            ..Default::default()
        });
        let old = m.clone();
        assert!(clone_for_packed_children(&mut m, &[expansion("z")]).is_empty());
        assert_eq!(m, old);
    }
    #[test]
    fn duplicate_function_value_names_do_not_match_descriptions() {
        let values = vec![value("x", &[1, 4]), value("x", &[1, 4])];
        let desc = vec![
            feature(&value("x", &[1, 4])).unwrap(),
            feature(&value("y", &[1, 4])).unwrap(),
        ];
        assert!(!description_matches(&desc, &values));
    }
    #[test]
    fn one_small_input_is_replaced_and_large_passthrough_is_unchanged() {
        let mut m = fixture(false);
        let mut target = child(
            vec![value("prepared", &[1, 4]), value("large", &[1, 4096])],
            vec![
                op("matmul", "z", &["prepared", "prepared"], &[1, 4]),
                op("concat", "joined", &["z", "large"], &[1, 4100]),
            ],
            "joined",
        );
        target.description.as_mut().unwrap().input[1].short_description =
            "Retained user feature metadata".into();
        let original = target.clone();
        pipeline_mut(&mut m).models[1] = target;
        assert_eq!(
            clone_for_packed_children(&mut m, &[expansion("z")]).len(),
            1
        );
        let target = &pipeline_mut(&mut m).models[1];
        assert_eq!(
            target.description.as_ref().unwrap().input[1],
            original.description.as_ref().unwrap().input[1]
        );
        assert_eq!(
            function(target).unwrap().inputs[1],
            function(&original).unwrap().inputs[1]
        );
        assert_eq!(
            target.description.as_ref().unwrap().output,
            original.description.as_ref().unwrap().output
        );
        assert_eq!(
            &block(target).unwrap().operations[3..],
            &block(&original).unwrap().operations
        );
    }
    #[test]
    fn retained_input_collisions_and_multiple_small_inputs_skip_atomically() {
        for (name, dims) in [
            ("a", vec![1, 4096]),
            ("root", vec![1, 4096]),
            ("other", vec![1, 4]),
        ] {
            let mut m = fixture(false);
            pipeline_mut(&mut m).models[1] = child(
                vec![value("prepared", &[1, 4]), value(name, &dims)],
                vec![op("matmul", "z", &["prepared", "prepared"], &[1, 4])],
                "z",
            );
            let old = m.clone();
            assert!(clone_for_packed_children(&mut m, &[expansion("z")]).is_empty());
            assert_eq!(m, old);
        }
    }
    #[test]
    fn logical_and_embedded_budgets_skip_without_mutation() {
        let mut m = fixture(false);
        let old = m.clone();
        assert!(clone_into_child(&mut m, 1, 1).unwrap().is_empty());
        assert_eq!(m, old);
        let first = &mut pipeline_mut(&mut m).models[0];
        let Some(model::Type::MlProgram(p)) = &mut first.r#type else {
            panic!()
        };
        let b = p
            .functions
            .get_mut("main")
            .unwrap()
            .block_specializations
            .get_mut("CoreML8")
            .unwrap();
        let huge = m::Value {
            r#type: Some(m::ValueType {
                r#type: Some(m::value_type::Type::TensorType(m::TensorType {
                    data_type: m::DataType::String as i32,
                    rank: 0,
                    ..Default::default()
                })),
            }),
            value: Some(m::value::Value::ImmediateValue(m::value::ImmediateValue {
                value: Some(m::value::immediate_value::Value::Tensor(m::TensorValue {
                    value: Some(m::tensor_value::Value::Strings(
                        m::tensor_value::RepeatedStrings {
                            values: vec!["x".repeat(65536)],
                        },
                    )),
                })),
            })),
            ..Default::default()
        };
        b.operations[1].attributes.insert("name".into(), huge);
        let old = m.clone();
        assert!(clone_for_packed_children(&mut m, &[expansion("z")]).is_empty());
        assert_eq!(m, old);
        let mut wide = value("wide", &[1024]);
        if let Some(m::value_type::Type::TensorType(t)) =
            wide.r#type.as_mut().and_then(|v| v.r#type.as_mut())
        {
            t.data_type = m::DataType::Float64 as i32;
        }
        assert_eq!(logical_bytes(&wide), Some(8192));
    }
}
