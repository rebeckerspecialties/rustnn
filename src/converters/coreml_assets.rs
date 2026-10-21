//! Compact native blob spans in place and externalize large immediate weights.
//! Planning validates every reference before mutating either model or storage.
use super::coreml_packed::tensor_type;
use crate::protos::coreml::{
    mil_spec as m,
    specification::{Model, model},
};
use std::{
    collections::HashMap,
    hash::{Hash, Hasher},
    ops::Range,
};
const FILE: &str = "@model_path/weights/weights.bin";
enum Source {
    Span(Range<usize>),
    Inline(Vec<u8>),
}
impl Source {
    fn bytes<'a>(&'a self, old: &'a [u8]) -> &'a [u8] {
        match self {
            Self::Span(r) => &old[r.clone()],
            Self::Inline(b) => b,
        }
    }
}
struct Entry {
    source: Source,
    dtype: u32,
    offset: usize,
}
fn values(
    model: &mut Model,
    visit: &mut impl FnMut(&mut m::Value) -> Result<(), String>,
) -> Result<(), String> {
    fn block(
        b: &mut m::Block,
        visit: &mut impl FnMut(&mut m::Value) -> Result<(), String>,
    ) -> Result<(), String> {
        for op in &mut b.operations {
            for v in op.attributes.values_mut() {
                visit(v)?;
            }
            for a in op.inputs.values_mut().flat_map(|a| &mut a.arguments) {
                if let Some(m::argument::binding::Binding::Value(v)) = &mut a.binding {
                    visit(v)?;
                }
            }
            for child in &mut op.blocks {
                block(child, visit)?;
            }
        }
        Ok(())
    }
    match &mut model.r#type {
        Some(model::Type::Pipeline(p)) => {
            for c in &mut p.models {
                values(c, visit)?;
            }
        }
        Some(model::Type::MlProgram(p)) => {
            for f in p.functions.values_mut() {
                for b in f.block_specializations.values_mut() {
                    block(b, visit)?;
                }
            }
        }
        _ => {}
    }
    Ok(())
}
/// Exact compaction of the contiguous, 64-byte-aligned version-2 weight format
/// emitted by RustNN. No arithmetic or quantization occurs. Validation errors
/// leave both arguments unchanged. Large live blobs are moved within the owned
/// buffer, avoiding a second model-sized allocation during on-device lowering.
pub fn compact(model: &mut Model, weights: &mut Vec<u8>) -> Result<(), String> {
    let mut storage = super::CoremlWeightStorage::memory(std::mem::take(weights));
    let result = compact_file_backed(model, &mut storage);
    *weights = storage.into_memory();
    result
}
/// Compact and externalize through owned local scratch storage.
pub fn compact_file_backed(
    model: &mut Model,
    weights: &mut super::CoremlWeightStorage,
) -> Result<(), String> {
    let mut entries = Vec::<Entry>::new();
    let mut bindings = Vec::new();
    let mut hashes = HashMap::<(u32, usize, u64), Vec<usize>>::new();
    let mut spans = Vec::new();
    values(model, &mut |v| {
        let blob = matches!(v.value, Some(m::value::Value::BlobFileValue(_)));
        let Some((dtype, shape)) = v.r#type.as_ref().and_then(tensor_type) else {
            if blob {
                return Err("untyped blob".into());
            }
            bindings.push(None);
            return Ok(());
        };
        let (dtype, width) = match m::DataType::try_from(dtype) {
            Ok(m::DataType::Float32) => (2, 4usize),
            Ok(m::DataType::Float16) => (1, 2),
            Ok(m::DataType::Int8) => (4, 1),
            Ok(m::DataType::Uint8) => (3, 1),
            _ => {
                if blob {
                    return Err("unsupported blob type".into());
                }
                bindings.push(None);
                return Ok(());
            }
        };
        let size = shape
            .iter()
            .try_fold(width, |a, &b| a.checked_mul(b as usize))
            .ok_or("constant size overflow")?;
        let source = if let Some(m::value::Value::BlobFileValue(b)) = &v.value {
            if b.file_name != FILE {
                return Err(format!("unsupported blob file {}", b.file_name));
            }
            let start = usize::try_from(b.offset).map_err(|_| "offset overflow")?;
            if start < 64 || start % 64 != 0 {
                return Err("unaligned blob metadata".into());
            }
            let end = start.checked_add(64).ok_or("offset overflow")?;
            let meta = weights.get(start..end).ok_or("short metadata")?;
            if u32::from_le_bytes(meta[..4].try_into().unwrap()) != 0xdeadbeef
                || u32::from_le_bytes(meta[4..8].try_into().unwrap()) != dtype
            {
                return Err("blob sentinel/type mismatch".into());
            }
            let len = usize::try_from(u64::from_le_bytes(meta[8..16].try_into().unwrap()))
                .map_err(|_| "length overflow")?;
            let pos = usize::try_from(u64::from_le_bytes(meta[16..24].try_into().unwrap()))
                .map_err(|_| "payload offset overflow")?;
            if len != size || pos != end {
                return Err("blob shape/length or contiguous payload mismatch".into());
            }
            let end = pos.checked_add(len).ok_or("payload overflow")?;
            weights.get(pos..end).ok_or("short payload")?;
            spans.push(start..end);
            Source::Span(pos..end)
        } else {
            if size < 64 {
                bindings.push(None);
                return Ok(());
            }
            let b = match value_payload(v) {
                Some(m::tensor_value::Value::Floats(f)) if dtype == 2 => {
                    f.values.iter().flat_map(|f| f.to_le_bytes()).collect()
                }
                Some(m::tensor_value::Value::Bytes(b)) => b.values.to_vec(),
                _ => {
                    bindings.push(None);
                    return Ok(());
                }
            };
            if b.len() != size {
                return Err("immediate shape/length mismatch".into());
            }
            Source::Inline(b)
        };
        let b = source.bytes(weights);
        let mut h = std::collections::hash_map::DefaultHasher::new();
        b.hash(&mut h);
        let key = (dtype, b.len(), h.finish());
        let existing = hashes
            .get(&key)
            .into_iter()
            .flatten()
            .copied()
            .find(|&i| entries[i].source.bytes(weights) == b);
        let i = if let Some(i) = existing {
            i
        } else {
            let i = entries.len();
            entries.push(Entry {
                source,
                dtype,
                offset: 0,
            });
            hashes.entry(key).or_default().push(i);
            i
        };
        bindings.push(Some(i));
        Ok(())
    })?;
    spans.sort_by_key(|r| r.start);
    spans.dedup();
    if spans.windows(2).any(|a| a[0].end > a[1].start) {
        return Err("overlapping blob entries".into());
    }
    let mut order = (0..entries.len()).collect::<Vec<_>>();
    order.sort_by_key(|&i| match &entries[i].source {
        Source::Span(r) => r.start,
        Source::Inline(_) => usize::MAX,
    });
    let mut length = 64usize;
    for &i in &order {
        let len = entries[i].source.bytes(weights).len();
        entries[i].offset = length;
        length = length
            .checked_add(64)
            .and_then(|n| n.checked_add(len))
            .and_then(|n| n.checked_add(63))
            .map(|n| n & !63)
            .ok_or("weight size overflow")?;
    }
    let count = u32::try_from(entries.len()).map_err(|_| "weight count overflow")?;
    weights
        .try_reserve(length.saturating_sub(weights.len()))
        .map_err(|e| e.to_string())?;
    weights.resize(weights.len().max(length), 0);
    for i in order {
        let e = &entries[i];
        let offset = e.offset;
        let len = e.source.bytes(weights).len();
        match &e.source {
            Source::Span(r) => weights.copy_within(r.clone(), offset + 64),
            Source::Inline(b) => weights[offset + 64..offset + 64 + b.len()].copy_from_slice(b),
        }
        weights[offset..offset + 64].fill(0);
        weights[offset..offset + 4].copy_from_slice(&0xdeadbeefu32.to_le_bytes());
        weights[offset + 4..offset + 8].copy_from_slice(&e.dtype.to_le_bytes());
        weights[offset + 8..offset + 16].copy_from_slice(&(len as u64).to_le_bytes());
        weights[offset + 16..offset + 24].copy_from_slice(&((offset + 64) as u64).to_le_bytes());
    }
    weights.truncate(length);
    weights[..64].fill(0);
    weights[..4].copy_from_slice(&count.to_le_bytes());
    weights[4..8].copy_from_slice(&2u32.to_le_bytes());
    let mut bindings = bindings.into_iter();
    values(model, &mut |v| {
        if let Some(i) = bindings.next().expect("identical model traversal") {
            v.value = Some(m::value::Value::BlobFileValue(m::value::BlobFileValue {
                file_name: FILE.into(),
                offset: entries[i].offset as u64,
            }));
        }
        Ok(())
    })?;
    Ok(())
}

fn value_payload(v: &m::Value) -> Option<&m::tensor_value::Value> {
    let m::value::Value::ImmediateValue(v) = v.value.as_ref()? else {
        return None;
    };
    let m::value::immediate_value::Value::Tensor(t) = v.value.as_ref()? else {
        return None;
    };
    t.value.as_ref()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::converters::{coreml_packed::named_type, weight_file_builder::WeightFileBuilder};
    use prost::Message;
    fn model(values: Vec<m::Value>) -> Model {
        let operations = values
            .into_iter()
            .enumerate()
            .map(|(i, val)| m::Operation {
                r#type: "const".into(),
                attributes: [("val".into(), val)].into(),
                outputs: vec![named_type(&format!("v{i}"), m::DataType::Float32, &[16])],
                ..Default::default()
            })
            .collect();
        let child = m::Block {
            operations,
            ..Default::default()
        };
        Model {
            r#type: Some(model::Type::MlProgram(m::Program {
                functions: [(
                    "main".into(),
                    m::Function {
                        block_specializations: [(
                            "CoreML8".into(),
                            m::Block {
                                operations: vec![m::Operation {
                                    r#type: "cond".into(),
                                    blocks: vec![child],
                                    ..Default::default()
                                }],
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
    fn blob(offset: u64) -> m::Value {
        m::Value {
            r#type: named_type("", m::DataType::Float32, &[16]).r#type,
            value: Some(m::value::Value::BlobFileValue(m::value::BlobFileValue {
                file_name: FILE.into(),
                offset,
            })),
            ..Default::default()
        }
    }
    #[test]
    fn removes_dead_spans_deduplicates_exact_bytes_and_keeps_nan_payloads() {
        let payload = [0x7fc01234u32, 0x80000000, 1, 0x7f800000]
            .repeat(4)
            .iter()
            .flat_map(|n| n.to_le_bytes())
            .collect::<Vec<_>>();
        let mut builder = WeightFileBuilder::new();
        builder.add_weight(0, 2, &vec![0; 256]).unwrap();
        let live = builder.add_weight(1, 2, &payload).unwrap();
        let duplicate = builder.add_weight(2, 2, &payload).unwrap();
        let mut weights = builder.finalize();
        let mut m = model(vec![blob(live), blob(duplicate)]);
        compact(&mut m, &mut weights).unwrap();
        assert_eq!(weights.len(), 192);
        assert_eq!(&weights[128..192], payload);
        values(&mut m, &mut |v| {
            if let Some(m::value::Value::BlobFileValue(b)) = &v.value {
                assert_eq!(b.offset, 64);
            }
            Ok(())
        })
        .unwrap();
        let previous = weights.clone();
        compact(&mut m, &mut weights).unwrap();
        assert_eq!(weights, previous);
    }
    #[test]
    fn invalid_later_blob_is_transactional() {
        let mut b = WeightFileBuilder::new();
        let offset = b.add_weight(0, 2, &[0; 64]).unwrap();
        let mut weights = b.finalize();
        let mut m = model(vec![blob(offset), blob(u64::MAX)]);
        let before = m.encode_to_vec();
        let old = weights.clone();
        assert!(compact(&mut m, &mut weights).is_err());
        assert_eq!(m.encode_to_vec(), before);
        assert_eq!(weights, old);
    }
    #[test]
    fn externalizes_and_shares_immediate_values_without_changing_bits() {
        let v = m::Value {
            r#type: named_type("", m::DataType::Float32, &[16]).r#type,
            value: Some(m::value::Value::ImmediateValue(m::value::ImmediateValue {
                value: Some(m::value::immediate_value::Value::Tensor(m::TensorValue {
                    value: Some(m::tensor_value::Value::Floats(
                        m::tensor_value::RepeatedFloats {
                            values: vec![f32::from_bits(0x7fc00017); 16],
                        },
                    )),
                })),
            })),
            ..Default::default()
        };
        let mut m = model(vec![v.clone(), v]);
        let mut weights = vec![];
        compact(&mut m, &mut weights).unwrap();
        assert_eq!(weights.len(), 192);
        assert_eq!(
            u32::from_le_bytes(weights[128..132].try_into().unwrap()),
            0x7fc00017
        );
    }
}
