//! Compact only documented source BlobFile entries; never edit compiled assets.

use std::collections::BTreeMap;

const ALIGNMENT: usize = 64;
const MAX_DEPTH: usize = 128;
type Result<T> = std::result::Result<T, RepackError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RepackError {
    /// Keep the original source route and record why it cannot be compacted.
    Unsupported(&'static str),
    /// Malformed or contradictory source data must not reach the compiler.
    Invalid(String),
}

impl std::fmt::Display for RepackError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported(reason) => write!(formatter, "unsupported source repack: {reason}"),
            Self::Invalid(reason) => write!(formatter, "invalid source repack: {reason}"),
        }
    }
}

impl std::error::Error for RepackError {}

impl From<&str> for RepackError {
    fn from(reason: &str) -> Self {
        Self::Invalid(reason.to_owned())
    }
}

impl From<String> for RepackError {
    fn from(reason: String) -> Self {
        Self::Invalid(reason)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PayloadMapping {
    pub old_metadata_offset: u64,
    pub new_metadata_offset: u64,
    pub blob_dtype: u32,
    pub payload_bytes: usize,
}

#[derive(Debug)]
pub struct RepackedChild {
    pub source: Vec<u8>,
    pub weights: Vec<u8>,
    pub mappings: Vec<PayloadMapping>,
}

#[derive(Clone, Copy)]
struct Field<'a> {
    number: u64,
    wire: u8,
    raw: &'a [u8],
    key: &'a [u8],
    payload: &'a [u8],
}

fn varint(bytes: &[u8], cursor: &mut usize) -> Result<u64> {
    let mut value = 0u64;
    for shift in (0..70).step_by(7) {
        let byte = *bytes.get(*cursor).ok_or("truncated source protobuf")?;
        *cursor += 1;
        if shift == 63 && byte > 1 {
            return Err("source protobuf varint overflows u64".into());
        }
        value |= u64::from(byte & 127) << shift;
        if byte & 128 == 0 {
            return Ok(value);
        }
    }
    Err("invalid source protobuf varint".into())
}

fn fields(bytes: &[u8]) -> Result<Vec<Field<'_>>> {
    let mut cursor = 0;
    let mut result = Vec::new();
    while cursor < bytes.len() {
        let start = cursor;
        let key = varint(bytes, &mut cursor)?;
        let key_end = cursor;
        if key >> 3 == 0 {
            return Err("source protobuf field number is zero".into());
        }
        let wire = (key & 7) as u8;
        let (payload_start, length) = match wire {
            0 => {
                let payload_start = cursor;
                varint(bytes, &mut cursor)?;
                (payload_start, cursor - payload_start)
            }
            1 => (cursor, 8),
            2 => {
                let length = usize::try_from(varint(bytes, &mut cursor)?)
                    .map_err(|_| "source protobuf length overflows usize")?;
                (cursor, length)
            }
            5 => (cursor, 4),
            _ => return Err("unsupported source protobuf wire type".into()),
        };
        let end = payload_start
            .checked_add(length)
            .filter(|&end| end <= bytes.len())
            .ok_or("truncated source protobuf field")?;
        cursor = end;
        result.push(Field {
            number: key >> 3,
            wire,
            raw: &bytes[start..end],
            key: &bytes[start..key_end],
            payload: &bytes[payload_start..end],
        });
    }
    Ok(result)
}

fn encode_varint(mut value: u64, output: &mut Vec<u8>) {
    while value >= 128 {
        output.push((value as u8 & 127) | 128);
        value >>= 7;
    }
    output.push(value as u8);
}

#[derive(Clone, Copy)]
enum Kind {
    Model,
    Program,
    Function,
    Block,
    Operation,
    Argument,
    Binding,
    Value,
    Blob,
    Immediate,
    TupleValue,
    ListValue,
    DictionaryValue,
    KeyValuePair,
    NamedType,
    ValueType,
    TensorType,
    TupleType,
    ListType,
    DictionaryType,
    StateType,
    MapFunction,
    MapBlock,
    MapArgument,
    MapValue,
}

// Tags come from the public Model.proto/MIL.proto schema, not string patterns.
fn nested(kind: Kind, number: u64) -> Option<Kind> {
    use Kind::*;
    match (kind, number) {
        (Model, 502) => Some(Program),
        (Program, 2) => Some(MapFunction),
        (Program | Function, 4) => Some(MapValue),
        (Function, 1) | (Block, 1) | (Operation, 3) => Some(NamedType),
        (Function, 3) => Some(MapBlock),
        (Block, 3) => Some(Operation),
        (Block, 4) | (Operation, 5) | (TensorType, 4) => Some(MapValue),
        (Operation, 2) => Some(MapArgument),
        (Operation, 4) => Some(Block),
        (Argument, 1) => Some(Binding),
        (Binding, 2) => Some(Value),
        (Value, 2) | (NamedType, 2) => Some(ValueType),
        (Value, 3) => Some(Immediate),
        (Value, 5) => Some(Blob),
        (Immediate, 2) => Some(TupleValue),
        (Immediate, 3) => Some(ListValue),
        (Immediate, 4) => Some(DictionaryValue),
        (TupleValue | ListValue, 1) => Some(Value),
        (DictionaryValue, 1) => Some(KeyValuePair),
        (KeyValuePair, 1 | 2) => Some(Value),
        (ValueType, 1) => Some(TensorType),
        (ValueType, 2) => Some(ListType),
        (ValueType, 3) => Some(TupleType),
        (ValueType, 4) => Some(DictionaryType),
        (ValueType, 5) => Some(StateType),
        (TupleType, 1) | (ListType, 1) | (StateType, 1) => Some(ValueType),
        (DictionaryType, 1 | 2) => Some(ValueType),
        (MapFunction, 2) => Some(Function),
        (MapBlock, 2) => Some(Block),
        (MapArgument, 2) => Some(Argument),
        (MapValue, 2) => Some(Value),
        _ => None,
    }
}

fn unique<'a>(fields: &[Field<'a>], number: u64, wire: u8) -> Result<Option<Field<'a>>> {
    let mut found = None;
    for &field in fields.iter().filter(|field| field.number == number) {
        if field.wire != wire || found.is_some() {
            return Err(format!("ambiguous source field {number}").into());
        }
        found = Some(field);
    }
    Ok(found)
}

fn scalar(fields: &[Field<'_>], number: u64) -> Result<u64> {
    unique(fields, number, 0)?
        .map(|field| varint(field.payload, &mut 0))
        .transpose()
        .map(|value| value.unwrap_or(0))
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Reference {
    offset: u64,
    header_type: u32,
    bytes: usize,
}

fn reference(value: &[u8]) -> Result<Option<Reference>> {
    let value = fields(value)?;
    let Some(blob) = unique(&value, 5, 2)? else {
        return Ok(None);
    };
    if unique(&value, 3, 2)?.is_some() {
        return Err("source Value contains both immediate and blob variants".into());
    }
    let blob = fields(blob.payload)?;
    let offset = scalar(&blob, 2)?;
    let path = unique(&blob, 1, 2)?.ok_or("missing source BlobFile path")?;
    let alternate_path = path.payload != b"@model_path/weights/weights.bin";
    if alternate_path {
        let path =
            std::str::from_utf8(path.payload).map_err(|_| "source BlobFile path is not UTF8")?;
        let relative = path
            .strip_prefix("@model_path/")
            .ok_or("invalid source BlobFile path prefix")?;
        if relative
            .split('/')
            .any(|part| part.is_empty() || matches!(part, "." | ".."))
            || relative
                .chars()
                .any(|character| character.is_control() || matches!(character, '\\' | ':' | '"'))
        {
            return Err("invalid source BlobFile path traversal or encoding".into());
        }
    }
    let value_type = unique(&value, 2, 2)?.ok_or("missing source BlobFile type")?;
    let value_type = fields(value_type.payload)?;
    let type_variants = (1..=5)
        .map(|number| unique(&value_type, number, 2))
        .collect::<Result<Vec<_>>>()?;
    if type_variants.iter().flatten().count() > 1 {
        return Err("ambiguous source BlobFile value type".into());
    }
    let tensor = match type_variants[0] {
        Some(tensor) => tensor,
        None if type_variants.iter().any(Option::is_some) => {
            return Err(RepackError::Unsupported("non-tensor source BlobFile type"));
        }
        None => return Err("missing source BlobFile tensor type".into()),
    };
    let tensor = fields(tensor.payload)?;
    let storage = match scalar(&tensor, 1)? {
        10 => Some((1, 2)),
        11 => Some((2, 4)),
        31 => Some((3, 1)),
        21 => Some((4, 1)),
        0 => return Err("source BlobFile tensor dtype is unspecified".into()),
        _ => None,
    };
    let rank =
        usize::try_from(scalar(&tensor, 2)?).map_err(|_| "invalid source BlobFile tensor rank")?;
    let dimensions: Vec<_> = tensor.iter().filter(|field| field.number == 3).collect();
    if dimensions.len() != rank {
        return Err("source BlobFile tensor rank/shape mismatch".into());
    }
    let mut elements: usize = 1;
    let mut non_static = false;
    for dimension in dimensions {
        if dimension.wire != 2 {
            return Err("invalid source BlobFile dimension encoding".into());
        }
        let dimension = fields(dimension.payload)?;
        let constant = unique(&dimension, 1, 2)?;
        let unknown = unique(&dimension, 2, 2)?;
        if constant.is_some() && unknown.is_some() {
            return Err("ambiguous source BlobFile dimension".into());
        }
        let constant = match (constant, unknown) {
            (Some(constant), _) => constant,
            (None, Some(_)) => {
                non_static = true;
                continue;
            }
            (None, None) => return Err("missing source BlobFile dimension".into()),
        };
        let extent = usize::try_from(scalar(&fields(constant.payload)?, 1)?)
            .map_err(|_| "source BlobFile extent overflows usize")?;
        elements = elements
            .checked_mul(extent)
            .ok_or("source BlobFile shape overflows usize")?;
    }
    let Some((header_type, element_bytes)) = storage else {
        return Err(RepackError::Unsupported("source BlobFile tensor dtype"));
    };
    if non_static {
        return Err(RepackError::Unsupported("non-static source BlobFile shape"));
    }
    let bytes = elements
        .checked_mul(element_bytes)
        .ok_or("source BlobFile byte size overflows usize")?;
    // Validate known tensor fields before falling back for another safe asset
    // path. An ambiguous or malformed source must never select fallback.
    if alternate_path {
        return Err(RepackError::Unsupported(
            "alternate source weight asset path",
        ));
    }
    Ok(Some(Reference {
        offset,
        header_type,
        bytes,
    }))
}

fn collect_known(
    kind: Kind,
    source: &[u8],
    depth: usize,
    output: &mut Vec<Reference>,
    standard_offsets: &mut Vec<u64>,
    pending: &mut Option<&'static str>,
) -> Result<()> {
    if depth > MAX_DEPTH {
        return Err("source MIL nesting exceeds supported depth".into());
    }
    if matches!(kind, Kind::Value) {
        // Entry boundaries do not depend on the tensor dtype. Retain the
        // offset even when another known field selects unsupported fallback.
        let value = fields(source)?;
        if let Some(blob) = unique(&value, 5, 2)? {
            let blob = fields(blob.payload)?;
            let offset = scalar(&blob, 2)?;
            if unique(&blob, 1, 2)?
                .is_some_and(|path| path.payload == b"@model_path/weights/weights.bin")
            {
                standard_offsets.push(offset);
            }
        }
        match reference(source) {
            Ok(Some(value)) => output.push(value),
            Ok(None) => {}
            Err(RepackError::Unsupported(reason)) => {
                pending.get_or_insert(reason);
            }
            Err(error) => return Err(error),
        }
    }
    for field in fields(source)? {
        if let Some(child) = nested(kind, field.number) {
            if field.wire != 2 {
                return Err("source MIL nested message has incorrect wire type".into());
            }
            collect_known(
                child,
                field.payload,
                depth + 1,
                output,
                standard_offsets,
                pending,
            )?;
        }
    }
    Ok(())
}

fn collect(kind: Kind, source: &[u8], depth: usize, output: &mut Vec<Reference>) -> Result<()> {
    let mut pending = None;
    collect_known(kind, source, depth, output, &mut vec![], &mut pending)?;
    pending.map_or(Ok(()), |reason| Err(RepackError::Unsupported(reason)))
}

fn rewrite(kind: Kind, source: &[u8], offsets: &BTreeMap<u64, u64>) -> Result<Vec<u8>> {
    let mut result = Vec::with_capacity(source.len());
    for field in fields(source)? {
        if matches!(kind, Kind::Blob) && field.number == 2 {
            if field.wire != 0 {
                return Err("source BlobFile offset has incorrect wire type".into());
            }
            let old = varint(field.payload, &mut 0)?;
            let new = *offsets
                .get(&old)
                .ok_or("source BlobFile offset was not inventoried")?;
            if old == new {
                result.extend_from_slice(field.raw);
            } else {
                result.extend_from_slice(field.key);
                encode_varint(new, &mut result);
            }
        } else if let Some(child) = nested(kind, field.number) {
            let rewritten = rewrite(child, field.payload, offsets)?;
            if rewritten == field.payload {
                result.extend_from_slice(field.raw);
            } else {
                result.extend_from_slice(field.key);
                encode_varint(rewritten.len() as u64, &mut result);
                result.extend_from_slice(&rewritten);
            }
        } else {
            result.extend_from_slice(field.raw);
        }
    }
    Ok(result)
}

#[derive(Clone, Copy)]
struct Entry<'a> {
    metadata: &'a [u8],
    payload: &'a [u8],
    dtype: u32,
}

fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn u64_at(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

fn aligned(value: usize) -> Result<usize> {
    value
        .checked_add(63)
        .map(|value| value & !63)
        .ok_or("source BlobFile length overflow".into())
}

fn entries_known<'a>(
    weights: &'a [u8],
    pending: &mut Option<&'static str>,
) -> Result<BTreeMap<u64, Entry<'a>>> {
    if weights.len() < ALIGNMENT || !weights.len().is_multiple_of(ALIGNMENT) {
        return Err("source weight file is truncated or misaligned".into());
    }
    if u32_at(weights, 4) == 0 {
        return Err("source weight file version is zero".into());
    }
    if u32_at(weights, 4) != 2 {
        return Err(RepackError::Unsupported("source weight file version"));
    }
    if weights[8..64].iter().any(|&byte| byte != 0) {
        return Err("source weight file reserved header bytes are nonzero".into());
    }
    let count = usize::try_from(u32_at(weights, 0)).map_err(|_| "weight entry count overflow")?;
    if count > (weights.len() - 64) / 64 {
        return Err("source weight entry count exceeds file size".into());
    }
    let mut result = BTreeMap::new();
    let mut cursor: usize = 64;
    for _ in 0..count {
        let metadata_end = cursor.checked_add(64).ok_or("weight metadata overflow")?;
        let metadata = weights
            .get(cursor..metadata_end)
            .ok_or("truncated source weight metadata")?;
        if u32_at(metadata, 0) != 0xDEADBEEF || metadata[24..64].iter().any(|&byte| byte != 0) {
            return Err("invalid source weight metadata".into());
        }
        let dtype = u32_at(metadata, 4);
        if dtype == 0 {
            return Err("source weight metadata dtype is unspecified".into());
        }
        let payload_location = usize::try_from(u64_at(metadata, 16))
            .map_err(|_| "source weight payload offset overflows usize")?;
        if payload_location < metadata_end || !payload_location.is_multiple_of(ALIGNMENT) {
            return Err("source weight payload overlaps metadata or is misaligned".into());
        }
        let length =
            usize::try_from(u64_at(metadata, 8)).map_err(|_| "weight payload exceeds usize")?;
        let end = payload_location
            .checked_add(length)
            .ok_or("weight payload overflow")?;
        let payload = weights
            .get(payload_location..end)
            .ok_or("truncated source weight payload")?;
        if !(1..=4).contains(&dtype) {
            pending.get_or_insert("source weight metadata dtype");
        }
        if payload_location != metadata_end {
            pending.get_or_insert("noncontiguous source weight payload layout");
        }
        let next = aligned(end)?;
        let padding = weights
            .get(end..next)
            .ok_or("truncated source weight padding")?;
        if padding.iter().any(|&byte| byte != 0) {
            return Err("source weight padding is nonzero".into());
        }
        result.insert(
            cursor as u64,
            Entry {
                metadata,
                payload,
                dtype,
            },
        );
        cursor = next;
    }
    if cursor != weights.len() {
        return Err("source weight count leaves unindexed bytes".into());
    }
    Ok(result)
}

fn entries(weights: &[u8]) -> Result<BTreeMap<u64, Entry<'_>>> {
    let mut pending = None;
    let entries = entries_known(weights, &mut pending)?;
    match pending {
        Some(reason) => Err(RepackError::Unsupported(reason)),
        None => Ok(entries),
    }
}

/// Return a derived child source and compact sidecar. The original inputs are
/// untouched; callers must retain original/derived hashes for qualification.
/// Payloads are copied byte-for-byte and metadata changes only its absolute
/// payload offset. This recognizes the exact v2 layout emitted by this crate.
pub fn repack_child(source: &[u8], weights: &[u8]) -> Result<Option<RepackedChild>> {
    let model = fields(source)?;
    const MODEL_TYPES: &[u64] = &[
        200, 201, 202, 300, 301, 302, 303, 304, 400, 401, 402, 403, 404, 500, 501, 502, 555, 556,
        560, 600, 601, 602, 603, 604, 606, 607, 609, 610, 900, 2000, 2001, 2002, 2003, 2004, 2005,
        2006, 3000,
    ];
    let mut model_types = 0;
    for &number in MODEL_TYPES {
        model_types += usize::from(unique(&model, number, 2)?.is_some());
    }
    if model_types > 1 {
        return Err("ambiguous source Model type".into());
    }
    if unique(&model, 502, 2)?.is_none() {
        return if model_types == 1 {
            Err(RepackError::Unsupported("non-MLProgram source model"))
        } else {
            Err("missing source Model type".into())
        };
    }
    let mut references = Vec::new();
    let mut standard_offsets = Vec::new();
    let mut pending = None;
    collect_known(
        Kind::Model,
        source,
        0,
        &mut references,
        &mut standard_offsets,
        &mut pending,
    )?;
    if references.is_empty() && pending.is_none() {
        return Ok(None);
    }
    let original_entries = match entries_known(weights, &mut pending) {
        Ok(entries) => entries,
        Err(RepackError::Unsupported(reason)) => {
            pending.get_or_insert(reason);
            // A different file version cannot be parsed as known v2 layout.
            return Err(RepackError::Unsupported(pending.unwrap()));
        }
        Err(error) => return Err(error),
    };
    for offset in standard_offsets {
        if !original_entries.contains_key(&offset) {
            return Err("source BlobFile offset is not an entry boundary".into());
        }
    }
    let mut used = BTreeMap::new();
    for reference in references {
        let entry = original_entries
            .get(&reference.offset)
            .ok_or("source BlobFile offset is not an entry boundary")?;
        if (1..=4).contains(&entry.dtype) && entry.dtype != reference.header_type
            || entry.payload.len() != reference.bytes
        {
            return Err("source BlobFile dtype/shape disagrees with its physical header".into());
        }
        if let Some(previous) = used.insert(reference.offset, reference.clone())
            && previous != reference
        {
            return Err("source BlobFile aliases have conflicting tensor types".into());
        }
    }
    if let Some(reason) = pending {
        return Err(RepackError::Unsupported(reason));
    }
    let total = used.keys().try_fold(64usize, |total, offset| {
        aligned(
            total
                .checked_add(64)
                .and_then(|value| value.checked_add(original_entries[offset].payload.len()))
                .ok_or("compact source weight size overflow")?,
        )
    })?;
    let mut compact = Vec::with_capacity(total);
    compact.extend_from_slice(&weights[..64]);
    compact[..4].copy_from_slice(&(used.len() as u32).to_le_bytes());
    let mut offsets = BTreeMap::new();
    let mut mappings = Vec::new();
    for (&old, reference) in &used {
        let entry = original_entries[&old];
        let new = compact.len() as u64;
        offsets.insert(old, new);
        compact.extend_from_slice(entry.metadata);
        let payload_location = usize::try_from(new).unwrap() + 16;
        compact[payload_location..payload_location + 8].copy_from_slice(&(new + 64).to_le_bytes());
        compact.extend_from_slice(entry.payload);
        compact.resize(aligned(compact.len())?, 0);
        mappings.push(PayloadMapping {
            old_metadata_offset: old,
            new_metadata_offset: new,
            blob_dtype: reference.header_type,
            payload_bytes: entry.payload.len(),
        });
    }
    let source = rewrite(Kind::Model, source, &offsets)?;
    // Re-inventory derived references and compare all original payload bytes.
    let mut derived = Vec::new();
    collect(Kind::Model, &source, 0, &mut derived)?;
    let derived_entries = entries(&compact)?;
    for reference in derived {
        let entry = derived_entries
            .get(&reference.offset)
            .ok_or("derived BlobFile offset is invalid")?;
        if entry.dtype != reference.header_type || entry.payload.len() != reference.bytes {
            return Err("derived BlobFile type differs from its source".into());
        }
    }
    for mapping in &mappings {
        if original_entries[&mapping.old_metadata_offset].payload
            != derived_entries[&mapping.new_metadata_offset].payload
        {
            return Err("source weight payload changed during repacking".into());
        }
    }
    Ok(Some(RepackedChild {
        source,
        weights: compact,
        mappings,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn integer(number: u64, value: u64) -> Vec<u8> {
        let mut result = vec![];
        encode_varint(number << 3, &mut result);
        encode_varint(value, &mut result);
        result
    }

    fn message(number: u64, bytes: &[u8]) -> Vec<u8> {
        let mut result = vec![];
        encode_varint((number << 3) | 2, &mut result);
        encode_varint(bytes.len() as u64, &mut result);
        result.extend_from_slice(bytes);
        result
    }

    fn concatenate(items: &[Vec<u8>]) -> Vec<u8> {
        items.iter().flatten().copied().collect()
    }

    fn tensor_value(dtype: u64, shape: &[u64], offset: u64) -> Vec<u8> {
        let mut tensor = concatenate(&[integer(1, dtype), integer(2, shape.len() as u64)]);
        for &extent in shape {
            tensor.extend(message(3, &message(1, &integer(1, extent))));
        }
        concatenate(&[
            message(2, &message(1, &tensor)),
            message(
                5,
                &concatenate(&[
                    message(1, b"@model_path/weights/weights.bin"),
                    integer(2, offset),
                    // Preserve an unknown BlobFile extension and its wire bytes.
                    message(127, b"opaque-extension"),
                ]),
            ),
        ])
    }

    fn model(values: &[Vec<u8>]) -> Vec<u8> {
        let mut operation = message(1, b"const");
        for (index, value) in values.iter().enumerate() {
            operation.extend(message(
                5,
                &concatenate(&[
                    message(1, format!("weight{index}").as_bytes()),
                    message(2, value),
                ]),
            ));
        }
        let block = concatenate(&[message(3, &operation), message(111, b"unknown-block")]);
        let function = message(
            3,
            &concatenate(&[message(1, b"CoreML7"), message(2, &block)]),
        );
        let program = message(
            2,
            &concatenate(&[message(1, b"main"), message(2, &function)]),
        );
        concatenate(&[
            integer(1, 9),
            message(502, &program),
            // This unrelated opaque field contains bytes resembling a BlobFile.
            message(1001, &tensor_value(10, &[1], 9999)),
        ])
    }

    fn weight_file(items: &[(u32, Vec<u8>)]) -> (Vec<u8>, Vec<u64>) {
        let mut weights = vec![0; 64];
        weights[..4].copy_from_slice(&(items.len() as u32).to_le_bytes());
        weights[4..8].copy_from_slice(&2u32.to_le_bytes());
        let mut offsets = vec![];
        for (dtype, payload) in items {
            let start = weights.len();
            offsets.push(start as u64);
            let mut metadata = vec![0; 64];
            metadata[..4].copy_from_slice(&0xDEADBEEFu32.to_le_bytes());
            metadata[4..8].copy_from_slice(&dtype.to_le_bytes());
            metadata[8..16].copy_from_slice(&(payload.len() as u64).to_le_bytes());
            metadata[16..24].copy_from_slice(&((start + 64) as u64).to_le_bytes());
            weights.extend(metadata);
            weights.extend_from_slice(payload);
            weights.resize(aligned(weights.len()).unwrap(), 0);
        }
        (weights, offsets)
    }

    #[test]
    fn drops_unreferenced_entries_and_preserves_all_typed_payload_bits() {
        for (mil, header, payload) in [
            (10, 1, vec![0, 128, 1, 0, 1, 124, 0, 252]),
            (11, 2, vec![0, 0, 0, 128, 1, 0, 128, 127]),
            (21, 4, vec![128, 127, 255, 0]),
            (31, 3, vec![0, 127, 128, 255]),
        ] {
            let (weights, offsets) = weight_file(&[(1, vec![1; 4096]), (header, payload.clone())]);
            let bytes = payload.len()
                / if mil == 10 {
                    2
                } else if mil == 11 {
                    4
                } else {
                    1
                };
            let source = model(&[tensor_value(mil, &[bytes as u64], offsets[1])]);
            let original_source = source.clone();
            let original_weights = weights.clone();
            let compact = repack_child(&source, &weights).unwrap().unwrap();
            assert!(compact.weights.len() < weights.len());
            assert_eq!(compact.weights.len(), 192);
            assert_eq!(u32_at(&compact.weights, 0), 1);
            assert_eq!(u32_at(&compact.weights, 68), header);
            assert_eq!(&compact.weights[128..128 + payload.len()], payload);
            assert_eq!(compact.mappings[0].old_metadata_offset, offsets[1]);
            assert_eq!(compact.mappings[0].new_metadata_offset, 64);
            assert_eq!(source, original_source);
            assert_eq!(weights, original_weights);
            let mut derived = vec![];
            collect(Kind::Model, &compact.source, 0, &mut derived).unwrap();
            assert_eq!(derived[0].offset, 64);
            // Restoring the one rewritten offset restores every source byte,
            // including unknown fields, unrelated literals, and original order.
            let restored =
                rewrite(Kind::Model, &compact.source, &[(64, offsets[1])].into()).unwrap();
            assert_eq!(restored, source);
        }
    }

    #[test]
    fn repeated_references_share_one_entry_but_distinct_entries_keep_their_tags() {
        let (weights, offsets) = weight_file(&[(1, vec![0, 128]), (4, vec![255])]);
        let first = tensor_value(10, &[1], offsets[0]);
        let second = tensor_value(21, &[], offsets[1]);
        let compact = repack_child(&model(&[first.clone(), first, second]), &weights)
            .unwrap()
            .unwrap();
        assert_eq!(compact.mappings.len(), 2);
        assert_eq!(u32_at(&compact.weights, 0), 2);
        assert_eq!(entries(&compact.weights).unwrap().len(), 2);
    }

    #[test]
    fn rejects_wrong_dtype_and_conflicting_alias_shapes() {
        let (weights, offsets) = weight_file(&[(3, vec![128, 255])]);
        assert!(repack_child(&model(&[tensor_value(21, &[2], offsets[0])]), &weights).is_err());
        assert!(
            repack_child(
                &model(&[
                    tensor_value(31, &[2], offsets[0]),
                    tensor_value(31, &[1], offsets[0]),
                ]),
                &weights
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_malformed_header_alignment_overlap_and_truncation() {
        let (weights, offsets) = weight_file(&[(1, vec![1, 0])]);
        let source = model(&[tensor_value(10, &[1], offsets[0])]);
        for (location, value) in [
            (0, 255),
            (4, 3),
            (8, 1),
            (64, 0),
            (68, 255),
            (80, 64),
            (88, 1),
        ] {
            let mut malformed = weights.clone();
            malformed[location] = value;
            assert!(repack_child(&source, &malformed).is_err(), "byte{location}");
        }
        assert!(repack_child(&source, &weights[..weights.len() - 1]).is_err());
        assert!(repack_child(&model(&[tensor_value(10, &[1], offsets[0] + 1)]), &weights).is_err());
        assert!(repack_child(&model(&[tensor_value(10, &[1], u64::MAX)]), &weights).is_err());
    }

    #[test]
    fn rejects_ambiguous_blob_fields_and_untrusted_paths() {
        let (weights, offsets) = weight_file(&[(1, vec![1, 0])]);
        let value = tensor_value(10, &[1], offsets[0]);
        let mut ambiguous = value.clone();
        ambiguous.extend(message(5, &message(1, b"@model_path/weights/weights.bin")));
        assert!(repack_child(&model(&[ambiguous]), &weights).is_err());
        let invalid = value
            .windows(b"weights/weights.bin".len())
            .position(|v| v == b"weights/weights.bin")
            .unwrap();
        let mut untrusted = value;
        untrusted[invalid..invalid + 3].copy_from_slice(b"../");
        assert!(repack_child(&model(&[untrusted]), &weights).is_err());
    }

    #[test]
    fn supports_nested_value_containers_and_does_not_touch_immediate_literals() {
        let (weights, offsets) = weight_file(&[(3, vec![0; 128]), (1, vec![1, 0])]);
        let referenced = tensor_value(10, &[1], offsets[1]);
        let tuple = message(3, &message(2, &message(1, &referenced)));
        let list = message(3, &message(3, &message(1, &referenced)));
        let dictionary = message(3, &message(4, &message(1, &message(2, &referenced))));
        let compact = repack_child(&model(&[tuple, list, dictionary]), &weights)
            .unwrap()
            .unwrap();
        assert_eq!(compact.mappings.len(), 1);
        let mut refs = vec![];
        collect(Kind::Model, &compact.source, 0, &mut refs).unwrap();
        assert_eq!(refs.len(), 3);
    }

    #[test]
    fn source_without_blob_does_not_need_or_retain_weights() {
        assert!(repack_child(&model(&[]), &[]).unwrap().is_none());
    }

    #[test]
    fn lazy_source_preparation_retains_originals_and_writes_only_referenced_entries() {
        use super::super::{Stage, StageExecution, prepare_child_source};
        let (weights, offsets) = weight_file(&[(1, vec![1; 4096]), (1, vec![0, 128, 1, 0])]);
        let source = model(&[tensor_value(10, &[2], offsets[1])]);
        let stage = Stage {
            source: source.clone(),
            inputs: vec![],
            outputs: vec![],
            release_after: vec![],
            has_weights: true,
            execution: StageExecution::Native,
        };
        let root = tempfile::tempdir().unwrap();
        let shared = root.path().join("shared.bin");
        std::fs::write(&shared, &weights).unwrap();
        let derived = tempfile::tempdir_in(root.path()).unwrap();
        assert!(
            prepare_child_source(&stage, derived.path(), &shared, Some(&weights))
                .unwrap()
                .is_none()
        );
        let expected = repack_child(&source, &weights).unwrap().unwrap();
        assert_eq!(
            std::fs::read(derived.path().join("model.mlmodel")).unwrap(),
            expected.source
        );
        assert_eq!(
            std::fs::read(derived.path().join("weights/weights.bin")).unwrap(),
            expected.weights
        );
        assert_eq!(stage.source, source);
        assert_eq!(std::fs::read(&shared).unwrap(), weights);
    }

    #[test]
    fn lazy_preparation_distinguishes_explicit_unsupported_fallback_from_corruption() {
        use super::super::{Stage, StageExecution, prepare_child_source};
        let (weights, offsets) = weight_file(&[(1, vec![1, 0])]);
        let source = model(&[tensor_value(10, &[1], offsets[0])]);
        let stage = Stage {
            source: source.clone(),
            inputs: vec![],
            outputs: vec![],
            release_after: vec![],
            has_weights: true,
            execution: StageExecution::Native,
        };
        let root = tempfile::tempdir().unwrap();
        let shared = root.path().join("shared.bin");
        let mut future = weights.clone();
        future[4..8].copy_from_slice(&3u32.to_le_bytes());
        std::fs::write(&shared, &future).unwrap();
        let derived = tempfile::tempdir_in(root.path()).unwrap();
        let diagnostic = prepare_child_source(&stage, derived.path(), &shared, Some(&future))
            .unwrap()
            .unwrap();
        assert!(
            diagnostic
                .reason
                .contains("retained original source: source weight file version")
        );
        assert!(diagnostic.compute_units.is_none());
        assert_eq!(
            std::fs::read(derived.path().join("model.mlmodel")).unwrap(),
            source
        );
        assert_eq!(
            std::fs::read(derived.path().join("weights/weights.bin")).unwrap(),
            future
        );
        let mut corrupt = weights;
        corrupt[64] = 0;
        std::fs::write(&shared, &corrupt).unwrap();
        let refused = tempfile::tempdir_in(root.path()).unwrap();
        assert!(prepare_child_source(&stage, refused.path(), &shared, Some(&corrupt)).is_err());
        assert!(!refused.path().join("model.mlmodel").exists());
        assert!(!refused.path().join("weights/weights.bin").exists());
    }

    fn assert_invalid(source: &[u8], weights: &[u8]) {
        assert!(
            matches!(repack_child(source, weights), Err(RepackError::Invalid(_))),
            "a malformed source must not use unsupported fallback"
        );
    }

    fn assert_unsupported(source: &[u8], weights: &[u8], reason: &'static str) {
        assert_eq!(
            repack_child(source, weights).unwrap_err(),
            RepackError::Unsupported(reason)
        );
    }

    #[test]
    fn typed_header_errors_distinguish_future_layout_from_corruption() {
        let (weights, offsets) = weight_file(&[(1, vec![1, 0])]);
        let source = model(&[tensor_value(10, &[1], offsets[0])]);
        let mut future = weights.clone();
        future[4..8].copy_from_slice(&3u32.to_le_bytes());
        assert_unsupported(&source, &future, "source weight file version");
        let mut unspecified = weights.clone();
        unspecified[4..8].copy_from_slice(&0u32.to_le_bytes());
        assert_invalid(&source, &unspecified);
        future = weights.clone();
        future[68..72].copy_from_slice(&255u32.to_le_bytes());
        assert_unsupported(&source, &future, "source weight metadata dtype");
        // A future dtype cannot hide a provably truncated payload.
        future[72..80].copy_from_slice(&u64::MAX.to_le_bytes());
        assert_invalid(&source, &future);
        unspecified = weights.clone();
        unspecified[68..72].copy_from_slice(&0u32.to_le_bytes());
        assert_invalid(&source, &unspecified);
        let mut separate = weights.clone();
        separate.resize(256, 0);
        separate[80..88].copy_from_slice(&192u64.to_le_bytes());
        separate[192..194].copy_from_slice(&[1, 0]);
        assert_unsupported(
            &source,
            &separate,
            "noncontiguous source weight payload layout",
        );
        separate[80..88].copy_from_slice(&64u64.to_le_bytes());
        assert_invalid(&source, &separate);
        for location in [0, 8, 64, 88] {
            let mut malformed = weights.clone();
            malformed[location] ^= 1;
            assert_invalid(&source, &malformed);
        }
        assert_invalid(&source, &weights[..weights.len() - 1]);
        assert_invalid(&model(&[tensor_value(31, &[2], offsets[0])]), &weights);
        assert_invalid(&model(&[tensor_value(10, &[2], offsets[0])]), &weights);
    }

    #[test]
    fn typed_source_errors_distinguish_future_types_from_ambiguous_fields() {
        let (weights, offsets) = weight_file(&[(1, vec![1, 0])]);
        assert_unsupported(
            &model(&[tensor_value(22, &[1], offsets[0])]),
            &weights,
            "source BlobFile tensor dtype",
        );
        assert_invalid(&model(&[tensor_value(0, &[1], offsets[0])]), &weights);
        let value = tensor_value(10, &[1], offsets[0]);
        let mut ambiguous = value.clone();
        ambiguous.extend(message(2, &message(2, &[])));
        assert_invalid(&model(&[ambiguous]), &weights);
        let mut value_fields = fields(&value).unwrap();
        let tensor_type = value_fields.remove(0).payload;
        let mut both = tensor_type.to_vec();
        both.extend(message(2, &[]));
        let ambiguous = concatenate(&[message(2, &both), value_fields[0].raw.to_vec()]);
        assert_invalid(&model(&[ambiguous]), &weights);
        let non_tensor = concatenate(&[message(2, &message(2, &[])), value_fields[0].raw.to_vec()]);
        assert_unsupported(
            &model(&[non_tensor]),
            &weights,
            "non-tensor source BlobFile type",
        );
        let mut both_models = model(std::slice::from_ref(&value));
        both_models.extend(message(500, &[]));
        assert_invalid(&both_models, &weights);
        assert_unsupported(&message(202, &[]), &weights, "non-MLProgram source model");
        let mut alternate = value.clone();
        let start = alternate
            .windows(b"weights.bin".len())
            .position(|bytes| bytes == b"weights.bin")
            .unwrap();
        alternate[start..start + 11].copy_from_slice(b"another.bin");
        assert_unsupported(
            &model(&[alternate]),
            &weights,
            "alternate source weight asset path",
        );
        let mut traversal = value;
        let start = traversal
            .windows(b"weights/weights.bin".len())
            .position(|bytes| bytes == b"weights/weights.bin")
            .unwrap();
        traversal[start..start + 3].copy_from_slice(b"../");
        assert_invalid(&model(&[traversal]), &weights);
    }

    #[test]
    fn unsupported_reference_does_not_hide_later_known_type_or_wire_corruption() {
        let (weights, offsets) = weight_file(&[(1, vec![1, 0]), (1, vec![2, 0])]);
        let future = tensor_value(22, &[1], offsets[0]);
        let wrong_known = tensor_value(11, &[1], offsets[1]);
        assert_invalid(&model(&[future.clone(), wrong_known]), &weights);
        let mut duplicate = tensor_value(10, &[1], offsets[1]);
        duplicate.extend(message(5, &message(1, b"@model_path/weights/weights.bin")));
        assert_invalid(&model(&[future, duplicate]), &weights);
        // An unsupported dtype within the same reference must still validate
        // its known rank/shape and offset fields before choosing fallback.
        let mut bad_shape = tensor_value(22, &[1], offsets[0]);
        let rank = bad_shape
            .windows(2)
            .position(|bytes| bytes == [16, 1])
            .unwrap();
        bad_shape[rank + 1] = 2;
        assert_invalid(&model(&[bad_shape]), &weights);
    }

    #[test]
    fn future_v2_entry_does_not_hide_later_header_or_reference_corruption() {
        let (mut weights, offsets) = weight_file(&[(1, vec![1, 0]), (1, vec![2, 0])]);
        let source = model(&[tensor_value(10, &[1], offsets[1])]);
        weights[68..72].copy_from_slice(&255u32.to_le_bytes());
        assert_unsupported(&source, &weights, "source weight metadata dtype");
        let mut malformed = weights.clone();
        malformed[offsets[1] as usize] ^= 1;
        assert_invalid(&source, &malformed);
        assert_invalid(&model(&[tensor_value(11, &[1], offsets[1])]), &weights);
        weights[80..88].copy_from_slice(&128u64.to_le_bytes());
        weights[offsets[1] as usize + 8..offsets[1] as usize + 16]
            .copy_from_slice(&u64::MAX.to_le_bytes());
        assert_invalid(&source, &weights);
    }

    #[test]
    fn future_dtype_still_requires_a_known_v2_entry_boundary() {
        let (weights, offsets) = weight_file(&[(1, vec![1, 0])]);
        assert_unsupported(
            &model(&[tensor_value(22, &[1], offsets[0])]),
            &weights,
            "source BlobFile tensor dtype",
        );
        // Misaligned, aligned-but-missing, and out-of-range offsets all have
        // known v2 semantics even when the referenced tensor dtype is future.
        for offset in [offsets[0] + 1, offsets[0] + 64, u64::MAX, 0] {
            assert_invalid(&model(&[tensor_value(22, &[1], offset)]), &weights);
        }
        let (empty, _) = weight_file(&[]);
        assert_invalid(&model(&[tensor_value(22, &[1], 64)]), &empty);

        // Another safe asset is not this sidecar: its offsets cannot be
        // validated against this file's entry inventory.
        let mut alternate = tensor_value(22, &[1], u64::MAX);
        let start = alternate
            .windows(b"weights.bin".len())
            .position(|bytes| bytes == b"weights.bin")
            .unwrap();
        alternate[start..start + 11].copy_from_slice(b"another.bin");
        assert_unsupported(
            &model(&[alternate]),
            &weights,
            "source BlobFile tensor dtype",
        );

        // A future file version does not establish the v2 boundary layout.
        let mut future_version = weights;
        future_version[4..8].copy_from_slice(&3u32.to_le_bytes());
        assert_unsupported(
            &model(&[tensor_value(22, &[1], u64::MAX)]),
            &future_version,
            "source BlobFile tensor dtype",
        );
    }
}
