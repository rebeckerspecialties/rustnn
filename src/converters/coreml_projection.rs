//! Source-derived block-local projection candidate. Only explicit constant
//! Cast/Sub/Reshape/Mul/Transpose closures are recognized; names and model
//! architecture do not participate. Qualification is deliberately opt-in.
use super::coreml_numerics::{Emit, V, named_arg, value_payload, variable};
use super::coreml_packed::{named_type, reference};
use crate::protos::coreml::{
    mil_spec as m,
    specification::{Model, model},
};
use std::collections::{BTreeMap, BTreeSet};

/// Keep constant Int8-to-Float32 conversions as packed CoreML leaves for CPU
/// qualification. This is distinct from the watchOS affine/column-loop profile.
/// Unit-scale decoding preserves every Int8 value exactly. GPU policies retain
/// casts until their compressed-weight compiler behavior is qualified.
pub fn compress_cpu_constants(model: &mut Model) -> usize {
    let Some(model::Type::MlProgram(program)) = model.r#type.as_mut() else {
        return 0;
    };
    let mut changed = 0;
    for function in program.functions.values_mut() {
        if function.block_specializations.len() != 1 {
            continue;
        }
        let Some(block) = function.block_specializations.get_mut(&function.opset) else {
            continue;
        };
        let bytes = block
            .operations
            .iter()
            .filter(|op| op.r#type == "const")
            .flat_map(|op| &op.outputs)
            .filter(|v| {
                variable(v).is_some_and(|v| v.dtype == m::DataType::Int8 && !v.shape.is_empty())
            })
            .map(|v| v.name.clone())
            .collect::<BTreeSet<_>>();
        let mut names = block
            .operations
            .iter()
            .flat_map(|op| &op.outputs)
            .chain(&function.inputs)
            .map(|v| v.name.clone())
            .collect::<BTreeSet<_>>();
        let before = changed;
        let mut result = Vec::new();
        for mut op in std::mem::take(&mut block.operations) {
            let admitted = op.r#type == "cast"
                && op.outputs.len() == 1
                && variable(&op.outputs[0]).is_some_and(|v| v.dtype == m::DataType::Float32)
                && named_arg(&op, "x").is_some_and(|x| bytes.contains(x));
            if admitted {
                let data = named_arg(&op, "x").unwrap().to_string();
                let rank = variable(&op.outputs[0]).unwrap().shape.len();
                let mut emit = Emit {
                    next: names.len(),
                    ops: vec![],
                    names: &mut names,
                };
                let scale = emit.constant(
                    m::DataType::Float32,
                    &vec![1; rank],
                    m::tensor_value::Value::Floats(m::tensor_value::RepeatedFloats {
                        values: vec![1.],
                    }),
                );
                result.extend(emit.ops);
                op.r#type = "constexpr_blockwise_shift_scale".into();
                op.inputs = [
                    ("data".into(), reference(&data)),
                    ("scale".into(), reference(&scale.name)),
                ]
                .into();
                changed += 1;
            }
            result.push(op);
        }
        block.operations = result;
        if changed != before {
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
    if changed > 0 {
        model.specification_version = model.specification_version.max(9);
    }
    changed
}

struct Packed {
    q: Vec<i8>,
    scales: Vec<f32>,
    k: u32,
    n: u32,
    width: u32,
    source_name: String,
    source_offset: Option<u64>,
    lookup_original: bool,
    /// Zero point subtracted from the stored bytes before scaling.
    zero: f32,
}
fn shape(op: &m::Operation) -> Option<Vec<u32>> {
    Some(variable(op.outputs.first()?)?.shape)
}
fn raw(value: &m::Value, weights: &[u8]) -> Option<Vec<u8>> {
    match value.value.as_ref()? {
        m::value::Value::BlobFileValue(v) => {
            if v.file_name != "@model_path/weights/weights.bin" {
                return None;
            }
            // Offsets address metadata, not the payload. Validate every span.
            let start = usize::try_from(v.offset).ok()?;
            let meta = weights.get(start..start.checked_add(64)?)?;
            if u32::from_le_bytes(meta[..4].try_into().ok()?) != 0xdeadbeef {
                return None;
            }
            let len = usize::try_from(u64::from_le_bytes(meta[8..16].try_into().ok()?)).ok()?;
            let pos = usize::try_from(u64::from_le_bytes(meta[16..24].try_into().ok()?)).ok()?;
            Some(weights.get(pos..pos.checked_add(len)?)?.to_vec())
        }
        _ => match value_payload(value)? {
            m::tensor_value::Value::Bytes(b) => Some(b.values.to_vec()),
            m::tensor_value::Value::Floats(b) => {
                Some(b.values.iter().flat_map(|v| v.to_le_bytes()).collect())
            }
            _ => None,
        },
    }
}
fn constant<'a>(name: &str, p: &BTreeMap<&str, &'a m::Operation>) -> Option<&'a m::Value> {
    let op = p.get(name)?;
    (op.r#type == "const").then(|| op.attributes.get("val"))?
}
fn floats(name: &str, p: &BTreeMap<&str, &m::Operation>, weights: &[u8]) -> Option<Vec<f32>> {
    let op = p.get(name)?;
    if op.r#type == "cast" {
        if !matches!(
            variable(op.outputs.first()?)?.dtype,
            m::DataType::Float32 | m::DataType::Float16
        ) {
            return None;
        }
        return floats(named_arg(op, "x")?, p, weights);
    }
    let dtype = variable(op.outputs.first()?)?.dtype;
    let raw = raw(constant(name, p)?, weights)?;
    match dtype {
        m::DataType::Float32 if raw.len() % 4 == 0 => Some(
            raw.chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect(),
        ),
        m::DataType::Float16 if raw.len() % 2 == 0 => Some(
            raw.chunks_exact(2)
                .map(|b| half::f16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32())
                .collect(),
        ),
        _ => None,
    }
}
fn parameter<'a>(
    op: &'a m::Operation,
    key: &str,
    producers: &BTreeMap<&str, &'a m::Operation>,
) -> Option<&'a m::tensor_value::Value> {
    let [binding] = op.inputs.get(key)?.arguments.as_slice() else {
        return None;
    };
    let value = match binding.binding.as_ref()? {
        m::argument::binding::Binding::Name(n) => constant(n, producers)?,
        m::argument::binding::Binding::Value(v) => v,
    };
    value_payload(value)
}
fn untransposed(op: &m::Operation, producers: &BTreeMap<&str, &m::Operation>) -> bool {
    ["transpose_x", "transpose_y"].iter().all(|key| {
        !op.inputs.contains_key(*key)
            || matches!(parameter(op,key,producers),
            Some(m::tensor_value::Value::Bools(v)) if v.values==[false])
    })
}
fn recognize(name: &str, p: &BTreeMap<&str, &m::Operation>, weights: &[u8]) -> Option<Packed> {
    recognize_in(name, p, weights, m::DataType::Float32)
}
/// Recognize the constant dequantization closure whose cast lands in `lane`.
fn recognize_in(
    name: &str,
    p: &BTreeMap<&str, &m::Operation>,
    weights: &[u8],
    lane: m::DataType,
) -> Option<Packed> {
    let r = p.get(name)?;
    if r.r#type != "reshape" {
        return None;
    }
    let dims = shape(r)?;
    let [n, k] = dims.as_slice() else { return None };
    if *n == 0 || *k == 0 || k % 32 != 0 {
        return None;
    }
    let mul = p.get(named_arg(r, "x")?)?;
    if mul.r#type != "mul" {
        return None;
    }
    for (a, b) in [("x", "y"), ("y", "x")] {
        let Some(result) = (|| {
            let data = p.get(named_arg(mul, a)?)?;
            let scale = p.get(named_arg(mul, b)?)?;
            if data.r#type != "reshape" || scale.r#type != "reshape" {
                return None;
            }
            let d = shape(data)?;
            let s = shape(scale)?;
            let [dn, g, w] = d.as_slice() else {
                return None;
            };
            if *dn != *n || g.checked_mul(*w)? != *k || *w % 32 != 0 || s != [*n, *g, 1] {
                return None;
            }
            let mut q = p.get(named_arg(data, "x")?)?;
            let mut zero = 0.;
            if q.r#type == "sub" {
                let z = floats(named_arg(q, "y")?, p, weights)?;
                if z.len() != 1 {
                    return None;
                }
                zero = z[0];
                q = p.get(named_arg(q, "x")?)?;
            }
            if q.r#type != "cast" || variable(q.outputs.first()?)?.dtype != lane {
                return None;
            }
            let qname = named_arg(q, "x")?;
            let original = p.get(qname)?;
            let dtype = variable(original.outputs.first()?)?.dtype;
            if shape(original)? != [*n, *k]
                || !matches!(dtype, m::DataType::Int8 | m::DataType::Uint8)
            {
                return None;
            }
            let bytes = raw(constant(qname, p)?, weights)?;
            if bytes.len() != (*n as usize).checked_mul(*k as usize)?
                || !zero.is_finite()
                || zero.trunc() != zero
            {
                return None;
            }
            let q = bytes
                .into_iter()
                .map(|v| {
                    let x = if dtype == m::DataType::Int8 {
                        v as i8 as f32
                    } else {
                        v as f32
                    };
                    let centered = x - zero;
                    (-128. ..=127.)
                        .contains(&centered)
                        .then_some(centered as i8)
                })
                .collect::<Option<Vec<_>>>()?;
            let scales = floats(named_arg(scale, "x")?, p, weights)?;
            if scales.len() != (*n as usize).checked_mul(*g as usize)?
                || scales.iter().any(|x| !x.is_finite())
            {
                return None;
            }
            Some(Packed {
                q,
                scales,
                k: *k,
                n: *n,
                width: *w,
                source_name: qname.into(),
                source_offset: match &constant(qname, p)?.value {
                    Some(m::value::Value::BlobFileValue(b))
                        if u64::from_le_bytes(
                            weights
                                .get(b.offset as usize + 16..b.offset as usize + 24)?
                                .try_into()
                                .ok()?,
                        ) == b.offset + 64 =>
                    {
                        Some(b.offset)
                    }
                    _ => None,
                },
                lookup_original: dtype == m::DataType::Int8 && zero == 0.,
                zero,
            })
        })() else {
            continue;
        };
        return Some(result);
    }
    None
}

// Resolve only an explicit Float32 constant matrix or its rank-two transpose.
// Preserve raw words: layout conversion must not round or flush source values.
fn constant_matrix(
    name: &str,
    p: &BTreeMap<&str, &m::Operation>,
    weights: &[u8],
) -> Option<(Vec<u8>, bool)> {
    let op = p.get(name)?;
    let (leaf, transposed) = if op.r#type == "transpose"
        && matches!(parameter(op, "perm", p), Some(m::tensor_value::Value::Ints(v)) if v.values == [1, 0])
    {
        (p.get(named_arg(op, "x")?)?, true)
    } else {
        (op, false)
    };
    let value = variable(leaf.outputs.first()?)?;
    if leaf.r#type != "const" || value.dtype != m::DataType::Float32 || value.shape.len() != 2 {
        return None;
    }
    let bytes = raw(leaf.attributes.get("val")?, weights)?;
    let expected = value
        .shape
        .iter()
        .try_fold(4usize, |a, &b| a.checked_mul(b as usize))?;
    (bytes.len() == expected).then_some((bytes, transposed))
}
impl Emit<'_> {
    fn tiled_constant_product(
        &mut self,
        x: &V,
        y: &V,
        out: &V,
        bytes: &[u8],
        transposed: bool,
        weights: &mut super::CoremlWeightStorage,
    ) -> Option<V> {
        if y.shape.len() != 2
            || x.shape.len() < 2
            || x.shape.len() > 4
            || x.shape[..x.shape.len() - 1].iter().any(|&d| d != 1)
        {
            return None;
        }
        let (k, n) = (y.shape[0], y.shape[1]);
        if k == 0 || k % 32 != 0 || n < 4096 || x.shape.last() != Some(&k) {
            return None;
        }
        let mut outputs = Vec::new();
        for start in (0..n).step_by(2048) {
            let count = 2048.min(n - start);
            let mut tile = Vec::with_capacity(k as usize * count as usize * 4);
            for row in 0..k {
                for column in start..start + count {
                    let index = if transposed {
                        column as usize * k as usize + row as usize
                    } else {
                        row as usize * n as usize + column as usize
                    } * 4;
                    tile.extend_from_slice(&bytes[index..index + 4]);
                }
            }
            let offset = append_blob(weights, 2, &tile);
            let w = self.op("const", vec![], m::DataType::Float32, &[k, count]);
            self.ops.last_mut().unwrap().attributes.insert(
                "val".into(),
                m::Value {
                    r#type: named_type("", m::DataType::Float32, &[k, count]).r#type,
                    value: Some(m::value::Value::BlobFileValue(m::value::BlobFileValue {
                        file_name: "@model_path/weights/weights.bin".into(),
                        offset,
                    })),
                    ..Default::default()
                },
            );
            let mut target = out.clone();
            *target.shape.last_mut()? = count;
            outputs.push(self.runtime_product(x, &w, &target)?);
        }
        Some(self.concat(&outputs, out.shape.len() - 1))
    }
    fn transpose(&mut self, x: &V, perm: &[usize]) -> V {
        let shape = perm.iter().map(|&i| x.shape[i]).collect::<Vec<_>>();
        let p = self.i(&perm.iter().map(|&i| i as i32).collect::<Vec<_>>(), false);
        self.op(
            "transpose",
            vec![("x", reference(&x.name)), ("perm", reference(&p.name))],
            x.dtype,
            &shape,
        )
    }
    fn pieces_axis(&mut self, x: &V, axis: usize, concat_axis: usize) -> V {
        let mut parts = Vec::new();
        let mut r = x.clone();
        for _ in 0..3 {
            let (h, l) = self.split(&r, &[axis], 131072.);
            parts.push(h);
            r = l;
        }
        parts.push(r);
        self.concat(&parts, concat_axis)
    }
    fn runtime_product(&mut self, x: &V, y: &V, out: &V) -> Option<V> {
        let rank = x.shape.len().max(y.shape.len());
        if !(2..=5).contains(&rank) || out.shape.len() != rank {
            return None;
        }
        let mut xs = vec![1; rank - x.shape.len()];
        xs.extend(&x.shape);
        let mut ys = vec![1; rank - y.shape.len()];
        ys.extend(&y.shape);
        let k = xs[rank - 1];
        let n = ys[rank - 1];
        let mut rows = xs[rank - 2];
        if k == 0 || !k.is_multiple_of(32) || ys[rank - 2] != k {
            return None;
        }
        let mut prefix = rank - 2;
        for i in 0..prefix {
            if (xs[i] != ys[i] && xs[i] != 1 && ys[i] != 1) || out.shape[i] != xs[i].max(ys[i]) {
                return None;
            }
        }
        // A trailing broadcast batch on the RHS is a contiguous set of left
        // rows. Fold it into M, retaining shared GQA keys/values without tiling.
        while prefix > 0 && ys[prefix - 1] == 1 {
            prefix -= 1;
            rows = rows.checked_mul(xs[prefix])?;
        }
        let product = |s: &[u32]| s.iter().try_fold(1u32, |a, &b| a.checked_mul(b));
        let xb = product(&xs[..prefix])?;
        let yb = product(&ys[..prefix])?;
        if xs[..prefix] != ys[..prefix] && xb != 1 && yb != 1 {
            return None;
        }
        let batch = xb.max(yb);
        let groups = k / 32;
        let a = self.reshape(x, &[xb, rows, groups, 32]);
        let a = self.transpose(&a, &[0, 2, 1, 3]);
        // Interleave activation slices within each row. This permits a rank-5
        // [batch, groups, rows, 16, columns] product, within CoreML's rank cap.
        let mut parts = Vec::new();
        let mut r = a;
        for _ in 0..3 {
            let (h, l) = self.split(&r, &[3], 131072.);
            parts.push(self.reshape(&h, &[xb, groups, rows, 1, 32]));
            r = l;
        }
        parts.push(self.reshape(&r, &[xb, groups, rows, 1, 32]));
        let a = self.concat(&parts, 3);
        let a = self.reshape(&a, &[xb, groups, rows.checked_mul(4)?, 32]);
        let b = self.reshape(y, &[yb, groups, 32, n]);
        let b = self.transpose(&b, &[0, 1, 3, 2]);
        let b = self.pieces_axis(&b, 3, 2);
        let b = self.transpose(&b, &[0, 1, 3, 2]);
        let flag = self.bool(false);
        let p = self.op(
            "matmul",
            vec![
                ("x", reference(&a.name)),
                ("y", reference(&b.name)),
                ("transpose_x", reference(&flag.name)),
                ("transpose_y", reference(&flag.name)),
            ],
            m::DataType::Float32,
            &[batch, groups, rows * 4, n.checked_mul(4)?],
        );
        let p = self.reshape(&p, &[batch, groups, rows, 16, n]);
        let p = self.transpose(&p, &[1, 3, 0, 2, 4]);
        let columns = batch.checked_mul(rows)?.checked_mul(n)?;
        let p = self.reshape(&p, &[groups * 16, 1, columns]);
        let result = self.aligned_sum(&p, &[0], groups * 16);
        Some(self.reshape(&result, &out.shape))
    }
    /// Replace a float16 dequantization closure by one `constexpr_blockwise_shift_scale`
    /// over the stored Int8/Uint8 matrix, so the exported bundle keeps the weights
    /// quantized and CoreML decompresses them itself. The group scales become a float16
    /// `[n, k/width]` blob and the zero point a constant offset in the data's dtype.
    fn blockwise_constexpr(
        &mut self,
        packed: &Packed,
        data_dtype: m::DataType,
        weights: &mut super::CoremlWeightStorage,
    ) -> V {
        let groups = packed.k / packed.width;
        let bytes = packed
            .scales
            .iter()
            .flat_map(|&s| half::f16::from_f32(s).to_bits().to_le_bytes())
            .collect::<Vec<_>>();
        let offset = append_blob(weights, 1, &bytes);
        let scale = self.op("const", vec![], m::DataType::Float16, &[packed.n, groups]);
        self.ops.last_mut().unwrap().attributes.insert(
            "val".into(),
            m::Value {
                r#type: named_type("", m::DataType::Float16, &[packed.n, groups]).r#type,
                value: Some(m::value::Value::BlobFileValue(m::value::BlobFileValue {
                    file_name: "@model_path/weights/weights.bin".into(),
                    offset,
                })),
                ..Default::default()
            },
        );
        let mut inputs = vec![
            ("data", reference(&packed.source_name)),
            ("scale", reference(&scale.name)),
        ];
        if packed.zero != 0. {
            let byte = if data_dtype == m::DataType::Int8 {
                packed.zero as i32 as i8 as u8
            } else {
                packed.zero as u8
            };
            let zero = self.constant(
                data_dtype,
                &[packed.n, groups],
                m::tensor_value::Value::Bytes(m::tensor_value::RepeatedBytes {
                    values: vec![byte; (packed.n * groups) as usize].into(),
                }),
            );
            inputs.push(("offset", reference(&zero.name)));
        }
        self.op(
            "constexpr_blockwise_shift_scale",
            inputs,
            m::DataType::Float16,
            &[packed.n, packed.k],
        )
    }
    fn packed_constant(
        &mut self,
        bytes: &[i8],
        shape: &[u32],
        weights: &mut super::CoremlWeightStorage,
    ) -> V {
        let offset = append_blob(
            weights,
            4,
            &bytes.iter().map(|&n| n as u8).collect::<Vec<_>>(),
        );
        let data = self.op("const", vec![], m::DataType::Int8, shape);
        self.ops.last_mut().unwrap().attributes.insert(
            "val".into(),
            m::Value {
                r#type: named_type("", m::DataType::Int8, shape).r#type,
                value: Some(m::value::Value::BlobFileValue(m::value::BlobFileValue {
                    file_name: "@model_path/weights/weights.bin".into(),
                    offset,
                })),
                ..Default::default()
            },
        );
        data
    }
    fn lookup(
        &mut self,
        indices: &V,
        packed: &Packed,
        weights: &mut super::CoremlWeightStorage,
        shape: &[u32],
    ) -> V {
        let table = if packed.lookup_original {
            V {
                name: packed.source_name.clone(),
                dtype: m::DataType::Int8,
                shape: vec![packed.n, packed.k],
            }
        } else {
            self.packed_constant(&packed.q, &[packed.n, packed.k], weights)
        };
        let token = self.reshape(indices, &[1]);
        let zero = self.i(&[0], false);
        let begin = self.concat(&[token.clone(), zero], 0);
        let size = self.i(&[1, packed.k as i32], false);
        let row = self.op(
            "slice_by_size",
            vec![
                ("x", reference(&table.name)),
                ("begin", reference(&begin.name)),
                ("size", reference(&size.name)),
            ],
            m::DataType::Int8,
            &[1, packed.k],
        );
        let dtype = self.constant(
            m::DataType::String,
            &[],
            m::tensor_value::Value::Strings(m::tensor_value::RepeatedStrings {
                values: vec!["fp32".into()],
            }),
        );
        let row = self.op(
            "cast",
            vec![
                ("x", reference(&row.name)),
                ("dtype", reference(&dtype.name)),
            ],
            m::DataType::Float32,
            &row.shape,
        );
        let groups = packed.k / packed.width;
        let row = self.reshape(&row, &[1, groups, packed.width]);
        let scales = self.constant(
            m::DataType::Float32,
            &[packed.n, groups, 1],
            m::tensor_value::Value::Floats(m::tensor_value::RepeatedFloats {
                values: packed.scales.clone(),
            }),
        );
        let zero = self.i(&[0, 0], false);
        let begin = self.concat(&[token, zero], 0);
        let size = self.i(&[1, groups as i32, 1], false);
        let scale = self.op(
            "slice_by_size",
            vec![
                ("x", reference(&scales.name)),
                ("begin", reference(&begin.name)),
                ("size", reference(&size.name)),
            ],
            m::DataType::Float32,
            &[1, groups, 1],
        );
        let product = self.b("mul", &row, &scale);
        self.reshape(&product, shape)
    }
    fn concat(&mut self, values: &[V], axis: usize) -> V {
        let a = self.i(&[axis as i32], true);
        let interleave = self.bool(false);
        let mut shape = values[0].shape.clone();
        shape[axis] = values.iter().map(|v| v.shape[axis]).sum();
        self.op(
            "concat",
            vec![
                (
                    "values",
                    m::Argument {
                        arguments: values
                            .iter()
                            .map(|v| m::argument::Binding {
                                binding: Some(m::argument::binding::Binding::Name(v.name.clone())),
                            })
                            .collect(),
                    },
                ),
                ("axis", reference(&a.name)),
                ("interleave", reference(&interleave.name)),
            ],
            values[0].dtype,
            &shape,
        )
    }
    fn pieces(&mut self, x: &V) -> V {
        let (a, r) = self.split(x, &[2], 16384.);
        let (b, c) = self.split(&r, &[2], 16384.);
        self.concat(&[a, b, c], 1)
    }
    fn weight(
        &mut self,
        bytes: &[i8],
        shape: &[u32],
        weights: &mut super::CoremlWeightStorage,
        watch: bool,
    ) -> V {
        let offset = append_blob(
            weights,
            4,
            &bytes.iter().map(|&n| n as u8).collect::<Vec<_>>(),
        );
        self.weight_at(offset, shape, watch)
    }
    fn weight_at(&mut self, offset: u64, shape: &[u32], watch: bool) -> V {
        let blob = m::Value {
            r#type: named_type("", m::DataType::Int8, shape).r#type,
            value: Some(m::value::Value::BlobFileValue(m::value::BlobFileValue {
                file_name: "@model_path/weights/weights.bin".into(),
                offset,
            })),
            ..Default::default()
        };
        if watch {
            // The iOS 16 affine constexpr schema uses typed attributes, not
            // named SSA inputs. Keep the blob on the operation so the bounded
            // column pass can recognize and move it inside its native loop.
            let value = |dtype, payload| {
                let argument = super::coreml_packed::immediate(dtype, &[], payload);
                match argument
                    .arguments
                    .into_iter()
                    .next()
                    .unwrap()
                    .binding
                    .unwrap()
                {
                    m::argument::binding::Binding::Value(v) => v,
                    _ => unreachable!(),
                }
            };
            let result = self.op(
                "constexpr_affine_dequantize",
                vec![],
                m::DataType::Float32,
                shape,
            );
            self.ops.last_mut().unwrap().attributes = [
                ("quantized_data".into(), blob),
                (
                    "scale".into(),
                    value(
                        m::DataType::Float32,
                        m::tensor_value::Value::Floats(m::tensor_value::RepeatedFloats {
                            values: vec![1.],
                        }),
                    ),
                ),
                (
                    "zero_point".into(),
                    value(
                        m::DataType::Int8,
                        m::tensor_value::Value::Bytes(m::tensor_value::RepeatedBytes {
                            values: vec![0].into(),
                        }),
                    ),
                ),
                (
                    "axis".into(),
                    value(
                        m::DataType::Int32,
                        m::tensor_value::Value::Ints(m::tensor_value::RepeatedInts {
                            values: vec![0],
                        }),
                    ),
                ),
            ]
            .into();
            return result;
        }
        let data = self.op("const", vec![], m::DataType::Int8, shape);
        self.ops
            .last_mut()
            .unwrap()
            .attributes
            .insert("val".into(), blob);
        let dtype = self.constant(
            m::DataType::String,
            &[],
            m::tensor_value::Value::Strings(m::tensor_value::RepeatedStrings {
                values: vec!["fp32".into()],
            }),
        );
        self.op(
            "cast",
            vec![
                ("x", reference(&data.name)),
                ("dtype", reference(&dtype.name)),
            ],
            m::DataType::Float32,
            shape,
        )
    }
    fn product(&mut self, x: &V, y: &V) -> V {
        let flag = self.bool(false);
        self.op(
            "matmul",
            vec![
                ("x", reference(&x.name)),
                ("y", reference(&y.name)),
                ("transpose_x", reference(&flag.name)),
                ("transpose_y", reference(&flag.name)),
            ],
            m::DataType::Float32,
            &[x.shape[0], x.shape[1], y.shape[2]],
        )
    }
    fn projection(
        &mut self,
        x: &V,
        packed: &Packed,
        weights: &mut super::CoremlWeightStorage,
        watch: bool,
        reuse: bool,
    ) -> Option<V> {
        let g = packed.k / 32;
        let shared_activation = if !watch || packed.n <= 4096 {
            let view = self.reshape(x, &[g, 1, 32]);
            Some(self.pieces(&view))
        } else {
            None
        };
        let mut tiles = Vec::new();
        for start in (0..packed.n).step_by(4096) {
            let n = 4096.min(packed.n - start);
            let mut q = Vec::new();
            let mut s = Vec::new();
            let mut residual = Vec::new();
            let mut units = Vec::new();
            let mut rounded = false;
            for group in 0..g {
                for j in 0..32 {
                    for col in start..start + n {
                        let index = (col * packed.k + group * 32 + j) as usize;
                        let value = packed.q[index];
                        let scale = packed.scales[(col * (packed.k / packed.width)
                            + group * 32 / packed.width)
                            as usize];
                        let exact = f64::from(value) * f64::from(scale);
                        let difference = f64::from(exact as f32) - exact;
                        let exp = ((scale.abs().to_bits() >> 23) & 255) as i32 - 127;
                        let unit = 2.0f64.powi(exp - 23);
                        let r = difference / unit;
                        if !r.is_finite() || r.trunc() != r || r.abs() > 127. {
                            return None;
                        }
                        rounded |= r != 0.;
                        q.push(value);
                        residual.push(r as i8);
                    }
                }
                for col in start..start + n {
                    let scale = packed.scales
                        [(col * (packed.k / packed.width) + group * 32 / packed.width) as usize];
                    s.push(scale);
                    let exp = ((scale.abs().to_bits() >> 23) & 255) as i32 - 127;
                    units.push(2.0f32.powi(exp - 23));
                }
            }
            if rounded && units.iter().any(|x| *x == 0. || !x.is_finite()) {
                return None;
            }
            let w = if reuse
                && packed.n <= 4096
                && let Some(offset) = packed.source_offset
            {
                // The source has one unobserved consumer chain, wholly replaced
                // by this product. Reuse its span after all admission checks.
                let at = offset as usize;
                weights[at + 4..at + 8].copy_from_slice(&4u32.to_le_bytes());
                for (to, &from) in weights[at + 64..at + 64 + q.len()].iter_mut().zip(&q) {
                    *to = from as u8;
                }
                self.weight_at(offset, &[g, 32, n], watch)
            } else {
                self.weight(&q, &[g, 32, n], weights, watch)
            };
            // The Watch profile keeps a wide projection's small activation
            // prelude local to each column tile. Native child cuts can forward
            // the original activation instead of expanded compensation values.
            // Other targets retain the faster shared-prelude profile.
            let a = shared_activation.clone().unwrap_or_else(|| {
                let view = self.reshape(x, &[g, 1, 32]);
                self.pieces(&view)
            });
            let p = self.product(&a, &w);
            let (h, l) = self.split(&p, &[1], 4096.);
            let scales = self.constant(
                m::DataType::Float32,
                &[g, 1, n],
                m::tensor_value::Value::Floats(m::tensor_value::RepeatedFloats { values: s }),
            );
            let h = self.b("mul", &h, &scales);
            let l = self.b("mul", &l, &scales);
            let h = self.reshape(&h, &[g * 3, 1, n]);
            let l = self.reshape(&l, &[g * 3, 1, n]);
            let mut terms = vec![h, l];
            if rounded {
                if units.iter().any(|x| *x == 0. || !x.is_finite()) {
                    return None;
                }
                let w = self.weight(&residual, &[g, 32, n], weights, watch);
                let p = self.product(&a, &w);
                let units = self.constant(
                    m::DataType::Float32,
                    &[g, 1, n],
                    m::tensor_value::Value::Floats(m::tensor_value::RepeatedFloats {
                        values: units,
                    }),
                );
                let correction = self.b("mul", &p, &units);
                terms.push(self.reshape(&correction, &[g * 3, 1, n]));
            }
            let terms = self.concat(&terms, 0);
            let sum = self.aligned_sum(&terms, &[0], terms.shape[0]);
            tiles.push(self.reshape(&sum, &[1, n]));
        }
        Some(if tiles.len() == 1 {
            tiles.remove(0)
        } else {
            self.concat(&tiles, 1)
        })
    }
}
fn append_blob(weights: &mut super::CoremlWeightStorage, dtype: u32, data: &[u8]) -> u64 {
    if weights.len() < 64 {
        weights.truncate(0);
        weights.resize(64, 0);
        weights[4..8].copy_from_slice(&2u32.to_le_bytes());
    }
    weights.resize((weights.len() + 63) & !63, 0);
    let offset = weights.len();
    let count = u32::from_le_bytes(weights[0..4].try_into().unwrap()) + 1;
    weights[0..4].copy_from_slice(&count.to_le_bytes());
    weights.extend_from_slice(&0xdeadbeefu32.to_le_bytes());
    weights.extend_from_slice(&dtype.to_le_bytes());
    weights.extend_from_slice(&(data.len() as u64).to_le_bytes());
    weights.extend_from_slice(&((offset + 64) as u64).to_le_bytes());
    weights.resize(offset + 64, 0);
    weights.extend_from_slice(data);
    weights.resize((weights.len() + 63) & !63, 0);
    offset as u64
}
/// Rewrite supported source-derived constant projection closures. This profile
/// is for qualification; it is not the default WebNN contraction lowering.
/// `watch` selects the watchOS-only compressed representation, allowing the
/// exact emitted target graph to be tested on a development host too.
pub fn apply(model: &mut Model, weights: &mut Vec<u8>, watch: bool) -> usize {
    let mut storage = super::CoremlWeightStorage::memory(std::mem::take(weights));
    let count = apply_storage(model, &mut storage, watch);
    *weights = storage.into_memory();
    count
}
/// Apply the same lowering with bounded compiler scratch stored in a local file.
pub fn apply_file_backed(
    model: &mut Model,
    weights: &mut super::CoremlWeightStorage,
    watch: bool,
) -> Result<usize, String> {
    let extra = weights
        .len()
        .checked_mul(3)
        .and_then(|n| n.checked_add(8 * 1024 * 1024))
        .ok_or("projection scratch extent overflow")?;
    weights.try_reserve(extra).map_err(|e| e.to_string())?;
    let count = apply_storage(model, weights, watch);
    weights.flush().map_err(|e| e.to_string())?;
    Ok(count)
}
fn apply_storage(
    model: &mut Model,
    weights: &mut super::CoremlWeightStorage,
    watch: bool,
) -> usize {
    let Some(model::Type::MlProgram(p)) = &mut model.r#type else {
        return 0;
    };
    let mut count = 0;
    for f in p.functions.values_mut() {
        for block in f.block_specializations.values_mut() {
            let old = std::mem::take(&mut block.operations);
            let producers = old
                .iter()
                .flat_map(|o| o.outputs.iter().map(move |v| (v.name.as_str(), o)))
                .collect::<BTreeMap<_, _>>();
            let mut users = BTreeMap::<&str, Vec<&m::Operation>>::new();
            for op in &old {
                for arg in op.inputs.values().flat_map(|a| &a.arguments) {
                    if let Some(m::argument::binding::Binding::Name(n)) = &arg.binding {
                        users.entry(n).or_default().push(op);
                    }
                }
            }
            let exclusive = |source: &str, target: &str| {
                let mut name = source;
                for _ in 0..16 {
                    if block.outputs.iter().any(|n| n == name) {
                        return false;
                    }
                    let Some(uses) = users.get(name) else {
                        return false;
                    };
                    let [next] = uses.as_slice() else {
                        return false;
                    };
                    if next.outputs.iter().any(|v| v.name == target) {
                        return true;
                    }
                    if !matches!(
                        next.r#type.as_str(),
                        "cast" | "sub" | "reshape" | "mul" | "transpose"
                    ) || next.outputs.len() != 1
                    {
                        return false;
                    }
                    name = &next.outputs[0].name;
                }
                false
            };
            let mut vars = f
                .inputs
                .iter()
                .filter_map(variable)
                .map(|v| (v.name.clone(), v))
                .collect::<BTreeMap<_, _>>();
            let mut names = f
                .inputs
                .iter()
                .map(|v| v.name.clone())
                .chain(producers.keys().map(|&n| n.into()))
                .collect::<BTreeSet<_>>();
            for op in &old {
                for v in &op.outputs {
                    if let Some(v) = variable(v) {
                        vars.insert(v.name.clone(), v);
                    }
                }
                let candidate = (|| {
                    if op.r#type != "matmul"
                        || op.outputs.len() != 1
                        || !untransposed(op, &producers)
                    {
                        return None;
                    }
                    let x = vars.get(named_arg(op, "x")?)?;
                    let y = producers.get(named_arg(op, "y")?)?;
                    if y.r#type != "transpose"
                        || !matches!(parameter(y,"perm",&producers),Some(m::tensor_value::Value::Ints(v)) if v.values==[1,0])
                        || x.dtype != m::DataType::Float32
                        || x.shape.len() < 2
                        || x.shape[..x.shape.len() - 1].iter().any(|&d| d != 1)
                    {
                        return None;
                    }
                    let packed = recognize(named_arg(y, "x")?, &producers, weights)?;
                    if x.shape.last() != Some(&packed.k) || shape(y)? != [packed.k, packed.n] {
                        return None;
                    }
                    Some((x.clone(), packed))
                })();
                if let Some((x, packed)) = candidate {
                    let mut e = Emit {
                        next: names.len(),
                        ops: vec![],
                        names: &mut names,
                    };
                    if let Some(result) = e.projection(
                        &x,
                        &packed,
                        weights,
                        watch,
                        exclusive(&packed.source_name, &op.outputs[0].name),
                    ) {
                        let out = variable(&op.outputs[0]).unwrap();
                        let result = e.reshape(&result, &out.shape);
                        e.ops.push(m::Operation {
                            r#type: "identity".into(),
                            inputs: [("x".into(), reference(&result.name))].into(),
                            outputs: op.outputs.clone(),
                            ..Default::default()
                        });
                        block.operations.extend(e.ops);
                        count += 1;
                        drop(packed);
                        continue;
                    }
                }
                // A float16 activation against a float16 dequantization closure: keep the
                // stored bytes and let CoreML's blockwise constant decompress them.
                let half_candidate = (|| {
                    if op.r#type != "matmul"
                        || op.outputs.len() != 1
                        || !untransposed(op, &producers)
                    {
                        return None;
                    }
                    let x = vars.get(named_arg(op, "x")?)?;
                    let y = producers.get(named_arg(op, "y")?)?;
                    if y.r#type != "transpose"
                        || !matches!(parameter(y,"perm",&producers),Some(m::tensor_value::Value::Ints(v)) if v.values==[1,0])
                        || x.dtype != m::DataType::Float16
                    {
                        return None;
                    }
                    let packed = recognize_in(
                        named_arg(y, "x")?,
                        &producers,
                        weights,
                        m::DataType::Float16,
                    )?;
                    if x.shape.last() != Some(&packed.k) || shape(y)? != [packed.k, packed.n] {
                        return None;
                    }
                    let data_dtype = variable(
                        producers
                            .get(packed.source_name.as_str())?
                            .outputs
                            .first()?,
                    )?
                    .dtype;
                    Some((x.clone(), packed, data_dtype))
                })();
                if let Some((x, packed, data_dtype)) = half_candidate {
                    let mut e = Emit {
                        next: names.len(),
                        ops: vec![],
                        names: &mut names,
                    };
                    let w = e.blockwise_constexpr(&packed, data_dtype, weights);
                    let out = variable(&op.outputs[0]).unwrap();
                    let no = e.bool(false);
                    let yes = e.bool(true);
                    let product = e.op(
                        "matmul",
                        vec![
                            ("x", reference(&x.name)),
                            ("y", reference(&w.name)),
                            ("transpose_x", reference(&no.name)),
                            ("transpose_y", reference(&yes.name)),
                        ],
                        m::DataType::Float16,
                        &out.shape,
                    );
                    e.ops.push(m::Operation {
                        r#type: "identity".into(),
                        inputs: [("x".into(), reference(&product.name))].into(),
                        outputs: op.outputs.clone(),
                        ..Default::default()
                    });
                    block.operations.extend(e.ops);
                    count += 1;
                    continue;
                }
                let runtime = (|| {
                    if op.r#type != "matmul"
                        || op.outputs.len() != 1
                        || !untransposed(op, &producers)
                    {
                        return None;
                    }
                    let x = vars.get(named_arg(op, "x")?)?;
                    let y = vars.get(named_arg(op, "y")?)?;
                    let out = variable(&op.outputs[0])?;
                    if x.dtype != m::DataType::Float32 || y.dtype != m::DataType::Float32 {
                        return None;
                    }
                    let mut e = Emit {
                        next: names.len(),
                        ops: vec![],
                        names: &mut names,
                    };
                    let result = if y.shape.len() == 2 && y.shape[1] >= 4096 {
                        if let Some((bytes, transposed)) =
                            constant_matrix(&y.name, &producers, weights)
                        {
                            e.tiled_constant_product(x, y, &out, &bytes, transposed, weights)?
                        } else {
                            e.runtime_product(x, y, &out)?
                        }
                    } else {
                        e.runtime_product(x, y, &out)?
                    };
                    e.ops.push(m::Operation {
                        r#type: "identity".into(),
                        inputs: [("x".into(), reference(&result.name))].into(),
                        outputs: op.outputs.clone(),
                        ..Default::default()
                    });
                    Some(e.ops)
                })();
                if let Some(ops) = runtime {
                    block.operations.extend(ops);
                    count += 1;
                    continue;
                }
                let lookup = (|| {
                    if op.r#type != "gather" || op.outputs.len() != 1 {
                        return None;
                    }
                    if !matches!(parameter(op,"axis",&producers),Some(m::tensor_value::Value::Ints(v)) if v.values==[0])
                    {
                        return None;
                    }
                    let table = named_arg(op, "x")?;
                    let indices = vars.get(named_arg(op, "indices")?)?;
                    if indices.dtype != m::DataType::Int32
                        || indices.shape.iter().product::<u32>() != 1
                    {
                        return None;
                    }
                    let packed = recognize(table, &producers, weights)?;
                    let out = variable(&op.outputs[0])?;
                    if out.shape.iter().product::<u32>() != packed.k {
                        return None;
                    }
                    let mut e = Emit {
                        next: names.len(),
                        ops: vec![],
                        names: &mut names,
                    };
                    let result = e.lookup(indices, &packed, weights, &out.shape);
                    e.ops.push(m::Operation {
                        r#type: "identity".into(),
                        inputs: [("x".into(), reference(&result.name))].into(),
                        outputs: op.outputs.clone(),
                        ..Default::default()
                    });
                    Some(e.ops)
                })();
                if let Some(ops) = lookup {
                    block.operations.extend(ops);
                    count += 1;
                    continue;
                }
                block.operations.push(op.clone());
            }
            // Remove only pure, unreachable SSA operations. Every original graph
            // output is a root, including observed dequantization intermediates.
            let mut needed = block.outputs.iter().cloned().collect::<BTreeSet<_>>();
            let mut kept = Vec::new();
            for op in block.operations.drain(..).rev() {
                if !op.blocks.is_empty() || op.outputs.iter().any(|o| needed.contains(&o.name)) {
                    for arg in op.inputs.values().flat_map(|a| &a.arguments) {
                        if let Some(m::argument::binding::Binding::Name(n)) = &arg.binding {
                            needed.insert(n.clone());
                        }
                    }
                    kept.push(op);
                }
            }
            kept.reverse();
            block.operations = kept;
        }
    }
    if count > 0 {
        for f in p.functions.values_mut() {
            if f.opset == "CoreML7" && f.block_specializations.len() == 1 {
                let block = f.block_specializations.remove("CoreML7").unwrap();
                f.opset = "CoreML8".into();
                f.block_specializations.insert("CoreML8".into(), block);
            }
        }
        model.specification_version = model.specification_version.max(9);
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::converters::{CoremlMlProgramConverter, GraphConverter};
    use crate::mlcontext::{MLGraphBuilder, MLNamedOperands, MLOperandDescriptor};
    use crate::operator_enums::MLOperandDataType;
    use prost::Message;

    #[cfg(all(
        target_os = "macos",
        feature = "coreml-runtime",
        feature = "coreml-export"
    ))]
    #[test]
    fn watch_affine_blob_attributes_load_before_and_after_bounded_expansion() {
        use crate::backend_selection::DeviceType;
        use crate::executors::coreml::{CoremlByteInput, compile_model, run_coreml_bytes};
        use std::collections::HashMap;
        let mut builder = MLGraphBuilder::new_uncompiled();
        let input = builder
            .input(
                "x",
                &MLOperandDescriptor::new(MLOperandDataType::Float32, vec![2, 1, 32]),
            )
            .unwrap();
        let placeholder = builder
            .constant_from_bytes(
                &MLOperandDescriptor::new(MLOperandDataType::Float32, vec![2, 32, 8]),
                vec![0; 512 * 4],
            )
            .unwrap();
        let output = builder.matmul(input, placeholder).unwrap();
        let graph = builder
            .finish_graph_info(&MLNamedOperands::from([("y", output)]))
            .unwrap();
        let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
        let mut model = Model::decode(converted.data.as_slice()).unwrap();
        let mut names = ["x".into(), "y".into()].into();
        let mut e = Emit {
            next: 0,
            ops: vec![],
            names: &mut names,
        };
        let mut storage = super::super::WeightFileBuilder::new().finalize_storage();
        let q = (0..512).map(|i| (i % 256) as u8 as i8).collect::<Vec<_>>();
        let weight = e.weight(&q, &[2, 32, 8], &mut storage, true);
        let affine = &e.ops[0];
        assert_eq!(affine.r#type, "constexpr_affine_dequantize");
        assert!(affine.inputs.is_empty());
        assert_eq!(affine.attributes.len(), 4);
        assert_eq!(
            raw(&affine.attributes["quantized_data"], &storage).unwrap(),
            q.iter().map(|&v| v as u8).collect::<Vec<_>>()
        );
        let x = V {
            name: "x".into(),
            dtype: m::DataType::Float32,
            shape: vec![2, 1, 32],
        };
        let product = e.product(&x, &weight);
        e.ops.push(m::Operation {
            r#type: "identity".into(),
            inputs: [("x".into(), reference(&product.name))].into(),
            outputs: vec![named_type("y", m::DataType::Float32, &[2, 1, 8])],
            ..Default::default()
        });
        let Some(model::Type::MlProgram(program)) = &mut model.r#type else {
            panic!()
        };
        let function = program.functions.get_mut("main").unwrap();
        let block = function
            .block_specializations
            .get_mut(&function.opset)
            .unwrap();
        block.operations = e.ops;
        block.outputs = vec!["y".into()];
        let weights = storage.into_memory();
        let input = (0..64)
            .map(|i| (i as f32 - 31.) * 0.125)
            .collect::<Vec<_>>();
        let bytes = input
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        let expected = (0..2)
            .flat_map(|g| (0..8).map(move |n| (g, n)))
            .map(|(g, n)| {
                (0..32)
                    .map(|k| input[g * 32 + k] * f32::from(q[(g * 32 + k) * 8 + n]))
                    .sum::<f32>()
            })
            .collect::<Vec<_>>();
        for bounded in [false, true] {
            let mut candidate = model.clone();
            if bounded {
                assert_eq!(
                    super::super::coreml_packed::bound_packed_matmuls(&mut candidate, 512).len(),
                    1
                );
            }
            let native = compile_model(
                candidate.encode_to_vec(),
                Some(weights.clone()),
                DeviceType::Cpu,
                false,
            )
            .unwrap();
            let outputs = run_coreml_bytes(
                &native,
                &HashMap::from([(
                    "x".into(),
                    CoremlByteInput {
                        data: &bytes,
                        descriptor: &graph.operands[graph.input_operands[0] as usize].descriptor,
                    },
                )]),
                &HashMap::from([(
                    "y".into(),
                    graph.operands[graph.output_operands[0] as usize]
                        .descriptor
                        .clone(),
                )]),
            )
            .unwrap();
            let actual = outputs["y"]
                .chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                .collect::<Vec<_>>();
            assert_eq!(actual, expected, "bounded={bounded}");
        }
    }

    #[cfg(all(
        target_os = "macos",
        feature = "coreml-runtime",
        feature = "coreml-export"
    ))]
    #[test]
    fn tiled_watch_and_shared_cpu_projections_keep_partial_column_and_signed_values() {
        use crate::backend_selection::DeviceType;
        use crate::executors::coreml::{CoremlByteInput, compile_model, run_coreml_bytes};
        use std::collections::HashMap;
        let n = 4097usize;
        let mut builder = MLGraphBuilder::new_uncompiled();
        let input = builder
            .input(
                "x",
                &MLOperandDescriptor::new(MLOperandDataType::Float32, vec![1, 32]),
            )
            .unwrap();
        let weight = builder
            .constant_from_bytes(
                &MLOperandDescriptor::new(MLOperandDataType::Float32, vec![32, n as u64]),
                vec![0; 32 * n * 4],
            )
            .unwrap();
        let output = builder.matmul(input, weight).unwrap();
        let graph = builder
            .finish_graph_info(&MLNamedOperands::from([("y", output)]))
            .unwrap();
        let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
        let original = Model::decode(converted.data.as_slice()).unwrap();
        let q = (0..n * 32)
            .map(|i| ((i * 17 % 255) as i32 - 127) as i8)
            .collect::<Vec<_>>();
        let values = (0..32)
            .map(|i| (i as f32 - 15.0) * 0.125)
            .collect::<Vec<_>>();
        let expected = (0..n)
            .map(|col| {
                (0..32)
                    .map(|k| f64::from(values[k]) * f64::from(q[col * 32 + k]) * 0.125)
                    .sum::<f64>() as f32
            })
            .collect::<Vec<_>>();
        let bytes = values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        for watch in [false, true] {
            let mut names = ["x".into(), "y".into()].into();
            let mut e = Emit {
                next: 0,
                ops: vec![],
                names: &mut names,
            };
            let mut storage = super::super::WeightFileBuilder::new().finalize_storage();
            let packed = Packed {
                q: q.clone(),
                scales: vec![0.125; n],
                k: 32,
                n: n as u32,
                width: 32,
                source_name: "weights".into(),
                source_offset: None,
                lookup_original: false,
                zero: 0.,
            };
            let x = V {
                name: "x".into(),
                dtype: m::DataType::Float32,
                shape: vec![1, 32],
            };
            let result = e
                .projection(&x, &packed, &mut storage, watch, false)
                .unwrap();
            e.ops.push(m::Operation {
                r#type: "identity".into(),
                inputs: [("x".into(), reference(&result.name))].into(),
                outputs: vec![named_type("y", m::DataType::Float32, &[1, n as u32])],
                ..Default::default()
            });
            let mut model = original.clone();
            let Some(model::Type::MlProgram(p)) = &mut model.r#type else {
                panic!()
            };
            let f = p.functions.get_mut("main").unwrap();
            let mut b = f.block_specializations.remove(&f.opset).unwrap();
            b.operations = e.ops;
            b.outputs = vec!["y".into()];
            f.opset = "CoreML8".into();
            f.block_specializations.insert(f.opset.clone(), b);
            model.specification_version = 9;
            if watch {
                assert_eq!(
                    super::super::coreml_packed::bound_packed_matmuls(&mut model, 2 * 1024 * 1024)
                        .len(),
                    2
                );
            }
            let native = compile_model(
                model.encode_to_vec(),
                Some(storage.into_memory()),
                DeviceType::Cpu,
                false,
            )
            .unwrap();
            let actual = run_coreml_bytes(
                &native,
                &HashMap::from([(
                    "x".into(),
                    CoremlByteInput {
                        data: &bytes,
                        descriptor: &graph.operands[graph.input_operands[0] as usize].descriptor,
                    },
                )]),
                &HashMap::from([(
                    "y".into(),
                    graph.operands[graph.output_operands[0] as usize]
                        .descriptor
                        .clone(),
                )]),
            )
            .unwrap();
            let actual = actual["y"]
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>();
            assert_eq!(actual, expected, "watch={watch}");
        }
    }

    #[test]
    fn cpu_compression_preserves_byte_payload_and_skips_runtime_casts() {
        let mut b = MLGraphBuilder::new_uncompiled();
        let desc = MLOperandDescriptor::new(MLOperandDataType::Int8, vec![256]);
        let bytes = (0..256).map(|v| v as u8).collect::<Vec<_>>();
        let c = b.constant_from_bytes(&desc, bytes.clone()).unwrap();
        let x = b.input("x", &desc).unwrap();
        let a = b.cast(c, MLOperandDataType::Float32).unwrap();
        let z = b.cast(x, MLOperandDataType::Float32).unwrap();
        let g = b
            .finish_graph_info(&MLNamedOperands::from([("a", a), ("z", z)]))
            .unwrap();
        let converted = CoremlMlProgramConverter.convert(&g).unwrap();
        let weights = converted.weights_data.unwrap();
        let mut model = Model::decode(converted.data.as_slice()).unwrap();
        assert_eq!(compress_cpu_constants(&mut model), 1);
        assert_eq!(compress_cpu_constants(&mut model), 0);
        let Some(model::Type::MlProgram(p)) = &model.r#type else {
            panic!()
        };
        let f = &p.functions["main"];
        assert_eq!(f.opset, "CoreML8");
        let block = &f.block_specializations[&f.opset];
        let op = block
            .operations
            .iter()
            .find(|o| o.r#type == "constexpr_blockwise_shift_scale")
            .unwrap();
        let data = named_arg(op, "data").unwrap();
        let leaf = block
            .operations
            .iter()
            .find(|o| o.outputs[0].name == data)
            .unwrap();
        assert_eq!(raw(&leaf.attributes["val"], &weights).unwrap(), bytes);
        assert!(
            block
                .operations
                .iter()
                .any(|o| o.r#type == "cast" && named_arg(o, "x") == Some("x"))
        );
    }

    #[test]
    fn float16_chain_becomes_a_blockwise_constexpr_consumed_by_a_transposed_matmul() {
        use crate::operator_options::MLDimension;
        let (n, k) = (64u32, 64u32);
        let mut b = MLGraphBuilder::new_uncompiled();
        let stored = (0..n * k).map(|i| (i * 7 % 251) as u8).collect::<Vec<_>>();
        let data = b
            .constant_from_bytes(
                &MLOperandDescriptor::new(MLOperandDataType::Uint8, vec![n as u64, k as u64]),
                stored.clone(),
            )
            .unwrap();
        let zero = b
            .constant_from_bytes(
                &MLOperandDescriptor::new(MLOperandDataType::Float16, vec![]),
                half::f16::from_f32(128.).to_bits().to_le_bytes().to_vec(),
            )
            .unwrap();
        let scale_values = (0..n * 2)
            .map(|i| 0.5 + i as f32 / 256.)
            .collect::<Vec<_>>();
        let scales = b
            .constant_from_bytes(
                &MLOperandDescriptor::new(MLOperandDataType::Float16, vec![n as u64, 2]),
                scale_values
                    .iter()
                    .flat_map(|&v| half::f16::from_f32(v).to_bits().to_le_bytes())
                    .collect(),
            )
            .unwrap();
        let dims = |v: &[u32]| {
            v.iter()
                .map(|&d| MLDimension::Static(d))
                .collect::<Vec<_>>()
        };
        let widened = b.cast(data, MLOperandDataType::Float16).unwrap();
        let centered = b.sub(widened, zero).unwrap();
        let grouped = b.reshape(centered, dims(&[n, 2, 32])).unwrap();
        let scale_cast = b.cast(scales, MLOperandDataType::Float16).unwrap();
        let scale_grouped = b.reshape(scale_cast, dims(&[n, 2, 1])).unwrap();
        let scaled = b.mul(grouped, scale_grouped).unwrap();
        let weight = b.reshape(scaled, dims(&[n, k])).unwrap();
        let transposed = b.transpose(weight).unwrap();
        let x = b
            .input(
                "x",
                &MLOperandDescriptor::new(MLOperandDataType::Float16, vec![1, 1, k as u64]),
            )
            .unwrap();
        let y = b.matmul(x, transposed).unwrap();
        let g = b
            .finish_graph_info(&MLNamedOperands::from([("y", y)]))
            .unwrap();
        let converted = CoremlMlProgramConverter.convert(&g).unwrap();
        let mut weights = converted.weights_data.unwrap();
        let mut model = Model::decode(converted.data.as_slice()).unwrap();
        assert_eq!(apply(&mut model, &mut weights, false), 1);
        let Some(model::Type::MlProgram(p)) = &model.r#type else {
            panic!()
        };
        let f = &p.functions["main"];
        assert_eq!(f.opset, "CoreML8");
        let block = &f.block_specializations[&f.opset];
        let producers = block
            .operations
            .iter()
            .flat_map(|o| o.outputs.iter().map(move |v| (v.name.as_str(), o)))
            .collect::<BTreeMap<_, _>>();
        let leaf = block
            .operations
            .iter()
            .find(|o| o.r#type == "constexpr_blockwise_shift_scale")
            .expect("one blockwise constant");
        let out = variable(&leaf.outputs[0]).unwrap();
        assert_eq!(
            (out.dtype, out.shape.clone()),
            (m::DataType::Float16, vec![n, k])
        );
        let data_leaf = producers[named_arg(leaf, "data").unwrap()];
        assert_eq!(
            variable(&data_leaf.outputs[0]).unwrap().dtype,
            m::DataType::Uint8
        );
        assert_eq!(raw(&data_leaf.attributes["val"], &weights).unwrap(), stored);
        let scale_leaf = producers[named_arg(leaf, "scale").unwrap()];
        let scale_var = variable(&scale_leaf.outputs[0]).unwrap();
        assert_eq!(
            (scale_var.dtype, scale_var.shape.clone()),
            (m::DataType::Float16, vec![n, 2])
        );
        assert_eq!(
            floats(&scale_leaf.outputs[0].name, &producers, &weights).unwrap(),
            scale_values
                .iter()
                .map(|&v| half::f16::from_f32(v).to_f32())
                .collect::<Vec<_>>()
        );
        let offset_leaf = producers[named_arg(leaf, "offset").unwrap()];
        assert_eq!(
            variable(&offset_leaf.outputs[0]).unwrap().dtype,
            m::DataType::Uint8
        );
        assert_eq!(
            raw(&offset_leaf.attributes["val"], &weights).unwrap(),
            vec![128u8; (n * 2) as usize]
        );
        let product = block
            .operations
            .iter()
            .find(|o| o.r#type == "matmul")
            .expect("one float16 matmul");
        assert_eq!(named_arg(product, "y"), Some(leaf.outputs[0].name.as_str()));
        assert!(
            matches!(parameter(product, "transpose_y", &producers), Some(m::tensor_value::Value::Bools(v)) if v.values == [true])
        );
        assert_eq!(
            variable(&product.outputs[0]).unwrap().dtype,
            m::DataType::Float16
        );
        assert!(
            !block.operations.iter().any(|o| o.r#type == "cast"
                && variable(&o.outputs[0])
                    .is_some_and(|v| v.dtype == m::DataType::Float16 && v.shape == [n, k])),
            "the widening cast of the stored bytes is gone"
        );
    }

    #[test]
    fn column_tiles_preserve_every_raw_word_and_partial_final_tile() {
        for transposed in [false, true] {
            let (k, n) = (32usize, 4097usize);
            let words: [u32; 7] = [
                0, 0x80000000, 1, 0x007fffff, 0x7fc01234, 0xff800000, 0x3f800001,
            ];
            let bytes = (0..k * n)
                .flat_map(|i| words[i % words.len()].to_le_bytes())
                .collect::<Vec<_>>();
            let x = V {
                name: "x".into(),
                dtype: m::DataType::Float32,
                shape: vec![1, k as u32],
            };
            let y = V {
                name: "y".into(),
                dtype: m::DataType::Float32,
                shape: vec![k as u32, n as u32],
            };
            let out = V {
                name: "out".into(),
                dtype: m::DataType::Float32,
                shape: vec![1, n as u32],
            };
            let mut names = ["x".into(), "y".into(), "out".into()].into();
            let mut e = Emit {
                next: 0,
                ops: vec![],
                names: &mut names,
            };
            let mut storage = super::super::WeightFileBuilder::new().finalize_storage();
            let result = e
                .tiled_constant_product(&x, &y, &out, &bytes, transposed, &mut storage)
                .unwrap();
            assert_eq!(result.shape, out.shape);
            let tiles = e
                .ops
                .iter()
                .filter(|o| {
                    o.r#type == "const"
                        && matches!(
                            o.attributes.get("val").and_then(|v| v.value.as_ref()),
                            Some(m::value::Value::BlobFileValue(_))
                        )
                })
                .collect::<Vec<_>>();
            assert_eq!(tiles.len(), 3);
            let mut column = 0;
            for tile in tiles {
                let count = shape(tile).unwrap()[1] as usize;
                let payload = raw(&tile.attributes["val"], &storage).unwrap();
                assert_eq!(payload.len(), k * count * 4);
                for row in 0..k {
                    for c in 0..count {
                        let source = if transposed {
                            (column + c) * k + row
                        } else {
                            row * n + column + c
                        };
                        assert_eq!(
                            &payload[(row * count + c) * 4..(row * count + c + 1) * 4],
                            &bytes[source * 4..source * 4 + 4]
                        );
                    }
                }
                column += count;
            }
            assert_eq!(column, n);
            assert!(e.ops.iter().all(|o| o.blocks.is_empty()));
        }
    }

    #[test]
    fn streamed_source_preserves_archive_and_manifest_payloads() {
        let words =
            (0..1025).map(|i| [0u32, 0x80000000, 1, 0x007fffff, 0x7fc01234, 0x3f800001][i % 6]);
        let bytes = words.flat_map(u32::to_le_bytes).collect::<Vec<_>>();
        let graph = serde_json::json!({"format":"webnn-graph-json","version":2,"inputs":{"x":{"dataType":"float32","shape":[1025]}},"consts":{"w":{"dataType":"float32","shape":[1025],"init":{"kind":"weights","ref":"w"}}},"nodes":[{"id":"sum","op":"add","inputs":["x","w"],"outputs":["y"],"options":{}}],"outputs":{"y":"y"}});
        for archive in [false, true] {
            let root = tempfile::tempdir().unwrap();
            let path = root.path().join("graph.json");
            std::fs::write(&path, serde_json::to_vec(&graph).unwrap()).unwrap();
            let payload = if archive {
                let view = safetensors::tensor::TensorView::new(
                    safetensors::Dtype::F32,
                    vec![1025],
                    &bytes,
                )
                .unwrap();
                let encoded = safetensors::serialize([("w", view)], None).unwrap();
                let path = root.path().join("graph.safetensors");
                std::fs::write(&path, &encoded).unwrap();
                (path, encoded)
            } else {
                std::fs::write(root.path().join("graph.manifest.json"),serde_json::to_vec(&serde_json::json!({"tensors":{"w":{"byteOffset":64,"byteLength":bytes.len()}}})).unwrap()).unwrap();
                let mut encoded = vec![0; 64];
                encoded.extend_from_slice(&bytes);
                let path = root.path().join("graph.weights");
                std::fs::write(&path, &encoded).unwrap();
                (path, encoded)
            };
            let original = CoremlMlProgramConverter
                .convert(&crate::load_graph_from_path(&path).unwrap())
                .unwrap();
            let (model, weights) = CoremlMlProgramConverter
                .convert_source_file_backed(&path)
                .unwrap();
            assert_eq!(
                Model::decode(model.as_slice()).unwrap(),
                Model::decode(original.data.as_slice()).unwrap()
            );
            assert_eq!(weights.as_ref(), original.weights_data.unwrap());
            assert_eq!(std::fs::read(&payload.0).unwrap(), payload.1);
            // A malformed later read fails before a native model can be loaded.
            let file = std::fs::OpenOptions::new()
                .write(true)
                .open(&payload.0)
                .unwrap();
            file.set_len(64).unwrap();
            assert!(
                CoremlMlProgramConverter
                    .convert_source_file_backed(&path)
                    .is_err()
            );
        }
    }

    #[test]
    fn consuming_and_file_backed_conversion_preserve_model_and_blobs() {
        let mut b = MLGraphBuilder::new_uncompiled();
        let desc = MLOperandDescriptor::new(MLOperandDataType::Float32, vec![8]);
        let bits = [
            0u32, 0x80000000, 1, 0x007fffff, 0x7fc01234, 0x7f800000, 0xff800000, 0x3f800001,
        ];
        let bytes = bits
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        let c = b.constant_from_bytes(&desc, bytes.clone()).unwrap();
        let x = b.input("x", &desc).unwrap();
        let y = b.add(x, c).unwrap();
        let g = b
            .finish_graph_info(&MLNamedOperands::from([("y", y)]))
            .unwrap();
        let borrowed = CoremlMlProgramConverter.convert(&g).unwrap();
        let owned = CoremlMlProgramConverter.convert_owned(g.clone()).unwrap();
        let (data, storage) = CoremlMlProgramConverter
            .convert_owned_file_backed(g)
            .unwrap();
        assert_eq!(
            Model::decode(borrowed.data.as_slice()).unwrap(),
            Model::decode(owned.data.as_slice()).unwrap()
        );
        assert_eq!(
            Model::decode(borrowed.data.as_slice()).unwrap(),
            Model::decode(data.as_slice()).unwrap()
        );
        assert_eq!(borrowed.weights_data, owned.weights_data);
        assert_eq!(borrowed.weights_data.unwrap().as_slice(), storage.as_ref());
    }
}
