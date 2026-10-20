//! Opt-in native Float32 corrections. These are MIL operations, not host kernels.
//! Rewrites retain an ordinary native result for values outside their admission
//! domain. Target qualification remains necessary before enabling this profile.
use super::coreml_packed::{immediate, named_type, reference, tensor_type};
use crate::protos::coreml::{
    mil_spec as m,
    specification::{Model, model},
};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone)]
pub(super) struct V {
    pub(super) name: String,
    pub(super) dtype: m::DataType,
    pub(super) shape: Vec<u32>,
}
pub(super) struct Emit<'a> {
    pub(super) ops: Vec<m::Operation>,
    pub(super) names: &'a mut BTreeSet<String>,
    pub(super) next: usize,
}
impl Emit<'_> {
    pub(super) fn op(
        &mut self,
        kind: &str,
        inputs: Vec<(&str, m::Argument)>,
        dtype: m::DataType,
        shape: &[u32],
    ) -> V {
        let name = loop {
            let n = format!("__rustnn_precise_{}", self.next);
            self.next += 1;
            if self.names.insert(n.clone()) {
                break n;
            }
        };
        self.ops.push(m::Operation {
            r#type: kind.into(),
            inputs: inputs.into_iter().map(|(k, v)| (k.into(), v)).collect(),
            outputs: vec![named_type(&name, dtype, shape)],
            ..Default::default()
        });
        V {
            name,
            dtype,
            shape: shape.to_vec(),
        }
    }
    pub(super) fn constant(
        &mut self,
        dtype: m::DataType,
        shape: &[u32],
        payload: m::tensor_value::Value,
    ) -> V {
        let arg = immediate(dtype, shape, payload);
        let m::argument::binding::Binding::Value(value) = arg.arguments[0].binding.clone().unwrap()
        else {
            unreachable!()
        };
        let v = self.op("const", vec![], dtype, shape);
        self.ops
            .last_mut()
            .unwrap()
            .attributes
            .insert("val".into(), value);
        v
    }
    pub(super) fn f(&mut self, value: f32) -> V {
        self.constant(
            m::DataType::Float32,
            &[],
            m::tensor_value::Value::Floats(m::tensor_value::RepeatedFloats {
                values: vec![value],
            }),
        )
    }
    pub(super) fn i(&mut self, values: &[i32], scalar: bool) -> V {
        self.constant(
            m::DataType::Int32,
            &if scalar {
                vec![]
            } else {
                vec![values.len() as u32]
            },
            m::tensor_value::Value::Ints(m::tensor_value::RepeatedInts {
                values: values.to_vec(),
            }),
        )
    }
    pub(super) fn bool(&mut self, value: bool) -> V {
        self.constant(
            m::DataType::Bool,
            &[],
            m::tensor_value::Value::Bools(m::tensor_value::RepeatedBools {
                values: vec![value],
            }),
        )
    }
    pub(super) fn u(&mut self, kind: &str, x: &V) -> V {
        self.op(kind, vec![("x", reference(&x.name))], x.dtype, &x.shape)
    }
    pub(super) fn b(&mut self, kind: &str, x: &V, y: &V) -> V {
        let len = x.shape.len().max(y.shape.len());
        let mut shape = vec![1; len];
        for v in [x, y] {
            for (a, b) in shape[len - v.shape.len()..].iter_mut().zip(&v.shape) {
                *a = (*a).max(*b);
            }
        }
        let dtype = if matches!(
            kind,
            "less"
                | "less_equal"
                | "greater"
                | "greater_equal"
                | "equal"
                | "logical_and"
                | "logical_or"
        ) {
            m::DataType::Bool
        } else {
            x.dtype
        };
        self.op(
            kind,
            vec![("x", reference(&x.name)), ("y", reference(&y.name))],
            dtype,
            &shape,
        )
    }
    pub(super) fn select(&mut self, c: &V, a: &V, b: &V) -> V {
        self.op(
            "select",
            vec![
                ("cond", reference(&c.name)),
                ("a", reference(&a.name)),
                ("b", reference(&b.name)),
            ],
            a.dtype,
            &a.shape,
        )
    }
    pub(super) fn reduce(&mut self, kind: &str, x: &V, axes: &[usize]) -> V {
        let av = self.i(&axes.iter().map(|&n| n as i32).collect::<Vec<_>>(), false);
        let keep = self.bool(true);
        let mut shape = x.shape.clone();
        for &a in axes {
            shape[a] = 1;
        }
        self.op(
            kind,
            vec![
                ("x", reference(&x.name)),
                ("axes", reference(&av.name)),
                ("keep_dims", reference(&keep.name)),
            ],
            x.dtype,
            &shape,
        )
    }
    pub(super) fn reshape(&mut self, x: &V, shape: &[u32]) -> V {
        let s = self.i(&shape.iter().map(|&n| n as i32).collect::<Vec<_>>(), false);
        self.op(
            "reshape",
            vec![("x", reference(&x.name)), ("shape", reference(&s.name))],
            x.dtype,
            shape,
        )
    }
    pub(super) fn all(&mut self, x: &V, axes: &[usize], count: u32) -> V {
        let dt = self.constant(
            m::DataType::String,
            &[],
            m::tensor_value::Value::Strings(m::tensor_value::RepeatedStrings {
                values: vec!["int32".into()],
            }),
        );
        let cast = self.op(
            "cast",
            vec![("x", reference(&x.name)), ("dtype", reference(&dt.name))],
            m::DataType::Int32,
            &x.shape,
        );
        let sum = self.reduce("reduce_sum", &cast, axes);
        let n = self.i(&[count as i32], true);
        self.b("equal", &sum, &n)
    }
    pub(super) fn finite(&mut self, x: &V) -> V {
        let a = self.u("abs", x);
        let max = self.f(f32::MAX);
        self.b("less_equal", &a, &max)
    }
    pub(super) fn extract(&mut self, x: &V, sigma: &V) -> (V, V) {
        let sum = self.b("add", x, sigma);
        // sigma >= abs(x) on the admitted finite domain. This relu preserves
        // the rounded nonnegative value and prevents algebraic cancellation.
        let rounded = self.u("relu", &sum);
        let h = self.b("sub", &rounded, sigma);
        let l = self.b("sub", x, &h);
        (h, l)
    }
    pub(super) fn split(&mut self, x: &V, axes: &[usize], factor: f32) -> (V, V) {
        let a = self.u("abs", x);
        let peak = if axes.is_empty() {
            a
        } else {
            self.reduce("reduce_max", &a, axes)
        };
        let c = self.f(factor);
        let sigma = self.b("mul", &peak, &c);
        self.extract(x, &sigma)
    }
    pub(super) fn aligned_sum(&mut self, x: &V, axes: &[usize], count: u32) -> V {
        let factor = 2.0f32.powi((32 - (count - 1).leading_zeros()) as i32 + 1);
        let (h, r) = self.split(x, axes, factor);
        let (m, l) = self.split(&r, axes, factor);
        let h = self.reduce("reduce_sum", &h, axes);
        let m = self.reduce("reduce_sum", &m, axes);
        let l = self.reduce("reduce_sum", &l, axes);
        let sum = self.b("add", &h, &m);
        self.b("add", &sum, &l)
    }
    fn softmax(&mut self, x: &V, ordinary: &V, axis: usize) -> V {
        let max = self.reduce("reduce_max", x, &[axis]);
        let shifted = self.b("sub", x, &max);
        let lo = self.f(-80.);
        let zero_tail = self.f(-104.);
        let normal = self.b("greater_equal", &shifted, &lo);
        let zero = self.b("less_equal", &shifted, &zero_tail);
        let domain = self.b("logical_or", &normal, &zero);
        let finite = self.finite(x);
        let domain = self.b("logical_and", &domain, &finite);
        let admitted = self.all(&domain, &[axis], x.shape[axis]);
        let exp = self.u("exp", &shifted);
        let sum = self.reduce("reduce_sum", &exp, &[axis]);
        let result = self.b("real_div", &exp, &sum);
        self.select(&admitted, &result, ordinary)
    }
    fn sigmoid(&mut self, x: &V, ordinary: &V) -> V {
        let low = self.f(-80.);
        let half = self.f(0.5);
        let below = self.b("minimum", x, &low);
        let scaled = self.b("mul", &below, &half);
        let exp = self.u("exp", &scaled);
        let tail = self.b("mul", &exp, &exp);
        let lower = self.b("less", x, &low);
        let result = self.select(&lower, &tail, ordinary);
        let valid = self.b("equal", x, x);
        self.select(&valid, &result, x)
    }
    fn softplus(&mut self, x: &V, ordinary: &V) -> V {
        // log1p(t) = 2 atanh(t/(2+t)), t = exp(-abs(x)). Here q <= 1/3;
        // truncating after q^15/15 contributes less than 1.03e-9 absolute
        // error. The negative normal tail never adds its tiny value to one.
        let a = self.u("abs", x);
        let minus = self.f(-1.);
        let negative = self.b("mul", &a, &minus);
        let t = self.u("exp", &negative);
        let two = self.f(2.);
        let denominator = self.b("add", &two, &t);
        let q = self.b("real_div", &t, &denominator);
        let q2 = self.b("mul", &q, &q);
        let mut p = self.f(1. / 15.);
        for d in [13, 11, 9, 7, 5, 3, 1] {
            let product = self.b("mul", &p, &q2);
            let coefficient = self.f(1. / d as f32);
            p = self.b("add", &product, &coefficient);
        }
        let twice = self.b("mul", &two, &q);
        let log1p = self.b("mul", &twice, &p);
        let zero = self.f(0.);
        let positive = self.b("maximum", x, &zero);
        let result = self.b("add", &positive, &log1p);
        // Preserve the ordinary native operation outside the qualified normal
        // domain. This pass does not claim to repair its subnormal tail.
        let low = self.f(-80.);
        let admitted = self.b("greater_equal", x, &low);
        let finite = self.finite(x);
        let admitted = self.b("logical_and", &admitted, &finite);
        self.select(&admitted, &result, ordinary)
    }
    fn division(&mut self, x: &V, y: &V, q: &V) -> V {
        let (qh, ql) = self.split(
            q,
            &q.shape.len().checked_sub(1).into_iter().collect::<Vec<_>>(),
            4096.,
        );
        let (yh, yl) = self.split(
            y,
            &y.shape.len().checked_sub(1).into_iter().collect::<Vec<_>>(),
            4096.,
        );
        let hh = self.b("mul", &qh, &yh);
        let lh = self.b("mul", &ql, &yh);
        let hl = self.b("mul", &qh, &yl);
        let ll = self.b("mul", &ql, &yl);
        let r = self.b("sub", x, &hh);
        let r = self.b("sub", &r, &lh);
        let r = self.b("sub", &r, &hl);
        let r = self.b("sub", &r, &ll);
        let d = self.b("real_div", &r, y);
        let corrected = self.b("add", q, &d);
        let low = self.f(2.0f32.powi(-40));
        let high = self.f(2.0f32.powi(40));
        let xabs = self.u("abs", x);
        let yabs = self.u("abs", y);
        let xl = self.b("greater_equal", &xabs, &low);
        let xh = self.b("less_equal", &xabs, &high);
        let yl = self.b("greater_equal", &yabs, &low);
        let yh = self.b("less_equal", &yabs, &high);
        let validx = self.b("logical_and", &xl, &xh);
        let validy = self.b("logical_and", &yl, &yh);
        let valid = self.b("logical_and", &validx, &validy);
        // The split shares a row maximum. Keep the element-wise admission,
        // while rejecting rows whose maxima can overflow or contain NaNs.
        // Zero neighbours are safe and must not disable a finite cell's repair.
        let qabs = self.u("abs", q);
        let qlimit = self.f(2.0f32.powi(80));
        let safeq = self.b("less_equal", &qabs, &qlimit);
        let safey = self.b("less_equal", &yabs, &high);
        let row = self.b("logical_and", &safeq, &safey);
        let axes = row
            .shape
            .len()
            .checked_sub(1)
            .into_iter()
            .collect::<Vec<_>>();
        let row = if let Some(&axis) = axes.first() {
            self.all(&row, &axes, row.shape[axis])
        } else {
            row
        };
        let valid = self.b("logical_and", &valid, &row);
        self.select(&valid, &corrected, q)
    }

    fn sqrt(&mut self, x: &V, ordinary: &V) -> V {
        let lower = self.f(2.0f32.powi(-64));
        let upper = self.f(2.0f32.powi(64));
        let low = self.b("less", x, &lower);
        let high = self.b("greater", x, &upper);
        let up = self.f(2.0f32.powi(48));
        let down = self.f(2.0f32.powi(-48));
        let xu = self.b("mul", x, &up);
        let xd = self.b("mul", x, &down);
        let work = self.select(&high, &xd, x);
        let work = self.select(&low, &xu, &work);
        let y = self.u("sqrt", &work);
        let c = self.f(4096.);
        let sigma = self.b("mul", &y, &c);
        let (h, l) = self.extract(&y, &sigma);
        let hh = self.b("mul", &h, &h);
        let hl = self.b("mul", &h, &l);
        let two = self.f(2.);
        let hl = self.b("mul", &hl, &two);
        let ll = self.b("mul", &l, &l);
        let r = self.b("sub", &work, &hh);
        let r = self.b("sub", &r, &hl);
        let r = self.b("sub", &r, &ll);
        let denominator = self.b("mul", &y, &two);
        let correction = self.b("real_div", &r, &denominator);
        let root = self.b("add", &y, &correction);
        let up = self.f(2.0f32.powi(24));
        let down = self.f(2.0f32.powi(-24));
        let xu = self.b("mul", &root, &up);
        let xd = self.b("mul", &root, &down);
        let result = self.select(&high, &xu, &root);
        let result = self.select(&low, &xd, &result);
        let zero = self.f(0.);
        let positive = self.b("greater", x, &zero);
        let finite = self.finite(x);
        let valid = self.b("logical_and", &positive, &finite);
        self.select(&valid, &result, ordinary)
    }
    fn mean(&mut self, x: &V, ordinary: &V, axes: &[usize], count: u32) -> V {
        let sum = self.aligned_sum(x, axes, count);
        let n = self.f(count as f32);
        let q = self.b("real_div", &sum, &n);
        let (h, l) = self.split(&q, axes, 4096.);
        let hn = self.b("mul", &h, &n);
        let ln = self.b("mul", &l, &n);
        let r = self.b("sub", &sum, &hn);
        let r = self.b("sub", &r, &ln);
        let delta = self.b("real_div", &r, &n);
        let repaired = self.b("add", &q, &delta);
        // Exclude overflow, subnormal input products and poorly scaled ranges.
        // The ordinary operator handles exceptional values without sanitizing.
        let absolute = self.u("abs", x);
        let maximum = self.reduce("reduce_max", &absolute, axes);
        let upper = self.f(2.0f32.powi(80));
        let safe = self.b("less_equal", &maximum, &upper);
        let lower = self.f(2.0f32.powi(-60));
        let zero = self.f(0.);
        let normal = self.b("greater_equal", &absolute, &lower);
        let zero = self.b("equal", &absolute, &zero);
        let normal = self.b("logical_or", &normal, &zero);
        let normal = self.all(&normal, axes, count);
        let finite = self.finite(x);
        let finite = self.all(&finite, axes, count);
        let safe = self.b("logical_and", &safe, &finite);
        let safe = self.b("logical_and", &safe, &normal);
        let result = self.reshape(&repaired, &ordinary.shape);
        let safe = self.reshape(&safe, &ordinary.shape);
        self.select(&safe, &result, ordinary)
    }
}
pub(super) fn named_arg<'a>(op: &'a m::Operation, key: &str) -> Option<&'a str> {
    let [a] = op.inputs.get(key)?.arguments.as_slice() else {
        return None;
    };
    if let Some(m::argument::binding::Binding::Name(n)) = &a.binding {
        Some(n)
    } else {
        None
    }
}
pub(super) fn value_payload(v: &m::Value) -> Option<&m::tensor_value::Value> {
    let m::value::Value::ImmediateValue(v) = v.value.as_ref()? else {
        return None;
    };
    let m::value::immediate_value::Value::Tensor(t) = v.value.as_ref()? else {
        return None;
    };
    t.value.as_ref()
}
pub(super) fn argument_payload<'a>(
    op: &'a m::Operation,
    key: &str,
    constants: &'a BTreeMap<String, m::Value>,
) -> Option<&'a m::tensor_value::Value> {
    let [a] = op.inputs.get(key)?.arguments.as_slice() else {
        return None;
    };
    match a.binding.as_ref()? {
        m::argument::binding::Binding::Name(n) => value_payload(constants.get(n)?),
        m::argument::binding::Binding::Value(v) => value_payload(v),
    }
}
pub(super) fn variable(v: &m::NamedValueType) -> Option<V> {
    let (dtype, shape) = tensor_type(v.r#type.as_ref()?)?;
    Some(V {
        name: v.name.clone(),
        dtype: m::DataType::try_from(dtype).ok()?,
        shape,
    })
}

/// Apply explicit Float32 softmax, sigmoid, square-root and mean corrections.
/// Returns counts by original MIL operator. This opt-in profile does not change
/// the WPT comparator or assert correctness for an entire model.
pub fn apply(model: &mut Model) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    fn visit(model: &mut Model, counts: &mut BTreeMap<String, usize>) {
        match &mut model.r#type {
            Some(model::Type::Pipeline(p)) => {
                for child in &mut p.models {
                    visit(child, counts)
                }
            }
            Some(model::Type::MlProgram(p)) => {
                for f in p.functions.values_mut() {
                    for b in f.block_specializations.values_mut() {
                        rewrite(b, &f.inputs, counts)
                    }
                }
            }
            _ => {}
        }
    }
    visit(model, &mut counts);
    counts
}
fn rewrite(
    block: &mut m::Block,
    inputs: &[m::NamedValueType],
    counts: &mut BTreeMap<String, usize>,
) {
    let mut vars: BTreeMap<_, _> = inputs
        .iter()
        .chain(&block.inputs)
        .filter_map(variable)
        .map(|v| (v.name.clone(), v))
        .collect();
    let mut names: BTreeSet<_> = vars
        .keys()
        .cloned()
        .chain(
            block
                .operations
                .iter()
                .flat_map(|o| o.outputs.iter().map(|v| v.name.clone())),
        )
        .collect();
    let mut constants = BTreeMap::new();
    let original = std::mem::take(&mut block.operations);
    for mut op in original {
        for b in &mut op.blocks {
            let outer = vars
                .values()
                .map(|v| named_type(&v.name, v.dtype, &v.shape))
                .collect::<Vec<_>>();
            rewrite(b, &outer, counts);
        }
        for v in &op.outputs {
            if let Some(v) = variable(v) {
                vars.insert(v.name.clone(), v);
            }
        }
        if op.r#type == "const"
            && let (Some(v), Some(out)) = (op.attributes.get("val"), op.outputs.first())
        {
            constants.insert(out.name.clone(), v.clone());
        }
        let eligible = matches!(
            op.r#type.as_str(),
            "softmax" | "sigmoid" | "softplus" | "sqrt" | "reduce_mean" | "real_div"
        );
        let x = named_arg(&op, "x").and_then(|n| vars.get(n)).cloned();
        let out = op.outputs.first().and_then(variable);
        if !eligible
            || op.outputs.len() != 1
            || x.as_ref().is_none_or(|v| {
                v.dtype != m::DataType::Float32 || v.shape.is_empty() || v.shape.contains(&0)
            })
            || out.as_ref().is_none_or(|v| v.dtype != m::DataType::Float32)
        {
            block.operations.push(op);
            continue;
        }
        let x = x.unwrap();
        let out = out.unwrap();
        let y = if op.r#type == "real_div" {
            let y = named_arg(&op, "y").and_then(|n| vars.get(n)).cloned();
            if y.as_ref()
                .is_none_or(|y| y.dtype != m::DataType::Float32 || y.shape.is_empty())
            {
                block.operations.push(op);
                continue;
            }
            y
        } else {
            None
        };
        let axes = if op.r#type == "softmax" {
            match argument_payload(&op, "axis", &constants) {
                Some(m::tensor_value::Value::Ints(i)) if i.values.len() == 1 => vec![i.values[0]],
                _ => {
                    block.operations.push(op);
                    continue;
                }
            }
        } else if op.r#type == "reduce_mean" {
            match argument_payload(&op, "axes", &constants) {
                Some(m::tensor_value::Value::Ints(i)) => i.values.clone(),
                _ => {
                    block.operations.push(op);
                    continue;
                }
            }
        } else {
            vec![]
        };
        let axes = axes
            .into_iter()
            .map(|i| if i < 0 { x.shape.len() as i32 + i } else { i })
            .map(|i| usize::try_from(i).ok().filter(|&i| i < x.shape.len()))
            .collect::<Option<Vec<_>>>();
        let Some(axes) = axes else {
            block.operations.push(op);
            continue;
        };
        let count = axes
            .iter()
            .try_fold(1u32, |n, &a| n.checked_mul(x.shape[a]));
        let Some(count) = count.filter(|&n| n > 0 && n <= 65536) else {
            block.operations.push(op);
            continue;
        };
        if op.r#type == "reduce_mean"
            && (axes.is_empty() || axes.iter().collect::<BTreeSet<_>>().len() != axes.len())
        {
            block.operations.push(op);
            continue;
        }
        let mut e = Emit {
            next: names.len(),
            ops: vec![],
            names: &mut names,
        };
        let ordinary = e.op(
            &op.r#type,
            op.inputs
                .iter()
                .map(|(k, v)| (k.as_str(), v.clone()))
                .collect(),
            out.dtype,
            &out.shape,
        );
        let result = match op.r#type.as_str() {
            "real_div" => e.division(&x, y.as_ref().unwrap(), &ordinary),
            "softmax" if axes[0] + 1 == x.shape.len() => {
                let rows = x.shape[..axes[0]].iter().product::<u32>();
                let flat = e.reshape(&x, &[rows, count]);
                let fallback = e.reshape(&ordinary, &[rows, count]);
                let value = e.softmax(&flat, &fallback, 1);
                e.reshape(&value, &out.shape)
            }
            "softmax" => e.softmax(&x, &ordinary, axes[0]),
            "sigmoid" => e.sigmoid(&x, &ordinary),
            "softplus" => e.softplus(&x, &ordinary),
            "sqrt" if x.shape.iter().product::<u32>() == 1 => {
                let flat = e.reshape(&x, &[1, 1]);
                let fallback = e.reshape(&ordinary, &[1, 1]);
                let value = e.sqrt(&flat, &fallback);
                e.reshape(&value, &out.shape)
            }
            "sqrt" => e.sqrt(&x, &ordinary),
            "reduce_mean" if axes == [x.shape.len() - 1] => {
                let rows = x.shape[..x.shape.len() - 1].iter().product::<u32>();
                let flat = e.reshape(&x, &[rows, count]);
                let fallback = e.reshape(&ordinary, &[rows, 1]);
                let value = e.mean(&flat, &fallback, &[1], count);
                e.reshape(&value, &out.shape)
            }
            "reduce_mean" => e.mean(&x, &ordinary, &axes, count),
            _ => unreachable!(),
        };
        e.ops.push(m::Operation {
            r#type: "identity".into(),
            inputs: [("x".into(), reference(&result.name))].into(),
            outputs: op.outputs,
            ..Default::default()
        });
        block.operations.extend(e.ops);
        *counts.entry(op.r#type).or_default() += 1;
    }
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "coreml-runtime",
    feature = "coreml-export"
))]
mod native_tests {
    use super::*;
    use crate::backend_selection::DeviceType;
    use crate::converters::{CoremlMlProgramConverter, GraphConverter};
    use crate::executors::coreml::{CoremlByteInput, compile_model, run_coreml_bytes};
    use crate::mlcontext::{MLGraphBuilder, MLNamedOperands, MLOperandDescriptor};
    use crate::operator_enums::MLOperandDataType;
    use prost::Message;
    use std::collections::HashMap;
    fn run(kind: &str, shape: Vec<u32>, data: &[f32], repair: bool) -> Vec<f32> {
        let mut b = MLGraphBuilder::new_uncompiled();
        let x = b
            .input(
                "x",
                &MLOperandDescriptor::new(
                    MLOperandDataType::Float32,
                    shape.into_iter().map(u64::from).collect(),
                ),
            )
            .unwrap();
        let y = match kind {
            "sqrt" => b.sqrt(x).unwrap(),
            "sigmoid" => b.sigmoid(x).unwrap(),
            "softplus" => b.softplus(x).unwrap(),
            "softmax" => b.softmax(x, 1).unwrap(),
            "mean" => b
                .reduce_mean_with_options(
                    x,
                    crate::operator_options::MLReduceOptions {
                        axes: Some(vec![1]),
                        keep_dimensions: true,
                        ..Default::default()
                    },
                )
                .unwrap(),
            _ => unreachable!(),
        };
        let g = b
            .finish_graph_info(&MLNamedOperands::from([("y", y)]))
            .unwrap();
        let c = CoremlMlProgramConverter.convert(&g).unwrap();
        let mut m = Model::decode(c.data.as_slice()).unwrap();
        if repair {
            let changes = apply(&mut m);
            assert_eq!(changes.values().sum::<usize>(), 1);
        }
        let native =
            compile_model(m.encode_to_vec(), c.weights_data, DeviceType::Cpu, false).unwrap();
        let bytes = data
            .iter()
            .flat_map(|n| n.to_le_bytes())
            .collect::<Vec<_>>();
        let out = run_coreml_bytes(
            &native,
            &HashMap::from([(
                "x".into(),
                CoremlByteInput {
                    data: &bytes,
                    descriptor: &g.operands[g.input_operands[0] as usize].descriptor,
                },
            )]),
            &HashMap::from([(
                "y".into(),
                g.operands[g.output_operands[0] as usize].descriptor.clone(),
            )]),
        )
        .unwrap();
        out["y"]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }
    #[test]
    fn aligned_mean_keeps_small_signed_cancellation_results() {
        // Every product and sum in these dyadic packets has an exact binary64
        // reference. A zero or sign loss cannot hide behind a relative budget.
        for width in [32usize, 128, 512] {
            let mut values = Vec::new();
            let mut expected = Vec::new();
            for (positive, negative) in [
                (f32::from_bits(1.0f32.to_bits() + 1), -1.0),
                (-f32::from_bits(1.0f32.to_bits() + 1), 1.0),
                (f32::from_bits(1048576.0f32.to_bits() + 1), -1048576.0),
            ] {
                values.extend((0..width).map(|i| if i % 2 == 0 { positive } else { negative }));
                expected.push(((f64::from(positive) + f64::from(negative)) / 2.) as f32);
            }
            let actual = run("mean", vec![3, width as u32], &values, true);
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.into_iter().zip(expected) {
                assert_eq!(actual.to_bits(), expected.to_bits(), "width {width}");
            }
        }
    }
    #[test]
    fn native_softplus_preserves_negative_normal_timesteps() {
        let values = [
            -80.0f32,
            -30.,
            -22.42,
            -15.850192,
            -10.,
            -1.,
            0.,
            1.,
            20.,
            80.,
            f32::MAX,
        ];
        let actual = run("softplus", vec![values.len() as u32], &values, true);
        for (&x, &y) in values.iter().zip(&actual) {
            let x = f64::from(x);
            let expected = (x.max(0.) + (-x.abs()).exp().ln_1p()) as f32;
            assert!(
                y.to_bits().abs_diff(expected.to_bits()) <= 4,
                "{x}: {y} vs {expected}"
            );
        }
    }
    #[test]
    fn native_division_scalar_and_broadcast_keep_two_ulp_and_classes() {
        for (scalar, finite) in [(false, false), (true, false), (false, true)] {
            let mut xs = [
                1.0f32,
                -1.,
                16777218.,
                -2.0f32.powi(-40),
                2.0f32.powi(40),
                0.,
                -0.,
                f32::from_bits(1),
                f32::MAX,
                f32::INFINITY,
                f32::NAN,
                7.,
            ];
            let ys: Vec<f32> = if scalar {
                vec![3.]
            } else {
                vec![3., -7., 13., 3., 7., 3., 3., 1., 1., 2., 2., 0.]
            };
            if finite {
                for x in &mut xs {
                    if !x.is_finite() || x.abs() > 2.0f32.powi(40) {
                        *x = 0.25;
                    }
                }
            }
            let ys = ys
                .into_iter()
                .map(|y| if finite && y == 0. { 3. } else { y })
                .collect::<Vec<_>>();
            let mut b = MLGraphBuilder::new_uncompiled();
            let x = b
                .input(
                    "x",
                    &MLOperandDescriptor::new(MLOperandDataType::Float32, vec![1, xs.len() as u64]),
                )
                .unwrap();
            let y = if scalar {
                b.constant_from_slice(
                    &MLOperandDescriptor::new(MLOperandDataType::Float32, vec![]),
                    &ys,
                )
                .unwrap()
            } else {
                b.input(
                    "d",
                    &MLOperandDescriptor::new(MLOperandDataType::Float32, vec![1, ys.len() as u64]),
                )
                .unwrap()
            };
            let z = b.div(x, y).unwrap();
            let g = b
                .finish_graph_info(&MLNamedOperands::from([("y", z)]))
                .unwrap();
            let c = CoremlMlProgramConverter.convert(&g).unwrap();
            let mut model = Model::decode(c.data.as_slice()).unwrap();
            assert_eq!(
                apply(&mut model).get("real_div").copied().unwrap_or(0),
                usize::from(!scalar)
            );
            let native = compile_model(
                model.encode_to_vec(),
                c.weights_data,
                DeviceType::Cpu,
                false,
            )
            .unwrap();
            let xb = xs.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
            let yb = ys.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
            let mut inputs = HashMap::from([(
                "x".into(),
                CoremlByteInput {
                    data: &xb,
                    descriptor: &g.operands[g.input_operands[0] as usize].descriptor,
                },
            )]);
            if !scalar {
                inputs.insert(
                    "d".into(),
                    CoremlByteInput {
                        data: &yb,
                        descriptor: &g.operands[g.input_operands[1] as usize].descriptor,
                    },
                );
            }
            let actual = run_coreml_bytes(
                &native,
                &inputs,
                &HashMap::from([(
                    "y".into(),
                    g.operands[g.output_operands[0] as usize].descriptor.clone(),
                )]),
            )
            .unwrap();
            for (i, bytes) in actual["y"].chunks_exact(4).enumerate() {
                let a = f32::from_le_bytes(bytes.try_into().unwrap());
                let e = (xs[i] as f64 / ys[if scalar { 0 } else { i }] as f64) as f32;
                assert!(
                    if e.is_nan() {
                        a.is_nan()
                    } else if !e.is_finite() {
                        a == e
                    } else {
                        a.is_finite() && (a == e || a.to_bits().abs_diff(e.to_bits()) <= 2)
                    },
                    "scalar={scalar} index={i}: {a:?} vs {e:?}"
                );
            }
        }
    }
    #[test]
    fn native_sqrt_full_float32_domain_retains_one_ulp_and_classes() {
        let mut x = vec![
            0.,
            -0.,
            f32::INFINITY,
            f32::NAN,
            -1.,
            f32::from_bits(1),
            f32::from_bits(0x7fffff),
            f32::MIN_POSITIVE,
            f32::MAX,
        ];
        let mut seed = 47u32;
        for _ in 0..1536 {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            x.push(f32::from_bits(1 + seed % 0x7f7fffff));
        }
        let expected = x
            .iter()
            .map(|&v| (v as f64).sqrt() as f32)
            .collect::<Vec<_>>();
        let y = run("sqrt", vec![1, x.len() as u32], &x, true);
        for (i, (&a, &e)) in y.iter().zip(&expected).enumerate() {
            assert!(
                if e.is_nan() {
                    a.is_nan()
                } else if !e.is_finite() {
                    a == e
                } else {
                    a.is_finite() && a.to_bits().abs_diff(e.to_bits()) <= 1
                },
                "sqrt[{i}] x={} actual={a:?} expected={e:?}",
                x[i]
            );
        }
    }
    #[test]
    fn native_softmax_finite_and_exceptional_rows() {
        let mut x = vec![0.; 6 * 128];
        for (row, tail) in [-5., -80., -90., -103., -104., f32::NEG_INFINITY]
            .into_iter()
            .enumerate()
        {
            x[row * 128] = 0.;
            x[row * 128 + 1..(row + 1) * 128].fill(tail);
        }
        let ordinary = run("softmax", vec![6, 128], &x, false);
        let actual = run("softmax", vec![6, 128], &x, true);
        for (row, tail) in [-5f64, -80., -90., -103., -104., f64::NEG_INFINITY]
            .into_iter()
            .enumerate()
        {
            let total = 1. + 127. * tail.exp();
            for col in 0..128 {
                let expected = (if col == 0 { 1. } else { tail.exp() } / total) as f32;
                let i = row * 128 + col;
                assert!(
                    actual[i].is_finite()
                        && actual[i].to_bits().abs_diff(expected.to_bits()) <= 387,
                    "softmax row={row} col={col}: {} vs {expected}",
                    actual[i]
                );
                if matches!(row, 2 | 3 | 5) {
                    // The fallback is native softmax. Its compiler context can
                    // change rounding; both paths retain the upstream budget.
                    assert!(
                        ordinary[i].is_finite()
                            && ordinary[i].to_bits().abs_diff(expected.to_bits()) <= 387
                    );
                }
            }
        }
    }
    #[test]
    fn native_sigmoid_tails_and_mean_cancellation() {
        let x = [
            -104.,
            -103.,
            -95.,
            -90.,
            -88.,
            -81.,
            -80.,
            -1.,
            0.,
            1.,
            80.,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ];
        let actual = run("sigmoid", vec![1, x.len() as u32], &x, true);
        for (&a, &x) in actual.iter().zip(&x) {
            let e = (1.0f64 / (1. + (-f64::from(x)).exp())) as f32;
            assert!(
                if e.is_nan() {
                    a.is_nan()
                } else {
                    a.is_finite() && a.to_bits().abs_diff(e.to_bits()) <= 34
                },
                "sigmoid {x}: {a} vs {e}"
            );
        }
        let mut x = vec![1.; 576];
        x[0] = 4096.;
        x[575] = -4096.;
        let actual = run("mean", vec![1, 576], &x, true);
        let e = (574.0f64 / 576.) as f32;
        assert!(
            actual[0].to_bits().abs_diff(e.to_bits()) <= 1,
            "mean {} vs {e}",
            actual[0]
        );
    }
}
