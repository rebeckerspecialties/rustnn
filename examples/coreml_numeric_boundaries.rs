//! Reduced arithmetic inventory, not a new composed WebNN tolerance contract.
//!
//! See docs/testing/wpt-test-guide.md. The report keeps standalone ULP checks,
//! nonfinite classes, signed zero and no-flush fidelity separate. In particular,
//! an Exp result within 32 ULP can still lose a deep subnormal. Consumer controls
//! use the actual once-rounded producer, not the ideal Exp result, so they do not
//! attribute the producer's approximation to Mul/Div. Do not move these stronger
//! checks into WPT before its composed-reference/underflow contract is agreed:
//! https://github.com/webmachinelearning/webnn/issues/950

use serde::{Deserialize, Serialize};

#[derive(Clone, Deserialize)]
struct Fixture {
    op: String,
    width: u32,
    ulp_budget: u32,
    input_bits: Vec<u32>,
    expected_bits: Vec<u32>,
    #[serde(default)]
    shape: Option<Vec<u32>>,
    #[serde(default)]
    axis: Option<u32>,
}

struct Geometry {
    shape: Vec<u32>,
    axis: Option<u32>,
}

impl Geometry {
    fn linear(count: usize) -> Self {
        Self {
            shape: vec![u32::try_from(count).expect("fixture length")],
            axis: None,
        }
    }

    fn rows(&self) -> Vec<Vec<usize>> {
        let axis = self.axis.expect("validated Softmax axis") as usize;
        let width = self.shape[axis] as usize;
        let inner: usize = self.shape[axis + 1..].iter().map(|&n| n as usize).product();
        let outer: usize = self.shape[..axis].iter().map(|&n| n as usize).product();
        (0..outer)
            .flat_map(|outer| {
                (0..inner).map(move |offset| {
                    (0..width)
                        .map(|i| (outer * width + i) * inner + offset)
                        .collect()
                })
            })
            .collect()
    }
}

impl Fixture {
    fn geometry(&self) -> Result<Geometry, String> {
        let shape = self
            .shape
            .clone()
            .unwrap_or_else(|| Geometry::linear(self.input_bits.len()).shape);
        let count = shape.iter().try_fold(1usize, |n, &size| {
            (size > 0).then_some(())?;
            n.checked_mul(size as usize)
        });
        if shape.is_empty()
            || count != Some(self.input_bits.len())
            || self.expected_bits.len() != self.input_bits.len()
        {
            return Err("fixture shape/count differs".into());
        }
        if self.op == "softmax" {
            let axis = self
                .axis
                .filter(|&axis| (axis as usize) < shape.len())
                .ok_or("invalid Softmax axis")?;
            if self.width != 32
                || shape[axis as usize]
                    .checked_mul(3)
                    .and_then(|n| n.checked_add(3))
                    != Some(self.ulp_budget)
                || self
                    .input_bits
                    .iter()
                    .any(|&bits| !value(bits, 32).is_finite())
            {
                return Err("finite Float32 Softmax fixture/budget differs".into());
            }
        } else if self.axis.is_some() {
            return Err("axis on non-Softmax fixture".into());
        }
        Ok(Geometry {
            shape,
            axis: self.axis,
        })
    }
}

fn fixtures() -> Vec<Fixture> {
    serde_json::from_str(include_str!("../tests/fixtures/numeric_boundaries.json")).unwrap()
}

#[derive(Debug, Serialize)]
struct SoftmaxFidelity {
    reference_bit_mismatches: usize,
    maximum_absolute_error: Option<f64>,
    row_sums_f64: Vec<Option<f64>>,
    reference_row_sums_f64: Vec<Option<f64>>,
}

// Observations only: neither exact reference bits nor exact unit sums are new
// WPT acceptance requirements. They expose legal local variation that can be
// amplified by later attention/projection/state operations in a real model.
fn softmax_fidelity(fixture: &Fixture, actual: &[u32]) -> Result<SoftmaxFidelity, String> {
    let geometry = fixture.geometry()?;
    if fixture.op != "softmax" || actual.len() != fixture.expected_bits.len() {
        return Err("Softmax observation input differs".into());
    }
    let row_sums = |bits: &[u32]| {
        geometry
            .rows()
            .iter()
            .map(|row| {
                let sum: f64 = row.iter().map(|&i| value(bits[i], 32)).sum();
                sum.is_finite().then_some(sum)
            })
            .collect()
    };
    let maximum_absolute_error = actual.iter().zip(&fixture.expected_bits).try_fold(
        0.0_f64,
        |maximum, (&actual, &expected)| {
            let error = (value(actual, 32) - value(expected, 32)).abs();
            error.is_finite().then_some(maximum.max(error))
        },
    );
    Ok(SoftmaxFidelity {
        reference_bit_mismatches: actual
            .iter()
            .zip(&fixture.expected_bits)
            .filter(|(a, e)| a != e)
            .count(),
        maximum_absolute_error,
        row_sums_f64: row_sums(actual),
        reference_row_sums_f64: row_sums(&fixture.expected_bits),
    })
}

fn value(bits: u32, width: u32) -> f64 {
    // Build binary64 using integer fields rather than hardware conversion.
    // A CoreML worker can enable FZ/FZ16: even widening a stored Float32
    // subnormal would then turn the diagnostic's reference into zero. Every
    // nonzero finite Half/Float32 value is a normal, exact binary64 value.
    // Keep this decoder independent of the production conversion helpers.
    let (fraction_bits, exponent_bits, bias) = match width {
        16 => (10, 5, 15),
        32 => (23, 8, 127),
        _ => panic!("unsupported diagnostic storage width"),
    };
    let sign = u64::from((bits >> (width - 1)) & 1) << 63;
    let exponent_mask = (1 << exponent_bits) - 1;
    let exponent = (bits >> fraction_bits) & exponent_mask;
    let fraction = bits & ((1 << fraction_bits) - 1);
    let widened = if exponent == exponent_mask {
        // Only NaN classification matters here; quiet the payload explicitly.
        sign | (0x7ff << 52) | if fraction == 0 { 0 } else { 1 << 51 }
    } else if exponent != 0 {
        let exponent = i64::from(exponent) - bias + 1023;
        sign | ((exponent as u64) << 52) | (u64::from(fraction) << (52 - fraction_bits))
    } else if fraction == 0 {
        sign
    } else {
        let top = 31 - fraction.leading_zeros();
        let exponent = 1 - bias - i64::from(fraction_bits) + i64::from(top) + 1023;
        let significand = u64::from(fraction) << (52 - top);
        sign | ((exponent as u64) << 52) | (significand & ((1 << 52) - 1))
    };
    f64::from_bits(widened)
}

fn magnitude_mask(width: u32) -> u32 {
    (1 << (width - 1)) - 1
}

fn ordered(bits: u32, width: u32) -> i64 {
    let magnitude = i64::from(bits & magnitude_mask(width));
    if bits & (1 << (width - 1)) != 0 {
        -magnitude
    } else {
        magnitude
    }
}

#[derive(Debug, Serialize)]
struct Comparison {
    actual_bits: u32,
    expected_bits: u32,
    finite_ulp: Option<u64>,
    within_budget_and_class: bool,
    flushed_nonzero: bool,
    zero_sign_changed: bool,
}

fn compare(actual: u32, expected: u32, width: u32, budget: u32) -> Comparison {
    let a = value(actual, width);
    let e = value(expected, width);
    let finite_ulp = (a.is_finite() && e.is_finite())
        .then(|| ordered(actual, width).abs_diff(ordered(expected, width)));
    Comparison {
        actual_bits: actual,
        expected_bits: expected,
        finite_ulp,
        within_budget_and_class: finite_ulp.map_or_else(
            || (a.is_nan() && e.is_nan()) || a == e,
            |distance| distance <= u64::from(budget),
        ),
        flushed_nonzero: a == 0.0 && e.is_finite() && e != 0.0,
        zero_sign_changed: a == 0.0 && e == 0.0 && actual != expected,
    }
}

// The selected factors are exact powers of two. All nonzero Exp tails become
// normal outputs without losing significand bits. Reference generation never
// calls a hardware Half narrowing instruction.
fn scaled_bits(bits: u32, width: u32) -> u32 {
    if width == 16 {
        let v = value(bits, width) * 1024.0;
        // These results are exactly representable, including the zero control.
        let magnitude = bits & 0x7fff;
        if magnitude == 0 {
            return bits;
        }
        assert!(magnitude < 0x400, "consumer requires a subnormal input");
        let top = 31 - magnitude.leading_zeros();
        let exponent = top + 1;
        let result = (bits & 0x8000) | (exponent << 10) | ((magnitude << (10 - top)) & 0x3ff);
        assert_eq!(value(result, width), v);
        result
    } else {
        ((value(bits, width) * 2f64.powi(100)) as f32).to_bits()
    }
}

#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
mod native {
    use super::*;
    use rustnn::backend_selection::{BackendDevice, DeviceType};
    use rustnn::graph::{
        DataType, GraphInfo, Operand, OperandDescriptor, OperandKind, to_dimension_vector,
    };
    use rustnn::mlcontext::{
        MLContext, MLContextOptions, MLNamedTensors, MLPowerPreference, MLTensorDescriptor,
    };
    use rustnn::operator_enums::MLOperandDataType;
    use rustnn::operators::Operation;
    use serde_json::json;

    fn predict(
        policy: DeviceType,
        width: u32,
        input: &[u32],
        unary: Option<&str>,
        consumer: Option<&str>,
        geometry: &Geometry,
    ) -> Result<Vec<u32>, String> {
        let dtype = if width == 16 {
            DataType::Float16
        } else {
            DataType::Float32
        };
        let mltype = if width == 16 {
            MLOperandDataType::Float16
        } else {
            MLOperandDataType::Float32
        };
        let descriptor = OperandDescriptor {
            data_type: dtype,
            shape: to_dimension_vector(&geometry.shape),
            pending_permutation: vec![],
        };
        let operand = |name: &str, kind| Operand {
            name: Some(name.into()),
            kind,
            descriptor: descriptor.clone(),
        };
        let mut source = GraphInfo {
            operands: vec![operand("input", OperandKind::Input)],
            input_operands: vec![0],
            ..Default::default()
        };
        let mut last = 0;
        if let Some(op) = unary {
            source
                .operands
                .push(operand("producer", OperandKind::Output));
            source.operations.push(
                Operation::from_json_attributes(
                    op,
                    &[0],
                    &[1],
                    &geometry
                        .axis
                        .map_or_else(|| json!({}), |axis| json!({"axis": axis})),
                )
                .ok_or_else(|| format!("unsupported unary fixture {op}"))?,
            );
            last = 1;
        }
        if let Some(op) = consumer {
            if unary.is_some() {
                source.operands[last as usize].kind = OperandKind::Intermediate;
            }
            let factor = source.operands.len() as u32;
            source.operands.push(operand("factor", OperandKind::Input));
            source.input_operands.push(factor);
            let output = source.operands.len() as u32;
            source.operands.push(operand("result", OperandKind::Output));
            source.operations.push(
                Operation::from_json_attributes(op, &[last, factor], &[output], &json!({}))
                    .ok_or_else(|| format!("unsupported consumer fixture {op}"))?,
            );
            last = output;
        }
        source.operands[last as usize].name = Some("result".into());
        source.output_operands = vec![last];
        let mut context = MLContext::create(
            &MLContextOptions::new(MLPowerPreference::Default, policy != DeviceType::Cpu)
                .with_rustnn_device_hint(BackendDevice::Coreml {
                    device_type: policy,
                }),
        )
        .map_err(|e| e.to_string())?;
        let mut graph = context
            .rustnn_build_graph(source)
            .map_err(|e| e.to_string())?;
        let Some(rustnn::mlcontext::LoadDiagnostics::Coreml(load)) =
            graph.rustnn_load_diagnostics()
        else {
            return Err("missing CoreML load diagnostics".into());
        };
        if load.requested_compute_units != load.loaded_compute_units {
            return Err(format!("policy changed during loading: {load:?}"));
        }
        let td = MLTensorDescriptor::new(
            mltype,
            geometry.shape.iter().map(|&n| u64::from(n)).collect(),
        );
        let tensor = context
            .create_tensor(&td.clone().to_writable())
            .map_err(|e| e.to_string())?;
        let output = context
            .create_tensor(&td.clone().to_readable())
            .map_err(|e| e.to_string())?;
        let bytes = |values: &[u32]| -> Vec<u8> {
            values
                .iter()
                .flat_map(|b| b.to_le_bytes().into_iter().take((width / 8) as usize))
                .collect()
        };
        context
            .write_tensor(&tensor, &bytes(input))
            .map_err(|e| e.to_string())?;
        let factor = if let Some(op) = consumer {
            let bits = match (width, op) {
                (16, "mul") => 0x6400,
                (16, _) => 0x1400,
                (_, "mul") => 0x71800000,
                _ => 0x0d800000,
            };
            let factor = context
                .create_tensor(&td.to_writable())
                .map_err(|e| e.to_string())?;
            context
                .write_tensor(&factor, &bytes(&vec![bits; input.len()]))
                .map_err(|e| e.to_string())?;
            Some(factor)
        } else {
            None
        };
        let mut inputs = MLNamedTensors::from([("input", &tensor)]);
        if let Some(factor) = &factor {
            inputs.insert("factor", factor);
        }
        context
            .dispatch(
                &mut graph,
                &inputs,
                &MLNamedTensors::from([("result", &output)]),
            )
            .map_err(|e| e.to_string())?;
        let mut result = vec![0u8; input.len() * (width / 8) as usize];
        context
            .read_tensor(&output, &mut result)
            .map_err(|e| e.to_string())?;
        Ok(result
            .chunks_exact((width / 8) as usize)
            .map(|b| {
                if width == 16 {
                    u32::from(u16::from_le_bytes(b.try_into().unwrap()))
                } else {
                    u32::from_le_bytes(b.try_into().unwrap())
                }
            })
            .collect())
    }

    pub fn run() -> Result<(), String> {
        let mut rows = vec![];
        let mut errors = vec![];
        for policy in [DeviceType::Cpu, DeviceType::Gpu, DeviceType::Npu] {
            for fixture in fixtures() {
                let geometry = fixture.geometry()?;
                let actual = match predict(
                    policy,
                    fixture.width,
                    &fixture.input_bits,
                    Some(&fixture.op),
                    None,
                    &geometry,
                ) {
                    Ok(actual) => actual,
                    Err(error) => {
                        errors.push(json!({"policy": format!("{policy:?}"), "op": fixture.op, "width": fixture.width, "error": error}));
                        continue;
                    }
                };
                let checks: Vec<_> = actual
                    .iter()
                    .zip(&fixture.expected_bits)
                    .map(|(&a, &e)| compare(a, e, fixture.width, fixture.ulp_budget))
                    .collect();
                if fixture.op == "softmax" {
                    rows.push(json!({"requested_policy": format!("{policy:?}"), "op": fixture.op, "width": fixture.width, "shape": geometry.shape, "axis": geometry.axis, "input_bits": fixture.input_bits, "reference": "stable Decimal Softmax at 100/140 digits, rounded once to Float32", "standalone_ulp_budget": fixture.ulp_budget, "budget_source": "WPT getSoftmaxPrecisionTolerance: 3 * axis extent + 3; not a normative model-error contract", "checks": checks, "source_fidelity_observations": softmax_fidelity(&fixture, &actual)?, "fidelity_scope": "Reference-bit and row-sum observations, not additional pass criteria or a composed-model gate"}));
                    continue;
                }
                rows.push(json!({"requested_policy": format!("{policy:?}"), "op": fixture.op, "width": fixture.width, "input_bits": fixture.input_bits, "reference": "once-rounded Decimal unary", "standalone_ulp_budget": fixture.ulp_budget, "checks": checks}));
                if fixture.op != "exp" {
                    continue;
                }
                // Keep only deep tails (including the underflow-to-zero control).
                // Avoid trusting a producer's finite classification to select tests.
                let limit = if fixture.width == 16 { 0x400 } else { 0x800000 };
                let indexes: Vec<_> = fixture
                    .expected_bits
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &b)| (b < limit).then_some(i))
                    .collect();
                let producer: Vec<_> = indexes.iter().map(|&i| actual[i]).collect();
                if producer.iter().any(|&b| b >= limit) {
                    errors.push(json!({"op": "exp", "width": fixture.width, "error": "invalid producer class; consumer attribution unavailable"}));
                    continue;
                }
                let input: Vec<_> = indexes.iter().map(|&i| fixture.input_bits[i]).collect();
                let expected: Vec<_> = producer
                    .iter()
                    .map(|&b| scaled_bits(b, fixture.width))
                    .collect();
                let reference_producer: Vec<_> =
                    indexes.iter().map(|&i| fixture.expected_bits[i]).collect();
                let reference_scaled: Vec<_> = reference_producer
                    .iter()
                    .map(|&b| scaled_bits(b, fixture.width))
                    .collect();
                for op in ["mul", "div"] {
                    for (name, unary, input, expected) in [
                        ("direct-actual", None, &producer, &expected),
                        (
                            "direct-reference",
                            None,
                            &reference_producer,
                            &reference_scaled,
                        ),
                        ("composed", Some("exp"), &input, &expected),
                    ] {
                        match predict(policy, fixture.width, input, unary, Some(op), &Geometry::linear(input.len())) {
                            Ok(result) => {
                                let checks: Vec<_> = result.iter().zip(expected).map(|(&a, &e)| compare(a, e, fixture.width, 0)).collect();
                                rows.push(json!({"requested_policy": format!("{policy:?}"), "op": op, "width": fixture.width, "route": name, "input_bits": input, "producer_bits": producer, "reference": "exact power-of-two scaling of stored input (direct) or actual isolated producer (composed); local fidelity, not composed WPT budget", "checks": checks}));
                            }
                            Err(error) => errors.push(json!({"policy": format!("{policy:?}"), "op": op, "width": fixture.width, "route": name, "error": error})),
                        }
                    }
                }
            }
        }
        println!(
            "{}",
            serde_json::to_string_pretty(
                &json!({"placement": "not measured", "rows": rows, "execution_errors": errors})
            )
            .unwrap()
        );
        if errors.is_empty() {
            Ok(())
        } else {
            Err("one or more diagnostics could not execute; see report".into())
        }
    }
}

fn main() {
    #[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
    if let Err(error) = native::run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
    #[cfg(not(all(target_os = "macos", feature = "coreml-runtime")))]
    {
        // The independent oracle and classifier tests are available on every host.
        for f in fixtures() {
            assert!(matches!(f.op.as_str(), "exp" | "sqrt" | "gelu" | "softmax"));
            f.geometry().expect("valid reference fixture");
            if f.op == "softmax" {
                let _ = softmax_fidelity(&f, &f.expected_bits).expect("valid Softmax observation");
            }
            assert_eq!(f.input_bits.len(), f.expected_bits.len());
            for b in f.expected_bits {
                let _ = compare(b, b, f.width, f.ulp_budget);
            }
        }
        let _ = scaled_bits(1, 16);
        eprintln!(
            "Run with coreml-runtime on macOS to collect native output; use make test-numeric-boundaries for portable checks."
        );
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_value_decoding() {
        // Explicit binary64 encodings, not another Half/Float32 conversion.
        for (width, bits, expected) in [
            (16, 0, 0),
            (16, 0x8000, 0x8000_0000_0000_0000),
            (16, 1, 0x3e70_0000_0000_0000),
            (16, 0x03ff, 0x3f0f_f800_0000_0000),
            (16, 0x0400, 0x3f10_0000_0000_0000),
            (16, 0x7bff, 0x40ef_fc00_0000_0000),
            (16, 0x7c00, 0x7ff0_0000_0000_0000),
            (16, 0xfc00, 0xfff0_0000_0000_0000),
            (32, 0, 0),
            (32, 0x8000_0000, 0x8000_0000_0000_0000),
            (32, 1, 0x36a0_0000_0000_0000),
            (32, 0x007f_ffff, 0x380f_ffff_c000_0000),
            (32, 0x0080_0000, 0x3810_0000_0000_0000),
            (32, 0x7f7f_ffff, 0x47ef_ffff_e000_0000),
            (32, 0x7f80_0000, 0x7ff0_0000_0000_0000),
            (32, 0xff80_0000, 0xfff0_0000_0000_0000),
        ] {
            assert_eq!(
                value(std::hint::black_box(bits), width).to_bits(),
                expected,
                "width={width}, bits={bits:#x}"
            );
            if bits > 0 && bits < magnitude_mask(width) {
                assert_eq!(
                    value(std::hint::black_box(bits | (1 << (width - 1))), width).to_bits(),
                    expected | (1 << 63)
                );
            }
        }
        for (width, bits) in [
            (16, 0x7e00),
            (16, 0x7c01),
            (32, 0x7fc0_0000),
            (32, 0x7f80_0001),
        ] {
            assert!(value(std::hint::black_box(bits), width).is_nan());
        }
    }

    #[test]
    fn diagnostic_value_decoding_preserves_both_storage_formats() {
        assert_value_decoding();
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn diagnostic_oracle_survives_worker_flush_modes_and_restores_fpcr() {
        std::thread::spawn(|| {
            struct Restore(u64);
            impl Drop for Restore {
                fn drop(&mut self) {
                    // SAFETY: restore only this dedicated worker's saved state.
                    unsafe {
                        core::arch::asm!("msr fpcr, {value}", value = in(reg) self.0,
                            options(nostack, preserves_flags));
                    }
                }
            }
            let original: u64;
            // SAFETY: read only the current test worker's control register.
            unsafe {
                core::arch::asm!("mrs {value}, fpcr", value = out(reg) original,
                    options(nostack, preserves_flags));
            }
            {
                let _restore = Restore(original);
                let flush_mask = (1 << 19) | (1 << 24);
                for mask in [0, 1 << 19, 1 << 24, flush_mask] {
                    let changed = (original & !flush_mask) | mask;
                    // SAFETY: FZ16/FZ affect only this worker, restored even on panic.
                    unsafe {
                        core::arch::asm!("msr fpcr, {value}", value = in(reg) changed,
                            options(nostack, preserves_flags));
                    }
                    assert_value_decoding();
                    distinguishes_allowed_ulp_from_stronger_no_flush_and_zero_sign_properties();
                    softmax_subnormal_and_nonfinite_observations_do_not_disappear();
                    consumer_control_detects_flushing_without_charging_exp_approximation_twice();
                }
            }
            let restored: u64;
            // SAFETY: verify this worker's original control register was restored.
            unsafe {
                core::arch::asm!("mrs {value}, fpcr", value = out(reg) restored,
                    options(nostack, preserves_flags));
            }
            assert_eq!(restored, original);
        })
        .join()
        .unwrap();
    }

    #[test]
    fn every_reference_preserves_itself_and_classification() {
        for f in fixtures() {
            assert_eq!(f.input_bits.len(), f.expected_bits.len());
            for bits in f.expected_bits {
                let c = compare(bits, bits, f.width, f.ulp_budget);
                assert!(c.within_budget_and_class && !c.flushed_nonzero && !c.zero_sign_changed);
            }
        }
    }

    #[test]
    fn softmax_inventory_includes_independently_rounded_finite_cases() {
        let cases: Vec<_> = fixtures()
            .into_iter()
            .filter(|fixture| fixture.op == "softmax")
            .collect();
        assert!(!cases.is_empty(), "missing finite Softmax diagnostics");
        for fixture in cases {
            assert_eq!(fixture.width, 32);
            assert!(
                fixture
                    .input_bits
                    .iter()
                    .all(|&bits| value(bits, 32).is_finite())
            );
            assert_eq!(fixture.input_bits.len(), fixture.expected_bits.len());
        }
    }

    #[test]
    fn softmax_rows_follow_the_declared_axis_and_singletons_are_one() {
        let first = fixtures().into_iter().find(|f| f.axis == Some(0)).unwrap();
        assert_eq!(
            first.geometry().unwrap().rows(),
            [vec![0, 2, 4], vec![1, 3, 5]]
        );
        let middle = fixtures()
            .into_iter()
            .find(|f| f.shape == Some(vec![2, 3, 2]))
            .unwrap();
        assert_eq!(
            middle.geometry().unwrap().rows(),
            [vec![0, 2, 4], vec![1, 3, 5], vec![6, 8, 10], vec![7, 9, 11]]
        );
        for fixture in fixtures().into_iter().filter(|f| f.op == "softmax") {
            let geometry = fixture.geometry().unwrap();
            let mut visited: Vec<_> = geometry.rows().into_iter().flatten().collect();
            visited.sort_unstable();
            assert_eq!(visited, (0..fixture.input_bits.len()).collect::<Vec<_>>());
            assert_eq!(
                fixture.ulp_budget,
                3 * geometry.shape[geometry.axis.unwrap() as usize] + 3
            );
            let observation = softmax_fidelity(&fixture, &fixture.expected_bits).unwrap();
            assert_eq!(observation.reference_bit_mismatches, 0);
            assert_eq!(observation.maximum_absolute_error, Some(0.0));
            if geometry.shape[geometry.axis.unwrap() as usize] == 1 {
                assert!(fixture.expected_bits.iter().all(|&bits| bits == 0x3f800000));
                assert!(observation.row_sums_f64.iter().all(|&sum| sum == Some(1.0)));
            }
        }
    }

    #[test]
    fn softmax_allowed_rounding_difference_is_retained_as_an_observation() {
        let fixture = fixtures()
            .into_iter()
            .find(|f| f.shape == Some(vec![4, 2]))
            .unwrap();
        // For odd n and small x=n*2^-23, logistic(x) is just below
        // 1/2+x/4, a Float32 midpoint. A Double intermediate can hit the
        // midpoint instead. Both answers still fit the current WPT budget.
        assert_eq!(fixture.expected_bits[0], 0x3f000001);
        let mut actual = fixture.expected_bits.clone();
        actual[0] += 1;
        let check = compare(actual[0], fixture.expected_bits[0], 32, fixture.ulp_budget);
        assert!(check.within_budget_and_class);
        assert_eq!(check.finite_ulp, Some(1));
        let observation = softmax_fidelity(&fixture, &actual).unwrap();
        assert_eq!(observation.reference_bit_mismatches, 1);
        assert_eq!(observation.maximum_absolute_error, Some(2f64.powi(-24)));
        let recorded = serde_json::to_value(&check).unwrap();
        assert_eq!(recorded["actual_bits"], actual[0]);
        assert_eq!(recorded["expected_bits"], fixture.expected_bits[0]);
    }

    #[test]
    fn softmax_subnormal_and_nonfinite_observations_do_not_disappear() {
        let fixture = fixtures()
            .into_iter()
            .find(|f| f.shape == Some(vec![3, 2]) && f.axis == Some(1))
            .unwrap();
        assert!(fixture.expected_bits[1] > 0 && fixture.expected_bits[1] < 0x800000);
        assert_eq!(fixture.expected_bits[3], 1);
        assert_eq!(fixture.expected_bits[5], 0);
        let outside = compare(0, fixture.expected_bits[1], 32, fixture.ulp_budget);
        assert!(!outside.within_budget_and_class && outside.flushed_nonzero);
        assert_eq!(outside.finite_ulp, Some(584_744));
        let check = compare(0, fixture.expected_bits[3], 32, fixture.ulp_budget);
        assert!(check.within_budget_and_class && check.flushed_nonzero);
        let mut actual = fixture.expected_bits.clone();
        actual[1] = 0;
        let observation = softmax_fidelity(&fixture, &actual).unwrap();
        assert_eq!(
            observation.maximum_absolute_error,
            Some(584_744.0 * f64::from_bits(0x36a0_0000_0000_0000))
        );
        actual.fill(0);
        actual[1] = fixture.expected_bits[1];
        let observation = softmax_fidelity(&fixture, &actual).unwrap();
        assert_eq!(
            observation.row_sums_f64[0],
            Some(584_744.0 * f64::from_bits(0x36a0_0000_0000_0000))
        );
        actual = fixture.expected_bits.clone();
        actual[0] = f32::NAN.to_bits();
        let observation = softmax_fidelity(&fixture, &actual).unwrap();
        assert_eq!(observation.reference_bit_mismatches, 1);
        assert_eq!(observation.maximum_absolute_error, None);
        assert_eq!(observation.row_sums_f64[0], None);
        assert!(
            !compare(actual[0], fixture.expected_bits[0], 32, fixture.ulp_budget)
                .within_budget_and_class
        );
    }

    #[test]
    fn malformed_softmax_fixtures_fail_before_execution() {
        let fixture = fixtures().into_iter().find(|f| f.op == "softmax").unwrap();
        for shape in [vec![], vec![0], vec![3], vec![u32::MAX; 3]] {
            let mut broken = fixture.clone();
            broken.shape = Some(shape);
            assert!(broken.geometry().is_err());
        }
        for axis in [None, Some(2), Some(u32::MAX)] {
            let mut broken = fixture.clone();
            broken.axis = axis;
            assert!(broken.geometry().is_err());
        }
        let mut broken = fixture.clone();
        broken.ulp_budget += 1;
        assert!(broken.geometry().is_err());
        broken = fixture.clone();
        broken.input_bits[0] = f32::INFINITY.to_bits();
        assert!(broken.geometry().is_err());
        assert!(softmax_fidelity(&fixture, &[]).is_err());
    }

    #[test]
    fn distinguishes_allowed_ulp_from_stronger_no_flush_and_zero_sign_properties() {
        for width in [16, 32] {
            let c = compare(0, 1, width, 1);
            assert!(c.within_budget_and_class && c.flushed_nonzero);
            let c = compare(0, 1 << (width - 1), width, 0);
            assert!(c.within_budget_and_class && c.zero_sign_changed);
            let inf = if width == 16 { 0x7c00 } else { 0x7f800000 };
            assert!(!compare(inf, inf - 1, width, u32::MAX).within_budget_and_class);
        }
    }

    #[test]
    fn sqrt_boundary_oracle_contains_exact_powers_and_domain_classes() {
        for f in fixtures().into_iter().filter(|f| f.op == "sqrt") {
            assert_eq!(f.expected_bits[0], 0);
            assert_eq!(f.expected_bits[1], 1 << (f.width - 1));
            assert!(value(f.expected_bits[9], f.width).is_nan());
            assert!(value(f.expected_bits[10], f.width).is_infinite());
            for i in [if f.width == 16 { 2usize } else { 3 }, 5, 7] {
                let root = value(f.expected_bits[i], f.width);
                assert_eq!(root * root, value(f.input_bits[i], f.width));
            }
        }
    }

    #[test]
    fn consumer_control_detects_flushing_without_charging_exp_approximation_twice() {
        for width in [16, 32] {
            for bits in [1, 2, 27, 103, 762] {
                let expected = scaled_bits(bits, width);
                assert!(value(expected, width).is_normal());
                assert!(!compare(0, expected, width, 1).within_budget_and_class);
                assert!(compare(expected, expected, width, 0).within_budget_and_class);
            }
        }
    }
}
