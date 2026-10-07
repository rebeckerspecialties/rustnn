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
}

fn fixtures() -> Vec<Fixture> {
    serde_json::from_str(include_str!("../tests/fixtures/numeric_boundaries.json")).unwrap()
}

fn value(bits: u32, width: u32) -> f64 {
    if width == 16 {
        half::f16::from_bits(bits as u16).to_f64()
    } else {
        f32::from_bits(bits) as f64
    }
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
            shape: to_dimension_vector(&[input.len() as u32]),
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
                Operation::from_json_attributes(op, &[0], &[1], &json!({}))
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
        let td = MLTensorDescriptor::new(mltype, vec![input.len() as u64]);
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
                let actual = match predict(
                    policy,
                    fixture.width,
                    &fixture.input_bits,
                    Some(&fixture.op),
                    None,
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
                        match predict(policy, fixture.width, input, unary, Some(op)) {
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
            assert!(matches!(f.op.as_str(), "exp" | "sqrt" | "gelu"));
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
