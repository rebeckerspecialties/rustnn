//! Emit MIL ASCII text from a [`MILProgram`].
//!
//! The output format matches Apple's `coremlc` output for `model.mil` inside
//! `.mlmodelc` bundles. Key rules derived from the reference output:
//!
//! - Opset tag: `CoreML8` → `ios18`, `CoreML7` → `ios17`, `CoreML6` → `ios16`.
//! - `const` ops:  `type name = const()[val = type(literal)];`
//! - Regular ops: `type name = opname(args sorted alpha);`
//! - Floats in C99 hex: `0x1.8p-3`, `0x0p+0`.
//! - Tensors with nested brackets: `[[[[0x1p+1]]]]`.
//! - Multi-output ops list each output as `type name`, comma-separated, with
//!   no enclosing brackets or parens — Apple's loader rejects both `(...)`
//!   and `[...]` LHS forms.

use crate::description::{FunctionDescription, ModelDescription, ShapeFlexibility, shape_text};
use crate::hex_float::{hex_float16, hex_float32, hex_float32_bytes};
use crate::sink::MILOutputSink;
use crate::types::*;

/// Emit the full MIL text for `program` into an in-memory string.
pub fn emit_to_string(program: &MILProgram) -> String {
    let mut sink = MILOutputSink::in_memory(4096);
    emit(program, &mut sink);
    String::from_utf8(sink.finalize().unwrap_or_default()).unwrap_or_default()
}

/// Emit the full MIL text for `program` into the given sink.
pub fn emit(program: &MILProgram, out: &mut MILOutputSink) {
    emit_program(program, out, false);
}

// Call only after validating the source program's asset paths. Pipeline children
// live exactly one level below the bundle root; they borrow its single assets.
pub(crate) fn emit_pipeline_child(program: &MILProgram, out: &mut MILOutputSink) {
    emit_program(program, out, true);
}

fn emit_program(program: &MILProgram, out: &mut MILOutputSink, shared_root: bool) {
    out.write_str("program(1.3)\n");
    out.write_str("[buildInfo = dict<string, string>(");
    out.write_str("{{\"coremlc-component-MIL\", \"MILTextCompiler\"}, ");
    out.write_str("{\"coremlc-version\", \"1.0.0\"}");
    out.write_str("})]\n");
    out.write_str("{\n");

    let description = ModelDescription::decode(&program.description_data);
    for (name, function) in &program.functions {
        emit_function(name, function, description.function(name), out, shared_root);
    }

    out.write_str("}");
}

fn emit_function(
    name: &str,
    function: &MILFunction,
    description: Option<&FunctionDescription>,
    out: &mut MILOutputSink,
    shared_root: bool,
) {
    let tag = opset_tag(&function.opset);
    out.write_str(&format!("    func {name}<{tag}>("));

    for (i, input) in function.inputs.iter().enumerate() {
        if i > 0 {
            out.write_str(", ");
        }
        out.write_str(&format!(
            "{} {}",
            input.r#type.text_representation(),
            input.name
        ));
    }

    out.write_str(")");
    if let Some(description) = description {
        emit_flexible_shapes(description, out);
    }
    out.write_str(" {\n");

    for op in &function.block.operations {
        emit_operation(op, out, shared_root);
    }

    let outs = function.block.outputs.join(", ");
    out.write_str(&format!("        }} -> ({outs});\n"));
}

fn emit_flexible_shapes(description: &FunctionDescription, out: &mut MILOutputSink) {
    let mut features: Vec<_> = description
        .inputs
        .iter()
        .filter(|feature| matches!(feature.flexibility, Some(ShapeFlexibility::Ranges(_))))
        .collect();
    features.sort_by(|a, b| a.name.cmp(&b.name));
    if features.is_empty() {
        return;
    }
    let defaults = features
        .iter()
        .map(|feature| {
            format!(
                "{{\"{}\", {}}}",
                feature.name,
                shape_text(&feature.default_shape())
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    let entries = features
        .iter()
        .filter_map(|feature| {
            let Some(ShapeFlexibility::Ranges(ranges)) = &feature.flexibility else {
                return None;
            };
            let ranges = ranges
                .iter()
                .map(|(lo, hi)| format!("[{lo}, {hi}]"))
                .collect::<Vec<_>>()
                .join(", ");
            Some(format!("{{\"{}\", [{ranges}]}}", feature.name))
        })
        .collect::<Vec<_>>()
        .join(", ");
    out.write_str(&format!(
        " [FlexibleShapeInformation = tuple<tuple<string, dict<string, tensor<int32, [?]>>>, tuple<string, dict<string, list<tensor<int32, [2]>, ?>>>>(((\"DefaultShapes\", {{{defaults}}}), (\"RangeDims\", {{{entries}}})))]"
    ));
}

fn emit_operation(op: &MILOperation, out: &mut MILOutputSink, shared_root: bool) {
    emit_operation_at(op, out, shared_root, 12);
}

fn emit_operation_at(op: &MILOperation, out: &mut MILOutputSink, shared_root: bool, depth: usize) {
    let padding = " ".repeat(depth);
    let indent = padding.as_str();

    let Some(first_output) = op.outputs.first() else {
        return;
    };

    if op.r#type == "const" {
        emit_const_operation(first_output, &op.attributes, indent, out, shared_root);
        return;
    }

    out.write_str(indent);

    if op.outputs.len() > 1 {
        // MIL multi-output ops list each output as `type name` separated by
        // commas, with no enclosing brackets or parens. The two alternative
        // forms we tried — `(t n, ...)` and `[t n, ...]` — are both rejected
        // by Apple's loader with "Unexpected token type: got '(' / '['
        // when expecting RBRACKET".
        for (i, o) in op.outputs.iter().enumerate() {
            if i > 0 {
                out.write_str(", ");
            }
            out.write_str(&format!("{} {}", o.r#type.text_representation(), o.name));
        }
        out.write_str(" = ");
    } else {
        out.write_str(&format!(
            "{} {} = ",
            first_output.r#type.text_representation(),
            first_output.name
        ));
    }

    out.write_str(&format!("{}(", op.r#type));

    let mut sorted_inputs: Vec<&(String, Vec<MILBinding>)> = op.inputs.iter().collect();
    sorted_inputs.sort_by(|a, b| a.0.cmp(&b.0));
    for (i, (key, bindings)) in sorted_inputs.iter().enumerate() {
        if i > 0 {
            out.write_str(", ");
        }
        out.write_str(&format!("{key} = "));
        emit_bindings(bindings, out, shared_root);
    }

    out.write_str(")");
    if !op.attributes.is_empty() {
        out.write_str("[");
        let mut attributes: Vec<_> = op.attributes.iter().collect();
        attributes.sort_by(|a, b| a.0.cmp(&b.0));
        for (i, (key, value)) in attributes.into_iter().enumerate() {
            if i > 0 {
                out.write_str(", ");
            }
            out.write_str(&format!("{key} = "));
            emit_typed_literal(value, out, shared_root);
        }
        out.write_str("]");
    }
    for block in &op.blocks {
        out.write_str("\n");
        out.write_str(&" ".repeat(depth + 4));
        out.write_str("(");
        for (index, input) in block.inputs.iter().enumerate() {
            if index > 0 {
                out.write_str(", ");
            }
            out.write_str(&format!(
                "{} {}",
                input.r#type.text_representation(),
                input.name
            ));
        }
        out.write_str(") {\n");
        for child in &block.operations {
            emit_operation_at(child, out, shared_root, depth + 8);
        }
        out.write_str(&format!(
            "{} }} -> ({})",
            " ".repeat(depth + 3),
            block.outputs.join(", ")
        ));
    }
    out.write_str(";\n");
}

fn emit_const_operation(
    output: &MILNamedType,
    attributes: &[(String, MILValue)],
    indent: &str,
    out: &mut MILOutputSink,
    shared_root: bool,
) {
    let type_str = output.r#type.text_representation();

    let val_attr = attributes.iter().find(|(k, _)| k == "val");

    let Some((_, value)) = val_attr else {
        out.write_str(&format!(
            "{indent}{type_str} {} = const()[];\n",
            output.name
        ));
        return;
    };

    out.write_str(&format!(
        "{indent}{type_str} {} = const()[val = ",
        output.name
    ));
    emit_typed_literal(value, out, shared_root);
    out.write_str("];\n");
}

fn emit_bindings(bindings: &[MILBinding], out: &mut MILOutputSink, shared_root: bool) {
    if bindings.len() == 1 {
        emit_binding(&bindings[0], out, shared_root);
        return;
    }
    out.write_str("(");
    for (i, b) in bindings.iter().enumerate() {
        if i > 0 {
            out.write_str(", ");
        }
        emit_binding(b, out, shared_root);
    }
    out.write_str(")");
}

fn emit_binding(binding: &MILBinding, out: &mut MILOutputSink, shared_root: bool) {
    match binding {
        MILBinding::Reference(name) => out.write_str(name),
        MILBinding::Immediate(value) => emit_typed_literal(value, out, shared_root),
    }
}

fn emit_typed_literal(value: &MILValue, out: &mut MILOutputSink, shared_root: bool) {
    if let Some(blob) = &value.blob {
        out.write_str(&format!("{}(", value.r#type.text_representation()));
        emit_blob_reference(blob, out, shared_root);
        out.write_str(")");
        return;
    }

    if value.r#type.is_scalar() {
        emit_scalar_literal(value, out);
    } else {
        out.write_str(&format!("{}(", value.r#type.text_representation()));
        // The decoder rejects dynamic immediate values; MIL values are concrete.
        let shape = value
            .r#type
            .concrete_shape()
            .expect("immediate tensor shape must be concrete");
        emit_nested_tensor_values(value, &shape, out);
        out.write_str(")");
    }
}

fn emit_blob_reference(blob: &MILBlobRef, out: &mut MILOutputSink, shared_root: bool) {
    // Apple's loader expects typed MIL literals here, not raw string/integer.
    // Without the `string(...)` and `uint64(...)` wrappers the parser reports
    // "Type declaration expected here" at the first character after `path = `.
    let filename = if shared_root {
        format!(
            "@model_path/../{}",
            blob.filename
                .strip_prefix("@model_path/")
                .expect("validated asset prefix")
        )
    } else {
        blob.filename.clone()
    };
    out.write_str(&format!(
        "BLOBFILE(path = string(\"{}\"), offset = uint64({}))",
        filename, blob.offset
    ));
}

fn emit_scalar_literal(value: &MILValue, out: &mut MILOutputSink) {
    let type_name = value.r#type.data_type.text_name();
    match &value.tensor {
        MILTensorData::Floats(v) => {
            let f = v.first().copied().unwrap_or(0.0);
            out.write_str(&format!("{type_name}({})", hex_float32(f)));
        }
        MILTensorData::Ints(v) => {
            out.write_str(&format!("{type_name}({})", v.first().copied().unwrap_or(0)));
        }
        MILTensorData::Bools(v) => {
            let b = v.first().copied().unwrap_or(false);
            out.write_str(&format!(
                "{type_name}({})",
                if b { "true" } else { "false" }
            ));
        }
        MILTensorData::Strings(v) => {
            let s: &str = v.first().map(|s| s.as_str()).unwrap_or("");
            out.write_str(&format!("{type_name}(\"{s}\")"));
        }
        MILTensorData::Fp16Bytes(d) => {
            if d.len() >= 2 {
                let bits = (d[0] as u16) | ((d[1] as u16) << 8);
                out.write_str(&format!("{type_name}({})", hex_float16(bits)));
            } else {
                out.write_str(&format!("{type_name}(0x0p+0)"));
            }
        }
        MILTensorData::LongInts(v) => {
            out.write_str(&format!("{type_name}({})", v.first().copied().unwrap_or(0)));
        }
        MILTensorData::Doubles(v) => {
            let d = v.first().copied().unwrap_or(0.0);
            out.write_str(&format!("{type_name}({})", hex_float32(d as f32)));
        }
    }
}

fn emit_nested_tensor_values(value: &MILValue, shape: &[usize], out: &mut MILOutputSink) {
    match &value.tensor {
        MILTensorData::Floats(values) => {
            // Uniform-value fast path: zero-padding, masked heads, and
            // post-quantisation constants hit this in sparse/test graphs.
            if let Some(&first) = values.first()
                && values.iter().all(|v| v.to_bits() == first.to_bits())
            {
                emit_uniform_tensor(shape, &hex_float32(first), out);
                return;
            }
            emit_dense_float32(values, shape, out);
        }
        MILTensorData::Ints(values) => {
            if let Some(&first) = values.first()
                && values.iter().all(|v| *v == first)
            {
                emit_uniform_tensor(shape, &first.to_string(), out);
                return;
            }
            let mut idx = 0usize;
            emit_nested_array(shape, 0, &mut idx, out, &|i| values[i].to_string());
        }
        MILTensorData::Bools(values) => {
            let mut idx = 0usize;
            emit_nested_array(shape, 0, &mut idx, out, &|i| {
                if values[i] {
                    "true".to_string()
                } else {
                    "false".to_string()
                }
            });
        }
        MILTensorData::Strings(values) => {
            let mut idx = 0usize;
            emit_nested_array(shape, 0, &mut idx, out, &|i| format!("\"{}\"", values[i]));
        }
        MILTensorData::Fp16Bytes(data) => {
            if data.len() >= 2 && data.len() % 2 == 0 && is_fp16_uniform(data) {
                let bits = (data[0] as u16) | ((data[1] as u16) << 8);
                emit_uniform_tensor(shape, &hex_float16(bits), out);
                return;
            }
            let mut idx = 0usize;
            emit_nested_array(shape, 0, &mut idx, out, &|i| {
                let off = i * 2;
                let bits = (data[off] as u16) | ((data[off + 1] as u16) << 8);
                hex_float16(bits)
            });
        }
        MILTensorData::LongInts(values) => {
            if let Some(&first) = values.first()
                && values.iter().all(|v| *v == first)
            {
                emit_uniform_tensor(shape, &first.to_string(), out);
                return;
            }
            let mut idx = 0usize;
            emit_nested_array(shape, 0, &mut idx, out, &|i| values[i].to_string());
        }
        MILTensorData::Doubles(values) => {
            if let Some(&first) = values.first()
                && values.iter().all(|v| v.to_bits() == first.to_bits())
            {
                emit_uniform_tensor(shape, &hex_float32(first as f32), out);
                return;
            }
            let mut idx = 0usize;
            emit_nested_array(shape, 0, &mut idx, out, &|i| hex_float32(values[i] as f32));
        }
    }
}

fn is_fp16_uniform(data: &[u8]) -> bool {
    if data.len() < 2 {
        return true;
    }
    let lo = data[0];
    let hi = data[1];
    let mut i = 2;
    while i + 1 < data.len() {
        if data[i] != lo || data[i + 1] != hi {
            return false;
        }
        i += 2;
    }
    true
}

/// Emit a nested-bracket tensor whose every leaf value is the same
/// pre-formatted string. Builds the innermost row once and replays it for
/// every outer-dimension slot — `O(product(outer dims))` instead of
/// `O(product(all dims))`.
fn emit_uniform_tensor(shape: &[usize], value_str: &str, out: &mut MILOutputSink) {
    if shape.is_empty() {
        out.write_str(value_str);
        return;
    }
    let inner_dim = *shape.last().unwrap_or(&0);
    let mut inner = String::with_capacity(inner_dim * (value_str.len() + 2) + 2);
    inner.push('[');
    for i in 0..inner_dim {
        if i > 0 {
            inner.push_str(", ");
        }
        inner.push_str(value_str);
    }
    inner.push(']');

    emit_uniform_nested(shape, 0, &inner, out);
}

fn emit_uniform_nested(shape: &[usize], depth: usize, inner: &str, out: &mut MILOutputSink) {
    if depth >= shape.len().saturating_sub(1) {
        out.write_str(inner);
        return;
    }
    out.write_str("[");
    let n = shape[depth];
    for i in 0..n {
        if i > 0 {
            out.write_str(", ");
        }
        emit_uniform_nested(shape, depth + 1, inner, out);
    }
    out.write_str("]");
}

/// Alloc-free nested-bracket emitter specialised for `Float32` arrays. Writes
/// each element via `hex_float32_bytes` into a 32-byte stack buffer, then
/// straight to the sink — no per-element `String` allocation.
fn emit_dense_float32(values: &[f32], shape: &[usize], out: &mut MILOutputSink) {
    if shape.is_empty() {
        let mut buf = [0u8; 32];
        let n = hex_float32_bytes(values.first().copied().unwrap_or(0.0), &mut buf);
        out.write_bytes(&buf[..n]);
        return;
    }
    let mut idx = 0usize;
    let mut buf = [0u8; 32];
    emit_dense_float32_rec(shape, 0, values, &mut buf, &mut idx, out);
}

fn emit_dense_float32_rec(
    shape: &[usize],
    depth: usize,
    values: &[f32],
    stack: &mut [u8; 32],
    index: &mut usize,
    out: &mut MILOutputSink,
) {
    if depth == shape.len() - 1 {
        out.write_str("[");
        let n = shape[depth];
        for i in 0..n {
            if i > 0 {
                out.write_str(", ");
            }
            let written = hex_float32_bytes(values[*index], stack);
            out.write_bytes(&stack[..written]);
            *index += 1;
        }
        out.write_str("]");
    } else {
        out.write_str("[");
        let n = shape[depth];
        for i in 0..n {
            if i > 0 {
                out.write_str(", ");
            }
            emit_dense_float32_rec(shape, depth + 1, values, stack, index, out);
        }
        out.write_str("]");
    }
}

fn emit_nested_array(
    shape: &[usize],
    depth: usize,
    index: &mut usize,
    out: &mut MILOutputSink,
    format: &dyn Fn(usize) -> String,
) {
    if depth == shape.len() - 1 {
        out.write_str("[");
        let n = shape[depth];
        for i in 0..n {
            if i > 0 {
                out.write_str(", ");
            }
            out.write_str(&format(*index));
            *index += 1;
        }
        out.write_str("]");
    } else if depth < shape.len() {
        out.write_str("[");
        let n = shape[depth];
        for i in 0..n {
            if i > 0 {
                out.write_str(", ");
            }
            emit_nested_array(shape, depth + 1, index, out, format);
        }
        out.write_str("]");
    }
}

fn opset_tag(opset: &str) -> String {
    match opset {
        "CoreML8" => "ios18".to_string(),
        "CoreML7" => "ios17".to_string(),
        "CoreML6" => "ios16".to_string(),
        "CoreML5" => "ios15".to_string(),
        _ => {
            if let Some(last) = opset.chars().last()
                && let Some(d) = last.to_digit(10)
            {
                return format!("ios{}", d + 10);
            }
            opset.to_lowercase()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_relocation_handles_attribute_and_immediate_binding_without_mutation() {
        let mut program = crate::decode(include_bytes!(
            "../tests/fixtures/flexible-weighted-legacy/input.mlmodel"
        ))
        .unwrap();
        let value = program.functions[0].1.block.operations[0].attributes[0]
            .1
            .clone();
        let mut binding_op = program.functions[0].1.block.operations[0].clone();
        binding_op.r#type = "identity".into();
        binding_op.outputs[0].name = "bound_asset".into();
        binding_op.attributes.clear();
        binding_op.inputs = vec![("x".into(), vec![MILBinding::Immediate(value)])];
        program.functions[0].1.block.operations.push(binding_op);
        let mut sink = MILOutputSink::in_memory(4096);
        emit_pipeline_child(&program, &mut sink);
        let emitted = String::from_utf8(sink.finalize().unwrap()).unwrap();
        assert_eq!(
            emitted
                .matches("@model_path/../weights/weights.bin")
                .count(),
            2
        );
        assert_eq!(emitted.matches("offset = uint64(64)").count(), 2);
        assert_eq!(
            crate::weights::references(&program)
                .unwrap()
                .into_keys()
                .collect::<Vec<_>>(),
            ["weights/weights.bin"]
        );
        assert!(!emit_to_string(&program).contains("@model_path/../"));
    }
}
