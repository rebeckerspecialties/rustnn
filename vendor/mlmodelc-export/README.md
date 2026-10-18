# mlmodelc-export

Compile a CoreML MLProgram protobuf into a runnable `.mlmodelc` bundle in pure
Rust, without invoking Apple's `coremlc`.

[![Crates.io](https://img.shields.io/crates/v/mlmodelc-export.svg)](https://crates.io/crates/mlmodelc-export)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

## Why this exists

Apple's `coremlc` is the tool that turns a `.mlmodel` (or `.mlpackage`)
protobuf into the directory layout that `MLModel(contentsOfURL:)` actually
loads at runtime — it lives inside Xcode's toolchain and is callable only
from a developer host on macOS. There is no equivalent on:

- **watchOS** — sandboxed, no compiler, only the loader.
- **iOS App Extensions** — same restriction.
- **iOS App Clips** — same restriction.
- **tvOS sandboxed contexts** — same restriction.
- **Linux/Windows CI** — no Apple toolchain at all.

This crate fills the gap. Feed it the protobuf bytes you'd otherwise hand to
`coremlc`, and it emits the same four files Apple's compiler produces:

```
example.mlmodelc/
├─ model.mil                 UTF-8 MIL text
├─ coremldata.bin            FunctionDescription / defaultFunctionName trailer
├─ metadata.json             I/O schema, op histogram, availability matrix
├─ analytics/
│  └─ coremldata.bin         minimal stub satisfying CoreML's analytics check
└─ weights/
   └─ weights.bin            optional, when the input has external weights
```

Golden tests compare `model.mil` (except build info) and `coremldata.bin`
byte-for-byte against captured `coremlc` output. `metadata.json` is compared
structurally, excluding generated class names and optimization statistics.
The exporter reports source operations, not Apple's optimized executable.

## Quick start

### Library

```rust
use mlmodelc_export::compile_to_bundle;

let mlmodel_bytes = std::fs::read("model.mlmodel")?;
let bundle = compile_to_bundle(&mlmodel_bytes, None)?;
bundle.write_to_dir("model.mlmodelc")?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

### Streaming variant (recommended for large models)

```rust
use mlmodelc_export::compile_to_dir;

let mlmodel_bytes = std::fs::read("model.mlmodel")?;
let stats = compile_to_dir(&mlmodel_bytes, None, "model.mlmodelc")?;
println!("emitted {} bytes of MIL text across {} ops", stats.output_bytes, stats.operation_count);
# Ok::<(), Box<dyn std::error::Error>>(())
```

`compile_to_dir` writes the MIL text directly to disk through a 256 KB flush
buffer — keeping resident memory bounded by the chunk size rather than the
full MIL text. Recommended on memory-constrained devices like 32-bit Apple
Watches where the jetsam limit is ~150 MB.

### CLI

```sh
cargo install mlmodelc-export
mlmodelc-export model.mlmodel ./model.mlmodelc
```

Useful for CI / build pipelines where you want a runnable bundle without
depending on a macOS host with Xcode installed.

### External weights

The CLI discovers every referenced blob relative to the source `.mlmodel`,
including the model inside a `.mlpackage`. It preserves filenames, offsets and
bytes: `weights/weight.bin`, `weights/weights.bin` and multiple/nested files are
supported. `--weights PATH` overrides a single referenced file; models with
multiple files must provide their named assets. Missing files, out-of-file
reference offsets and unsafe paths fail before bundle emission.

Library callers can keep using `compile_to_bundle` / `compile_to_dir` for one
blob. For multiple blobs, use `referenced_weight_paths` to discover the relative
paths and supply a `BTreeMap<String, Vec<u8>>` to
`compile_to_bundle_with_weight_files` / `compile_to_dir_with_weight_files`.
The streaming API borrows those buffers; only MIL emission is streamed, not
source weight loading. In-memory bundles retain the legacy `weights_bin` field
for `weights/weights.bin` and use `weight_files` for other names. Code constructing
`MlmodelcBundle` directly must initialize the additional map.

This validates asset presence and reference offsets, not the blob's internal
format. Source-package auto-discovery cannot follow symlinks outside the model
directory, and bundle writes reject existing destination symlinks. As before,
callers must not concurrently modify source/output directories during export;
filesystem write failures are not an atomic directory transaction.

### Pipeline precision boundaries

Flat CoreML Pipelines containing MLPrograms use the same compile APIs. Their
stages remain separate programs, preserving intermediate FP16 tensor storage
between an FP32-to-FP16 cast and its consumers. Wrapper and child descriptions
retain their input defaults, shape ranges and output types.

Export writes each distinct external asset once at the root. Validated child
references are relocated to that shared root, so ordinary app-directory copies
retain one asset without depending on hard links. Buffered bundles retain one
owned buffer per distinct asset.
Tests check exact cast rounding, both outputs of a weighted two-stage model,
copied-bundle asset counts and repeated dynamic resizing. Empty pipelines, nested
pipelines and non-MLProgram children produce explicit errors.

## On-device Validation

This crate originated as a Swift package (`MILTextCompiler`) inside the
`metal-info-app` watchOS validation harness for [rustnn]. We ported it to
Rust to align with rustnn's Rust-first architecture and so the crate could
be reused outside that harness. The full motivation is documented in
[rustnn/rustnn#110](https://github.com/rustnn/rustnn/issues/110).

When paired with **rustnn**'s WebNN-graph → CoreML MLProgram frontend, the
combination has been validated end-to-end on real **Apple Watch SE 2
hardware** (arm64_32, watchOS 11, ~150 MB jetsam ceiling) — a device
where Apple ships *no* CoreML model compiler at all. Every model below is
compiled at runtime on the watch itself, then loaded with
`MLModel(contentsOf:)` and exercised with real inputs:

| Workload | rustnn frontend output | Result on Watch SE 2 |
|---|---|---|
| **W3C WebNN WPT operator coverage** | per-op MLProgram fixtures | **44/44 ops pass** (100%) — abs, add, ceil, clamp, concat, conv2d, conv_transpose2d, div, elu, exp, expand, floor, hard_sigmoid, hard_swish, instance_normalization, leaky_relu, linear, log, matmul, max, min, mul, neg, pow, reduce_l1/l2/log_sum/log_sum_exp/max/mean/min/product/sum/sum_square, relu, reshape, sigmoid, slice, softmax, split, sqrt, sub, tanh, transpose |
| **LeNet MNIST classifier** | `conv2d → relu → averagePool2d → … → softmax` | **10/10 correct** at p ≥ 0.92 (compile 12 ms, load 305 ms, predict 29 ms total) |
| **Char-level transformer (Shakespeare)** | `layerNorm → matmul → gelu → matmul → softmax` | **5/5 top-3 hits** vs host reference (compile 12 ms, load 363 ms, predict 43 ms) |
| **MNIST autoencoder decoder** | latent → fully-connected → conv stack → output | **10/10 reconstructions, mean MSE 0.000000** vs host reference |

Peak resident memory across all four workloads stayed under **10 MB** — well
below the arm64_32 jetsam ceiling that originally motivated the streaming
emission path and the alloc-free hex-float byte formatter described below.

[rustnn]: https://github.com/rustnn/rustnn

## Testing methodology

The crate uses **golden-file directory comparison** against two committed
references per fixture:

| Reference | Source | Role |
|---|---|---|
| `expected-macos.mlmodelc/` | output of `xcrun coremlc compile` on macOS | **positive contract** — our output must match |
| `observed-watchos-broken.mlmodelc/` | output of Apple's arm64_32 watchOS stub | **negative contract** — our output must not match |

The watchOS stub is the same compiler binary but running with `arm64_32`
constraints — it returns silently after writing a truncated `coremldata.bin`
(missing the 16-byte loader trailer) and refusing to emit `model.mil` or
`metadata.json`. This is the failure mode the crate exists to avoid; pinning
it as a checked-in negative reference protects against future regressions.

See [`tests/fixtures/README.md`](tests/fixtures/README.md) for instructions
on adding new fixtures.

## Status

`v0.1` — works for the small-graph cases in our test fixtures. Round-trip
verified against `xcrun coremlc compile` (Xcode 26.4, `coremlc` 3520.4.1):

- `model.mil` byte-identical except for the `buildInfo` producer string
- `coremldata.bin` byte-identical (source ModelDescription and per-function trailers)
- `metadata.json` I/O schemas, shape constraints, function selection and
  producer-defined metadata are preserved

Ranged flexible shapes preserve unknown MIL extents as `?`, together with the
source default shapes and bounds. Literal zero remains a fixed empty extent;
this does not imply that every CoreML operation can predict with empty tensors.
Both single-function and multi-function descriptions are retained, including a
non-first default function and descriptions larger than 255 bytes. Additional
reference fixtures were captured with coremlc 3520.5.1 and 3600.25.1 (Xcode 27).

User-defined metadata survives in both `coremldata.bin` and `metadata.json`.
This includes producer annotations used to interpret normalized feature names;
the exporter preserves those names and does not invent its own encoding policy.
On macOS 15+, check exact predictions and runtime metadata with:

```sh
swift tests/runtime_identifiers.swift target/release/mlmodelc-export tests/fixtures/generate_identifiers.py
```

Append `--check-invalid` to also reproduce the loader rejection of an unescaped
`state` input on affected CoreML versions. CI requires the normalized positive
case, without requiring future runtimes to reject the negative control.

Enumerated input shapes, variable-rank tensors and variadic dimensions currently
return an explicit unsupported-format error instead of emitting an invalid model.
For callers constructing intermediate types directly, `MILType.shape` now holds
`MILDimension::{Constant, Unknown}`; `MILType::new` still accepts `Vec<usize>`.

On macOS 15+, exercise exact numerical results while growing and shrinking
inputs, including the selected default function:

```sh
cargo build --release
swift tests/runtime_shapes.swift target/release/mlmodelc-export tests/fixtures
```

This compiles the checked-in protobufs through the Rust exporter before loading
them with CoreML. It does not load the precompiled golden references. The 36
exact predictions include a legacy single-function model with external weights,
growing and shrinking its input on the same loaded model. Its small source
fixture and independent expected values need no model download. These macOS
runtime checks do not substitute for physical iOS/watchOS validation. Separate
exact checks cover FP16 subnormal constants cast to FP32 and int8 compressed
weights with typed dequantization parameters.
The runtime suite also generates single- and multiple-blob packages with
`weight.bin` and nested filenames, checking exact predictions through `1→4→2→1`
input resizing after exporting without `--weights` or manual file copying.

For large dense Float32 constants (>10⁵ elements) the streaming path uses an
allocation-free hex-float byte formatter (see
[`src/hex_float.rs`](src/hex_float.rs)), giving ~3-5× faster emission than
the naive `String`-allocating implementation. This was the critical
optimisation for getting jetsam-safe compile times on arm64_32 watch SoCs.

## Native control flow

Nested MIL `while_loop` and `cond` bodies retain their typed block arguments,
operations and return values. External weight discovery, bundle relocation and
metadata also visit nested blocks. This is required when packed weights are
expanded inside a bounded native loop; dropping a body is not a valid export.
The decoder rejects more than 64 nested blocks before emission.
The streaming named-weight API accepts `AsRef<[u8]>` owners, including memory
mappings, so the caller need not copy each mapped asset into a `Vec<u8>`.

Run the reproducible fixture and exact native runtime checks entirely offline
with cached Rust dependencies:

```sh
python3 -B tests/control-flow/generate.py --check
cargo test --offline --locked
cargo build --offline --locked --release
swift tests/runtime_control_flow.swift target/release/mlmodelc-export tests/control-flow
```

The fixture combines a loop, an inner branch and an Int8 BLOBFILE constant.
The runtime check uses only final outputs and exercises the requested compute
policies; policy selection does not establish GPU or Neural Engine placement.
Exporter support does not by itself establish WebNN numerical conformance or a
whole-model memory/performance bound. Callers supply the native MIL lowering.

## License

Dual licensed at your option:

- MIT License ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
