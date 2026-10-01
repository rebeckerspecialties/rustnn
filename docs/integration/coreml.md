# CoreML Backend

The `coreml` backend runs WebNN graphs through Apple CoreML on macOS. The converter
(`src/converters/coreml_mlprogram.rs`) emits an MLProgram, the MIL-based model format, and the
backend (`src/backends/coreml.rs`) compiles it once and keeps the compiled model for repeated
dispatch. The Objective-C bridge lives in `src/executors/coreml.rs` and `src/executors/coreml_shim.mm`.

## Requirements

- macOS with a CoreML version that accepts MLProgram models (macOS 13 or newer; on-device
  validation has also been done on iOS 18 and watchOS 11 builds of rustnn).
- The `coreml-runtime` Cargo feature. On Linux and Windows the feature compiles to shims whose
  calls always fail, so `cargo check --features coreml-runtime` works everywhere but the backend
  is never selected off macOS.
- Xcode command line tools for the Objective-C++ shim (`build.rs` compiles it with `cc`).

## Selection and devices

`BackendDevice::Coreml { device_type }` maps the WebNN hints to CoreML compute units:

| Hint | `DeviceType` | `MLComputeUnits` |
|---|---|---|
| not accelerated | `Cpu` | `cpuOnly` |
| accelerated, `Default` or `HighPerformance` | `Gpu` | `cpuAndGPU` |
| accelerated, `LowPower` | `Npu` | `cpuAndNeuralEngine` |

CoreML decides at run time which units execute which layers; the hint is a ceiling, not a
guarantee. The WPT harness pins `accelerated = false` (CPU) so that results are deterministic.

For the unified `MLContext` path, inspect `graph.rustnn_load_diagnostics()` after
building and match `LoadDiagnostics::Coreml`. It retains the requested policy, successful policy/route, and earlier errors even
when CPU-only or URL fallback succeeds. Route preparation errors have no compute-unit policy;
a deliberately selected URL route is not reported as a failed in-memory attempt. The same
summary is logged once at debug level under `rustnn::executors::coreml::load`, not per dispatch.

This distinguishes load fallback from CPU scheduling within an accelerator-enabled model.
It helps investigate sustained CPU thermal throttling without attributing that throttling to
an unobserved GPU/Neural Engine workload. Neither a successful policy nor this diagnostic
establishes placement, energy savings or prediction-time fallback; use separate device traces.

## How a graph runs

1. The converter lowers the graph to MIL programs. Rank-0 operands are promoted to `[1]` at the
   model boundary, comparison and logical results are produced as `uint8`, and reductions with
   empty axes, `resample2d` on arbitrary axis pairs and the stable `reduceLogSumExp` form are
   lowered explicitly because MIL has no direct equivalent.
2. Float16 weights remain a separate shared blob (`ConvertedGraph::weights_data`). MLProgram
   graphs compile locally from a URL: the in-memory route changes represented values on
   multiple tested CoreML stacks. This does not force CPU-only execution.
   Real precision boundaries are materialized as native Pipeline children where needed.
   The children expose live values only and reuse the original weight storage; the public
   WebNN graph, types and shapes are unchanged.
   Float16 PReLU and GEMM evaluate their complete comparison/affine computation in
   Float32 before the final Half conversion, avoiding flushed subnormals and
   premature dot-product overflow. Floating triangular copies use the same
   protected transport path; integer triangular remains unchanged.
3. `MLGraphBuilder::build` compiles the model with `MLModel` and keeps the compiled model;
   `dispatch` binds `MLMultiArray`s over the tensor storage and runs a prediction.
4. The legacy CLI path (`--convert coreml --run-coreml`) tries the compute-unit configurations in
   turn and reports each attempt; `--coreml-compiled-output <dir>` stores the compiled
   `.mlmodelc` for reuse.

## Tensor names

Input and output names remain independent in the RustNN API, including names containing
spaces, Unicode, punctuation, leading digits or MIL keywords. The converter records JSON
logical-to-physical bindings in creator-defined `rustnn.webnn.input_aliases` and
`rustnn.webnn.output_aliases` metadata. RustNN applies them automatically; standalone
consumers should apply them when binding and retrieving CoreML features.
Previous exports with only the `rustnn.coreml.name_encoding=hex-v1` marker remain
readable; unmarked third-party models keep literal feature names.
Proven equal copy outputs may share one physical result, avoiding CoreML's omission of
duplicate scalar/dynamic features. RustNN supplies each requested logical output tensor;
unequal computations and real dtype conversions remain separate. Returning an original
input or constant directly remains invalid under WebNN's build rules.
For produced identity/same-type-cast chains rooted in an input, the serialized
`rustnn.webnn.output_passthroughs` map identifies the logical input and original descriptor.
RustNN validates dtype, actual shape and byte length and snapshots that input before
prediction, supplying independent output copies even if CoreML omits or changes a copy
feature. Standalone consumers should honor these proven-copy bindings too; they must not
infer an input/output alias from matching names or values.

## Converter-private input views

Precision lowerings can request a compact native Half input through creator-defined
`rustnn.webnn.compact_input_views` metadata: a JSON array of `{source, view}` bindings
between declared physical input features. These are backend details, not additional
WebNN inputs; the original graph, logical dtype and shape remain unchanged.

RustNN binds each private rank-one feature using the source's actual element count,
including growing and shrinking bounded dimensions. Contiguous source arrays share
their pointer with a native view whose deallocator retains the source owner. Padded
or strided arrays instead receive an exact raw-Half copy, preserving subnormals,
signed zero and NaN bits without a Float32 round trip. The original feature remains
bound for actual-shape queries and other consumers. Metadata, native dtype, shape
constraints and storage layout are checked before prediction; standalone consumers
of these exports must supply the same private bindings.

## Testing

```bash
make test-wpt-coreml              # full WPT suite, expected failures are non-fatal
make test-wpt-coreml-report       # same, plus the JSON report
make wpt-sync-coreml              # regenerate tests/wpt_conformance/coreml_expected_failures.txt
make test-coreml                  # ordinary native unit and integration tests
```

CoreML has no PASS snapshots; failing trials are listed in
`tests/wpt_conformance/coreml_expected_failures.txt` and must be executed on every run. CI runs
the suite on macOS for every pull request.

## Known limits

The remaining expected failures come from the platform rather than from missing lowerings:

- Integer operations are computed in float32, so values near the int32 and all int64 extremes
  lose precision (`abs`, `clamp`, `relu`, `neg` on wide integer types).
- Tensors of rank 6 and above are rejected.
- Pooling has no dilation parameter; `maxPool2d` with `ceil` rounding and all-padding windows
  differs at the border.
- `pad` in `edge` and `reflection` mode is limited to two dimensions; `gatherND` above rank 5 and
  out-of-bounds gather and scatter indices follow CoreML's clamping rather than the WebNN text.
- The `shape` extension operation is not lowered.

See the [WPT conformance dashboard](https://rustnn.github.io/rustnn/wpt-conformance/) for the
current per-operation status.
