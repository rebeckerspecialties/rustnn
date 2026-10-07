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
  is never selected on non-Apple targets.
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
3. `MLGraphBuilder::build` prepares the model for repeated dispatch. A standalone native
   model retains its `MLModel`. A precision Pipeline validates child features and their
   dependencies, compiles children lazily, and retains at most one loaded child model.
   Returned typed arrays remain live until their last consumer; compiled URLs are reused
   on later predictions and removed when the graph is dropped.
4. The legacy CLI path (`--convert coreml --run-coreml`) tries the compute-unit configurations in
   turn and reports each attempt; `--coreml-compiled-output <dir>` stores the compiled
   `.mlmodelc` for reuse.

Host-side numeric conversions into float16 use direct round-to-nearest, ties-to-even
from the source value, preserving subnormals and signed zero independently of hardware
half-conversion support. Matching storage types are copied bit-for-bit, including NaN
payloads. This boundary conversion does not change CoreML's internal arithmetic policy.

## Bounded precision pipelines

The executor reads original Pipeline child protobuf bytes without reconstructing unknown
fields. It checks unique feature names, dependency order, native types, declared bounds
and actual shapes. Output aliases and original input/constant copy proofs remain intact.
An immutable, graph-owned weight mapping is shared between children; each derived native
child contains only its referenced weight entries. Unsupported future weight layouts
retain the original source route with diagnostics; malformed references are rejected.

A complete, source-proven Float32-to-Half or Half-to-Float32 Cast child uses integer-bit
conversion rather than a native cast that may flush subnormals or erase a rounding
boundary. The proof checks the entire known-wire program and its feature descriptors,
not a node name or observed value. Other children still execute through CoreML under the
requested compute-unit permissions. A host-only graph reports the `TypedHost` route and
`NOT_APPLICABLE` loaded compute units, not an accelerator policy or fallback.

Run `make test-coreml-pipeline` for source-validation, weight-repacking, exhaustive Half
encoding/midpoint and retained grow/shrink dispatch regressions. Bounded loading is a
correctness/resource-lifetime mechanism; persistent tensor reuse is a separate path.
Apple's source-model compilation and specification-data asset APIs are unavailable on
watchOS. This executor rejects those routes explicitly rather than invoking unavailable
selectors; offline compiled-child loading still requires separate integration.

## Source-proven Float32 matrix products

Qualified static Float32 matrix children use an exact host dot product with one
round-to-nearest-even result per output. A bounded Float64 interval certifies a unique
Float32 result; ambiguous intervals, nonfinite inputs or unsuitable host rounding use
an exact signed-integer accumulator instead. Scratch is capped at 64 KiB. Original
Float32 constant blobs and lossless views are borrowed, not transposed into a second
learned-weight tensor; runtime operands use validated native strides.

The converter marks only static, source-proven matrix closures. Dynamic extents,
post-result layout adapters and unsupported constant arithmetic retain their native
route. A marked source with corrupted descriptors, flags, views or weight spans is
rejected rather than silently replacing the exact route with native arithmetic. The
backend-independent `GraphInfo`, public types and original payloads are unchanged.

`make test-coreml-matmul` covers signed cancellation, transpose/broadcast mappings,
bounded scratch, rejection controls and 256 predictions feeding each actual output
back as the next input. Its closed-form dyadic state oracle is independent of the
matrix implementation. This stronger implementation-fidelity contract is distinct
from WebNN's per-operation allowances: WPT deliberately excludes catastrophic
cancellation ([WPT #38679](https://github.com/web-platform-tests/wpt/pull/38679)), while
accumulator precision and composed budgets are discussed in
[#948](https://github.com/webmachinelearning/webnn/issues/948) and
[#950](https://github.com/webmachinelearning/webnn/issues/950).

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
For produced copy chains rooted in an input, the serialized
`rustnn.webnn.output_passthroughs` map identifies the logical input and original descriptor.
RustNN validates dtype, actual shape and byte length and snapshots that input before
prediction, supplying independent output copies even if CoreML omits or changes a copy
feature. Standalone consumers should honor these proven-copy bindings too; they must not
infer an input/output alias from matching names or values.
Bounded copies accept each actual shape within the declared bound, including repeated
grow/shrink predictions. `int8` and `uint8` identity and same-type cast kernels use exact
`int32` temporaries; graph descriptors and stored bytes retain their original dtype.
Signed constants used by these copies or casts to `int32`/`float32` retain their raw byte
payload under a private unsigned interpretation, then recover the signed values in MIL.
Copy proofs also cover same-shape reshape, identity transpose and static full-span
unit-stride slice. Original constant copies use version-1
`rustnn.webnn.output_constant_copies` metadata, with one original raw base64 payload per
output-reachable source and independent output bindings. Consumers must validate the
descriptor and encoded/decoded byte lengths. This adds approximately four encoded bytes
per three source bytes, once per referenced constant, not once per output or for all weights.
The native graph still runs; these proofs do not replace arithmetic-derived results.

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
