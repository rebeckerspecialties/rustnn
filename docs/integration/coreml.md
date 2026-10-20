# CoreML Backend

The `coreml` backend runs WebNN graphs through Apple CoreML on Apple targets. The converter
(`src/converters/coreml_mlprogram.rs`) emits an MLProgram, the MIL-based model format, and the
backend (`src/backends/coreml.rs`) compiles it once and keeps the compiled model for repeated
dispatch. The Objective-C bridge lives in `src/executors/coreml.rs` and `src/executors/coreml_shim.mm`.

## Requirements

- A CoreML version that accepts MLProgram models (macOS 13 or newer). The native Rust
  backend also supports iOS; mobile validation uses iOS 18 as its deployment target.
- The vendored `objc` message-dispatch patch selects the Apple ABI on tvOS,
  watchOS and visionOS. Keep this prerequisite when updating the dependency.
  watchOS additionally requires `coreml-export` for local bundle compilation.
- The `coreml-runtime` Cargo feature. On Linux and Windows the feature compiles to shims whose
  calls always fail, so `cargo check --features coreml-runtime` works everywhere but the backend
  is never selected on those platforms.
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

1. The converter lowers the graph to a MIL program. Rank-0 operands are promoted to `[1]` at the
   model boundary, comparison and logical results are produced as `uint8`, and reductions with
   empty axes, `resample2d` on arbitrary axis pairs and the stable `reduceLogSumExp` form are
   lowered explicitly because MIL has no direct equivalent.
2. Float16 weights are written as a separate weights blob (`ConvertedGraph::weights_data`), and
   the model is packaged as an in-memory asset together with that blob. Graphs for which the
   in-memory compiler is known to misbehave (`gather` with a rank-0 index) are written to a
   temporary directory and compiled from its URL instead (`supports_in_memory_asset`).
3. `MLGraphBuilder::build` compiles the model with `MLModel` and keeps the compiled model;
   `dispatch` binds `MLMultiArray`s over the tensor storage and runs a prediction.
4. The legacy CLI path (`--convert coreml --run-coreml`) tries the compute-unit configurations in
   turn and reports each attempt; `--coreml-compiled-output <dir>` stores the compiled
   `.mlmodelc` for reuse.

## Reusing tensor storage

With `RustNNOptions::coreml.reuse_tensor_storage` enabled, CoreML contexts keep float32,
float16 and int32 tensors in owned storage with
retained `MLMultiArray` views. Dispatch binds compatible input arrays directly. An output can
become the next graph's input without `read_tensor`/`write_tensor` or a temporary input array;
for KV caches, alternate two distinct tensor sets. A dispatch cannot bind one tensor as both
input and output.

Buffers of at least 16 KiB are page-aligned for CoreML's output-backing performance
recommendation; smaller scalar and index buffers use 16-byte alignment.

When supported, `outputBackings` proposes the destination array to CoreML. Only a returned
array with the same object identity counts as accepted. Backings require a fully static
graph (including intermediate operands) and a fixed output feature. A fixed-size output
alone is not sufficient when the graph has dynamic dimensions. Ineligible outputs,
declined backings, strided results and dtype conversions are copied
into the destination's owned storage. Returned arrays are never adopted as tensor storage,
because CoreML may alias them to an input or another output. This is not a guarantee of zero
copies inside CoreML, GPU or Neural Engine drivers.

Both host and retained-storage paths use the same checked numeric conversion and array
layout rules. Equal element widths do not imply equal types (for example int32 and
float32). Same-type copies preserve integer bits and float16 storage widths; overlapping
strided views are gathered before writing the destination.

Numeric conversions into float16 round directly from the source value using integer
round-to-nearest, ties-to-even, retaining all discarded bits. This preserves subnormals
and signed zero independently of hardware half-conversion support. Matching storage
types are copied bit-for-bit, including NaN payloads; CoreML's internal arithmetic policy
is unchanged.

`RustNNOptions::coreml` controls this experimental path. `reuse_tensor_storage` defaults to
`false`, preserving the byte-buffer reference implementation until an application measures
a benefit on its workload. `output_backings` defaults to `true` when reuse is enabled and
can be disabled independently to measure persistent input storage alone. The
WebNN API and compute-unit selection are unchanged. Other data types retain host storage and
the existing conversion path. Zero-extent prediction remains unsupported.

With `dynamic-inputs`, reserve the maximum capacity before a decode loop and resize the
active shape between dispatches. Shape changes rebuild the array view, but do not allocate
another data buffer while they fit the reserved capacity. Reserve replaces storage with
zeroed bytes; growth during resize preserves the existing prefix.

`MLContext::rustnn_backend_statistics()` returns `Some(BackendStatistics::Coreml(...))`
for this backend; backends that do not report statistics return `None`.
The CoreML variant reports cumulative host reads/writes, native allocations, direct input
bindings, proposed/accepted backings and logical copied payloads.
These count rustnn-side work, not total memory traffic or internal CoreML allocations. The
reported compute-unit policy includes load fallback but does not measure accelerator
placement. Take counter differences around the decode loop to exclude prefill and setup.

## Testing

```bash
make test-wpt-coreml              # full WPT suite, expected failures are non-fatal
make test-wpt-coreml-report       # same, plus the JSON report
make wpt-sync-coreml              # regenerate tests/wpt_conformance/coreml_expected_failures.txt
make test-coreml                  # unit and integration tests
WPT_COREML_TENSOR_MODE=persistent make test-wpt-coreml
WPT_COREML_TENSOR_MODE=backings make test-wpt-coreml
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

## Local compilation on Apple devices

Build with `coreml-runtime,coreml-export` to parse a local WebNN source bundle,
lower it in RustNN and emit its compiled CoreML bundle in the application process.
This route uses the immutable Git-pinned `mlmodelc-export` prerequisite and loads
the result through CoreML. The dependency can be replaced by a released crate once nested-block support is available. It needs no
Mac helper, Python, compiler subprocess or network connection after deployment;
Cargo dependencies and the application binary can be prepared on a development
machine. watchOS requires this route because its public CoreML API does not
compile source models. The same route is available on iOS, tvOS, visionOS and
macOS. Cross-compilation alone does not qualify a physical device.

Load `.webnn` or `.json` and its local weights with `load_graph_from_path`, then
pass the result to `MLContext::rustnn_build_graph`. Load diagnostics report
`CoremlLoadRoute::RustExport`. Export or load errors are returned to the caller;
this route does not fall back to a remote compiler or host inference stage.
Enable `RustNNOptions.coreml.reuse_tensor_storage` to carry compatible tensors
between dispatches without Rust materializing their bytes. Read only the outputs
your application actually needs. CoreML may still copy storage internally.

The Rust compiler route changes bundle emission, not the numerical lowering.
It does not by itself qualify the experimental SmolLM2 corrections, guarantee
GPU/Neural Engine placement, or establish a Watch memory bound. The source loader
currently owns resolved weight bytes, so parsing and lowering peak memory must
also be measured on the device. Existing operator tolerances and model accuracy
gates remain separate validation requirements.

### Explicit Watch packed-weight scheduling

`coreml_packed::bound_packed_matmuls` preserves single-consumer, unit-scale Int8
leaves and static rank-three K=32 contractions, expanding one output-column tile
inside MIL. Unsupported scale/type/shape/transpose/fanout cases stay unchanged.
`ConvertedGraph::bound_coreml_packed_matmuls` retains external weight bytes.
The default-disabled `RustNNOptions.coreml.packed_weight_expansion_bytes` enables
this scheduling only on watchOS; other platforms return a graph-build error.
Changed functions require CoreML8 (`slice_update`); the logical tile cap does not
bound physical memory or qualify arithmetic. The direct utility supports host
qualification of generated graphs. No source numerical profile is enabled here.

### File-backed source weights

`CoremlMlProgramConverter::convert_source_file_backed` consumes local WebNN
constants one payload at a time into `CoremlWeightStorage` scratch. The resolver
is pinned to its reviewed consuming API. The portable loader keeps its contract.
`coreml_assets::compact_file_backed` validates all recursive blob references before
relocating live spans and externalizing large immediate values; it does not
quantize weights or change arithmetic. Borrowed conversion remains available.
Peak setup memory still includes one source tensor and native compiler workspace.
A scratch file and a smaller weight blob do not imply a process-memory bound.

### Native feature-provider sessions

`CoremlNativeSession` is an opt-in Apple provider API alongside the portable
WebNN tensor interface. Native pipeline children forward retained `MLFeatureValue`
objects through versioned slots and release intermediates after their last use.
Repeated feature names do not overwrite retained earlier versions. No Rust
arithmetic closure or intermediate readback is used; CoreML may still copy
internally. Explicit `CoremlComputeUnits` permissions fail without retrying a
CPU-only policy and do not prove actual placement. Setup-only load callbacks
never inspect runtime tensors. Read terminal outputs only as the application
requires. File-backed compiler storage is dropped before native preparation.

### Experimental native numerical profile

`coreml_numerics::apply` emits native MIL guarded softmax, sigmoid tails, scalar
sqrt/mean residuals, division and softplus corrections with native fallbacks and
ReLU rounding barriers. It is never enabled by the portable converter by default.
No WPT budget, expected-failure list, or full-model reference is changed.
Operator accuracy and carried-model accuracy are distinct gates. The preserved
A profile still fails one TinyStories SE2 logit at step 163; later B/C softmax
and XS boundary experiments are excluded. Extreme-range/policy and three-model
physical qualification remain required before deployment recommendation.

### Experimental block-local contractions

`coreml_projection::apply_file_backed` uses width-32 block-local aligned pieces
for admitted constant projections and runtime-weight attention, retaining
ordinary native paths outside admission. Static weight transformation happens
during compilation; inference remains native. Watch profile affine leaves and
non-Watch CPU constant compression are explicit options, not a new default
backend policy. This does not resize KV context or relax contraction/model
gates. Cross-model extreme-range and accelerated physical qualification remain
open; no GPU/ANE placement or universal goodput/memory claim is made.

### Native partition and Watch producer placement

`coreml_partition::split` cuts static small frontiers while retaining exact
interfaces and constant references. Its weight budget includes constexpr
attributes but is a scheduling heuristic rather than physical residency.
`coreml_local_producers::clone_for_packed_children` can copy bounded pure
producer closures into recorded packed children, preserving operations and
constants, passthrough descriptors, retained outputs and lexical captures.
Ambiguous or over-budget candidates are skipped transactionally. Production
wiring is Watch-only; the failed non-Watch cloning experiments are excluded.
Partitioning never executes host arithmetic or reads an intermediate tensor.

### Experimental offline source composition

`CoremlNativeSession::compile_webnn_source_with_progress(path, policy)` parses
local WebNN text/JSON and weight sidecars, lowers, applies the opt-in numerical
and projection profiles, compacts live blobs, partitions, exports and loads
native children in the deployed process. It uses no network, Mac helper or
compiler subprocess. Watch affine/bounded expansion and producer placement are
compile-time watchOS choices. The source graph declares its context; the compiler
does not shorten it. `examples/coreml_source.rs` demonstrates setup.

`compile_webnn_source_cached_with_progress(path, policy, cache_dir, key)` adds a
persistent compiled-bundle cache owned by the application. `cache_dir/<key>/`
holds the exported bundle, the native child routing plan and a completion
marker; a complete entry is loaded without reading the source (`cache/hit`
notification), otherwise the source is lowered as above, exported into a staging
directory beside the entry and published with one atomic rename (`cache/miss`).
An entry that fails to load is removed and rebuilt once. The key is the caller's
identity for the source bundle, the native library and the policy, all of which
change the exported bundle; the cache never evicts other keys, so the
application removes entries whose source bundle is gone. The Rustdoc of the
method is the contract.
The [qualification snapshot](../testing/native-source-qualification-v124.json)
binds historical dirty-source A builds, not fresh split-head device validation.
TinyStories SE2 accuracy, Mamba XS load/returned-state qualification, Watch SE2
and accelerated execution remain open. Smol response quality/context and token
cap issues remain release blockers. Physical final-head qualification must
precede performance claims; budgets and frozen references stay unchanged.

See the [native source-port handoff](../testing/native-coreml-source-handoff.md)
for the draft branch sequence, immutable prerequisites and outstanding gates.
