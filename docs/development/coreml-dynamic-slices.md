# Runtime-sized CoreML slices

With `dynamic-inputs`, the existing `Slice.sizes` representation accepts an
explicit named dimension as well as a fixed integer. The CoreML lowering builds
the size vector from runtime input shapes and passes it to MIL `slice_by_size`.
It does not substitute the dimension's allocation maximum.

For example, an input `[1, 3, sequence<=4, 64]` can select the second half of each
head using starts `[0, 0, 0, 32]` and sizes
`[1, 3, {name: "sequence", maxSize: 4}, 32]`. Its output is
`[1, 3, sequence, 32]`, including when the same compiled model is dispatched with
different sequence lengths.

The supported subset has constant starts, unit strides, and nonempty dimension
names bound to graph input shapes. Names are opaque equality labels, not
arithmetic expressions. Arbitrary shape-tensor operands and derived dimensions
remain part of the [dynamic-shape work](https://github.com/rustnn/rustnn/pull/225),
following the [WebNN proposal](https://github.com/webmachinelearning/webnn/pull/945).
Unsupported relationships fail explicitly rather than using a guessed size.
LiteRT, TensorRT, and CANN reject named slice sizes until they implement runtime
lowering; their numeric slice paths are unchanged. ONNX retains its existing
runtime-size path and rejects overflowing end bounds explicitly.

The CoreML context checks active slice bounds before prediction, including fixed
slices on dynamic inputs. A literal size `8` remains eight: an active input of
four elements does not turn it into a four-element slice. Runtime checks retain
only shape bindings and slice metadata, not model weights or tensor values.
Starts, size arithmetic, and MIL int32 shape boundaries use checked conversions.
Graph interchange preserves `starts`, `sizes`, and `strides`, including the
distinction between numeric and named sizes.

This does not repair an export that already replaced a shape dependency with a
literal maximum. In particular, SmolLM's existing literal `4096` slice still needs
the original ONNX shape dependency preserved by its exporter. See the
[SmolLM tracker](https://github.com/rustnn/rustnn/issues/222).

The lowering uses the existing MIL `slice_by_size` operation, without an opset
increase. Zero-extent prediction support is a separate limitation. Tests cover
repeated nonzero shapes, exact selected values, fixed-size counterexamples,
invalid active bounds, unknown labels, strides, and integer overflow. These are
experimental RustNN regressions, not claims of conformance to an adopted dynamic
WebNN API; upstream WPT currently exercises the static slice API.
