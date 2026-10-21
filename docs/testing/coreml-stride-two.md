# Static stride-two CoreML lowering

TinyStories-8M's precision-preserving lowering produced many even/odd slices.
On A12 and A13, the unrevised CPU+GPU model exited during native loading; replacing
those slices enabled complete, accurate conversations with traced GPU execution.
The change is a general MIL layout rewrite, not a model-name or device-name rule.

For an even dimension `N`, a full-axis parity slice is equivalent to:

```text
[..., N, ...] -> reshape [..., N/2, 2, ...]
             -> unit-stride slice [..., N/2, 1, ...] at parity 0 or 1
             -> reshape [..., N/2, ...]
```

Only static FP16/FP32 tensors of rank 1-4, a single stride-two axis, full bounds,
and absent/false masks qualify. The temporary rank is at most five. Dynamic,
partial, odd-sized, masked, integer and nested-block cases keep their original
lowering. Arithmetic, casts, weights, output names and native pipeline boundaries
are unchanged. The pass runs after arithmetic lowering, immediately before model
serialization, including inside existing native pipeline children.

The pass is `rewrite_model()` in `src/converters/coreml_mlprogram/stride_two.rs`.
The CoreML converter's `GraphConverter::convert()` calls it once the MLProgram is
assembled, immediately before the model is serialized. When combining this with
[precision-boundary lowering (#279)](https://github.com/rustnn/rustnn/pull/279),
keep that call at the end of `convert()`, after #279's precision boundaries are
materialized, rather than in the program-building helper that #279 factors out of
`convert()`. This ensures slices introduced by later arithmetic lowering are
rewritten, without changing the resulting pipeline partitioning.

## Device evidence (13 October 2026)

These are the preserved integration runs of the same floating-point rewrite,
not a fresh phone run of this standalone branch. The extracted pass was replayed
against the sealed full TinyStories graph: every eligible slice was replaced and
every resulting operation, binding, type, output and attribute matched the sealed
candidate in each of its native children. The standalone pass leaves experimental
partition metadata untouched rather than depending on it.

Each admitted run held one model through five independent three-prompt
conversations. Every serving logit vector and every returned-KV boundary word
matched the qualified reference/hash replay. The initial gate requires finite
values, matching token/argmax and `abs(actual-reference) <= 2e-5 + 1e-4*abs(reference)`;
this application gate is separate from the operator-level WebNN tolerance tests,
whose current results are on the
[WPT conformance dashboard](https://rustnn.github.io/rustnn/wpt-conformance/).

| Device / OS | Requested policy | Inclusive tokens/s | Median TTFT (s) | Peak / held charged memory (MB) |
| --- | --- | ---: | ---: | ---: |
| XS Max, A12 / iOS 18.7.10 | CPU | 2.556 | 3.112 | 201.5 / 39.1 |
| XS Max, A12 / iOS 18.7.10 | CPU+GPU | 2.261 | 3.393 | 638.3 / 545.3 |
| XS Max, A12 / iOS 18.7.10 | All | 2.266 | 3.335 | 777.1 / 680.3 |
| SE2, A13 / iOS 26.7.1 | CPU | 3.712 | 2.248 | 202.9 / 44.6 |
| SE2, A13 / iOS 26.7.1 | CPU+GPU | 3.699 | 2.267 | 782.6 / 684.3 |
| SE2, A13 / iOS 26.7.1 | All | 3.692 | 2.267 | 805.1 / 714.8 |

Goodput includes scheduled recovery: at most 20 seconds of native work followed
by at least 30 seconds of rest, plus thermal/lifecycle waits. Startup and initial
qualification are excluded. Peak is lifetime charged footprint, not RSS or a
resettable reply-only peak. GPU does not outperform CPU here: the benefit is
making the accurate native GPU path usable, not claiming a speed/energy win.

Separate placement traces joined model prediction spans to GPU hardware buffers
on both phones. An XS All trace also observed GPU work, with no ANE activity; an
exact-convolution positive control exercised the ANE tracing channel. CPU+ANE is
not qualified for this full model: the XS run exited during loading, and SE2 failed
the logit gate at step 163. No power/thermal-spreading claim is made from these
timing runs.

SmolLM2's larger native-memory investigation is separate and remains tracked in
[fork issue #38](https://github.com/rebeckerspecialties/rustnn/issues/38).

## Reproduction and provenance

`make test-coreml-stride-two` checks generic ranks, axes, both parities, opaque
floating storage patterns, naming, no-op guards, chained consumers and native
pipeline traversal. On macOS it also exercises the exported model across all
four compute permissions, requiring exact outputs and rejecting GPU load fallback.
The rewrite changes only the MIL layout of an already supported `slice`; the
[backend operator support report](../development/backend-operator-support.md)
remains the reference for which operations the CoreML converter lowers.

Preserved evidence identifiers (SHA-256):

- Phone runtime rewrite source: `4ab6ef18b949bc2ca910fa4b79169b72f9c6d66b66f205c8c3ca786019da47a5`
- Conversation build record: `f91e8eaca3d7913b863a9e5a073955cfc1f36ac73da30dd472b1c9424d3cdab2`
- XS CPU+GPU run `xs-tiny-gpu-clean-v60b`, audit: `4e5f4b7d1b04243a64a6d741434c833b56822e9d7381ef3f9ca1409ca484babe`
- SE2 CPU+GPU run `se2-tiny-gpu-clean-v60a`, audit: `963a2d35cb7e14b29bb51ff4f3375e7b205f72e7dba77fc3ab9d7a87161e71d9`
- Unrewritten full graph: `f6fb8f8543c8d4ad44c7a7a0a7a3ed708786aeb47180e65187097dc9b3440886`
