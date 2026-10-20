# Native CoreML source-port handoff

This draft series separates local compilation, storage, native forwarding and
opt-in numerical policy. The public follow-up plan is
[issue #38](https://github.com/rebeckerspecialties/rustnn/issues/38#issuecomment-6281305443).
Review each PR against its immediate predecessor. Retarget/rebase dependent PRs
as prerequisites merge; the final source branch contains the assembled series.
No mirror branch, WPT baseline or stride-two PR #56 is part of these changes.

## Draft branches

| Review scope | Branch | Draft PR |
| --- | --- | --- |
| Compile CoreML bundles locally on Apple devices | `feat/coreml-device-local-export` | [#59](https://github.com/rebeckerspecialties/rustnn/pull/59) |
| Bound native packed-weight column expansion on watchOS | `feat/coreml-watch-packed-expansion` | [#60](https://github.com/rebeckerspecialties/rustnn/pull/60) |
| Consume WebNN source weights into file-backed CoreML storage | `feat/coreml-file-backed-source-weights` | [#61](https://github.com/rebeckerspecialties/rustnn/pull/61) |
| Forward native pipeline features with versioned last-use release | `feat/coreml-native-feature-forwarding` | [#62](https://github.com/rebeckerspecialties/rustnn/pull/62) |
| Add an opt-in native numerical correction profile | `feat/coreml-native-numerical-profile` | [#63](https://github.com/rebeckerspecialties/rustnn/pull/63) |
| Lower admitted projections and attention with block-local native pieces | `feat/coreml-block-local-projections` | [#64](https://github.com/rebeckerspecialties/rustnn/pull/64) |
| Partition native children and bound Watch producer placement | `feat/coreml-native-partition-placement` | [#65](https://github.com/rebeckerspecialties/rustnn/pull/65) |
| Compose offline WebNN source compilation into native sessions | `feat/coreml-webnn-source-sessions` | [#66](https://github.com/rebeckerspecialties/rustnn/pull/66) |

Dependency prerequisites are [mlmodelc-export #9](https://github.com/rebeckerspecialties/mlmodelc-export/pull/9)
and [webnn-graph #25](https://github.com/rustnn/webnn-graph/pull/25).
Cargo pins immutable public commits, with no dependency on a local experiment
checkout. The vendored objc package records its single Apple ABI patch and license.

## Clean-checkout validation

Check out `feat/coreml-webnn-source-sessions` from the RustNN fork, then use:

```bash
make test
make test-coreml COREML_FEATURES=coreml-runtime,coreml-export
make docs-api DOCS_API_FEATURES=coreml-runtime,coreml-export
make ci-docs
cargo run --features coreml-runtime,coreml-export --example coreml_source -- /path/to/model.webnn cpu
```

Development dependency fetching is permitted. Deployed applications use local
WebNN text/JSON and weight sidecars and perform parsing, structural checks,
lowering, bundle emission and native prediction on the edge device. The example
checks setup only. A load is not a prediction or accuracy qualification.
Native providers route intermediate `MLFeatureValue` objects without Rust
inference arithmetic or tensor readback; CoreML may still make internal copies.
Explicit compute permissions do not establish observed accelerator placement.

Each split head has a CoreML runtime/export library check; the composition head
also checks all targets. Host tests and documentation results are recorded with
exact pushed heads in the handoff receipts. Hosted CI runs the Rust, WPT and
RustNNPT workflows for pull requests whose base is `main` or a `feat/coreml-*`
branch, so stacked drafts get the same matrix as the foundation PR. Do not infer
a passing hosted matrix from the absence of checks on older runs.

On a Mac where a toolchain `bin` directory precedes `~/.cargo/bin` on `PATH`,
`rust-toolchain.toml` is bypassed and `cargo` reports that the crate requires
rustc 1.97; run through `rustup run 1.97.0 cargo ...` or fix `PATH`.

## Accuracy and release gates

The numerical source is production profile A, with the same admission/fallback
logic, reference packets and WPT allowances. No expected-failure list or gate is
relaxed. Later softmax B/C, diagnostic fixed-boundary E131 and generic-boundary
E133 work remains separate experimental evidence and is not enabled here.
The historical [qualification snapshot](native-source-qualification-v124.json)
binds its recorded source/artifact builds, not fresh physical split-head tests.

Before readiness:

- Resolve the preserved TinyStories SE2 step-163 full-model failure under the
  original gate; argmax agreement does not turn it into a pass.
- Complete Mamba XS load attribution, first prediction and original returned-state
  qualification. A load-only control is insufficient.
- Rerun final-head Smol, TinyStories and Mamba on target devices, including
  Watch SE2 arm64_32; measure lifecycle memory, cold/warm TTFT, sustained
  multi-turn goodput and thermals after accuracy passes.
- Preserve the earlier full-context model control while investigating Smol reply
  quality. Artificial reply caps explain truncation, not all odd or repeated
  answers. The compiler does not silently change context length.
- Qualify accelerated execution separately with exact-process placement traces
  and original accuracy gates; CPU receipts do not qualify GPU or ANE.
- Demonstrate the deployed app running source setup and multiple turns offline.
  Cross-compilation and Mac-side test execution do not prove that device gate.

## WPT follow-up queue

Keep new browser conformance work separate from this native-port series. Candidate
missing domains include sigmoid negative tails under its existing allowance,
finite wide-gap/large-offset softmax rows under the upstream axis budget,
scalar sqrt/mean boundaries only when they violate the public comparator,
signed-byte constant gather/cast/scaling paths, and independent constant matmul
branches with different output widths and ordinary per-output contraction budgets.
Each needs browser-positive and deliberately faulty/fixed controls, not only host
helpers or native compiler evidence. Private MIL representation, memory limits,
placement, model-logit gates and exact-bit tests are not public WebNN contracts.

Do not duplicate open WPT [#63155](https://github.com/web-platform-tests/wpt/pull/63155),
[#63661](https://github.com/web-platform-tests/wpt/pull/63661),
[#63662](https://github.com/web-platform-tests/wpt/pull/63662), or
[#63663](https://github.com/web-platform-tests/wpt/pull/63663).
Accumulator and composed-budget questions belong in WebNN
[#948](https://github.com/webmachinelearning/webnn/issues/948) and
[#950](https://github.com/webmachinelearning/webnn/issues/950) until agreed.
