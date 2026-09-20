# Native CoreML int32 precision isolation

These are portable, uncompiled MLProgram sources derived from the triangular
regression. Run `make -f diagnostics/Makefile probe` on macOS. Each model is
compiled on the executing machine. The Swift harness bypasses RustNN, verifies
typed input readback, and reads actual int32 output buffers using their strides.
The report records exact values, dtype, shape, and whether CoreML used a supplied
output backing. Diagnostic jobs report numerical failures rather than treating
successful prediction as conformance.

## Observed results

| CPU-only URL-compiled model | M4, macOS 27.0 (26A428) | Hosted arm64, macOS 26.6.2 (25G83) |
| --- | --- | --- |
| Identity / native band / select | Large integers rounded | Large integers rounded |
| Band plus subtraction (PR #235) | Exact tested values | Large integers rounded |
| Integer multiply by a 0/1 mask | Exact tested values | Large integers rounded |
| Gather-based mask | Exact tested values | Large integers rounded |
| `16777217 - 16777216` | `1` | `0` |

Inputs read back exactly on both systems. Results are declared and returned as
int32, with shape `[3,3]` and strides `[3,1]`. Explicit int32 `outputBackings`
are used (pointer identity verified) but do not change the results. Neither
the fast-prediction hint nor switching the subtraction models among CoreML5,
CoreML6, CoreML7, CoreML8, and CoreML9 changes this distinction.

The small subtraction result rules out rounding only during final output
materialization: information is lost before or during native integer arithmetic.
This does not distinguish native input adaptation from operator implementation.
Hardware and OS differ between the two machines, so this is not an isolated OS
version comparison and does not establish behavior on iOS or watchOS.

Hosted evidence:

- [Native operator controls](https://github.com/rebeckerspecialties/rustnn/actions/runs/35538433203)
- [Output backings and small-result subtraction](https://github.com/rebeckerspecialties/rustnn/actions/runs/35538517758)
- [Operator-set comparison](https://github.com/rebeckerspecialties/rustnn/actions/runs/35538610708)

The URL-compilation workaround in #235 is insufficient on the hosted runtime.
Do not loosen the integer assertion, silently accept rounded values, or infer
that a green diagnostic job means these models meet exact WebNN semantics.
