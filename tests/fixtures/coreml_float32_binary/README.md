# Float32 binary arithmetic reference

`cases.bin` contains 29,800 records of four little-endian `uint32` words:
operation (`0` = Mul, `1` = Div), left operand bits, right operand bits, and
nearest-even result bits. NaNs are compared by class, not payload or sign;
zeros retain their sign.

Regenerate with:

```sh
node scripts/generate_coreml_float32_binary.mjs tests/fixtures/coreml_float32_binary --packed
```

The generator uses BigInt rational arithmetic and integer nearest-even rounding
without floating-point arithmetic in the reference calculation. It covers
boundary values, deterministic seeded pairs, special values, and 1,292 cases
where a subnormal input contributes to a normal output. It also cross-checks
the generated expectations against JavaScript binary64 arithmetic followed by
Float32 conversion. The packed SHA-256 is
`95b40f118bf2576705b57148ba29809ba9bfb7b6aac9f146a44b1087c5ac5676`.

The exact-rounding kernel and public-tensor checks are local regressions; they
do not tighten WPT's existing Mul 1 ULP or Div 2 ULP allowances. The separate
Exp compositions check independently rounded intermediate values, both with
and without public fanout. They do not assume that every conforming upstream
Exp approximation yields the same composed result.
