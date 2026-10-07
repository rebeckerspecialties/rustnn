#!/usr/bin/env python3
"""Independent Decimal reference for the small CoreML boundary diagnostic.

No CoreML, NumPy or host Half conversion is used. Run through Make; --check
compares two working precisions and the checked-in, once-rounded bit patterns.
"""

import argparse
import decimal
import json
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/numeric_boundaries.json"
PI = decimal.Decimal(
    "3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067982148086513282306647093844609550582231725359408128481"
)


def decode(bits, width):
    fraction_bits, exponent_bits, bias = (10, 5, 15) if width == 16 else (23, 8, 127)
    sign = bits >> (width - 1)
    exponent = (bits >> fraction_bits) & ((1 << exponent_bits) - 1)
    fraction = bits & ((1 << fraction_bits) - 1)
    if exponent == (1 << exponent_bits) - 1:
        result = decimal.Decimal("NaN" if fraction else "Infinity")
    else:
        mantissa = fraction if exponent == 0 else (1 << fraction_bits) + fraction
        power = (1 - bias if exponent == 0 else exponent - bias) - fraction_bits
        result = decimal.Decimal(mantissa) * decimal.Decimal(2) ** power
    return result.copy_negate() if sign else result


def encode(value, width):
    fraction_bits, exponent_bits, bias = (10, 5, 15) if width == 16 else (23, 8, 127)
    sign = int(value.is_signed()) << (width - 1)
    infinity = ((1 << exponent_bits) - 1) << fraction_bits
    if value.is_nan():
        return infinity | (1 << (fraction_bits - 1))
    if value.is_infinite():
        return sign | infinity
    value = abs(Fraction(value))
    if not value:
        return sign
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if value < Fraction(2) ** exponent:
        exponent -= 1
    quantum = max(exponent, 1 - bias) - fraction_bits
    scaled = value / Fraction(2) ** quantum
    quotient, remainder = divmod(scaled.numerator, scaled.denominator)
    quotient += (2 * remainder > scaled.denominator or
                 (2 * remainder == scaled.denominator and quotient & 1))
    if quotient == 1 << (fraction_bits + 1):
        quotient >>= 1
        exponent += 1
    if exponent > bias:
        return sign | infinity
    if quotient < 1 << fraction_bits:
        return sign | quotient
    return sign | ((max(exponent, 1 - bias) + bias) << fraction_bits) | (quotient - (1 << fraction_bits))


def reference(op, value):
    if value.is_nan():
        return decimal.Decimal("NaN")
    if op == "exp":
        return value.exp()
    if op == "sqrt":
        return decimal.Decimal("NaN") if value < 0 else value.sqrt()
    # GELU's exact formula, including its undefined -Infinity * 0 product.
    if value == decimal.Decimal("-Infinity"):
        return decimal.Decimal("NaN")
    if value == decimal.Decimal("Infinity"):
        return value
    if value.is_zero():
        return value
    # Beyond these tails the once-rounded binary32 result is x or signed zero.
    if abs(value) >= 16:
        return decimal.Decimal("-0") if value < 0 else value
    z = value / decimal.Decimal(2).sqrt()
    term = z
    total = term
    n = 0
    while True:
        n += 1
        term *= -z * z / n
        updated = total + term / (2 * n + 1)
        if updated == total:
            break
        total = updated
    erf = 2 * total / PI.sqrt()
    return value * (1 + erf) / 2


def generate(precision):
    with decimal.localcontext() as ctx:
        ctx.prec = precision
        rows = []
        for width in (16, 32):
            tail = [-18, -17, -16, -12, -10] if width == 16 else [-104, -103, -100, -95, -90, -88]
            exp_inputs = [encode(decimal.Decimal(x), width) for x in tail + [-1, 0, 1]]
            sqrt_inputs = ([0, 0x8000, 1, 2, 0x3ff, 0x400, 0x401, 0x3c00, 0x7bff, 0xbc00, 0x7c00, 0xfc00, 0x7e00]
                           if width == 16 else [0, 0x80000000, 1, 2, 0x7fffff, 0x800000, 0x800001, 0x3f800000, 0x7f7fffff, 0xbf800000, 0x7f800000, 0xff800000, 0x7fc00000])
            unary = [("exp", exp_inputs, 1 if width == 16 else 32), ("sqrt", sqrt_inputs, 1)]
            if width == 32:
                # Half GELU already has an exhaustive suite in the GELU PR.
                gelu_inputs = [encode(decimal.Decimal(x), width) for x in [-65504, -10, -5, -1, 1, 5, 10, 65504]]
                gelu_inputs += [0, 0x80000000, 1, 0x80000001, 0x7fffff, 0x807fffff, 0x7f800000, 0xff800000, 0x7fc00000]
                unary.append(("gelu", gelu_inputs, 18))
            for op, inputs, budget in unary:
                rows.append({"op": op, "width": width, "ulp_budget": budget,
                             "input_bits": inputs,
                             "expected_bits": [encode(reference(op, decode(x, width)), width) for x in inputs]})
        return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = generate(100)
    assert result == generate(140), "reference changes with Decimal precision"
    if args.check:
        assert result == json.loads(FIXTURE.read_text()), "checked-in reference drift"
        print("Independent boundary reference and higher-precision cross-check passed")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
