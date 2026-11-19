"""Finite endpoint intervals using explicitly directed MPFR operations.

No NumPy/BLAS/Numba arithmetic enters a certificate. Python floats denote
their exact binary values; strings denote exact rational numbers. Overflow,
underflow, invalid operations and division by zero fail closed.
"""
from dataclasses import dataclass
from fractions import Fraction
import math
import gmpy2 as g


def rational(value):
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Nonfinite input")
        return g.mpq(*value.as_integer_ratio())
    if isinstance(value, Fraction):
        return g.mpq(value.numerator, value.denominator)
    return g.mpq(value)


class Arithmetic:
    def __init__(self, bits=128):
        if not isinstance(bits, int) or bits < 8:
            raise ValueError("bits must be an integer >= 8")
        self.bits = bits
        options = dict(precision=bits, trap_invalid=True, trap_divzero=True,
                       trap_overflow=True, trap_underflow=True)
        self.down = g.context(round=g.RoundDown, **options)
        self.up = g.context(round=g.RoundUp, **options)
        self.nearest = g.context(round=g.RoundToNearest, **options)

    def raw(self, lo, hi):
        # gmpy2 context arithmetic on two mpq operands remains exact mpq.
        # Explicitly convert rational inputs to directed MPFR endpoints.
        if not isinstance(lo, g.mpfr):
            with self.down:
                lo = g.mpfr(lo)
        if not isinstance(hi, g.mpfr):
            with self.up:
                hi = g.mpfr(hi)
        if not g.is_finite(lo) or not g.is_finite(hi) or lo > hi:
            raise ArithmeticError("Invalid or nonfinite enclosure")
        return Interval(self, lo, hi)

    def interval(self, lo, hi=None):
        lo, hi = rational(lo), rational(lo if hi is None else hi)
        if lo > hi:
            raise ValueError("Reversed interval")
        return self.raw(self.down.add(lo, 0), self.up.add(hi, 0))

    def sum(self, values):
        result = self.interval(0)
        for value in values:
            result = result + value
        return result


@dataclass(frozen=True)
class Interval:
    arithmetic: Arithmetic
    lo: object
    hi: object

    def _coerce(self, other):
        if not isinstance(other, Interval):
            other = self.arithmetic.interval(other)
        if other.arithmetic is not self.arithmetic:
            raise ValueError("Cannot mix arithmetic instances")
        return other

    def __add__(self, other):
        other = self._coerce(other)
        a = self.arithmetic
        return a.raw(a.down.add(self.lo, other.lo), a.up.add(self.hi, other.hi))

    __radd__ = __add__

    def __neg__(self):
        a = self.arithmetic
        return a.raw(a.down.minus(self.hi), a.up.minus(self.lo))

    def __sub__(self, other):
        return self + (-self._coerce(other))

    def __rsub__(self, other):
        return self._coerce(other) - self

    def __mul__(self, other):
        other = self._coerce(other)
        a = self.arithmetic
        if self.lo == self.hi:
            c = self.lo
            if c >= 0:
                return a.raw(a.down.mul(c, other.lo), a.up.mul(c, other.hi))
            return a.raw(a.down.mul(c, other.hi), a.up.mul(c, other.lo))
        if other.lo == other.hi:
            return other * self
        pairs = [(x, y) for x in (self.lo, self.hi) for y in (other.lo, other.hi)]
        return a.raw(min(a.down.mul(x, y) for x, y in pairs),
                     max(a.up.mul(x, y) for x, y in pairs))

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = self._coerce(other)
        if other.lo <= 0 <= other.hi:
            raise ZeroDivisionError("Interval divisor contains zero")
        a = self.arithmetic
        return self * a.raw(a.down.div(1, other.hi), a.up.div(1, other.lo))

    def abs_power(self, p):
        """Exact range shape, with outward-rounded integer powers."""
        if not isinstance(p, int) or p < 1:
            raise ValueError("Only positive integer powers are supported")
        a = self.arithmetic
        # MPFR negation/abs via context methods, never ambient arithmetic.
        low_abs = min(a.down.abs(self.lo), a.down.abs(self.hi))
        high_abs = max(a.up.abs(self.lo), a.up.abs(self.hi))
        if self.lo <= 0 <= self.hi:
            low_abs = 0
        return a.raw(a.down.pow(low_abs, p), a.up.pow(high_abs, p))

    def root(self, p):
        if not isinstance(p, int) or p < 1 or self.lo < 0:
            raise ValueError("Root requires a nonnegative interval and integer p >= 1")
        a = self.arithmetic
        return a.raw(a.down.rootn(self.lo, p), a.up.rootn(self.hi, p))

    def intersect(self, other):
        other = self._coerce(other)
        return self.arithmetic.raw(max(self.lo, other.lo), min(self.hi, other.hi))

    def hull(self, other):
        other = self._coerce(other)
        return self.arithmetic.raw(min(self.lo, other.lo), max(self.hi, other.hi))

    def contains(self, value):
        value = rational(value)
        return self.lo <= value <= self.hi

    @property
    def width(self):
        return self.arithmetic.up.sub(self.hi, self.lo)

    @property
    def exact_width(self):
        return rational(self.hi) - rational(self.lo)

    def serialise(self):
        # Rational strings preserve endpoints exactly; decimal printouts do not.
        return {"lower": str(rational(self.lo)), "upper": str(rational(self.hi))}


def tanh_ranges(x):
    """Ranges of tanh, sech^2, and -2*tanh*sech^2, through extrema.

    sech avoids cancellation in 1-tanh^2 for saturated inputs. Both the
    critical location and value are intervals, not decimal approximations.
    """
    a = x.arithmetic
    t = a.raw(a.down.tanh(x.lo), a.up.tanh(x.hi))

    def sech_square(z):
        return a.raw(a.down.sech(z), a.up.sech(z)).abs_power(2)

    ga, gb = sech_square(x.lo), sech_square(x.hi)
    first = ga.hull(gb)
    if x.lo <= 0 <= x.hi:
        first = first.hull(1)

    def second(z):
        tz = a.raw(a.down.tanh(z), a.up.tanh(z))
        return -2 * tz * sech_square(z)

    h = second(x.lo).hull(second(x.hi))
    if not hasattr(a, "_tanh_extrema"):
        sqrt3 = a.interval(3).root(2)
        inv = a.interval(1) / sqrt3
        critical = a.raw(a.down.atanh(inv.lo), a.up.atanh(inv.hi))
        extremum = a.interval(4) / (3 * sqrt3)
        a._tanh_extrema = critical, extremum
    critical, extremum = a._tanh_extrema
    for location, value in ((-critical, extremum), (critical, -extremum)):
        if location.lo <= x.hi and location.hi >= x.lo:
            h = h.hull(value)
    return t, first, h
