"""Validated constant-envelope integration on exact rational dyadic boxes.

All subdivision geometry is rational and exact. Per-cell ranges are intersected
with their ancestors' ranges, both of which enclose the same function on the
child. Global endpoints are accumulated with MPFR directed rounding. Marking
uses exact rational widths of the computed local integral intervals.
"""
from dataclasses import dataclass
from itertools import product
from time import perf_counter
from verified_arithmetic import rational


@dataclass(frozen=True)
class Box:
    bounds: tuple
    depth: int = 0

    def __post_init__(self):
        b = tuple((rational(lo), rational(hi)) for lo, hi in self.bounds)
        if not b or any(lo >= hi for lo, hi in b):
            raise ValueError("Box must have finite, positive side lengths")
        object.__setattr__(self, "bounds", b)

    @property
    def volume(self):
        v = rational(1)
        for lo, hi in self.bounds:
            v *= hi - lo
        return v

    def children(self):
        # No binary64 midpoint conversion: arbitrarily small dyadic children
        # remain geometrically distinct. max_depth controls resource use.
        axes = []
        for lo, hi in self.bounds:
            middle = (lo + hi) / 2
            axes.append(((lo, middle), (middle, hi)))
        return tuple(Box(tuple(b), self.depth + 1) for b in product(*axes))


@dataclass(frozen=True)
class Cell:
    box: Box
    integrand: object
    integral: object

    @property
    def indicator(self):
        return self.integral.exact_width


class Integrator:
    def __init__(self, arithmetic, enclosure, domain, *, p=2):
        if not isinstance(p, int) or p < 1:
            raise ValueError("p must be a positive integer")
        self.a, self.enclosure, self.p = arithmetic, enclosure, p
        self.started = perf_counter()
        self.evaluations = 0
        self.cells = [self.evaluate(domain)]
        self.history = []
        self.status = "initialized"
        self.record()

    def evaluate(self, box, parent_range=None):
        f = self.enclosure(box.bounds)
        if f.arithmetic is not self.a or f.lo < 0:
            raise ValueError("Expected nonnegative integrand in the same arithmetic")
        if parent_range is not None:
            f = f.intersect(parent_range)
        self.evaluations += 1
        return Cell(box, f, f * self.a.interval(box.volume))

    def total(self):
        return self.a.sum(c.integral for c in self.cells)

    def record(self):
        interval = self.total()
        norm = interval.root(self.p)
        record = {"leaves": len(self.cells), "evaluations": self.evaluations,
                  "seconds": perf_counter() - self.started,
                  "integral": interval.serialise(), "norm": norm.serialise(),
                  "gap_upper": str(rational(interval.width)),
                  "norm_gap_upper": str(rational(norm.width))}
        self.history.append(record)
        return interval

    def run(self, *, policy="adaptive", theta="0.35", max_leaves=256,
            tolerance=0, max_steps=10000, max_depth=1000, stagnation_steps=8):
        theta, tolerance = rational(theta), rational(tolerance)
        if policy not in ("adaptive", "uniform") or not 0 < theta < 1 or tolerance < 0:
            raise ValueError("Invalid policy, theta, or tolerance")
        if (max_leaves < len(self.cells) or max_steps < 0 or max_depth < 0
                or stagnation_steps < 1):
            raise ValueError("Invalid budget")
        previous = self.total().width
        stagnant = 0
        for _ in range(max_steps):
            if self.total().width <= tolerance:
                self.status = "tolerance_reached"
                return self
            ranked = sorted(range(len(self.cells)),
                            key=lambda i: self.cells[i].indicator, reverse=True)
            if policy == "uniform":
                marked = ranked
            else:
                total_indicator = sum((c.indicator for c in self.cells), rational(0))
                if total_indicator == 0:
                    self.status = "rounding_stagnation"
                    return self
                selected, cutoff, marked = rational(0), None, []
                for i in ranked:
                    e = self.cells[i].indicator
                    if cutoff is not None and e < cutoff:
                        break
                    marked.append(i)
                    selected += e
                    if cutoff is None and selected >= theta * total_indicator:
                        cutoff = e
            child_count = 2 ** len(self.cells[0].box.bounds)
            # Never truncate the bulk set or uniform level to fit a budget.
            if len(self.cells) + len(marked) * (child_count - 1) > max_leaves:
                self.status = "leaf_budget_exhausted"
                return self
            if any(self.cells[i].box.depth >= max_depth for i in marked):
                self.status = "depth_budget_exhausted"
                return self
            selected_set = set(marked)
            new_cells = [c for i, c in enumerate(self.cells) if i not in selected_set]
            for i in marked:
                parent = self.cells[i]
                new_cells.extend(self.evaluate(child, parent.integrand)
                                 for child in parent.box.children())
            self.cells = new_cells
            current = self.record().width
            stagnant = stagnant + 1 if current >= previous else 0
            previous = current
            if stagnant >= stagnation_steps:
                self.status = "gap_stagnation"
                return self
        self.status = "tolerance_reached" if self.total().width <= tolerance else "step_budget_exhausted"
        return self

    def serialise(self):
        return {"status": self.status, "history": self.history,
                "cells": [{"bounds": [[str(lo), str(hi)] for lo, hi in c.box.bounds],
                           "depth": c.box.depth,
                           "integrand": c.integrand.serialise(),
                           "integral": c.integral.serialise(),
                           "indicator": str(c.indicator)} for c in self.cells]}
