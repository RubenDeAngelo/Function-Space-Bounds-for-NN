import json
import random
import unittest
import gmpy2 as g
from verified_arithmetic import Arithmetic, rational, tanh_ranges
from verified_network import VerifiedNetwork
from verified_integration import Box, Integrator


class ArithmeticTests(unittest.TestCase):
    def test_exact_rational_oracle(self):
        rng, a = random.Random(19), Arithmetic(17)
        for _ in range(200):
            x = rational(rng.randint(-10000, 10000)) / rng.randint(1, 1000)
            y = rational(rng.randint(-10000, 10000)) / rng.randint(1, 1000)
            ix, iy = a.interval(x), a.interval(y)
            for computed, exact in ((ix + iy, x + y), (ix - iy, x - y),
                                    (ix * iy, x * y)):
                self.assertTrue(computed.contains(exact))
            if not iy.lo <= 0 <= iy.hi:
                self.assertTrue((ix / iy).contains(x / y))

    def test_cancellation_and_global_sum(self):
        a = Arithmetic(53)
        result = a.sum([a.interval(2**60), a.interval(1), a.interval(-2**60)])
        self.assertTrue(result.contains(1))
        self.assertGreater(result.width, 0)
        self.assertTrue((a.interval(-3, 2).abs_power(2)).contains(0))
        self.assertTrue((a.interval(-3, 2).abs_power(2)).contains(9))

    def test_interval_extreme_products_and_fail_closed(self):
        a = Arithmetic(17)
        for l, u, v, w in ((-7, -2, -5, 3), (0, 1, -2, 4), (-3, 5, -7, -1)):
            answer = a.interval(l, u) * a.interval(v, w)
            for x in (l, u):
                for y in (v, w):
                    self.assertTrue(answer.contains(x*y))
        small_range = Arithmetic(53)
        small_range.down.emax = small_range.up.emax = 20
        with self.assertRaises((OverflowError, ArithmeticError)):
            small_range.interval(2**19) * small_range.interval(2**19)
        small_range.down.emin = small_range.up.emin = -20
        with self.assertRaises((ArithmeticError, OverflowError)):
            small_range.interval("1e-100")

    def test_input_and_export_semantics(self):
        a = Arithmetic()
        self.assertTrue(a.interval(0.1).contains(rational(0.1)))
        self.assertNotEqual(rational(0.1), rational("0.1"))
        interval = a.interval("1/3", "2/3")
        record = json.loads(json.dumps(interval.serialise()))
        self.assertEqual(rational(interval.lo), rational(record["lower"]))
        self.assertEqual(rational(interval.hi), rational(record["upper"]))
        for bad in (float("inf"), float("nan")):
            with self.assertRaises((ValueError, OverflowError)):
                a.interval(bad)
        with self.assertRaises(ZeroDivisionError):
            a.interval(1) / a.interval(-1, 1)
        with self.assertRaises(ValueError):
            a.interval(1) + Arithmetic().interval(1)
        with self.assertRaises(ValueError):
            a.interval(1).abs_power(1.5)

    def test_roots_and_near_zero(self):
        a, ref = Arithmetic(80), Arithmetic(512)
        x = a.interval("1e-200", "1e-199")
        root = x.root(2)
        self.assertLessEqual(root.lo, ref.interval("1e-200").root(2).lo)
        self.assertGreaterEqual(root.hi, ref.interval("1e-199").root(2).hi)
        self.assertEqual(a.interval(0).root(3).lo, 0)

    def test_tanh_extrema_saturation(self):
        a, ref = Arithmetic(80), Arithmetic(512)
        for lo, hi in ((-3, 3), ("0.65", "0.67"), ("-0.67", "-0.65"),
                       (100, 101), (-101, -100), ("1e-100", "2e-100")):
            ranges = tanh_ranges(a.interval(lo, hi))
            for j in range(21):
                point = rational(lo) + (rational(hi) - rational(lo)) * j / 20
                rr = tanh_ranges(ref.interval(point))
                for low, high in zip(ranges, rr):
                    self.assertLessEqual(low.lo, high.lo)
                    self.assertGreaterEqual(low.hi, high.hi)
        # Saturated derivatives remain strictly positive, not 1-tanh^2 == 0.
        self.assertGreater(tanh_ranges(a.interval(100, 101))[1].lo, 0)

    def test_ambient_context_independence(self):
        a = Arithmetic(100)
        expected = tanh_ranges(a.interval("0.1", "0.2"))
        old = g.get_context().copy()
        try:
            g.get_context().precision = 8
            g.get_context().round = g.RoundToZero
            actual = tanh_ranges(a.interval("0.1", "0.2"))
            self.assertEqual([(x.lo, x.hi) for x in expected],
                             [(x.lo, x.hi) for x in actual])
        finally:
            g.set_context(old)


class NetworkTests(unittest.TestCase):
    def test_mixed_derivative_count_and_multiple_outputs(self):
        a = Arithmetic(80)
        net = VerifiedNetwork(a, [[[1, 1]], [[1], [-2]]], [[0], [0, 0]])
        bounds = (("0.1", "0.1"), ("0.2", "0.2"))
        jets = net.evaluate(bounds)
        self.assertEqual(len(jets), 2)
        self.assertEqual(len(jets[0].hessian), 3)
        for x, y in zip(jets[0].hessian, jets[1].hessian):
            self.assertTrue((-2*x).intersect(y).width >= 0)
        combined = net.sobolev_integrand(bounds)
        ref = Arithmetic(512)
        t, first, second = tanh_ranges(ref.interval("0.3"))
        expected = 5 * (t.abs_power(2) + 2*first.abs_power(2) + 3*second.abs_power(2))
        self.assertLessEqual(combined.lo, expected.lo)
        self.assertGreaterEqual(combined.hi, expected.hi)

    def test_affine_derivatives_and_exact_coefficients(self):
        a = Arithmetic(53)
        network = VerifiedNetwork(a, [[[2, -3]], [[4]]], [[1], [-2]], "affine")
        jet, = network.evaluate(((0, 1), (0, 1)))
        self.assertTrue(jet.gradient[0].contains(8))
        self.assertTrue(jet.gradient[1].contains(-12))
        self.assertTrue(all(h.lo == h.hi == 0 for h in jet.hessian))
        cancellation = VerifiedNetwork(a, [[[2**60, 1, -2**60]]], [[0]])
        self.assertTrue(cancellation.evaluate(((1, 1),) * 3)[0].value.contains(1))

    def test_chain_rule_against_high_precision_differentiation(self):
        import mpmath as mp
        mp.mp.dps = 100
        a = Arithmetic(80)
        w = [[["3/2", "-2/3"], ["-1/4", "5/3"]],
             [["2/5", "-4/3"], ["7/8", "1/9"]], [["3/7", "-5/4"]]]
        b = [["1/7", "-1/5"], ["2/3", "-1/11"], ["1/13"]]
        network = VerifiedNetwork(a, w, b)

        def m(q):
            q = rational(q)
            return mp.mpf(int(q.numerator)) / int(q.denominator)

        def function(x, y):
            values = [x, y]
            for k, (weights, biases) in enumerate(zip(w, b)):
                values = [sum(m(c) * v for c, v in zip(row, values)) + m(bias)
                          for row, bias in zip(weights, biases)]
                if k < len(w) - 1:
                    values = [mp.tanh(v) for v in values]
            return values[0]

        for x, y in (("0.1", "-0.2"), ("0.5", "0.7"), ("-0.8", "0.1")):
            jet, = network.evaluate(((x, x), (y, y)))
            point = (m(x), m(y))
            targets = [function(*point)] + [mp.diff(function, point, n)
                       for n in ((1, 0), (0, 1), (2, 0), (1, 1), (0, 2))]
            for interval, target in zip((jet.value,) + jet.gradient + jet.hessian, targets):
                # High-precision numerical differential oracle, not proof of inclusion.
                self.assertTrue(interval.contains(mp.nstr(target, 95)))
            bounds = network.sobolev_integrand(((x, x), (y, y)))
            expected = sum(t*t for t in targets)
            self.assertTrue(bounds.contains(mp.nstr(expected, 95)))

    def test_torch_adapter_and_unsupported_graphs(self):
        import torch
        a = Arithmetic()
        model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Tanh(),
                                    torch.nn.Linear(3, 1)).double()
        network = VerifiedNetwork.from_torch(model, a)
        self.assertEqual(network.weights[0][0][0], rational(model[0].weight[0, 0].item()))
        self.assertEqual(network.serialise(),
                         VerifiedNetwork.from_record(network.serialise(), a).serialise())
        with self.assertRaises(ValueError):
            VerifiedNetwork.from_torch(torch.nn.Sequential(torch.nn.Linear(2, 1),
                                                          torch.nn.ReLU()), a)
        with self.assertRaises(ValueError):
            VerifiedNetwork(a, [[[1]]], [[0]], "sigmoid")


class IntegrationTests(unittest.TestCase):
    def test_exact_geometry_small_boxes(self):
        lo = rational(1)
        hi = lo + rational(1) / 2**2000
        box = Box(((lo, hi), (0, 1)))
        children = box.children()
        self.assertEqual(sum((b.volume for b in children), rational(0)), box.volume)
        self.assertTrue(all(b.bounds[0][0] < b.bounds[0][1] for b in children))

    def test_analytic_integral_and_norm(self):
        a = Arithmetic(80)
        net = VerifiedNetwork(a, [[[2]]], [[1]])
        domain = Box(((-1, 1),))
        for policy in ("adaptive", "uniform"):
            run = Integrator(a, net.sobolev_integrand, domain).run(
                policy=policy, max_leaves=64)
            integral = run.total()
            self.assertTrue(integral.contains(rational(38)/3))
            root = integral.root(2)
            self.assertLessEqual(a.down.pow(root.lo, 2), rational(38)/3)
            self.assertGreaterEqual(a.up.pow(root.hi, 2), rational(38)/3)
            gaps = [rational(r["gap_upper"]) for r in run.history]
            self.assertLess(gaps[-1], gaps[0])
            self.assertLessEqual(run.history[-1]["leaves"], 64)
            for c in run.cells:
                self.assertTrue(c.box.volume > 0)

    def test_nonlinear_analytic_integral(self):
        a, ref = Arithmetic(80), Arithmetic(512)
        network = VerifiedNetwork(a, [[[1]], [[1]]], [[0], [0]])
        run = Integrator(a, lambda b: network.sobolev_integrand(b, order=0),
                         Box(((0, 1),))).run(max_leaves=128)
        # Integral_0^1 tanh(x)^2 dx = 1 - tanh(1).
        exact = ref.interval(1) - tanh_ranges(ref.interval(1))[0]
        self.assertLessEqual(run.total().lo, exact.lo)
        self.assertGreaterEqual(run.total().hi, exact.hi)

    def test_zero_budgets_tolerance_and_stagnation(self):
        a, box = Arithmetic(53), Box(((0, 1),))
        constant = lambda b: a.interval(4)
        run = Integrator(a, constant, box).run(max_steps=0)
        self.assertEqual(run.status, "tolerance_reached")
        self.assertTrue(run.total().contains(4))
        run = Integrator(a, lambda b: a.interval(1, 2), box).run(
            max_leaves=100, stagnation_steps=2)
        self.assertEqual(run.status, "gap_stagnation")
        run = Integrator(a, lambda b: a.interval(1, 2), box).run(max_leaves=1)
        self.assertEqual(run.status, "leaf_budget_exhausted")
        run = Integrator(a, lambda b: a.interval(1, 2), box).run(max_depth=0)
        self.assertEqual(run.status, "depth_budget_exhausted")
        run = Integrator(a, lambda b: a.interval(1, 2), box).run(max_steps=0)
        self.assertEqual(run.status, "step_budget_exhausted")


if __name__ == "__main__":
    unittest.main()
