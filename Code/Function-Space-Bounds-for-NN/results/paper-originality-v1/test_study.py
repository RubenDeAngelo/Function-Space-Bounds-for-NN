import unittest
from fractions import Fraction as Q
from experiment import *

class StudyTests(unittest.TestCase):
    def test_independent_interval_oracle(self):
        for net,quantity in [(clipped_hat(16),'L2'),(cancellation_hat(16),'L2'),
                             (spline([Q(1,3),Q(2,3)],[1,-2]),'H1')]:
            oracle=Oracle(net,quantity,'range')
            for l,u in [(Q(0),Q(1)),(Q(0),Q(1,256)),(Q(1,4),Q(1,2)),(Q(3,4),Q(1))]:
                lo,hi=exact_relu_range(net,l,u,quantity)
                c=oracle.cell(l,u)
                self.assertLessEqual(c.lower,lo*(u-l))
                self.assertGreaterEqual(c.upper,hi*(u-l))

    def test_affine_moments_and_boundary(self):
        n=spline([Q(1,3)],[1])
        self.assertIsNone(affine_on_cell(n,Q(0),Q(1)))
        self.assertIsNotNone(affine_on_cell(n,Q(1,3),Q(1)))
        self.assertEqual(exact_piecewise_integral(n,[Q(1,3)],'H1'),Q(62,81))
        self.assertEqual(exact_piecewise_integral(n,[Q(1,3)],'semi'),Q(2,3))
        self.assertEqual(exact_piecewise_integral(n,[Q(1,3)],'L2'),Q(8,81))

    def test_normalized_peak(self):
        for n in [8,16,32]:
            net=clipped_hat(n)
            self.assertEqual(exact_piecewise_integral(net,[Q(1,2*n*n),Q(1,n*n)],'L2'),Q(1,3))
            r=run(net,'L2','range','uniform',n)
            self.assertEqual(Q(r['final']['gap']),n)
            self.assertEqual(r['final']['counts']['cell_calls'],2*n-1)
            a=run(net,'L2','range','maxgap',n)
            self.assertEqual(a['final']['leaves'],n)
            self.assertEqual(a['final']['counts']['cell_calls'],2*n-1)

    def test_hinge_decay_and_roundoff_limit(self):
        n=spline([Q(1,3)],[1])
        for leaves in [4,8,16]:
            r=run(n,'semi','range','maxgap',leaves)
            # 128-bit nonbinary coefficient conversion can only widen the exact oracle.
            ideal=Q(1,2**(leaves-1));got=Q(r['final']['gap'])
            self.assertGreaterEqual(got,ideal)
            self.assertLess(got-ideal,Q(1,2**110))
        h=run(n,'H1','hybrid','maxgap',16)
        self.assertGreater(h['final']['counts']['exact_polynomial_integrals'],0)
        self.assertLessEqual(Q(h['final']['lower']),Q(62,81))
        self.assertGreaterEqual(Q(h['final']['upper']),Q(62,81))

    def test_midpoint_on_analytic_tanh_integral(self):
        net={'activation':'tanh','weights':[[['1']],[['1']]],'biases':[['0'],['0']]}
        oracle=Oracle(net,'L2','midpoint')
        endpoint=oracle.net.evaluate([(1,1)],0)[0].value
        exact=oracle.a.interval(1)-endpoint  # integral_0^1 tanh(x)^2 dx = 1-tanh(1)
        r=run(net,'L2','midpoint','maxgap',16)
        self.assertLessEqual(Q(r['final']['lower']),fq(exact.lo))
        self.assertGreaterEqual(Q(r['final']['upper']),fq(exact.hi))
        old=run(net,'L2','range','maxgap',16)
        self.assertLess(Q(r['final']['gap']),Q(old['final']['gap']))

if __name__=='__main__':unittest.main()
