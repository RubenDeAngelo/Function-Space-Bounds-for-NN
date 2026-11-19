"""Independent chain-rule checks specific to reverse suffix propagation."""
import unittest
from verified_arithmetic import Arithmetic, rational, tanh_ranges
from verified_network_reverse import VerifiedNetwork

class ReverseTests(unittest.TestCase):
    def test_nondiagonal_suffix_hessian_on_boxes(self):
        # f=tanh(tanh(x+2y)-2*tanh(3x-y)); its intermediate Hessian
        # has nonzero off-diagonals, testing the D H D contribution.
        a=Arithmetic(96); ref=Arithmetic(512)
        net=VerifiedNetwork(a, [[[1,2],[3,-1]], [[1,-2]], [[1]]],
                            [[0,0],[0],[0]])
        for box in ((("-1/4","1/3"),("-1/5","2/5")),
                    (("2","5/2"),("-2","-3/2"))):
            jet,=net.evaluate(box)
            for r in range(5):
                for s in range(5):
                    x,y=[rational(lo)+(rational(hi)-rational(lo))*k/4
                         for (lo,hi),k in zip(box,(r,s))]
                    u,du,ddu=tanh_ranges(ref.interval(x+2*y))
                    v,dv,ddv=tanh_ranges(ref.interval(3*x-y))
                    z=u-2*v; value,dz,ddz=tanh_ranges(z)
                    g=[du-6*dv,2*du+2*dv]
                    h=[ddu-18*ddv,2*ddu+6*ddv,4*ddu-2*ddv]
                    targets=[value]+[dz*t for t in g]+[
                        ddz*g[i]*g[j]+dz*h[k]
                        for k,(i,j) in enumerate(((0,0),(0,1),(1,1)))]
                    for enclosure,target in zip((jet.value,)+jet.gradient+jet.hessian,targets):
                        self.assertLessEqual(enclosure.lo,target.lo)
                        self.assertGreaterEqual(enclosure.hi,target.hi)

    def test_single_affine_layer_and_orders(self):
        a=Arithmetic(80)
        net=VerifiedNetwork(a, [[[2,-3],[-1,4]]], [[1,2]])
        for order in (0,1,2):
            jets=net.evaluate(((0,1),(-1,2)),order)
            self.assertEqual(len(jets),2)
            self.assertEqual(len(jets[0].gradient),2 if order else 0)
            self.assertEqual(len(jets[0].hessian),3 if order==2 else 0)
            if order==2:
                self.assertTrue(all(h.lo==h.hi==0 for j in jets for h in j.hessian))

    def test_nested_boxes_and_shrinking_widths(self):
        a=Arithmetic(128)
        net=VerifiedNetwork(a, [[[1,2],[-3,1]], [[1,-2],[2,1]], [[1,-1]]],
                            [[0,1],[1,0],[0]])
        previous=None
        widths=[]
        for k in range(1,9):
            e=rational(1)/2**k
            jet,=net.evaluate(((0,e),(0,e)))
            entries=(jet.value,)+jet.gradient+jet.hessian
            if previous:
                for outer,inner in zip(previous,entries):
                    self.assertLessEqual(outer.lo,inner.lo)
                    self.assertGreaterEqual(outer.hi,inner.hi)
            widths.append(max(float(x.width) for x in entries))
            previous=entries
        self.assertLess(widths[-1],widths[-2]*0.6)
