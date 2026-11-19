import unittest
from verified_arithmetic import Arithmetic
from verified_network import VerifiedNetwork
from verified_integration import Integrator,Box
class ReLUTests(unittest.TestCase):
    def test_known_weak_norms(self):
        a=Arithmetic(128);n=VerifiedNetwork(a,[[[1]],[[1]]],[[0],[0]],'relu')
        for order,exact in [(0,'1/3'),(1,'4/3')]:
            run=Integrator(a,lambda b:n.sobolev_integrand(b,order=order),Box(((-1,1),)))
            run.run(max_leaves=128)
            self.assertTrue(run.total().contains(exact))
    def test_multilayer_and_zero(self):
        a=Arithmetic();n=VerifiedNetwork(a,[[[1],[-1]],[[1,1]],[[1]]],[[0,0],[0],[0]],'relu')
        for pt,derivative in [(-2,-1),(2,1)]:
            j=n.evaluate([(pt,pt)],1)[0];self.assertTrue(j.value.contains(abs(pt)));self.assertTrue(j.gradient[0].contains(derivative))
        n=VerifiedNetwork(a,[[[0]],[[1]]],[[0],[0]],'relu');self.assertEqual(n.evaluate([[-1,1]],1)[0].gradient[0].hi,0)
    def test_h2_rejected(self):
        with self.assertRaises(ValueError):VerifiedNetwork(Arithmetic(),[[[1]]],[[0]],'relu').evaluate([[-1,1]],2)
if __name__=='__main__':unittest.main()
