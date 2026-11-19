import copy
import unittest
import mpmath as mp
from experiment_records import validate_record
from verified_arithmetic import Arithmetic, rational
from verified_network import VerifiedNetwork
from verified_integration import Box, Integrator
from verified_residual import tanh_poisson_forcing, poisson_residual_integrand


class ResidualTests(unittest.TestCase):
    def test_forcing_at_saturation_and_critical_points(self):
        a=Arithmetic(80)
        with mp.workdps(100):
            for x,y in [(0,0),(1,1),(.5,.5),(.125,.375)]:
                f=tanh_poisson_forcing(a,((x,x),(y,y)))
                z=mp.mpf(x)+mp.mpf(y)-1;t=mp.tanh(z);exact=-4*(1-t*t)*t
                # Compare exact rationals enclosing the high-precision decimal oracle.
                self.assertLessEqual(mp.mpf(str(rational(f.lo).numerator))/int(rational(f.lo).denominator),exact)
                self.assertGreaterEqual(mp.mpf(str(rational(f.hi).numerator))/int(rational(f.hi).denominator),exact)
        whole=tanh_poisson_forcing(a,((0,1),(0,1)))
        self.assertTrue(whole.contains(0))
        self.assertLess(whole.lo, -1.5);self.assertGreater(whole.hi,1.5)

    def test_exact_solution_residual_and_subdivision(self):
        a=Arithmetic(100)
        network=VerifiedNetwork(a,[[[1,1]],[[1]]],[[-1],[0]])
        # This network is exactly tanh(x+y-1), hence its true residual is zero.
        parent=poisson_residual_integrand(network,((0,1),(0,1)))
        point=poisson_residual_integrand(network,((.5,.5),(.5,.5)))
        self.assertTrue(parent.contains(0));self.assertTrue(point.contains(0))
        self.assertLess(point.width,parent.width)
        run=Integrator(a,lambda b:poisson_residual_integrand(network,b),Box(((0,1),(0,1))))
        run.run(max_leaves=64)
        self.assertTrue(run.total().contains(0));validate_record(run.serialise(),((0,1),(0,1)))

    def test_point_samples_of_non_solution(self):
        a=Arithmetic(80);ref=Arithmetic(512)
        network=VerifiedNetwork(a,[[[1,2]],[[3]]],[[-1],[.25]])
        exact=VerifiedNetwork(ref,[[[1,2]],[[3]]],[[-1],[.25]])
        envelope=poisson_residual_integrand(network,((0,1),(0,1)))
        for x,y in [(0,0),(.25,.75),(1,1)]:
            sample=poisson_residual_integrand(exact,((x,x),(y,y)))
            self.assertLessEqual(envelope.lo,sample.lo);self.assertGreaterEqual(envelope.hi,sample.hi)


class RecordTests(unittest.TestCase):
    def test_record_validator_rejects_corruption(self):
        a=Arithmetic(80);run=Integrator(a,lambda b:a.interval(*b[0]).abs_power(2),Box(((0,1),)))
        run.run(max_leaves=16);record=run.serialise();validate_record(record,((0,1),))
        bad=copy.deepcopy(record);bad['cells'][0]['bounds'][0][1]='100'
        with self.assertRaises(AssertionError):validate_record(bad,((0,1),))
        bad=copy.deepcopy(record);bad['history'][-1]['norm']['upper']='0'
        with self.assertRaises(AssertionError):validate_record(bad,((0,1),))
        bad=copy.deepcopy(record);bad['cells'].append(copy.deepcopy(bad['cells'][0]))
        with self.assertRaises(AssertionError):validate_record(bad,((0,1),))

    def test_disk_target_definition(self):
        import torch
        from exp_verified_studies import disk_target
        x=torch.tensor([[0.,0.],[1.,0.],[1.25,0.],[1.5,0.],[2.,0.]],dtype=torch.float64)
        self.assertTrue(torch.equal(disk_target(x,.5),torch.tensor([[1.],[1.],[.5],[0.],[0.]],dtype=torch.float64)))


if __name__=='__main__':unittest.main()
