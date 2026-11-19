import copy
import unittest
import mpmath as mp
from experiment_records import validate_record
from verified_arithmetic import Arithmetic, rational
from verified_network import VerifiedNetwork
from verified_integration import Box, Integrator
from verified_residual import tanh_poisson_forcing, poisson_residual_integrand


class ResidualTests(unittest.TestCase):
    def test_forcing_points_and_critical_range(self):
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
        for x,y in [(0,0),(.25,.25),(.25,.75),(1,1)]:
            self.assertTrue(poisson_residual_integrand(network,((x,x),(y,y))).contains(0))
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
            with mp.workdps(100):
                t=mp.tanh(mp.mpf(x)+2*mp.mpf(y)-1);z=mp.tanh(mp.mpf(x)+mp.mpf(y)-1)
                oracle=(-30*t*(1-t*t)+4*z*(1-z*z))**2
                point=poisson_residual_integrand(network,((x,x),(y,y)))
                lo,hi=rational(point.lo),rational(point.hi)
                self.assertLessEqual(mp.mpf(str(lo.numerator))/int(lo.denominator),oracle)
                self.assertGreaterEqual(mp.mpf(str(hi.numerator))/int(hi.denominator),oracle)


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


class ComparisonTests(unittest.TestCase):
    def test_unintended_changes_rejected(self):
        from compare_verified_runs import compare_settings
        a={'network':'a','domain':[[0,1]],'theta':'7/20','bits':128}
        b={**a,'theta':'1/20'}
        with self.assertRaises(ValueError):compare_settings(a,b)
        self.assertIn('theta',compare_settings(a,b,vary=['theta']))
        with self.assertRaises(ValueError):compare_settings(a,{**b,'network':'b'},vary=['theta'])
        self.assertIn('network',compare_settings(a,{**a,'network':'b'},different_network=True))


class LegacyConfigurationTests(unittest.TestCase):
    def test_sample_and_static_counts(self):
        import ast
        from pathlib import Path
        import numpy as np
        root=Path(__file__).resolve().parents[1]
        for name in ['exp_sobolev_2d_disk_convergence.py','exp_combined_lp_sobolev_convergence.py']:
            tree=ast.parse((root/name).read_text());ns={'np':np}
            defs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('h','smooth_disk','generate_balanced_2d_coin_data')]
            exec(compile(ast.Module(body=defs,type_ignores=[]),name,'exec'),ns)
            x,y=ns['generate_balanced_2d_coin_data'](N=20,epsilon=.5)
            self.assertEqual(len(x),20);self.assertEqual(len(y),20);self.assertTrue(np.isfinite(y).all())
        tree=ast.parse((root/'exp_sobolev_2d_disk_heatmaps.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='compute_static_gap')
        ns={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),'static','exec'),ns)
        gap,q,count=ns['compute_static_gap'](None,-1.5,1.5,10,lambda *a,**k:(1.,2.))
        self.assertEqual(count,9);self.assertEqual(gap,13.5)



class PolicyValidationTests(unittest.TestCase):
    def test_status_claim_is_checked(self):
        from experiment_records import validate_policy
        a=Arithmetic(80);run=Integrator(a,lambda b:a.interval(*b[0]).abs_power(2),Box(((0,1),)))
        run.run(max_leaves=16);r=run.serialise()
        manifest={'domain':[[0,1]],'theta':'7/20','max_leaves':16,'tolerance':'0'}
        self.assertTrue(validate_policy(r,manifest,'adaptive'))
        bad=copy.deepcopy(r);bad['status']='tolerance_reached'
        with self.assertRaises(AssertionError):validate_policy(bad,manifest,'adaptive')


if __name__=='__main__':unittest.main()
