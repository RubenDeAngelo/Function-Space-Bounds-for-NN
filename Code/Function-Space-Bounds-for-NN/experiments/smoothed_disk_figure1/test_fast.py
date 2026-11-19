import unittest,random,json,time
from pathlib import Path
from backend import Oracle,Q,children,moment
from fast_oracle import FastOracle

def rec(w,b):return {'activation':'relu','weights':[[[str(v) for v in row] for row in a] for a in w],
                     'biases':[[str(v) for v in a] for a in b]}

class Tests(unittest.TestCase):
    def compare_paths(self,r,seed,count=12):
        old=Oracle(r);new=FastOracle(r);rng=random.Random(seed)
        for _ in range(count):
            box=((Q(-3,2),Q(3,2)),)*2;pa=pb=None
            for j in range(11):
                a=old.cell(box,pa);b=new.cell(box,pb)
                self.assertEqual((a.lower,a.upper,a.resolved,a.fallback),(b.lower,b.upper,b.resolved,b.fallback))
                pa,pb=a,b;box=children(box)[rng.randrange(4)]

    def test_frozen_checkpoint_paths(self):
        self.compare_paths(json.loads((Path(__file__).parent/'model/network.json').read_text()),713,30)

    def test_random_binary_deep_graphs(self):
        rng=random.Random(141)
        for _ in range(20):
            widths=[2,3,4,2,1]
            w=[[[Q(rng.randint(-4,4),4) for _ in range(a)] for _ in range(b)] for a,b in zip(widths,widths[1:])]
            bs=[[Q(rng.randint(-3,3),8) for _ in range(b)] for b in widths[1:]]
            self.compare_paths(rec(w,bs),rng.randrange(100000),6)

    def test_zero_cancellation_and_nondyadic_breakpoint(self):
        for r in [rec([[[0,0]],[[1]]],[[0],[0]]),
                  rec([[[3,0],[-3,0]],[[1,-1]]],[[-1,1],[0]]),
                  rec([[[1,1],[1,1]],[[1,-1]]],[[-1,-1],[1]])]:
            self.compare_paths(r,731,15)

    def test_affine_moments(self):
        rng=random.Random(19)
        for _ in range(30):
            a,b,c=(Q(rng.randint(-5,5),4) for _ in range(3))
            r=rec([[[a,b]]],[[c]]);o=FastOracle(r)
            j=rng.randrange(10);t=1<<j;i,k=rng.randrange(t),rng.randrange(t)
            box=((Q(-3,2)+Q(3*i,t),Q(-3,2)+Q(3*(i+1),t)),
                 (Q(-3,2)+Q(3*k,t),Q(-3,2)+Q(3*(k+1),t)))
            q=o.cell(box)
            self.assertEqual(q.lower,moment((c,a,b),box,False));self.assertEqual(q.lower,q.upper)

    def test_nondyadic_coefficients_rejected(self):
        with self.assertRaises(AssertionError):FastOracle(rec([[[Q(1,3),0]]],[[0]]))

if __name__=='__main__':unittest.main()
