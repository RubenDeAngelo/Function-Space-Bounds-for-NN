import unittest,random,json
from fractions import Fraction as F
from backend import Network,Oracle,Q,run,children,moment,volume
from independent import expand,integral

def rec(w,b):return {'activation':'relu','weights':[[[str(x) for x in r] for r in m] for m in w],'biases':[[str(x) for x in r] for r in b]}
def hinge(d=2,a='1/3',normal=None):return rec([[normal or [1]+[0]*(d-1)],[[1]]],[[-F(a)],[1]])

def cases():
 return {
 'hinge':hinge(),
 'aligned':hinge(a='1/2'),
 'oblique':hinge(a='2/3',normal=[1,1]),
 'almost_parallel':rec([[[1,1],[1,'1001/1000']],[[1,-1]]],[['-1/3','-1/3'],[1]]),
 'close_planes':rec([[[1,0],[1,0]],[[1,-1]]],[['-1/3',str(-F(1,3)-F(1,1024))],[1]]),
 'cancellation':rec([[[1,1],[1,1]],[[1,-1]]],[['-1/3','-1/3'],[1]]),
 'zero_neuron':rec([[[0,0],[1,0]],[[3,1]]],[[0,'-1/3'],[1]]),
 'deep_bent':rec([[[1,0],[0,1]],[[1,-1],[-1,1]],[[1,1]]],[['-1/4','-1/3'],['-1/7','-1/9'],[1]])}

class Tests(unittest.TestCase):
 def test_independent_affine_moments(self):
  rng=random.Random(19)
  for _ in range(25):
   f=tuple(Q(rng.randint(-5,5),3) for _ in range(3));box=((Q(-1,2),Q(3,4)),(Q(1,3),Q(5,3)))
   n=rec([[[0,0]],[[0]]],[[0],[0]])
   pieces=[([(F(-1,2),F(1,3)),(F(3,4),F(1,3)),(F(3,4),F(5,3)),(F(-1,2),F(5,3))],[tuple(F(str(x)) for x in f)])]
   for h1 in [False,True]:self.assertEqual(F(str(moment(f,box,h1))),integral(pieces,box,h1))
 def test_independent_regions_and_every_leaf(self):
  for name,record in cases().items():
   pieces=expand(record,[(0,1),(0,1)])
   for q in ['L2','H1']:
    for rule in ['range','hybrid']:
     for policy in ['uniform','maxgap']:
      r=run(record,[(0,1),(0,1)],q,rule,policy,64,scan=True)
      truth=integral(pieces,[(0,1),(0,1)],q=='H1')
      self.assertLessEqual(F(r['final']['lower']),truth);self.assertGreaterEqual(F(r['final']['upper']),truth)
      for c in r['cells']:
       v=integral(pieces,c['box'],q=='H1')
       self.assertLessEqual(F(c['lower']),v);self.assertGreaterEqual(F(c['upper']),v)
       if c['resolved']:self.assertEqual(F(c['lower']),v);self.assertEqual(F(c['upper']),v)
 def test_random_deep_regions(self):
  rng=random.Random(141)
  for _ in range(10):
   widths=[2,3,2,1];w=[[[str(F(rng.randint(-3,3),2)) for _ in range(a)] for _ in range(b)] for a,b in zip(widths,widths[1:])];bs=[[str(F(rng.randint(-2,2),3)) for _ in range(b)] for b in widths[1:]]
   n={'activation':'relu','weights':w,'biases':bs};p=expand(n,[(0,1),(0,1)])
   for q in ['L2','H1']:
    r=run(n,[(0,1),(0,1)],q,'hybrid','maxgap',64,scan=True)
    for c in r['cells']:
     v=integral(p,c['box'],q=='H1');self.assertLessEqual(F(c['lower']),v);self.assertGreaterEqual(F(c['upper']),v)
 def test_slope_includes_zero(self):
  o=Oracle(hinge(),True,False)
  for box in [((Q(1,3),Q(2,3)),(Q(0),Q(1))),((Q(0),Q(1,3)),(Q(0),Q(1)))]:
   c=o.cell(box);self.assertGreaterEqual(c.gap,0)
 def test_stable_parent_descendants(self):
  for n in cases().values():
   o=Oracle(n,True,True)
   for box in children(((Q(0),Q(1)),)*2):
    p=o.cell(box)
    for c in children(box):o.cell(c,p)
 def test_known_hinge_integrals_2d_3d(self):
  for d in [2,3]:
   for q,truth in [('L2',Q(125,81)),('H1',Q(179,81))]:
    r=run(hinge(d),[(0,1)]*d,q,'hybrid','maxgap',64,scan=True)
    self.assertLessEqual(Q(r['final']['lower']),truth);self.assertGreaterEqual(Q(r['final']['upper']),truth)
 def test_positive_tolerance_and_zero(self):
  for policy in ['uniform','maxgap']:
   r=run(hinge(a='1/2'),[(0,1)]*2,'H1','hybrid',policy,64,'1/1000',scan=True)
   self.assertEqual(r['status'],'tolerance_reached');self.assertEqual(Q(r['final']['gap']),0)
 def test_exact_constant_and_degenerate(self):
  n=rec([[[0,0],[1,0]],[[1,0]]],[[0,0],[1]])
  r=run(n,[(0,1)]*2,'H1','hybrid','maxgap',64)
  self.assertEqual(r['status'],'exact');self.assertEqual(Q(r['final']['lower']),1)
 def test_inclusion_isotonic_fallback(self):
  for n in cases().values():
   o=Oracle(n,True,False);p=o.cell(((Q(0),Q(1)),)*2)
   for box in children(p.box):
    c=o.cell(box);self.assertLessEqual(p.fallback[0],c.fallback[0]);self.assertGreaterEqual(p.fallback[1],c.fallback[1])
 def test_matching_hinge_lower_bounds(self):
  for d in [2,3]:
   for q,k in [('L2',0),('H1',1)]:
    c=Q(2,3) if k==0 else Q(1)
    for policy in ['uniform','maxgap']:
     r=run(hinge(d),[(0,1)]*d,q,'hybrid',policy,64,scan=True)
     f=r['final'];g=Q(f['gap']);n=f['leaves'];power=d if policy=='uniform' else d-1
     self.assertGreaterEqual(g**power*n**(2-k),c**power)
 def test_deep_hinge_embedding(self):
  base=hinge()
  deep=rec([[[1,0],[0,0]],[[1,0],[0,0]],[[1,0],[0,0]],[[1,0]]],[['-1/3',0],[0,0],[0,0],[1]])
  for q in ['L2','H1']:
   for policy in ['uniform','maxgap']:
    a=run(base,[(0,1)]*2,q,'hybrid',policy,64);b=run(deep,[(0,1)]*2,q,'hybrid',policy,64)
    self.assertEqual([(h['lower'],h['upper']) for h in a['history']],[(h['lower'],h['upper']) for h in b['history']])
 def test_literal_lipschitz_enclosure_subclass(self):
  for d in [2,3]:
   for a in ['1/4','1/3','3/4']:
    row=[1]+[0]*(d-1)
    n=rec([[row,row],[[1,-1]]],[[-F(a),-F(a)],[1]])
    o=Oracle(n,False,True);root=o.cell(((Q(0),Q(1)),)*d);nodes=[root]
    for depth in range(3):
     nxt=[]
     for parent in nodes:
      pv=volume(parent.box);pf=(parent.lower/pv,parent.upper/pv)
      self.assertLessEqual(pf[0],1);self.assertGreaterEqual(pf[1],1)
      self.assertLessEqual(pf[1]-pf[0],4*max(u-l for l,u in parent.box))
      for box in children(parent.box):
       c=o.cell(box,parent);cv=volume(box);cf=(c.lower/cv,c.upper/cv)
       self.assertLessEqual(pf[0],cf[0]);self.assertGreaterEqual(pf[1],cf[1]);nxt.append(c)
     nodes=nxt
 def test_input_validation(self):
  with self.assertRaises(ValueError):Network({'activation':'tanh'})
  with self.assertRaises(ValueError):run(hinge(),[(0,1)]*2,'H2','range','maxgap',4)

if __name__=='__main__':unittest.main()
