"""Exact rational ReLU certificates on dyadic boxes. Requires gmpy2.
All coefficients/geometry/moments/sums/comparisons are rational. No float
operation enters an endpoint. Floats below are only elapsed time/diagnostics.
"""
from dataclasses import dataclass
from collections import Counter
from itertools import product
import heapq,time
import gmpy2 as g
Q=g.mpq
ZERO=(Q(0),Q(0))
def add(a,b):return a[0]+b[0],a[1]+b[1]
def scale(c,a):return (c*a[0],c*a[1]) if c>=0 else (c*a[1],c*a[0])
def mul(a,b):
 v=[x*y for x in a for y in b];return min(v),max(v)
def square(a):return (Q(0) if a[0]<=0<=a[1] else min(a[0]**2,a[1]**2),max(a[0]**2,a[1]**2))
def isum(xs):
 lo=hi=Q(0)
 for a,b in xs:lo+=a;hi+=b
 return lo,hi
def volume(box):
 ans=Q(1)
 for l,u in box:ans*=u-l
 return ans
def children(box):return list(product(*[[(l,(l+u)/2),((l+u)/2,u)] for l,u in box]))
def bitlen(q):return max(abs(q.numerator).bit_length(),q.denominator.bit_length())

class Network:
 def __init__(self,record):
  if record['activation']!='relu':raise ValueError('ReLU only')
  self.w=[[[Q(x) for x in row] for row in w] for w in record['weights']]
  self.b=[[Q(x) for x in bb] for bb in record['biases']]
  self.d=len(self.w[0][0]);n=self.d
  if len(self.w)!=len(self.b):raise ValueError('Layer mismatch')
  for w,b in zip(self.w,self.b):
   if not w or len(w)!=len(b) or any(len(row)!=n for row in w):raise ValueError('Shape mismatch')
   n=len(w)
  if n!=1:raise ValueError('Scalar output required')
 def range(self,box,h1,counts):
  counts['range_network_calls']+=1
  vals=list(box);ders=[[(Q(int(i==j)),Q(int(i==j))) for j in range(self.d)] for i in range(self.d)] if h1 else None
  for k,(w,bs) in enumerate(zip(self.w,self.b)):
   vv=[];dd=[]
   for row,b in zip(w,bs):
    v=add((b,b),isum(scale(c,x) for c,x in zip(row,vals)))
    if h1:grad=[isum(scale(c,ds[j]) for c,ds in zip(row,ders)) for j in range(self.d)]
    counts['range_weight_terms']+=len(row)*(1+self.d*int(h1))
    if k<len(self.w)-1:
     slope=ZERO if v[1]<0 else ((Q(1),Q(1)) if v[0]>0 else (Q(0),Q(1)))
     v=max(Q(0),v[0]),max(Q(0),v[1])
     if h1:grad=[mul(slope,x) for x in grad]
    vv.append(v)
    if h1:dd.append(grad)
   vals=vv
   if h1:ders=dd
  f=square(vals[0])
  if h1:f=add(f,isum(square(x) for x in ders[0]))
  return f
 def affine(self,box,counts):
  counts['affine_cell_tests']+=1
  forms=[tuple([Q(0)]+[Q(int(i==j)) for j in range(self.d)]) for i in range(self.d)]
  for k,(w,bs) in enumerate(zip(self.w,self.b)):
   nxt=[]
   for row,b in zip(w,bs):
    f=tuple((b if j==0 else Q(0))+sum((c*x[j] for c,x in zip(row,forms)),Q(0)) for j in range(self.d+1))
    counts['affine_weight_terms']+=len(row)*(self.d+1)
    if k<len(self.w)-1:
     counts['neuron_sign_tests']+=1
     lo=hi=f[0]
     for c,(l,u) in zip(f[1:],box):lo+=c*(l if c>=0 else u);hi+=c*(u if c>=0 else l)
     if hi<=0:f=(Q(0),)*(self.d+1)
     elif lo<0:return None
    nxt.append(f)
   forms=nxt
  return forms[0]
 def point(self,x):
  z=list(x)
  for k,(w,b) in enumerate(zip(self.w,self.b)):
   z=[bb+sum((c*v for c,v in zip(row,z)),Q(0)) for row,bb in zip(w,b)]
   if k<len(self.w)-1:z=[max(Q(0),v) for v in z]
  return z[0]

def moment(f,box,h1):
 mid=[(l+u)/2 for l,u in box]
 mean=f[0]+sum((c*x for c,x in zip(f[1:],mid)),Q(0))
 val=mean**2+sum((c*c*(u-l)**2/12 for c,(l,u) in zip(f[1:],box)),Q(0))
 if h1:val+=sum((c*c for c in f[1:]),Q(0))
 return volume(box)*val

@dataclass
class Cell:
 box:tuple
 lower:object
 upper:object
 fallback:object
 resolved:bool
 @property
 def gap(self):return self.upper-self.lower
 def record(self):return {'box':[[str(x) for x in t] for t in self.box],'lower':str(self.lower),'upper':str(self.upper),'resolved':self.resolved}

class Oracle:
 def __init__(self,record,h1=False,hybrid=True):
  self.net=Network(record);self.h1=h1;self.hybrid=hybrid;self.counts=Counter()
 def cell(self,box,parent=None):
  self.counts['cell_calls']+=1
  f=self.net.affine(box,self.counts) if self.hybrid else None
  if f is not None:
   self.counts['moment_calls']+=1;q=moment(f,box,self.h1);v=Cell(box,q,q,None,True)
  else:
   r=self.net.range(box,self.h1,self.counts)
   if parent is not None and parent.fallback is not None:r=max(r[0],parent.fallback[0]),min(r[1],parent.fallback[1])
   if r[0]>r[1]:raise ArithmeticError('Empty range intersection')
   vol=volume(box);v=Cell(box,vol*r[0],vol*r[1],r,False)
  self.counts['max_endpoint_bits']=max(self.counts['max_endpoint_bits'],bitlen(v.lower),bitlen(v.upper))
  if parent is not None and parent.resolved:assert v.resolved
  return v

def run(record,domain,quantity,rule,policy,cap,tolerance='0',save_cells=True,scan=False):
 if quantity not in ['L2','H1'] or rule not in ['range','hybrid'] or policy not in ['uniform','maxgap']:raise ValueError('Bad policy')
 box=tuple(tuple(Q(x) for x in v) for v in domain);o=Oracle(record,quantity=='H1',rule=='hybrid')
 if len(box)!=o.net.d or any(l>=u for l,u in box):raise ValueError('Invalid domain')
 b=2**len(box);tol=Q(tolerance);t0=time.perf_counter();c=o.cell(box);active={0:c};heap=[(-c.gap,0)];uid=1
 lo,hi=c.lower,c.upper;hist=[];checks=0
 def snapshot():
  hist.append({'leaves':len(active),'lower':str(lo),'upper':str(hi),'gap':str(hi-lo),
               'resolved_leaves':sum(c.resolved for c in active.values()),'counts':dict(o.counts)})
 snapshot()
 while True:
  if hi-lo<=tol:status='tolerance_reached' if tol>0 else 'exact';break
  if len(active)+b-1>cap:status='leaf_budget_exhausted';break
  if policy=='maxgap':
   _,i=heapq.heappop(heap)
   if scan:assert i==min(active,key=lambda j:(-active[j].gap,j))
   selected=[i]
  else:
   if len(active)*b>cap:status='leaf_budget_exhausted';break
   selected=list(active)
  for i in selected:
   p=active.pop(i);new=[]
   for k in children(p.box):
    c=o.cell(k,p);new.append(c);active[uid]=c;heapq.heappush(heap,(-c.gap,uid));uid+=1
   assert sum((c.gap for c in new),Q(0))<=p.gap
   lo+=sum((c.lower for c in new),Q(0))-p.lower
   hi+=sum((c.upper for c in new),Q(0))-p.upper;checks+=1
  snapshot()
 assert lo==sum((c.lower for c in active.values()),Q(0)) and hi==sum((c.upper for c in active.values()),Q(0))
 assert o.counts['cell_calls']==1+b*(len(active)-1)//(b-1)
 assert sum((volume(c.box) for c in active.values()),Q(0))==volume(box)
 return {'quantity':quantity,'rule':rule,'policy':policy,'cap':cap,'tolerance':str(tol),'status':status,
         'seconds':time.perf_counter()-t0,'history':hist,'final':hist[-1],
         'subadditivity_checks':checks,'cells':[c.record() for c in active.values()] if save_cells else []}
