"""Standalone controlled study; frozen MPFR backend, rational geometry/aggregation.

Run with Python and gmpy2 installed. No paper-project files are imported or edited.
All endpoint strings are exact rationals; timings are observations, not rankings.
"""
from pathlib import Path
from fractions import Fraction as Q
from dataclasses import dataclass
from collections import Counter
import csv, hashlib, heapq, json, platform, sys, time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'backend'))
from verified_arithmetic import Arithmetic, rational
from verified_network import VerifiedNetwork
import gmpy2

def fq(x):
    return Q(str(rational(x)))

def isum(xs):
    lo = hi = Q(0)
    for a, b in xs:
        lo += a; hi += b
    return lo, hi

def iadd(a,b): return a[0]+b[0], a[1]+b[1]
def imul(a,b):
    v=[x*y for x in a for y in b]
    return min(v),max(v)
def iscale(c,a): return imul((c,c),a)
def isquare(a):
    return (Q(0) if a[0]<=0<=a[1] else min(a[0]**2,a[1]**2),max(a[0]**2,a[1]**2))

def exact_relu_range(record,l,u,quantity):
    """Independent exact rational interval propagation, including weak gradient."""
    jets=[((l,u),(Q(1),Q(1)))]
    for k,(ww,bb) in enumerate(zip(record['weights'],record['biases'])):
        out=[]
        for row,bias in zip(ww,bb):
            val=iadd(isum(iscale(Q(c),j[0]) for c,j in zip(row,jets)),(Q(bias),Q(bias)))
            der=isum(iscale(Q(c),j[1]) for c,j in zip(row,jets))
            if k<len(record['weights'])-1:
                slope=(Q(0),Q(0)) if val[1]<=0 else ((Q(1),Q(1)) if val[0]>=0 else (Q(0),Q(1)))
                val=max(Q(0),val[0]),max(Q(0),val[1])
                der=imul(slope,der)
            out.append((val,der))
        jets=out
    terms=[]
    for val,der in jets:
        if quantity!='semi':terms.append(isquare(val))
        if quantity!='L2':terms.append(isquare(der))
    return isum(terms)

def affine_on_cell(record,l,u,counts=None):
    """Exact, sufficient activation test; returns affine outputs or None."""
    forms=[(Q(1),Q(0))]
    for k,(ww,bb) in enumerate(zip(record['weights'],record['biases'])):
        out=[]
        for row,bias in zip(ww,bb):
            aa=sum((Q(c)*a for c,(a,b) in zip(row,forms)),Q(0))
            zz=Q(bias)+sum((Q(c)*b for c,(a,b) in zip(row,forms)),Q(0))
            if k<len(record['weights'])-1:
                if counts is not None:counts['neuron_sign_tests']+=1
                lo,hi=sorted((aa*l+zz,aa*u+zz))
                if hi<=0:aa=zz=Q(0)
                elif lo<0:return None
            out.append((aa,zz))
        forms=out
    return forms

def polynomial_integral(forms,l,u,quantity):
    result=Q(0)
    for a,b in forms:
        if quantity!='semi':
            result+=a*a*(u**3-l**3)/3+a*b*(u*u-l*l)+b*b*(u-l)
        if quantity!='L2':result+=a*a*(u-l)
    return result

def exact_piecewise_integral(record,knots,quantity):
    points=sorted({Q(0),Q(1),*(Q(x) for x in knots if 0<Q(x)<1)})
    total=Q(0)
    for l,u in zip(points,points[1:]):
        form=affine_on_cell(record,l,u)
        if form is None:raise AssertionError('supplied knots do not resolve network')
        total+=polynomial_integral(form,l,u,quantity)
    return total

def rec(weights,biases):
    return {'activation':'relu','weights':[[[str(c) for c in row] for row in w] for w in weights],
            'biases':[[str(b) for b in bb] for bb in biases]}

def spline(knots,coeff,b0=0,b1=0):
    return rec([[[1]]*(len(knots)+1),[[b1,*coeff]]],[[0,*[-Q(a) for a in knots]],[b0]])

def clipped_hat(n):
    aa=2*n*n
    return rec([[[aa],[-aa]],[[-1,-1]],[[n]]],[[-1,1],[1],[0]])

def cancellation_hat(n):
    aa=2*n*n
    return rec([[[aa],[aa],[aa]],[[n,-2*n,n]]],[[0,-1,-2],[0]])

@dataclass
class Cell:
    l: Q
    u: Q
    lower: Q
    upper: Q
    f_range: object
    method: str
    @property
    def gap(self):return self.upper-self.lower

class Oracle:
    def __init__(self,record,quantity,method,bits=128):
        self.record,self.quantity,self.method=record,quantity,method
        self.a=Arithmetic(bits)
        self.net=VerifiedNetwork.from_record(record,self.a)
        self.counts=Counter()

    def cell(self,l,u,parent=None):
        self.counts['cell_calls']+=1
        forms=None
        if self.method=='hybrid':
            self.counts['affine_cell_tests']+=1
            forms=affine_on_cell(self.record,l,u,self.counts)
            if forms is not None:
                self.counts['exact_polynomial_integrals']+=1
                q=polynomial_integral(forms,l,u,self.quantity)
                return Cell(l,u,q,q,None,'exact_polynomial')
        order=0 if self.quantity=='L2' else 1
        self.counts['network_order'+str(order)]+=1
        jets=self.net.evaluate([(l,u)],order=order)
        terms=[]
        for jet in jets:
            if self.quantity!='semi':terms.append(jet.value.abs_power(2))
            if self.quantity!='L2':terms.extend(x.abs_power(2) for x in jet.gradient)
        ff=self.a.sum(terms)
        if parent is not None and parent.f_range is not None:
            ff=ff.intersect(parent.f_range)
        ii=ff*self.a.interval(u-l)
        if self.method=='midpoint':
            if self.quantity!='L2' or self.record['activation']!='tanh':
                raise ValueError('midpoint study supports smooth L2 only')
            self.counts['network_order2']+=1
            jj=self.net.evaluate([(l,u)],order=2)[0]
            fsecond=2*jj.gradient[0].abs_power(2)+2*jj.value*jj.hessian[0]
            bound=max(abs(fq(fsecond.lo)),abs(fq(fsecond.hi)))
            self.counts['network_order0']+=1
            mid=(l+u)/2
            vv=self.net.evaluate([(mid,mid)],order=0)[0].value.abs_power(2)
            qq=vv*self.a.interval(u-l)
            error=self.a.interval(bound*(u-l)**3/24)
            candidate=self.a.raw(self.a.down.sub(qq.lo,error.hi),self.a.up.add(qq.hi,error.hi))
            ii=ii.intersect(candidate)
        self.counts['range_fallbacks']+=1
        return Cell(l,u,fq(ii.lo),fq(ii.hi),ff,self.method)

def run(record,quantity,method,policy,max_leaves=128,tolerance=None,bits=128):
    oracle=Oracle(record,quantity,method,bits)
    start=time.perf_counter()
    c=oracle.cell(Q(0),Q(1));active={0:c};heap=[(-c.gap,0)];next_id=1
    history=[];audits=[]
    def snapshot():
        lo=sum((x.lower for x in active.values()),Q(0));hi=sum((x.upper for x in active.values()),Q(0))
        h={'leaves':len(active),'lower':str(lo),'upper':str(hi),'gap':str(hi-lo),
           'counts':dict(oracle.counts),'seconds':time.perf_counter()-start}
        assert all(x.l<x.u and x.lower<=x.upper for x in active.values())
        history.append(h)
        return hi-lo
    gap=snapshot()
    while True:
        if tolerance is not None and gap<=tolerance:status='tolerance_reached';break
        if len(active)>=max_leaves:status='leaf_budget_exhausted';break
        if gap==0:status='exact';break
        if policy=='maxgap':
            neg,i=heapq.heappop(heap)
            assert -neg==active[i].gap
            # Independent scan verifies every marking decision, including ties.
            assert i==min(active,key=lambda k:(-active[k].gap,k))
            selected=[i]
        else:
            if 2*len(active)>max_leaves:status='leaf_budget_exhausted';break
            selected=list(active)
        for i in selected:
            parent=active.pop(i);mid=(parent.l+parent.u)/2;new=[]
            for l,u in [(parent.l,mid),(mid,parent.u)]:
                child=oracle.cell(l,u,parent);active[next_id]=child
                heapq.heappush(heap,(-child.gap,next_id));new.append(child);next_id+=1
            # Ranges/hybrid support the theorem's subadditivity, up to MPFR rounding.
            audits.append({'parent_gap':str(parent.gap),'child_gap':str(sum(x.gap for x in new)),
                           'parent_method':parent.method})
        gap=snapshot()
    lo,hi=Q(history[-1]['lower']),Q(history[-1]['upper'])
    rooted=oracle.a.interval(lo,hi).root(2)
    return {'quantity':quantity,'method':method,'policy':policy,'precision_bits':bits,
            'max_leaves':max_leaves,'tolerance':None if tolerance is None else str(tolerance),
            'status':status,'history':history,'final':history[-1],
            'norm_interval':rooted.serialise(),
            'method_leaves':dict(Counter(x.method for x in active.values())),
            'rounding_subadditivity_violations':sum(Q(x['child_gap'])>Q(x['parent_gap']) for x in audits),
            'cells':[{'l':str(x.l),'u':str(x.u),'lower':str(x.lower),'upper':str(x.upper),
                      'method':x.method} for x in sorted(active.values(),key=lambda c:c.l)]}

def main():
    results=ROOT/'results'
    if (results/'records.json').exists():raise SystemExit('Refusing to overwrite completed records; use a copied study directory.')
    records=[]
    def add(name,record,quantity,method,policy,budget,tol=None,exact=None,bits=128):
        r=run(record,quantity,method,policy,budget,tol,bits);r['case']=name;r['network']=record
        if exact is not None:
            assert Q(r['final']['lower'])<=exact<=Q(r['final']['upper'])
            r['exact_integral']=str(exact)
        records.append(r)
        return r
    # Fixed class, predefined offsets, signs, and clustered breakpoints.
    cases=[('hinge-third',spline([Q(1,3)],[1]),[Q(1,3)]),
           ('hinge-dyadic',spline([Q(3,8)],[1]),[Q(3,8)]),
           ('three-hinges',spline([Q(1,7),Q(2,5),Q(5,7)],[1,Q(-1,2),Q(1,3)],Q(1,5),Q(-1,4)),[Q(1,7),Q(2,5),Q(5,7)]),
           ('clustered-hinges',spline([Q(1,3),Q(1,3)+Q(1,1024),Q(1,3)+Q(1,512)],[1,-2,1]),[Q(1,3),Q(1,3)+Q(1,1024),Q(1,3)+Q(1,512)])]
    for name,net,knots in cases:
        for quantity in ['semi','H1']:
            exact=exact_piecewise_integral(net,knots,quantity)
            for method in ['range','hybrid']:
                for policy in ['maxgap','uniform']:
                    add(name,net,quantity,method,policy,128,Q(1,1000000),exact)
    # Same fixed integral; fixed architecture; two equivalent representations.
    for n in [16,32,64,128,256]:
        for label,net in [('clipped',clipped_hat(n)),('cancellation',cancellation_hat(n))]:
            for method in ['range','hybrid']:
                for policy in ['maxgap','uniform']:
                    add(f'normalized-hat-{n}-{label}',net,'L2',method,policy,n,exact=Q(1,3))
    # Affine control: no localized feature, full range oracle is exact.
    net=spline([],[],b1=1)
    for method in ['range','hybrid']:
        for policy in ['maxgap','uniform']:
            add('affine-control',net,'L2',method,policy,128,exact=Q(1,3))
    # Real frozen networks; independent validated midpoint comparator.
    for label in ['trained','untrained']:
        net=json.loads((ROOT/'backend'/f'{label}-tanh.json').read_text())
        for method in ['range','midpoint']:
            for policy in ['maxgap','uniform']:
                add(f'{label}-tanh-L2',net,'L2',method,policy,128,Q(1,1000))
    # Finite precision check where a one-hinge rational moment is not binary.
    net=spline([Q(1,3)],[1]);exact=exact_piecewise_integral(net,[Q(1,3)],'H1')
    for bits in [64,256]:add('precision-hinge',net,'H1','hybrid','maxgap',32,Q(1,1000000),exact,bits)
    results.mkdir(exist_ok=True)
    (results/'records.json').write_text(json.dumps(records,indent=2)+'\n')
    fields=['case','quantity','method','policy','precision_bits','status','leaves','lower','upper','gap','cell_calls','network_order0','network_order1','network_order2','affine_cell_tests','exact_polynomial_integrals','neuron_sign_tests','seconds']
    with (results/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in records:
            row={k:r[k] for k in fields if k in r};row.update({k:r['final'][k] for k in ['leaves','lower','upper','gap','seconds']})
            row.update({k:r['final']['counts'].get(k,0) for k in fields if k in ['cell_calls','network_order0','network_order1','network_order2','affine_cell_tests','exact_polynomial_integrals','neuron_sign_tests']})
            w.writerow(row)
    env={'python':sys.version,'platform':platform.platform(),'gmpy2':gmpy2.version(),'mpfr':gmpy2.mpfr_version(),
         'record_count':len(records),'aggregation':'exact rational sum of validated local integral endpoints',
         'rounding':'directed MPFR except exact rational affine-cell tests and polynomial moments',
         'command':'python experiment.py','no_training':True}
    (results/'environment.json').write_text(json.dumps(env,indent=2)+'\n')
    print('Completed',len(records),'runs; every analytically known integral enclosed.',flush=True)

if __name__=='__main__':main()
