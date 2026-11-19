"""Independent piecewise-affine expansion checks every ReLU leaf integral."""
from experiment import ROOT,Q,polynomial_integral
import json

def expand(record):
    pieces=[(Q(0),Q(1),[(Q(1),Q(0))])]
    for k,(weights,biases) in enumerate(zip(record['weights'],record['biases'])):
        new=[]
        for l,u,forms in pieces:
            pre=[]
            for row,bias in zip(weights,biases):
                aa=sum((Q(c)*a for c,(a,b) in zip(row,forms)),Q(0))
                bb=Q(bias)+sum((Q(c)*b for c,(a,b) in zip(row,forms)),Q(0))
                pre.append((aa,bb))
            if k==len(record['weights'])-1:new.append((l,u,pre));continue
            # Solve all affine zero equations explicitly, then classify open pieces.
            pts=sorted({l,u,*[-b/a for a,b in pre if a and l<-b/a<u]})
            for p,q in zip(pts,pts[1:]):
                t=(p+q)/2
                new.append((p,q,[(a,b) if a*t+b>0 else (Q(0),Q(0)) for a,b in pre]))
        pieces=new
    return pieces

def integrate(pieces,l,u,quantity):
    ans=Q(0)
    for a,b,forms in pieces:
        p,q=max(a,l),min(b,u)
        if p<q:ans+=polynomial_integral(forms,p,q,quantity)
    return ans

def main():
    records=json.loads((ROOT/'results/records.json').read_text())
    checks={'runs':len(records),'relu_leaves_checked':0,'relu_runs_checked':0,
            'partition_checks':0,'selection':'each decision independently scanned during execution',
            'known_integrals':'exact piecewise-affine expansion, independently locating all zero crossings'}
    for r in records:
        cells=r['cells'];last=Q(0);lo=hi=Q(0)
        for c in cells:
            assert Q(c['l'])==last and Q(c['l'])<Q(c['u'])
            last=Q(c['u']);lo+=Q(c['lower']);hi+=Q(c['upper'])
        assert last==1
        assert lo==Q(r['final']['lower']) and hi==Q(r['final']['upper'])
        assert hi-lo==Q(r['final']['gap'])
        assert r['final']['counts']['cell_calls']==2*len(cells)-1
        assert len(cells)==r['final']['leaves']
        if r['status']=='tolerance_reached':assert hi-lo<=Q(r['tolerance'])
        if r['status']=='exact':assert lo==hi
        checks['partition_checks']+=1
        if r['network']['activation']=='relu':
            pieces=expand(r['network'])
            exact=integrate(pieces,Q(0),Q(1),r['quantity'])
            assert str(exact)==r['exact_integral']
            for c in cells:
                truth=integrate(pieces,Q(c['l']),Q(c['u']),r['quantity'])
                assert Q(c['lower'])<=truth<=Q(c['upper']), (r['case'],c,truth)
                if c['method']=='exact_polynomial':assert Q(c['lower'])==truth==Q(c['upper'])
                checks['relu_leaves_checked']+=1
            checks['relu_runs_checked']+=1
    # All four independently generated smooth-network bounds have common intersection.
    for name in ['trained-tanh-L2','untrained-tanh-L2']:
        rr=[r for r in records if r['case']==name]
        assert max(Q(r['final']['lower']) for r in rr)<=min(Q(r['final']['upper']) for r in rr)
    checks['smooth_policy_interval_intersection_checks']=2
    checks['representation_lower_bound_checks']=0
    for r in records:
        if r['case'].startswith('normalized-hat-') and r['case'].endswith('-cancellation') and r['method']=='range':
            n=int(r['case'].split('-')[2])
            assert Q(r['final']['gap'])>=16*n**4*(1-Q(1,n*n))**3
            checks['representation_lower_bound_checks']+=1
    checks['rounding_subadditivity_violations']=sum(r['rounding_subadditivity_violations'] for r in records)
    checks['formal_verification']=False
    (ROOT/'results/validation.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
