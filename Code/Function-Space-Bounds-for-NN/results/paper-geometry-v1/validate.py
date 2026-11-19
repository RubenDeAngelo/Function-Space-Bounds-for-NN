"""Saved-record audit and independent polygon leaf checks for small controls."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import json,math
from independent import expand,integral
ROOT=Path(__file__).resolve().parent

def main():
 report=Counter();summary=[]
 for p in sorted((ROOT/'results').glob('*.json')):
  r=json.loads(p.read_text());f=r['final'];d=len(r['domain']);b=2**d
  cells=r['cells'];vol=F(0);lo=hi=F(0)
  for c in cells:
   v=F(1)
   for l,u in c['box']:assert F(l)<F(u);v*=F(u)-F(l)
   assert F(c['lower'])<=F(c['upper']);vol+=v;lo+=F(c['lower']);hi+=F(c['upper'])
  domainvol=F(1)
  for l,u in r['domain']:domainvol*=F(u)-F(l)
  assert vol==domainvol and str(lo)==f['lower'] and str(hi)==f['upper'] and str(hi-lo)==f['gap']
  keys=set()
  for c in cells:
   indices=[];depths=[]
   for (l,u),(a,z) in zip(c['box'],r['domain']):
    h=(F(u)-F(l))/(F(z)-F(a));inv=1/h
    assert inv.denominator==1 and inv.numerator>0 and inv.numerator&(inv.numerator-1)==0
    depth=inv.numerator.bit_length()-1;index=(F(l)-F(a))/(F(z)-F(a))/h
    assert index.denominator==1 and 0<=index<inv
    indices.append(int(index));depths.append(depth)
   assert len(set(depths))==1
   key=(depths[0],tuple(indices));assert key not in keys;keys.add(key)
  for depth,indices in keys:
   for level in range(depth):assert (level,tuple(i//(1<<(depth-level)) for i in indices)) not in keys
  report['dyadic_partition_checks']+=1
  assert len(cells)==f['leaves'];assert f['counts']['cell_calls']==1+b*(len(cells)-1)//(b-1)
  assert r['subadditivity_checks']==(len(cells)-1)//(b-1)
  assert all(F(a['gap'])>=F(z['gap']) for a,z in zip(r['history'],r['history'][1:]))
  if r['status']=='tolerance_reached':assert F(f['gap'])<=F(r['tolerance'])
  if r['status']=='exact':assert lo==hi
  if r['case'].startswith('control') and d==2:
   pieces=expand(r['network'],r['domain']);truth=integral(pieces,r['domain'],r['quantity']=='H1')
   assert lo<=truth<=hi
   for c in cells:
    t=integral(pieces,c['box'],r['quantity']=='H1');assert F(c['lower'])<=t<=F(c['upper'])
    if c['resolved']:assert F(c['lower'])==t==F(c['upper'])
    report['independent_2d_leaf_integrals']+=1
   report['independent_2d_runs']+=1
  report['runs']+=1;report[r['status']]+=1;report['leaves']+=len(cells)
  row={k:r[k] for k in ['case','quantity','rule','policy','cap','tolerance','status','seconds']}
  row.update(f)
  scale=1<<128
  def lower_root(x):return math.isqrt((x.numerator*scale*scale)//x.denominator)
  a=lower_root(lo);z=lower_root(hi)
  if F(z*z,scale*scale)<hi:z+=1
  row['norm_interval']={'lower':str(F(a,scale)),'upper':str(F(z,scale))}
  assert F(a,scale)**2<=lo and F(z,scale)**2>=hi
  summary.append(row)
 (ROOT/'validation.json').write_text(json.dumps(dict(report),indent=2)+'\n')
 (ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(json.dumps(dict(report),indent=2))
if __name__=='__main__':main()
