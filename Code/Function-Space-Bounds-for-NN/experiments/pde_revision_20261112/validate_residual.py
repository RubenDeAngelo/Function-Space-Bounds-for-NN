"""Independent high-precision differentiation, inclusion, and history checks.
Tests support the implementation; the certificate rests on directed interval inclusion.
"""
from pathlib import Path
from fractions import Fraction as F
import subprocess,json,random,csv
import mpmath as mp
E=Path(__file__).resolve().parent
mp.mp.dps=100
rng=random.Random(20261112)
def export(net,path):
 lines=[str(len(net['weights']))]
 for w,b in zip(net['weights'],net['biases']):
  lines.append(f'{len(w[0])} {len(w)}')
  lines.extend(' '.join(row+[bias]) for row,bias in zip(w,b))
 path.write_text('\n'.join(lines)+'\n')
def M(q):
 q=F(q);return mp.mpf(q.numerator)/q.denominator
def oracle(path,kind,boxes):
 p=subprocess.run([str(E/'residual_kernel'),str(path),kind,'oracle'],input='\n'.join(' '.join(map(str,b)) for b in boxes)+'\n',text=True,capture_output=True,check=True)
 vals=[tuple(map(F,s.split())) for s in p.stdout.splitlines()]
 assert len(vals)==len(boxes)
 return vals
def value(net,x,y):
 v=[x,y]
 for k,(w,b) in enumerate(zip(net['weights'],net['biases'])):
  v=[mp.fsum(M(c)*z for c,z in zip(row,v))+M(bias) for row,bias in zip(w,b)]
  if k<len(net['weights'])-1:v=list(map(mp.tanh,v))
 return v[0]
def reference(net,kind,x,y):
 v=lambda a,b:value(net,a,b)
 vx=mp.diff(lambda a:v(a,y),x);vy=mp.diff(lambda b:v(x,b),y)
 vxx=mp.diff(lambda a:v(a,y),x,2);vyy=mp.diff(lambda b:v(x,b),y,2)
 if kind=='poisson':r=vxx+vyy-2*mp.diff(mp.tanh,x+y-1,2)
 else:
  u=lambda a,b:mp.sin(a)*(1-mp.exp(-2*(1-a)))*b*b*(1-mp.exp(-3*(1-b)))
  ux=mp.diff(lambda a:u(a,y),x);uy=mp.diff(lambda b:u(x,b),y)
  lap=mp.diff(lambda a:u(a,y),x,2)+mp.diff(lambda b:u(x,b),y,2)
  f=-lap+(3-x)*ux+(4-y)*uy+u(x,y)
  r=-vxx-vyy+(3-x)*vx+(4-y)*vy+v(x,y)-f
 return r*r
small={'activation':'tanh','weights':[[['1','1']],[['1']]],'biases':[['-1'],['0']]}
(E/'validation').mkdir(exist_ok=True);export(small,E/'validation/exact-poisson.txt')
boxes=[]
for _ in range(20):
 d=rng.randint(1,6);den=2**d;x=F(rng.randrange(den),den);y=F(rng.randrange(den),den)
 boxes.append((x,x+F(1,den),y,y+F(1,den)))
for lo,hi in oracle(E/'validation/exact-poisson.txt','poisson',boxes):assert lo<=0<=hi
checks=0
for name,kind in [('poisson_model','poisson'),('cdr_model','cdr')]:
 net=json.loads((E/name/'network.json').read_text());export(net,E/name/'network.txt')
 testboxes=[(F(0),F(1),F(0),F(1)),(F(1,4),F(5,16),F(1,2),F(9,16))]
 for _ in range(4):
  x=F(rng.randrange(129),128);y=F(rng.randrange(129),128);testboxes.append((x,x,y,y))
 intervals=oracle(E/name/'network.txt',kind,testboxes)
 for box,(lo,hi) in zip(testboxes,intervals):
  assert 0<=lo<=hi
  # Center and two opposite corners; singleton tests constrain rounding-scale widths.
  for x,y in dict.fromkeys([(box[0],box[2]),(box[1],box[3]),((box[0]+box[1])/2,(box[2]+box[3])/2)]):
   r=reference(net,kind,M(x),M(y));assert M(lo)<=r<=M(hi),(name,box,lo,hi,mp.nstr(r,20));checks+=1
for path in (E/'runs').glob('*-*.csv'):
 if path.name.startswith('corner'):continue
 rows=list(csv.DictReader(path.open()));oldlo=F(0);oldhi=None
 for row in rows:
  n=int(row['leaves']);lo=F(row['lower']);hi=F(row['upper']);gap=F(row['gap'])
  assert 0<=lo<=hi and hi-lo==gap and int(row['evaluations'])==1+4*(n-1)//3
  assert lo>=oldlo and (oldhi is None or hi<=oldhi)
  oldlo,oldhi=lo,hi
result={'status':'passed','independent_100_decimal_digit_residual_values':checks,'exact_solution_zero_residual_boxes':len(boxes),'method':'mpmath numerical differentiation of the value-only network and manufactured solution; rational history checks','not_a_formal_software_verification':True}
(E/'validation/result.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
