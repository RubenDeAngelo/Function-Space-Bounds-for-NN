"""Fixed-protocol joint tanh training; no interval claims for training metrics."""
from pathlib import Path
import json,hashlib,time,sys,platform
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
OUT=HERE/'cdr_model'
if OUT.exists():raise SystemExit('Refusing to overwrite existing training output')
OUT.mkdir()
torch.set_num_threads(1);torch.set_default_dtype(torch.float64);torch.manual_seed(42)
torch.use_deterministic_algorithms(True)
class ShiftedHalton:
 def __init__(self,seed):self.index=1;self.shift=np.random.default_rng(seed).random(2)
 def random(self,n):
  out=np.empty((n,2))
  for j,base in enumerate((2,3)):
   for row,index in enumerate(range(self.index,self.index+n)):
    value=0.;scale=1./base
    while index:
     index,digit=divmod(index,base);value+=digit*scale;scale/=base
    out[row,j]=(value+self.shift[j])%1.
  self.index+=n
  return out
rng=ShiftedHalton(1042)
x=torch.tensor(rng.random(3500));bound=[]
for axis,side in [(0,0),(0,1),(1,0),(1,1)]:
 p=torch.tensor(rng.random(875));p[:,axis]=side;bound.append(p)
np.savez_compressed(OUT/'training_points.npz',interior=x.numpy(),boundary=torch.cat(bound).numpy())
# u=a(x)b(y), used analytically to define forcing, never as value labels.
def forcing(p):
 x,y=p[:,0:1],p[:,1:2];ex=torch.exp(-2*(1-x));ey=torch.exp(-3*(1-y))
 a=torch.sin(x)*(1-ex);ax=torch.cos(x)*(1-ex)-2*torch.sin(x)*ex
 axx=-torch.sin(x)*(1-ex)-4*torch.cos(x)*ex-4*torch.sin(x)*ex
 b=y*y*(1-ey);by=2*y*(1-ey)-3*y*y*ey;byy=2*(1-ey)-12*y*ey-9*y*y*ey
 return -(axx*b+a*byy)+(3-x)*ax*b+(4-y)*a*by+a*b
f=forcing(x).detach();x.requires_grad_(True)
model=torch.nn.Sequential(torch.nn.Linear(2,32),torch.nn.Tanh(),torch.nn.Linear(32,32),torch.nn.Tanh(),torch.nn.Linear(32,32),torch.nn.Tanh(),torch.nn.Linear(32,1))
opt=torch.optim.Adam(model.parameters(),lr=.001)
started=time.perf_counter();history=[]
for epoch in range(10000):
 opt.zero_grad(set_to_none=True);x.grad=None
 v=model(x);g=torch.autograd.grad(v.sum(),x,create_graph=True)[0]
 lap=sum(torch.autograd.grad(g[:,i].sum(),x,create_graph=True)[0][:,i:i+1] for i in range(2))
 r=-lap+(3-x[:,0:1])*g[:,0:1]+(4-x[:,1:2])*g[:,1:2]+v-f
 interior=r.square().mean();boundary=torch.tensor(0.)
 for j,b in enumerate(bound):
  if j==2:
   b=b.detach().requires_grad_(True);vb=model(b);gb=torch.autograd.grad(vb.sum(),b,create_graph=True)[0][:,1:2];boundary=boundary+gb.square().mean()
  else:boundary=boundary+model(b).square().mean()
 loss=interior+boundary;loss.backward();opt.step()
 if epoch%500==0 or epoch==9999:
  row={'step':epoch+1,'loss':loss.item(),'interior_MSE':interior.item(),'boundary_MSE_sum':boundary.item(),'seconds':time.perf_counter()-started};history.append(row);print(json.dumps(row),flush=True)
model.eval();torch.save(model.state_dict(),OUT/'checkpoint.pt')
def frac(v):
 n,d=float(v).as_integer_ratio();return f'{n}/{d}'
record={'activation':'tanh','weights':[],'biases':[]}
for layer in list(model)[::2]:
 record['weights'].append([[frac(v) for v in row] for row in layer.weight.detach().tolist()]);record['biases'].append([frac(v) for v in layer.bias.detach().tolist()])
(OUT/'network.json').write_text(json.dumps(record,indent=2)+'\n')
meta={'architecture':[2,32,32,32,1],'seed':42,'sampling_seed':1042,'dtype':'float64','device':'cpu','threads':1,'torch':str(torch.__version__),'numpy':np.__version__,'python':sys.version,'platform':platform.platform(),'steps':10000,'optimizer':'Adam','learning_rate':.001,'weight_decay':0,'interior_points':3500,'boundary_points_per_facet':875,'sampling':'one randomly shifted base-(2,3) Halton stream, fixed points; coordinate fixed for each facet','boundary_order':['x=0 Dirichlet','x=1 Dirichlet','y=0 outward Neumann','y=1 Dirichlet'],'epsilon':1,'training_scope':'PDE and boundary residuals only; no exact-solution value labels','history':history,'seconds':time.perf_counter()-started,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'network_sha256':hashlib.sha256((OUT/'network.json').read_bytes()).hexdigest()}
(OUT/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
