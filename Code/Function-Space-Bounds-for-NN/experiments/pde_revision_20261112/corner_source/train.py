"""Train Cartesian shallow ReLU networks using only PDE energy and soft boundary loss."""
import argparse, hashlib, json, math, time
from pathlib import Path
import numpy as np
import torch
import yaml
OMEGA=1.5*math.pi
AREA=0.5*OMEGA
PERIMETER=2+OMEGA

def sample(n,m,g):
    uv=torch.rand((n,2),generator=g)
    r=uv[:,0].sqrt(); phi=OMEGA*uv[:,1]
    x=torch.stack((r*phi.cos(),r*phi.sin()),1)
    forcing=(32/9)*torch.sin(math.pi*uv[:,1])
    uvb=torch.rand((m,2),generator=g)
    length=PERIMETER*uvb[:,0]; t=uvb[:,1]
    arc=torch.stack(((OMEGA*t).cos(),(OMEGA*t).sin()),1)
    edge0=torch.stack((t,torch.zeros_like(t)),1)
    edge1=torch.stack((torch.zeros_like(t),-t),1)
    y=torch.where((length<1)[:,None],edge0,torch.where((length<2)[:,None],edge1,arc))
    return x,forcing,y

def evaluate(x,w,b,a,b0):
    z=x@w.T+b
    v=torch.relu(z)@a+b0
    # Almost-everywhere input gradient, with parameter differentiation retained.
    dv=((z>0).to(z.dtype)*a)@w
    return v,dv

def train(seed,cfg):
    torch.manual_seed(seed)
    g=torch.Generator().manual_seed(100000+seed)
    w=torch.nn.Parameter(torch.randn(64,2)) # std sqrt(2/fan_in)=1
    b=torch.nn.Parameter(torch.zeros(64))
    a=torch.nn.Parameter(torch.empty(64).uniform_(-1/8,1/8))
    b0=torch.nn.Parameter(torch.zeros(()))
    opt=torch.optim.Adam([w,b,a,b0],lr=cfg['learning_rate'])
    history=[]; start=time.perf_counter()
    for it in range(cfg['updates']):
        if it==cfg['learning_rate_switch']:
            for group in opt.param_groups: group['lr']=cfg['learning_rate_final']
        x,f,y=sample(cfg['interior_batch'],cfg['boundary_batch'],g)
        v,dv=evaluate(x,w,b,a,b0)
        vb,_=evaluate(y,w,b,a,b0)
        energy=AREA*(0.5*dv.square().sum(1).double()-f.double()*v.double()).mean()
        boundary=PERIMETER*vb.double().square().mean()
        loss=energy+cfg['beta']/2*boundary
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
        if (it+1)%1000==0:
            row=dict(step=it+1,loss=loss.item(),energy=energy.item(),boundary=boundary.item(),seconds=time.perf_counter()-start)
            history.append(row);print('TRAIN',seed,json.dumps(row),flush=True)
    folder=Path('models')/f'seed{seed}';folder.mkdir(parents=True,exist_ok=True)
    arrays=[p.detach().cpu().numpy().astype('<f4') for p in [w,b,a,b0]]
    raw=b''.join(p.tobytes() for p in arrays)
    (folder/'weights.bin').write_bytes(raw)
    torch.save(dict(w=w.detach(),b=b.detach(),a=a.detach(),b0=b0.detach()),folder/'checkpoint.pt')
    # Separate, double precision validation; never used for optimization or model selection.
    gd=torch.Generator().manual_seed(900000+seed)
    wd,bd,ad,b0d=[p.detach().double() for p in [w,b,a,b0]]
    stats=np.zeros(5); count=0
    with torch.no_grad():
        for k in range(32):
            x,f,y=sample(8192,4096,gd);x=x.double();y=y.double()
            v,dv=evaluate(x,wd,bd,ad,b0d);vb,_=evaluate(y,wd,bd,ad,b0d)
            r=x.square().sum(1).sqrt();phi=torch.atan2(x[:,1],x[:,0]);phi=torch.where(phi<0,phi+2*math.pi,phi)
            exact=(r.pow(2/3)-r.square())*torch.sin(2*phi/3)
            ur=((2/3)*r.pow(-1/3)-2*r)*torch.sin(2*phi/3)
            up=(2/3)*(r.pow(2/3)-r.square())*torch.cos(2*phi/3)
            ux=phi.cos()*ur-phi.sin()*up/r;uy=phi.sin()*ur+phi.cos()*up/r
            graderr=(dv-torch.stack((ux,uy),1)).square().sum(1)
            force=(32/9)*torch.sin(2*phi/3)
            stats+=np.array([AREA*(0.5*dv.square().sum(1)-force*v).mean().item(),PERIMETER*vb.square().mean().item(),AREA*(v-exact).square().mean().item(),AREA*graderr.mean().item(),0])
            count+=1
    stats/=count
    manifest=dict(seed=seed,architecture=[2,64,1],sha256=hashlib.sha256(raw).hexdigest(),byte_order='little',layout=['w[64,2]','b[64]','a[64]','b0'],parameter_dtype='float32',training_seconds=time.perf_counter()-start,torch=torch.__version__,config=cfg,history=history,diagnostics_noncertified=dict(energy=stats[0],boundary_square=stats[1],l2_error_square=stats[2],gradient_error_square=stats[3],h1_error=math.sqrt(stats[2]+stats[3])))
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2));print('FINISHED',seed,json.dumps(manifest['diagnostics_noncertified']),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='config/sector_relu.yaml');p.add_argument('--seeds',type=int,nargs='+');args=p.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text());torch.set_num_threads(cfg['training_threads']);torch.set_num_interop_threads(1)
    for seed in args.seeds if args.seeds is not None else cfg['seeds']:train(seed,cfg)
