"""Train/replay the compact paper suite. Render and validate without training.

All training/diagnostic arithmetic is ordinary float64. Only verified_* modules
compute certificates. Full exact records, source snapshots and checkpoints persist.
"""
import argparse
from fractions import Fraction
import json
import os
from pathlib import Path
import shutil
from time import perf_counter

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent/'.cache/matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(Path(__file__).resolve().parent/'.cache'))
from experiment_records import dump, sha, source_snapshot, validate_record
from verified_arithmetic import Arithmetic
from verified_network import VerifiedNetwork
from verified_integration import Box, Integrator
from verified_residual import poisson_residual_integrand


def model_new(architecture, seed):
    import torch
    from lib_nn_model import FlexibleNet
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return FlexibleNet(architecture, activation='tanh').double()


def save_model(folder, model, metadata, a):
    import torch
    folder.mkdir(parents=True, exist_ok=False)
    torch.save(model.state_dict(), folder/'checkpoint.pt')
    dump(folder/'network.json', VerifiedNetwork.from_torch(model, a).serialise())
    metadata.update({'checkpoint_sha256': sha(folder/'checkpoint.pt'),
                     'network_sha256': sha(folder/'network.json'), 'dtype': 'float64',
                     'activation': 'tanh', 'coefficient_semantics': 'stored binary64 values as exact real coefficients'})
    dump(folder/'model.json', metadata)


def load_model(folder, a):
    import torch
    info = json.loads((folder/'model.json').read_text())
    assert sha(folder/'checkpoint.pt') == info['checkpoint_sha256']
    assert sha(folder/'network.json') == info['network_sha256']
    model = model_new(info['architecture'], 0)
    model.load_state_dict(torch.load(folder/'checkpoint.pt', map_location='cpu', weights_only=True))
    for name,digest in info.get('data_sha256',{}).items():
        assert sha(folder/name)==digest
    network = VerifiedNetwork.from_torch(model, a)
    assert network.serialise() == json.loads((folder/'network.json').read_text())
    return network


def disk_target(x, epsilon):
    import torch
    r = torch.linalg.vector_norm(x, dim=1)
    y = torch.zeros_like(r); y[r <= 1] = 1
    mask = (r > 1) & (r < 1+epsilon)
    # Only evaluate the smooth transition inside its domain: no zero denominators.
    t = r[mask]-1
    y[mask] = torch.sigmoid(1/t - 1/(epsilon-t))
    return y[:,None]


def train_models(output, config, a):
    import numpy as np
    import torch
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    specs = []; start = perf_counter(); models = output/'models'; models.mkdir()
    c = config['one_d']
    x = torch.linspace(0,1,c['points'],dtype=torch.float64)[:,None]
    y = torch.exp(-100*(x-.5)**2)
    for seed in c['seeds']:
        model = model_new(c['architecture'], seed)
        for state in ('untrained','trained'):
            if state == 'trained':
                opt = torch.optim.Adam(model.parameters(), lr=c['lr'], weight_decay=c['weight_decay'])
                for _ in range(c['epochs']):
                    opt.zero_grad(); loss=(model(x)-y).square().mean(); loss.backward(); opt.step()
            model.eval(); name=f'one-d-seed{seed}-{state}'; folder=models/name
            grid=torch.linspace(0,1,1001,dtype=torch.float64)[:,None]
            with torch.no_grad(): err=model(grid)-torch.exp(-100*(grid-.5)**2)
            save_model(folder,model,{'architecture':c['architecture'],'seed':seed,'state':state,
                'training':c,'grid_max_abs_error':err.abs().max().item()},a)
            np.savez_compressed(folder/'training_data.npz',x=x.numpy(),y=y.numpy())
            for order in config['orders']:
                specs.append({'id':f'{name}-H{order}','model':name,'kind':'sobolev','order':order,
                              'domain':c['domain'],'max_leaves':c['max_leaves'],'group':'one_d','state':state,'seed':seed})
            print('Saved',name,flush=True)
    c=config['disk']; model=model_new(c['architecture'],c['seed'])
    rng=np.random.default_rng(c['seed']+1)
    x=torch.tensor(rng.uniform(-1.5,1.5,(c['points'],2)),dtype=torch.float64)
    y=disk_target(x,c['epsilon']); assert len(x)==c['points']
    opt=torch.optim.Adam(model.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
    for epoch in range(c['epochs']):
        opt.zero_grad(); loss=(model(x)-y).square().mean(); loss.backward();opt.step()
        if epoch%2000==0: print('disk',epoch,float(loss.detach()),flush=True)
    model.eval();line=torch.linspace(-1.5,1.5,161,dtype=torch.float64)
    gx,gy=torch.meshgrid(line,line,indexing='xy'); grid=torch.stack((gx.flatten(),gy.flatten()),1)
    with torch.no_grad(): pred=model(grid);truth=disk_target(grid,c['epsilon'])
    save_model(models/'disk',model,{'architecture':c['architecture'],'seed':c['seed'],'training':c,
        'grid_max_abs_error':(pred-truth).abs().max().item()},a)
    np.savez_compressed(models/'disk/training_data.npz',x=x.numpy(),y=y.numpy())
    np.savez_compressed(models/'disk/fit_grid.npz',x=gx.numpy(),y=gy.numpy(),target=truth.numpy().reshape(gx.shape),prediction=pred.numpy().reshape(gx.shape))
    for order in config['orders']:
        specs.append({'id':f'disk-H{order}','model':'disk','kind':'sobolev','order':order,
                      'domain':c['domain'],'max_leaves':c['max_leaves'],'group':'disk'})
    c=config['pinn']; model=model_new(c['architecture'],c['seed'])
    generator=torch.Generator(device='cpu').manual_seed(c['data_seed'])
    opt=torch.optim.Adam(model.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
    first_data=None
    for epoch in range(c['epochs']):
        opt.zero_grad()
        x=torch.rand((c['interior_points_per_epoch'],2),generator=generator,dtype=torch.float64).requires_grad_(True)
        value=model(x); g=torch.autograd.grad(value.sum(),x,create_graph=True)[0]
        lap=sum(torch.autograd.grad(g[:,i].sum(),x,create_graph=True)[0][:,i:i+1] for i in range(2))
        z=x.sum(dim=1,keepdim=True)-1;t=torch.tanh(z);f=-4*(1-t*t)*t
        b=c['boundary_points_per_edge_per_epoch'];r=torch.rand((b,1),generator=generator,dtype=torch.float64)
        s=torch.rand((b,1),generator=generator,dtype=torch.float64);zero=torch.zeros_like(r);one=torch.ones_like(r)
        boundary=torch.cat([torch.cat(v,1) for v in ((r,zero),(r,one),(zero,s),(one,s))])
        loss_r=(lap-f).square().mean();loss_b=(model(boundary)-torch.tanh(boundary.sum(1,keepdim=True)-1)).square().mean()
        loss=loss_r+loss_b;loss.backward();opt.step()
        if epoch==0: first_data=(x.detach().numpy().copy(),boundary.numpy().copy())
        if epoch%2000==0:print('pinn',epoch,float(loss.detach()),flush=True)
    model.eval();line=torch.linspace(0,1,101,dtype=torch.float64);gx,gy=torch.meshgrid(line,line,indexing='xy')
    grid=torch.stack((gx.flatten(),gy.flatten()),1).requires_grad_(True)
    pred=model(grid);grad=torch.autograd.grad(pred.sum(),grid,create_graph=True)[0]
    lap=sum(torch.autograd.grad(grad[:,i].sum(),grid,retain_graph=True)[0][:,i] for i in range(2))
    z=grid.sum(1)-1;t=torch.tanh(z);res=lap+4*(1-t*t)*t
    save_model(models/'pinn',model,{'architecture':c['architecture'],'seed':c['seed'],'training':c,
        'final_training_interior_MSE':float(loss_r.detach()),'final_training_boundary_MSE':float(loss_b.detach()),
        'diagnostic_grid_max_abs_residual':float(res.detach().abs().max()),
        'sampling':'fresh interior and boundary points each epoch; independent saved CPU generator state'},a)
    np.savez_compressed(models/'pinn/sampling.npz',first_interior=first_data[0],first_boundary=first_data[1],final_generator_state=generator.get_state().numpy())
    np.savez_compressed(models/'pinn/fit_grid.npz',x=gx.numpy(),y=gy.numpy(),target=torch.tanh(z).detach().numpy().reshape(gx.shape),prediction=pred.detach().numpy().reshape(gx.shape),residual=res.detach().numpy().reshape(gx.shape))
    specs.append({'id':'pinn-residual','model':'pinn','kind':'poisson_residual','domain':c['domain'],
                  'max_leaves':c['max_leaves'],'group':'pinn'})
    for folder in models.iterdir():
        info=json.loads((folder/'model.json').read_text())
        info['data_sha256']={p.name:sha(p) for p in sorted(folder.glob('*.npz'))}
        dump(folder/'model.json',info)
    dump(output/'experiments.json',specs)
    dump(output/'training_summary.json',{'seconds':perf_counter()-start,'threads':1,'device':'cpu','deterministic':True})
    return specs


def certify(output,config,specs):
    a=Arithmetic(config['bits']); runs=output/'runs';runs.mkdir()
    summaries=[]
    for spec in specs:
        net=load_model(output/'models'/spec['model'],a)
        enclosure=(lambda b:poisson_residual_integrand(net,b)) if spec['kind']=='poisson_residual' else (lambda b:net.sobolev_integrand(b,order=spec['order']))
        folder=runs/spec['id'];folder.mkdir()
        manifest={**spec,'theta':config['theta'],'bits':config['bits'],'tolerance':config['tolerance'],'p':2,
                  'enclosure_method':'natural interval AD with ancestor intersections',
                  'mixed_derivatives':'one per multi-index','network_sha256':sha(output/'models'/spec['model']/'network.json')}
        dump(folder/'manifest.json',manifest)
        for policy in ('adaptive','uniform'):
            run=Integrator(a,enclosure,Box(tuple(tuple(pair) for pair in spec['domain'])))
            run.run(policy=policy,theta=config['theta'],tolerance=config['tolerance'],max_leaves=spec['max_leaves'])
            record=run.serialise();validate_record(record,spec['domain']);dump(folder/f'{policy}.json',record)
            h=record['history'][-1]
            summaries.append({'id':spec['id'],'policy':policy,'status':record['status'],**h})
            print(spec['id'],policy,h['leaves'],float(Fraction(h['gap_upper'])),flush=True)
    dump(output/'summary.json',summaries)


def validate_suite(folder):
    env=json.loads((folder/'environment.json').read_text())
    for name,digest in env['source_sha256'].items():assert sha(folder/'sources'/name)==digest
    config=json.loads((folder/'config.json').read_text());a=Arithmetic(config['bits'])
    checks=[]
    for spec in json.loads((folder/'experiments.json').read_text()):
        net=load_model(folder/'models'/spec['model'],a)
        manifest=json.loads((folder/'runs'/spec['id']/'manifest.json').read_text())
        assert manifest['network_sha256']==sha(folder/'models'/spec['model']/'network.json')
        for policy in ('adaptive','uniform'):
            record=json.loads((folder/'runs'/spec['id']/f'{policy}.json').read_text())
            checks.append({'id':spec['id'],'policy':policy,**validate_record(record,spec['domain'])})
    dump(folder/'validation.json',checks)
    print('Validated',len(checks),'policy records',flush=True)


def render_suite(folder):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle
    specs=json.loads((folder/'experiments.json').read_text());figures=folder/'figures';figures.mkdir(exist_ok=True)
    colors={'adaptive':'#0072B2','uniform':'#D55E00'}
    def history(ax,spec):
        for policy in colors:
            r=json.loads((folder/'runs'/spec['id']/f'{policy}.json').read_text())
            ax.loglog([h['leaves'] for h in r['history']],[float(Fraction(h['gap_upper'])) for h in r['history']],
                color=colors[policy],linestyle='-' if policy=='adaptive' else '--',alpha=.8,
                label=policy.capitalize() if spec.get('seed',0)==0 else None)
        ax.set(xlabel='Leaf boxes $N$',ylabel='Integral enclosure gap $U-L$');ax.grid(alpha=.2,which='both')
    fig,axes=plt.subplots(2,2,figsize=(9,6.4),layout='constrained')
    for i,state in enumerate(('untrained','trained')):
        for j,order in enumerate((1,2)):
            for spec in specs:
                if spec['group']=='one_d' and spec['state']==state and spec['order']==order:history(axes[i,j],spec)
            axes[i,j].set_title(f'{state.capitalize()}: squared $H^{order}$ norm');axes[i,j].legend()
    for ext in ('png','pdf'):fig.savefig(figures/f'one_d.{ext}',dpi=200)
    plt.close(fig)
    for group in ('disk','pinn'):
        selected=[s for s in specs if s['group']==group]
        fig,axes=plt.subplots(len(selected),3,figsize=(12,3.8*len(selected)),layout='constrained',squeeze=False)
        for row,spec in zip(axes,selected):
            history(row[0],spec); row[0].legend()
            row[0].set_title('Squared residual integral' if group=='pinn' else f'Squared $H^{spec["order"]}$ norm')
            records=[json.loads((folder/'runs'/spec['id']/f'{p}.json').read_text()) for p in colors]
            positive=[float(Fraction(c['indicator'])) for r in records for c in r['cells'] if Fraction(c['indicator'])>0]
            lo,hi=min(positive),max(positive);norm=LogNorm(lo,max(hi,lo*1.001))
            for ax,policy,record in zip(row[1:],colors,records):
                patches=[];values=[]
                for cell in record['cells']:
                    (x0,x1),(y0,y1)=[[float(Fraction(x)) for x in pair] for pair in cell['bounds']]
                    patches.append(Rectangle((x0,y0),x1-x0,y1-y0));values.append(float(Fraction(cell['indicator'])))
                col=PatchCollection(patches,cmap='viridis',norm=norm,edgecolors='white',linewidths=.12)
                col.set_array(np.ma.masked_less_equal(values,0));ax.add_collection(col)
                b=spec['domain'];ax.set(xlim=b[0],ylim=b[1],xlabel='$x$',ylabel='$y$',title=f'{policy.capitalize()}: {len(patches)} boxes');ax.set_aspect('equal')
            fig.colorbar(col,ax=list(row[1:]),label='Local integral gap contribution',shrink=.8)
        for ext in ('png','pdf'):fig.savefig(figures/f'{group}.{ext}',dpi=200)
        plt.close(fig)
    # Explicitly approximate diagnostics, separate from enclosure-contribution maps.
    for group in ('disk','pinn'):
        data=np.load(folder/'models'/group/'fit_grid.npz');fig,axes=plt.subplots(1,3,figsize=(10,3),layout='constrained')
        zs=[data['target'],data['prediction'],abs(data['prediction']-data['target']) if group=='disk' else abs(data['residual'])]
        titles=['Target','Network','Sampled absolute fit error' if group=='disk' else 'Sampled absolute residual']
        for ax,z,title in zip(axes,zs,titles):
            im=ax.pcolormesh(data['x'],data['y'],z,shading='auto');ax.set(title=title,xlabel='$x$',ylabel='$y$');ax.set_aspect('equal');fig.colorbar(im,ax=ax)
        fig.savefig(figures/f'{group}_diagnostic.png',dpi=180);plt.close(fig)


def check_config(config):
    if not 0<Fraction(config['theta'])<1 or Fraction(config['tolerance'])<0:
        raise ValueError('Invalid tolerance/theta')
    if config['bits']<8 or not config['orders'] or any(x not in (1,2) for x in config['orders']):
        raise ValueError('Invalid precision or derivative orders')
    if config['one_d']['target']!='exp(-100*(x-0.5)^2)' or config['pinn']['target']!='tanh(x+y-1)':
        raise ValueError('This runner supports only the documented fixed targets')
    for group,dim,domain in [('one_d',1,[[0,1]]),('disk',2,[[-1.5,1.5],[-1.5,1.5]]),('pinn',2,[[0,1],[0,1]])]:
        c=config[group]
        if c['domain']!=domain or c['architecture'][0]!=dim or c['architecture'][-1]!=1:
            raise ValueError('Unsupported domain/architecture')
        for key in ('epochs','max_leaves'):
            if not isinstance(c[key],int) or c[key]<1:raise ValueError('Invalid budget')
        if c['lr']<=0 or c['weight_decay']<0:raise ValueError('Invalid training settings')
    if not config['one_d']['seeds'] or len(set(config['one_d']['seeds']))!=len(config['one_d']['seeds']):
        raise ValueError('Expected distinct seeds')
    if config['one_d']['points']<2 or config['disk']['points']<1 or config['disk']['epsilon']<=0:
        raise ValueError('Invalid training data settings')
    if min(config['pinn']['interior_points_per_epoch'],config['pinn']['boundary_points_per_edge_per_epoch'])<1:
        raise ValueError('Invalid PINN sampling')


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    for command in ('run','replay'):
        q=sub.add_parser(command);q.add_argument('--output',type=Path,required=True)
        if command=='run':q.add_argument('--config',type=Path,default=Path('experiments/paper.json'))
        else:q.add_argument('--input',type=Path,required=True)
    for command in ('render','validate'):
        q=sub.add_parser(command);q.add_argument('directory',type=Path)
    args=p.parse_args()
    if args.command in ('render','validate'):
        (render_suite if args.command=='render' else validate_suite)(args.directory);return
    config=json.loads((args.config if args.command=='run' else args.input/'config.json').read_text())
    check_config(config)
    args.output.mkdir(parents=True,exist_ok=False);dump(args.output/'config.json',config);source_snapshot(args.output)
    if args.command=='run':specs=train_models(args.output,config,Arithmetic(config['bits']))
    else:
        shutil.copytree(args.input/'models',args.output/'models');specs=json.loads((args.input/'experiments.json').read_text());dump(args.output/'experiments.json',specs)
    certify(args.output,config,specs);validate_suite(args.output);render_suite(args.output)


if __name__=='__main__':main()
