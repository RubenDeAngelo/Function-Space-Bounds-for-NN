from pathlib import Path
import torch,numpy as np
from lib_nn_model import FlexibleNet
from experiment_records import dump
root=Path(__file__).resolve().parent.parent
torch.set_num_threads(1);results={}
for name,lo,hi in [('two-bumps',-1,1),('disk',-1.5,1.5)]:
    model=FlexibleNet([2,32,32,32,1],'relu').double();model.load_state_dict(torch.load(root/name/'checkpoint.pt',weights_only=True));rows=[]
    for n in [256,512,1024]:
        axis=lo+(np.arange(n)+.5)*(hi-lo)/n;l2=h1=0.
        for chunk in np.array_split(axis,16):
            xx,yy=np.meshgrid(axis,chunk);x=torch.tensor(np.column_stack([xx.ravel(),yy.ravel()]),requires_grad=True)
            v=model(x);g=torch.autograd.grad(v.sum(),x)[0]
            l2+=float(v.square().sum().detach());h1+=float((v.square().flatten()+g.square().sum(1)).sum().detach())
        factor=((hi-lo)/n)**2;rows.append({'grid':n,'L2':(l2*factor)**.5,'H1':(h1*factor)**.5})
    results[name]=rows;print(name,rows,flush=True)
dump(root/'sampled_norms.json',results)
