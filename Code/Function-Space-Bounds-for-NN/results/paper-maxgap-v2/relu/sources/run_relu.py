from pathlib import Path
import json,sys,shutil
from fractions import Fraction
import numpy as np
import torch
from verified_arithmetic import Arithmetic
from verified_network import VerifiedNetwork
from verified_integration import Box,Integrator
from experiment_records import dump,sha,validate_record,validate_policy
from exp_verified_two_bumps import train,render,load_saved_network
from exp_verified_studies import disk_target

ROOT=Path(__file__).resolve().parent.parent
BASE=Path('/Users/moritz/Projekte/FunctionSpaceNormsOfNeuralNetworks_Research_Paper/Code/Function-Space-Bounds-for-NN')
torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
a=Arithmetic(128)
# Exact analytical ReLU checks: relu(x), its inactive region, and H2 rejection.
n=VerifiedNetwork(a,[[[1]],[[1]]],[[0],[0]],'relu')
for box,vr,gr in [([[-2,-1]],(0,0),(0,0)),([[1,2]],(1,2),(1,1)),([[-1,2]],(0,2),(0,1))]:
    j=n.evaluate(box,1)[0]
    assert (j.value.lo,j.value.hi)==vr and (j.gradient[0].lo,j.gradient[0].hi)==gr
try:n.evaluate([[-1,1]],2);raise AssertionError('H2 accepted')
except ValueError:pass
summary=[]
for name,domain,budget in [('two-bumps',[[-1,1],[-1,1]],4096),('disk',[[-1.5,1.5],[-1.5,1.5]],1024)]:
    folder=ROOT/name;folder.mkdir()
    if name=='two-bumps':
        model=train(folder,3000,20261020,(32,32,32),BASE/'results/two-bumps-deep-l2/training_data.npz',250,1.0)
        net=VerifiedNetwork.from_torch(model,a);dump(folder/'network.json',net.serialise())
        assert load_saved_network(folder,a).serialise()==net.serialise()
    else:
        from lib_nn_model import FlexibleNet
        torch.manual_seed(20261024);model=FlexibleNet([2,32,32,32,1],'relu').double()
        data=BASE/'results/validated-paper-suite-v1/models/disk/training_data.npz'
        with np.load(data) as d:x=torch.tensor(d['x']);y=torch.tensor(d['y'])
        assert torch.equal(y,disk_target(x,.5))
        opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.001)
        from time import perf_counter
        start=perf_counter()
        for epoch in range(10000):
            opt.zero_grad();loss=(model(x)-y).square().mean();loss.backward();opt.step()
            if epoch%2000==0:print('disk training',epoch,float(loss.detach()),flush=True)
        model.eval();line=np.linspace(-1.5,1.5,161);xx,yy=np.meshgrid(line,line)
        grid=torch.tensor(np.column_stack([xx.ravel(),yy.ravel()]))
        with torch.no_grad():prediction=model(grid);truth=disk_target(grid,.5);err=prediction-truth
        torch.save(model.state_dict(),folder/'checkpoint.pt');shutil.copy2(data,folder/'training_data.npz')
        np.savez_compressed(folder/'fit_grid.npz',x=xx,y=yy,target=truth.numpy().reshape(xx.shape),prediction=prediction.numpy().reshape(xx.shape))
        dump(folder/'training.json',{'activation':'relu','architecture':[2,32,32,32,1],'seed':20261024,'epochs':10000,'adam_lr':.001,'weight_decay':.001,'epsilon':.5,'points':5000,'training_data_source':str(data),'training_data_source_sha256':sha(data),'checkpoint_sha256':sha(folder/'checkpoint.pt'),'training_data_sha256':sha(folder/'training_data.npz'),'metrics':{'grid_MSE':float(err.square().mean()),'grid_max_abs_error':float(err.abs().max()),'training_MSE':float((model(x)-y).square().mean().detach()),'seconds':perf_counter()-start}})
        net=VerifiedNetwork.from_torch(model,a);dump(folder/'network.json',net.serialise())
        check=FlexibleNet([2,32,32,32,1],'relu').double();check.load_state_dict(torch.load(folder/'checkpoint.pt',weights_only=True));assert VerifiedNetwork.from_torch(check,a).serialise()==net.serialise()
    # Independent ordinary-autograd diagnostic at random points, away from kinks.
    rng=np.random.default_rng(728);points=rng.uniform(domain[0][0],domain[0][1],(50,2))
    for pt in points:
        t=torch.tensor(pt[None],requires_grad=True);v=model(t);g=torch.autograd.grad(v.sum(),t)[0]
        j=net.evaluate([(float(z),float(z)) for z in pt],1)[0]
        assert abs(float(j.value.lo)-float(v.detach()))<1e-12
        assert all(float(gi.lo)-1e-12<=float(g[0,k])<=float(gi.hi)+1e-12 for k,gi in enumerate(j.gradient))
    for norm,order in [('L2',0),('H1',1)]:
        runfolder=folder/norm;runfolder.mkdir();records={}
        manifest={'domain':domain,'theta':'7/20','tolerance':'0','max_leaves':budget,'bits':128,'p':2,'norm':norm,'order':order,'network_sha256':sha(folder/'network.json'),'method':'directed MPFR interval forward AD; ReLU weak gradient a.e.; ambiguous slopes [0,1]; ancestor intersections'}
        dump(runfolder/'manifest.json',manifest)
        for policy in ['adaptive','uniform']:
            run=Integrator(a,lambda b:net.sobolev_integrand(b,order=order),Box(tuple(map(tuple,domain))))
            run.run(policy=policy,theta='7/20',max_leaves=budget,tolerance='0')
            r=run.serialise();validate_record(r,domain);validate_policy(r,manifest,policy)
            dump(runfolder/(policy+'.json'),r);records[policy]=r
            h=r['history'][-1];summary.append({'experiment':name,'norm_name':norm,'policy':policy,'status':r['status'],**h})
            print(name,norm,policy,h['leaves'],h['seconds'],{k:float(Fraction(v)) for k,v in h['norm'].items()},flush=True)
        # renderer has fixed [-1,1] partition axes, so only use it for bumps.
        if name=='two-bumps':
            shutil.copy2(folder/'fit_grid.npz',runfolder/'fit_grid.npz');render(runfolder,records,norm)
        dump(ROOT/'summary.json',summary)
dump(ROOT/'validation.json',{'analytical_relu_tests':'passed','H2_rejected':True,'autograd_point_checks_per_model':50,'record_and_policy_checks':8,'checkpoint_exact_record_checks':'passed'})
