from pathlib import Path
import json
from fractions import Fraction
import numpy as np
import os
os.environ.setdefault("MPLCONFIGDIR",str(root_cache := Path(__file__).resolve().parent/".cache/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME",str(root_cache.parent))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent.parent
fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
for i,name in enumerate(['two-bumps','disk']):
    for j,norm in enumerate(['L2','H1']):
        ax=axes[i,j]
        for policy in ['adaptive','uniform']:
            r=json.loads((root/name/norm/(policy+'.json')).read_text())
            ax.loglog([h['leaves'] for h in r['history']],[float(Fraction(h['gap_upper'])) for h in r['history']],'o-',markersize=3,label=policy)
        ax.set(title=f'{name}, {norm}',xlabel='Leaf boxes',ylabel='Squared-norm integral gap U − L');ax.grid(alpha=.25);ax.legend()
fig.savefig(root/'convergence.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(2,3,figsize=(11,7),layout='constrained')
for i,name in enumerate(['two-bumps','disk']):
    with np.load(root/name/'fit_grid.npz') as d:
        for ax,z,title in zip(axes[i],[d['target'],d['prediction'],abs(d['target']-d['prediction'])],['Target','ReLU network','Absolute fit error']):
            im=ax.pcolormesh(d['x'],d['y'],z,shading='auto');ax.set(title=f'{name}: {title}',xlabel='x',ylabel='y');ax.set_aspect('equal');fig.colorbar(im,ax=ax)
fig.savefig(root/'fits.png',dpi=180)
