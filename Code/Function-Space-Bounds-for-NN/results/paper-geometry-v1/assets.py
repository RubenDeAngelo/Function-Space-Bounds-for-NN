from pathlib import Path
import json,argparse,os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/relu-geometry-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fractions import Fraction as Q
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--latex',required=True,type=Path);args=p.parse_args();latex=args.latex
rr=[json.loads(p.read_text()) for p in (ROOT/'results').glob('*.json')]
def find(case,q,rule,pol):return next(r for r in rr if r['case']==case and r['quantity']==q and r['rule']==rule and r['policy']==pol and r['cap']==1024)
rows=[]
for arch,label in [('deep','D'),('shallow200','S')]:
 for target,tlabel in [('two-bumps','Bumps'),('disk','Disk')]:
  for q in ['L2','H1']:
   r=[find(f'trained-{arch}-{target}',q,rule,pol) for rule,pol in [('range','uniform'),('range','maxgap'),('hybrid','uniform'),('hybrid','maxgap')]]
   vals=[f"{float(Q(x['final']['gap'])):.6g}" for x in r]
   norm='$L^2$' if q=='L2' else '$H^1$'
   rows.append(' & '.join([label,tlabel,norm,*vals,str(r[-1]['final']['resolved_leaves'])])+r' \\')
text=r'''\begin{tabular}{@{}lllrrrrr@{}}\toprule
Arch. & Target & Norm & Range/u & Range/a & Hybrid/u & Hybrid/a & Resolved/a\\\midrule
'''+ '\n'.join(rows)+ '\n'+r'\bottomrule\end{tabular}'+'\n'
(latex/'tables/geometry_gaps.tex').write_text(text)
plt.rcParams.update({'font.size':10,'axes.titlesize':10,'legend.fontsize':9,'pdf.fonttype':42})
fig,axes=plt.subplots(2,2,figsize=(7.2,5.8),layout='constrained')
for ax,(case,q,title) in zip(axes.flat,[('control-hinge','L2','Non-dyadic hinge: squared L2'),('control-hinge','H1','Non-dyadic hinge: squared H1'),('trained-deep-disk','L2','Frozen deep disk: squared L2'),('trained-shallow200-disk','H1','Frozen shallow disk: squared H1')]):
 for rule,color in [('range','#B46B1C'),('hybrid','#176B78')]:
  for pol,ls in [('uniform','--'),('maxgap','-')]:
   r=find(case,q,rule,pol);h=[x for x in r['history'] if Q(x['gap'])>0]
   ax.loglog([x['leaves'] for x in h],[float(Q(x['gap'])) for x in h],color=color,ls=ls,label=f'{rule}/{pol}')
 ax.set(title=title,xlabel='Leaf cubes',ylabel='Full integral gap');ax.grid(alpha=.2);ax.legend()
fig.savefig(latex/'images/geometry_certification.pdf');plt.close(fig)
print('Saved geometry figure and all-eight-case terminal table.')
