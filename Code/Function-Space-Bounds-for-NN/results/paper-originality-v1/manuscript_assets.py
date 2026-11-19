"""Render manuscript figure and tables from the immutable saved records."""
from pathlib import Path
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--latex',required=True,type=Path)
args=parser.parse_args()
args.latex.mkdir(parents=True,exist_ok=True)
(args.latex/'images').mkdir(exist_ok=True)
(args.latex/'tables').mkdir(exist_ok=True)
manuscript_root=args.latex.resolve()
from pathlib import Path
from fractions import Fraction as Q
import json,os,tempfile
ROOT=Path(__file__).resolve().parent
cache=Path(tempfile.gettempdir())/'codex-originality-plot-cache'
cache.mkdir(exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR',str(cache/'matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME',str(cache))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
rr=json.loads((ROOT/'results/records.json').read_text())
plt.rcParams.update({'font.size':12})
fig,axes=plt.subplots(2,2,figsize=(7.5,6.5),layout='constrained')
colors={'range':'#b46b1c','hybrid':'#176b78','midpoint':'#176b78'}
for ax,case,quantity,title,xmode in [
    (axes[0,0],'hinge-third','H1','Full H1: effect of local integration','leaves'),
    (axes[0,1],'trained-tanh-L2','L2','Trained tanh: attained tolerance','calls'),
    (axes[1,0],'normalized-hat-128-clipped','L2','Same hat: clipped representation','leaves'),
    (axes[1,1],'normalized-hat-128-cancellation','L2','Same hat: cancelling representation','leaves')]:
    for r in rr:
        if r['case']!=case or r['quantity']!=quantity:continue
        h=[a for a in r['history'] if Q(a['gap'])>0]
        xx=[a['leaves'] if xmode=='leaves' else a['counts']['cell_calls'] for a in h]
        yy=[float(Q(a['gap'])) for a in h]
        ax.plot(xx,yy,color=colors[r['method']],linestyle='-' if r['policy']=='maxgap' else '--',
                linewidth=1.6,label=r['method']+' / '+r['policy'])
        if r['status']=='exact':
            ax.annotate('hybrid: exact at '+str(r['final']['leaves'])+' leaves',
                        xy=(.03,.08),xycoords='axes fraction',fontsize=11,color=colors['hybrid'])
    if case=='hinge-third':ax.axhline(1e-6,color='gray',lw=.8,ls=':')
    if case=='trained-tanh-L2':ax.axhline(1e-3,color='gray',lw=.8,ls=':')
    ax.set_yscale('log');ax.set_title(title,fontsize=11)
    ax.set_xlabel('Leaf intervals' if xmode=='leaves' else 'Cell calls (derivative costs differ)')
    ax.set_ylabel('Full integral enclosure gap')
    ax.tick_params(labelsize=10);ax.grid(alpha=.2);ax.legend(fontsize=10,loc="upper right")
fig.suptitle('Controlled certification study — recorded states, no fitted rates',fontsize=11)
fig.savefig(manuscript_root/'images/controlled_certification.pdf')
print('Saved controlled_certification.pdf')

import re
rows=[]
for case,label,quantity in [('hinge-third','Hinge $H^1$','H1'),('three-hinges','Three hinges $H^1$','H1'),('trained-tanh-L2','Trained $L^2$','L2')]:
 for r in rr:
  if r['case']==case and r['quantity']==quantity and (r['method']!='range' or case!='three-hinges'):
   f=r['final'];method={'range':'Range','hybrid':'Hybrid','midpoint':'Midpoint'}[r['method']]
   value=f'{float(Q(f["gap"])):.6g}'
   value=re.sub(r'([\d.]+)e([+-]\d+)',lambda m:m[1]+r'\times10^{'+str(int(m[2]))+'}',value)
   rows.append(f"{label} & {method} & {r['policy']} & {f['leaves']} & ${value}$ & "+('Tol.' if r['status']=='tolerance_reached' else 'cap')+r' \\')
(manuscript_root/'tables/controlled_success.tex').write_text('\\begin{tabular}{@{}lllrrl@{}}\\toprule\nCase & Rule & Partition & Leaves & Gap & Status\\\\\\midrule\n'+'\n'.join(rows)+'\n\\bottomrule\\end{tabular}\n')
rows=[]
for n in [16,32,64,128,256]:
 vals=[]
 for graph in ['clipped','cancellation']:
  for policy in ['maxgap','uniform']:
   r=next(r for r in rr if r['case']==f'normalized-hat-{n}-{graph}' and r['method']=='range' and r['policy']==policy)
   v=Q(r['final']['gap']);vals.append(str(v.numerator) if v.denominator==1 else f'{float(v):.7g}')
 rows.append(str(n)+' & '+' & '.join('$'+x+'$' for x in vals)+r' \\')
(manuscript_root/'tables/hat_representation.tex').write_text('\\begin{tabular}{@{}rrrrr@{}}\\toprule\n$n$ & Clip/maxgap & Clip/uniform & Cancel/maxgap & Cancel/uniform\\\\\\midrule\n'+'\n'.join(rows)+'\n\\bottomrule\\end{tabular}\n')
