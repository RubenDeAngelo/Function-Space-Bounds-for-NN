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
fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
colors={'range':'#b46b1c','hybrid':'#176b78','midpoint':'#176b78'}
for ax,case,quantity,title,xmode in [
    (axes[0,0],'hinge-third','H1','Full H1: local integration changes the rate','leaves'),
    (axes[0,1],'trained-tanh-L2','L2','Frozen trained tanh: attained tolerance','calls'),
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
                        xy=(.03,.08),xycoords='axes fraction',fontsize=9,color=colors['hybrid'])
    if case=='hinge-third':ax.axhline(1e-6,color='gray',lw=.8,ls=':')
    if case=='trained-tanh-L2':ax.axhline(1e-3,color='gray',lw=.8,ls=':')
    ax.set_yscale('log');ax.set_title(title,fontsize=11)
    ax.set_xlabel('Leaf intervals' if xmode=='leaves' else 'Cell calls (derivative costs differ)')
    ax.set_ylabel('Full integral enclosure gap')
    ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Controlled certification study — recorded states, no fitted rates',fontsize=14)
fig.savefig(ROOT/'results/controlled-comparison.png',dpi=180)
print('Saved controlled-comparison.png')
