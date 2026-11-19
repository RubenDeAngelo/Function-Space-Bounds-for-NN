"""Render all active convergence panels from unchanged records, with applicable guides."""
from pathlib import Path
from fractions import Fraction as Q
import sys,os,json,csv,hashlib
sys.set_int_max_str_digits(0)
HERE=Path(__file__).resolve().parent;C=HERE.parents[1];P=C.parents[1];L=P/'LaTeX'
os.environ['MPLCONFIGDIR']='/private/tmp/slope-triangles-mpl'
os.environ['XDG_CACHE_HOME']='/private/tmp/slope-triangles-xdg'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rate_guides import GUIDES,power,exponential
plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.titlesize':10,'axes.labelsize':10,'legend.fontsize':9})
STYLE={'maxgap':('#2264a4','-','o'),'uniform':('#cf632e','--','s')}
INPUTS={};PANELS=[];ART=[]
def read(path,kind='json'):
 path=Path(path);INPUTS[str(path.relative_to(P))]=hashlib.sha256(path.read_bytes()).hexdigest()
 return json.loads(path.read_text()) if kind=='json' else list(csv.DictReader(path.open()))
def setup(ax,panel,title,semilog=False):
 ax.set_gid(panel);ax.set_xscale('linear' if semilog else 'log');ax.set_yscale('log')
 ax.set(title=title,xlabel='Leaf intervals $N$' if semilog or panel.startswith('control-') else 'Leaf boxes $N$',ylabel='Full integral gap')
 ax.grid(alpha=.2);PANELS.append({'panel':panel,'xscale':ax.get_xscale(),'yscale':'log'})
def curve(ax,rows,policy):
 color,ls,marker=STYLE[policy]
 rows=[r for r in rows if Q(r['gap'])>0];assert rows
 xs=[int(r['leaves']) for r in rows];ys=[float(Q(r['gap'])) for r in rows]
 ax.plot(xs,ys,color=color,ls=ls,lw=1.3,label=policy.title())
 idx=sorted(set(range(0,len(rows),max(1,len(rows)//14)))|{len(rows)-1})
 ax.plot([xs[i] for i in idx],[ys[i] for i in idx],ls='none',marker=marker,ms=3,color=color)
 PANELS[-1].setdefault('curves',[]).append({'policy':policy,'records':len(rows),'leaves':xs,'gaps_as_floats':ys})
def finish(fig,name):
 fig.canvas.draw()
 # Every axis has at least one validated theorem reference.
 for ax in fig.axes:assert any(g['panel']==ax.get_gid() for g in GUIDES),ax.get_gid()
 fig.savefig(L/'images'/name,metadata={'CreationDate':None,'ModDate':None})
 fig.savefig(HERE/'validation'/name.replace('.pdf','.png'),dpi=150)
 ART.append('images/'+name);plt.close(fig)
def hybrid_guides(ax,q,factor=4,orange_position=(.64,.55)):
 a=2 if q=='L2' else 1
 power(ax,a,STYLE['maxgap'][0],(.64,.88),factor,policy='maxgap',theorem='thm:deep_relu_separation')
 power(ax,a/2,STYLE['uniform'][0],orange_position,factor,policy='uniform',theorem='thm:deep_relu_separation')
# Figure 1 line panel. Composite is rebuilt from its existing LaTeX source.
rows=read(C/'experiments/smoothed_disk_figure1/data/history.csv','csv')
fig,ax=plt.subplots(figsize=(7.5,2.8),layout='constrained');setup(ax,'large-disk-L2','')
for pol in STYLE:
 data=[{'leaves':r['leaves'],'gap':r['gap_exact']} for r in rows if r['policy']==pol];curve(ax,data,pol)
ax.set_ylabel(r'Enclosure width of $\|v\|_{L^2}^2$');ax.legend(loc='lower left');hybrid_guides(ax,'L2',orange_position=(.27,.38));finish(fig,'disk_gap_revised.pdf')
# One-dimensional controls: exponential and algebraic rates use different axes.
records=read(C/'results/paper-originality-v1/results/records.json')
controls={(r['case'],r['quantity'],r['policy'],r['max_leaves']):r for r in records if r['method']=='hybrid' and r['precision_bits']==128}
fig,axs=plt.subplots(2,2,figsize=(7.4,5.5),layout='constrained')
for i,(case,title,m) in enumerate([('hinge-third','Hinge at 1/3',1),('three-hinges','Signed three-hinge spline',3)]):
 for j,pol in enumerate(STYLE):
  ax=axs[i,j];setup(ax,f'control-{case}-{pol}',title+': '+pol.title(),pol=='maxgap');ax.set_xlabel('Leaf intervals $N$')
  curve(ax,controls[case,'H1',pol,128]['history'],pol);ax.axhline(1e-6,color='.5',lw=.7,ls=':')
  if pol=='maxgap':exponential(ax,m,STYLE[pol][0])
  else:power(ax,1,STYLE[pol][0],(.57,.83),factor=4,policy=pol,theorem='thm:spline_separation')
finish(fig,'controlled_certification.pdf')
# Hinge control in two dimensions.
fig,axs=plt.subplots(1,2,figsize=(7.4,2.9),layout='constrained')
for ax,q in zip(axs,['L2','H1']):
 setup(ax,'hinge-2D-'+q,r'Non-dyadic hinge: squared $'+('L^2' if q=='L2' else 'H^1')+'$')
 for pol in STYLE:
  paths=list((C/'results/paper-geometry-v1/results').glob(f'control-hinge-{q}-hybrid-{pol}-*.json'));assert len(paths)==1
  r=read(paths[0]);curve(ax,r['history'],pol)
 ax.legend(loc='lower left');hybrid_guides(ax,q)
finish(fig,'geometry_controls_revised.pdf')
# Full relative-width histories on the frozen disk checkpoints.
fig,axs=plt.subplots(2,2,figsize=(7.4,5.6),layout='constrained')
for i,arch in enumerate(['deep','shallow200']):
 for j,q in enumerate(['L2','H1']):
  ax=axs[i,j];setup(ax,f'disk-{arch}-{q}',('Deep' if i==0 else 'Shallow')+r' disk: squared $'+('L^2' if q=='L2' else 'H^1')+'$')
  for pol in STYLE:
   data=read(C/f'experiments/relative_width_20261111/runs/trained-{arch}-disk-{q}-{pol}.csv','csv')
   for r in data:r['gap']=str(Q(r['upper'])-Q(r['lower']))
   curve(ax,data,pol)
  ax.legend(loc='lower left');hybrid_guides(ax,q,factor=16 if q=='H1' else 4,orange_position=(.22,.42) if q=='H1' else (.27,.38))
finish(fig,'disk_extended_certification.pdf')
# Smooth residuals: shared ideal-arithmetic Lipschitz-width benchmark.
fig,axs=plt.subplots(1,2,figsize=(7.4,2.9),layout='constrained')
for ax,case,title in zip(axs,['poisson','cdr'],['Poisson','Convection--diffusion--reaction']):
 setup(ax,'tanh-'+case,title);ax.set_ylabel('Full residual integral gap')
 for pol in STYLE:curve(ax,read(C/f'experiments/pde_revision_20261112/runs/{case}-{pol}.csv','csv'),pol)
 ax.axhline(1e-4,color='.4',ls=':',lw=.8,label='Requested gap');ax.legend(loc='upper right')
 power(ax,.5,'.3',(.16,.38),factor=16,policy='both',theorem='thm: dyadic rate; eq: uniform rate')
finish(fig,'tanh_pde_gaps.pdf')
# Hybrid hat trajectories have no applicable theorem rate: retain exact outcomes in a table.
lines=[]
for graph,label in [('clipped','Clipped'),('cancellation','Cancelling')]:
 for pol in STYLE:
  r=controls[f'normalized-hat-128-{graph}','L2',pol,128];f=r['final'];lo,hi,gap=map(Q,[f['lower'],f['upper'],f['gap']]);assert lo<=Q(1,3)<=hi and hi-lo==gap
  def tex(q):return str(q.numerator) if q.denominator==1 else rf'\frac{{{q.numerator}}}{{{q.denominator}}}'
  status='Exact' if r['status']=='exact' else 'Cap'
  lines.append(f"{label} & {pol.title()} & {f['leaves']} & {f['counts']['cell_calls']} & $[{tex(lo)},\\ {tex(hi)}]$ & {status} \\\\")
(L/'tables/representation_hybrid_summary.tex').write_text(r'\begin{tabular}{@{}llrrll@{}}\toprule'+'\n'+r'Graph & Policy & Leaves & Calls & Integral enclosure & Stop\\\midrule'+'\n'+'\n'.join(lines)+'\n'+r'\bottomrule\end{tabular}'+'\n')
assert len(PANELS)==13 and len(GUIDES)==20
(HERE/'validation/plot_metadata.json').write_text(json.dumps({'panels':PANELS,'guides':GUIDES,'input_sha256':INPUTS,'figure_outputs':ART,'triangle_count':len(GUIDES),'panel_count':len(PANELS)},indent=2)+'\n')
for path,sha in INPUTS.items():assert hashlib.sha256((P/path).read_bytes()).hexdigest()==sha
print('Rendered 13 convergence panels with 20 validated theorem-reference triangles; input hashes unchanged.')
