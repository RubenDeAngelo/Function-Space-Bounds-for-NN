from pathlib import Path
from fractions import Fraction as F
from decimal import Decimal, localcontext, ROUND_FLOOR,ROUND_CEILING
from math import isqrt
import os,json,csv,sys,importlib.util
sys.set_int_max_str_digits(0)
E=Path(__file__).resolve().parent;C=E.parents[1];P=C.parents[1];L=P/'LaTeX'
os.environ['XDG_CACHE_HOME']='/private/tmp/pde-revision-xdg'
os.environ['MPLCONFIGDIR']='/private/tmp/pde-revision-mpl'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.titlesize':11,'axes.labelsize':10,'legend.fontsize':9})
styles={'maxgap':('#2264a4','-','o'),'uniform':('#cf632e','--','s')}
def plot(axis,rows,policy,x='leaves',y='gap'):
 color,ls,marker=styles[policy];axis.loglog([int(r[x]) for r in rows],[float(F(r[y])) for r in rows],color=color,ls=ls,lw=1.3,label=policy.title());indices=sorted(set(range(0,len(rows),max(1,len(rows)//16)))|{len(rows)-1});axis.plot([int(rows[i][x]) for i in indices],[float(F(rows[i][y])) for i in indices],ls='none',marker=marker,ms=3,color=color)
def finish(fig,name):fig.savefig(L/'images'/name,metadata={'CreationDate':None,'ModDate':None});plt.close(fig)
# Keep hinge controls separate, so trained curves can extend to their full histories.
fig,axs=plt.subplots(1,2,figsize=(7.4,2.9),layout='constrained')
for ax,q in zip(axs,['L2','H1']):
 for policy in styles:
  matches=list((C/'results/paper-geometry-v1/results').glob(f'control-hinge-{q}-hybrid-{policy}-*.json'))
  if not matches:matches=list((C/'results/paper-geometry-v1/results').glob(f'control-non_dyadic-{q}-hybrid-{policy}-*.json'))
  assert len(matches)==1,matches
  rec=json.loads(matches[0].read_text());plot(ax,[r for r in rec['history'] if F(r['gap'])>0],policy)
 ax.set(title=r'Non-dyadic hinge: squared $'+('L^2' if q=='L2' else 'H^1')+'$',xlabel='Leaf boxes $N$',ylabel='Full integral gap');ax.grid(alpha=.2);ax.legend()
finish(fig,'geometry_controls_revised.pdf')
fig,axs=plt.subplots(2,2,figsize=(7.4,5.6),layout='constrained')
for row,arch in enumerate(['deep','shallow200']):
 for col,q in enumerate(['L2','H1']):
  ax=axs[row,col]
  for policy in styles:
   rows=list(csv.DictReader((C/f'experiments/relative_width_20261111/runs/trained-{arch}-disk-{q}-{policy}.csv').open()))
   for r in rows:r['gap']=str(F(r['upper'])-F(r['lower']))
   plot(ax,rows,policy)
  ax.set(title=('Deep' if arch=='deep' else 'Shallow')+r' disk: squared $'+('L^2' if q=='L2' else 'H^1')+'$',xlabel='Leaf boxes $N$',ylabel='Full integral gap');ax.grid(alpha=.2);ax.legend()
finish(fig,'disk_extended_certification.pdf')
# Only the approved energy and unweighted boundary residual are extracted.
rows=list(csv.DictReader((E/'runs/corner-energy-boundary.csv').open()));last=rows[-1]
fig,axs=plt.subplots(2,1,figsize=(3.8,4.0),layout='constrained')
for ax,q,title in zip(axs,['J','B2'],['Neural PDE energy',r'Squared boundary residual']):
 ax.loglog([int(r['interior'])+int(r['boundary']) for r in rows],[float(F(r[q+'_hi'])-F(r[q+'_lo'])) for r in rows],'-o',color='#2264a4',ms=3)
 ax.set(title=title,xlabel='Total interior and boundary cells',ylabel='Full integral gap');ax.grid(alpha=.2)
finish(fig,'corner_energy_boundary_gaps.pdf')
def decimal(q,up,places=9):
 with localcontext() as ctx:
  ctx.prec=100
  return format((Decimal(q.numerator)/Decimal(q.denominator)).quantize(Decimal(10)**-places,rounding=ROUND_CEILING if up else ROUND_FLOOR),'f')
def root(q,up):
 scale=2**128;n=isqrt((q.numerator*scale*scale)//q.denominator)
 if up and n*n*q.denominator<q.numerator*scale*scale:n+=1
 return F(n,scale)
lo,hi=F(last['B2_lo']),F(last['B2_hi'])
lines=[]
for label,a,b in [('$J(v)$',F(last['J_lo']),F(last['J_hi'])),('$B_2(v)$',lo,hi),('$B_\\partial(v)$',root(lo,False),root(hi,True))]:
 assert F(decimal(a,False))<=a and F(decimal(b,True))>=b
 lines.append(f'{label} & $[{decimal(a,False)},\\ {decimal(b,True)}]$ \\\\')
(L/'tables/corner_certificates.tex').write_text(r'\begin{tabular}{@{}ll@{}}\toprule'+'\n'+r'Quantity & Certified interval\\\midrule'+'\n'+'\n'.join(lines)+'\n'+r'\bottomrule\end{tabular}'+'\n')
print('Figures and outward-rounded corner table generated.')
fig,axs=plt.subplots(1,2,figsize=(7.4,2.9),layout='constrained')
lines=[];summary=[]
for ax,case,title in zip(axs,['poisson','cdr'],['Poisson', 'Convection--diffusion--reaction']):
 for policy in styles:
  path=E/f'runs/{case}-{policy}.csv';rows=list(csv.DictReader(path.open()));plot(ax,rows,policy)
  last=rows[-1];lo,hi=F(last['lower']),F(last['upper']);assert hi-lo>F(1,10000)
  assert 'leaf_budget_exhausted' in Path(str(path)+'.meta').read_text()
  n=int(last['leaves']);calls=int(last['evaluations']);assert n==65536 and calls==87381
  a,b=('0' if lo==0 else decimal(lo,False)),decimal(hi,True);ra,rb=('0' if lo==0 else decimal(root(lo,False),False)),decimal(root(hi,True),True)
  assert F(a)<=lo<=hi<=F(b)
  assert F(ra)**2<=lo and F(rb)**2>=hi
  lines.append(f"{'Poisson' if case=='poisson' else 'CDR'} & {policy.title()} & {n:,} & {calls:,} & $[{a},\\ {b}]$ & $[{ra},\\ {rb}]$ \\\\")
  summary.append({'case':case,'policy':policy,'leaves':n,'evaluations':calls,'integral_interval':[a,b],'residual_norm_interval':[ra,rb],'stop':'leaf_budget_exhausted','seconds':last['seconds']})
 ax.axhline(1e-4,color='.4',ls=':',lw=.8,label='Requested gap')
 ax.set(title=title,xlabel='Leaf boxes $N$',ylabel='Full residual integral gap');ax.grid(alpha=.2);ax.legend()
finish(fig,'tanh_pde_gaps.pdf')
(L/'tables/tanh_pde_certificates.tex').write_text(r'\begin{tabular}{@{}llrrll@{}}\toprule'+'\n'+r'Case & Policy & $N$ & $E$ & $I$ & $\sqrt I$\\\midrule'+'\n'+'\n'.join(lines)+'\n'+r'\bottomrule\end{tabular}'+'\n')
(E/'runs/printed-residual-certificates.json').write_text(json.dumps(summary,indent=2)+'\n')
