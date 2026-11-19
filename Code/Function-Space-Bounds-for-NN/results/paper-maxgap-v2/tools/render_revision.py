"""Render manuscript assets from saved records; never interpolate certificates."""
from pathlib import Path
from fractions import Fraction as Q
import argparse, csv, json, math, os
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/maxgap-paper-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME', '/private/tmp/maxgap-paper-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm

parser = argparse.ArgumentParser()
parser.add_argument('--artifact', type=Path, default=Path(__file__).parent/'artifact')
parser.add_argument('--latex', type=Path, default=Path(__file__).parent/'LaTeX')
args = parser.parse_args()
root, latex = args.artifact, args.latex
images = latex/'images'
images.mkdir(parents=True, exist_ok=True)
(latex/'tables').mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 14, 'axes.titlesize': 14, 'axes.labelsize': 14,
                     'legend.fontsize': 14, 'xtick.labelsize': 14, 'ytick.labelsize': 14,
                     'pdf.fonttype': 42})
styles = {'maxgap': ('#0072B2', '-', 'Maxgap'),
          'uniform': ('#D55E00', '--', 'Uniform'),
          'dorfler': ('#009E73', ':', 'Dörfler')}
load = lambda p: json.loads(p.read_text())
cases = []
for spec in load(root/'tanh'/'experiments.json'):
    cases.append(dict(id=spec['id'], activation='tanh', domain=spec['domain'],
                      path=root/'tanh'/'runs'/spec['id']))
for arch in ['deep', 'shallow200']:
    for target in ['two-bumps', 'disk']:
        for norm in ['L2', 'H1']:
            path = root/'relu'/arch/target/norm
            cases.append(dict(id=f'relu-{arch}-{target}-{norm}', activation='relu',
                              arch=arch, target=target, norm=norm,
                              domain=load(path/'manifest.json')['domain'], path=path))
for c in cases:
    c['records'] = {p: load(c['path']/(p+'.json')) for p in styles}
    c['final'] = {p: dict(r['history'][-1], status=r['status']) for p, r in c['records'].items()}
    a, u = (c['final'][p] for p in ['maxgap', 'uniform'])
    assert a['leaves'] == u['leaves'] and a['evaluations'] == u['evaluations']
    assert a['status'] == u['status'] == 'leaf_budget_exhausted'
    c['reduction'] = 100*(1-Q(a['gap_upper'])/Q(u['gap_upper']))
byid = {c['id']: c for c in cases}

def history(ax, c, title, policies=('maxgap', 'uniform')):
    for p in policies:
        col, ls, label = styles[p]
        h = c['records'][p]['history']
        ax.loglog([v['evaluations'] for v in h], [float(Q(v['gap_upper'])) for v in h],
                  color=col, linestyle=ls, linewidth=1.5, label=label)
    ax.set(title=title, xlabel='Enclosure evaluations', ylabel=r'Integral gap $U-L$')
    ax.grid(alpha=.22)
    ax.legend()

def heatmaps(fig, axes, c):
    records = [c['records'][p] for p in ['maxgap', 'uniform']]
    positive = [float(Q(v['indicator'])) for r in records for v in r['cells'] if Q(v['indicator'])>0]
    norm = LogNorm(min(positive), max(positive))
    for ax, r, policy in zip(axes, records, ['maxgap', 'uniform']):
        patches, vals = [], []
        for cell in r['cells']:
            (x0,x1),(y0,y1) = [[float(Q(x)) for x in pair] for pair in cell['bounds']]
            patches.append(Rectangle((x0,y0),x1-x0,y1-y0))
            vals.append(float(Q(cell['indicator'])))
        pc = PatchCollection(patches, cmap='viridis', norm=norm, edgecolor='white', linewidth=.12)
        pc.set_array(np.ma.masked_less_equal(vals, 0))
        ax.add_collection(pc)
        ax.set(xlim=c['domain'][0], ylim=c['domain'][1], xlabel='x', ylabel='y',
               title=f"{styles[policy][2]}, N={len(r['cells'])}")
        ax.set_aspect('equal')
    fig.colorbar(pc, ax=list(axes), shrink=.75, label='Local integral gap')

def save(fig, name):
    fig.savefig(images/(name+'.pdf'), bbox_inches='tight')
    fig.savefig(images/(name+'.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

fig, axes = plt.subplots(1,3,figsize=(10.4,4.0),layout='constrained')
c = byid['two-bumps-L2']
history(axes[0], c, 'Two bumps: squared $L^2$')
heatmaps(fig, axes[1:], c)
save(fig, 'maxgap_two_bumps')
for order in [1,2]:
    fig, axes = plt.subplots(2,3,figsize=(9.4,6.4),layout='constrained')
    for i, state in enumerate(['untrained','trained']):
        for seed in range(3):
            history(axes[i,seed], byid[f'one-d-seed{seed}-{state}-H{order}'], f'{state.capitalize()}, seed {seed}')
    save(fig, f'maxgap_one_d_H{order}')
fig, axes = plt.subplots(2,3,figsize=(10.4,7.8),layout='constrained')
for i, order in enumerate([1,2]):
    c = byid[f'disk-H{order}']
    history(axes[i,0], c, f'Disk: squared $H^{order}$')
    heatmaps(fig, axes[i,1:], c)
save(fig, 'maxgap_disk')
fig, axes = plt.subplots(1,3,figsize=(10.4,4.0),layout='constrained')
c = byid['pinn-residual']
history(axes[0], c, 'Squared interior residual')
heatmaps(fig, axes[1:], c)
save(fig, 'maxgap_pinn')
for arch in ['deep','shallow200']:
    fig, axes = plt.subplots(2,2,figsize=(7.0,5.3),layout='constrained')
    for i, target in enumerate(['two-bumps','disk']):
        for j, norm in enumerate(['L2','H1']):
            history(axes[i,j], byid[f'relu-{arch}-{target}-{norm}'],
                    f"{'Two bumps' if target=='two-bumps' else 'Disk'}: squared ${'L^2' if norm=='L2' else 'H^1'}$")
    save(fig, f'maxgap_relu_{arch}')

# Supplemental curves retain the original bulk runs on the same networks.
for order in [1,2]:
    fig, axes = plt.subplots(2,3,figsize=(9.4,6.4),layout='constrained')
    for i,state in enumerate(['untrained','trained']):
        for seed in range(3):
            history(axes[i,seed], byid[f'one-d-seed{seed}-{state}-H{order}'],
                    f'{state.capitalize()}, seed {seed}', tuple(styles))
    save(fig, f'supp_one_d_H{order}')
fig, axes = plt.subplots(2,2,figsize=(7,5.3),layout='constrained')
for ax, name in zip(axes.flat, ['two-bumps-L2','disk-H1','disk-H2','pinn-residual']):
    history(ax, byid[name], name, tuple(styles))
save(fig, 'supp_two_d')
for arch in ['deep','shallow200']:
    fig, axes = plt.subplots(2,2,figsize=(7,5.3),layout='constrained')
    for i,target in enumerate(['two-bumps','disk']):
        for j,norm in enumerate(['L2','H1']):
            history(axes[i,j], byid[f'relu-{arch}-{target}-{norm}'], f'{target}, {norm}', tuple(styles))
    save(fig, f'supp_relu_{arch}')

def outward(q, upper=False, digits=4):
    q = Q(q)
    if not q: return '0'
    assert q>0
    exponent = len(str(q.numerator))-len(str(q.denominator))
    while q < Q(10)**exponent: exponent -= 1
    while q >= Q(10)**(exponent+1): exponent += 1
    step = Q(10)**(exponent-digits+1)
    k = math.ceil(q/step) if upper else math.floor(q/step)
    z = k*step
    # Keep decimal conversion exact; integer division controls every digit.
    places = max(0, digits-1-exponent)
    if -2<=exponent<=3:
        integer = int(z*10**places)
        return str(integer) if places==0 else str(integer//10**places)+'.'+str(integer%10**places).zfill(places)
    mant = z/Q(10)**exponent
    places=digits-1
    integer=int(mant*10**places)
    return str(integer//10**places)+'.'+str(integer%10**places).zfill(places)+r'\!\times\!10^{'+str(exponent)+'}'

def interval(r):
    lo,hi = Q(r['lower']),Q(r['upper'])
    a,b=outward(lo),outward(hi,True)
    return f'$[{a},{b}]$'

labels = {'two-bumps-L2':r'Two bumps $L^2$', 'disk-H1':r'Disk $H^1$',
          'disk-H2':r'Disk $H^2$', 'pinn-residual':'Interior residual'}
rows=[]
for name in labels:
    c=byid[name]
    for p in ['maxgap','uniform']:
        r=c['final'][p]
        rows.append(f"{labels[name]} & {styles[p][2]} & {r['leaves']:,} & {interval(r['integral'])} & {interval(r['norm'])} "+r'\\')
(latex/'tables'/'maxgap_endpoints.tex').write_text(
    r'\begin{tabular}{@{}llrll@{}}\toprule'+'\n'+
    r'Study & Policy & $N$ & Enclosure of $J$ & Enclosure of $\sqrt J$\\\midrule'+'\n'+
    '\n'.join(rows)+'\n'+r'\bottomrule\end{tabular}'+'\n')
rows=[]
for c in cases:
    if c['activation']!='relu': continue
    a,u=(c['final'][p] for p in ['maxgap','uniform'])
    rows.append(f"{'Deep' if c['arch']=='deep' else 'Shallow'} & {'Two bumps' if c['target']=='two-bumps' else 'Disk'} & ${'L^2' if c['norm']=='L2' else 'H^1'}$ & {float(Q(a['gap_upper'])):.6g} & {float(Q(u['gap_upper'])):.6g} & {float(c['reduction']):.2f}\\% "+r'\\')
(latex/'tables'/'relu_gaps.tex').write_text(
    r'\begin{tabular}{@{}lllrrr@{}}\toprule'+'\n'+
    r'Architecture & Target & Norm & Maxgap & Uniform & Reduction\\\midrule'+'\n'+
    '\n'.join(rows)+'\n'+r'\bottomrule\end{tabular}'+'\n')

flat=[]
for c in cases:
    for p,r in c['final'].items():
        flat.append(dict(id=c['id'],policy=p,leaves=r['leaves'],evaluations=r['evaluations'],
                         seconds=r['seconds'],status=r['status'],gap_upper=r['gap_upper'],
                         integral_lower=r['integral']['lower'],integral_upper=r['integral']['upper'],
                         norm_lower=r['norm']['lower'],norm_upper=r['norm']['upper']))
with (latex/'all_endpoints.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)

supp = [r'''\documentclass[11pt,a4paper]{article}
\usepackage[margin=22mm]{geometry}
\usepackage{amsmath,graphicx,booktabs,longtable,float,hyperref}
\title{Supplement: maxgap and bulk marking on frozen networks}
\author{}\date{}
\begin{document}\maketitle
The figures compare maxgap, uniform refinement and the original D\"orfler runs
with bulk fraction $7/20$. Every curve uses the same frozen network within a
panel. The horizontal axis counts cumulative enclosure evaluations, including
parents; the vertical axis is the full integral enclosure gap. Only recorded
certificates are plotted. Connecting segments guide the eye and do not certify
intermediate values. D\"orfler batches can stop below the leaf budget, so their
terminal values are not equal-work comparisons. All runs use 128-bit directed
MPFR arithmetic and zero requested stopping tolerance; all terminate at their
leaf budgets. Timings in the artifact are single-run observations, with different
history and marking overheads, and do not establish a runtime advantage.
''']
for name, caption in [
    ('supp_two_d',r'Tanh two-dimensional studies: two-bump $L^2$, disk $H^1/H^2$, and squared interior residual.'),
    ('supp_one_d_H1',r'Tanh squared $H^1$ norms: untrained (top) and trained (bottom), seeds 0--2.'),
    ('supp_one_d_H2',r'Tanh squared $H^2$ norms: untrained (top) and trained (bottom), seeds 0--2.'),
    ('supp_relu_deep',r'Deep ReLU networks $(2,32,32,32,1)$: two-bump and disk $L^2/H^1$ studies.'),
    ('supp_relu_shallow200',r'Shallow ReLU networks $(2,200,1)$: two-bump and disk $L^2/H^1$ studies.')]:
    supp.append(r'\begin{figure}[H]\centering\includegraphics[width=\textwidth]{images/'+name+r'.pdf}\caption{'+caption+r'}\end{figure}\clearpage')
supp += [r'''\section*{Terminal enclosures}
The following endpoints are rounded outward to four significant digits.
$J$ denotes the squared norm or squared interior residual. Exact rational
endpoints, upward-rounded gaps, work counts and timings for all three policies
are in \texttt{all\_endpoints.csv} in the accompanying artifact.
\small
\begin{longtable}{@{}lllrr@{}}
\toprule Case & Policy & $N$ & Enclosure of $J$ & Enclosure of $\sqrt J$\\\midrule\endhead
''']
for c in cases:
    name=c['id'].replace('one-d-seed','1D s').replace('-untrained-',r' U ').replace('-trained-',r' T ').replace('relu-shallow200-','ReLU S ').replace('relu-deep-','ReLU D ').replace('two-bumps','bumps').replace('pinn-residual','residual')
    for p in ['maxgap','uniform','dorfler']:
        r=c['final'][p]
        supp.append(f"{name} & {styles[p][2]} & {r['leaves']} & {interval(r['integral'])} & {interval(r['norm'])} "+r'\\')
supp += [r'\bottomrule\end{longtable}\end{document}']
(latex/'maxgap_supplement.tex').write_text('\n'.join(supp)+'\n')
print(f'Rendered {len(cases)} cases; exported {len(flat)} terminal records.')
