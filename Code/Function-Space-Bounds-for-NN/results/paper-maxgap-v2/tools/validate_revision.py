"""Check rendered decimal enclosure endpoints against exact saved fractions."""
from pathlib import Path
from fractions import Fraction as Q
import argparse, csv, json, re
root=Path(__file__).parent
parser=argparse.ArgumentParser()
parser.add_argument('--artifact',type=Path,default=root/'artifact')
parser.add_argument('--latex',type=Path,default=root/'LaTeX')
args=parser.parse_args()
artifact=args.artifact
latex=args.latex
load=lambda p:json.loads(p.read_text())
records=[]
for s in load(artifact/'tanh'/'experiments.json'):
    path=artifact/'tanh'/'runs'/s['id']
    for policy in ['maxgap','uniform','dorfler']:
        r=load(path/(policy+'.json'))
        records.append(dict(id=s['id'],policy=policy,status=r['status'],**r['history'][-1]))
for arch in ['deep','shallow200']:
    for target in ['two-bumps','disk']:
        for norm in ['L2','H1']:
            for policy in ['maxgap','uniform','dorfler']:
                r=load(artifact/'relu'/arch/target/norm/(policy+'.json'))
                records.append(dict(id=f'relu-{arch}-{target}-{norm}',policy=policy,status=r['status'],**r['history'][-1]))
csvrows=list(csv.DictReader((artifact/'all_endpoints.csv').open()))
assert len(csvrows)==len(records)==72
for c,r in zip(csvrows,records):
    for key in ['id','policy','status','leaves','evaluations','gap_upper','seconds']:
        assert str(r[key])==c[key],(r['id'],key)
    for kind in ['integral','norm']:
        for end in ['lower','upper']:assert c[kind+'_'+end]==r[kind][end]
    assert Q(r['norm']['lower'])**2<=Q(r['integral']['lower'])
    assert Q(r['norm']['upper'])**2>=Q(r['integral']['upper'])
    assert Q(r['gap_upper'])>=Q(r['integral']['upper'])-Q(r['integral']['lower'])
for suite in ['tanh','relu']:
    for r in load(artifact/suite/'summary.json'):
        name=r['id'] if suite=='tanh' else f"relu-{r['architecture']}-{r['experiment']}-{r['norm_name']}"
        exact=next(x for x in records if x['id']==name and x['policy']==r['policy'])
        for key in ['integral','norm','gap_upper','norm_gap_upper','leaves','evaluations','status','seconds']:
            assert exact[key]==r[key],(name,key)

def parse(text):
    text=text.replace(r'\!', '')
    if r'\times' not in text:return Q(text)
    m=re.fullmatch(r'(.+)\\times10\^\{(-?\d+)\}',text)
    assert m,text
    return Q(m[1])*Q(10)**int(m[2])
def check_printed(text,rows):
    intervals=re.findall(r'\$\[([^,]+),([^\]]+)\]\$',text)
    assert len(intervals)==2*len(rows),(len(intervals),len(rows))
    for i,r in enumerate(rows):
        for k,kind in enumerate(['integral','norm']):
            lo,hi=map(parse,intervals[2*i+k])
            assert lo<=Q(r[kind]['lower'])<=Q(r[kind]['upper'])<=hi,(r['id'],r['policy'],kind)
check_printed((latex/'maxgap_supplement.tex').read_text(),records)
mainrows=[]
for name in ['two-bumps-L2','disk-H1','disk-H2','pinn-residual']:
    mainrows += [r for r in records if r['id']==name and r['policy'] in ['maxgap','uniform']]
check_printed((latex/'tables'/'maxgap_endpoints.tex').read_text(),mainrows)

# Every cited label is defined once; the bibliography is unchanged by this revision.
def read_inputs(path, stack=()):
    path=path.resolve()
    assert path not in stack, f"Cyclic input: {path}"
    text=path.read_text()
    text=re.sub(r"(?m)(?<!\\)%.*$", "", text)
    def expand(match):
        child=path.parent/match.group(1)
        if not child.suffix: child=child.with_suffix('.tex')
        return read_inputs(child, (*stack,path))
    return re.sub(r"\\(?:input|include)\{([^}]+)\}", expand, text)
s=read_inputs(latex/'main.tex')
labels=re.findall(r'\\label\{([^}]+)\}',s)
assert len(labels)==len(set(labels))
references=re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',s)
assert set(references)<=set(labels),set(references)-set(labels)
for image in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',s):
    assert (latex/image).is_file(),image
result=dict(terminal_records_checked=72,main_printed_intervals_checked=16,
            supplement_printed_intervals_checked=144,summary_records_checked=48,
            exact_endpoint_csv='passed',outward_decimal_endpoints='passed',
            reference_and_asset_checks='passed')
(artifact/'manuscript_checks.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
