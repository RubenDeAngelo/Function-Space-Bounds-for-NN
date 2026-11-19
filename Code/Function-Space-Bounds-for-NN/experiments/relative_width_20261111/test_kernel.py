from pathlib import Path
from fractions import Fraction as F
import importlib.util,sys,random,json,csv,subprocess,os
HERE=Path(__file__).resolve().parent
REF=Path(os.environ.get('PAPER_CODE_ROOT',str(HERE.parents[1])))/'results/paper-geometry-v1'
def load(name):
 spec=importlib.util.spec_from_file_location(name,REF/f'{name}.py');m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
backend=load('backend');ind=load('independent')
# Import export without rerunning the trained-certificate validation.
source=(HERE/'prepare_validate.py').read_text().split('\nrecords=')[0]
ns={'__file__':str(HERE/'prepare_validate.py')};exec(source,ns);export=ns['export']
rng=random.Random(20261111)
passed=0
for n in range(20):
 widths=[2,3,2,1] if n%2 else [2,4,1]
 r={'activation':'relu','weights':[],'biases':[]}
 for a,b in zip(widths,widths[1:]):
  r['weights'].append([[str(F(rng.randrange(-4,5),4)) for _ in range(a)] for _ in range(b)])
  r['biases'].append([str(F(rng.randrange(-2,3),4)) for _ in range(b)])
 # Exercise identically zero and cancelling neurons.
 if n%5==0:r['weights'][0][0]=['0','0'];r['biases'][0][0]='0'
 domain=[['-3/2','3/2']]*2 if n%3 else [['-1','1']]*2
 model=HERE/'validation/random-model.txt';export(r,domain,model)
 pieces=ind.expand(r,domain)
 for q in ['L2','H1']:
  exact=ind.integral(pieces,domain,q=='H1')
  for p in ['uniform','maxgap']:
   cap=64;ref=backend.run(r,domain,q,'hybrid',p,cap,'0',save_cells=False)
   out=HERE/'validation/random.csv'
   run=subprocess.run([str(HERE/'kernel'),str(model),q,p,str(cap),str(out),'fixed'],capture_output=True,text=True)
   assert run.returncode==0,run.stderr
   result=list(csv.DictReader(out.open()))[-1];f=ref['final']
   assert F(result['lower'])==F(str(f['lower'])) and F(result['upper'])==F(str(f['upper'])),(n,q,p)
   assert F(result['lower'])<=exact<=F(result['upper']),(n,q,p,'independent integral')
   assert int(result['leaves'])==f['leaves'],(n,q,p,'leaves')
   passed+=1
print(f'{passed} randomized policy/quantity comparisons match reference endpoints and enclose independent polygon integrals.')
(HERE/'validation/randomized.json').write_text(json.dumps({'seed':20261111,'networks':20,'comparisons':passed,'independent_integrals':40,'all_passed':True},indent=2)+'\n')
