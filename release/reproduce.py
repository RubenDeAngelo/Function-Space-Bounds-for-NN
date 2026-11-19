"""Reproduce paper evidence in a disposable copy, preserving released records."""
from pathlib import Path
import argparse,csv,datetime,hashlib,json,os,shlex,shutil,subprocess,sys,time
from fractions import Fraction

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checks',action='store_true',help='Build kernels and run tests, selected replays and record audits')
p.add_argument('--build-kernels',action='store_true')
p.add_argument('--render',action='store_true',help='Regenerate current figures and tables from saved records')
p.add_argument('--build-paper',action='store_true',help='Build both manuscript PDFs in the working copy')
p.add_argument('--paper-source',type=Path,help='Private manuscript LaTeX directory, needed only for --build-paper')
p.add_argument('--full-pde',action='store_true',help='Replay all four 65536-leaf PDE trajectories')
p.add_argument('--geometry-replay',action='store_true',help='Replay all 88 current hybrid geometry records')
args=p.parse_args()
if not any(getattr(args,k) for k in ['checks','build_kernels','render','build_paper','full_pde','geometry_replay']):p.error('Choose --checks, --render, --build-paper or a replay option')
if args.build_paper and (args.paper_source is None or not (args.paper_source/'main.tex').is_file()):p.error('--build-paper requires --paper-source /path/to/the/private/LaTeX-directory')
subprocess.run([sys.executable,str(ROOT/'release/verify.py')],check=True)
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
work=ROOT/'.release-work'/stamp;work.mkdir(parents=True)
for name in ['Code','LaTeX','release']:
    shutil.copytree(ROOT/name,work/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
C=work/'Code/Function-Space-Bounds-for-NN';L=work/'LaTeX'
if args.paper_source:
    # Keep the draft manuscript local; copy build inputs without changing it.
    shutil.copytree(args.paper_source,L,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.aux','*.log','*.fls','*.fdb_latexmk','*.synctex.gz'))
env=os.environ.copy();env.update({'PYTHONDONTWRITEBYTECODE':'1','MPLCONFIGDIR':str(work/'mpl-cache'),'XDG_CACHE_HOME':str(work/'xdg-cache'),'PAPER_CODE_ROOT':str(C),'PAPER_LATEX_ROOT':str(L)})
env.pop('PYTHONPATH',None)
report={'started_utc':stamp,'python':sys.version,'steps':[],'status':'running'}
def run(label,cmd,cwd=None):
    print(label,flush=True);start=time.monotonic()
    dest=work/(f'{len(report["steps"])+1:02d}-'+label+'.log')
    with dest.open('w') as f:
        proc=subprocess.run([str(x) for x in cmd],cwd=cwd or work,env=env,stdout=f,stderr=subprocess.STDOUT)
    step={'name':label,'command':[str(x) for x in cmd],'exit_code':proc.returncode,'seconds':time.monotonic()-start,'log':dest.name}
    report['steps'].append(step);(work/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    if proc.returncode:
        print(dest.read_text()[-12000:]);raise SystemExit(f'{label} failed; see {dest}')
    print('  passed:',dest.read_text()[-1400:].strip(),flush=True)
def py(label,script,cwd=None,*extra):run(label,[sys.executable,'-B',script,*extra],cwd)
def compile_kernels():
    cxx=shlex.split(env.get('CXX','clang++' if shutil.which('clang++') else 'g++'))
    includes=[];libs=[]
    for var in ['GMP_PREFIX','MPFR_PREFIX']:
        prefix=env.get(var)
        if prefix:
            includes.append('-I'+str(Path(prefix)/'include'));libs.append('-L'+str(Path(prefix)/'lib'))
    if not includes and shutil.which('pkg-config'):
        info=subprocess.run(['pkg-config','--cflags','--libs','gmp','mpfr'],capture_output=True,text=True)
        if info.returncode==0:
            for flag in shlex.split(info.stdout):
                (includes if flag.startswith('-I') else libs).append(flag)
    flags=['-O3','-std=c++17',*includes]
    rel=C/'experiments/relative_width_20261111';pde=C/'experiments/pde_revision_20261112'
    run('compile-relative-width',[*cxx,*flags,rel/'kernel.cpp',*libs,'-lgmpxx','-lgmp','-o',rel/'kernel'])
    run('compile-pde',[*cxx,*flags,pde/'residual_kernel.cpp',*libs,'-lmpfr','-lgmpxx','-lgmp','-o',pde/'residual_kernel'])
    compiler_version=subprocess.check_output([*cxx,'--version'],text=True)
    strict=['-ffp-model=strict','-ffp-contract=off'] if 'clang' in compiler_version.lower() else ['-frounding-math','-fno-fast-math','-ffp-contract=off']
    run('compile-corner',[*cxx,*flags,*strict,pde/'corner_source/main.cpp',*libs,'-lmpfr','-lgmpxx','-lgmp','-o',pde/'corner_source/verify_rebuilt'])
    run('corner-arithmetic-selftest',[pde/'corner_source/verify_rebuilt','--selftest'])

if args.checks or args.build_kernels or args.full_pde:compile_kernels()
if args.checks:
    # Check immutable original manifests before scripts refresh scratch-copy ledgers.
    py('guide-and-frozen-data-audit',C/'experiments/slope_triangles_20261112/validate.py')
    for package in ['paper-originality-v1','paper-geometry-v1']:
        folder=C/'results'/package
        run(package+'-tests',[sys.executable,'-B','-m','unittest','-v','test_study'],folder)
        py(package+'-independent-audit',folder/'validate.py',folder)
    py('replay-40-hybrid-controls',C/'experiments/hybrid_submission/replay.py',None,'--suite','controls','--output',str(work/'replay-controls'))
    disk=C/'experiments/smoothed_disk_figure1'
    run('disk-kernel-tests',[sys.executable,'-B','-m','unittest','-v','test_fast'],disk)
    py('disk-partition-audit',disk/'validate.py',disk)
    rel=C/'experiments/relative_width_20261111'
    py('replay-16-trained-certificates',rel/'prepare_validate.py',rel)
    py('randomized-independent-kernel-tests',rel/'test_kernel.py',rel)
    pde=C/'experiments/pde_revision_20261112'
    py('independent-pde-residual-tests',pde/'validate_residual.py',pde)
    py('replay-four-pde-checkpoints',pde/'replay.py',pde)
    # Re-extract every imported J/B2 endpoint from original hexadecimal records.
    raw=[json.loads(line) for line in (pde/'corner_source/source_run.jsonl').read_text().splitlines() if line.strip()]
    saved=list(csv.DictReader((pde/'runs/corner-energy-boundary.csv').open()))
    assert len(raw)==len(saved)
    for a,b in zip(raw,saved):
        for key in ['steps','interior','boundary']:assert a[key]==int(b[key])
        assert a['metrics']['evaluations']==int(b['evaluations'])
        for quantity in ['J','B2']:
            for side in ['lo','hi']:
                assert Fraction(float.fromhex(a['intervals'][quantity]['hex_'+side]))==Fraction(b[quantity+'_'+side])
    report['corner_saved_records_checked']=len(saved)
    py('audit-all-relative-width-histories',rel/'summarize.py',rel)
if args.geometry_replay:
    py('replay-88-hybrid-geometry',C/'experiments/hybrid_submission/replay.py',None,'--suite','geometry','--output',str(work/'replay-geometry'))
if args.full_pde:
    py('replay-four-full-pde-histories',C/'experiments/pde_revision_20261112/replay.py',None,'--full')
if args.render:
    # Order matters: the final slope renderer supersedes earlier plot variants.
    run('render-base-assets',[sys.executable,'-B',L/'render_relu_submission_assets.py','--records-root',C/'results','--disk-history',C/'experiments/smoothed_disk_figure1/data/history.csv','--latex',L])
    py('render-relative-width-tables',C/'experiments/relative_width_20261111/summarize.py')
    py('render-pde-tables',C/'experiments/pde_revision_20261112/render_assets.py')
    py('render-current-reference-guides',C/'experiments/slope_triangles_20261112/render.py')
    # PDE generation refreshes scratch files, so its original frozen-manifest check
    # is performed before rendering, not incorrectly asserted afterward.
    run('assemble-figure-one',['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error','-outdir=images','disk_figure_revised.tex'],L)
if args.build_paper:
    for document in ['main.tex','maxgap_supplement.tex']:
        run('build-'+Path(document).stem,['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error',document],L)
report['status']='passed';report['work_directory']=str(work)
(work/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print('Completed; released evidence unchanged. Report:',work/'report.json',flush=True)
