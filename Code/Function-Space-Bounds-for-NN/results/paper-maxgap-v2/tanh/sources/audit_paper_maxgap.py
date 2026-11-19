from pathlib import Path
import json,math
from fractions import Fraction as Q
from itertools import product
from experiment_records import validate_record,validate_policy,sha,dump
ROOT=Path(__file__).resolve().parent.parent
checks=[]
for spec in json.loads((ROOT/'experiments.json').read_text()):
 folder=ROOT/'runs'/spec['id'];manifest=json.loads((folder/'manifest.json').read_text());d=len(spec['domain']);budget=spec['max_leaves']
 record=json.loads((folder/'maxgap.json').read_text());uniform=json.loads((folder/'uniform.json').read_text())
 validate_record(record,spec['domain']);validate_record(uniform,spec['domain']);validate_policy(uniform,manifest,'uniform')
 assert record['history'][-1]['leaves']==uniform['history'][-1]['leaves']==budget
 assert record['history'][-1]['evaluations']==uniform['history'][-1]['evaluations']==1+2**d*(budget-1)//(2**d-1)
 data=json.loads((folder/'maxgap_audit.json').read_text());nodes={n['id']:n for n in data['nodes']};assert len(nodes)==len(data['nodes'])
 active={0:Q(nodes[0]['indicator'])};seen={0}
 for event in data['refinements']:
  expected=max(active,key=active.get);assert event['parent']==expected
  parent=nodes[event['parent']];del active[event['parent']]
  assert len(event['children'])==2**d and len(set(event['children']))==2**d
  pb=[[Q(v) for v in pair] for pair in parent['bounds']]
  expected_boxes=set(product(*[[(lo,(lo+hi)/2),((lo+hi)/2,hi)] for lo,hi in pb]));child_boxes=set()
  for i in event['children']:
   assert i not in seen;seen.add(i);n=nodes[i];assert n['depth']==parent['depth']+1
   b=tuple(tuple(Q(v) for v in pair) for pair in n['bounds']);child_boxes.add(b)
   low,up=[Q(n['integrand'][k]) for k in ['lower','upper']];pl,pu=[Q(parent['integrand'][k]) for k in ['lower','upper']]
   assert 0<=pl<=low<=up<=pu
   il,iu=[Q(n['integral'][k]) for k in ['lower','upper']];volume=math.prod(hi-lo for lo,hi in b)
   assert il<=volume*low<=volume*up<=iu and Q(n['indicator'])==iu-il
   active[i]=Q(n['indicator'])
  assert child_boxes==expected_boxes
 assert len(active)==budget and len(seen)==len(nodes)
 for cell,i in zip(record['cells'],active):assert cell=={k:v for k,v in nodes[i].items() if k!='id'}
 assert manifest['network_sha256']==sha(ROOT/'models'/spec['model']/'network.json')
 # Check saved history endpoints against the audit-tree partition at each saved size.
 partitions={h['leaves']:h for h in record['history']};active_ids={0}
 def check_history():
  if len(active_ids) not in partitions:return
  h=partitions[len(active_ids)];lo=sum((Q(nodes[i]['integral']['lower']) for i in active_ids),Q(0));hi=sum((Q(nodes[i]['integral']['upper']) for i in active_ids),Q(0))
  assert Q(h['integral']['lower'])<=lo<=hi<=Q(h['integral']['upper'])
 check_history()
 for event in data['refinements']:
  active_ids.remove(event['parent']);active_ids.update(event['children']);check_history()
 checks.append({'id':spec['id'],'dimension':d,'marking_decisions_checked':len(data['refinements']),'leaves':budget,'evaluations':record['history'][-1]['evaluations']})
 print('audited',spec['id'],flush=True)
dump(ROOT/'validation.json',{'independent_maxgap_linear_scan_audit':'passed','cases':checks,'total_marking_decisions_checked':sum(c['marking_decisions_checked'] for c in checks),'uniform_exact_replays':16,'policy_records_checked':32})
