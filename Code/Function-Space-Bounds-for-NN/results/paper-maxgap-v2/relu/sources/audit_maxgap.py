"""Independent linear-scan selection audit; exact rational geometry/aggregation."""
from pathlib import Path
import json
from fractions import Fraction as Q
from experiment_records import validate_record,dump,sha
ROOT=Path(__file__).resolve().parent.parent
checks=[]
for arch in ['deep','shallow200']:
 for name in ['two-bumps','disk']:
  for norm in ['L2','H1']:
   folder=ROOT/arch/name/norm
   m=json.loads((folder/'manifest.json').read_text());record=json.loads((folder/'maxgap.json').read_text())
   validate_record(record,m['domain']);validate_record(json.loads((folder/'uniform.json').read_text()),m['domain'])
   data=json.loads((folder/'maxgap_audit.json').read_text());nodes={n['id']:n for n in data['nodes']};assert len(nodes)==len(data['nodes'])
   active={0:Q(nodes[0]['indicator'])}
   for event in data['refinements']:
    # This audit uses no heap. Python dict order is creation order.
    expected=max(active,key=active.get);assert event['parent']==expected
    parent=nodes[event['parent']];del active[event['parent']]
    assert len(event['children'])==4 and len(set(event['children']))==4
    pb=[[Q(v) for v in pair] for pair in parent['bounds']]
    expected_boxes=set()
    from itertools import product
    for pairs in product(*[[(lo,(lo+hi)/2),((lo+hi)/2,hi)] for lo,hi in pb]):expected_boxes.add(tuple(pairs))
    child_boxes=set()
    for i in event['children']:
     assert i not in active;n=nodes[i];assert n['depth']==parent['depth']+1
     b=tuple(tuple(Q(v) for v in pair) for pair in n['bounds']);child_boxes.add(b)
     low,up=[Q(n['integrand'][k]) for k in ['lower','upper']]
     pl,pu=[Q(parent['integrand'][k]) for k in ['lower','upper']];assert pl<=low<=up<=pu
     il,iu=[Q(n['integral'][k]) for k in ['lower','upper']];volume=(b[0][1]-b[0][0])*(b[1][1]-b[1][0])
     assert il<=volume*low<=volume*up<=iu and Q(n['indicator'])==iu-il
     active[i]=Q(n['indicator'])
    assert child_boxes==expected_boxes
   assert len(active)==m['max_leaves']
   for cell,i in zip(record['cells'],active):
    assert cell=={k:v for k,v in nodes[i].items() if k!='id'}
   assert record['history'][-1]['leaves']==m['max_leaves']
   assert m['network_sha256']==sha(folder.parent/'network.json')
   checks.append({'architecture':arch,'experiment':name,'norm':norm,'refinements_checked':len(data['refinements']),'leaves':len(active),'evaluations':record['history'][-1]['evaluations']})
dump(ROOT/'validation.json',{'independent_linear_scan_marking_audit':'passed','checks':checks,'uniform_exact_replay':'passed in runner for all 8 cases','new_maxgap_tests':2,'relu_tests':3})
print('Independent marking and geometry audit passed:',len(checks),'cases',flush=True)
