import unittest
from verified_arithmetic import Arithmetic,rational
from verified_integration import Box
from maxgap import MaxGapIntegrator
class MaxGapTests(unittest.TestCase):
 def test_one_box_and_stable_ties(self):
  a=Arithmetic();run=MaxGapIntegrator(a,lambda b:a.interval(0,1),Box(((0,1),(0,1))))
  run.run_maxgap(16)
  self.assertEqual([e['parent'] for e in run.refinements],[0,1,2,3,4])
  self.assertEqual(len(run.cells),16);self.assertEqual(run.evaluations,21)
 def test_actual_maximum(self):
  a=Arithmetic();run=MaxGapIntegrator(a,lambda b:a.interval(b[0][0]**2,b[0][1]**2),Box(((0,1),(0,1))))
  run.run_maxgap(64);nodes={n['id']:n for n in run.node_records};active={0}
  for e in run.refinements:
   maximum=max(rational(nodes[i]['indicator']) for i in active)
   self.assertEqual(rational(nodes[e['parent']]['indicator']),maximum)
   self.assertEqual(len(e['children']),4);active.remove(e['parent']);active.update(e['children'])
  self.assertEqual(len(active),64)
if __name__=='__main__':unittest.main()
