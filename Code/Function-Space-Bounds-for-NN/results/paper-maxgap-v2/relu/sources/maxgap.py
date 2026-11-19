"""One maximum integral-gap leaf per step; exact cached indicators and stable ties."""
import heapq
from verified_integration import Integrator
from verified_arithmetic import rational

class MaxGapIntegrator(Integrator):
    def run_maxgap(self, max_leaves):
        if max_leaves < 1: raise ValueError('positive budget required')
        active={0:self.cells[0]};heap=[(-active[0].indicator,0)];next_id=1
        self.refinements=[];self.node_records=[]
        def node_record(i,c):
            return {'id':i,'bounds':[[str(v) for v in pair] for pair in c.box.bounds],
                    'depth':c.box.depth,'integrand':c.integrand.serialise(),'integral':c.integral.serialise(),'indicator':str(c.indicator)}
        self.node_records.append(node_record(0,active[0]))
        while len(active)+3<=max_leaves:
            neg,i=heapq.heappop(heap);parent=active.pop(i)
            assert -neg==parent.indicator
            children=[]
            for box in parent.box.children():
                c=self.evaluate(box,parent.integrand);j=next_id;next_id+=1
                active[j]=c;heapq.heappush(heap,(-c.indicator,j));children.append(j)
                self.node_records.append(node_record(j,c))
            self.refinements.append({'parent':i,'children':children})
            # Save at uniform levels, every 32 splits, and at the final count.
            n=len(active)
            if len(self.refinements)%32==0 or n in (4,16,64,256,1024,4096) or n+3>max_leaves:
                self.cells=list(active.values());self.record()
        self.cells=list(active.values())
        if self.history[-1]['leaves']!=len(active):self.record()
        self.status='leaf_budget_exhausted'
        return self
