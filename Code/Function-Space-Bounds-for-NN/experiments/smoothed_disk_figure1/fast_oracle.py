"""Same 2D dyadic hybrid oracle, evaluated with exact integers and cached forms.

Stored binary weights are dyadic. Matrix products use NumPy object arrays of
GMP integers, never machine integers or floating point. Layerwise common
denominators avoid normalising a rational after every multiply/add. Caching
preactivation affine coefficients depends only on preceding activation masks.
The sign test, fallback range and integral endpoints equal backend.Oracle.
"""
from functools import lru_cache
from collections import Counter
import numpy as np
import gmpy2 as g
from backend import Network,Cell,Q,volume,square

class FastOracle:
    def __init__(self,record):
        self.net=Network(record);assert self.net.d==2
        self.counts=Counter();self.w=[];self.b=[];self.ds=[];self.ps=[]
        self.wn=[];self.an=[];self.bn=[];P=g.mpz(1)
        for w,b in zip(self.net.w,self.net.b):
            ds=[x.denominator for row in w for x in row]+[x.denominator for x in b]
            assert all(d>0 and int(d)&(int(d)-1)==0 for d in ds)
            D=g.mpz(max(ds));wi=tuple(tuple(g.mpz(x*D) for x in row) for row in w)
            bi=tuple(g.mpz(x*D) for x in b)
            self.w.append(wi);self.b.append(bi);self.ds.append(D);P*=D;self.ps.append(P)
            self.wn.append(np.array(wi,dtype=object));self.an.append(np.array([[abs(x) for x in row] for row in wi],dtype=object))
            self.bn.append(np.array(bi,dtype=object))

    @lru_cache(maxsize=12000)
    def forms(self,prefix):
        k=len(prefix)
        if k==0:return tuple((b,*w) for w,b in zip(self.w[0],self.b[0]))
        prev=self.forms(prefix[:-1]);mask=prefix[-1]
        active=[(j,f) for j,f in enumerate(prev) if mask>>j&1]
        P=self.ps[k-1]
        return tuple(tuple((b*P if c==0 else g.mpz(0))+
            sum((row[j]*f[c] for j,f in active),g.mpz(0)) for c in range(3))
            for row,b in zip(self.w[k],self.b[k]))

    @staticmethod
    def geometry(box):
        h=box[0][1]-box[0][0];t=Q(3)/h
        assert t.denominator==1 and box[1][1]-box[1][0]==h
        t=g.mpz(t);ix=(box[0][0]+Q(3,2))/h;iy=(box[1][0]+Q(3,2))/h
        assert ix.denominator==iy.denominator==1
        return g.mpz(ix),g.mpz(iy),t

    def affine(self,box):
        ix,iy,t=self.geometry(box);x=-3*t+6*ix;y=-3*t+6*iy
        prefix=()
        for k in range(len(self.w)-1):
            mask=0
            for j,(b,ax,ay) in enumerate(self.forms(prefix)):
                z=2*t*b+ax*x+ay*y
                lo=z+min(0,6*ax)+min(0,6*ay)
                hi=z+max(0,6*ax)+max(0,6*ay)
                if hi<=0:pass
                elif lo>=0:mask|=1<<j
                else:return None
            prefix+=mask,
        return self.forms(prefix)[0],self.ps[-1]

    def natural_range(self,box):
        ix,iy,t=self.geometry(box);den=2*t
        lo=np.array([-3*t+6*ix,-3*t+6*iy],dtype=object);hi=lo+6
        for k,(W,A,b,D) in enumerate(zip(self.wn,self.an,self.bn,self.ds)):
            center=W.dot(lo+hi)+2*den*b
            radius=A.dot(hi-lo)
            # Both numerators are exactly even by expansion of lo+hi, hi-lo.
            lower=center-radius;upper=center+radius
            if self.counts['cell_calls']<100:
                assert all(v%2==0 for v in lower) and all(v%2==0 for v in upper)
            lo=np.array([v//2 for v in lower],dtype=object)
            hi=np.array([v//2 for v in upper],dtype=object);den*=D
            if k<len(self.w)-1:
                lo=np.array([max(g.mpz(0),v) for v in lo],dtype=object)
                hi=np.array([max(g.mpz(0),v) for v in hi],dtype=object)
        return square((Q(lo[0],den),Q(hi[0],den)))

    def cell(self,box,parent=None):
        self.counts['cell_calls']+=1
        if parent is not None and parent.resolved:
            f=getattr(parent,'form',None)
            if f is None:f=self.affine(parent.box);assert f is not None;parent.form=f
        else:f=self.affine(box)
        if f is not None:
            (b,ax,ay),P=f;ix,iy,t=self.geometry(box)
            r=2*t*b+ax*(-3*t+6*ix+3)+ay*(-3*t+6*iy+3)
            q=Q(9*(r*r+3*(ax*ax+ay*ay)),4*t**4*P**2)
            c=Cell(box,q,q,None,True);c.form=f
        else:
            r=self.natural_range(box)
            if parent is not None and parent.fallback is not None:
                r=max(r[0],parent.fallback[0]),min(r[1],parent.fallback[1])
            assert r[0]<=r[1]
            v=volume(box);c=Cell(box,v*r[0],v*r[1],r,False)
        return c
