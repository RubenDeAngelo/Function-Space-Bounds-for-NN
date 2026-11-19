"""Independent 2D activation-region expansion and triangle integration.
Uses fractions.Fraction, not the backend's gmpy2 or box-moment formula.
Intended for small falsification networks, not the trained 32-neuron models.
"""
from fractions import Fraction as F

def clip(poly,a,b,c):
 out=[]
 for p,q in zip(poly,poly[1:]+poly[:1]):
  vp=a*p[0]+b*p[1]+c;vq=a*q[0]+b*q[1]+c
  if vp>=0:out.append(p)
  if (vp<0<vq) or (vq<0<vp):
   t=vp/(vp-vq);out.append(tuple(x+t*(y-x) for x,y in zip(p,q)))
 return list(dict.fromkeys(out))
def area(poly):return abs(sum((p[0]*q[1]-p[1]*q[0] for p,q in zip(poly,poly[1:]+poly[:1])),F(0)))/2 if len(poly)>=3 else F(0)
def expand(record,domain):
 (l,u),(v,w)=[tuple(F(x) for x in k) for k in domain]
 pieces=[([(l,v),(u,v),(u,w),(l,w)],[(F(0),F(1),F(0)),(F(0),F(0),F(1))])]
 for k,(weights,biases) in enumerate(zip(record['weights'],record['biases'])):
  allnew=[]
  for poly,oldforms in pieces:
   pres=[tuple((F(b) if j==0 else F(0))+sum((F(c)*f[j] for c,f in zip(row,oldforms)),F(0)) for j in range(3)) for row,b in zip(weights,biases)]
   if k==len(record['weights'])-1:allnew.append((poly,pres));continue
   work=[(poly,[])]
   for f in pres:
    nxt=[]
    for p,forms in work:
     vs=[f[0]+f[1]*x+f[2]*y for x,y in p]
     if max(vs)<=0:nxt.append((p,forms+[(F(0),)*3]))
     elif min(vs)>=0:nxt.append((p,forms+[f]))
     else:
      for sign,ff in [(1,f),(-1,(F(0),)*3)]:
       pp=clip(p,sign*f[1],sign*f[2],sign*f[0])
       if area(pp)>0:nxt.append((pp,forms+[ff]))
    work=nxt
   allnew.extend(work)
  pieces=allnew
  if len(pieces)>10000:raise RuntimeError('Tiny-network validator region limit')
 return pieces

def integrate_polygon(poly,f,h1):
 if len(poly)<3:return F(0)
 ans=F(0)
 for i in range(1,len(poly)-1):
  tri=[poly[0],poly[i],poly[i+1]];ar=area(tri)
  z=[f[0]+f[1]*x+f[2]*y for x,y in tri]
  ans+=ar*(sum(x*x for x in z)+z[0]*z[1]+z[0]*z[2]+z[1]*z[2])/6
  if h1:ans+=ar*(f[1]**2+f[2]**2)
 return ans

def integral(pieces,box,h1=False):
 (l,u),(v,w)=[tuple(F(str(x)) for x in k) for k in box];total=F(0)
 for poly,forms in pieces:
  p=poly
  for a,b,c in [(1,0,-l),(-1,0,u),(0,1,-v),(0,-1,w)]:
   if not p:break
   p=clip(p,F(a),F(b),F(c))
  total+=integrate_polygon(p,forms[0],h1)
 return total
