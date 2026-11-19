"""Theorem reference triangles in data coordinates, never estimated from curves."""
import math
GUIDES=[]
def _anchor(ax,position):
 xlim,ylim=ax.get_xlim(),ax.get_ylim()
 assert min(ylim)>0 and ax.get_yscale()=='log'
 x=(10**(math.log10(xlim[0])+position[0]*math.log10(xlim[1]/xlim[0])) if ax.get_xscale()=='log' else xlim[0]+position[0]*(xlim[1]-xlim[0]))
 y=10**(math.log10(ylim[0])+position[1]*math.log10(ylim[1]/ylim[0]))
 return x,y,xlim,ylim

def power(ax,alpha,color,position=(.65,.85),factor=4,label=None,policy=None,theorem=None):
 assert ax.get_xscale()=='log' and alpha>0 and factor>1
 x,y,xlim,ylim=_anchor(ax,position);xx=x*factor;yy=y/factor**alpha
 assert xx<xlim[1] and yy>ylim[0]
 verts=[[x,y],[xx,y],[xx,yy],[x,y]]
 ax.plot(*zip(*verts),color=color,lw=.95,ls='--',zorder=4)
 exponent='1/2' if alpha==.5 else f'{alpha:g}'
 ax.annotate(label or rf'$N^{{-{exponent}}}$',(xx,math.sqrt(y*yy)),xytext=(3,0),textcoords='offset points',va='center',color=color,fontsize=10)
 ax.set_xlim(xlim);ax.set_ylim(ylim)
 observed=math.log(yy/y)/math.log(xx/x);assert math.isclose(observed,-alpha,abs_tol=1e-12)
 GUIDES.append({'panel':ax.get_gid(),'kind':'power','policy':policy,'theorem':theorem,'alpha':alpha,'vertices':verts,'verified_log_slope':observed})

def exponential(ax,m,color,position=(.68,.80),steps=2):
 assert ax.get_xscale()=='linear' and m>=1
 x,y,xlim,ylim=_anchor(ax,position);dx=steps*m;xx=x+dx;yy=y/2**steps
 assert xx<xlim[1] and yy>ylim[0]
 verts=[[x,y],[xx,y],[xx,yy],[x,y]]
 ax.plot(*zip(*verts),color=color,lw=.95,ls='--',zorder=4)
 label=r'$2^{-N}$' if m==1 else rf'$2^{{-N/{m}}}$'
 ax.annotate(label,(xx,math.sqrt(y*yy)),xytext=(3,0),textcoords='offset points',va='center',color=color,fontsize=10)
 ax.set_xlim(xlim);ax.set_ylim(ylim)
 observed=math.log2(yy/y)/(xx-x);assert math.isclose(observed,-1/m,abs_tol=1e-12)
 GUIDES.append({'panel':ax.get_gid(),'kind':'exponential','policy':'maxgap','theorem':'thm:spline_separation','m':m,'vertices':verts,'verified_semilog2_slope':observed})
