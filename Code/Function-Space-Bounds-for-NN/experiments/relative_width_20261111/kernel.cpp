// Exact 2D dyadic hybrid integration with GMP. No floating-point endpoints.
#include <gmpxx.h>
#include <vector>
#include <array>
#include <map>
#include <queue>
#include <fstream>
#include <iostream>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <cassert>
using Z=mpz_class; using Q=mpq_class;
using Form=std::array<Z,3>; using Range=std::pair<Q,Q>;
struct Layer {size_t n,m;std::vector<std::vector<Z>> w;std::vector<Z>b;Z den,prod;};
struct Cell {unsigned depth;unsigned long x,y,id;Q lo,hi,rlo,rhi; bool resolved;Q gap()const{return hi-lo;}};
struct Order {bool operator()(Cell const&a,Cell const&b)const {Q ga=a.gap(),gb=b.gap();return ga==gb?a.id>b.id:ga<gb;}};
struct Oracle {
 std::vector<Layer> layers;Z rn,rd;bool h1;unsigned long calls=0,tests=0,range_calls=0,moments=0,form_builds=0;
 std::map<std::vector<std::string>,std::vector<Form>> cache;
 Oracle(std::string path,bool h):h1(h){std::ifstream f(path);size_t L;f>>rn>>rd>>L;Z P=1;for(size_t k=0;k<L;k++){Layer a;f>>a.n>>a.m>>a.den;a.w.resize(a.n,std::vector<Z>(a.m));a.b.resize(a.n);for(size_t i=0;i<a.n;i++){for(auto &v:a.w[i])f>>v;f>>a.b[i];}P*=a.den;a.prod=P;layers.push_back(a);}if(!f)throw std::runtime_error("bad model");}
 std::vector<Form> const&forms(std::vector<std::string>const&prefix){auto it=cache.find(prefix);if(it!=cache.end())return it->second;
  size_t k=prefix.size();auto const&a=layers[k];std::vector<Form> fs(a.n);form_builds++;
  if(k==0){assert(a.m==2);for(size_t i=0;i<a.n;i++)fs[i]={a.b[i],a.w[i][0],a.w[i][1]};}
  else {auto prevprefix=prefix;prevprefix.pop_back();auto const&prev=forms(prevprefix);Z P=layers[k-1].prod;
   for(size_t i=0;i<a.n;i++){fs[i]={a.b[i]*P,Z(0),Z(0)};for(size_t j=0;j<a.m;j++)if(prefix.back()[j]=='1')for(int c=0;c<3;c++)fs[i][c]+=a.w[i][j]*prev[j][c];}}
  return cache.emplace(prefix,std::move(fs)).first->second;
 }
 bool affine(unsigned d,unsigned long ix,unsigned long iy,Form&out){tests++;Z t=Z(1)<<d;Z xn=rn*(2*Z(ix)-t),yn=rn*(2*Z(iy)-t),den=rd*t,step=2*rn;std::vector<std::string>prefix;
  for(size_t k=0;k+1<layers.size();k++){auto const&fs=forms(prefix);std::string mask(fs.size(),'0');for(size_t i=0;i<fs.size();i++){auto const&f=fs[i];Z low=f[0]*den+f[1]*xn+f[2]*yn;Z hi=low;
    for(int c=1;c<3;c++){Z v=f[c]*step;if(v<0)low+=v;else hi+=v;}
    if(hi<=0){}else if(low>=0)mask[i]='1';else return false;}prefix.push_back(mask);}
  out=forms(prefix)[0];return true;
 }
 Range natural(unsigned d,unsigned long ix,unsigned long iy){range_calls++;Z t=Z(1)<<d,den=rd*t,step=2*rn;std::vector<Z>lo={rn*(2*Z(ix)-t),rn*(2*Z(iy)-t)},hi={lo[0]+step,lo[1]+step};
  std::vector<std::array<Z,2>>dl(2),du(2);for(int i=0;i<2;i++)for(int j=0;j<2;j++)dl[i][j]=du[i][j]=(i==j);Z P=1;
  for(size_t k=0;k<layers.size();k++){auto const&a=layers[k];std::vector<Z>nl(a.n),nu(a.n);std::vector<std::array<Z,2>>el(a.n),eu(a.n);
   for(size_t i=0;i<a.n;i++){nl[i]=nu[i]=a.b[i]*den*P;for(size_t j=0;j<a.m;j++){auto const&w=a.w[i][j];if(w==0)continue;
     nl[i]+=w*(w>0?lo[j]:hi[j]);nu[i]+=w*(w>0?hi[j]:lo[j]);if(h1)for(int c=0;c<2;c++){el[i][c]+=w*(w>0?dl[j][c]:du[j][c]);eu[i][c]+=w*(w>0?du[j][c]:dl[j][c]);}}
    if(k+1<layers.size()){if(h1)for(int c=0;c<2;c++){if(nu[i]<0)el[i][c]=eu[i][c]=0;else if(nl[i]<=0){if(el[i][c]>0)el[i][c]=0;if(eu[i][c]<0)eu[i][c]=0;}}
     if(nl[i]<0)nl[i]=0;if(nu[i]<0)nu[i]=0;}}
   lo=std::move(nl);hi=std::move(nu);dl=std::move(el);du=std::move(eu);P=a.prod;}
  auto sq=[](Z const&l,Z const&u,Z const&denom){Z l2=l*l,u2=u*u,mn=(l<=0&&u>=0)?Z(0):std::min(l2,u2),mx=std::max(l2,u2);Q a(mn,denom),b(mx,denom);a.canonicalize();b.canonicalize();return Range(a,b);};
  Z vden=den*P;vden*=vden;auto r=sq(lo[0],hi[0],vden);if(h1){Z gden=P*P;for(int c=0;c<2;c++){auto s=sq(dl[0][c],du[0][c],gden);r.first+=s.first;r.second+=s.second;}}r.first.canonicalize();r.second.canonicalize();return r;
 }
 Cell cell(unsigned d,unsigned long x,unsigned long y,unsigned long id,Cell const*parent=nullptr){calls++;Cell c;c.depth=d;c.x=x;c.y=y;c.id=id;Form f;Z t=Z(1)<<d;Q side(2*rn,rd*t);side.canonicalize();Q vol=side*side;
  c.resolved=affine(d,x,y,f);if(c.resolved){moments++;Q P(layers.back().prod);Q cx(-Q(rn,rd)+side*(Q(x)+Q(1,2))),cy(-Q(rn,rd)+side*(Q(y)+Q(1,2)));Q mean=(Q(f[0])+Q(f[1])*cx+Q(f[2])*cy)/P;Q a2=(Q(f[1]*f[1])+Q(f[2]*f[2]))/(P*P);c.lo=c.hi=vol*(mean*mean+a2*side*side/12+(h1?a2:Q(0)));}
  else {auto r=natural(d,x,y);if(parent){r.first=std::max(r.first,parent->rlo);r.second=std::min(r.second,parent->rhi);}assert(r.first<=r.second);c.rlo=r.first;c.rhi=r.second;c.lo=vol*r.first;c.hi=vol*r.second;}
  c.lo.canonicalize();c.hi.canonicalize();assert(c.lo<=c.hi);return c;
 }
};
bool reached(Q const&lo,Q const&hi){return hi==0||200*(hi-lo)<=hi+lo;}
int main(int argc,char**argv){try{
 if(argc!=7)throw std::runtime_error("model quantity policy cap output mode");bool h1=std::string(argv[2])=="H1";Oracle o(argv[1],h1);std::string policy=argv[3],mode=argv[6];unsigned long cap=std::stoul(argv[4]);std::ofstream out(argv[5]);auto start=std::chrono::steady_clock::now();auto sec=[&](){return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();};
 Cell root=o.cell(0,0,0,0);Q lo=root.lo,hi=root.hi;unsigned long N=1,uid=1;std::priority_queue<Cell,std::vector<Cell>,Order> heap;std::vector<Cell>level;if(root.gap()>0){heap.push(root);level.push_back(root);}unsigned depth=0;unsigned long nextrecord=4;bool tol=mode=="tolerance";
 out<<"leaves,lower,upper,relative_width,evaluated_cells,positive_leaves,seconds\n";
 auto snapshot=[&](size_t positive){Q rel=hi==0?Q(0):Q(2*(hi-lo)/(hi+lo));out<<N<<','<<lo<<','<<hi<<','<<rel<<','<<o.calls<<','<<positive<<','<<sec()<<'\n';out.flush();std::cerr<<policy<<' '<<(h1?"H1":"L2")<<" N="<<N<<" relative="<<rel.get_d()<<" calls="<<o.calls<<" seconds="<<sec()<<std::endl;};
 snapshot(root.gap()>0?1:0);while(N<cap&&hi>lo&&(!tol||!reached(lo,hi))){
  if(policy=="maxgap") {Cell p=heap.top();heap.pop();Q cl=0,cu=0;for(unsigned i=0;i<2;i++)for(unsigned j=0;j<2;j++){Cell c=o.cell(p.depth+1,2*p.x+i,2*p.y+j,uid++,&p);cl+=c.lo;cu+=c.hi;if(c.gap()>0)heap.push(std::move(c));}assert(p.lo<=cl&&cu<=p.hi);lo+=cl-p.lo;hi+=cu-p.hi;N+=3;
   if(N>=nextrecord||N==cap||(tol&&reached(lo,hi))||hi==lo){snapshot(heap.size());if(N>=nextrecord)nextrecord*=4;}}
  else {std::vector<Cell>newlevel;for(Cell const&p:level){Q cl=0,cu=0;for(unsigned i=0;i<2;i++)for(unsigned j=0;j<2;j++){Cell c=o.cell(p.depth+1,2*p.x+i,2*p.y+j,uid++,&p);cl+=c.lo;cu+=c.hi;if(c.gap()>0)newlevel.push_back(std::move(c));}assert(p.lo<=cl&&cu<=p.hi);lo+=cl-p.lo;hi+=cu-p.hi;}level=std::move(newlevel);N*=4;depth++;snapshot(level.size());}
 }
 if(policy=="maxgap"&&N<nextrecord&&N!=cap&&!(tol&&reached(lo,hi))&&hi!=lo)snapshot(heap.size());
 out.close();std::ofstream meta(std::string(argv[5])+".meta");meta<<"status "<<(reached(lo,hi)?"relative_tolerance_reached":hi==lo?"exact":"cap")<<"\nseconds "<<sec()<<"\nevaluated_cells "<<o.calls<<"\nrange_calls "<<o.range_calls<<"\nmoment_calls "<<o.moments<<"\nform_builds "<<o.form_builds<<"\n";
 }catch(std::exception const&e){std::cerr<<e.what()<<std::endl;return 1;}}
