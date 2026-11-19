#include "interval.hpp"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <chrono>
#include <queue>
#include <unordered_map>
#include <memory>
#include <functional>
#include <cstring>
#include <sys/resource.h>
using Clock=std::chrono::steady_clock;
double since(Clock::time_point a){return std::chrono::duration<double>(Clock::now()-a).count();}
struct Box{int kind=0,dr=0,dz=0,root=0;uint64_t nr=0,nz=0;std::string path;};
M P0(const M&p){return rat(-3,2)*cosM(p*rat(2,3));}
M PX(const M&p){return rat(1,2)*(rat(-3,5)*cosM(p*rat(5,3))+M(3)*cosM(p*rat(1,3)));}
M PY(const M&p){return rat(1,2)*(M(3)*sinM(p*rat(1,3))-rat(3,5)*sinM(p*rat(5,3)));}
struct Geo{M X,Y,A,S,m0,m1,m2,H0,H1,L,L1,L2;std::array<M,6> arc;M S0,Sx,Sy;};
Geo raw_geometry(const Box&b){
 Geo g;static M pi=M::pi();static M omega=pi*rat(3,2);
 M r0(dyadic(b.nr,b.dr)),r1(dyadic(b.nr+1,b.dr)),z0(dyadic(b.nz,b.dz)),z1(dyadic(b.nz+1,b.dz));
 M a=omega*z0,c=omega*z1,R=span(r0,r1);
 if(b.kind==1||b.kind==2){
  if(b.kind==1){g.X=R;g.Y=M(0);}else{g.X=M(0);g.Y=-R;}
  M ca=cbrtM(r0),cb=cbrtM(r1);
  g.H0=rat(3,2)*(ipow(cb,2)-ipow(ca,2))-rat(1,2)*(ipow(r1,2)-ipow(r0,2));
  g.H1=rat(3,5)*(ipow(cb,5)-ipow(ca,5))-rat(1,3)*(ipow(r1,3)-ipow(r0,3));
  if(mpfr_sgn(g.H0.l)<0)mpfr_set_zero(g.H0.l,1);
  if(mpfr_sgn(g.H1.l)<0)mpfr_set_zero(g.H1.l,1);
  g.L=r1-r0;g.L1=rat(1,2)*(ipow(r1,2)-ipow(r0,2));g.L2=rat(1,3)*(ipow(r1,3)-ipow(r0,3));
  return g;
 }
 M angle=span(a,c);g.X=cosM(angle);g.Y=sinM(angle);
 g.S0=P0(c)-P0(a);g.Sx=PX(c)-PX(a);g.Sy=PY(c)-PY(a);
 if(mpfr_sgn(g.S0.l)<0)mpfr_set_zero(g.S0.l,1);
 if(b.kind==0){
  g.X=R*g.X;g.Y=R*g.Y;
  M R0=rat(1,2)*(ipow(r1,2)-ipow(r0,2));M R1=rat(1,3)*(ipow(r1,3)-ipow(r0,3));
  g.A=R0*(c-a);g.S=rat(32,9)*sinM(pi*span(z0,z1));
  g.m0=rat(32,9)*R0*g.S0;g.m1=rat(32,9)*R1*g.Sx;g.m2=rat(32,9)*R1*g.Sy;
 }else{
  g.L=c-a;g.arc[0]=g.L;g.arc[1]=sinM(c)-sinM(a);g.arc[2]=cosM(a)-cosM(c);
  M d=rat(1,4)*(sinM(M(2)*c)-sinM(M(2)*a));
  g.arc[3]=rat(1,2)*g.L+d;g.arc[4]=rat(1,2)*g.L-d;
  g.arc[5]=rat(1,2)*(ipow(sinM(c),2)-ipow(sinM(a),2));
 }
 return g;
}
// Cache geometry by exact one-dimensional dyadic intervals. Each verifier process
// starts with cold caches. All cache costs remain in end-to-end timing.
Geo geometry(const Box&b){
 static std::unordered_map<std::string,Geo> radial,angular;
 auto rk=std::to_string(b.nr)+"/"+std::to_string(b.dr);
 auto ri=radial.find(rk);
 if(ri==radial.end()){Box q=b;q.kind=1;ri=radial.emplace(rk,raw_geometry(q)).first;}
 const Geo&rr=ri->second;
 if(b.kind==1)return rr;
 if(b.kind==2){Geo g=rr;g.X=M(0);g.Y=-rr.X;return g;}
 auto zk=std::to_string(b.nz)+"/"+std::to_string(b.dz);
 auto zi=angular.find(zk);
 if(zi==angular.end()){Box q=b;q.kind=3;Geo g=raw_geometry(q);
  M z0(dyadic(b.nz,b.dz)),z1(dyadic(b.nz+1,b.dz));g.S=rat(32,9)*sinM(M::pi()*span(z0,z1));
  zi=angular.emplace(zk,g).first;}
 Geo g=zi->second;
 if(b.kind==0){g.X=rr.X*g.X;g.Y=rr.X*g.Y;g.A=rr.L1*g.L;g.m0=rat(32,9)*rr.L1*g.S0;g.m1=rat(32,9)*rr.L2*g.Sx;g.m2=rat(32,9)*rr.L2*g.Sy;}
 return g;
}
struct Model{float w[64][2],b[64],a[64],b0;};
Model load(const std::string&p){Model m;std::ifstream f(p,std::ios::binary);f.read(reinterpret_cast<char*>(&m),sizeof(m));if(f.gcount()!=sizeof(m)||f.peek()!=EOF)throw std::runtime_error("wrong model byte size");for(size_t i=0;i<sizeof(m)/4;++i){float x;std::memcpy(&x,reinterpret_cast<char*>(&m)+i*4,4);if(!std::isfinite(x))throw std::runtime_error("nonfinite model");}return m;}
struct Metrics{double geo=0,network=0,coeff=0,aggregation=0,priority=0;uint64_t evaluated=0,accepted=0,inconclusive=0,cache_hits=0,cache_misses=0;};
template<class T>struct Eval{
 using V=I<T>;Model m;bool detect;Metrics stats;std::unordered_map<uint64_t,std::array<V,3>>cache;
 explicit Eval(Model model,bool d):m(model),detect(d){}
 std::array<V,5> operator()(const Box&b,bool&affine,double&area){
  auto ts=Clock::now();Geo g=geometry(b);stats.geo+=since(ts);++stats.evaluated;
  ts=Clock::now();V X=g.X.native<T>(),Y=g.Y.native<T>(),val(T(m.b0)),D1(0),D2(0);uint64_t mask=0;bool resolved=true;
  for(int j=0;j<64;++j){
   V pre=V(T(m.w[j][0]))*X+V(T(m.w[j][1]))*Y+V(T(m.b[j]));
   V act(std::max(T(0),pre.l),std::max(T(0),pre.h));val=val+V(T(m.a[j]))*act;
   int state=pre.l>0?1:(pre.h<0?-1:0);if(state==1)mask|=uint64_t(1)<<j;if(state==0){resolved=false;++stats.inconclusive;}
   if(state!=-1){V q1=V(T(m.a[j]))*V(T(m.w[j][0])),q2=V(T(m.a[j]))*V(T(m.w[j][1]));if(state==0){q1=V(std::min(T(0),q1.l),std::max(T(0),q1.h));q2=V(std::min(T(0),q2.l),std::max(T(0),q2.h));}D1=D1+q1;D2=D2+q2;}
  }
  stats.network+=since(ts);ts=Clock::now();affine=detect&&resolved;std::array<V,3>cf;
  if(affine){++stats.accepted;auto it=cache.find(mask);if(it==cache.end()){++stats.cache_misses;cf={V(T(m.b0)),V(0),V(0)};for(int j=0;j<64;++j)if(mask&(uint64_t(1)<<j)){cf[0]=cf[0]+V(T(m.a[j]))*V(T(m.b[j]));cf[1]=cf[1]+V(T(m.a[j]))*V(T(m.w[j][0]));cf[2]=cf[2]+V(T(m.a[j]))*V(T(m.w[j][1]));}cache.emplace(mask,cf);}else{++stats.cache_hits;cf=it->second;}val=intersect(val,cf[0]+cf[1]*X+cf[2]*Y);}
  stats.coeff+=since(ts);ts=Clock::now();std::array<V,5>out{};
  auto n=[](const M&x){return x.native<T>();};
  if(b.kind==0){V A=n(g.A);area=double(A.h);if(affine)out[0]=V(T(0.5))*A*(sq(cf[1])+sq(cf[2]))-(cf[0]*n(g.m0)+cf[1]*n(g.m1)+cf[2]*n(g.m2));else out[0]=A*(V(T(0.5))*(sq(D1)+sq(D2))-n(g.S)*val);}
  else if(b.kind==1||b.kind==2){
   if(affine){V d0=cf[0],d1=b.kind==1?cf[1]:-cf[2];out[1]=n(rat(-2,3))*(d0*n(g.H0)+d1*n(g.H1));out[3]=sq(d0)*n(g.L)+V(T(2))*d0*d1*n(g.L1)+sq(d1)*n(g.L2);}
   else{out[1]=n(rat(-2,3))*n(g.H0)*val;out[3]=n(g.L)*sq(val);}
   out[3].l=std::max(T(0),out[3].l);
  }else{
   if(affine){out[1]=n(rat(-4,3))*(cf[0]*n(g.S0)+cf[1]*n(g.Sx)+cf[2]*n(g.Sy));
    out[2]=sq(cf[0])*n(g.arc[0])+sq(cf[1])*n(g.arc[3])+sq(cf[2])*n(g.arc[4])+V(T(2))*(cf[0]*cf[1]*n(g.arc[1])+cf[0]*cf[2]*n(g.arc[2])+cf[1]*cf[2]*n(g.arc[5]));}
   else{out[1]=n(rat(-4,3))*n(g.S0)*val;out[2]=n(g.L)*sq(val);}
   out[2].l=std::max(T(0),out[2].l);out[3]=out[2];
  }
  out[4]=V(T(3))*out[0]-V(T(3))*out[1]+out[2];for(auto&x:out)x.check();stats.network+=since(ts);return out;
 }
};
struct Options{std::string model,precision="64",history_in,history_out;bool detect=true;double seconds=600;size_t leaves=1048576,boundary=262144;int depth=40;size_t splits=0;};
template<class T,class A>struct Engine{
 using V=I<T>;using S=I<A>;
 struct Node{Box box;std::array<S,5>sum{};int parent=-1,left=-1,right=-1;bool affine=false;double area=0;};
 struct Item{mpq_class gap;int index,root;std::string path;};
 struct Less{bool operator()(const Item&a,const Item&b)const{int c=cmp(a.gap,b.gap);if(c)return c<0;if(a.root!=b.root)return a.root>b.root;return a.path>b.path;}};
 Eval<T> eval;Options opt;std::vector<Node>nodes;std::priority_queue<Item,std::vector<Item>,Less>heap;size_t interior=0,boundary=0,affine_i=0,affine_b=0,steps=0;double affine_area=0;int maxdr=0,maxdz=0;Clock::time_point start;std::ofstream hist;std::ifstream replay;
 Engine(Model m,Options o):eval(m,o.detect),opt(o),start(Clock::now()){
  if(!o.history_out.empty())hist.open(o.history_out);if(!o.history_in.empty()){replay.open(o.history_in);if(!replay)throw std::runtime_error("missing history");}
  nodes.reserve(200000);
  nodes.clear();nodes.push_back(Node{});std::vector<Box>roots;
  for(int k=0;k<4;++k){Box b;b.nz=k;b.dz=2;b.root=roots.size();roots.push_back(b);}
  for(int k=1;k<=2;++k){Box b;b.kind=k;b.root=roots.size();roots.push_back(b);}
  for(int k=0;k<4;++k){Box b;b.kind=3;b.nz=k;b.dz=2;b.root=roots.size();roots.push_back(b);}
  std::function<void(int,int,int)>build=[&](int idx,int a,int b){if(b-a==1){nodes[idx].box=roots[a];evaluate(idx);return;}int mid=(a+b)/2,l=nodes.size();nodes.emplace_back();int r=nodes.size();nodes.emplace_back();nodes[idx].left=l;nodes[idx].right=r;nodes[l].parent=idx;nodes[r].parent=idx;build(l,a,mid);build(r,mid,b);combine(idx);};build(0,0,roots.size());
 }
 void evaluate(int idx){Node&n=nodes[idx];auto arr=eval(n.box,n.affine,n.area);for(int j=0;j<5;++j)n.sum[j]=S(A(arr[j].l),A(arr[j].h));
  if(n.box.kind==0){++interior;if(n.affine){++affine_i;affine_area+=n.area;}}else{++boundary;if(n.affine)++affine_b;}
  auto ts=Clock::now();heap.push({width(arr[4]),idx,n.box.root,n.box.path});eval.stats.priority+=since(ts);maxdr=std::max(maxdr,n.box.dr);maxdz=std::max(maxdz,n.box.dz);
 }
 void combine(int idx){auto ts=Clock::now();for(int j=0;j<5;++j)nodes[idx].sum[j]=nodes[nodes[idx].left].sum[j]+nodes[nodes[idx].right].sum[j];eval.stats.aggregation+=since(ts);}
 bool split(std::string&status){int idx;
  if(replay.is_open()){if(!(replay>>idx)){status="history_end";return false;}if(idx<0||idx>=int(nodes.size())||nodes[idx].left!=-1)throw std::runtime_error("invalid replay node");}
  else{if(heap.empty())throw std::runtime_error("empty heap");auto ts=Clock::now();idx=heap.top().index;heap.pop();eval.stats.priority+=since(ts);}
  Box p=nodes[idx].box;int axis=p.kind==0?(p.dr<=p.dz?0:1):(p.kind==3?1:0);
  if((axis==0?p.dr:p.dz)>=opt.depth){status="depth_limit";return false;}
  if((p.kind==0&&interior>=opt.leaves)||(p.kind!=0&&boundary>=opt.boundary)){status="leaf_limit";return false;}
  if(hist)hist<<idx<<'\n';Box a=p,b=p;a.path+='0';b.path+='1';
  if(axis==0){a.nr=2*p.nr;b.nr=2*p.nr+1;a.dr=b.dr=p.dr+1;}else{a.nz=2*p.nz;b.nz=2*p.nz+1;a.dz=b.dz=p.dz+1;}
  if(p.kind==0){--interior;if(nodes[idx].affine){--affine_i;affine_area-=nodes[idx].area;}}else{--boundary;if(nodes[idx].affine)--affine_b;}
  int l=nodes.size();nodes.emplace_back();int r=nodes.size();nodes.emplace_back();nodes[idx].left=l;nodes[idx].right=r;nodes[l].parent=idx;nodes[r].parent=idx;nodes[l].box=a;nodes[r].box=b;evaluate(l);evaluate(r);
  for(int i=idx;i>=0;i=nodes[i].parent)combine(i);++steps;return true;
 }
 S total(){S t=nodes[0].sum[4]+(M::pi()*rat(1,2)).native<A>();if(t.h<0)throw std::runtime_error("negative estimator upper bound");t.l=std::max(A(0),t.l);return t;}
 void output(const std::string&status){S t=total();double sec=since(start);M upper(mpq_class(double(t.h)));M sr;mpfr_sqrt(sr.h,upper.h,MPFR_RNDU);double err=mpfr_get_d(sr.h,MPFR_RNDU);
  std::cout<<std::setprecision(17)<<"{\"status\":\""<<status<<"\",\"seconds\":"<<sec<<",\"steps\":"<<steps<<",\"interior\":"<<interior<<",\"boundary\":"<<boundary<<",\"affine_interior\":"<<affine_i<<",\"affine_boundary\":"<<affine_b<<",\"affine_area_estimate\":"<<affine_area<<",\"max_dr\":"<<maxdr<<",\"max_dz\":"<<maxdz<<",\"error_upper\":"<<err<<",\"intervals\":{";
  const char*names[]={"J","C","Ba2","B2","T"};for(int j=0;j<5;++j){S a=j==4?t:nodes[0].sum[j];if(j)std::cout<<',';std::cout<<'"'<<names[j]<<"\":{\"lo\":"<<double(a.l)<<",\"hi\":"<<double(a.h)<<",\"hex_lo\":\""<<std::hexfloat<<double(a.l)<<"\",\"hex_hi\":\""<<double(a.h)<<std::defaultfloat<<"\"}";}
  auto&s=eval.stats;struct rusage ru;getrusage(RUSAGE_SELF,&ru);
  std::cout<<"},\"metrics\":{\"geometry_seconds\":"<<s.geo<<",\"network_seconds\":"<<s.network<<",\"coefficient_seconds\":"<<s.coeff<<",\"aggregation_seconds\":"<<s.aggregation<<",\"priority_seconds\":"<<s.priority<<",\"evaluations\":"<<s.evaluated<<",\"accepted_cumulative\":"<<s.accepted<<",\"inconclusive_signs\":"<<s.inconclusive<<",\"coefficient_cache_hits\":"<<s.cache_hits<<",\"coefficient_cache_misses\":"<<s.cache_misses<<",\"peak_rss_bytes\":"<<ru.ru_maxrss<<"}}"<<std::endl;
 }
 void run(){mpq_class goals[]={mpq_class(1,100),mpq_class(1,1000),mpq_class(1,10000)};bool seen[3]={};output("initial");size_t next=64;std::string status="running";
  while(true){S t=total();auto gap=width(t);for(int i=0;i<3;++i)if(!seen[i]&&gap<=goals[i]){seen[i]=true;output(std::string("target_")+std::to_string(i));}
   if(seen[2]&&!replay.is_open()&&!opt.splits){status="target_reached";break;}
   if(since(start)>=opt.seconds){status="time_limit";break;}
   if(opt.splits&&steps>=opt.splits){status="split_limit";break;}
   if(!split(status))break;
   if(steps>=next){output("checkpoint");next*=2;}
  }
  output(status);
 }
};
void need(bool b,const char*s){if(!b)throw std::runtime_error(s);}
template<class T>void arithmetic_tests(){
 need(std::numeric_limits<T>::is_iec559,"non IEEE type");need(std::fegetround()==FE_TONEAREST,"not round to nearest");
 T tiny=std::numeric_limits<T>::denorm_min();volatile T one=T(1);volatile T tt=tiny;need(T(tt*one)==tiny,"flush-to-zero");
 std::vector<T>v={T(0),T(1),T(-1),T(0.1),T(-0.7),T(1e-10),tiny,T(2)*tiny,std::numeric_limits<T>::min()};
 for(T x:v)for(T y:v){for(int op=0;op<3;++op){I<T>a(x),b(y),c=op==0?a+b:(op==1?a-b:a*b);mpq_class xx{double(x)},yy{double(y)},q;if(op==0)q=xx+yy;else if(op==1)q=xx-yy;else q=xx*yy;need(!std::isfinite(c.l)||mpq_class(double(c.l))<=q,"lower arithmetic failure");need(!std::isfinite(c.h)||q<=mpq_class(double(c.h)),"upper arithmetic failure");}}
 unsigned long long state=42;
 for(int i=0;i<3000;++i){state=state*6364136223846793005ULL+1;T x=T(int64_t(state)>>20)/T(1000000);state=state*6364136223846793005ULL+1;T y=T(int64_t(state)>>20)/T(1000000);I<T>z=I<T>(x)*I<T>(y);mpq_class q=mpq_class(double(x))*mpq_class(double(y));need(mpq_class(double(z.l))<=q&&q<=mpq_class(double(z.h)),"random multiply");}
 I<T>s=sq(I<T>(T(-2),T(3)));need(s.l==0&&s.h>=9,"square crossing zero");
}
void selftest(){
 arithmetic_tests<float>();arithmetic_tests<double>();
 Box b;Geo g=geometry(b);M p=M::pi();auto a=g.A.native<double>(),total=(p*rat(3,4)).native<double>();need(a.l<=total.h&&total.l<=a.h,"area");
 auto check=[](const M&m,const mpq_class&q){M exact(q);need(mpfr_cmp_q(m.l,q.get_mpq_t())<=0&&mpfr_cmp_q(m.h,q.get_mpq_t())>=0,"moment rational");};
 check(g.m0,mpq_class(16,3));check(g.m1,mpq_class(-64,45));check(g.m2,mpq_class(64,45));b.kind=1;Geo h=geometry(b);check(h.H0,mpq_class(1));check(h.H1,mpq_class(4,15));
 // Affine x and constant cases use independently derived whole-domain integrals.
 for(int kind=0;kind<3;++kind){Model m{};for(auto&bb:m.b)bb=3;if(kind==1)m.b0=1;if(kind==2){m.w[0][0]=1;m.a[0]=1;m.b0=-3;}
  Options o;o.seconds=1;Engine<double,double>e(m,o);auto t=e.total();M exact=kind==0?p*rat(1,2):(kind==1?p*M(2):p*rat(19,8));auto r=exact.native<double>();need(t.l<=r.h&&r.l<=t.h,"whole-domain estimator");
  if(kind==2)need(e.affine_i==4&&e.affine_b==6,"affine acceptance");
  Engine<float,float>ef(m,o);auto tf=ef.total();need(double(tf.l)<=r.h&&r.l<=double(tf.h),"float32 whole-domain estimator");
  Engine<float,double>eh(m,o);auto th=eh.total();need(th.l<=r.h&&r.l<=th.h,"hybrid whole-domain estimator");
 }
 // Extrema inside angular intervals must be enclosed.
 auto sine=sinM(span(M(0),p)).native<double>();need(sine.h>=1,"missed sine extremum");
 std::cout<<"All arithmetic, affine-network, area, moment, and trigonometric tests passed.\n";
}
int main(int argc,char**argv){try{
 if(argc==2&&std::string(argv[1])=="--selftest"){selftest();return 0;}
 Options o;for(int i=1;i<argc;++i){std::string k=argv[i];if(k=="--general"){o.detect=false;continue;}if(i+1>=argc)throw std::runtime_error("missing option value");std::string v=argv[++i];if(k=="--model")o.model=v;else if(k=="--precision")o.precision=v;else if(k=="--seconds")o.seconds=std::stod(v);else if(k=="--leaves")o.leaves=std::stoull(v);else if(k=="--boundary")o.boundary=std::stoull(v);else if(k=="--depth")o.depth=std::stoi(v);else if(k=="--splits")o.splits=std::stoull(v);else if(k=="--history-out")o.history_out=v;else if(k=="--history-in")o.history_in=v;else throw std::runtime_error("unknown option "+k);}
 need(std::fegetround()==FE_TONEAREST,"rounding mode changed");auto start=Clock::now();Model m=load(o.model); // Model load is added to reported timing below by constructor start override.
 if(o.precision=="32"){Engine<float,float>e(m,o);e.start=start;e.run();}else if(o.precision=="64"){Engine<double,double>e(m,o);e.start=start;e.run();}else if(o.precision=="32/64"){Engine<float,double>e(m,o);e.start=start;e.run();}else throw std::runtime_error("unknown precision");
 return 0;
 }catch(const std::exception&e){std::cerr<<"VERIFIER FAILURE: "<<e.what()<<'\n';return 2;}}
