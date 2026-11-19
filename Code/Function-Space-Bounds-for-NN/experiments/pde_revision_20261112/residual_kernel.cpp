// Directed 128-bit MPFR enclosure of 2D tanh network residual integrals.
// Model coefficients are required to be represented exactly at working precision.
#include <mpfr.h>
#include <gmpxx.h>
#include <vector>
#include <array>
#include <queue>
#include <fstream>
#include <iostream>
#include <chrono>
#include <cassert>
#include <string>
#include <algorithm>
#include <stdexcept>
using Q=mpq_class;
struct I {
 mpfr_t l,h;
 I(){mpfr_init2(l,128);mpfr_init2(h,128);mpfr_set_zero(l,1);mpfr_set_zero(h,1);}
 I(long x):I(){mpfr_set_si(l,x,MPFR_RNDD);mpfr_set_si(h,x,MPFR_RNDU);}
 I(Q const&q):I(){mpfr_set_q(l,q.get_mpq_t(),MPFR_RNDD);mpfr_set_q(h,q.get_mpq_t(),MPFR_RNDU);}
 I(I const&a):I(){mpfr_set(l,a.l,MPFR_RNDD);mpfr_set(h,a.h,MPFR_RNDU);}
 I&operator=(I const&a){if(this!=&a){mpfr_set(l,a.l,MPFR_RNDD);mpfr_set(h,a.h,MPFR_RNDU);}return *this;}
 ~I(){mpfr_clear(l);mpfr_clear(h);}
 Q low()const{Q q;mpfr_get_q(q.get_mpq_t(),l);return q;}
 Q high()const{Q q;mpfr_get_q(q.get_mpq_t(),h);return q;}
 void check()const{if(!mpfr_number_p(l)||!mpfr_number_p(h)||mpfr_greater_p(l,h))throw std::runtime_error("invalid interval");}
};
I operator+(I const&a,I const&b){I r;mpfr_add(r.l,a.l,b.l,MPFR_RNDD);mpfr_add(r.h,a.h,b.h,MPFR_RNDU);return r;}
I operator-(I const&a){I r;mpfr_neg(r.l,a.h,MPFR_RNDD);mpfr_neg(r.h,a.l,MPFR_RNDU);return r;}
I operator-(I const&a,I const&b){return a+(-b);}
I operator*(I const&a,I const&b){I r;mpfr_set_inf(r.l,1);mpfr_set_inf(r.h,-1);mpfr_t t;mpfr_init2(t,128);for(auto x:{a.l,a.h})for(auto y:{b.l,b.h}){mpfr_mul(t,x,y,MPFR_RNDD);if(mpfr_less_p(t,r.l))mpfr_set(r.l,t,MPFR_RNDD);mpfr_mul(t,x,y,MPFR_RNDU);if(mpfr_greater_p(t,r.h))mpfr_set(r.h,t,MPFR_RNDU);}mpfr_clear(t);return r;}
I sq(I const&a){I r;mpfr_t t;mpfr_init2(t,128);mpfr_sqr(r.l,a.l,MPFR_RNDD);mpfr_sqr(t,a.h,MPFR_RNDD);if(mpfr_less_p(t,r.l))mpfr_set(r.l,t,MPFR_RNDD);if(mpfr_sgn(a.l)<=0&&mpfr_sgn(a.h)>=0)mpfr_set_zero(r.l,1);mpfr_sqr(r.h,a.l,MPFR_RNDU);mpfr_sqr(t,a.h,MPFR_RNDU);if(mpfr_greater_p(t,r.h))mpfr_set(r.h,t,MPFR_RNDU);mpfr_clear(t);return r;}
I expI(I const&a){I r;mpfr_exp(r.l,a.l,MPFR_RNDD);mpfr_exp(r.h,a.h,MPFR_RNDU);return r;}
I sin01(I const&a){if(mpfr_sgn(a.l)<0||mpfr_cmp_ui(a.h,1)>0)throw std::runtime_error("sin domain");I r;mpfr_sin(r.l,a.l,MPFR_RNDD);mpfr_sin(r.h,a.h,MPFR_RNDU);return r;}
I cos01(I const&a){if(mpfr_sgn(a.l)<0||mpfr_cmp_ui(a.h,1)>0)throw std::runtime_error("cos domain");I r;mpfr_cos(r.l,a.h,MPFR_RNDD);mpfr_cos(r.h,a.l,MPFR_RNDU);return r;}
std::array<I,3> tanhJet(I const&z){I t;mpfr_tanh(t.l,z.l,MPFR_RNDD);mpfr_tanh(t.h,z.h,MPFR_RNDU);I first=I(1)-sq(t);I second=I(-2)*t*first;return {t,first,second};}
// Components: value, x derivative, y derivative, xx derivative, yy derivative.
using Jet=std::array<I,5>;
struct Layer{std::vector<std::vector<I>>w;std::vector<I>b;};
struct Network{
 std::vector<Layer>ls;
 Network(std::string path){std::ifstream f(path);unsigned L=0,previous=2;if(!(f>>L)||L==0)throw std::runtime_error("invalid model header");for(unsigned k=0;k<L;k++){unsigned m=0,n=0;if(!(f>>m>>n)||m!=previous||n==0)throw std::runtime_error("invalid layer dimensions");previous=n;Layer a;a.w.resize(n);a.b.resize(n);for(unsigned i=0;i<n;i++){for(unsigned j=0;j<m;j++){Q q;f>>q;I c(q);if(c.low()!=q||c.high()!=q)throw std::runtime_error("inexact model coefficient");a.w[i].push_back(c);}Q b;f>>b;a.b[i]=I(b);if(a.b[i].low()!=b||a.b[i].high()!=b)throw std::runtime_error("inexact bias");}ls.push_back(a);}if(!f||ls.front().w.front().size()!=2||ls.back().w.size()!=1)throw std::runtime_error("invalid model");}
 Jet eval(I const&x,I const&y)const{
  std::vector<Jet>v(2);v[0][0]=x;v[1][0]=y;v[0][1]=I(1);v[1][2]=I(1);
  for(unsigned k=0;k<ls.size();k++){auto const&a=ls[k];std::vector<Jet>out(a.w.size());for(unsigned i=0;i<out.size();i++){
   out[i][0]=a.b[i];for(unsigned j=0;j<v.size();j++){auto const&c=a.w[i][j];bool pos=mpfr_sgn(c.l)>=0;for(int p=0;p<5;p++){
    // c is an exact scalar; directed fused multiply-add encloses each sum.
    mpfr_fma(out[i][p].l,c.l,pos?v[j][p].l:v[j][p].h,out[i][p].l,MPFR_RNDD);
    mpfr_fma(out[i][p].h,c.h,pos?v[j][p].h:v[j][p].l,out[i][p].h,MPFR_RNDU);
   }}
   if(k+1<ls.size()){auto t=tanhJet(out[i][0]);out[i][3]=t[2]*sq(out[i][1])+t[1]*out[i][3];out[i][4]=t[2]*sq(out[i][2])+t[1]*out[i][4];out[i][1]=t[1]*out[i][1];out[i][2]=t[1]*out[i][2];out[i][0]=t[0];}
  }v=std::move(out);}for(auto const&a:v[0])a.check();return v[0];
 }
};
I forceCDR(I const&x,I const&y){I ex=expI(I(-2)*(I(1)-x)),ey=expI(I(-3)*(I(1)-y));I sx=sin01(x),cx=cos01(x);I a=sx*(I(1)-ex),ax=cx*(I(1)-ex)-I(2)*sx*ex,axx=-sx*(I(1)-ex)-I(4)*cx*ex-I(4)*sx*ex;I b=sq(y)*(I(1)-ey),by=I(2)*y*(I(1)-ey)-I(3)*sq(y)*ey,byy=I(2)*(I(1)-ey)-I(12)*y*ey-I(9)*sq(y)*ey;return -(axx*b+a*byy)+(I(3)-x)*ax*b+(I(4)-y)*a*by+a*b;}
I residual(Network const&net,I const&x,I const&y,std::string const&kind){Jet v=net.eval(x,y);I r;if(kind=="poisson"){auto t=tanhJet(x+y-I(1));r=v[3]+v[4]-I(2)*t[2];}else if(kind=="cdr"){r=-(v[3]+v[4])+(I(3)-x)*v[1]+(I(4)-y)*v[2]+v[0]-forceCDR(x,y);}else throw std::runtime_error("kind");return sq(r);}
struct Cell{unsigned d;unsigned long x,y,id;Q rl,rh,lo,hi;Q gap()const{return hi-lo;}};
struct Order{bool operator()(Cell const&a,Cell const&b)const{Q ga=a.gap(),gb=b.gap();return ga==gb?a.id>b.id:ga<gb;}};
int main(int argc,char**argv){try{
 if(argc<4)throw std::runtime_error("model kind oracle|policy [cap output tolerance]");Network net(argv[1]);std::string kind=argv[2],policy=argv[3];
 if(policy=="oracle"){std::string a,b,c,d;while(std::cin>>a>>b>>c>>d){I r;
 I x,y;I xl{Q(a)},xu{Q(b)},yl{Q(c)},yu{Q(d)};mpfr_set(x.l,xl.l,MPFR_RNDD);mpfr_set(x.h,xu.h,MPFR_RNDU);mpfr_set(y.l,yl.l,MPFR_RNDD);mpfr_set(y.h,yu.h,MPFR_RNDU);x.check();y.check();if(mpfr_sgn(x.l)<0||mpfr_cmp_ui(x.h,1)>0||mpfr_sgn(y.l)<0||mpfr_cmp_ui(y.h,1)>0)throw std::runtime_error("oracle outside unit square");r=residual(net,x,y,kind);std::cout<<r.low()<<' '<<r.high()<<'\n';}return 0;}
 if(argc!=7||(policy!="maxgap"&&policy!="uniform"))throw std::runtime_error("arguments");unsigned long cap=std::stoul(argv[4]);Q tolerance(argv[6]);if(cap<1||(cap-1)%3||tolerance<0)throw std::runtime_error("invalid cap or tolerance");if(policy=="uniform"){unsigned long n=cap;while(n>1&&n%4==0)n/=4;if(n!=1)throw std::runtime_error("uniform cap must be a power of four");}std::ofstream out(argv[5]);if(!out)throw std::runtime_error("cannot write history");auto start=std::chrono::steady_clock::now();unsigned long calls=0,uid=1;
 auto cell=[&](unsigned d,unsigned long ix,unsigned long iy,unsigned long id,Cell const*p){mpz_class den=mpz_class(1)<<d;I x,y;mpfr_set_q(x.l,Q(mpz_class(ix),den).get_mpq_t(),MPFR_RNDD);mpfr_set_q(x.h,Q(mpz_class(ix+1),den).get_mpq_t(),MPFR_RNDU);mpfr_set_q(y.l,Q(mpz_class(iy),den).get_mpq_t(),MPFR_RNDD);mpfr_set_q(y.h,Q(mpz_class(iy+1),den).get_mpq_t(),MPFR_RNDU);I f=residual(net,x,y,kind);Q l=f.low(),h=f.high();if(p){l=std::max(l,p->rl);h=std::min(h,p->rh);}if(l<0||l>h)throw std::runtime_error("range");Q vol(1,den*den);calls++;return Cell{d,ix,iy,id,l,h,l*vol,h*vol};};
 Cell root=cell(0,0,0,0,nullptr);Q lo=root.lo,hi=root.hi;unsigned long N=1,next=4;std::priority_queue<Cell,std::vector<Cell>,Order>heap;std::vector<Cell>level;heap.push(root);level.push_back(root);
 out<<"leaves,evaluations,lower,upper,gap,seconds\n";auto snap=[&](){out<<N<<','<<calls<<','<<lo<<','<<hi<<','<<Q(hi-lo)<<','<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<'\n';out.flush();std::cerr<<kind<<' '<<policy<<" N="<<N<<" gap="<<Q(hi-lo).get_d()<<std::endl;};snap();
 while(N<cap&&hi-lo>tolerance){
  if(policy=="maxgap"){
   Cell p=heap.top();heap.pop();if(p.d>=50)throw std::runtime_error("depth limit");Q cl=0,ch=0;for(unsigned i=0;i<2;i++)for(unsigned j=0;j<2;j++){Cell c=cell(p.d+1,2*p.x+i,2*p.y+j,uid++,&p);cl+=c.lo;ch+=c.hi;heap.push(c);}if(cl<p.lo||ch>p.hi)throw std::runtime_error("nesting");lo+=cl-p.lo;hi+=ch-p.hi;N+=3;if(N>=next||N==cap||hi-lo<=tolerance){snap();if(N>=next)next*=4;}
  }else{std::vector<Cell>nl;for(auto const&p:level){Q cl=0,ch=0;for(unsigned i=0;i<2;i++)for(unsigned j=0;j<2;j++){Cell c=cell(p.d+1,2*p.x+i,2*p.y+j,uid++,&p);cl+=c.lo;ch+=c.hi;nl.push_back(c);}if(cl<p.lo||ch>p.hi)throw std::runtime_error("nesting");lo+=cl-p.lo;hi+=ch-p.hi;}level=std::move(nl);N*=4;snap();}
 }
 std::ofstream meta(std::string(argv[5])+".meta");if(!meta||!out)throw std::runtime_error("output failure");meta<<"status "<<(hi-lo<=tolerance?"tolerance_reached":"leaf_budget_exhausted")<<"\nbits 128\nleaves "<<N<<"\nevaluations "<<calls<<"\n";
 }catch(std::exception const&e){std::cerr<<e.what()<<std::endl;return 1;}}
