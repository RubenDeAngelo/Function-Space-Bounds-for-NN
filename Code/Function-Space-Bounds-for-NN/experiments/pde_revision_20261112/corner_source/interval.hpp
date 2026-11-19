#pragma once
#include <cmath>
#include <limits>
#include <stdexcept>
#include <algorithm>
#include <cfenv>
#include <mpfr.h>
#include <gmpxx.h>
#include <array>
#include <vector>
// Assumptions checked by the build/tests: IEEE binary32/binary64, round-to-nearest,
// source-type rounding, gradual underflow, no reassociation or FMA contraction.
// Each basic result is padded by one representable neighbour on the relevant side.
// This encloses a correctly rounded operation even when it rounded toward that side.
template<class T>struct I{
 T l=0,h=0;
 I()=default;explicit I(T a):l(a),h(a){} I(T a,T b):l(a),h(b){check();}
 void check()const{if(std::isnan(l)||std::isnan(h)||l>h)throw std::runtime_error("invalid interval");}
 bool zero()const{return l==0&&h==0;}
};
template<class T>T dn(T a){if(std::isnan(a))throw std::runtime_error("NaN operation");return std::nextafter(a,-std::numeric_limits<T>::infinity());}
template<class T>T up(T a){if(std::isnan(a))throw std::runtime_error("NaN operation");return std::nextafter(a,std::numeric_limits<T>::infinity());}
template<class T>I<T> operator+(I<T>a,I<T>b){if(a.zero())return b;if(b.zero())return a;return {dn(T(a.l+b.l)),up(T(a.h+b.h))};}
template<class T>I<T> operator-(I<T>a){return {-a.h,-a.l};}
template<class T>I<T> operator-(I<T>a,I<T>b){return a+(-b);}
template<class T>I<T> operator*(I<T>a,I<T>b){if(a.zero()||b.zero())return I<T>(0);T lo=std::numeric_limits<T>::infinity(),hi=-lo;for(T x:{a.l,a.h})for(T y:{b.l,b.h}){T z=T(x*y);lo=std::min(lo,dn(z));hi=std::max(hi,up(z));}return {lo,hi};}
template<class T>I<T> sq(I<T>a){if(a.zero())return a;T m=std::max(std::abs(a.l),std::abs(a.h));T n=a.l<=0&&a.h>=0?0:std::min(std::abs(a.l),std::abs(a.h));return {n==0?T(0):std::max(T(0),dn(T(n*n))),up(T(m*m))};}
template<class T>I<T> intersect(I<T>a,I<T>b){return {std::max(a.l,b.l),std::min(a.h,b.h)};}
template<class T>mpq_class width(I<T>a){if(!std::isfinite(a.l)||!std::isfinite(a.h))throw std::runtime_error("unbounded priority");return mpq_class(double(a.h))-mpq_class(double(a.l));}
// MPFR interval service: every arithmetic result has explicitly directed endpoints.
struct M{
 mpfr_t l,h;
 M(){mpfr_init2(l,256);mpfr_init2(h,256);mpfr_set_zero(l,1);mpfr_set_zero(h,1);}
 M(long n):M(){mpfr_set_si(l,n,MPFR_RNDD);mpfr_set_si(h,n,MPFR_RNDU);}
 M(const mpq_class&q):M(){mpfr_set_q(l,q.get_mpq_t(),MPFR_RNDD);mpfr_set_q(h,q.get_mpq_t(),MPFR_RNDU);}
 M(const M&a):M(){mpfr_set(l,a.l,MPFR_RNDD);mpfr_set(h,a.h,MPFR_RNDU);}
 M&operator=(const M&a){if(this!=&a){mpfr_set(l,a.l,MPFR_RNDD);mpfr_set(h,a.h,MPFR_RNDU);}return *this;}
 ~M(){mpfr_clear(l);mpfr_clear(h);}
 static M pi(){M r;mpfr_const_pi(r.l,MPFR_RNDD);mpfr_const_pi(r.h,MPFR_RNDU);return r;}
 template<class T>I<T> native()const{if constexpr(sizeof(T)==4)return {mpfr_get_flt(l,MPFR_RNDD),mpfr_get_flt(h,MPFR_RNDU)};else return {mpfr_get_d(l,MPFR_RNDD),mpfr_get_d(h,MPFR_RNDU)};}
};
inline M operator+(const M&a,const M&b){M r;mpfr_add(r.l,a.l,b.l,MPFR_RNDD);mpfr_add(r.h,a.h,b.h,MPFR_RNDU);return r;}
inline M operator-(const M&a){M r;mpfr_neg(r.l,a.h,MPFR_RNDD);mpfr_neg(r.h,a.l,MPFR_RNDU);return r;}
inline M operator-(const M&a,const M&b){return a+(-b);}
inline M operator*(const M&a,const M&b){M r;mpfr_set_inf(r.l,1);mpfr_set_inf(r.h,-1);mpfr_t t;mpfr_init2(t,256);for(auto x:{a.l,a.h})for(auto y:{b.l,b.h}){mpfr_mul(t,x,y,MPFR_RNDD);if(mpfr_less_p(t,r.l))mpfr_set(r.l,t,MPFR_RNDD);mpfr_mul(t,x,y,MPFR_RNDU);if(mpfr_greater_p(t,r.h))mpfr_set(r.h,t,MPFR_RNDU);}mpfr_clear(t);return r;}
inline M rat(long a,long b=1){return M(mpq_class(a,b));}
inline M ipow(M a,int n){M r(1);for(int i=0;i<n;++i)r=r*a;return r;}
inline M cbrtM(const M&a){M r;mpfr_cbrt(r.l,a.l,MPFR_RNDD);mpfr_cbrt(r.h,a.h,MPFR_RNDU);return r;}
inline M trig(const M&a,bool cosine){M r;auto fn=cosine?mpfr_cos:mpfr_sin;fn(r.l,a.l,MPFR_RNDD);fn(r.h,a.l,MPFR_RNDU);mpfr_t t;mpfr_init2(t,256);fn(t,a.h,MPFR_RNDD);if(mpfr_less_p(t,r.l))mpfr_set(r.l,t,MPFR_RNDD);fn(t,a.h,MPFR_RNDU);if(mpfr_greater_p(t,r.h))mpfr_set(r.h,t,MPFR_RNDU);mpfr_clear(t);
 // All callers use angles within [-16,16]. Critical points for both functions
 // are covered by k*pi/2, -12<=k<=12. Include extrema on uncertain overlap.
 if(mpfr_cmp_si(a.l,-16)<0||mpfr_cmp_si(a.h,16)>0)throw std::runtime_error("trig range outside proved domain");
 static const auto critical=[](){std::vector<M>v;M p=M::pi();for(int k=-12;k<=12;++k)v.push_back(p*rat(k,2));return v;}();
 for(int k=-12;k<=12;++k){bool relevant=cosine?(k%2==0):(k%2!=0);if(!relevant)continue;const M&c=critical[k+12];if(mpfr_cmp(a.l,c.h)<=0&&mpfr_cmp(a.h,c.l)>=0){int k4=(k%4+4)%4;bool max=cosine?k4==0:k4==1;if(max)mpfr_set_si(r.h,1,MPFR_RNDU);else mpfr_set_si(r.l,-1,MPFR_RNDD);}}
 return r;}
inline M sinM(const M&a){return trig(a,false);}inline M cosM(const M&a){return trig(a,true);}
inline M span(const M&a,const M&b){M r;mpfr_set(r.l,a.l,MPFR_RNDD);mpfr_set(r.h,b.h,MPFR_RNDU);return r;}
inline mpq_class dyadic(uint64_t n,unsigned d){mpq_class q;mpz_class den(1);den<<=d;q=mpq_class(mpz_class(static_cast<unsigned long>(n)),den);return q;}
