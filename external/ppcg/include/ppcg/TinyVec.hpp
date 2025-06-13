/// @file tinyvec.hpp
/// @brief Tiny vectors of dimension 3.
/// @date 2023-07-01
#ifndef  _PPCG_TINYVEC_HPP_
#define  _PPCG_TINYVEC_HPP_

#include "ppcg/environment.hpp"

namespace PPCG {

/// @class Vec3T
/// 
/// @brief Tiny vectors of dimension 3.
///
/// The main use of tiny vectors is to represent a triplet of three
/// indices (Index3) and three real numbers (Point3). 
template <class F> class Vec3T {
private:
  F v_[3];
public:
  enum{ X=0, Y=1, Z=2 };
  //------------CONSTRUCTOR AND DESTRUCTOR 
  Vec3T()              { v_[0]=F(0);    v_[1]=F(0);    v_[2]=F(0); }
  Vec3T(const F* f)    { v_[0]=f[0];    v_[1]=f[1];    v_[2]=f[2]; }
  Vec3T(const F a, const F b, const F c)   { v_[0]=a;       v_[1]=b;       v_[2]=c; }
  Vec3T(const Vec3T& c){ v_[0]=c.v_[0]; v_[1]=c.v_[1]; v_[2]=c.v_[2]; }
  ~Vec3T() {}
  //------------POINTER and ACCESS
  operator F*()             { return &v_[0]; }
  operator const F*() const { return &v_[0]; }
  F* Data()                 { return &v_[0]; }  //access array
  F& operator()(Int i);
  const F& operator()(Int i) const;
  F& operator[](Int i);
  const F& operator[](Int i) const;
  //------------ASSIGN
  Vec3T& operator= ( const Vec3T& c ) { v_[0] =c.v_[0]; v_[1] =c.v_[1]; v_[2] =c.v_[2]; return *this; }
  Vec3T& operator+=( const Vec3T& c ) { v_[0]+=c.v_[0]; v_[1]+=c.v_[1]; v_[2]+=c.v_[2]; return *this; }
  Vec3T& operator-=( const Vec3T& c ) { v_[0]-=c.v_[0]; v_[1]-=c.v_[1]; v_[2]-=c.v_[2]; return *this; }
  Vec3T& operator*=( const F& s )     { v_[0]*=s;       v_[1]*=s;       v_[2]*=s;       return *this; }
  Vec3T& operator/=( const F& s )     { v_[0]/=s;       v_[1]/=s;       v_[2]/=s;       return *this; }
  //-----------LENGTH
  F l1( void )     const  { F sum=F(0); for(Int i=0; i<3; i++) sum=sum+std::abs(v_[i]); return sum; }
  F linfty( void ) const  { F cur=F(0); for(Int i=0; i<3; i++) cur=std::max(cur,std::abs(v_[i])); return cur; }
  F l2( void )     const  { F sum=F(0); for(Int i=0; i<3; i++) sum=sum+std::abs(v_[i]*v_[i]); return std::sqrt(sum); }
  F prod( void )   const  { return v_[0]*v_[1]*v_[2]; }
};

// Commonly used Vec3T types

typedef Vec3T<Real>   Point3;
typedef Vec3T<Int>    Index3;

} // namespace PPCG

#include "ppcg/TinyVec_impl.hpp"

#endif // _PPCG_TINYVEC_HPP_
