/// @file tinyvec_impl.hpp
/// @brief Implementation of tiny vectors of dimension 2 and 3.
/// @date 2023-07-01
#ifndef  _PPCG_TINYVEC_IMPL_HPP_
#define  _PPCG_TINYVEC_IMPL_HPP_

namespace PPCG{

// *********************************************************************
// Tiny vectors of dimension 3.
// *********************************************************************

template <class F> 
  inline F&
  Vec3T<F>::operator() ( Int i ) 
  {
    if( i < 0 || i > 2 ){
      ErrorHandling( "Index is out of bound." );
    }
    return v_[i];
  }         // -----  end of method Vec3T::operator()  ----- 

template <class F> 
  inline const F&
  Vec3T<F>::operator() ( Int i ) const
  {
    if( i < 0 || i > 2 ){
      ErrorHandling( "Index is out of bound." );
    }
    return v_[i];
  }         // -----  end of method Vec3T::operator()  ----- 

template <class F> 
  inline F&
  Vec3T<F>::operator[] ( Int i ) 
  {
    if( i < 0 || i > 2 ){
      ErrorHandling( "Index is out of bound." );
    }
    return v_[i];
  }         // -----  end of method Vec3T::operator[]  ----- 

template <class F> 
  inline const F&
  Vec3T<F>::operator[] ( Int i ) const
  {
    if( i < 0 || i > 2 ){
      ErrorHandling( "Index is out of bound." );
    }
    return v_[i];
  }         // -----  end of method Vec3T::operator[]  ----- 

// *********************************************************************
// Vec3T: Compare
// *********************************************************************
template <class F> inline bool operator==(const Vec3T<F>& a, const Vec3T<F>& b) {
  return (a[0]==b[0] && a[1]==b[1] && a[2]==b[2]);
}
template <class F> inline bool operator!=(const Vec3T<F>& a, const Vec3T<F>& b) {
  return !(a==b);
}
template <class F> inline bool operator> (const Vec3T<F>& a, const Vec3T<F>& b) {
  for(Int i=0; i<3; i++) {
    if(     a[i]>b[i])      return true;
    else if(a[i]<b[i])      return false;
  }
  return false;
}
template <class F> inline bool operator< (const Vec3T<F>& a, const Vec3T<F>& b) {
  for(Int i=0; i<3; i++) {
    if(     a[i]<b[i])      return true;
    else if(a[i]>b[i])      return false;
  }
  return false;
}
template <class F> inline bool operator>=(const Vec3T<F>& a, const Vec3T<F>& b) {
  for(Int i=0; i<3; i++) {
    if(     a[i]>b[i])      return true;
    else if(a[i]<b[i])      return false;
  }
  return true;
}
template <class F> inline bool operator<=(const Vec3T<F>& a, const Vec3T<F>& b) {
  for(Int i=0; i<3; i++) {
    if(     a[i]<b[i])      return true;
    else if(a[i]>b[i])      return false;
  }
  return true;
}

// *********************************************************************
// Vec3T: Numerical operations
// *********************************************************************
template <class F> inline Vec3T<F> operator- (const Vec3T<F>& a) {
  Vec3T<F> r;  for(Int i=0; i<3; i++) r[i] = -a[i]; return r;
}
template <class F> inline Vec3T<F> operator+ (const Vec3T<F>& a, const Vec3T<F>& b) {
  Vec3T<F> r;  for(Int i=0; i<3; i++) r[i] = a[i]+b[i]; return r; 
}
template <class F> inline Vec3T<F> operator- (const Vec3T<F>& a, const Vec3T<F>& b) {
  Vec3T<F> r;  for(Int i=0; i<3; i++) r[i] = a[i]-b[i]; return r;
}
template <class F> inline Vec3T<F> operator* (F scl, const Vec3T<F>& a) {
  Vec3T<F> r;  for(Int i=0; i<3; i++) r[i] = scl*a[i];  return r;
}
template <class F> inline Vec3T<F> operator* (const Vec3T<F>& a, F scl) {
  Vec3T<F> r;  for(Int i=0; i<3; i++) r[i] = scl*a[i];  return r;
}
template <class F> inline Vec3T<F> operator/ (const Vec3T<F>& a, F scl) {
  Vec3T<F> r;  for(Int i=0; i<3; i++) r[i] = a[i]/scl;  return r;
}
template <class F> inline F operator* (const Vec3T<F>& a, const Vec3T<F>& b) {
  F sum=F(0); for(Int i=0; i<3; i++) sum=sum+a(i)*b(i); return sum;
}
template <class F> inline F dot       (const Vec3T<F>& a, const Vec3T<F>& b) {
  return a*b;
}
template <class F> inline Vec3T<F> operator^ (const Vec3T<F>& a, const Vec3T<F>& b) {
  return Vec3T<F>(a(1)*b(2)-a(2)*b(1), a(2)*b(0)-a(0)*b(2), a(0)*b(1)-a(1)*b(0));
}
template <class F> inline Vec3T<F> cross     (const Vec3T<F>& a, const Vec3T<F>& b) { 
  return a^b; 
}

// *********************************************************************
// Vec3T: Element wise numerical operations
// *********************************************************************
template <class F> inline Vec3T<F> ewmin(const Vec3T<F>& a, const Vec3T<F>& b) {
  Vec3T<F> r;  for(Int i=0; i<3; i++) r[i] = std::min(a[i], b[i]); return r;
}
template <class F> inline Vec3T<F> ewmax(const Vec3T<F>& a, const Vec3T<F>& b) {
  Vec3T<F> r;  for(Int i=0; i<3; i++) r[i] = std::max(a[i], b[i]); return r;
}
template <class F> inline Vec3T<F> ewabs(const Vec3T<F>& a) {
  Vec3T<F> r;  for(Int i=0; i<3; i++) r[i] = std::abs(a[i]); return r;
}
template <class F> inline Vec3T<F> ewmul(const Vec3T<F>&a, const Vec3T<F>& b) {
  Vec3T<F> r;  for(Int i=0; i<3; i++) r[i] = a[i]*b[i]; return r;
}
template <class F> inline Vec3T<F> ewdiv(const Vec3T<F>&a, const Vec3T<F>& b) { 
  Vec3T<F> r;  for(Int i=0; i<3; i++) r[i] = a[i]/b[i]; return r;
}
template <class F> inline Vec3T<F> ewrnd(const Vec3T<F>&a) { //round
  Vec3T<F> r;  for(Int i=0; i<3; i++)    r[i] = round(a[i]);  return r;
}

// *********************************************************************
// Vec3T: Accumulative boolean operations
// *********************************************************************
template <class F> inline bool allequ(const Vec3T<F>& a, const Vec3T<F>& b) {
  bool res = true;  for(Int i=0; i<3; i++)   res = res && (a(i)==b(i));  return res;
}
template <class F> inline bool allneq(const Vec3T<F>& a, const Vec3T<F>& b) {
  return !(a==b);
}
template <class F> inline bool allgtt(const Vec3T<F>& a, const Vec3T<F>& b) {
  bool res = true;  for(Int i=0; i<3; i++)   res = res && (a(i)> b(i));  return res; 
}
template <class F> inline bool alllst(const Vec3T<F>& a, const Vec3T<F>& b) {
  bool res = true;  for(Int i=0; i<3; i++)   res = res && (a(i)< b(i));  return res; 
}
template <class F> inline bool allgoe(const Vec3T<F>& a, const Vec3T<F>& b) {
  bool res = true;  for(Int i=0; i<3; i++)    res = res && (a(i)>=b(i));  return res; 
}
template <class F> inline bool allloe(const Vec3T<F>& a, const Vec3T<F>& b) {
  bool res = true;  for(Int i=0; i<3; i++)   res = res && (a(i)<=b(i));  return res; 
}


// *********************************************************************
// Vec3T: Input and output
// *********************************************************************
template <class F> std::istream& operator>>(std::istream& is, Vec3T<F>& a) {
  for(Int i=0; i<3; i++) is>>a[i]; return is;
}
template <class F> std::ostream& operator<<(std::ostream& os, const Vec3T<F>& a) { 
  for(Int i=0; i<3; i++) os<<a[i]<<" "; return os;
}

} // namespace PPCG

#endif // _PPCG_TINYVEC_IMPL_HPP_
