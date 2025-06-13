/// @file NumVec.hpp
/// @brief  Numerical vector.
/// @date 2023-07-01
#ifndef _PPCG_NUMVEC_HPP_
#define _PPCG_NUMVEC_HPP_

#include "ppcg/environment.hpp"

namespace  PPCG {
/// @class NumVec
///
/// @brief Numerical vector.
/// 
/// NumVec is a portable encapsulation of a pointer to represent a 1D
/// vector. The main difference between NumVec<F> and std::vector<F> is
/// that NumVec<F> allows the vector to not owning the data, by
/// specifying (owndata_ == false).
template <class F> class NumVec
{
public:
  Int  m_;                                // The number of elements 
  bool owndata_;                          // Whether it owns the data
  F* data_;                               // The pointer for the actual data
public:
  NumVec(Int m = 0);
  NumVec(Int m, bool owndata, F* data);
  NumVec(const NumVec& C);
  ~NumVec();

  NumVec& operator=(const NumVec& C);

  void Resize ( Int m );

  const F& operator()(Int i) const;  
  F& operator()(Int i);  
  const F& operator[](Int i) const;
  F& operator[](Int i);

  bool IsOwnData() const { return owndata_; }

  F*   Data() const { return data_; }

  Int  m () const { return m_; }

  Int Size() const { return m_; }
};

// Commonly used
typedef NumVec<bool>       BolNumVec;
typedef NumVec<Int>        IntNumVec;
typedef NumVec<Real>       DblNumVec;
typedef NumVec<Complex>    CpxNumVec;


// Utilities
template <class F> inline void SetValue( NumVec<F>& vec, F val );
template <class F> inline Real Energy( const NumVec<F>& vec );
template <class F> inline Real findMin( const NumVec<F>& vec );
template <class F> inline Real findMax( const NumVec<F>& vec );
template <class F> inline void Sort( NumVec<F>& vec );

} // namespace PPCG

# include "ppcg/NumVec_impl.hpp"

#endif // _PPCG_NUMVEC_HPP_
