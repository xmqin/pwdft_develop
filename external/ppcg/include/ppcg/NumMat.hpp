/// @file NumMat.hpp
/// @brief Numerical matrix.
/// @date 2023-07-01
#ifndef _PPCG_NUMMAT_HPP_
#define _PPCG_NUMMAT_HPP_

#include "ppcg/environment.hpp"

namespace  PPCG{
/// @class NumMat
///
/// @brief Numerical matrix.
///
/// NumMat is a portable encapsulation of a pointer to represent a 2D
/// matrix, which can either own (owndata == true) or view (owndata ==
/// false) a piece of data. 

template <class F>
  class NumMat
  {
  public:
    Int m_, n_;
    bool owndata_;
    F* data_;
  public:
    NumMat(Int m=0, Int n=0);

    NumMat(Int m, Int n, bool owndata, F* data);

    NumMat(const NumMat& C);

    ~NumMat();

    NumMat& operator=(const NumMat& C);

    void Resize(Int m, Int n);

    const F& operator()(Int i, Int j) const;  

    F& operator()(Int i, Int j);  

    bool IsOwnData() const { return owndata_; }

    F* Data() const { return data_; }

    F* VecData(Int j)  const; 

    Int m() const { return m_; }

    Int n() const { return n_; }

    Int Size() const { return m_ * n_; }

  };

// Commonly used
typedef NumMat<bool>     BolNumMat;
typedef NumMat<Int>      IntNumMat;
typedef NumMat<Real>     DblNumMat;
typedef NumMat<Complex>  CpxNumMat;

// Utilities
template <class F> inline void SetValue(NumMat<F>& M, F val);
template <class F> inline Real Energy(const NumMat<F>& M);
template <class F> inline void Transpose ( const NumMat<F>& A, NumMat<F>& B );
template <class F> inline void Symmetrize( NumMat<F>& A );

} // namespace PPCG

#include "ppcg/NumMat_impl.hpp"

#endif // _PPCG_NUMMAT_HPP_

