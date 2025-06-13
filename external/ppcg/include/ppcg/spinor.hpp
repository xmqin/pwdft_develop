/// @file spinor.hpp
/// @brief Spinor (wavefunction) for the global domain.
/// @date 2023-07-01
#ifndef _PPCG_SPINOR_HPP_
#define _PPCG_SPINOR_HPP_

#include  "ppcg/environment.hpp"
#include  "ppcg/NumVec.hpp"
#include  "ppcg/NumTns.hpp"
#include  "ppcg/domain.hpp"
#include  "ppcg/fourier.hpp"
#include  "ppcg/utility.hpp"
#include  "ppcg/lapack.hpp"

namespace PPCG {

class Spinor {
private:
  Domain            domain_;                // mesh should be used here for general cases 
#ifdef _COMPLEX_
  NumTns<Complex>   wavefun_;               // Local data of the wavefunction 
#else
  NumTns<Real>      wavefun_;               // Local data of the wavefunction 
#endif
  IntNumVec         wavefunIdx_;
  Int               numStateTotal_;
  Int               blocksize_;

  IntNumVec         numProcPotrf_;

public:
  // *********************************************************************
  // Constructor and destructor
  // *********************************************************************
  Spinor(); 
  ~Spinor();
#ifdef _COMPLEX_
  Spinor( const Domain &dm, const Int numComponent, const Int numStateTotal, Int numStateLocal,
      const bool owndata, Complex* data );

  void Setup( const Domain &dm, const Int numComponent, const Int numStateTotal, const Int numStateLocal,
      const Complex val = static_cast<Complex>(0,0) ); 

  void Setup( const Domain &dm, const Int numComponent, const Int numStateTotal, const Int numStateLocal,
      const bool owndata, Complex* data );
#else
  Spinor( const Domain &dm, const Int numComponent, const Int numStateTotal, Int numStateLocal,
      const bool owndata, Real* data );

  void Setup( const Domain &dm, const Int numComponent, const Int numStateTotal, const Int numStateLocal,
      const Real val = static_cast<Real>(0) ); 

  void Setup( const Domain &dm, const Int numComponent, const Int numStateTotal, const Int numStateLocal,
      const bool owndata, Real* data );
#endif
  // *********************************************************************
  // Inquiries
  // *********************************************************************
  Int NumGridTotal()  const { return wavefun_.m(); }
  Int NumComponent()  const { return wavefun_.n(); }
  Int NumState()      const { return wavefun_.p(); }
  Int NumStateTotal() const { return numStateTotal_; }
  Int Blocksize()     const { return blocksize_; }

  IntNumVec&  WavefunIdx() { return wavefunIdx_; }
  const IntNumVec&  WavefunIdx() const { return wavefunIdx_; }
  Int&  WavefunIdx(const Int k) { return wavefunIdx_(k); }
  const Int&  WavefunIdx(const Int k) const { return wavefunIdx_(k); }

#ifdef _COMPLEX_
  NumTns<Complex>& Wavefun() { return wavefun_; } 
  const NumTns<Complex>& Wavefun() const { return wavefun_; } 
  Complex& Wavefun(const Int i, const Int j, const Int k) {return wavefun_(i,j,k); }
  const Complex& Wavefun(const Int i, const Int j, const Int k) const {return wavefun_(i,j,k); }
#else
  NumTns<Real>& Wavefun() { return wavefun_; } 
  const NumTns<Real>& Wavefun() const { return wavefun_; } 
  Real& Wavefun(const Int i, const Int j, const Int k) {return wavefun_(i,j,k); }
  const Real& Wavefun(const Int i, const Int j, const Int k) const {return wavefun_(i,j,k); }
#endif

  // *********************************************************************
  // Access
  // *********************************************************************

  // *********************************************************************
  // Operations
  // *********************************************************************

  // Perform all operations of matrix vector multiplication on a fine grid.
#ifdef _COMPLEX_
  void AddMultSpinorFine( Fourier& fft, const DblNumVec& vtot, const DblNumVec& vnlc, const IntNumVec& index, 
      const IntNumVec& nGproj, const DblNumVec& coef, NumTns<Complex>& a3 );

  void AddTeterPrecond( Fourier* fftPtr, NumTns<Complex>& a3 );
#else
  void AddMultSpinorFine( Fourier& fft, const DblNumVec& vtot, const DblNumVec& vnlc, const IntNumVec& index, 
      const IntNumVec& nGproj, const DblNumVec& coef, NumTns<Real>& a3 );

  void AddMultSpinorFineR2C( Fourier& fft, const DblNumVec& vtot, const DblNumVec& vnlc, const IntNumVec& index,  
      const IntNumVec& nGproj, const DblNumVec& coef, NumTns<Real>& a3 );

  void AddTeterPrecond( Fourier* fftPtr, NumTns<Real>& a3 );
#endif
};  // Spinor

} // namespace PPCG

#endif // _PPCG_SPINOR_HPP_
