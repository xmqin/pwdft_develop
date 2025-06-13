/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

Author: Lin Lin and Wei Hu

This file is part of DGDFT. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

(1) Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.
(2) Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.
(3) Neither the name of the University of California, Lawrence Berkeley
National Laboratory, U.S. Dept. of Energy nor the names of its contributors may
be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

You are under no obligation whatsoever to provide any bug fixes, patches, or
upgrades to the features, functionality or performance of the source code
("Enhancements") to anyone; however, if you choose to make your Enhancements
available either publicly, or directly to Lawrence Berkeley National
Laboratory, without imposing a separate written license agreement for such
Enhancements, then you hereby grant the following license: a non-exclusive,
royalty-free perpetual license to install, use, modify, prepare derivative
works, incorporate into other computer software, distribute, and sublicense
such enhancements or derivative works thereof, in binary and source code form.
 */
/// @file spinor.hpp
/// @brief Spinor (wavefunction) for the global domain or extended
/// element.
/// @date 2012-10-06
#ifndef _SPINOR_HPP_
#define _SPINOR_HPP_

#include  "environment.hpp"
#include  "numvec_impl.hpp"
#include  "numtns_impl.hpp"
#include  "domain.hpp"
#include  "fourier.hpp"
#include  "utility.hpp"
#include  "lapack.hpp"
#include  "esdf.hpp"

#ifdef GPU
#include  "cu_numvec_impl.hpp"
#include  "cu_numtns_impl.hpp"
#include  "cublas.hpp"
#endif

namespace pwdft{

using namespace pwdft::esdf;

class Spinor {
private:
  Domain            domain_;                // mesh should be used here for general cases 
#ifdef _COMPLEX_
  NumTns<Complex>   wavefun_;               // Local data of the wavefunction 
  NumTns<Complex>   wavefunR_;              // wavefunction in real space, stored only for hybrid functional
  Point3            kpoint_;                // K-point coordinate of the Bloch wavefunction
  Int               ik_;                    // Global k index of the Bloch wavefunction
  Int               ikLocal_;               // Local k index of the Bloch wavefunction
  //CpxNumMat         G_;
#else
  NumTns<Real>      wavefun_;               // Local data of the wavefunction 
  NumTns<Complex>   wavefunG_;
  //DblNumMat         G_;
#endif
  // For partition and column-row transformation of wavefunction
  IntNumVec         wavefunIdx_;
  IntNumVec         gridIdx_;
  Int               numStateTotal_;
  Int               numGridTotal_;
  Int               numComponent_;
  Int               numStateLocal_;
  Int               numGridLocal_;

  Int               mblocksize_;
  Int               nblocksize_;
  IntNumVec         sendcounts;
  IntNumVec         recvcounts;
  IntNumVec         senddispls;
  IntNumVec         recvdispls;
  IntNumMat         sendk;
  IntNumMat         recvk;
  // For density fitting
  Int               numMu_;
  IntNumVec         pivQR_;
  IntNumVec         numProcPotrf_;

#ifdef GPU
  // not use wavefun_ in the GPU implementation.
  cuNumTns<Real>   cu_wavefun_;
#endif

public:
  // *********************************************************************
  // Constructor and destructor
  // *********************************************************************
  Spinor(); 
  ~Spinor();
#ifdef _COMPLEX_
  // Added for setup of Bloch wavefunctions
  Spinor( const Domain &dm, const Int numComponent, const Int numStateTotal,
      const Complex val, const Point3 kpoint = Point3(0.0, 0.0, 0.0 ), 
      const Int ik = 0, const Int ikLocal = 0 );

  Spinor( const Domain &dm, const Int numComponent, const Int numStateTotal,
      const bool owndata, Complex* data, const Point3 kpoint = Point3(0.0, 0.0, 0.0 ), 
      const Int ik = 0, const Int ikLocal = 0 );

  void Setup( const Domain &dm, const Int numComponent, const Int numStateTotal,
      const Complex val, const Point3 kpoint = Point3(0.0, 0.0, 0.0 ), 
      const Int ik = 0, const Int ikLocal = 0 );

  void Setup( const Domain &dm, const Int numComponent, const Int numStateTotal,
      const bool owndata, Complex* data, const Point3 kpoint = Point3(0.0, 0.0, 0.0 ), 
      const Int ik = 0, const Int ikLocal = 0 );
#else
  Spinor( const Domain &dm, const Int numComponent, const Int numStateTotal,
      const Real val = static_cast<Real>(0) );

  Spinor( const Domain &dm, const Int numComponent, const Int numStateTotal,
      const bool owndata, Real* data );

  void Setup( const Domain &dm, const Int numComponent, const Int numStateTotal,
      const Real val = static_cast<Real>(0) ); 

  void Setup( const Domain &dm, const Int numComponent, const Int numStateTotal,
      const bool owndata, Real* data );

  Spinor( const Domain &dm, const Int numComponent, const Int numStateTotal,
      const Complex val = static_cast<Complex>(0) );

  Spinor( const Domain &dm, const Int numComponent, const Int numStateTotal,
      const bool owndata, Complex* data );

  void Setup( const Domain &dm, const Int numComponent, const Int numStateTotal,
      const Complex val = static_cast<Complex>(0) );

  void Setup( const Domain &dm, const Int numComponent, const Int numStateTotal,
      const bool owndata, Complex* data );
#endif

#ifdef GPU
  Spinor( const Domain &dm, const Int numComponent, const Int numStateTotal,
      const bool owndata, Real* data, bool isGPU );
  void SetupGPU( const Domain &dm, const Int numComponent, const Int numStateTotal,
      const bool owndata, Real* data );
#endif

  // *********************************************************************
  // Inquiries
  // *********************************************************************
  Int NumGrid()       const { return numGridLocal_; }
  Int NumGridTotal()  const { return numGridTotal_; }
  Int NumComponent()  const { return numComponent_; }
#ifdef _COMPLEX_
  Int NumState()      const { return wavefun_.p(); }
#else
  Int NumState()      const { if( esdfParam.isUseSphereCut ) return wavefunG_.p();
                              else return wavefun_.p(); }
#endif
  Int NumStateTotal() const { return numStateTotal_; }
  Int MBlocksize()     const { return mblocksize_; }
  Int NBlocksize()     const { return nblocksize_; }

  IntNumVec&  WavefunIdx() { return wavefunIdx_; }
  IntNumVec&  GridIdx() { return gridIdx_; }
  const IntNumVec&  WavefunIdx() const { return wavefunIdx_; }
  const IntNumVec&  GridIdx() const { return gridIdx_; }
  Int&  WavefunIdx(const Int k) { return wavefunIdx_(k); }
  Int&  GridIdx(const Int k) { return gridIdx_(k); }
  const Int&  WavefunIdx(const Int k) const { return wavefunIdx_(k); }
  const Int&  GridIdx(const Int k) const { return gridIdx_(k); }

  const IntNumVec&  SendCounts() const { return sendcounts; }
  const IntNumVec&  RecvCounts() const { return recvcounts; }
  const IntNumVec&  SendDispls() const { return senddispls; }
  const IntNumVec&  RecvDispls() const { return recvdispls; }
  const IntNumMat&  Sendk()      const { return sendk; }
  const IntNumMat&  Recvk()      const { return recvk; }

  IntNumVec&  PivQR() { return pivQR_; }            
#ifdef _COMPLEX_
  Point3 Kpoint()     const { return kpoint_; }
  Int Ik()            const { return ik_; }
  Int IkLocal()       const { return ikLocal_; }

  NumTns<Complex>& Wavefun() { return wavefun_; } 
  const NumTns<Complex>& Wavefun() const { return wavefun_; } 
  Complex& Wavefun(const Int i, const Int j, const Int k) {return wavefun_(i,j,k); }
  const Complex& Wavefun(const Int i, const Int j, const Int k) const {return wavefun_(i,j,k); }

  NumTns<Complex>& WavefunR() { return wavefunR_; }
  const NumTns<Complex>& WavefunR() const { return wavefunR_; }
  Complex& WavefunR(const Int i, const Int j, const Int k) {return wavefunR_(i,j,k); }
  const Complex& WavefunR(const Int i, const Int j, const Int k) const {return wavefunR_(i,j,k); }
#else
  NumTns<Real>& Wavefun() { return wavefun_; } 
  const NumTns<Real>& Wavefun() const { return wavefun_; } 
  Real& Wavefun(const Int i, const Int j, const Int k) {return wavefun_(i,j,k); }
  const Real& Wavefun(const Int i, const Int j, const Int k) const {return wavefun_(i,j,k); }

  NumTns<Complex>& WavefunG() { return wavefunG_; }
  const NumTns<Complex>& WavefunG() const { return wavefunG_; }
  Complex& WavefunG(const Int i, const Int j, const Int k) {return wavefunG_(i,j,k); }
  const Complex& WavefunG(const Int i, const Int j, const Int k) const {return wavefunG_(i,j,k); }
#endif

#ifdef GPU
  cuNumTns<Real>& cuWavefun() { return cu_wavefun_; }
  const cuNumTns<Real>& cuWavefun() const { return cu_wavefun_; }
#endif

  // *********************************************************************
  // Access
  // *********************************************************************

  // *********************************************************************
  // Operations
  // *********************************************************************
  void Normalize();

  // Perform all operations of matrix vector multiplication on a fine grid.
#ifdef _COMPLEX_
  void AddTeterPrecond( Fourier* fftPtr, DblNumVec& teter, NumTns<Complex>& a3 );

  void AddMultSpinorFine( Fourier& fft, const std::vector<DblNumVec>& ekin, 
      const DblNumVec& vtot, const std::vector<PseudoPot>& pseudo, NumTns<Complex>& a3 );
  
  void AddMultSpinorFine( Fourier& fft, const std::vector<DblNumVec>& ekin,
      const DblNumMat& vtot, const std::vector<PseudoPot>& pseudo, NumTns<Complex>& a3 );

  // @brief Apply the exchange operator to the spinor by solving
  // Poisson like equations
  // EXX: Spinor with exact exchange.
  void Spinor::AddMultSpinorEXX ( Fourier& fft,
      std::vector<Spinor>& psik,
      const std::vector<CpxNumTns>& phik,
      const DblNumTns& exxgkk,
      Real  exxFraction,
      Int   spinIndex,
      Int   nspin,
      const std::vector<DblNumVec>& occupationRatek,
      std::vector<CpxNumTns>& a3 );

  void AddMultSpinorEXXDF ( Fourier& fft,
    const std::vector<Spinor>& psik,
    const std::vector<CpxNumTns>& phik,
    const DblNumTns& exxgkkR2C,
    Real  exxFraction,
    Int   spinIndex,
    Int   nspin,
    const std::vector<DblNumVec>& occupationRatek, 
    std::string& hybridDFType,
    std::string hybridDFKmeansWFType,
    const Real hybridDFKmeansWFAlpha,
    Real  hybridDFKmeansTolerance,
    Int   hybridDFKmeansMaxIter,
    const Real numMuFac,
    const Real numGaussianRandomFac,
    const Int numProcScaLAPACK,
    const Real hybridDFTolerance,
    const Int BlockSizeScaLAPACK,
    bool isFixColumnDF,
    std::vector<CpxNumMat>& VxMat,
    std::vector<CpxNumTns>& a3 );

    void AddMultSpinorEXXDFConv ( Fourier& fft,
    const std::vector<Spinor>& psik,
    const std::vector<CpxNumTns>& phik,
    const DblNumTns& exxgkkR2C,
    Real  exxFraction,
    Int   spinIndex,
    Int   nspin,
    const std::vector<DblNumVec>& occupationRatek,
    std::string& hybridDFType,
    std::string hybridDFKmeansWFType,
    const Real hybridDFKmeansWFAlpha,
    Real  hybridDFKmeansTolerance,
    Int   hybridDFKmeansMaxIter,
    const Real numMuFac,
    const Real numGaussianRandomFac,
    const Int numProcScaLAPACK,
    const Real hybridDFTolerance,
    const Int BlockSizeScaLAPACK,
    bool isFixColumnDF,
    std::vector<CpxNumMat>& VxMat,
    std::vector<CpxNumTns>& a3 );

    void Recip2Real( Fourier& fft );

    // ISDF functions
    void Spinor::ISDF_SelectIP( CpxNumMat& psiCol, CpxNumMat& phiCol,
        std::string& hybridDFType,
        std::string hybridDFKmeansWFType,
        const Real hybridDFKmeansWFAlpha,
        Real  hybridDFKmeansTolerance,
        Int   hybridDFKmeansMaxIter,
        const Real hybridDFTolerance,
        Int   numMu, const Real numGaussianRandomFac,
        Int mb, Int nb );

    void Spinor::ISDF_CalculateIV( Fourier& fft,
        CpxNumMat& psiCol, CpxNumMat& phiCol,
        DblNumVec& occupationRate, const DblNumTns& exxgkk,
        Real exxFraction, Int numMu, Int mb, Int nb,
        std::vector<IntNumVec>& idxMu, IntNumVec& ndispls,
        CpxNumTns& VXiRow, CpxNumTns& KMuNu );
#else
  void AddTeterPrecond( Fourier* fftPtr, DblNumVec& teter, NumTns<Real>& a3 );

  void AddTeterPrecond( Fourier* fftPtr, DblNumVec& teter, NumTns<Complex>& a3 );

  void AddMultSpinorFineR2C( Fourier& fft, const DblNumVec& ekin,
      const DblNumVec& vtot, const std::vector<PseudoPot>& pseudo, NumTns<Real>& a3 );

  void AddMultSpinorFineR2C( Fourier& fft, const DblNumVec& ekin,
      const DblNumVec& vtot, const std::vector<PseudoPot>& pseudo, NumTns<Complex>& a3 );

  void AddMultSpinorEXX ( Fourier& fft,
      const NumTns<Real>& phi,
      const DblNumVec& exxgkkR2C,
      Real  exxFraction,
      Int   nspin,
      const DblNumVec& occupationRate,
      NumTns<Real>& a3 );

  void AddMultSpinorEXX ( Fourier& fft,
      const NumTns<Complex>& phi,
      const DblNumVec& exxgkkR2C,
      Real  exxFraction,
      Int   nspin,
      const DblNumVec& occupationRate,
      NumTns<Complex>& a3 );
#endif

#ifdef GPU
  void AddTeterPrecondGPU( Fourier* fftPtr, DblNumVec& teter, cuNumTns<Real>& a3 );

  void AddTeterPrecondGPU( Fourier* fftPtr, DblNumVec& teter, NumTns<Real>& a3 );

  void AddMultSpinorFineR2CGPU( Fourier& fft, DblNumVec& ekin, const DblNumVec& vtot,
      const std::vector<PseudoPot>& pseudo, cuNumTns<Real>& a3 );

  void AddMultSpinorEXXGPU ( Fourier& fft,
      const NumTns<Real>& phi,
      const DblNumVec& exxgkkR2CFine,
      Real  exxFraction,
      Real  numSpin,
      const DblNumVec& occupationRate,
      cuNumTns<Real>& a3 );

#endif

};  // Spinor

} // namespace pwdft

#endif // _SPINOR_HPP_
