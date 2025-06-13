/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

Author: Lin Lin

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
/// @file fourier.cpp
/// @brief Sequential and Distributed Fourier wrapper.
/// @date 2011-11-01
/// @date 2015-05-02 Add some dual grid functions
/// @date 2023-11-23 Add sphere truncated grid functions
#ifndef _FOURIER_HPP_
#define _FOURIER_HPP_

#include  "environment.hpp"
#include  "domain.hpp"
#include  "numvec_impl.hpp"
#include  "esdf.hpp"

#ifdef GPU
#include "cu_numvec_impl.hpp"
#include <assert.h>
#endif

namespace pwdft{

// *********************************************************************
// Sequential FFTW interface
// *********************************************************************

/// @struct Fourier
/// @brief Sequential FFTW interface.
struct Fourier {
  Domain           domain;
  bool             isInitialized;
  Int              numGridTotal;
  Int              numGridTotalFine;
  // plans
  fftw_plan backwardPlan;
  fftw_plan forwardPlan;
  fftw_plan backwardPlanFine;
  fftw_plan forwardPlanFine;
  fftw_plan backwardPlanR2C;
  fftw_plan forwardPlanR2C;
  fftw_plan backwardPlanR2CFine;
  fftw_plan forwardPlanR2CFine;

#ifdef GPU
  cufftHandle cuPlanR2C[NSTREAM];
  cufftHandle cuPlanR2CFine[NSTREAM];
  cufftHandle cuPlanC2R[NSTREAM];
  cufftHandle cuPlanC2RFine[NSTREAM];
  cufftHandle cuPlanC2CFine[NSTREAM];
  cufftHandle cuPlanC2C[NSTREAM];
#endif

  unsigned  plannerFlag;

  // Laplacian operator related
  DblNumVec                gkk;
  std::vector<CpxNumVec>   ik;

  DblNumVec                gkkFine;
  std::vector<CpxNumVec>   ikFine;

  // Temporary vectors that can also be used globally
  CpxNumVec                inputComplexVec;     
  CpxNumVec                outputComplexVec;     

  CpxNumVec                inputComplexVecFine;     
  CpxNumVec                outputComplexVecFine;     

  // Real data Fourier transform
  Int                      numGridTotalR2C;
  Int                      numGridTotalR2CFine;

  DblNumVec                gkkR2C;
  std::vector<CpxNumVec>   ikR2C;
  std::vector<IntNumVec>   iKR2C;

  DblNumVec                gkkR2CFine;
  std::vector<CpxNumVec>   ikR2CFine;

  // Temporary vectors that can also be used globally
  DblNumVec                inputVecR2C;     
  CpxNumVec                outputVecR2C;     

  DblNumVec                inputVecR2CFine;     
  CpxNumVec                outputVecR2CFine;     

  /// @brief index array for mapping a coarse grid to a fine grid
  IntNumVec                idxFineGrid;
  IntNumVec                idxFineGridR2C;

  /// index array for mapping the grid of Fock operator
  /// to a coarse grid
  IntNumVec                idxCoarseFock;

  /// index array for mapping the grid of Fock operator
  /// to a fine grid
  IntNumVec                idxFineFock;
#ifdef _COMPLEX_
  /// @brief index array for mapping a sphere cutted
  /// wavefunction grid to a coarse grid
  ///
  /// Cutoff: |k + G|^2 < ecutWavefunction * 2.0
  std::vector<IntNumVec>   idxCoarseCut;
  std::vector<IntNumVec>   idxCoarseCutSCF;
  
  /// @brief index array for mapping a sphere cutted
  /// wavefunction grid to a fine grid 
  ///
  /// Cutoff: |k + G|^2 < ecutWavefunction * 2.0
  std::vector<IntNumVec>   idxFineCut;
  std::vector<IntNumVec>   idxFineCutSCF;

  /// @brief index array for mapping a sphere cutted
  /// density grid to a fine grid 
  ///
  /// Cutoff: |k + G|^2 < ecutWavefunction * 8.0
  IntNumVec                idxFineCutDensity;
#else
  IntNumVec                idxCoarseCut;
  IntNumVec                idxFineCut;
  IntNumVec                idxFineCutDensity;

  std::pair<IntNumVec, IntNumVec> idxCoarsePadding;
  std::pair<IntNumVec, IntNumVec> idxFinePadding;

  // Debug purpose
  IntNumVec                idxR2C;
#endif

  Fourier();
  ~Fourier();

  void Initialize( const Domain& dm );
  void InitializeFine( const Domain& dm );
  void InitializeSphere( Domain& dm );
};

void FFTWExecute( Fourier& fft, fftw_plan& plan );

#ifdef GPU
void cuFFTExecuteInverse( Fourier& fft, cufftHandle &plan, int fft_type, cuCpxNumVec &cu_psi_in, cuCpxNumVec &cu_psi_out );
void cuFFTExecuteInverse( Fourier& fft, cufftHandle &plan, int fft_type, cuCpxNumVec &cu_psi_in, cuCpxNumVec &cu_psi_out , int nbands);
void cuFFTExecuteInverse2( Fourier& fft, cufftHandle &plan, int fft_type, cuCpxNumVec &cu_psi_in, cuCpxNumVec &cu_psi_out );
void cuFFTExecuteForward( Fourier& fft, cufftHandle &plan, int fft_type, cuCpxNumVec &cu_psi_in, cuCpxNumVec &cu_psi_out );
void cuFFTExecuteForward2( Fourier& fft, cufftHandle &plan, int fft_type, cuCpxNumVec &cu_psi_in, cuCpxNumVec &cu_psi_out );
void cuFFTExecuteInverse( Fourier& fft, cufftHandle &plan, int fft_type, cuDblNumVec &cu_psi_in, cuDblNumVec &cu_psi_out );
void cuFFTExecuteForward( Fourier& fft, cufftHandle &plan, int fft_type, cuDblNumVec &cu_psi_in, cuDblNumVec &cu_psi_out );
#endif

} // namespace pwdft

#endif // _FOURIER_HPP_
