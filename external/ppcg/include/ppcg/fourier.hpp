/// @file fourier.cpp
/// @brief Sequential Fourier wrapper.
/// @date 2023-07-01
#ifndef _PPCG_FOURIER_HPP_
#define _PPCG_FOURIER_HPP_

#include  "ppcg/environment.hpp"
#include  "ppcg/domain.hpp"
#include  "ppcg/NumVec.hpp"

// *********************************************************************
// Sequential FFTW interface
// *********************************************************************

/// @struct Fourier
/// @brief Sequential FFTW interface.
namespace PPCG {

struct Fourier {
  Domain           domain;
  bool             isInitialized;
  Int              numGridTotal;
  Int              numGridTotalFine;
  Int              FFTtype;
  // plans
  fftw_plan backwardPlan;
  fftw_plan forwardPlan;
  fftw_plan backwardPlanFine;
  fftw_plan forwardPlanFine;
  fftw_plan backwardPlanR2C;
  fftw_plan forwardPlanR2C;
  fftw_plan backwardPlanR2CFine;
  fftw_plan forwardPlanR2CFine;

  unsigned  plannerFlag;

  // Laplacian operator related
  DblNumVec                gkk;
  std::vector<CpxNumVec>   ik;
  DblNumVec                TeterPrecond;

  DblNumVec                gkkFine;
  std::vector<CpxNumVec>   ikFine;
  // FIXME Teter should be moved to Hamiltonian
  DblNumVec                TeterPrecondFine;

  // Temporary vectors that can also be used globally
  CpxNumVec                inputComplexVec;     
  CpxNumVec                outputComplexVec;     

  CpxNumVec                inputComplexVecFine;     
  CpxNumVec                outputComplexVecFine;     

  // Real data Fourier transform
  Int       numGridTotalR2C;
  Int       numGridTotalR2CFine;

  DblNumVec                gkkR2C;
  std::vector<CpxNumVec>   ikR2C;
  DblNumVec                TeterPrecondR2C;

  DblNumVec                gkkR2CFine;
  std::vector<CpxNumVec>   ikR2CFine;
  DblNumVec                TeterPrecondR2CFine;

  // Temporary vectors that can also be used globally
  DblNumVec                inputVecR2C;     
  CpxNumVec                outputVecR2C;     

  DblNumVec                inputVecR2CFine;     
  CpxNumVec                outputVecR2CFine;     

  /// @brief index array for mapping a coarse grid to a fine grid
  IntNumVec                idxFineGrid;
  IntNumVec                idxFineGridR2C;


  Fourier();
  ~Fourier();

  void Initialize( const Domain& dm );
  void InitializeFine( const Domain& dm );
};

void FFTWExecute( Fourier& fft, fftw_plan& plan );

} // namespace PPCG

#endif // _PPCG_FOURIER_HPP_
