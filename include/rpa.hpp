#ifndef _RPA_HPP_
#define _RPA_HPP_

#include "environment.hpp"
#include "numvec_impl.hpp"
#include "numtns_impl.hpp"
#include "domain.hpp"
#include "fourier.hpp"
#include "spinor.hpp"
#include "hamiltonian.hpp"
#include "utility.hpp"
#include "lapack.hpp"
#include "esdf.hpp"
#include <cmath>

namespace pwdft{

class RPA {
private:
  Fourier*            fftPtr_; 
  Spinor*             psiPtr_;

  Int nocc_;  // Number of valence bands for constructing operators
  Int nvir_;  // Number of conduction bands for constructing operators
  Int nspin_;
  DblNumVec eigVal_; 
  DblNumVec coulG_;  

  bool restart_rpa_;             
  std::string freq_int_method_; // int method for frequency int, guass-legendre / clenshaw-curtis
  Int nw_;
  std::string coulomb_trunc_;
  Real radius_;
  bool spherecut_recip_;

public:
  // *********************************************************************
  // Lifecycle
  // *********************************************************************
  RPA();
  ~RPA();

  void Setup(Hamiltonian &ham, Spinor &psi, Fourier &fft);
  
  // *********************************************************************
  // Operations
  // *********************************************************************

  Real CalculateEnergyRPA();

  void Coeffs_gausslegint( DblNumVec &omega, DblNumVec &weight, double xmin, double xmax, int n );

  void Coeffs_clenshawint( DblNumVec &omega, DblNumVec &weight, double length, int n );

  void KR_product( DblNumMat &psi, Int m, Int n1, Int n2, DblNumMat &psiphi ); 

  void FFTR2C( DblNumMat &psiphi, CpxNumMat &PsiPhi );

  void CalculateCoulomb( DblNumVec &Vcoul );

  Int CalculateDeterminant( CpxNumMat &matrix ); 

}; // -----  end of class RPA  ----- 

} // namespace pwdft
#endif // _RPA_HPP_
