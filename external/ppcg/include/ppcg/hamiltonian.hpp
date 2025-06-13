/// @file hamiltonian.hpp
/// @brief Hamiltonian class for planewave basis diagonalization method.
/// @date 2023-07-01
#ifndef _PPCG_HAMILTONIAN_HPP_
#define _PPCG_HAMILTONIAN_HPP_

#include  "ppcg/environment.hpp"
#include  "ppcg/NumVec.hpp"
#include  "ppcg/domain.hpp"
#include  "ppcg/spinor.hpp"
#include  "ppcg/utility.hpp"

// *********************************************************************
// Base Hamiltonian class 
// *********************************************************************

/// @brief Pure virtual class for handling different types of
/// Hamiltonian.

namespace PPCG {

class Hamiltonian {
protected:
  Domain                      domain_;
  // Total local potential
  DblNumVec                   vtot_;  
  // Projectors in separable nonlocal potential
  DblNumVec                   vnlc_;
  // Index of non-zero values of projectors
  IntNumVec                   index_;
  // Number of non-zero values of projectors
  IntNumVec                   nGproj_;
  // weights of projectors
  DblNumVec                   coef_;       

public:

  // *********************************************************************
  // Lifecycle
  // *********************************************************************
  Hamiltonian();
  ~Hamiltonian();

  Hamiltonian(const Domain &domain, Int Nbeta, const double* vtot, const double* vnlc, const int* index,
                 const int* nGproj, const double* coef);
  // *********************************************************************
  // Operations
  // *********************************************************************
#ifdef _COMPLEX_
  void MultSpinor(Spinor& psi, NumTns<Complex>& a3, Fourier& fft);
#else
  void MultSpinor(Spinor& psi, NumTns<Real>& a3, Fourier& fft);
#endif
  // *********************************************************************
  // Access
  // *********************************************************************
  DblNumVec&  Vtot() { return vtot_; }
  DblNumVec&  Vnlc() { return vnlc_; }
  DblNumVec&  Coef() { return coef_; }                 
};

} // namespace PPCG

#endif // _PPCG_HAMILTONIAN_HPP_
