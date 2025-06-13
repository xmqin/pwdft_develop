#ifndef _GW_HPP_
#define _GW_HPP_

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

// *********************************************************************
// Base GW class
// *********************************************************************

class GW {

private:
  Int nv_oper_;  // Number of valence bands for constructing operators
  Int nc_oper_;  // Number of conduction bands for constructing operators
  Int n_oper_;   // Number of energy bands for constructing operators
  Int nv_ener_;  // Calculate the number of valence bands of energy
  Int nc_ener_;  // Calculate the number of conduction bands of energy
  Int n_ener_;   //  Calculate the number of energy bands
  Int ntot_;     // real space Corase grid
  Int ntotFine_; // real space rho grid
  Int nv_;
  Int vcrank_ratio_;
  Int vsrank_ratio_;
  Int ssrank_ratio_;
  Int nspin_;         // spinor
  DblNumMat density_; // rho
  DblNumMat vxc_;     // exchange-correlation potential
  DblNumVec eigVal_;  // energy
  DblNumVec eVxc_gw_; // exchange-correlation energy
  Real vol_;          // Vol
  Domain domain_;     // domain

public:
  // *********************************************************************
  // Constructor and destructor
  // *********************************************************************
  void Setup(Hamiltonian &ham, Spinor &psi, Fourier &fft, const Domain &dm);

  void FFTR2C(Fourier &fft, DblNumMat &psiphi, CpxNumMat &PsiPhi);
  void FFTC2C(Fourier &fft, DblNumMat &psiphi, CpxNumMat &PsiPhi);

  void KR_product(DblNumMat &psiRow, Int n1, Int n2, Int nbands, Int ntotLocal, std::string KR_type,DblNumMat &psiphiRow);
  void CalculateCoulomb(Fourier &fft, DblNumVec &Vcoul, Real &Vcoul0);

  IntNumVec sortArray(int start, int end);
  IntNumVec getColsMapForward(int width);
  void ExchangeCols(IntNumVec  colsMapForward, DblNumVec & input);
  void ExchangeCols(IntNumVec  colsMapForward, DblNumMat & input);
  void computeIdxNum(int start,int end,int &idx,int &num_local_indexes);
  void printCpx(CpxNumMat &Mat);

  void inverse(CpxNumMat &epsilonblock);
  void GW_ISDF(DblNumMat &psiCol,DblNumMat &psiRow, const Int &n1, const Int &n2, const Int &n3, Int &ntotLocal, Int &rk, DblNumMat &psiphizetaRow, IntNumVec &pivQR_);
  std::vector<std::complex<double>> landen(std::complex<double> k, double tol = 2.22e-16);
  void calculateK(double &k, std::complex<double> &K, double tol = 2.22e-16);
  std::complex<double> calculateKPrime(double k, double tol = 2.22e-16);
  double polyval(const std::vector<double> &coeffs, double x);
  void calculateEllipk(double L, double &K, double &Kp);
  void ellipjc(std::complex<double> u, double L, std::complex<double> &sn, std::complex<double> &cn, std::complex<double> &dn, bool flag = false);

  double calculateFrobeniusNorm(const DblNumMat &fromatrix);
  std::vector<double> extractNumbers(const std::vector<double>& vector,double newGap);
  void integrand(std::complex<double> t, double k, double m, double M, std::complex<double> &lambda, std::complex<double> &dlambda);
  void COmegaCstar(DblNumMat &psiCol, CpxNumMat &phiMu, CpxNumMat &psiMu, DblNumMat &results);

  void CalculateEpsilon(Hamiltonian &ham, Spinor &psi, Fourier &fft, const Domain &dm);
  void CalculateEpsilon_ISDF(Hamiltonian &ham, Spinor &psi, Fourier &fft, const Domain &dm);
};

} // namespace pwdft

#endif // _GW_HPP_
