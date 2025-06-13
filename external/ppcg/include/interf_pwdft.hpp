/// @file blas.hpp
/// PPCG library interface for PWDFT
/// @date 2023-07-01
#ifndef _PPCG_INTERF_PWDFT_HPP_
#define _PPCG_INTERF_PWDFT_HPP_

#include  "ppcg/environment.hpp"
#include  "ppcg/NumVec.hpp"
#include  "ppcg/NumMat.hpp"
#include  "ppcg/domain.hpp"
#include  "ppcg/fourier.hpp"
#include  "ppcg/hamiltonian.hpp"
#include  "ppcg/spinor.hpp"
#include  "ppcg/eigensolver.hpp"
#include  "ppcg/blas.hpp"

//namespace PPCG {

  //extern std::ofstream statusOFS;

//}

void ppcg_solveReal_mpi(int Iter, int Ncom, int Nstate, int Nbeta,
                        int* nG, int* nGFine, 
                        double* unitcell, MPI_Comm comm,
                        double* vtot, double* vnl, int *index,
                        int* nGproj, double* coef,
                        double* Psi_in,
                        int eigMaxIter, double eigMinTolerance, double eigTolNow,
                        int sbsize,
                        double* EigVals,
                        double* Psi_out
                       );


#endif // _PPCG_INTERF_PWDFT_HPP_

