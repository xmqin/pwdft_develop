/// @file eigensolver.hpp
/// @brief Eigensolver in the global domain.
/// @date 2023-07-01
#ifndef _PPCG_EIGENSOLVER_HPP_
#define _PPCG_EIGENSOLVER_HPP_

#include  "ppcg/environment.hpp"
#include  "ppcg/NumVec.hpp"
#include  "ppcg/domain.hpp"
#include  "ppcg/fourier.hpp"
#include  "ppcg/hamiltonian.hpp"
#include  "ppcg/spinor.hpp"

namespace PPCG {

class EigenSolver
{
private:

  Hamiltonian*        hamPtr_;
  Fourier*            fftPtr_;
  Spinor*             psiPtr_;

  DblNumVec           eigVal_;
  DblNumVec           resVal_;


public:

  // ********************  LIFECYCLE   *******************************

  EigenSolver ();

  ~EigenSolver();

  // ********************  OPERATORS   *******************************

  void Setup(
      Hamiltonian& ham,
      Spinor& psi,
      Fourier& fft );

  // ********************  OPERATIONS  *******************************
  void PPCGSolveReal(
      Int          Iter,
      Int          numEig,
      Int          eigMaxIter,
      Real         eigMinTolerance,
      Real         eigTolerance,
      Int          sbsize );


  // ********************  ACCESS      *******************************
  DblNumVec& EigVal() { return eigVal_; }
  DblNumVec& ResVal() { return resVal_; }


  Hamiltonian& Ham()  {return *hamPtr_;}
  Spinor&      Psi()  {return *psiPtr_;}
  Fourier&     FFT()  {return *fftPtr_;}

}; // -----  end of class  EigenSolver  ----- 

} // namespace PPCG

#endif // _PPCG_EIGENSOLVER_HPP_
