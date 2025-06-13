/// @file hamiltonian.cpp
/// @brief Hamiltonian class for planewave basis diagonalization method.
/// @brief Eigensolver in the global domain.
#include  "ppcg/hamiltonian.hpp"
#include  "ppcg/blas.hpp"
#include  "ppcg/lapack.hpp"

namespace PPCG {

Hamiltonian::Hamiltonian() { }
Hamiltonian::~Hamiltonian() { }

// Construct Hamiltonian by inputed potentials
Hamiltonian::Hamiltonian(const Domain &domain, Int nbeta, const double* vtot, const double* vnlc, 
    const int* index, const int* nGproj, const double* coef)
{
  domain_ = domain;
  Int ntot = domain_.NumGridTotalFine();
  // Local potential
  vtot_.Resize( ntot );
  blas::Copy(ntot, vtot, 1, vtot_.Data(), 1);
  // Nonlocal potential
  nGproj_.Resize( nbeta );
  coef_.Resize( nbeta );
  //blas::Copy(nbeta, nGproj, 1, nGproj_.Data(), 1);
  for( Int i = 0; i < nbeta; i++ ){
    nGproj_[i] = nGproj[i];
  }
  blas::Copy(nbeta, coef, 1, coef_.Data(), 1);

  Int nGtotal = 0;
  for( Int i = 0; i < nbeta; i++ ) {
    nGtotal += nGproj_[i];
  }
  vnlc_.Resize( nGtotal ); index_.Resize( nGtotal );
  blas::Copy(nGtotal, vnlc, 1, vnlc_.Data(), 1);
  //blas::Copy(nGtotal, index, 1, index_.Data(), 1);
  for( Int i = 0; i < nGtotal; i++ ){
    index_[i] = index[i];
  }
}
// Real version
void
Hamiltonian::MultSpinor    ( Spinor& psi, NumTns<Real>& a3, Fourier& fft )
{
  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

  Int ntot      = fft.domain.NumGridTotal();
  Int ntotFine  = fft.domain.NumGridTotalFine();
  Int numStateTotal = psi.NumStateTotal();
  Int numStateLocal = psi.NumState();
  NumTns<Real>& wavefun = psi.Wavefun();
  Int ncom = wavefun.n();

  Int ntotR2C = fft.numGridTotalR2C;

  Real timeSta, timeEnd;
  Real timeSta1, timeEnd1;

  Real timeGemm = 0.0;
  Real timeAlltoallv = 0.0;
  Real timeAllreduce = 0.0;

  SetValue( a3, 0.0 );

  GetTime( timeSta );
  psi.AddMultSpinorFineR2C( fft, vtot_, vnlc_, index_, nGproj_, coef_, a3 );
  GetTime( timeEnd );
#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Time for psi.AddMultSpinorFineR2C is " <<
    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

  return ;
}         // -----  end of method KohnSham::MultSpinor  ----- 

} // namespace PPCG


