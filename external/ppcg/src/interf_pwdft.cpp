/// @file interf_pwdft.cpp
/// @brief The interface for pwdft to call PPCG library
/// @date 2023/6/30
#include  "interf_pwdft.hpp"
#include  "ppcg/environment.hpp"
#include  "ppcg/NumVec.hpp"
#include  "ppcg/NumMat.hpp"
#include  "ppcg/domain.hpp"
#include  "ppcg/fourier.hpp"
#include  "ppcg/hamiltonian.hpp"
#include  "ppcg/spinor.hpp"
#include  "ppcg/eigensolver.hpp"
#include  "ppcg/blas.hpp"

using namespace PPCG;

void ppcg_solveReal_mpi(int Iter, int Ncom, int Nstate, int Nbeta, 
                        int* nG, int* nGFine,
                        double *unitcell, MPI_Comm comm,
                        double* vtot, double* vnlc, int *index,
                        int* nGproj, double* coef,
                        double* Psi_in,
                        int eigMaxIter, double eigMinTolerance, double eigTolNow,
                        Int sbsize,
                        double* EigVals,
                        double* Psi_out
                       )
{
    std::stringstream  ss;
    ss << "ppcg." << Iter;
    statusOFS.open( ss.str().c_str() );

// 1. Construct computation domain
    Domain dm;  
    // assign grid information and communication domain
    dm.comm = comm; dm.rowComm = comm; dm.colComm = comm; 
    //blas::Copy(DIM, nG, 1, dm.numGrid.Data(), 1);
    //blas::Copy(DIM, nGFine, 1, dm.numGridFine.Data(), 1);
    for( Int i = 0; i < DIM; i++ ){
      dm.numGrid[i] = nG[i];
      dm.numGridFine[i] = nGFine[i];
    }

    DblNumMat& M = dm.supercell;
    blas::Copy(DIM*DIM, unitcell, 1, M.Data(), 1);
    // Calculate the reciprocal lattice vectors by recipcell = 2pi*(inv(supercell))'
    // each row of recipcell represents reciprocal lattice vector in one direction
    DblNumMat adjointM;
    adjointM.Resize( DIM, DIM ); SetValue( adjointM, D_ZERO );
    for( Int i = 0; i < DIM; i++ ){
        for( Int j = 0; j < DIM; j++ ){
        adjointM(j,i) = M( (i+1)%DIM, (j+1)%DIM ) * M( (i+2)%DIM, (j+2)%DIM )
                            - M( (i+1)%DIM, (j+2)%DIM ) * M( (i+2)%DIM, (j+1)%DIM );
        }
    }
    // factor = det(M)
    Real factor = D_ZERO;
    for( Int i = 0; i < DIM; i++ ){
        factor += M(0,i) * adjointM(i,0);
    }
    factor = 2 * PI / factor;

    for( Int i = 0; i < DIM; i++ ){
        for( Int j = 0; j < DIM; j++ ){
        dm.recipcell(i,j) = adjointM(j,i) * factor;
        }
    }

//  2. Build Hamiltonian by local and nonlocal potential
    Hamiltonian ham = Hamiltonian(dm, Nbeta, vtot, vnlc, index, nGproj, coef);

//  3. Initialize spinor 
    Int mpirank, mpisize;
    MPI_Comm_rank( MPI_COMM_WORLD, &mpirank );
    MPI_Comm_size( MPI_COMM_WORLD, &mpisize );

    Int NstateLocal, blocksize;
    if ( Nstate <=  mpisize ) {
      blocksize = 1;

      if ( mpirank < Nstate ){
        NstateLocal = 1; // blocksize == 1;
      }
      else {
        NstateLocal = 0;
      }
    }
    else {  // numStateTotal >  mpisize

      if ( Nstate % mpisize == 0 ){
        blocksize = Nstate / mpisize;
        NstateLocal = blocksize ;
      }
      else {
        blocksize = Nstate / mpisize;
        NstateLocal = blocksize ;
        if ( mpirank < ( Nstate% mpisize ) ) {
          NstateLocal = NstateLocal + 1 ;
        }
      }
    }
 
    Spinor psi = Spinor(dm, Ncom, Nstate, NstateLocal, true, Psi_in);

// 4. Initialize FFT
    Fourier fft;

    fft.Initialize( dm );

    fft.InitializeFine( dm );

// 5. Assemble eigensolver by ham, spinor and fft
    EigenSolver eigSol;
    eigSol.Setup( ham, psi, fft );

// 6. Call PPCGReal solver to obtain first Nstate small eigenvalues and eigenvectors
    eigSol.PPCGSolveReal(Iter, Nstate, eigMaxIter, eigMinTolerance, eigTolNow, sbsize );
    Int Ntot = dm.NumGridTotal();

    blas::Copy(Ntot*NstateLocal*Ncom, eigSol.Psi().Wavefun().Data(), 1, Psi_out, 1);
    blas::Copy(Nstate, eigSol.EigVal().Data(), 1, EigVals, 1);

    statusOFS.close( ss.str().c_str() );
}


