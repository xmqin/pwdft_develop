/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

Author: Lin Lin, Wei Hu, Amartya Banerjee, Weile Jia

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
/// @file lobpcg.cpp
/// @brief LOBPCG eigensolver for both real-value Hamiltonian
/// @date 2024-06-19 extract codes for LOBPCG from eigensolver.cpp
#include  "eigensolver.hpp"
#include  "utility.hpp"
#include  "blas.hpp"
#include  "lapack.hpp"
#include  "scalapack.hpp"
#include  "mpi_interf.hpp"

using namespace pwdft::scalapack;
using namespace pwdft::esdf;

namespace pwdft{
#ifndef _COMPLEX_
// Basic version of PPCG with columnwise sweep
void
EigenSolver::LOBPCGSolveReal    (
    Int          numEig,
    Int          scfIter,
    Int          eigMaxIter,
    Real         eigMinTolerance,
    Real         eigTolerance)
{
  // *********************************************************************
  // Initialization
  // *********************************************************************
  MPI_Barrier( mpi_comm_ );

  Int mpirank;  MPI_Comm_rank(mpi_comm_, &mpirank);
  Int mpisize;  MPI_Comm_size(mpi_comm_, &mpisize);

  bool realspace = esdfParam.isUseRealSpace;

  Int mb = esdfParam.BlockSizeGrid;
  Int nb = esdfParam.BlockSizeState;

  // The spinor used for hamiltonian diagonalization
  Spinor& psiTemp = *psiPtr_;

  Int ntot = psiTemp.NumGridTotal();
  Int ntotLocal = psiTemp.NumGrid();
  Int ncom = psiTemp.NumComponent();
  Int noccLocal = psiTemp.NumState();
  Int noccTotal = psiTemp.NumStateTotal();

  // The spin-up and spin-down Hamiltonian are diagonalized  
  // individually for spin-unrestricted case, so ncom = 1
  Int nspin = hamPtr_->NumDensityComponent();
  Int spinswitch = hamPtr_->SpinSwitch();
  if( nspin == 2 ){
    noccLocal /= 2;
    noccTotal /= 2;
  }
  // For spin-noncollinear case, the spin-up and spin-down components
  // are chunked in the same way and communicated in pair
  Int height = ntot;
  Int height2 = height * ncom;
  Int width = noccTotal;
  Int lda = 3 * width;

  Int widthLocal = noccLocal;
  Int heightLocal = ntotLocal;
  Int heightLocal2 = heightLocal * ncom;

  // The number of unconverged bands
  Int notconv = numEig;

  if( numEig > width ){
    std::ostringstream msg;
    msg 
      << "Number of eigenvalues requested  = " << numEig << std::endl
      << "which is larger than the number of columns in psi = " << width << std::endl;
    ErrorHandling( msg.str().c_str() );
  }

  if( !realspace ){
    // S = ( X | W | P ) is a triplet used for LOBPCG.  
    // W is the preconditioned residual
    CpxNumMat  S( heightLocal2, 3*width ), AS( heightLocal2, 3*width ); 
    // AMat = S' * (AS),  BMat = S' * S
    // 
    // AMat = (X'*AX   X'*AW   X'*AP)
    //      = (  *     W'*AW   W'*AP)
    //      = (  *       *     P'*AP)
    //
    // BMat = (X'*X   X'*W   X'*P)
    //      = (  *    W'*W   W'*P)
    //      = (  *      *    P'*P)
    DblNumMat  AMat( 3*width, 3*width ), BMat( 3*width, 3*width );
    DblNumMat  AMatT1( 3*width, 3*width );

    // Temporary buffer array.
    // The unpreconditioned residual will also be saved in Xtemp
    DblNumMat  XTX( width, width );
    DblNumMat  XTXtemp( width, width );
    DblNumMat  XTXtemp1( width, width );

    CpxNumMat  Xtemp( heightLocal2, width );

    // rexNorm Grobal matrix  similar to numEig 
    DblNumVec  resNormLocal ( width ); 
    SetValue( resNormLocal, 0.0 );
    DblNumVec  resNorm( width );
    SetValue( resNorm, 0.0 );
    Real       resMax, resMin;

    // For convenience
    CpxNumMat  X( heightLocal2, width, false, S.VecData(0) );
    CpxNumMat  W( heightLocal2, width, false, S.VecData(width) );
    CpxNumMat  P( heightLocal2, width, false, S.VecData(2*width) );
    CpxNumMat AX( heightLocal2, width, false, AS.VecData(0) );
    CpxNumMat AW( heightLocal2, width, false, AS.VecData(width) );
    CpxNumMat AP( heightLocal2, width, false, AS.VecData(2*width) );

    CpxNumMat  Xcol( height2, widthLocal );
    CpxNumMat  Wcol( height2, widthLocal );
    CpxNumMat AXcol( height2, widthLocal );
    CpxNumMat AWcol( height2, widthLocal );

    bool isRestart = false;
    // numSet = 2    : Steepest descent (Davidson), only use (X | W)
    //        = 3    : Conjugate gradient, use all the triplet (X | W | P)
    Int numSet = 2;

    // numLocked is the number of converged vectors
    Int numLockedLocal = 0, numLockedSaveLocal = 0;
    Int numLockedTotal = 0, numLockedSaveTotal = 0; 
    Int numLockedSave = 0;
    Int numActiveLocal = 0;
    Int numActiveTotal = 0;

    const Int numLocked = 0;  // Never perform locking in this version
    const Int numActive = width;

    bool isConverged = false;

    // Initialization
    SetValue( S, Complex( 0.0, 0.0) );
    SetValue( AS, Complex( 0.0, 0.0) );

    DblNumVec  eigValS(lda);
    SetValue( eigValS, 0.0 );

    // Initialize X by the data in psi
    lapack::Lacpy( 'A', height2, widthLocal, psiTemp.WavefunG().MatData(spinswitch*noccLocal), height2, 
      Xcol.Data(), height2 );

    AlltoallForward( mb, nb, ncom, Xcol, X, mpi_comm_ );

    // *********************************************************************
    // Main loop
    // *********************************************************************
    if( scfIter == 1 ){
      EigenSolver::Orthogonalize( heightLocal2, width, X, Xtemp, XTX, XTXtemp1 );

      AlltoallBackward( mb, nb, ncom, X, Xcol, mpi_comm_ );
    }

    // Applying the Hamiltonian matrix
    {
      Spinor spnTemp(fftPtr_->domain, ncom, noccTotal, false, Xcol.Data());
      NumTns<Complex> tnsTemp(ntot, ncom, noccLocal, false, AXcol.Data());

      hamPtr_->MultSpinor( spnTemp, tnsTemp, *fftPtr_ );

      EigenSolver::NonlocalMultX( heightLocal, noccTotal, X, AX );

      AlltoallForwardAdd( mb, nb, ncom, AXcol, AX, mpi_comm_ );
    }

    // Start the main loop
    Int iter = 0;

    do{

      iter++;

      if( iter == 1 || isRestart == true )
        numSet = 2;
      else
        numSet = 3;

      // XTX <- X' * (AX)
      blas::Gemm( 'C', 'N', width, width, heightLocal2, 1.0, X.Data(),
          heightLocal2, AX.Data(), heightLocal2, 0.0, XTXtemp1.Data(), width );
      SetValue( XTX, 0.0 );
      MPI_Allreduce( XTXtemp1.Data(), XTX.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm_ );
      lapack::Lacpy( 'A', width, width, XTX.Data(), width, AMat.Data(), lda );

      // Compute the residual.
      // R <- AX - X*(X'*AX)
      lapack::Lacpy( 'A', heightLocal2, width, AX.Data(), heightLocal2, Xtemp.Data(), heightLocal2 );
      blas::Gemm( 'N', 'N', heightLocal2, width, width, -1.0, 
          X.Data(), heightLocal2, AMat.Data(), lda, 1.0, Xtemp.Data(), heightLocal2 );

      // Compute the norm of the residual
      SetValue( resNormLocal, 0.0 );
      for( Int k = 0; k < width; k++ ){
        resNormLocal(k) = Energy(CpxNumVec(heightLocal2, false, Xtemp.VecData(k)));
      }

      SetValue( resNorm, 0.0 );
      MPI_Allreduce( resNormLocal.Data(), resNorm.Data(), width, MPI_DOUBLE, 
          MPI_SUM, mpi_comm_ );

      if ( mpirank == 0 ){
        for( Int k = 0; k < width; k++ ){
          resNorm(k) = std::sqrt( resNorm(k) ) / std::max( 1.0, std::abs( XTX(k,k) ) );
        }
      }
      MPI_Bcast(resNorm.Data(), width, MPI_DOUBLE, 0, mpi_comm_);

      resMax = *(std::max_element( resNorm.Data(), resNorm.Data() + numEig ) );
      resMin = *(std::min_element( resNorm.Data(), resNorm.Data() + numEig ) );

      notconv = 0;
      for( Int i = 0; i < numEig; i++ ){
        if( resNorm[i] > eigTolerance ){
          notconv ++;
        }
      }

      if( notconv == 0 ) isConverged = true;

      numActiveTotal = width - numLockedTotal;
      numActiveLocal = widthLocal - numLockedLocal;

      AlltoallBackward( mb, nb, ncom, Xtemp, Xcol, mpi_comm_ );

      // Compute the preconditioned residual W = T*R.
      // The residual is saved in Xtemp
      {
        Spinor spnTemp(fftPtr_->domain, ncom, noccTotal, false, Xcol.VecData(numLockedLocal));
        NumTns<Complex> tnsTemp(ntot, ncom, widthLocal-numLockedLocal, false, Wcol.VecData(numLockedLocal));

        SetValue( tnsTemp, Complex(0.0, 0.0) );
        spnTemp.AddTeterPrecond( fftPtr_, hamPtr_->Teter(), tnsTemp );
      }

      Real norm = 0.0; 
      // Normalize the preconditioned residual
      for( Int k = numLockedLocal; k < widthLocal; k++ ){
        norm = Energy(CpxNumVec(height2, false, Wcol.VecData(k)));
        norm = std::sqrt( norm );
        blas::Scal( height2, 1.0 / norm, Wcol.VecData(k), 1 );
      }

      AlltoallForward( mb, nb, ncom, Wcol, W, mpi_comm_ );

      // Normalize the conjugate direction
      Real normPLocal[width];
      Real normP[width];
      if( numSet == 3 ){
        for( Int k = numLockedLocal; k < width; k++ ){
          normPLocal[k] = Energy(CpxNumVec(heightLocal2, false, P.VecData(k)));
          normP[k] = 0.0;
        }
        MPI_Allreduce( &normPLocal[0], &normP[0], width, MPI_DOUBLE, MPI_SUM, mpi_comm_ );
        for( Int k = numLockedLocal; k < width; k++ ){
          norm = std::sqrt( normP[k] );
          blas::Scal( heightLocal2, 1.0 / norm, P.VecData(k), 1 );
          blas::Scal( heightLocal2, 1.0 / norm, AP.VecData(k), 1 );
        }
      }

      // Compute AMat

      // Compute AW = A*W
      {
        Spinor spnTemp(fftPtr_->domain, ncom, noccTotal, false, Wcol.VecData(numLockedLocal));
        NumTns<Complex> tnsTemp(ntot, ncom, widthLocal-numLockedLocal, false, AWcol.VecData(numLockedLocal));

        hamPtr_->MultSpinor( spnTemp, tnsTemp, *fftPtr_ );
   
        EigenSolver::NonlocalMultX( heightLocal, noccTotal, W, AW );

        AlltoallForwardAdd( mb, nb, ncom, AWcol, AW, mpi_comm_ );
      }

      // Compute X' * (AW)
      // Instead of saving the block at &AMat(0,width+numLocked), the data
      // is saved at &AMat(0,width) to guarantee a continuous data
      // arrangement of AMat.  The same treatment applies to the blocks
      // below in both AMat and BMat.
      blas::Gemm( 'C', 'N', width, numActive, heightLocal2, 1.0, X.Data(),
          heightLocal2, AW.VecData(numLocked), heightLocal2, 
          0.0, XTXtemp1.Data(), width );
      SetValue( XTXtemp, 0.0 );
      MPI_Allreduce( XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm_ );
      lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, &AMat(0,width), lda );

      // Compute W' * (AW)
      blas::Gemm( 'C', 'N', numActive, numActive, heightLocal2, 1.0,
          W.VecData(numLocked), heightLocal2, AW.VecData(numLocked), heightLocal2, 
          0.0, XTXtemp1.Data(), width );
      SetValue( XTXtemp, 0.0 );
      MPI_Allreduce( XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm_ );
      lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, &AMat(width,width), lda );

      if( numSet == 3 ){
        // Compute X' * (AP)
        blas::Gemm( 'C', 'N', width, numActive, heightLocal2, 1.0,
            X.Data(), heightLocal2, AP.VecData(numLocked), heightLocal2, 
            0.0, XTXtemp1.Data(), width );
        SetValue( XTXtemp, 0.0 );
        MPI_Allreduce( XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm_ );
        lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, &AMat(0, width+numActive), lda );

        // Compute W' * (AP)
        blas::Gemm( 'C', 'N', numActive, numActive, heightLocal2, 1.0,
            W.VecData(numLocked), heightLocal2, AP.VecData(numLocked), heightLocal2, 
            0.0, XTXtemp1.Data(), width );
        SetValue( XTXtemp, 0.0 );
        MPI_Allreduce( XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm_ );
        lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, &AMat(width, width+numActive), lda );

        // Compute P' * (AP)
        blas::Gemm( 'C', 'N', numActive, numActive, heightLocal2, 1.0,
            P.VecData(numLocked), heightLocal2, AP.VecData(numLocked), heightLocal2, 
            0.0, XTXtemp1.Data(), width );
        SetValue( XTXtemp, 0.0 );
        MPI_Allreduce( XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm_ );
        lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, &AMat(width+numActive, width+numActive), lda );
      }

      // Compute BMat (overlap matrix)

      // Compute X'*X
      blas::Gemm( 'C', 'N', width, width, heightLocal2, 1.0, 
          X.Data(), heightLocal2, X.Data(), heightLocal2, 
          0.0, XTXtemp1.Data(), width );
      SetValue( XTXtemp, 0.0 );
      MPI_Allreduce( XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm_ );
      lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, &BMat(0,0), lda );

      // Compute X'*W
      blas::Gemm( 'C', 'N', width, numActive, heightLocal2, 1.0,
          X.Data(), heightLocal2, W.VecData(numLocked), heightLocal2,
          0.0, XTXtemp1.Data(), width );
      SetValue( XTXtemp, 0.0 );
      MPI_Allreduce( XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm_ );
      lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, &BMat(0,width), lda );

      // Compute W'*W
      blas::Gemm( 'C', 'N', numActive, numActive, heightLocal2, 1.0,
          W.VecData(numLocked), heightLocal2, W.VecData(numLocked), heightLocal2,
          0.0, XTXtemp1.Data(), width );
      SetValue( XTXtemp, 0.0 );
      MPI_Allreduce( XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm_ );
      lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, &BMat(width, width), lda );

      if( numSet == 3 ){
        // Compute X'*P
        blas::Gemm( 'C', 'N', width, numActive, heightLocal2, 1.0,
            X.Data(), heightLocal2, P.VecData(numLocked), heightLocal2, 
            0.0, XTXtemp1.Data(), width );
        SetValue( XTXtemp, 0.0 );
        MPI_Allreduce( XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm_ );
        lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, &BMat(0, width+numActive), lda );

        // Compute W'*P
        blas::Gemm( 'C', 'N', numActive, numActive, heightLocal2, 1.0,
            W.VecData(numLocked), heightLocal2, P.VecData(numLocked), heightLocal2,
            0.0, XTXtemp1.Data(), width );
        SetValue( XTXtemp, 0.0 );
        MPI_Allreduce( XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm_ );
        lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, &BMat(width, width+numActive), lda );

        // Compute P'*P
        blas::Gemm( 'C', 'N', numActive, numActive, heightLocal2, 1.0,
            P.VecData(numLocked), heightLocal2, P.VecData(numLocked), heightLocal2,
            0.0, XTXtemp1.Data(), width );
        SetValue( XTXtemp, 0.0 );
        MPI_Allreduce( XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm_ );
        lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, &BMat(width+numActive, width+numActive), lda );
      } // if( numSet == 3 )

      // Rayleigh-Ritz procedure
      // AMat * C = BMat * C * Lambda
      // Assuming the dimension (needed) for C is width * width, then
      //     ( C_X )
      //     ( --- )
      // C = ( C_W )
      //     ( --- )
      //     ( C_P )
      Int numCol;
      if( numSet == 3 ){
        // Conjugate gradient
        numCol = width + 2 * numActiveTotal;
      }
      else{
        numCol = width + numActiveTotal;
      }

      EigenSolver::SolveGenEig( lda, numCol, width, AMat, BMat, eigValS );

      // All processors synchronize the information
      MPI_Bcast(AMat.Data(), lda*lda, MPI_DOUBLE, 0, mpi_comm_);
      MPI_Bcast(BMat.Data(), lda*lda, MPI_DOUBLE, 0, mpi_comm_);
      MPI_Bcast(eigValS.Data(), lda, MPI_DOUBLE, 0, mpi_comm_);

      if( numSet == 2 ){
        // Update the eigenvectors 
        // X <- X * C_X + W * C_W
        blas::Gemm( 'N', 'N', heightLocal2, width, width, 1.0,
            X.Data(), heightLocal2, &AMat(0,0), lda,
            0.0, Xtemp.Data(), heightLocal2 );

        blas::Gemm( 'N', 'N', heightLocal2, width, numActive, 1.0,
            W.VecData(numLocked), heightLocal2, &AMat(width,0), lda,
            1.0, Xtemp.Data(), heightLocal2 );

        // Save the result into X
        lapack::Lacpy( 'A', heightLocal2, width, Xtemp.Data(), heightLocal2, 
            X.Data(), heightLocal2 );

        // P <- W
        lapack::Lacpy( 'A', heightLocal2, numActive, W.VecData(numLocked), 
            heightLocal2, P.VecData(numLocked), heightLocal2 );
      } 
      else{ // numSet == 3
        // Compute the conjugate direction
        // P <- W * C_W + P * C_P
        blas::Gemm( 'N', 'N', heightLocal2, width, numActive, 1.0,
            W.VecData(numLocked), heightLocal2, &AMat(width, 0), lda, 
            0.0, Xtemp.Data(), heightLocal2 );

        blas::Gemm( 'N', 'N', heightLocal2, width, numActive, 1.0,
            P.VecData(numLocked), heightLocal2, &AMat(width+numActive,0), lda,
            1.0, Xtemp.Data(), heightLocal2 );

        lapack::Lacpy( 'A', heightLocal2, numActive, Xtemp.VecData(numLocked), 
            heightLocal2, P.VecData(numLocked), heightLocal2 );

        // Update the eigenvectors
        // X <- X * C_X + P
        blas::Gemm( 'N', 'N', heightLocal2, width, width, 1.0, 
            X.Data(), heightLocal2, &AMat(0,0), lda, 
            1.0, Xtemp.Data(), heightLocal2 );

        lapack::Lacpy( 'A', heightLocal2, width, Xtemp.Data(), heightLocal2,
            X.Data(), heightLocal2 );
      } // if ( numSet == 2 )
  
      // Update AX and AP
      if( numSet == 2 ){
        // AX <- AX * C_X + AW * C_W
        blas::Gemm( 'N', 'N', heightLocal2, width, width, 1.0,
            AX.Data(), heightLocal2, &AMat(0,0), lda,
            0.0, Xtemp.Data(), heightLocal2 );

        blas::Gemm( 'N', 'N', heightLocal2, width, numActive, 1.0,
            AW.VecData(numLocked), heightLocal2, &AMat(width,0), lda,
            1.0, Xtemp.Data(), heightLocal2 );

        lapack::Lacpy( 'A', heightLocal2, width, Xtemp.Data(), heightLocal2,
            AX.Data(), heightLocal2 );

        // AP <- AW
        lapack::Lacpy( 'A', heightLocal2, numActive, AW.VecData(numLocked), heightLocal2,
            AP.VecData(numLocked), heightLocal2 );
      }
      else{
        // AP <- AW * C_W + A_P * C_P
        blas::Gemm( 'N', 'N', heightLocal2, width, numActive, 1.0, 
            AW.VecData(numLocked), heightLocal2, &AMat(width,0), lda,
            0.0, Xtemp.Data(), heightLocal2 );

        blas::Gemm( 'N', 'N', heightLocal2, width, numActive, 1.0,
            AP.VecData(numLocked), heightLocal2, &AMat(width+numActive, 0), lda,
            1.0, Xtemp.Data(), heightLocal2 );

        lapack::Lacpy( 'A', heightLocal2, numActive, Xtemp.VecData(numLocked),
            heightLocal2, AP.VecData(numLocked), heightLocal2 );

        // AX <- AX * C_X + AP
        blas::Gemm( 'N', 'N', heightLocal2, width, width, 1.0,
            AX.Data(), heightLocal2, &AMat(0,0), lda,
            1.0, Xtemp.Data(), heightLocal2 );

        lapack::Lacpy( 'A', heightLocal2, width, Xtemp.Data(), heightLocal2, 
            AX.Data(), heightLocal2 );
      } // if ( numSet == 2 )
    } while( (iter < eigMaxIter) && (resMax > eigTolerance) );

    // *********************************************************************
    // Post processing
    // *********************************************************************

    // Obtain the eigenvalues and eigenvectors
    // XTX should now contain the matrix X' * (AX), and X is an
    // orthonormal set
    if ( mpirank == 0 ){
      lapack::Syevd( 'V', 'U', width, XTX.Data(), width, eigValS.Data() );
    }

    MPI_Bcast(XTX.Data(), width*width, MPI_DOUBLE, 0, mpi_comm_);
    MPI_Bcast(eigValS.Data(), width, MPI_DOUBLE, 0, mpi_comm_);

    // X <- X*C
    blas::Gemm( 'N', 'N', heightLocal2, width, width, 1.0, X.Data(),
        heightLocal2, XTX.Data(), width, 0.0, Xtemp.Data(), heightLocal2 );

    lapack::Lacpy( 'A', heightLocal2, width, Xtemp.Data(), heightLocal2,
        X.Data(), heightLocal2 );

    // Save the eigenvalues and eigenvectors back to the eigensolver data
    // structure
    blas::Copy( width, eigValS.Data(), 1, &eigVal_[noccTotal*spinswitch], 1 );
    blas::Copy( width, resNorm.Data(), 1, &resVal_[noccTotal*spinswitch], 1 );

    AlltoallBackward( mb, nb, ncom, X, Xcol, mpi_comm_ );

    lapack::Lacpy( 'A', height2, widthLocal, Xcol.Data(), height2, 
        psiTemp.WavefunG().MatData(noccLocal*spinswitch), height2 );

    if( isConverged ){
      statusOFS << std::endl << "After " << iter 
        << " iterations, LOBPCG has converged."  << std::endl
        << "The maximum norm of the residual is " 
        << resMax << std::endl << std::endl
        << "The minimum norm of the residual is " 
        << resMin << std::endl << std::endl;
    }
    else{
      statusOFS << std::endl << "After " << iter 
        << " iterations, LOBPCG did not converge. " << std::endl
        << "The maximum norm of the residual is " 
        << resMax << std::endl << std::endl
        << "The minimum norm of the residual is " 
        << resMin << std::endl << std::endl;
    }
  }
  else{
    // S = ( X | W | P ) is a triplet used for LOBPCG.  
    // W is the preconditioned residual
    DblNumMat  S( heightLocal2, 3*width ), AS( heightLocal2, 3*width ); 
    // AMat = S' * (AS),  BMat = S' * S
    // 
    // AMat = (X'*AX   X'*AW   X'*AP)
    //      = (  *     W'*AW   W'*AP)
    //      = (  *       *     P'*AP)
    //
    // BMat = (X'*X   X'*W   X'*P)
    //      = (  *    W'*W   W'*P)
    //      = (  *      *    P'*P)
    DblNumMat  AMat( 3*width, 3*width ), BMat( 3*width, 3*width );
    DblNumMat  AMatT1( 3*width, 3*width );

    // Temporary buffer array.
    // The unpreconditioned residual will also be saved in Xtemp
    DblNumMat  XTX( width, width );
    DblNumMat  XTXtemp( width, width );
    DblNumMat  XTXtemp1( width, width );

    DblNumMat  Xtemp( heightLocal2, width );

    // rexNorm Grobal matrix similar to numEig 
    DblNumVec  resNormLocal ( width ); 
    SetValue( resNormLocal, 0.0 );
    DblNumVec  resNorm( width );
    SetValue( resNorm, 0.0 );
    Real       resMax, resMin;

    // For convenience
    DblNumMat  X( heightLocal2, width, false, S.VecData(0) );
    DblNumMat  W( heightLocal2, width, false, S.VecData(width) );
    DblNumMat  P( heightLocal2, width, false, S.VecData(2*width) );
    DblNumMat AX( heightLocal2, width, false, AS.VecData(0) );
    DblNumMat AW( heightLocal2, width, false, AS.VecData(width) );
    DblNumMat AP( heightLocal2, width, false, AS.VecData(2*width) );

    DblNumMat  Xcol( height2, widthLocal );
    DblNumMat  Wcol( height2, widthLocal );
    DblNumMat AXcol( height2, widthLocal );
    DblNumMat AWcol( height2, widthLocal );

    bool isRestart = false;
    // numSet = 2    : Steepest descent (Davidson), only use (X | W)
    //        = 3    : Conjugate gradient, use all the triplet (X | W | P)
    Int numSet = 2;

    // numLocked is the number of converged vectors
    Int numLockedLocal = 0, numLockedSaveLocal = 0;
    Int numLockedTotal = 0, numLockedSaveTotal = 0; 
    Int numLockedSave = 0;
    Int numActiveLocal = 0;
    Int numActiveTotal = 0;

    const Int numLocked = 0;  // Never perform locking in this version
    const Int numActive = width;

    bool isConverged = false;

    // Initialization
    SetValue( S, 0.0 );
    SetValue( AS, 0.0 );

    DblNumVec  eigValS(lda);
    SetValue( eigValS, 0.0 );

    // Initialize X by the data in psi
    lapack::Lacpy( 'A', height2, widthLocal, psiTemp.Wavefun().MatData(spinswitch*noccLocal), height2, 
      Xcol.Data(), height2 );

    AlltoallForward( mb, nb, Xcol, X, mpi_comm_ );

    // *********************************************************************
    // Main loop
    // *********************************************************************
    if( scfIter == 1 ){
      EigenSolver::Orthogonalize( heightLocal2, width, X, Xtemp, XTX, XTXtemp1 );

      AlltoallBackward( mb, nb, X, Xcol, mpi_comm_ );
    }

    // Applying the Hamiltonian matrix
    {
      Spinor spnTemp(fftPtr_->domain, ncom, noccTotal, false, Xcol.Data());
      NumTns<Real> tnsTemp(ntot, ncom, noccLocal, false, AXcol.Data());

      hamPtr_->MultSpinor( spnTemp, tnsTemp, *fftPtr_ );

      AlltoallForward( mb, nb, AXcol, AX, mpi_comm_ );
    }

    // Start the main loop
    Int iter = 0;

    do{

      iter++;

      if( iter == 1 || isRestart == true )
        numSet = 2;
      else
        numSet = 3;

      // XTX <- X' * (AX)
      blas::Gemm( 'T', 'N', width, width, heightLocal2, 1.0, X.Data(),
          heightLocal2, AX.Data(), heightLocal2, 0.0, XTXtemp1.Data(), width );
      SetValue( XTX, 0.0 );
      MPI_Allreduce( XTXtemp1.Data(), XTX.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm_ );
      lapack::Lacpy( 'A', width, width, XTX.Data(), width, AMat.Data(), lda );

      // Compute the residual.
      // R <- AX - X*(X'*AX)
      lapack::Lacpy( 'A', heightLocal2, width, AX.Data(), heightLocal2, Xtemp.Data(), heightLocal2 );
      blas::Gemm( 'N', 'N', heightLocal2, width, width, -1.0, 
          X.Data(), heightLocal2, AMat.Data(), lda, 1.0, Xtemp.Data(), heightLocal2 );

      // Compute the norm of the residual
      SetValue( resNormLocal, 0.0 );
      for( Int k = 0; k < width; k++ ){
        resNormLocal(k) = Energy(DblNumVec(heightLocal2, false, Xtemp.VecData(k)));
      }

      SetValue( resNorm, 0.0 );
      MPI_Allreduce( resNormLocal.Data(), resNorm.Data(), width, MPI_DOUBLE, 
          MPI_SUM, mpi_comm_ );

      if ( mpirank == 0 ){
        for( Int k = 0; k < width; k++ ){
          resNorm(k) = std::sqrt( resNorm(k) ) / std::max( 1.0, std::abs( XTX(k,k) ) );
        }
      }
      MPI_Bcast(resNorm.Data(), width, MPI_DOUBLE, 0, mpi_comm_);

      resMax = *(std::max_element( resNorm.Data(), resNorm.Data() + numEig ) );
      resMin = *(std::min_element( resNorm.Data(), resNorm.Data() + numEig ) );

      notconv = 0;
      for( Int i = 0; i < numEig; i++ ){
        if( resNorm[i] > eigTolerance ){
          notconv ++;
        }
      }

      if( notconv == 0 ) isConverged = true;

      numActiveTotal = width - numLockedTotal;
      numActiveLocal = widthLocal - numLockedLocal;

      AlltoallBackward( mb, nb, Xtemp, Xcol, mpi_comm_ );

      // Compute the preconditioned residual W = T*R.
      // The residual is saved in Xtemp
      {
        Spinor spnTemp(fftPtr_->domain, ncom, noccTotal, false, Xcol.VecData(numLockedLocal));
        NumTns<Real> tnsTemp(ntot, ncom, widthLocal-numLockedLocal, false, Wcol.VecData(numLockedLocal));

        SetValue( tnsTemp, 0.0 );
        spnTemp.AddTeterPrecond( fftPtr_, hamPtr_->Teter(), tnsTemp );
      }

      Real norm = 0.0; 
      // Normalize the preconditioned residual
      for( Int k = numLockedLocal; k < widthLocal; k++ ){
        norm = Energy(DblNumVec(height2, false, Wcol.VecData(k)));
        norm = std::sqrt( norm );
        blas::Scal( height2, 1.0 / norm, Wcol.VecData(k), 1 );
      }

      // Normalize the conjugate direction
      Real normPLocal[width];
      Real normP[width];
      if( numSet == 3 ){
        for( Int k = numLockedLocal; k < width; k++ ){
          normPLocal[k] = Energy(DblNumVec(heightLocal2, false, P.VecData(k)));
          normP[k] = 0.0;
        }
        MPI_Allreduce( &normPLocal[0], &normP[0], width, MPI_DOUBLE, MPI_SUM, mpi_comm_ );
        for( Int k = numLockedLocal; k < width; k++ ){
          norm = std::sqrt( normP[k] );
          blas::Scal( heightLocal2, 1.0 / norm, P.VecData(k), 1 );
          blas::Scal( heightLocal2, 1.0 / norm, AP.VecData(k), 1 );
        }
      }

      // Compute AMat

      // Compute AW = A*W
      {
        Spinor spnTemp(fftPtr_->domain, ncom, noccTotal, false, Wcol.VecData(numLockedLocal));
        NumTns<Real> tnsTemp(ntot, ncom, widthLocal-numLockedLocal, false, AWcol.VecData(numLockedLocal));

        hamPtr_->MultSpinor( spnTemp, tnsTemp, *fftPtr_ );
      }

      // Convert from column format to row format
      // Only convert W and AW
      AlltoallForward( mb, nb, Wcol, W, mpi_comm_ );
      AlltoallForward( mb, nb, AWcol, AW, mpi_comm_ );

      // Compute X' * (AW)
      // Instead of saving the block at &AMat(0,width+numLocked), the data
      // is saved at &AMat(0,width) to guarantee a continuous data
      // arrangement of AMat.  The same treatment applies to the blocks
      // below in both AMat and BMat.
      blas::Gemm( 'T', 'N', width, numActive, heightLocal2, 1.0, X.Data(),
          heightLocal2, AW.VecData(numLocked), heightLocal2, 
          0.0, XTXtemp1.Data(), width );
      SetValue( XTXtemp, 0.0 );
      MPI_Allreduce( XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm_ );
      lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, &AMat(0,width), lda );

      // Compute W' * (AW)
      blas::Gemm( 'T', 'N', numActive, numActive, heightLocal2, 1.0,
          W.VecData(numLocked), heightLocal2, AW.VecData(numLocked), heightLocal2, 
          0.0, XTXtemp1.Data(), width );
      SetValue( XTXtemp, 0.0 );
      MPI_Allreduce( XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm_ );
      lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, &AMat(width,width), lda );

      if( numSet == 3 ){
        // Compute X' * (AP)
        blas::Gemm( 'T', 'N', width, numActive, heightLocal2, 1.0,
            X.Data(), heightLocal2, AP.VecData(numLocked), heightLocal2, 
            0.0, XTXtemp1.Data(), width );
        SetValue( XTXtemp, 0.0 );
        MPI_Allreduce( XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm_ );
        lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, &AMat(0, width+numActive), lda );

        // Compute W' * (AP)
        blas::Gemm( 'T', 'N', numActive, numActive, heightLocal2, 1.0,
            W.VecData(numLocked), heightLocal2, AP.VecData(numLocked), heightLocal2, 
            0.0, XTXtemp1.Data(), width );
        SetValue( XTXtemp, 0.0 );
        MPI_Allreduce( XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm_ );
        lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, &AMat(width, width+numActive), lda );

        // Compute P' * (AP)
        blas::Gemm( 'T', 'N', numActive, numActive, heightLocal2, 1.0,
            P.VecData(numLocked), heightLocal2, AP.VecData(numLocked), heightLocal2, 
            0.0, XTXtemp1.Data(), width );
        SetValue( XTXtemp, 0.0 );
        MPI_Allreduce( XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm_ );
        lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, &AMat(width+numActive, width+numActive), lda );
      }

      // Compute BMat (overlap matrix)

      // Compute X'*X
      blas::Gemm( 'T', 'N', width, width, heightLocal2, 1.0, 
          X.Data(), heightLocal2, X.Data(), heightLocal2, 
          0.0, XTXtemp1.Data(), width );
      SetValue( XTXtemp, 0.0 );
      MPI_Allreduce( XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm_ );
      lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, &BMat(0,0), lda );

      // Compute X'*W
      blas::Gemm( 'T', 'N', width, numActive, heightLocal2, 1.0,
          X.Data(), heightLocal2, W.VecData(numLocked), heightLocal2,
          0.0, XTXtemp1.Data(), width );
      SetValue( XTXtemp, 0.0 );
      MPI_Allreduce( XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm_ );
      lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, &BMat(0,width), lda );

      // Compute W'*W
      blas::Gemm( 'T', 'N', numActive, numActive, heightLocal2, 1.0,
          W.VecData(numLocked), heightLocal2, W.VecData(numLocked), heightLocal2,
          0.0, XTXtemp1.Data(), width );
      SetValue( XTXtemp, 0.0 );
      MPI_Allreduce( XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm_ );
      lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, &BMat(width, width), lda );

      if( numSet == 3 ){
        // Compute X'*P
        blas::Gemm( 'T', 'N', width, numActive, heightLocal2, 1.0,
            X.Data(), heightLocal2, P.VecData(numLocked), heightLocal2, 
            0.0, XTXtemp1.Data(), width );
        SetValue( XTXtemp, 0.0 );
        MPI_Allreduce( XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm_ );
        lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, &BMat(0, width+numActive), lda );

        // Compute W'*P
        blas::Gemm( 'T', 'N', numActive, numActive, heightLocal2, 1.0,
            W.VecData(numLocked), heightLocal2, P.VecData(numLocked), heightLocal2,
            0.0, XTXtemp1.Data(), width );
        SetValue( XTXtemp, 0.0 );
        MPI_Allreduce( XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm_ );
        lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, &BMat(width, width+numActive), lda );

        // Compute P'*P
        blas::Gemm( 'T', 'N', numActive, numActive, heightLocal2, 1.0,
            P.VecData(numLocked), heightLocal2, P.VecData(numLocked), heightLocal2,
            0.0, XTXtemp1.Data(), width );
        SetValue( XTXtemp, 0.0 );
        MPI_Allreduce( XTXtemp1.Data(), XTXtemp.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm_ );
        lapack::Lacpy( 'A', width, width, XTXtemp.Data(), width, &BMat(width+numActive, width+numActive), lda );
      } // if( numSet == 3 )

      // Rayleigh-Ritz procedure
      // AMat * C = BMat * C * Lambda
      // Assuming the dimension (needed) for C is width * width, then
      //     ( C_X )
      //     ( --- )
      // C = ( C_W )
      //     ( --- )
      //     ( C_P )
      Int numCol;
      if( numSet == 3 ){
        // Conjugate gradient
        numCol = width + 2 * numActiveTotal;
      }
      else{
        numCol = width + numActiveTotal;
      }

      EigenSolver::SolveGenEig( lda, numCol, width, AMat, BMat, eigValS );

      // All processors synchronize the information
      MPI_Bcast(AMat.Data(), lda*lda, MPI_DOUBLE, 0, mpi_comm_);
      MPI_Bcast(BMat.Data(), lda*lda, MPI_DOUBLE, 0, mpi_comm_);
      MPI_Bcast(eigValS.Data(), lda, MPI_DOUBLE, 0, mpi_comm_);

      if( numSet == 2 ){
        // Update the eigenvectors 
        // X <- X * C_X + W * C_W
        blas::Gemm( 'N', 'N', heightLocal2, width, width, 1.0,
            X.Data(), heightLocal2, &AMat(0,0), lda,
            0.0, Xtemp.Data(), heightLocal2 );

        blas::Gemm( 'N', 'N', heightLocal2, width, numActive, 1.0,
            W.VecData(numLocked), heightLocal2, &AMat(width,0), lda,
            1.0, Xtemp.Data(), heightLocal2 );

        // Save the result into X
        lapack::Lacpy( 'A', heightLocal2, width, Xtemp.Data(), heightLocal2, 
            X.Data(), heightLocal2 );

        // P <- W
        lapack::Lacpy( 'A', heightLocal2, numActive, W.VecData(numLocked), 
            heightLocal2, P.VecData(numLocked), heightLocal2 );
      } 
      else{ // numSet == 3
        // Compute the conjugate direction
        // P <- W * C_W + P * C_P
        blas::Gemm( 'N', 'N', heightLocal2, width, numActive, 1.0,
            W.VecData(numLocked), heightLocal2, &AMat(width, 0), lda, 
            0.0, Xtemp.Data(), heightLocal2 );

        blas::Gemm( 'N', 'N', heightLocal2, width, numActive, 1.0,
            P.VecData(numLocked), heightLocal2, &AMat(width+numActive,0), lda,
            1.0, Xtemp.Data(), heightLocal2 );

        lapack::Lacpy( 'A', heightLocal2, numActive, Xtemp.VecData(numLocked), 
            heightLocal2, P.VecData(numLocked), heightLocal2 );

        // Update the eigenvectors
        // X <- X * C_X + P
        blas::Gemm( 'N', 'N', heightLocal2, width, width, 1.0, 
            X.Data(), heightLocal2, &AMat(0,0), lda, 
            1.0, Xtemp.Data(), heightLocal2 );

        lapack::Lacpy( 'A', heightLocal2, width, Xtemp.Data(), heightLocal2,
            X.Data(), heightLocal2 );
      } // if ( numSet == 2 )
  
      // Update AX and AP
      if( numSet == 2 ){
        // AX <- AX * C_X + AW * C_W
        blas::Gemm( 'N', 'N', heightLocal2, width, width, 1.0,
            AX.Data(), heightLocal2, &AMat(0,0), lda,
            0.0, Xtemp.Data(), heightLocal2 );

        blas::Gemm( 'N', 'N', heightLocal2, width, numActive, 1.0,
            AW.VecData(numLocked), heightLocal2, &AMat(width,0), lda,
            1.0, Xtemp.Data(), heightLocal2 );

        lapack::Lacpy( 'A', heightLocal2, width, Xtemp.Data(), heightLocal2,
            AX.Data(), heightLocal2 );

        // AP <- AW
        lapack::Lacpy( 'A', heightLocal2, numActive, AW.VecData(numLocked), heightLocal2,
            AP.VecData(numLocked), heightLocal2 );
      }
      else{
        // AP <- AW * C_W + A_P * C_P
        blas::Gemm( 'N', 'N', heightLocal2, width, numActive, 1.0, 
            AW.VecData(numLocked), heightLocal2, &AMat(width,0), lda,
            0.0, Xtemp.Data(), heightLocal2 );

        blas::Gemm( 'N', 'N', heightLocal2, width, numActive, 1.0,
            AP.VecData(numLocked), heightLocal2, &AMat(width+numActive, 0), lda,
            1.0, Xtemp.Data(), heightLocal2 );

        lapack::Lacpy( 'A', heightLocal2, numActive, Xtemp.VecData(numLocked),
            heightLocal2, AP.VecData(numLocked), heightLocal2 );

        // AX <- AX * C_X + AP
        blas::Gemm( 'N', 'N', heightLocal2, width, width, 1.0,
            AX.Data(), heightLocal2, &AMat(0,0), lda,
            1.0, Xtemp.Data(), heightLocal2 );

        lapack::Lacpy( 'A', heightLocal2, width, Xtemp.Data(), heightLocal2, 
            AX.Data(), heightLocal2 );
      } // if ( numSet == 2 )
    } while( (iter < eigMaxIter) && (resMax > eigTolerance) );

    // *********************************************************************
    // Post processing
    // *********************************************************************

    // Obtain the eigenvalues and eigenvectors
    // XTX should now contain the matrix X' * (AX), and X is an
    // orthonormal set
    if ( mpirank == 0 ){
      lapack::Syevd( 'V', 'U', width, XTX.Data(), width, eigValS.Data() );
    }

    MPI_Bcast(XTX.Data(), width*width, MPI_DOUBLE, 0, mpi_comm_);
    MPI_Bcast(eigValS.Data(), width, MPI_DOUBLE, 0, mpi_comm_);

    // X <- X*C
    blas::Gemm( 'N', 'N', heightLocal2, width, width, 1.0, X.Data(),
        heightLocal2, XTX.Data(), width, 0.0, Xtemp.Data(), heightLocal2 );

    lapack::Lacpy( 'A', heightLocal2, width, Xtemp.Data(), heightLocal2,
        X.Data(), heightLocal2 );

    // Save the eigenvalues and eigenvectors back to the eigensolver data
    // structure
    blas::Copy( width, eigValS.Data(), 1, &eigVal_[noccTotal*spinswitch], 1 );
    blas::Copy( width, resNorm.Data(), 1, &resVal_[noccTotal*spinswitch], 1 );

    AlltoallBackward( mb, nb, X, Xcol, mpi_comm_ );

    lapack::Lacpy( 'A', height2, widthLocal, Xcol.Data(), height2, 
        psiTemp.Wavefun().MatData(noccLocal*spinswitch), height2 );

    if( isConverged ){
      statusOFS << std::endl << "After " << iter 
        << " iterations, LOBPCG has converged."  << std::endl
        << "The maximum norm of the residual is " 
        << resMax << std::endl << std::endl
        << "The minimum norm of the residual is " 
        << resMin << std::endl << std::endl;
    }
    else{
      statusOFS << std::endl << "After " << iter 
        << " iterations, LOBPCG did not converge. " << std::endl
        << "The maximum norm of the residual is " 
        << resMax << std::endl << std::endl
        << "The minimum norm of the residual is " 
        << resMin << std::endl << std::endl;
    }
  }    // ---- end of if( !realspace ) ----
 
  return;
}         // -----  end of method EigenSolver::LOBPCGSolveReal  ----- 

#endif

} // namespace pwdft
