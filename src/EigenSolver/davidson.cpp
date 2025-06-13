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
/// @file davidson.cpp
/// @brief Davidson eigensolver for real-value Hamiltonian
/// @date 2024-06-19 extract codes for Davidson algorithm from eigensolver.cpp
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
// NOTE: This is the scalable version.
void
EigenSolver::DavidsonSolveReal    (
    Int          numEig,
    Int          scfIter,
    Int          maxdim,
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
    noccTotal /= 2;
    noccLocal /= 2;
  }
  // For spin-noncollinear case, the spin-up and spin-down components
  // are chunked in the same way and communicated in pair
  Int height = ntot;
  Int height2 = height * ncom;
  Int width = noccTotal;
  Int lda = maxdim * width;

  Int widthLocal = noccLocal;
  Int heightLocal = ntotLocal;
  Int heightLocal2 = heightLocal * ncom;

  // The number of active bands, i.e. unconverged bands 
  Int naLocal = widthLocal;
  Int naTotal = width;
  // The size of subspace
  Int nbLocal = widthLocal;
  Int nbTotal = width;

  if( numEig > width ){
    std::ostringstream msg;
    msg 
      << "Number of eigenvalues requested  = " << numEig << std::endl
      << "which is larger than the number of columns in psi = " << width << std::endl;
    ErrorHandling( msg.str().c_str() );
  }

  if( !realspace ){
    // S = ( X | W ) is maxdim fold used for Davidson
    // W is the preconditioned residual
    CpxNumMat  S( heightLocal2, lda ), AS( heightLocal2, lda ); 
    // AMat = S' * (AS),  BMat = S' * S
    // 
    // AMat = (X'*AX   X'*AW)
    //      = (  *     W'*AW)
    //
    // BMat = (X'*X   X'*W)
    //      = (  *    W'*W)
    DblNumMat  AMat( lda, lda ), BMat( lda, lda );
    DblNumMat  AMatsave( lda, lda ), BMatsave( lda, lda );

    // Temporary buffer array with variant size
    DblNumMat  XTX( width, width );
    DblNumMat  XTXtemp( width, width );

    // The unpreconditioned residual will also be saved in Xtemp
    CpxNumMat  Xtemp( heightLocal2, maxdim * width );
    CpxNumMat  Xcoltemp( height2, maxdim * widthLocal );

    // For convenience
    CpxNumMat  X( heightLocal2, width, false, S.VecData(0) );
    CpxNumMat  W( heightLocal2, width, false, S.VecData(width) );
    CpxNumMat  AX( heightLocal2, width, false, AS.VecData(0) );
    CpxNumMat  AW( heightLocal2, width, false, AS.VecData(width) );

    // FIXME the memory needed is not equal for each process
    // how to solve this problem ?
    CpxNumMat  Xcol( height2, widthLocal );
    CpxNumMat  Wcol( height2, 100*widthLocal );
    CpxNumMat  AXcol( height2, widthLocal );
    CpxNumMat  AWcol( height2, 100*widthLocal );

    IntNumVec  conv( width );

    DblNumVec  eigValS(width), eigValStemp(lda);

    DblNumVec  resNorm(width);

    bool isConverged = false;

    // Initialization
    SetValue( S, Complex( 0.0, 0.0) );
    SetValue( AS, Complex( 0.0, 0.0) );

    SetValue( AMat, 0.0 );
    SetValue( BMat, 0.0 );
    SetValue( AMatsave, 0.0 );
    SetValue( BMatsave, 0.0 );

    SetValue( conv, I_ZERO );

    SetValue( eigValS, 0.0 );
    SetValue( eigValStemp, 0.0);
    SetValue( resNorm, 0.0 );

    // Initialize X by the data in psi
    lapack::Lacpy( 'A', height2, widthLocal, psiTemp.WavefunG().MatData(spinswitch*noccLocal), height2, 
        Xcol.Data(), height2 );

    AlltoallForward( mb, nb, ncom, Xcol, X, mpi_comm_ );

    // *********************************************************************
    // Main loop
    // *********************************************************************
    if( scfIter == 1 ){
      EigenSolver::Orthogonalize( heightLocal2, width, X, Xtemp, XTX, XTXtemp );

      AlltoallBackward( mb, nb, ncom, X, Xcol, mpi_comm_ );
    }

    // Applying the Hamiltonian matrix
    {
      Spinor spnTemp(fftPtr_->domain, ncom, width, false, Xcol.Data() );
      NumTns<Complex> tnsTemp(ntot, ncom, widthLocal, false, AXcol.Data());

      hamPtr_->MultSpinor( spnTemp, tnsTemp, *fftPtr_ );

      EigenSolver::NonlocalMultX( heightLocal, noccTotal, X, AX );           

      AlltoallForwardAdd( mb, nb, ncom, AXcol, AX, mpi_comm_ );
    }

    // AMat <- X' * (AX)
    blas::Gemm( 'C', 'N', width, width, heightLocal2, 1.0, X.Data(),
        heightLocal2, AX.Data(), heightLocal2, 0.0, XTXtemp.Data(), width );

    SetValue( XTX, 0.0 );
    MPI_Allreduce( XTXtemp.Data(), XTX.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm_ );

    lapack::Lacpy( 'A', width, width, XTX.Data(), width, AMat.Data(), lda );
    
    lapack::Lacpy( 'A', width, width, XTX.Data(), width, AMatsave.Data(), lda );

    // BMat <- X' * (X)
    blas::Gemm( 'C', 'N', width, width, heightLocal2, 1.0, X.Data(),
        heightLocal2, X.Data(), heightLocal2, 0.0, XTXtemp.Data(), width );

    SetValue( XTX, 0.0 );
    MPI_Allreduce( XTXtemp.Data(), XTX.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm_ );

    lapack::Lacpy( 'A', width, width, XTX.Data(), width, BMat.Data(), lda );

    lapack::Lacpy( 'A', width, width, XTX.Data(), width, BMatsave.Data(), lda );

    // Solve the first generalized eigenvalue problem for RMM in subspace X
    Int numCol = width;

    EigenSolver::SolveGenEig( lda, numCol, width, AMat, BMat, eigValStemp );

    // All processors synchronize the information
    MPI_Bcast(AMat.Data(), lda*lda, MPI_DOUBLE, 0, mpi_comm_);
    MPI_Bcast(BMat.Data(), lda*lda, MPI_DOUBLE, 0, mpi_comm_);
    MPI_Bcast(eigValStemp.Data(), lda, MPI_DOUBLE, 0, mpi_comm_);

    blas::Copy( numCol, eigValStemp.Data(), 1, eigValS.Data(), 1 );

    // Iterations

    for( Int iter = 1; iter <= eigMaxIter; iter++ ){

      statusOFS << "Davidson iter " << iter << std::endl;

      // Move the unconverged eigenvectors and eigenvalues to front
      Int j = 0;
      for( Int i = 0; i < width; i++ ){
        if( !conv(i) ){
          if( j != i ){
            blas::Copy( lda, AMat.VecData(i), 1, AMat.VecData(j), 1 );
          }

          eigValStemp(j+nbTotal) = eigValS(i);

          j++;
        }
      }  
      
      // Compute residual of eigenpairs W = HX - XE for rotated unconverged X in subspace [X | W]
      // and push them to W in order
      blas::Gemm( 'N', 'N', heightLocal2, naTotal, nbTotal, -1.0, 
          X.Data(), heightLocal2, AMat.Data(), lda, 0.0, W.VecData(nbTotal-width), heightLocal2 );
      
      for( Int j = 0; j < naTotal; j++ ){
        for( Int i = 0; i < heightLocal2; i++){
          W(i,j+nbTotal-width) *= eigValStemp(j+nbTotal);
        }
      }

      blas::Gemm( 'N', 'N', heightLocal2, naTotal, nbTotal, 1.0, 
          AX.Data(), heightLocal2, AMat.Data(), lda, 1.0, W.VecData(nbTotal-width), heightLocal2 ); 

      // Transform W from row to column
      // only the new added part of W has to be converted
      {
        CpxNumMat WsubRow( heightLocal2, naTotal, false, W.VecData(nbTotal-width) );
        CpxNumMat WsubCol( height2, naLocal, false, Xcoltemp.VecData(0) );

        AlltoallBackward( mb, nb, ncom, WsubRow, WsubCol, mpi_comm_ );
      }

      // Compute W = TW
      {
        Spinor spnTemp(fftPtr_->domain, ncom, naTotal, false, Xcoltemp.Data() );
        NumTns<Complex> tnsTemp(ntot, ncom, naLocal, false, Wcol.VecData(nbLocal-widthLocal));

        SetValue( tnsTemp, Complex(0.0, 0.0));
        spnTemp.AddTeterPrecond( fftPtr_, hamPtr_->Teter(), tnsTemp );
      }

      // Convert from column format to row format
      {
        CpxNumMat WsubRow( heightLocal2, naTotal, false, W.VecData(nbTotal-width) );
        CpxNumMat WsubCol( height2, naLocal, false, Wcol.VecData(nbLocal-widthLocal) );

        AlltoallForward( mb, nb, ncom, WsubCol, WsubRow, mpi_comm_ );
      }

      // Normalize columns of W
      Real normLocal = 0.0, norm = 0.0;
      for( Int k = 0; k < naTotal; k++ ){
        normLocal = Energy(CpxNumVec(heightLocal2, false, W.VecData(k+nbTotal-width)));
        norm = 0.0;
        MPI_Allreduce( &normLocal, &norm, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_ );
        eigValStemp(k) = norm;
        norm = std::sqrt( norm );
        blas::Scal( heightLocal2, 1.0 / norm, W.VecData(k+nbTotal-width), 1 );
      }

      {
        CpxNumMat WsubRow( heightLocal2, naTotal, false, W.VecData(nbTotal-width) );
        CpxNumMat WsubCol( height2, naLocal, false, Wcol.VecData(nbLocal-widthLocal) );

        AlltoallBackward( mb, nb, ncom, WsubRow, WsubCol, mpi_comm_ );
      }

      // Compute AW = A*W
      {
        Spinor spnTemp(fftPtr_->domain, ncom, naTotal, false, Wcol.VecData(nbLocal-widthLocal));
        NumTns<Complex> tnsTemp(ntot, ncom, naLocal, false, AWcol.VecData(nbLocal-widthLocal));

        hamPtr_->MultSpinor( spnTemp, tnsTemp, *fftPtr_ );
        
        CpxNumMat AWsubCol( height2, naLocal, false, AWcol.VecData(nbLocal-widthLocal) );
        CpxNumMat WsubRow( heightLocal2, naTotal, false, W.VecData(nbTotal-width) );
        CpxNumMat AWsubRow( heightLocal2, naTotal, false, AW.VecData(nbTotal-width) );

        EigenSolver::NonlocalMultX( heightLocal, naTotal, WsubRow, AWsubRow, mpi_comm_ );

        AlltoallForwardAdd( mb, nb, ncom, AWsubCol, AWsubRow, mpi_comm_ );
      }

      // Complete the enlarged AMat and BMat
      // The core part
      lapack::Lacpy( 'A', nbTotal, nbTotal, AMatsave.Data(), lda, AMat.Data(), lda );
      lapack::Lacpy( 'A', nbTotal, nbTotal, BMatsave.Data(), lda, BMat.Data(), lda );

      XTX.Resize( nbTotal+naTotal, naTotal );
      XTXtemp.Resize( nbTotal+naTotal, naTotal );

      blas::Gemm( 'C', 'N', nbTotal+naTotal, naTotal, heightLocal2, 1.0, S.Data(),
          heightLocal2, AW.VecData(nbTotal-width), heightLocal2, 0.0, XTXtemp.Data(), nbTotal+naTotal );

      SetValue( XTX, D_ZERO );
      MPI_Allreduce( XTXtemp.Data(), XTX.Data(), naTotal*(nbTotal+naTotal), MPI_DOUBLE, MPI_SUM, mpi_comm_ );

      lapack::Lacpy( 'A', nbTotal+naTotal, naTotal, XTX.Data(), 
          nbTotal+naTotal, &AMat(0,nbTotal), lda );

      lapack::Lacpy( 'A', nbTotal+naTotal, naTotal, XTX.Data(),
          nbTotal+naTotal, &AMatsave(0,nbTotal), lda );

      blas::Gemm( 'C', 'N', nbTotal+naTotal, naTotal, heightLocal2, 1.0, X.Data(),
          heightLocal2, W.VecData(nbTotal-width), heightLocal2, 0.0, XTXtemp.Data(), nbTotal+naTotal );

      SetValue( XTX, D_ZERO );
      MPI_Allreduce( XTXtemp.Data(), XTX.Data(), naTotal*(nbTotal+naTotal), MPI_DOUBLE, MPI_SUM, mpi_comm_ );

      lapack::Lacpy( 'A', nbTotal+naTotal, naTotal, XTX.Data(),
          nbTotal+naTotal, &BMat(0,nbTotal), lda );

      lapack::Lacpy( 'A', nbTotal+naTotal, naTotal, XTX.Data(),
          nbTotal+naTotal, &BMatsave(0,nbTotal), lda );
      // Solve RMM problem in the enlarged subspace
      nbTotal = nbTotal + naTotal;
      nbLocal = nbLocal + naLocal;

      numCol = nbTotal;

      EigenSolver::SolveGenEig( lda, numCol, width, AMat, BMat, eigValStemp );

      MPI_Bcast(AMat.Data(), lda*lda, MPI_DOUBLE, 0, mpi_comm_);
      MPI_Bcast(BMat.Data(), lda*lda, MPI_DOUBLE, 0, mpi_comm_);
      MPI_Bcast(eigValStemp.Data(), lda, MPI_DOUBLE, 0, mpi_comm_);

      // Determine whether the band converges
      naTotal = 0;
      for( Int i = 0; i < width; i++ ){
        resNorm(i) = std::abs(eigValS(i) - eigValStemp(i));
        if( resNorm(i) < eigTolerance ){
          conv(i) = 1;
        }
        else{
          conv(i) = 0;
          naTotal ++;
        }      
      }

      Int nres;
      IntNumVec nlocal( mpisize );
      CalculateSizeAlltoall( naTotal, nb, nres, nlocal, mpi_comm_ );

      naLocal = nlocal( mpirank );

      isConverged = ( naTotal == 0 );

      blas::Copy( width, eigValStemp.Data(), 1, eigValS.Data(), 1 );

      if( isConverged || nbTotal + naTotal > lda || iter == eigMaxIter ){
        // *********************************************************************
        // Post processing
        // *********************************************************************

        // X <- X*C
        blas::Gemm( 'N', 'N', heightLocal2, width, nbTotal, 1.0, X.Data(),
            heightLocal2, AMat.Data(), lda, 0.0, Xtemp.Data(), heightLocal2 );

        lapack::Lacpy( 'A', heightLocal2, width, Xtemp.Data(), heightLocal2,
            X.Data(), heightLocal2 );

        // The size of matrix changes back to ntot * width
        AlltoallBackward( mb, nb, ncom, X, Xcol, mpi_comm_ );

        if( isConverged || iter == eigMaxIter ){
          lapack::Lacpy( 'A', height2, widthLocal, Xcol.Data(), height2, 
              psiTemp.WavefunG().MatData(noccLocal*spinswitch), height2 );

          blas::Copy( width, eigValS.Data(), 1, &eigVal_[noccTotal*spinswitch], 1 );
          blas::Copy( width, resNorm.Data(), 1, &resVal_[noccTotal*spinswitch], 1 );

          if( isConverged ){
            statusOFS << std::endl << "After " << iter
              << " iterations, Davidson has converged."  << std::endl;
          }
          else{
            statusOFS << std::endl << "After " << iter
              << " iterations, Davidson did not converge. " << std::endl
              << "The number of unconverged bands is  "
              << naTotal << std::endl << std::endl;
          }
        
          return;
        }
        else{
          // Restart if the size of subspace exceeds the max number ( default as 2 * width )
          blas::Gemm( 'N', 'N', heightLocal2, width, nbTotal, 1.0, AX.Data(),
              heightLocal2, AMat.Data(), lda, 0.0, W.Data(), heightLocal2 );

          lapack::Lacpy( 'A', heightLocal2, width, W.Data(), heightLocal2,
              AX.Data(), heightLocal2 );

          nbTotal = width;
          nbLocal = widthLocal;

          for( Int i = 0; i < nbTotal; i++ ){
            for( Int j = 0; j < nbTotal; j++ ){
              AMat(i,j) = D_ZERO;
              BMat(i,j) = D_ZERO;
              AMatsave(i,j) = D_ZERO;
              BMatsave(i,j) = D_ZERO;
            }
          }

          for( Int i = 0; i < nbTotal; i++ ){
            AMatsave(i,i) = eigValS(i);
            BMatsave(i,i) = D_ONE;
            AMat(i,i) = D_ONE;
            BMat(i,i) = D_ONE;
          }
        }    // ---- end of if( isConverged || iter == eigMaxIter ) ----
      }    // ---- end of if( isConverged || nbTotal + naTotal > lda || iter == eigMaxIter ) ----
    }    // for (iter)
  }
  else{
    // S = ( X | W ) is maxdim fold used for Davidson
    // W is the preconditioned residual
    DblNumMat  S( heightLocal2, lda ), AS( heightLocal2, lda ); 
    // AMat = S' * (AS),  BMat = S' * S
    // 
    // AMat = (X'*AX   X'*AW)
    //      = (  *     W'*AW)
    //
    // BMat = (X'*X   X'*W)
    //      = (  *    W'*W)
    DblNumMat  AMat( lda, lda ), BMat( lda, lda );
    DblNumMat  AMatsave( lda, lda ), BMatsave( lda, lda );

    // Temporary buffer array with variant size
    DblNumMat  XTX( width, width );
    DblNumMat  XTXtemp( width, width );

    // The unpreconditioned residual will also be saved in Xtemp
    DblNumMat  Xtemp( heightLocal2, maxdim * width );
    DblNumMat  Xcoltemp( height2, maxdim * widthLocal );

    // For convenience
    DblNumMat  X( heightLocal2, width, false, S.VecData(0) );
    DblNumMat  W( heightLocal2, width, false, S.VecData(width) );
    DblNumMat  AX( heightLocal2, width, false, AS.VecData(0) );
    DblNumMat  AW( heightLocal2, width, false, AS.VecData(width) );

    // FIXME the memory needed is not equal for each process
    // how to solve this problem ?
    DblNumMat  Xcol( height2, widthLocal );
    DblNumMat  Wcol( height2, 100*widthLocal );
    DblNumMat  AXcol( height2, widthLocal );
    DblNumMat  AWcol( height2, 100*widthLocal );

    IntNumVec  conv( width );

    DblNumVec  eigValS(width), eigValStemp(lda);

    DblNumVec  resNorm(width);

    bool isConverged = false;

    // Initialization
    SetValue( S, 0.0 );
    SetValue( AS, 0.0 );

    SetValue( AMat, 0.0 );
    SetValue( BMat, 0.0 );
    SetValue( AMatsave, 0.0 );
    SetValue( BMatsave, 0.0 );

    SetValue( conv, I_ZERO );

    SetValue( eigValS, 0.0 );
    SetValue( eigValStemp, 0.0);
    SetValue( resNorm, 0.0 );

    // Initialize X by the data in psi
    lapack::Lacpy( 'A', height2, widthLocal, psiTemp.Wavefun().MatData(spinswitch*noccLocal), height2, 
        Xcol.Data(), height2 );

    AlltoallForward( mb, nb, Xcol, X, mpi_comm_ );

    // *********************************************************************
    // Main loop
    // *********************************************************************
    if( scfIter == 1 ){
      EigenSolver::Orthogonalize( heightLocal2, width, X, Xtemp, XTX, XTXtemp );

      AlltoallBackward( mb, nb, X, Xcol, mpi_comm_ );
    }

    // Applying the Hamiltonian matrix
    {
      Spinor spnTemp(fftPtr_->domain, ncom, width, false, Xcol.Data() );
      NumTns<Real> tnsTemp(ntot, ncom, widthLocal, false, AXcol.Data());

      hamPtr_->MultSpinor( spnTemp, tnsTemp, *fftPtr_ );

      AlltoallForward( mb, nb, AXcol, AX, mpi_comm_ );
    }

    // AMat <- X' * (AX)
    blas::Gemm( 'T', 'N', width, width, heightLocal2, 1.0, X.Data(),
        heightLocal2, AX.Data(), heightLocal2, 0.0, XTXtemp.Data(), width );

    SetValue( XTX, 0.0 );
    MPI_Allreduce( XTXtemp.Data(), XTX.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm_ );

    lapack::Lacpy( 'A', width, width, XTX.Data(), width, AMat.Data(), lda );
    
    lapack::Lacpy( 'A', width, width, XTX.Data(), width, AMatsave.Data(), lda );

    // BMat <- X' * (X)
    blas::Gemm( 'T', 'N', width, width, heightLocal2, 1.0, X.Data(),
        heightLocal2, X.Data(), heightLocal2, 0.0, XTXtemp.Data(), width );

    SetValue( XTX, 0.0 );
    MPI_Allreduce( XTXtemp.Data(), XTX.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm_ );

    lapack::Lacpy( 'A', width, width, XTX.Data(), width, BMat.Data(), lda );

    lapack::Lacpy( 'A', width, width, XTX.Data(), width, BMatsave.Data(), lda );

    // Solve the first generalized eigenvalue problem for RMM in subspace X
    Int numCol = width;

    EigenSolver::SolveGenEig( lda, numCol, width, AMat, BMat, eigValStemp );

    // All processors synchronize the information
    MPI_Bcast(AMat.Data(), lda*lda, MPI_DOUBLE, 0, mpi_comm_);
    MPI_Bcast(BMat.Data(), lda*lda, MPI_DOUBLE, 0, mpi_comm_);
    MPI_Bcast(eigValStemp.Data(), lda, MPI_DOUBLE, 0, mpi_comm_);

    blas::Copy( numCol, eigValStemp.Data(), 1, eigValS.Data(), 1 );

    // Iterations

    for( Int iter = 1; iter <= eigMaxIter; iter++ ){

      // Move the unconverged eigenvectors and eigenvalues to front
      Int j = 0;
      for( Int i = 0; i < width; i++ ){
        if( !conv(i) ){
          if( j != i ){
            blas::Copy( lda, AMat.VecData(i), 1, AMat.VecData(j), 1 );
          }

          eigValStemp(j+nbTotal) = eigValS(i);

          j++;
        }
      }  
      
      // Compute residual of eigenpairs W = HX - XE for rotated unconverged X in subspace [X | W]
      // and push them to W in order
      blas::Gemm( 'N', 'N', heightLocal2, naTotal, nbTotal, -1.0, 
          X.Data(), heightLocal2, AMat.Data(), lda, 0.0, W.VecData(nbTotal-width), heightLocal2 );
      
      for( Int j = 0; j < naTotal; j++ ){
        for( Int i = 0; i < heightLocal2; i++){
          W(i,j+nbTotal-width) *= eigValStemp(j+nbTotal);
        }
      }

      blas::Gemm( 'N', 'N', heightLocal2, naTotal, nbTotal, 1.0, 
          AX.Data(), heightLocal2, AMat.Data(), lda, 1.0, W.VecData(nbTotal-width), heightLocal2 ); 

      // Transform W from row to column
      // only the new added part of W has to be converted
      {
        DblNumMat WsubRow( heightLocal2, naTotal, false, W.VecData(nbTotal-width) );
        DblNumMat WsubCol( height2, naLocal, false, Xcoltemp.VecData(0) );

        AlltoallBackward( mb, nb, WsubRow, WsubCol, mpi_comm_ );
      }

      // Compute W = TW
      {
        Spinor spnTemp(fftPtr_->domain, ncom, naTotal, false, Xcoltemp.Data() );
        NumTns<Real> tnsTemp(ntot, ncom, naLocal, false, Wcol.VecData(nbLocal-widthLocal));

        SetValue( tnsTemp, 0.0);
        spnTemp.AddTeterPrecond( fftPtr_, hamPtr_->Teter(), tnsTemp );
      }

      // Convert from column format to row format
      {
        DblNumMat WsubRow( heightLocal2, naTotal, false, W.VecData(nbTotal-width) );
        DblNumMat WsubCol( height2, naLocal, false, Wcol.VecData(nbLocal-widthLocal) );

        AlltoallForward( mb, nb, WsubCol, WsubRow, mpi_comm_ );
      }

      // Normalize columns of W
      Real normLocal = 0.0, norm = 0.0;
      for( Int k = 0; k < naTotal; k++ ){
        normLocal = Energy(DblNumVec(heightLocal2, false, W.VecData(k+nbTotal-width)));
        norm = 0.0;
        MPI_Allreduce( &normLocal, &norm, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_ );
        eigValStemp(k) = norm;
        norm = std::sqrt( norm );
        blas::Scal( heightLocal2, 1.0 / norm, W.VecData(k+nbTotal-width), 1 );
      }

      {
        DblNumMat WsubRow( heightLocal2, naTotal, false, W.VecData(nbTotal-width) );
        DblNumMat WsubCol( height2, naLocal, false, Wcol.VecData(nbLocal-widthLocal) );

        AlltoallBackward( mb, nb, WsubRow, WsubCol, mpi_comm_ );
      }

      // Compute AW = A*W
      {
        Spinor spnTemp(fftPtr_->domain, ncom, naTotal, false, Wcol.VecData(nbLocal-widthLocal));
        NumTns<Real> tnsTemp(ntot, ncom, naLocal, false, AWcol.VecData(nbLocal-widthLocal));

        hamPtr_->MultSpinor( spnTemp, tnsTemp, *fftPtr_ );
        
        DblNumMat AWsubCol( height2, naLocal, false, AWcol.VecData(nbLocal-widthLocal) );
        DblNumMat WsubRow( heightLocal2, naTotal, false, W.VecData(nbTotal-width) );
        DblNumMat AWsubRow( heightLocal2, naTotal, false, AW.VecData(nbTotal-width) );

        AlltoallForward( mb, nb, AWsubCol, AWsubRow, mpi_comm_ );
      }

      // Complete the enlarged AMat and BMat
      // The core part
      lapack::Lacpy( 'A', nbTotal, nbTotal, AMatsave.Data(), lda, AMat.Data(), lda );
      lapack::Lacpy( 'A', nbTotal, nbTotal, BMatsave.Data(), lda, BMat.Data(), lda );

      XTX.Resize( nbTotal+naTotal, naTotal );
      XTXtemp.Resize( nbTotal+naTotal, naTotal );

      blas::Gemm( 'T', 'N', nbTotal+naTotal, naTotal, heightLocal2, 1.0, S.Data(),
          heightLocal2, AW.VecData(nbTotal-width), heightLocal2, 0.0, XTXtemp.Data(), nbTotal+naTotal );

      SetValue( XTX, D_ZERO );
      MPI_Allreduce( XTXtemp.Data(), XTX.Data(), naTotal*(nbTotal+naTotal), MPI_DOUBLE, MPI_SUM, mpi_comm_ );

      lapack::Lacpy( 'A', nbTotal+naTotal, naTotal, XTX.Data(), 
          nbTotal+naTotal, &AMat(0,nbTotal), lda );

      lapack::Lacpy( 'A', nbTotal+naTotal, naTotal, XTX.Data(),
          nbTotal+naTotal, &AMatsave(0,nbTotal), lda );

      blas::Gemm( 'T', 'N', nbTotal+naTotal, naTotal, heightLocal2, 1.0, X.Data(),
          heightLocal2, W.VecData(nbTotal-width), heightLocal2, 0.0, XTXtemp.Data(), nbTotal+naTotal );

      SetValue( XTX, D_ZERO );
      MPI_Allreduce( XTXtemp.Data(), XTX.Data(), naTotal*(nbTotal+naTotal), MPI_DOUBLE, MPI_SUM, mpi_comm_ );

      lapack::Lacpy( 'A', nbTotal+naTotal, naTotal, XTX.Data(),
          nbTotal+naTotal, &BMat(0,nbTotal), lda );

      lapack::Lacpy( 'A', nbTotal+naTotal, naTotal, XTX.Data(),
          nbTotal+naTotal, &BMatsave(0,nbTotal), lda );
      // Solve RMM problem in the enlarged subspace
      nbTotal = nbTotal + naTotal;
      nbLocal = nbLocal + naLocal;

      numCol = nbTotal;

      EigenSolver::SolveGenEig( lda, numCol, width, AMat, BMat, eigValStemp );

      MPI_Bcast(AMat.Data(), lda*lda, MPI_DOUBLE, 0, mpi_comm_);
      MPI_Bcast(BMat.Data(), lda*lda, MPI_DOUBLE, 0, mpi_comm_);
      MPI_Bcast(eigValStemp.Data(), lda, MPI_DOUBLE, 0, mpi_comm_);

      // Determine whether the band converges
      naTotal = 0;
      for( Int i = 0; i < width; i++ ){
        resNorm(i) = std::abs(eigValS(i) - eigValStemp(i));
        if( resNorm(i) < eigTolerance ){
          conv(i) = 1;
        }
        else{
          conv(i) = 0;
          naTotal ++;
        }      
      }

      Int nres;
      IntNumVec nlocal( mpisize );
      CalculateSizeAlltoall( naTotal, nb, nres, nlocal, mpi_comm_ );

      naLocal = nlocal( mpirank );

      isConverged = ( naTotal == 0 );

      blas::Copy( width, eigValStemp.Data(), 1, eigValS.Data(), 1 );

      if( isConverged || nbTotal + naTotal > lda || iter == eigMaxIter ){
        // *********************************************************************
        // Post processing
        // *********************************************************************

        // X <- X*C
        blas::Gemm( 'N', 'N', heightLocal2, width, nbTotal, 1.0, X.Data(),
            heightLocal2, AMat.Data(), lda, 0.0, Xtemp.Data(), heightLocal2 );

        lapack::Lacpy( 'A', heightLocal2, width, Xtemp.Data(), heightLocal2,
            X.Data(), heightLocal2 );

        // The size of matrix changes back to ntot * width
        AlltoallBackward( mb, nb, X, Xcol, mpi_comm_ );

        if( isConverged || iter == eigMaxIter ){
          lapack::Lacpy( 'A', height2, widthLocal, Xcol.Data(), height2, 
              psiTemp.Wavefun().MatData(noccLocal*spinswitch), height2 );

          blas::Copy( width, eigValS.Data(), 1, &eigVal_[noccTotal*spinswitch], 1 );
          blas::Copy( width, resNorm.Data(), 1, &resVal_[noccTotal*spinswitch], 1 );

          if( isConverged ){
            statusOFS << std::endl << "After " << iter
              << " iterations, Davidson has converged."  << std::endl;
          }
          else{
            statusOFS << std::endl << "After " << iter
              << " iterations, Davidson did not converge. " << std::endl
              << "The number of unconverged bands is  "
              << naTotal << std::endl << std::endl;
          }
        
          return;
        }
        else{
          // Restart if the size of subspace exceeds the max number ( default as 2 * width )
          blas::Gemm( 'N', 'N', heightLocal2, width, nbTotal, 1.0, AX.Data(),
              heightLocal2, AMat.Data(), lda, 0.0, W.Data(), heightLocal2 );

          lapack::Lacpy( 'A', heightLocal2, width, W.Data(), heightLocal2,
              AX.Data(), heightLocal2 );

          nbTotal = width;
          nbLocal = widthLocal;

          for( Int i = 0; i < nbTotal; i++ ){
            for( Int j = 0; j < nbTotal; j++ ){
              AMat(i,j) = D_ZERO;
              BMat(i,j) = D_ZERO;
              AMatsave(i,j) = D_ZERO;
              BMatsave(i,j) = D_ZERO;
            }
          }

          for( Int i = 0; i < nbTotal; i++ ){
            AMatsave(i,i) = eigValS(i);
            BMatsave(i,i) = D_ONE;
            AMat(i,i) = D_ONE;
            BMat(i,i) = D_ONE;
          }
        }    // ---- end of if( isConverged || iter == eigMaxIter ) ----
      }    // ---- end of if( isConverged || nbTotal + naTotal > lda || iter == eigMaxIter ) ----
    }    // for (iter)
  }    // ---- end of if( !realspace ) ----
}         // -----  end of method EigenSolver::DavidsonSolveReal  ----- 

#endif

} // namespace pwdft
