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
/// @file ppcg.cpp
/// @brief PPCG eigensolver for real-value Hamiltonian
/// @date 2024-06-19 extract codes for PPCG from eigensolver.cpp
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
EigenSolver::PPCGSolveReal (
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
  Int height2 = ntot * ncom;
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
    // S = ( X | W | P ) is a triplet used for PPCG.  
    // W is the preconditioned residual
    CpxNumMat  S( heightLocal2, 3*width ), AS( heightLocal2, 3*width ); 

    // Temporary buffer array.
    // The unpreconditioned residual will also be saved in Xtemp
    DblNumMat  XTX( width, width );
    DblNumMat  XTXtemp( width, width );

    CpxNumMat  Xtemp( heightLocal2, width );

    // rexNorm Grobal matrix similar to numEig
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
    SetValue( S, Complex(0.0, 0.0) );
    SetValue( AS, Complex(0.0, 0.0) );

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
      EigenSolver::Orthogonalize( heightLocal2, width, X, Xtemp, XTX, XTXtemp );

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
          heightLocal2, AX.Data(), heightLocal2, 0.0, XTXtemp.Data(), width );
      SetValue( XTX, 0.0 );
      MPI_Allreduce( XTXtemp.Data(), XTX.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm_ );

      // Compute the residual.
      // R <- AX - X*(X'*AX)
      lapack::Lacpy( 'A', heightLocal2, width, AX.Data(), heightLocal2, Xtemp.Data(), heightLocal2 );
      blas::Gemm( 'N', 'N', heightLocal2, width, width, -1.0, 
          X.Data(), heightLocal2, XTX.Data(), width, 1.0, Xtemp.Data(), heightLocal2 );

      // Compute the norm of the residual block
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

      AlltoallBackward( mb, nb, ncom, Xtemp, Xcol, mpi_comm_ );

      // Compute W = TW
      {
        Spinor spnTemp(fftPtr_->domain, ncom, noccTotal, false, Xcol.VecData(numLockedLocal));
        NumTns<Complex> tnsTemp(ntot, ncom, widthLocal-numLockedLocal, false, Wcol.VecData(numLockedLocal));

        SetValue( tnsTemp, Complex(0.0, 0.0) );
        spnTemp.AddTeterPrecond( fftPtr_, hamPtr_->Teter(), tnsTemp );

        AlltoallForward( mb, nb, ncom, Wcol, W, mpi_comm_ );
      }

      // Compute AW = A*W
      {
        Spinor spnTemp(fftPtr_->domain, ncom, noccTotal, false, Wcol.VecData(numLockedLocal));
        NumTns<Complex> tnsTemp(ntot, ncom, widthLocal-numLockedLocal, false, AWcol.VecData(numLockedLocal));

        hamPtr_->MultSpinor( spnTemp, tnsTemp, *fftPtr_ );

        EigenSolver::NonlocalMultX( heightLocal, noccTotal, W, AW );

        AlltoallForwardAdd( mb, nb, ncom, AWcol, AW, mpi_comm_ );
      }

      // W = W - X(X'W), AW = AW - AX(X'W)
      blas::Gemm( 'C', 'N', width, width, heightLocal2, 1.0, X.Data(),
          heightLocal2, W.Data(), heightLocal2, 0.0, XTXtemp.Data(), width );
      SetValue( XTX, 0.0 );
      MPI_Allreduce( XTXtemp.Data(), XTX.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm_ );

      blas::Gemm( 'N', 'N', heightLocal2, width, width, -1.0, 
          X.Data(), heightLocal2, XTX.Data(), width, 1.0, W.Data(), heightLocal2 );

      blas::Gemm( 'N', 'N', heightLocal2, width, width, -1.0, 
          AX.Data(), heightLocal2, XTX.Data(), width, 1.0, AW.Data(), heightLocal2 );

      // Normalize columns of W
      Real normLocal[width]; 
      Real normGlobal[width];

      for( Int k = numLockedLocal; k < width; k++ ){
        normLocal[k] = Energy(CpxNumVec(heightLocal2, false, W.VecData(k)));
        normGlobal[k] = 0.0;
      }
 
      MPI_Allreduce( &normLocal[0], &normGlobal[0], width, MPI_DOUBLE, MPI_SUM, mpi_comm_ );
      for( Int k = numLockedLocal; k < width; k++ ){
        Real norm = std::sqrt( normGlobal[k] );
        blas::Scal( heightLocal2, 1.0 / norm, W.VecData(k), 1 );
        blas::Scal( heightLocal2, 1.0 / norm, AW.VecData(k), 1 );
      }

      // P = P - X(X'P), AP = AP - AX(X'P)
      if( numSet == 3 ){
        blas::Gemm( 'C', 'N', width, width, heightLocal2, 1.0, X.Data(),
            heightLocal2, P.Data(), heightLocal2, 0.0, XTXtemp.Data(), width );
        SetValue( XTX, 0.0 );
        MPI_Allreduce( XTXtemp.Data(), XTX.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm_ );

        blas::Gemm( 'N', 'N', heightLocal2, width, width, -1.0, 
            X.Data(), heightLocal2, XTX.Data(), width, 1.0, P.Data(), heightLocal2 );

        blas::Gemm( 'N', 'N', heightLocal2, width, width, -1.0, 
            AX.Data(), heightLocal2, XTX.Data(), width, 1.0, AP.Data(), heightLocal2 );

        // Normalize the conjugate direction
        for( Int k = numLockedLocal; k < width; k++ ){
          normLocal[k] = Energy(CpxNumVec(heightLocal2, false, P.VecData(k)));
          normGlobal[k] = 0.0;
        }
        MPI_Allreduce( &normLocal[0], &normGlobal[0], width, MPI_DOUBLE, MPI_SUM, mpi_comm_ );
        for( Int k = numLockedLocal; k < width; k++ ){
          Real norm = std::sqrt( normGlobal[k] );
          blas::Scal( heightLocal2, 1.0 / norm, P.VecData(k), 1 );
          blas::Scal( heightLocal2, 1.0 / norm, AP.VecData(k), 1 );
        }
      }

      // Perform the sweep
      Int sbSize = esdfParam.PPCGsbSize;
      Int nsb = (width + sbSize - 1) / sbSize ; 
      bool isDivid = ( width % sbSize == 0 );
      // Leading dimension of buff matrix
      Int sbSize1 = sbSize;
      Int sbSize2 = sbSize * 2;
      Int sbSize3 = sbSize * 3;

      // AMat and BMat have variant size dependent on sbSize for each small subspace
      DblNumMat AMat, BMat;

      // contains all nsb 3-by-3 matrices
      DblNumMat  AMatAll( sbSize3, sbSize3*nsb ), BMatAll( sbSize3, sbSize3*nsb ); 
      // contains local parts of all nsb 3-by-3 matrices
      DblNumMat  AMatAllLocal( sbSize3, sbSize3*nsb ), BMatAllLocal( sbSize3, sbSize3*nsb );
      SetValue( AMatAll, D_ZERO ); SetValue( BMatAll, D_ZERO );
      SetValue( AMatAllLocal, D_ZERO ); SetValue( BMatAllLocal, D_ZERO );

      // LOCKING NOT SUPPORTED, loop over all columns 
      for( Int k = 0; k < nsb; k++ ){

        if( (k == nsb - 1) && (!isDivid) )
          sbSize = width % sbSize;
        else
          sbSize = esdfParam.PPCGsbSize;

        // fetch indiviual columns
        CpxNumMat  x( heightLocal2, sbSize, false, X.VecData(sbSize1*k) );
        CpxNumMat  w( heightLocal2, sbSize, false, W.VecData(sbSize1*k) );
        CpxNumMat ax( heightLocal2, sbSize, false, AX.VecData(sbSize1*k) );
        CpxNumMat aw( heightLocal2, sbSize, false, AW.VecData(sbSize1*k) );

        // Compute AMatAllLoc and BMatAllLoc            
        // AMatAllLoc
        blas::Gemm( 'C', 'N', sbSize, sbSize, heightLocal2, 1.0, x.Data(),
            heightLocal2, ax.Data(), heightLocal2, 
            0.0, &AMatAllLocal(0,sbSize3*k), sbSize3 );

        blas::Gemm( 'C', 'N', sbSize, sbSize, heightLocal2, 1.0, w.Data(),
            heightLocal2, aw.Data(), heightLocal2, 
            0.0, &AMatAllLocal(sbSize1,sbSize3*k+sbSize1), sbSize3 );

        blas::Gemm( 'C', 'N', sbSize, sbSize, heightLocal2, 1.0, x.Data(),
            heightLocal2, aw.Data(), heightLocal2, 
            0.0, &AMatAllLocal(0,sbSize3*k+sbSize1), sbSize3 );

        // BMatAllLoc            
        blas::Gemm( 'C', 'N', sbSize, sbSize, heightLocal2, 1.0, x.Data(),
            heightLocal2, x.Data(), heightLocal2, 
            0.0, &BMatAllLocal(0,sbSize3*k), sbSize3 );

        blas::Gemm( 'C', 'N', sbSize, sbSize, heightLocal2, 1.0, w.Data(),
            heightLocal2, w.Data(), heightLocal2, 
            0.0, &BMatAllLocal(sbSize1,sbSize3*k+sbSize1), sbSize3 );

        blas::Gemm( 'C', 'N', sbSize, sbSize, heightLocal2, 1.0, x.Data(),
            heightLocal2, w.Data(), heightLocal2, 
            0.0, &BMatAllLocal(0,sbSize3*k+sbSize1), sbSize3 );

        if ( numSet == 3 ){

          CpxNumMat  p( heightLocal2, sbSize, false, P.VecData(sbSize1*k) );
          CpxNumMat ap( heightLocal2, sbSize, false, AP.VecData(sbSize1*k) );

          // AMatAllLoc
          blas::Gemm( 'C', 'N', sbSize, sbSize, heightLocal2, 1.0, p.Data(),
              heightLocal2, ap.Data(), heightLocal2, 
              0.0, &AMatAllLocal(sbSize2,sbSize3*k+sbSize2), sbSize3 );

          blas::Gemm( 'C', 'N', sbSize, sbSize, heightLocal2, 1.0, x.Data(),
              heightLocal2, ap.Data(), heightLocal2, 
              0.0, &AMatAllLocal(0, sbSize3*k+sbSize2), sbSize3 );

          blas::Gemm( 'C', 'N', sbSize, sbSize, heightLocal2, 1.0, w.Data(),
              heightLocal2, ap.Data(), heightLocal2, 
              0.0, &AMatAllLocal(sbSize1, sbSize3*k+sbSize2), sbSize3 );

          // BMatAllLoc
          blas::Gemm( 'C', 'N', sbSize, sbSize, heightLocal2, 1.0, p.Data(),
              heightLocal2, p.Data(), heightLocal2, 
              0.0, &BMatAllLocal(sbSize2,sbSize3*k+sbSize2), sbSize3 );

          blas::Gemm( 'C', 'N', sbSize, sbSize, heightLocal2, 1.0, x.Data(),
              heightLocal2, p.Data(), heightLocal2, 
              0.0, &BMatAllLocal(0, sbSize3*k+sbSize2), sbSize3 );

          blas::Gemm( 'C', 'N', sbSize, sbSize, heightLocal2, 1.0, w.Data(),
              heightLocal2, p.Data(), heightLocal2, 
              0.0, &BMatAllLocal(sbSize1, sbSize3*k+sbSize2), sbSize3 );
        }  // ---- End of if( numSet == 3 ) ----
      }  // for (k)             
      
      MPI_Allreduce( AMatAllLocal.Data(), AMatAll.Data(), sbSize3*sbSize3*nsb, MPI_DOUBLE, MPI_SUM, mpi_comm_ );
    
      MPI_Allreduce( BMatAllLocal.Data(), BMatAll.Data(), sbSize3*sbSize3*nsb, MPI_DOUBLE, MPI_SUM, mpi_comm_ );

      // Solve nsb small eigenproblems and update columns of X 
      for( Int k = 0; k < nsb; k++ ){

        if( (k == nsb - 1) && !isDivid )
          sbSize = width % sbSize;
        else
          sbSize = esdfParam.PPCGsbSize;

        Real eigs[3*sbSize];
        DblNumMat  cx( sbSize, sbSize ), cw( sbSize, sbSize ), cp( sbSize, sbSize);
        CpxNumMat tmp( heightLocal2, sbSize );            

        // small eigensolve
        AMat.Resize( sbSize*3, sbSize*3 );
        BMat.Resize( sbSize*3, sbSize*3 );
        if( k < nsb - 1 ){
          lapack::Lacpy( 'A', sbSize3, sbSize3, &AMatAll(0,sbSize3*k), sbSize3, AMat.Data(), sbSize3 );
          lapack::Lacpy( 'A', sbSize3, sbSize3, &BMatAll(0,sbSize3*k), sbSize3, BMat.Data(), sbSize3 );
        }
        else{
          for( Int i = 0; i < 3; i++ ){
            for( Int j = 0; j < 3; j++ ){
              lapack::Lacpy( 'A', sbSize, sbSize, &AMatAll(sbSize1*i,sbSize3*k+sbSize1*j), 
                  sbSize3, &AMat(sbSize*i,sbSize*j), sbSize*3 );
              lapack::Lacpy( 'A', sbSize, sbSize, &BMatAll(sbSize1*i,sbSize3*k+sbSize1*j), 
                  sbSize3, &BMat(sbSize*i,sbSize*j), sbSize*3 );
            }
          }
        }   

        Int dim = (numSet == 3) ? 3*sbSize : 2*sbSize;
        lapack::Sygvd(1, 'V', 'U', dim, AMat.Data(), 3*sbSize, BMat.Data(), 3*sbSize, eigs);

        // fetch indiviual columns
        CpxNumMat  x( heightLocal2, sbSize, false, X.VecData(sbSize1*k) );
        CpxNumMat  w( heightLocal2, sbSize, false, W.VecData(sbSize1*k) );
        CpxNumMat  p( heightLocal2, sbSize, false, P.VecData(sbSize1*k) );
        CpxNumMat ax( heightLocal2, sbSize, false, AX.VecData(sbSize1*k) );
        CpxNumMat aw( heightLocal2, sbSize, false, AW.VecData(sbSize1*k) );
        CpxNumMat ap( heightLocal2, sbSize, false, AP.VecData(sbSize1*k) );

        lapack::Lacpy( 'A', sbSize, sbSize, &AMat(0,0), 3*sbSize, cx.Data(), sbSize );
        lapack::Lacpy( 'A', sbSize, sbSize, &AMat(sbSize,0), 3*sbSize, cw.Data(), sbSize );

        // p = w*cw + p*cp; x = x*cx + p; ap = aw*cw + ap*cp; ax = ax*cx + ap;
        if( numSet == 3 ){
        
          lapack::Lacpy( 'A', sbSize, sbSize, &AMat(2*sbSize,0), 3*sbSize, cp.Data(), sbSize );
       
          // tmp <- p*cp 
          blas::Gemm( 'N', 'N', heightLocal2, sbSize, sbSize, 1.0,
              p.Data(), heightLocal2, cp.Data(), sbSize,
              0.0, tmp.Data(), heightLocal2 );

          // p <- w*cw + tmp
          blas::Gemm( 'N', 'N', heightLocal2, sbSize, sbSize, 1.0,
              w.Data(), heightLocal2, cw.Data(), sbSize,
              1.0, tmp.Data(), heightLocal2 );

          lapack::Lacpy( 'A', heightLocal2, sbSize, tmp.Data(), heightLocal2, p.Data(), heightLocal2 );

          // tmp <- ap*cp 
          blas::Gemm( 'N', 'N', heightLocal2, sbSize, sbSize, 1.0,
              ap.Data(), heightLocal2, cp.Data(), sbSize,
              0.0, tmp.Data(), heightLocal2 );

          // ap <- aw*cw + tmp
          blas::Gemm( 'N', 'N', heightLocal2, sbSize, sbSize, 1.0,
              aw.Data(), heightLocal2, cw.Data(), sbSize,
              1.0, tmp.Data(), heightLocal2 );
          lapack::Lacpy( 'A', heightLocal2, sbSize, tmp.Data(), heightLocal2, ap.Data(), heightLocal2 );
        }else{
          // p <- w*cw
          blas::Gemm( 'N', 'N', heightLocal2, sbSize, sbSize, 1.0,
              w.Data(), heightLocal2, cw.Data(), sbSize,
              0.0, p.Data(), heightLocal2 );

          // ap <- aw*cw
          blas::Gemm( 'N', 'N', heightLocal2, sbSize, sbSize, 1.0,
              aw.Data(), heightLocal2, cw.Data(), sbSize,
              0.0, ap.Data(), heightLocal2 );
        }

        // x <- x*cx + p
        lapack::Lacpy( 'A', heightLocal2, sbSize, p.Data(), heightLocal2, tmp.Data(), heightLocal2 );    
        blas::Gemm( 'N', 'N', heightLocal2, sbSize, sbSize, 1.0,
            x.Data(), heightLocal2, cx.Data(), sbSize,
            1.0, tmp.Data(), heightLocal2 );
        lapack::Lacpy( 'A', heightLocal2, sbSize, tmp.Data(), heightLocal2, x.Data(), heightLocal2 );

        // ax <- ax*cx + ap
        lapack::Lacpy( 'A', heightLocal2, sbSize, ap.Data(), heightLocal2, tmp.Data(), heightLocal2 );    
        blas::Gemm( 'N', 'N', heightLocal2, sbSize, sbSize, 1.0,
            ax.Data(), heightLocal2, cx.Data(), sbSize,
            1.0, tmp.Data(), heightLocal2 );
        lapack::Lacpy( 'A', heightLocal2, sbSize, tmp.Data(), heightLocal2, ax.Data(), heightLocal2 );
      }

      // CholeskyQR of the updated block X, AX is transformed meanwhile
      {
        EigenSolver::Orthogonalize( heightLocal2, width, X, Xtemp, XTX, XTXtemp );

        if( use_scala_ ){
          blas::Gemm( 'N', 'N', heightLocal2, width, width, 1.0,
              X.Data(), heightLocal2, XTX.Data(), width,
              0.0, Xtemp.Data(), heightLocal2 );
          lapack::Lacpy( 'A', heightLocal2, width, Xtemp.Data(), heightLocal2, AX.Data(), heightLocal2 );
        }
        else{
          blas::Trsm( 'R', 'U', 'N', 'N', heightLocal2, width, 1.0, XTX.Data(), width,
              AX.Data(), heightLocal2 ); 
        }   
      }
    } while( (iter < eigMaxIter) && (resMax > eigTolerance) );

    // *********************************************************************
    // Post processing
    // *********************************************************************

    // Obtain the eigenvalues and eigenvectors
    // if isConverged == true then XTX should contain the matrix X' * (AX); and X is an
    // orthonormal set
    {
      blas::Gemm( 'C', 'N', width, width, heightLocal2, 1.0, X.Data(),
          heightLocal2, AX.Data(), heightLocal2, 0.0, XTXtemp.Data(), width );
      SetValue( XTX, 0.0 );
      MPI_Allreduce( XTXtemp.Data(), XTX.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm_ );
    }

    if( use_scala_ )
    { 
      if( contxt_ >= 0 )
      {
        Int numKeep = width; 
        Int lda = width;

        scalapack::ScaLAPACKMatrix<Real> square_mat_scala;
        scalapack::ScaLAPACKMatrix<Real> eigvecs_scala;

        scalapack::Descriptor descReduceSeq, descReducePar;
        Real timeEigScala_sta, timeEigScala_end;

        // Leading dimension provided
        descReduceSeq.Init( numKeep, numKeep, numKeep, numKeep, I_ZERO, I_ZERO, contxt_, lda );

        // Automatically comptued Leading Dimension
        descReducePar.Init( numKeep, numKeep, scaBlockSize_, scaBlockSize_, I_ZERO, I_ZERO, contxt_ );

        square_mat_scala.SetDescriptor( descReducePar );
        eigvecs_scala.SetDescriptor( descReducePar );


        DblNumMat&  square_mat = XTX;
        // Redistribute the input matrix over the process grid
        SCALAPACK(pdgemr2d)(&numKeep, &numKeep, square_mat.Data(), &I_ONE, &I_ONE, descReduceSeq.Values(), 
            &square_mat_scala.LocalMatrix()[0], &I_ONE, &I_ONE, square_mat_scala.Desc().Values(), &contxt_ );

        // Make the ScaLAPACK call
        char uplo = 'U';
        std::vector<Real> temp_eigs(lda);

        scalapack::Syevd(uplo, square_mat_scala, temp_eigs, eigvecs_scala );

        // Copy the eigenvalues
        for(Int copy_iter = 0; copy_iter < lda; copy_iter ++){
          eigValS[copy_iter] = temp_eigs[copy_iter];
        }

        // Redistribute back eigenvectors
        SetValue(square_mat, 0.0 );
        SCALAPACK(pdgemr2d)( &numKeep, &numKeep, eigvecs_scala.Data(), &I_ONE, &I_ONE, square_mat_scala.Desc().Values(),
            square_mat.Data(), &I_ONE, &I_ONE, descReduceSeq.Values(), &contxt_ );
      }
    }
    else // PWSolver == "PPCGScaLAPACK"
    {
      if ( mpirank == 0 ){
        lapack::Syevd( 'V', 'U', width, XTX.Data(), width, eigValS.Data() );
      }
    }

    MPI_Bcast(XTX.Data(), width*width, MPI_DOUBLE, 0, mpi_comm_);
    MPI_Bcast(eigValS.Data(), width, MPI_DOUBLE, 0, mpi_comm_);

    // X <- X*C
    blas::Gemm( 'N', 'N', heightLocal2, width, width, 1.0, X.Data(),
        heightLocal2, XTX.Data(), width, 0.0, Xtemp.Data(), heightLocal2 );

    lapack::Lacpy( 'A', heightLocal2, width, Xtemp.Data(), heightLocal2,
        X.Data(), heightLocal2 );

    // AX <- AX*C
    blas::Gemm( 'N', 'N', heightLocal2, width, width, 1.0, AX.Data(),
        heightLocal2, XTX.Data(), width, 0.0, Xtemp.Data(), heightLocal2 );

    lapack::Lacpy( 'A', heightLocal2, width, Xtemp.Data(), heightLocal2,
        AX.Data(), heightLocal2 );

    // Compute norms of individual eigenpairs 
    for(Int j=0; j < width; j++){
      for(Int i=0; i < heightLocal2; i++){
        Xtemp(i,j) = AX(i,j) - X(i,j)*eigValS(j);  
      }
    } 

    SetValue( resNormLocal, 0.0 );
    for( Int k = 0; k < width; k++ ){
      resNormLocal(k) = 0.0;
      resNormLocal(k) = Energy(CpxNumVec(heightLocal2, false, Xtemp.VecData(k)));
    }
    
    SetValue( resNorm, 0.0 );
    MPI_Allreduce( resNormLocal.Data(), resNorm.Data(), width, MPI_DOUBLE, 
        MPI_SUM, mpi_comm_ );

    if ( mpirank == 0 ){
      for( Int k = 0; k < width; k++ ){
        resNorm(k) = std::sqrt( resNorm(k) ) / std::max( 1.0, std::abs( eigValS(k) ) );
      }
    }
    MPI_Bcast(resNorm.Data(), width, MPI_DOUBLE, 0, mpi_comm_);

    resMax = *(std::max_element( resNorm.Data(), resNorm.Data() + numEig ) );
    resMin = *(std::min_element( resNorm.Data(), resNorm.Data() + numEig ) );

    // Save the eigenvalues and eigenvectors back to the eigensolver data
    // structure
    blas::Copy( width, eigValS.Data(), 1, &eigVal_[noccTotal*spinswitch], 1 );
    blas::Copy( width, resNorm.Data(), 1, &resVal_[noccTotal*spinswitch], 1 );

    AlltoallBackward( mb, nb, ncom, X, Xcol, mpi_comm_ );

    lapack::Lacpy( 'A', height2, widthLocal, Xcol.Data(), height2, 
        psiTemp.WavefunG().MatData(noccLocal*spinswitch), height2 );

    if( isConverged ){
      statusOFS << std::endl << "After " << iter
        << " iterations, PPCG has converged."  << std::endl
        << "The maximum norm of the residual is "
        << resMax << std::endl << std::endl
        << "The minimum norm of the residual is "
        << resMin << std::endl << std::endl;
    }
    else{
      statusOFS << std::endl << "After " << iter
        << " iterations, PPCG did not converge. " << std::endl
        << "The maximum norm of the residual is "
        << resMax << std::endl << std::endl
        << "The minimum norm of the residual is "
        << resMin << std::endl << std::endl;
    }
  }
  else
  {  
    // S = ( X | W | P ) is a triplet used for PPCG.  
    // W is the preconditioned residual
    DblNumMat  S( heightLocal2, 3*width ), AS( heightLocal2, 3*width ); 

    // Temporary buffer array.
    // The unpreconditioned residual will also be saved in Xtemp
    DblNumMat  XTX( width, width );
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

      // Compute the residual.
      // R <- AX - X*(X'*AX)
      lapack::Lacpy( 'A', heightLocal2, width, AX.Data(), heightLocal2, Xtemp.Data(), heightLocal2 );
      blas::Gemm( 'N', 'N', heightLocal2, width, width, -1.0, 
          X.Data(), heightLocal2, XTX.Data(), width, 1.0, Xtemp.Data(), heightLocal2 );

      // Compute the norm of the residual block
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

      AlltoallBackward( mb, nb, Xtemp, Xcol, mpi_comm_ );

      // Compute W = TW
      {
        Spinor spnTemp(fftPtr_->domain, ncom, noccTotal, false, Xcol.VecData(numLockedLocal));
        NumTns<Real> tnsTemp(ntot, ncom, widthLocal-numLockedLocal, false, Wcol.VecData(numLockedLocal));

        SetValue( tnsTemp, 0.0 );
        spnTemp.AddTeterPrecond( fftPtr_, hamPtr_->Teter(), tnsTemp );
      }

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

      // W = W - X(X'W), AW = AW - AX(X'W)
      blas::Gemm( 'T', 'N', width, width, heightLocal2, 1.0, X.Data(),
          heightLocal2, W.Data(), heightLocal2, 0.0, XTXtemp1.Data(), width );
      SetValue( XTX, 0.0 );
      MPI_Allreduce( XTXtemp1.Data(), XTX.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm_ );

      blas::Gemm( 'N', 'N', heightLocal2, width, width, -1.0, 
          X.Data(), heightLocal2, XTX.Data(), width, 1.0, W.Data(), heightLocal2 );

      blas::Gemm( 'N', 'N', heightLocal2, width, width, -1.0, 
          AX.Data(), heightLocal2, XTX.Data(), width, 1.0, AW.Data(), heightLocal2 );

      // Normalize columns of W
      Real normLocal[width]; 
      Real normGlobal[width];

      for( Int k = numLockedLocal; k < width; k++ ){
        normLocal[k] = Energy(DblNumVec(heightLocal2, false, W.VecData(k)));
        normGlobal[k] = 0.0;
      }
 
      MPI_Allreduce( &normLocal[0], &normGlobal[0], width, MPI_DOUBLE, MPI_SUM, mpi_comm_ );
      for( Int k = numLockedLocal; k < width; k++ ){
        Real norm = std::sqrt( normGlobal[k] );
        blas::Scal( heightLocal2, 1.0 / norm, W.VecData(k), 1 );
        blas::Scal( heightLocal2, 1.0 / norm, AW.VecData(k), 1 );
      }

      // P = P - X(X'P), AP = AP - AX(X'P)
      if( numSet == 3 ){
        blas::Gemm( 'T', 'N', width, width, heightLocal2, 1.0, X.Data(),
            heightLocal2, P.Data(), heightLocal2, 0.0, XTXtemp1.Data(), width );
        SetValue( XTX, 0.0 );
        MPI_Allreduce( XTXtemp1.Data(), XTX.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm_ );

        blas::Gemm( 'N', 'N', heightLocal2, width, width, -1.0, 
            X.Data(), heightLocal2, XTX.Data(), width, 1.0, P.Data(), heightLocal2 );

        blas::Gemm( 'N', 'N', heightLocal2, width, width, -1.0, 
            AX.Data(), heightLocal2, XTX.Data(), width, 1.0, AP.Data(), heightLocal2 );

        // Normalize the conjugate direction
        for( Int k = numLockedLocal; k < width; k++ ){
          normLocal[k] = Energy(DblNumVec(heightLocal2, false, P.VecData(k)));
          normGlobal[k] = 0.0;
        }
        MPI_Allreduce( &normLocal[0], &normGlobal[0], width, MPI_DOUBLE, MPI_SUM, mpi_comm_ );
        for( Int k = numLockedLocal; k < width; k++ ){
          Real norm = std::sqrt( normGlobal[k] );
          blas::Scal( heightLocal2, 1.0 / norm, P.VecData(k), 1 );
          blas::Scal( heightLocal2, 1.0 / norm, AP.VecData(k), 1 );
        }
      }

      // Perform the sweep
      Int sbSize = esdfParam.PPCGsbSize;
      Int nsb = (width + sbSize - 1) / sbSize ; 
      bool isDivid = ( width % sbSize == 0 );
      // Leading dimension of buff matrix
      Int sbSize1 = sbSize;
      Int sbSize2 = sbSize * 2;
      Int sbSize3 = sbSize * 3;

      // AMat and BMat have variant size dependent on sbSize for each small subspace
      DblNumMat AMat, BMat;

      // contains all nsb 3-by-3 matrices
      DblNumMat  AMatAll( sbSize3, sbSize3*nsb ), BMatAll( sbSize3, sbSize3*nsb ); 
      // contains local parts of all nsb 3-by-3 matrices
      DblNumMat  AMatAllLocal( sbSize3, sbSize3*nsb ), BMatAllLocal( sbSize3, sbSize3*nsb );
      SetValue( AMatAll, D_ZERO ); SetValue( BMatAll, D_ZERO );
      SetValue( AMatAllLocal, D_ZERO ); SetValue( BMatAllLocal, D_ZERO );

      // LOCKING NOT SUPPORTED, loop over all columns 
      for( Int k = 0; k < nsb; k++ ){

        if( (k == nsb - 1) && (!isDivid) )
          sbSize = width % sbSize;
        else
          sbSize = esdfParam.PPCGsbSize;

        // fetch indiviual columns
        DblNumMat  x( heightLocal2, sbSize, false, X.VecData(sbSize1*k) );
        DblNumMat  w( heightLocal2, sbSize, false, W.VecData(sbSize1*k) );
        DblNumMat ax( heightLocal2, sbSize, false, AX.VecData(sbSize1*k) );
        DblNumMat aw( heightLocal2, sbSize, false, AW.VecData(sbSize1*k) );

        // Compute AMatAllLoc and BMatAllLoc            
        // AMatAllLoc
        blas::Gemm( 'T', 'N', sbSize, sbSize, heightLocal2, 1.0, x.Data(),
            heightLocal2, ax.Data(), heightLocal2, 
            0.0, &AMatAllLocal(0,sbSize3*k), sbSize3 );

        blas::Gemm( 'T', 'N', sbSize, sbSize, heightLocal2, 1.0, w.Data(),
            heightLocal2, aw.Data(), heightLocal2, 
            0.0, &AMatAllLocal(sbSize1,sbSize3*k+sbSize1), sbSize3 );

        blas::Gemm( 'T', 'N', sbSize, sbSize, heightLocal2, 1.0, x.Data(),
            heightLocal2, aw.Data(), heightLocal2, 
            0.0, &AMatAllLocal(0,sbSize3*k+sbSize1), sbSize3 );

        // BMatAllLoc            
        blas::Gemm( 'T', 'N', sbSize, sbSize, heightLocal2, 1.0, x.Data(),
            heightLocal2, x.Data(), heightLocal2, 
            0.0, &BMatAllLocal(0,sbSize3*k), sbSize3 );

        blas::Gemm( 'T', 'N', sbSize, sbSize, heightLocal2, 1.0, w.Data(),
            heightLocal2, w.Data(), heightLocal2, 
            0.0, &BMatAllLocal(sbSize1,sbSize3*k+sbSize1), sbSize3 );

        blas::Gemm( 'T', 'N', sbSize, sbSize, heightLocal2, 1.0, x.Data(),
            heightLocal2, w.Data(), heightLocal2, 
            0.0, &BMatAllLocal(0,sbSize3*k+sbSize1), sbSize3 );

        if ( numSet == 3 ){

          DblNumMat  p( heightLocal2, sbSize, false, P.VecData(sbSize1*k) );
          DblNumMat ap( heightLocal2, sbSize, false, AP.VecData(sbSize1*k) );

          // AMatAllLoc
          blas::Gemm( 'T', 'N', sbSize, sbSize, heightLocal2, 1.0, p.Data(),
              heightLocal2, ap.Data(), heightLocal2, 
              0.0, &AMatAllLocal(sbSize2,sbSize3*k+sbSize2), sbSize3 );

          blas::Gemm( 'T', 'N', sbSize, sbSize, heightLocal2, 1.0, x.Data(),
              heightLocal2, ap.Data(), heightLocal2, 
              0.0, &AMatAllLocal(0, sbSize3*k+sbSize2), sbSize3 );

          blas::Gemm( 'T', 'N', sbSize, sbSize, heightLocal2, 1.0, w.Data(),
              heightLocal2, ap.Data(), heightLocal2, 
              0.0, &AMatAllLocal(sbSize1, sbSize3*k+sbSize2), sbSize3 );

          // BMatAllLoc
          blas::Gemm( 'T', 'N', sbSize, sbSize, heightLocal2, 1.0, p.Data(),
              heightLocal2, p.Data(), heightLocal2, 
              0.0, &BMatAllLocal(sbSize2,sbSize3*k+sbSize2), sbSize3 );

          blas::Gemm( 'T', 'N', sbSize, sbSize, heightLocal2, 1.0, x.Data(),
              heightLocal2, p.Data(), heightLocal2, 
              0.0, &BMatAllLocal(0, sbSize3*k+sbSize2), sbSize3 );

          blas::Gemm( 'T', 'N', sbSize, sbSize, heightLocal2, 1.0, w.Data(),
              heightLocal2, p.Data(), heightLocal2, 
              0.0, &BMatAllLocal(sbSize1, sbSize3*k+sbSize2), sbSize3 );
        }  // ---- End of if( numSet == 3 ) ----
      }  // for (k)             
      
      MPI_Allreduce( AMatAllLocal.Data(), AMatAll.Data(), sbSize3*sbSize3*nsb, MPI_DOUBLE, MPI_SUM, mpi_comm_ );
    
      MPI_Allreduce( BMatAllLocal.Data(), BMatAll.Data(), sbSize3*sbSize3*nsb, MPI_DOUBLE, MPI_SUM, mpi_comm_ );

      // Solve nsb small eigenproblems and update columns of X 
      for( Int k = 0; k < nsb; k++ ){

        if( (k == nsb - 1) && !isDivid )
          sbSize = width % sbSize;
        else
          sbSize = esdfParam.PPCGsbSize;

        Real eigs[3*sbSize];
        DblNumMat  cx( sbSize, sbSize ), cw( sbSize, sbSize ), cp( sbSize, sbSize);
        DblNumMat tmp( heightLocal2, sbSize );            

        // small eigensolve
        AMat.Resize( sbSize*3, sbSize*3 );
        BMat.Resize( sbSize*3, sbSize*3 );
        if( k < nsb - 1 ){
          lapack::Lacpy( 'A', sbSize3, sbSize3, &AMatAll(0,sbSize3*k), sbSize3, AMat.Data(), sbSize3 );
          lapack::Lacpy( 'A', sbSize3, sbSize3, &BMatAll(0,sbSize3*k), sbSize3, BMat.Data(), sbSize3 );
        }
        else{
          for( Int i = 0; i < 3; i++ ){
            for( Int j = 0; j < 3; j++ ){
              lapack::Lacpy( 'A', sbSize, sbSize, &AMatAll(sbSize1*i,sbSize3*k+sbSize1*j), 
                  sbSize3, &AMat(sbSize*i,sbSize*j), sbSize*3 );
              lapack::Lacpy( 'A', sbSize, sbSize, &BMatAll(sbSize1*i,sbSize3*k+sbSize1*j), 
                  sbSize3, &BMat(sbSize*i,sbSize*j), sbSize*3 );
            }
          }
        }   

        Int dim = (numSet == 3) ? 3*sbSize : 2*sbSize;
        lapack::Sygvd(1, 'V', 'U', dim, AMat.Data(), 3*sbSize, BMat.Data(), 3*sbSize, eigs);

        // fetch indiviual columns
        DblNumMat  x( heightLocal2, sbSize, false, X.VecData(sbSize1*k) );
        DblNumMat  w( heightLocal2, sbSize, false, W.VecData(sbSize1*k) );
        DblNumMat  p( heightLocal2, sbSize, false, P.VecData(sbSize1*k) );
        DblNumMat ax( heightLocal2, sbSize, false, AX.VecData(sbSize1*k) );
        DblNumMat aw( heightLocal2, sbSize, false, AW.VecData(sbSize1*k) );
        DblNumMat ap( heightLocal2, sbSize, false, AP.VecData(sbSize1*k) );

        lapack::Lacpy( 'A', sbSize, sbSize, &AMat(0,0), 3*sbSize, cx.Data(), sbSize );
        lapack::Lacpy( 'A', sbSize, sbSize, &AMat(sbSize,0), 3*sbSize, cw.Data(), sbSize );

        // p = w*cw + p*cp; x = x*cx + p; ap = aw*cw + ap*cp; ax = ax*cx + ap;
        if( numSet == 3 ){
        
          lapack::Lacpy( 'A', sbSize, sbSize, &AMat(2*sbSize,0), 3*sbSize, cp.Data(), sbSize );
       
          // tmp <- p*cp 
          blas::Gemm( 'N', 'N', heightLocal2, sbSize, sbSize, 1.0,
              p.Data(), heightLocal2, cp.Data(), sbSize,
              0.0, tmp.Data(), heightLocal2 );

          // p <- w*cw + tmp
          blas::Gemm( 'N', 'N', heightLocal2, sbSize, sbSize, 1.0,
              w.Data(), heightLocal2, cw.Data(), sbSize,
              1.0, tmp.Data(), heightLocal2 );

          lapack::Lacpy( 'A', heightLocal2, sbSize, tmp.Data(), heightLocal2, p.Data(), heightLocal2 );

          // tmp <- ap*cp 
          blas::Gemm( 'N', 'N', heightLocal2, sbSize, sbSize, 1.0,
              ap.Data(), heightLocal2, cp.Data(), sbSize,
              0.0, tmp.Data(), heightLocal2 );

          // ap <- aw*cw + tmp
          blas::Gemm( 'N', 'N', heightLocal2, sbSize, sbSize, 1.0,
              aw.Data(), heightLocal2, cw.Data(), sbSize,
              1.0, tmp.Data(), heightLocal2 );
          lapack::Lacpy( 'A', heightLocal2, sbSize, tmp.Data(), heightLocal2, ap.Data(), heightLocal2 );
        }else{
          // p <- w*cw
          blas::Gemm( 'N', 'N', heightLocal2, sbSize, sbSize, 1.0,
              w.Data(), heightLocal2, cw.Data(), sbSize,
              0.0, p.Data(), heightLocal2 );

          // ap <- aw*cw
          blas::Gemm( 'N', 'N', heightLocal2, sbSize, sbSize, 1.0,
              aw.Data(), heightLocal2, cw.Data(), sbSize,
              0.0, ap.Data(), heightLocal2 );
        }

        // x <- x*cx + p
        lapack::Lacpy( 'A', heightLocal2, sbSize, p.Data(), heightLocal2, tmp.Data(), heightLocal2 );    
        blas::Gemm( 'N', 'N', heightLocal2, sbSize, sbSize, 1.0,
            x.Data(), heightLocal2, cx.Data(), sbSize,
            1.0, tmp.Data(), heightLocal2 );
        lapack::Lacpy( 'A', heightLocal2, sbSize, tmp.Data(), heightLocal2, x.Data(), heightLocal2 );

        // ax <- ax*cx + ap
        lapack::Lacpy( 'A', heightLocal2, sbSize, ap.Data(), heightLocal2, tmp.Data(), heightLocal2 );    
        blas::Gemm( 'N', 'N', heightLocal2, sbSize, sbSize, 1.0,
            ax.Data(), heightLocal2, cx.Data(), sbSize,
            1.0, tmp.Data(), heightLocal2 );
        lapack::Lacpy( 'A', heightLocal2, sbSize, tmp.Data(), heightLocal2, ax.Data(), heightLocal2 );
      }

      // CholeskyQR of the updated block X, AX is transformed meanwhile
      {
        EigenSolver::Orthogonalize( heightLocal2, width, X, Xtemp, XTX, XTXtemp1 );

        if( use_scala_ ){
          blas::Gemm( 'N', 'N', heightLocal2, width, width, 1.0,
              X.Data(), heightLocal2, XTX.Data(), width,
              0.0, Xtemp.Data(), heightLocal2 );
          lapack::Lacpy( 'A', heightLocal2, width, Xtemp.Data(), heightLocal2, AX.Data(), heightLocal2 );
        }
        else{
          blas::Trsm( 'R', 'U', 'N', 'N', heightLocal2, width, 1.0, XTX.Data(), width,
              AX.Data(), heightLocal2 ); 
        }   
      }
    } while( (iter < eigMaxIter) && (resMax > eigTolerance) );

    // *********************************************************************
    // Post processing
    // *********************************************************************

    // Obtain the eigenvalues and eigenvectors
    // if isConverged == true then XTX should contain the matrix X' * (AX); and X is an
    // orthonormal set
    {
      blas::Gemm( 'T', 'N', width, width, heightLocal2, 1.0, X.Data(),
          heightLocal2, AX.Data(), heightLocal2, 0.0, XTXtemp1.Data(), width );
      SetValue( XTX, 0.0 );
      MPI_Allreduce( XTXtemp1.Data(), XTX.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm_ );
    }

    if( use_scala_ )
    { 
      if( contxt_ >= 0 )
      {
        Int numKeep = width; 
        Int lda = width;

        scalapack::ScaLAPACKMatrix<Real> square_mat_scala;
        scalapack::ScaLAPACKMatrix<Real> eigvecs_scala;

        scalapack::Descriptor descReduceSeq, descReducePar;
        Real timeEigScala_sta, timeEigScala_end;

        // Leading dimension provided
        descReduceSeq.Init( numKeep, numKeep, numKeep, numKeep, I_ZERO, I_ZERO, contxt_, lda );

        // Automatically comptued Leading Dimension
        descReducePar.Init( numKeep, numKeep, scaBlockSize_, scaBlockSize_, I_ZERO, I_ZERO, contxt_ );

        square_mat_scala.SetDescriptor( descReducePar );
        eigvecs_scala.SetDescriptor( descReducePar );


        DblNumMat&  square_mat = XTX;
        // Redistribute the input matrix over the process grid
        SCALAPACK(pdgemr2d)(&numKeep, &numKeep, square_mat.Data(), &I_ONE, &I_ONE, descReduceSeq.Values(), 
            &square_mat_scala.LocalMatrix()[0], &I_ONE, &I_ONE, square_mat_scala.Desc().Values(), &contxt_ );

        // Make the ScaLAPACK call
        char uplo = 'U';
        std::vector<Real> temp_eigs(lda);

        scalapack::Syevd(uplo, square_mat_scala, temp_eigs, eigvecs_scala );

        // Copy the eigenvalues
        for(Int copy_iter = 0; copy_iter < lda; copy_iter ++){
          eigValS[copy_iter] = temp_eigs[copy_iter];
        }

        // Redistribute back eigenvectors
        SetValue(square_mat, 0.0 );
        SCALAPACK(pdgemr2d)( &numKeep, &numKeep, eigvecs_scala.Data(), &I_ONE, &I_ONE, square_mat_scala.Desc().Values(),
            square_mat.Data(), &I_ONE, &I_ONE, descReduceSeq.Values(), &contxt_ );
      }
    }
    else // PWSolver == "PPCGScaLAPACK"
    {
      if ( mpirank == 0 ){
        lapack::Syevd( 'V', 'U', width, XTX.Data(), width, eigValS.Data() );
      }
    }

    MPI_Bcast(XTX.Data(), width*width, MPI_DOUBLE, 0, mpi_comm_);
    MPI_Bcast(eigValS.Data(), width, MPI_DOUBLE, 0, mpi_comm_);

    // X <- X*C
    blas::Gemm( 'N', 'N', heightLocal2, width, width, 1.0, X.Data(),
        heightLocal2, XTX.Data(), width, 0.0, Xtemp.Data(), heightLocal2 );

    lapack::Lacpy( 'A', heightLocal2, width, Xtemp.Data(), heightLocal2,
        X.Data(), heightLocal2 );

    // AX <- AX*C
    blas::Gemm( 'N', 'N', heightLocal2, width, width, 1.0, AX.Data(),
        heightLocal2, XTX.Data(), width, 0.0, Xtemp.Data(), heightLocal2 );

    lapack::Lacpy( 'A', heightLocal2, width, Xtemp.Data(), heightLocal2,
        AX.Data(), heightLocal2 );

    // Compute norms of individual eigenpairs 
    for(Int j=0; j < width; j++){
      for(Int i=0; i < heightLocal2; i++){
        Xtemp(i,j) = AX(i,j) - X(i,j)*eigValS(j);  
      }
    } 

    SetValue( resNormLocal, 0.0 );
    for( Int k = 0; k < width; k++ ){
      resNormLocal(k) = 0.0;
      resNormLocal(k) = Energy(DblNumVec(heightLocal2, false, Xtemp.VecData(k)));
    }
    
    SetValue( resNorm, 0.0 );
    MPI_Allreduce( resNormLocal.Data(), resNorm.Data(), width, MPI_DOUBLE, 
        MPI_SUM, mpi_comm_ );

    if ( mpirank == 0 ){
      for( Int k = 0; k < width; k++ ){
        resNorm(k) = std::sqrt( resNorm(k) ) / std::max( 1.0, std::abs( eigValS(k) ) );
      }
    }
    MPI_Bcast(resNorm.Data(), width, MPI_DOUBLE, 0, mpi_comm_);

    resMax = *(std::max_element( resNorm.Data(), resNorm.Data() + numEig ) );
    resMin = *(std::min_element( resNorm.Data(), resNorm.Data() + numEig ) );

    // Save the eigenvalues and eigenvectors back to the eigensolver data
    // structure
    blas::Copy( width, eigValS.Data(), 1, &eigVal_[noccTotal*spinswitch], 1 );
    blas::Copy( width, resNorm.Data(), 1, &resVal_[noccTotal*spinswitch], 1 );

    AlltoallBackward( mb, nb, X, Xcol, mpi_comm_ );

    lapack::Lacpy( 'A', height2, widthLocal, Xcol.Data(), height2, 
        psiTemp.Wavefun().MatData(noccLocal*spinswitch), height2 );

    if( isConverged ){
      statusOFS << std::endl << "After " << iter
        << " iterations, PPCG has converged."  << std::endl
        << "The maximum norm of the residual is "
        << resMax << std::endl << std::endl
        << "The minimum norm of the residual is "
        << resMin << std::endl << std::endl;
    }
    else{
      statusOFS << std::endl << "After " << iter
        << " iterations, PPCG did not converge. " << std::endl
        << "The maximum norm of the residual is "
        << resMax << std::endl << std::endl
        << "The minimum norm of the residual is "
        << resMin << std::endl << std::endl;
    }
  } //    ---- end of ()
 
  return ;
}      // -----  end of method EigenSolver::PPCGSolveReal -----

#endif

} // namespace pwdft
