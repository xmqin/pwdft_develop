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
/// @file eigensolver.cpp
/// @brief Eigensolver in the global domain or extended element.
/// @date 2014-04-25 First version of parallelized version. This does
/// not scale well.
/// @date 2014-08-07 Intra-element parallelization.  This has much
/// improved scalability.
/// @date 2016-04-04 Adjust some parameters for controlling the number
/// of iterations dynamically.
/// @date 2016-04-07 Add Chebyshev filtering.
/// @date 2023-11-01 Add Davidson. Generate PPCG and LOBPCG to 
/// two-component hamiltonian
#include  "eigensolver.hpp"
#include  "utility.hpp"
#include  "blas.hpp"
#include  "lapack.hpp"
#include  "scalapack.hpp"
#include  "mpi_interf.hpp"

using namespace pwdft::scalapack;
using namespace pwdft::esdf;
using namespace pwdft::DensityComponent;
using namespace pwdft::SpinTwo;

namespace pwdft{

EigenSolver::EigenSolver() {
  // IMPORTANT: 
  // Set contxt_ here. Otherwise if an empty Eigensolver realization
  // is used, there could be error in the exit
  contxt_ = -1;
}

EigenSolver::~EigenSolver() {
  // Finish Cblacs
  if(contxt_ >= 0) {
    Cblacs_gridexit( contxt_ );
  }
}

#ifndef _COMPLEX_
void EigenSolver::Setup(
    Hamiltonian& ham,
    Spinor& psi,
    Fourier& fft ) {

  hamPtr_ = &ham;
  psiPtr_ = &psi;
  fftPtr_ = &fft;

  eigVal_.Resize(psiPtr_->NumStateTotal());  SetValue(eigVal_, 0.0);
  resVal_.Resize(psiPtr_->NumStateTotal());  SetValue(resVal_, 0.0);

  mpi_comm_ = fftPtr_->domain.comm;

  scaBlockSize_      = esdfParam.scaBlockSize;

  numProcScaLAPACK_  = esdfParam.numProcScaLAPACKPW; 

  PWDFT_Cheby_use_scala_ = esdfParam.PWDFT_Cheby_use_scala;

  // Setup BLACS
  if( esdfParam.PWSolver == "LOBPCGScaLAPACK" || esdfParam.PWSolver == "PPCGScaLAPACK" || 
      esdfParam.PWSolver == "DavidsonScaLAPACK" ||
      (esdfParam.PWSolver == "CheFSI" && PWDFT_Cheby_use_scala_) )
  {
    use_scala_ = true;

    for( Int i = IRound(sqrt(double(numProcScaLAPACK_))); 
        i <= numProcScaLAPACK_; i++){
      nprow_ = i; npcol_ = numProcScaLAPACK_ / nprow_;
      if( nprow_ * npcol_ == numProcScaLAPACK_ ) break;
    }

    IntNumVec pmap(numProcScaLAPACK_);
    // Take the first numProcScaLAPACK processors for diagonalization
    for ( Int i = 0; i < numProcScaLAPACK_; i++ ){
      pmap[i] = i;
    }

    Cblacs_get(0, 0, &contxt_);

    Cblacs_gridmap(&contxt_, &pmap[0], nprow_, nprow_, npcol_);
  }
  else{
    use_scala_ = false;
  }

  return;
}         // -----  end of method EigenSolver::Setup ( Real version ) -----

void EigenSolver::SolveGenEig( Int lda, Int numCol, Int width, DblNumMat& AMat, DblNumMat& BMat, 
    DblNumVec& eigValS )
{
  // Get width eigenvalues by diagonalizing the matrix with size of numCol

  Int mpirank;  MPI_Comm_rank(mpi_comm_, &mpirank);

  if( use_scala_ ){
    // Solve the generalized eigenvalue problem using ScaLAPACK
    // NOTE: This uses a simplified implementation with Hegst / Syevd / Trsm. 
    // For ill-conditioned matrices this might be unstable. So BE CAREFUL
    if( contxt_ >= 0 ){
      // Note: No need for symmetrization of A, B matrices due to
      // the usage of symmetric version of the algorithm

      // For stability reason, need to find a well conditioned
      // submatrix of B to solve the generalized eigenvalue problem. 
      // This is done by possibly repeatedly doing potrf until
      // info == 0 (no error)
      bool factorizeB = true;
      Int numKeep = numCol;
      scalapack::ScaLAPACKMatrix<Real> BMatSca;
      scalapack::Descriptor descReduceSeq, descReducePar;

      while( factorizeB ){
        // Redistributed the B matrix

        // Provided LDA
        descReduceSeq.Init( numKeep, numKeep, numKeep, numKeep, I_ZERO, I_ZERO, contxt_, lda );
        // Automatically comptued LDA
        descReducePar.Init( numKeep, numKeep, scaBlockSize_, scaBlockSize_, I_ZERO, I_ZERO, contxt_ );
        BMatSca.SetDescriptor( descReducePar );
        // Redistribute the matrix due to the changed size. 
        SCALAPACK(pdgemr2d)(&numKeep, &numKeep, BMat.Data(), &I_ONE, &I_ONE, descReduceSeq.Values(), 
            &BMatSca.LocalMatrix()[0], &I_ONE, &I_ONE, BMatSca.Desc().Values(), &contxt_ );

        // Factorize
        Int info;
        char uplo = 'U';
        SCALAPACK(pdpotrf)(&uplo, &numKeep, BMatSca.Data(), &I_ONE,
            &I_ONE, BMatSca.Desc().Values(), &info);
        if( info == 0 ){
          factorizeB = false;
        }
        else if( info > width + 1 ){
          // Reduce numKeep and solve again
          // NOTE: (int) is in fact redundant due to integer operation
          numKeep = (int)((info + width)/2);
          // Need to modify the descriptor
          statusOFS << "pdpotrf returns info = " << info << std::endl;
          statusOFS << "retry with size = " << numKeep << std::endl;
        }
        else if (info > 0 && info <=width + 1){
          std::ostringstream msg;
          msg << "pdpotrf: returns info = " << info << std::endl
            << "Not enough columns. The matrix is very ill conditioned." << std::endl;
          ErrorHandling( msg );
        }
        else if( info < 0 ){
          std::ostringstream msg;
          msg << "pdpotrf: runtime error. Info = " << info << std::endl;
          ErrorHandling( msg );
        }
      } // while (factorizeB)

      scalapack::ScaLAPACKMatrix<Real> AMatSca, ZMatSca;
      AMatSca.SetDescriptor( descReducePar );
      ZMatSca.SetDescriptor( descReducePar );

      SCALAPACK(pdgemr2d)(&numKeep, &numKeep, AMat.Data(), &I_ONE, &I_ONE, descReduceSeq.Values(), 
          &AMatSca.LocalMatrix()[0], &I_ONE, &I_ONE, AMatSca.Desc().Values(), &contxt_ );
      // Solve the generalized eigenvalue problem
      std::vector<Real> eigs(lda);
      // Keep track of whether Potrf is stable or not.
      scalapack::Sygst( 1, 'U', AMatSca, BMatSca );

      scalapack::Syevd('U', AMatSca, eigs, ZMatSca );

      scalapack::Trsm('L', 'U', 'N', 'N', 1.0, BMatSca, ZMatSca);

      // Copy the eigenvalues
      for( Int i = 0; i < width; i++ ){
        eigValS[i] = eigs[i];
      }

      // Copy the eigenvectors back to the 0-th processor
      SetValue( AMat, 0.0 );
      SCALAPACK(pdgemr2d)( &numKeep, &numKeep, ZMatSca.Data(), &I_ONE, &I_ONE, ZMatSca.Desc().Values(),
          AMat.Data(), &I_ONE, &I_ONE, descReduceSeq.Values(), &contxt_ );
    } // solve generalized eigenvalue problem
  } // Parallel version
  else{
    if ( mpirank == 0 ) {
      DblNumVec  sigma2(lda);
      DblNumVec  invsigma(lda);
      SetValue( sigma2, 0.0 );
      SetValue( invsigma, 0.0 );

      // Symmetrize A and B first.  This is important.
      for( Int j = 0; j < numCol; j++ ){
        for( Int i = j+1; i < numCol; i++ ){
          AMat(i,j) = AMat(j,i);
          BMat(i,j) = BMat(j,i);
        }
      }

      lapack::Syevd( 'V', 'U', numCol, BMat.Data(), lda, sigma2.Data() );

      Int numKeep = 0;
      for( Int i = numCol-1; i>=0; i-- ){
        if( sigma2(i) / sigma2(numCol-1) >  1e-8 )
          numKeep++;
        else
          break;
      }

      for( Int i = 0; i < numKeep; i++ ){
        invsigma(i) = 1.0 / std::sqrt( sigma2(i+numCol-numKeep) );
      }

      if( numKeep < width ){
        std::ostringstream msg;
        msg 
          << "width   = " << width << std::endl
          << "numKeep =  " << numKeep << std::endl
          << "there are not enough number of columns." << std::endl;
        ErrorHandling( msg.str().c_str() );
      }

      DblNumMat AMatT1( lda, lda );
      SetValue( AMatT1, 0.0 );
      // Evaluate S^{-1/2} (U^T A U) S^{-1/2}
      blas::Gemm( 'N', 'N', numCol, numKeep, numCol, 1.0,
          AMat.Data(), lda, BMat.VecData(numCol-numKeep), lda,
          0.0, AMatT1.Data(), lda );

      blas::Gemm( 'T', 'N', numKeep, numKeep, numCol, 1.0,
          BMat.VecData(numCol-numKeep), lda, AMatT1.Data(), lda, 
          0.0, AMat.Data(), lda );

      for( Int j = 0; j < numKeep; j++ ){
        for( Int i = 0; i < numKeep; i++ ){
          AMat(i,j) *= invsigma(i)*invsigma(j);
        }
      }

      // Solve the standard eigenvalue problem
      DblNumVec eigs( numKeep );
      lapack::Syevd( 'V', 'U', numKeep, AMat.Data(), lda, eigs.Data() );        
      blas::Copy( width, eigs.Data(), 1, eigValS.Data(), 1 );

      // Compute the correct eigenvectors and save them in AMat
      for( Int j = 0; j < numKeep; j++ ){
        for( Int i = 0; i < numKeep; i++ ){
          AMat(i,j) *= invsigma(i);
        }
      }

      blas::Gemm( 'N', 'N', numCol, numKeep, numKeep, 1.0,
          BMat.VecData(numCol-numKeep), lda, AMat.Data(), lda,
          0.0, AMatT1.Data(), lda );

      lapack::Lacpy( 'A', numCol, numKeep, AMatT1.Data(), lda, 
          AMat.Data(), lda );
    } // mpirank == 0
  } // sequential version
}

void EigenSolver::Orthogonalize( Int heightLocal, Int width, 
    DblNumMat& X, DblNumMat& Xtemp, DblNumMat& XTX, 
    DblNumMat& XTXtemp ) 
{
  // Perform orthogonalization for row divided X  
  Int mpirank;  MPI_Comm_rank(mpi_comm_, &mpirank);

  blas::Gemm( 'T', 'N', width, width, heightLocal, 1.0, X.Data(), 
      heightLocal, X.Data(), heightLocal, 0.0, XTXtemp.Data(), width );

  SetValue( XTX, D_ZERO );
  MPI_Allreduce( XTXtemp.Data(), XTX.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm_ );

  if( use_scala_ )
  {
    if( contxt_ >= 0 )
    {
      Int numKeep = width;
      Int lda = width;

      scalapack::ScaLAPACKMatrix<Real> square_mat_scala;

      scalapack::Descriptor descReduceSeq, descReducePar;

      // Leading dimension provided
      descReduceSeq.Init( numKeep, numKeep, numKeep, numKeep, I_ZERO, I_ZERO, contxt_, lda );

      // Automatically comptued Leading Dimension
      descReducePar.Init( numKeep, numKeep, scaBlockSize_, scaBlockSize_, I_ZERO, I_ZERO, contxt_ );

      square_mat_scala.SetDescriptor( descReducePar );

      DblNumMat&  square_mat = XTX;
      // Redistribute the input matrix over the process grid
      SCALAPACK(pdgemr2d)(&numKeep, &numKeep, square_mat.Data(), &I_ONE, &I_ONE, descReduceSeq.Values(),
          &square_mat_scala.LocalMatrix()[0], &I_ONE, &I_ONE, square_mat_scala.Desc().Values(), &contxt_ );

      char uplo = 'U';
      char diag = 'N';

      // Call PZPOTRF to do cholesky decomposition
      scalapack::Potrf(uplo, square_mat_scala );

      // Call PZTRTRI to do matrix inversion
      scalapack::Trtri(uplo, diag, square_mat_scala );

      // Redistribute back decomposed matrix
      SetValue( square_mat, D_ZERO );
      SCALAPACK(pdgemr2d)( &numKeep, &numKeep, square_mat_scala.Data(), &I_ONE, &I_ONE, square_mat_scala.Desc().Values(),
          square_mat.Data(), &I_ONE, &I_ONE, descReduceSeq.Values(), &contxt_ );
    }

    MPI_Bcast(XTX.Data(), width*width, MPI_DOUBLE, 0, mpi_comm_);

    // Set the lower triangular part of XTX to zero
    Int numKeep = width;
    for( Int j = 0; j < numKeep; j++ ){
      for( Int i = j + 1; i < numKeep; i++ ){
        XTX(i, j) = D_ZERO;
      }
    }

    // Calculate X * U^(-1)
    blas::Gemm( 'N', 'N', heightLocal, numKeep, numKeep, 1.0,
      X.Data(), heightLocal, XTX.Data(), numKeep,
      0.0, Xtemp.Data(), heightLocal );

    lapack::Lacpy( 'A', heightLocal, width, Xtemp.Data(), heightLocal, X.Data(), heightLocal );
  }
  else
  {
    if ( mpirank == 0) {
      lapack::Potrf( 'U', width, XTX.Data(), width );
    }

    MPI_Bcast(XTX.Data(), width*width, MPI_DOUBLE, 0, mpi_comm_);

    // X <- X * U^{-1} is orthogonal
    blas::Trsm( 'R', 'U', 'N', 'N', heightLocal, width, 1.0, XTX.Data(), width, 
        X.Data(), heightLocal );
  }  // ---- End of if( _use_scala_ ) ----

  return;
}

void EigenSolver::Orthogonalize( Int heightLocal, Int width, 
    CpxNumMat& X, CpxNumMat& Xtemp, DblNumMat& XTX, 
    DblNumMat& XTXtemp ) 
{
  // Perform orthogonalization for row divided X  
  Int mpirank;  MPI_Comm_rank(mpi_comm_, &mpirank);

  blas::Gemm( 'C', 'N', width, width, heightLocal, 1.0, X.Data(), 
      heightLocal, X.Data(), heightLocal, 0.0, XTXtemp.Data(), width );

  SetValue( XTX, D_ZERO );
  MPI_Allreduce( XTXtemp.Data(), XTX.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm_ );

  if( use_scala_ )
  {
    if( contxt_ >= 0 )
    {
      Int numKeep = width;
      Int lda = width;

      scalapack::ScaLAPACKMatrix<Real> square_mat_scala;

      scalapack::Descriptor descReduceSeq, descReducePar;

      // Leading dimension provided
      descReduceSeq.Init( numKeep, numKeep, numKeep, numKeep, I_ZERO, I_ZERO, contxt_, lda );

      // Automatically comptued Leading Dimension
      descReducePar.Init( numKeep, numKeep, scaBlockSize_, scaBlockSize_, I_ZERO, I_ZERO, contxt_ );

      square_mat_scala.SetDescriptor( descReducePar );

      DblNumMat&  square_mat = XTX;
      // Redistribute the input matrix over the process grid
      SCALAPACK(pdgemr2d)(&numKeep, &numKeep, square_mat.Data(), &I_ONE, &I_ONE, descReduceSeq.Values(),
          &square_mat_scala.LocalMatrix()[0], &I_ONE, &I_ONE, square_mat_scala.Desc().Values(), &contxt_ );

      char uplo = 'U';
      char diag = 'N';

      // Call PZPOTRF to do cholesky decomposition
      scalapack::Potrf(uplo, square_mat_scala );

      // Call PZTRTRI to do matrix inversion
      scalapack::Trtri(uplo, diag, square_mat_scala );

      // Redistribute back decomposed matrix
      SetValue( square_mat, D_ZERO );
      SCALAPACK(pdgemr2d)( &numKeep, &numKeep, square_mat_scala.Data(), &I_ONE, &I_ONE, square_mat_scala.Desc().Values(),
          square_mat.Data(), &I_ONE, &I_ONE, descReduceSeq.Values(), &contxt_ );
    }

    MPI_Bcast(XTX.Data(), width*width, MPI_DOUBLE, 0, mpi_comm_);

    // Set the lower triangular part of XTX to zero
    Int numKeep = width;
    for( Int j = 0; j < numKeep; j++ ){
      for( Int i = j + 1; i < numKeep; i++ ){
        XTX(i, j) = D_ZERO;
      }
    }

    // Calculate X * U^(-1)
    blas::Gemm( 'N', 'N', heightLocal, numKeep, numKeep, 1.0,
      X.Data(), heightLocal, XTX.Data(), numKeep,
      0.0, Xtemp.Data(), heightLocal );

    lapack::Lacpy( 'A', heightLocal, width, Xtemp.Data(), heightLocal, X.Data(), heightLocal );
  }
  else
  {
    if ( mpirank == 0) {
      lapack::Potrf( 'U', width, XTX.Data(), width );
    }

    MPI_Bcast(XTX.Data(), width*width, MPI_DOUBLE, 0, mpi_comm_);

    // X <- X * U^{-1} is orthogonal
    blas::Trsm( 'R', 'U', 'N', 'N', heightLocal, width, 1.0, XTX.Data(), width, 
        X.Data(), heightLocal );
  }  // ---- End of if( _use_scala_ ) ----

  return;
}

void EigenSolver::NonlocalMultX( Int heightLocal, Int width,
    CpxNumMat& X, CpxNumMat& VnlX )   
{
  // Calculate Vnlc * X when reciprocal method is used and X is row partitioned
  CpxNumMat& vnlc = hamPtr_->Vnlc()[0].first;
  DblNumVec& wgt = hamPtr_->Vnlc()[0].second;
  Int nbeta = wgt.m();
  Int nblock_band = 1; // When band number is too large, increase it to reduce memory burden
  Int bandSize = width / nblock_band;

  DblNumMat weightLocal( nbeta, bandSize );
  DblNumMat weight( nbeta, bandSize );
  CpxNumMat Xtemp( heightLocal, bandSize );
  CpxNumMat VnlXtemp( heightLocal, bandSize );

  for( Int ib = 0; ib < nblock_band; ib++ ){
     
    SetValue( weightLocal , 0.0 );
    SetValue( weight , 0.0 );

    for( Int j = 0; j < bandSize; j++ ){
      blas::Copy( heightLocal, &X(0,ib*bandSize+j), 1, Xtemp.VecData(j), 1 );        
    }

    blas::Gemm( 'C', 'N', nbeta, bandSize, heightLocal, 1.0, vnlc.Data(),
        heightLocal, Xtemp.Data(), heightLocal, 0.0, weightLocal.Data(), nbeta );

    MPI_Allreduce( weightLocal.Data(), weight.Data(), nbeta*bandSize, MPI_DOUBLE, MPI_SUM, mpi_comm_ );

    for( Int l = 0; l < bandSize; l++ ){
      for( Int i = 0; i < nbeta; i++ ){
        weight(i,l) *= wgt[i];
      }
    }

    blas::Gemm( 'N', 'N', heightLocal, bandSize, nbeta, 1.0, vnlc.Data(), heightLocal, weight.Data(),
        nbeta, 0.0, VnlXtemp.Data(), heightLocal );

    for( Int j = 0; j < bandSize; j++ ){
      blas::Copy( heightLocal, VnlXtemp.VecData(j), 1, &VnlX(0,ib*bandSize+j), 1 );
    }
  } // for (ib)

  return;
}
#endif

} // namespace pwdft
