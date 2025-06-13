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

#ifdef _COMPLEX_
void EigenSolver::Setup(
    Hamiltonian& ham,
    std::vector<Spinor>& psi,
    Fourier& fft ) {

  hamPtr_ = &ham;
  psiPtr_ = &psi;
  fftPtr_ = &fft;

  Int nkLocal = psi.size();

  eigVal_.resize( nkLocal );  
  resVal_.resize( nkLocal ); 
  for( Int i = 0; i < nkLocal; i++ ){
    eigVal_[i].Resize( psi[i].NumStateTotal() ); SetValue( eigVal_[i], 0.0 );
    resVal_[i].Resize( psi[i].NumStateTotal() ); SetValue( resVal_[i], 0.0 );
  }

  mpi_comm_ = fftPtr_->domain.colComm_kpoint;

  scaBlockSize_ = esdfParam.scaBlockSize;

  numProcScaLAPACK_  = esdfParam.numProcScaLAPACKPW / esdfParam.NumGroupKpoint ; 

  PWDFT_Cheby_use_scala_ = esdfParam.PWDFT_Cheby_use_scala;

  // Setup BLACS
  if( esdfParam.PWSolver == "LOBPCGScaLAPACK" || esdfParam.PWSolver == "PPCGScaLAPACK" || 
      esdfParam.PWSolver == "DavidsonScaLAPACK" ||
      (esdfParam.PWSolver == "CheFSI" && PWDFT_Cheby_use_scala_) )
  {
    use_scala_ = true;

    int rowsize;
    int rowrank;
    MPI_Comm_size( fft.domain.rowComm_kpoint, &rowsize );
    MPI_Comm_rank( fft.domain.rowComm_kpoint, &rowrank );

    for( Int i = IRound(sqrt(double(numProcScaLAPACK_))); 
        i <= numProcScaLAPACK_; i++){
      nprow_ = i; npcol_ = numProcScaLAPACK_ / nprow_;
      if( nprow_ * npcol_ == numProcScaLAPACK_ ) break;
    }

    IntNumVec pmap(numProcScaLAPACK_);
    // Take the first numProcScaLAPACK processors for diagonalization
    for ( Int i = 0; i < numProcScaLAPACK_; i++ ){
      pmap[i] = rowrank + i * rowsize;
    }

    Cblacs_get(0, 0, &contxt_);

    Cblacs_gridmap(&contxt_, &pmap[0], nprow_, nprow_, npcol_);
  }
  else{
    use_scala_ = false;
  }

  return;
}         // -----  end of method EigenSolver::Setup( Complex version ) ----- 

void EigenSolver::SolveGenEig( Int lda, Int numCol, Int width, CpxNumMat& AMat, CpxNumMat& BMat, 
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
      scalapack::ScaLAPACKMatrix<Complex> BMatSca;
      scalapack::Descriptor descReduceSeq, descReducePar;

      while( factorizeB ){
        // Redistributed the B matrix

        // Provided LDA
        descReduceSeq.Init( numKeep, numKeep, numKeep, numKeep, I_ZERO, I_ZERO, contxt_, lda );
        // Automatically comptued LDA
        descReducePar.Init( numKeep, numKeep, scaBlockSize_, scaBlockSize_, I_ZERO, I_ZERO, contxt_ );
        BMatSca.SetDescriptor( descReducePar );
        // Redistribute the matrix due to the changed size. 
        SCALAPACK(pzgemr2d)(&numKeep, &numKeep, BMat.Data(), &I_ONE, &I_ONE, descReduceSeq.Values(), 
            &BMatSca.LocalMatrix()[0], &I_ONE, &I_ONE, BMatSca.Desc().Values(), &contxt_ );

        // Factorize
        Int info;
        char uplo = 'U';
        SCALAPACK(pzpotrf)(&uplo, &numKeep, BMatSca.Data(), &I_ONE,
            &I_ONE, BMatSca.Desc().Values(), &info);
        if( info == 0 ){
          factorizeB = false;
        }
        else if( info > width + 1 ){
          // Reduce numKeep and solve again
          // NOTE: (int) is in fact redundant due to integer operation
          numKeep = (int)((info + width)/2);
          // Need to modify the descriptor
          statusOFS << "pzpotrf returns info = " << info << std::endl;
          statusOFS << "retry with size = " << numKeep << std::endl;
        }
        else if (info > 0 && info <=width + 1){
          std::ostringstream msg;
          msg << "pzpotrf: returns info = " << info << std::endl
            << "Not enough columns. The matrix is very ill conditioned." << std::endl;
          ErrorHandling( msg );
        }
        else if( info < 0 ){
          std::ostringstream msg;
          msg << "pzpotrf: runtime error. Info = " << info << std::endl;
          ErrorHandling( msg );
        }
      } // while (factorizeB)

      scalapack::ScaLAPACKMatrix<Complex> AMatSca, ZMatSca;
      AMatSca.SetDescriptor( descReducePar );
      ZMatSca.SetDescriptor( descReducePar );

      SCALAPACK(pzgemr2d)(&numKeep, &numKeep, AMat.Data(), &I_ONE, &I_ONE, descReduceSeq.Values(), 
          &AMatSca.LocalMatrix()[0], &I_ONE, &I_ONE, AMatSca.Desc().Values(), &contxt_ );
      // Solve the generalized eigenvalue problem
      std::vector<Real> eigs(lda);
      // Keep track of whether Potrf is stable or not.
      scalapack::Hegst( 1, 'U', AMatSca, BMatSca );

      scalapack::Syevd('U', AMatSca, eigs, ZMatSca );

      scalapack::Trsm('L', 'U', 'N', 'N', 1.0, BMatSca, ZMatSca);

      // Copy the eigenvalues
      for( Int i = 0; i < width; i++ ){
        eigValS[i] = eigs[i];
      }

      // Copy the eigenvectors back to the 0-th processor
      SetValue( AMat, Complex(0.0,0.0) );
      SCALAPACK(pzgemr2d)( &numKeep, &numKeep, ZMatSca.Data(), &I_ONE, &I_ONE, ZMatSca.Desc().Values(),
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
      for( Int i = 0; i < numCol; i++ ){
        AMat(i,i) = Complex( AMat(i,i).real(), 0.0 );
        BMat(i,i) = Complex( BMat(i,i).real(), 0.0 );
      }

      for( Int j = 0; j < numCol; j++ ){
        for( Int i = j+1; i < numCol; i++ ){
          AMat(i,j) = std::conj( AMat(j,i) );
          BMat(i,j) = std::conj( BMat(j,i) );
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

      CpxNumMat AMatT1( lda, lda );
      SetValue( AMatT1, Complex(0.0,0.0) );
      // Evaluate S^{-1/2} (U^T A U) S^{-1/2}
      blas::Gemm( 'N', 'N', numCol, numKeep, numCol, 1.0,
          AMat.Data(), lda, BMat.VecData(numCol-numKeep), lda,
          0.0, AMatT1.Data(), lda );

      blas::Gemm( 'C', 'N', numKeep, numKeep, numCol, 1.0,
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
    CpxNumMat& X, CpxNumMat& Xtemp, CpxNumMat& XTX, 
    CpxNumMat& XTXtemp ) 
{
  // Perform orthogonalization for row divided X  
  Int mpirank;  MPI_Comm_rank(mpi_comm_, &mpirank);

  blas::Gemm( 'C', 'N', width, width, heightLocal, 1.0, X.Data(), 
      heightLocal, X.Data(), heightLocal, 0.0, XTXtemp.Data(), width );

  SetValue( XTX, Z_ZERO );
  MPI_Allreduce( XTXtemp.Data(), XTX.Data(), width*width, MPI_DOUBLE_COMPLEX, MPI_SUM, mpi_comm_ );

  if( use_scala_ )
  {
    if( contxt_ >= 0 )
    {
      Int numKeep = width;
      Int lda = width;

      scalapack::ScaLAPACKMatrix<Complex> square_mat_scala;

      scalapack::Descriptor descReduceSeq, descReducePar;

      // Leading dimension provided
      descReduceSeq.Init( numKeep, numKeep, numKeep, numKeep, I_ZERO, I_ZERO, contxt_, lda );

      // Automatically comptued Leading Dimension
      descReducePar.Init( numKeep, numKeep, scaBlockSize_, scaBlockSize_, I_ZERO, I_ZERO, contxt_ );

      square_mat_scala.SetDescriptor( descReducePar );

      CpxNumMat&  square_mat = XTX;
      // Redistribute the input matrix over the process grid
      SCALAPACK(pzgemr2d)(&numKeep, &numKeep, square_mat.Data(), &I_ONE, &I_ONE, descReduceSeq.Values(),
          &square_mat_scala.LocalMatrix()[0], &I_ONE, &I_ONE, square_mat_scala.Desc().Values(), &contxt_ );

      char uplo = 'U';
      char diag = 'N';

      // Call PZPOTRF to do cholesky decomposition
      scalapack::Potrf(uplo, square_mat_scala );

      // Call PZTRTRI to do matrix inversion
      scalapack::Trtri(uplo, diag, square_mat_scala );

      // Redistribute back decomposed matrix
      SetValue( square_mat, Z_ZERO );
      SCALAPACK(pzgemr2d)( &numKeep, &numKeep, square_mat_scala.Data(), &I_ONE, &I_ONE, square_mat_scala.Desc().Values(),
          square_mat.Data(), &I_ONE, &I_ONE, descReduceSeq.Values(), &contxt_ );
    }

    MPI_Bcast(XTX.Data(), width*width, MPI_DOUBLE_COMPLEX, 0, mpi_comm_);

    // Set the lower triangular part of XTX to zero
    Int numKeep = width;
    for( Int j = 0; j < numKeep; j++ ){
      for( Int i = j + 1; i < numKeep; i++ ){
        XTX(i, j) = Z_ZERO;
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

    MPI_Bcast(XTX.Data(), width*width, MPI_DOUBLE_COMPLEX, 0, mpi_comm_);

    // X <- X * U^{-1} is orthogonal
    blas::Trsm( 'R', 'U', 'N', 'N', heightLocal, width, 1.0, XTX.Data(), width, 
        X.Data(), heightLocal );
  }  // ---- End of if( _use_scala_ ) ----

  return;
}

void EigenSolver::NonlocalMultX( Int ikLocal, Int heightLocal, Int width, Int ncom,
    CpxNumMat& X, CpxNumMat& VnlX )   
{
  // Calculate Vnlc * X when reciprocal method is used and X is row partitioned
  bool lspinorb = esdfParam.SpinOrbitCoupling;

  CpxNumMat& vnlc = hamPtr_->Vnlc()[ikLocal].first;

  if( !lspinorb ){
    DblNumVec& wgt = hamPtr_->Vnlc()[ikLocal].second;
    Int nbeta = wgt.m();
    Int nblock_band = 1; // When band number is too large, increase it to reduce memory burden
    Int bandSize = width / nblock_band;

    CpxNumMat weightLocal( nbeta, bandSize );
    CpxNumMat weight( nbeta, bandSize );
    CpxNumMat Xtemp( heightLocal, bandSize );
    CpxNumMat VnlXtemp( heightLocal, bandSize );

    for( Int k = 0; k < ncom; k++ ){
      for( Int ib = 0; ib < nblock_band; ib++ ){
        
        SetValue( weightLocal , Complex(0.0, 0.0) );
        SetValue( weight , Complex(0.0, 0.0) );

        for( Int j = 0; j < bandSize; j++ ){
          blas::Copy( heightLocal, &X(heightLocal*k,ib*bandSize+j), 1, Xtemp.VecData(j), 1 );        
        }

        blas::Gemm( 'C', 'N', nbeta, bandSize, heightLocal, 1.0, vnlc.Data(),
            heightLocal, Xtemp.Data(), heightLocal, 0.0, weightLocal.Data(), nbeta );

        MPI_Allreduce( weightLocal.Data(), weight.Data(), nbeta*bandSize, MPI_DOUBLE_COMPLEX, MPI_SUM, mpi_comm_ );

        for( Int l = 0; l < bandSize; l++ ){
          for( Int i = 0; i < nbeta; i++ ){
            weight(i,l) *= wgt[i];
          }
        }

        blas::Gemm( 'N', 'N', heightLocal, bandSize, nbeta, 1.0, vnlc.Data(), heightLocal, weight.Data(),
            nbeta, 0.0, VnlXtemp.Data(), heightLocal );

        for( Int j = 0; j < bandSize; j++ ){
          blas::Copy( heightLocal, VnlXtemp.VecData(j), 1, &VnlX(heightLocal*k,ib*bandSize+j), 1 );
        }
      } // for (ib)
    } // for (k)
  }
  else{
    CpxNumTns Xtemp( heightLocal, width, ncom );
    CpxNumTns VnlXtemp( heightLocal, width, ncom );
    SetValue( Xtemp, Z_ZERO );
    SetValue( VnlXtemp, Z_ZERO );

    // To ensure memory coherence, put the component index to the end
    for( Int k = 0; k < ncom; k++ ){
      for( Int j = 0; j < width; j++ ){
        blas::Copy( heightLocal, &X(heightLocal*k,j), 1, Xtemp.VecData(j,k), 1 );
      }
    }

    std::vector<CpxNumTns>& coef = hamPtr_->Coef();

    Int idxpp = 0;
    for( Int a = 0; a < coef.size(); a++ ){
      const CpxNumTns& coefMat = coef[a];
      Int nbeta = coefMat.m();

      CpxNumTns weightLocal( nbeta, width, ncom );
      CpxNumTns weight( nbeta, width, ncom );

      SetValue( weightLocal , Complex(0.0, 0.0) );
      SetValue( weight , Complex(0.0, 0.0) );
      
      CpxNumTns coefw( nbeta, width, ncom );

      for( Int k = 0; k < ncom; k++ ){
        blas::Gemm( 'C', 'N', nbeta, width, heightLocal, 1.0, vnlc.VecData(idxpp),
            heightLocal, Xtemp.MatData(k), heightLocal, 0.0, weightLocal.MatData(k), nbeta );
      }

      MPI_Allreduce( weightLocal.Data(), weight.Data(), nbeta*width*ncom, MPI_DOUBLE_COMPLEX, MPI_SUM, mpi_comm_ );

      // Matrix-matrix multiplications between coefMats and weights
      blas::Gemm( 'N', 'N', nbeta, width, nbeta, 1.0, coefMat.MatData(RHO), nbeta,
          weight.MatData(UP), nbeta, 0.0, coefw.MatData(UP), nbeta );
      blas::Gemm( 'N', 'N', nbeta, width, nbeta, 1.0, coefMat.MatData(MAGX), nbeta,
          weight.MatData(DN), nbeta, 1.0, coefw.MatData(UP), nbeta );

      blas::Gemm( 'N', 'N', nbeta, width, nbeta, 1.0, coefMat.MatData(MAGY), nbeta,
          weight.MatData(UP), nbeta, 0.0, coefw.MatData(DN), nbeta );
      blas::Gemm( 'N', 'N', nbeta, width, nbeta, 1.0, coefMat.MatData(MAGZ), nbeta,
          weight.MatData(DN), nbeta, 1.0, coefw.MatData(DN), nbeta );

      for( Int k = 0; k < ncom; k++ ){
        blas::Gemm( 'N', 'N', heightLocal, width, nbeta, 1.0, vnlc.VecData(idxpp),
            heightLocal, coefw.MatData(k), nbeta, 1.0, VnlXtemp.MatData(k), heightLocal );
      }

      idxpp += nbeta;   
    } // for (a)

    for( Int k = 0; k < ncom; k++ ){
      for( Int j = 0; j < width; j++ ){
        blas::Copy( heightLocal, VnlXtemp.VecData(j,k), 1, &VnlX(heightLocal*k,j), 1 );
      }
    }
  } // ---- end of if( !lspinorb ) ----

  return;
}
#endif

} // namespace pwdft
