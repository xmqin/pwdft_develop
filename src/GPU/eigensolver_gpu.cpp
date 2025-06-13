/// @file eigensolver_gpu.cpp
/// @brief GPU-ified functions in class EigenSolver.
/// @date 2024-02-04
#include  "eigensolver.hpp"
#include  "utility.hpp"
#include  "blas.hpp"
#include  "lapack.hpp"
#include  "scalapack.hpp"
#include  "mpi_interf.hpp"

#ifdef GPU
#include  "cublas.hpp"
#include  "cuda_utils.h"
#include  "cu_nummat_impl.hpp"
#endif

using namespace pwdft::scalapack;
using namespace pwdft::esdf;

namespace pwdft{

void
EigenSolver::PPCGSolveRealGPU (
    Int          numEig,
    Int          scfIter,
    Int          eigMaxIter,
    Real         eigMinTolerance,
    Real         eigTolerance )
{
  // *********************************************************************
  // Initialization
  // *********************************************************************
  MPI_Barrier(fftPtr_->domain.comm);

  MPI_Comm mpi_comm = fftPtr_->domain.comm;
  Int mpirank;  MPI_Comm_rank(mpi_comm, &mpirank);
  Int mpisize;  MPI_Comm_size(mpi_comm, &mpisize);

  /* init the CUDA Device */
  cublasStatus_t status;
  cublasSideMode_t right  = CUBLAS_SIDE_RIGHT;
  cublasFillMode_t up     = CUBLAS_FILL_MODE_UPPER;
  cublasDiagType_t nondiag   = CUBLAS_DIAG_NON_UNIT;
  cublasOperation_t cu_transT = CUBLAS_OP_T;
  cublasOperation_t cu_transN = CUBLAS_OP_N;
  cublasOperation_t cu_transC = CUBLAS_OP_C;

  Spinor& psiTemp = *psiPtr_;

  Int ntot = psiTemp.NumGridTotal();
  Int ntotLocal = psiTemp.NumGrid();
  Int ncom = psiTemp.NumComponent();
  Int noccLocal = psiTemp.NumState();
  Int noccTotal = psiTemp.NumStateTotal();
  
  Int nspin = hamPtr_->NumDensityComponent();
  Int spinswitch = hamPtr_->SpinSwitch();
  if( nspin == 2 ){
    noccLocal /= 2;
    noccTotal /= 2;
  }

  Int height = ntot;
  Int width = noccTotal;
  Int lda = 3 * width;

  Int widthLocal = noccLocal;
  Int heightLocal = ntotLocal;

  // The number of unconverged bands
  Int notconv = numEig;
  eigTolerance = std::sqrt( eigTolerance );

  // Arrays for CPU-GPU communication
  DblNumVec sendbuf(height*widthLocal);
  DblNumVec recvbuf(heightLocal*width);
  const IntNumVec& sendcounts = psiTemp.SendCounts();
  const IntNumVec& recvcounts = psiTemp.RecvCounts();
  const IntNumVec& senddispls = psiTemp.SendDispls();
  const IntNumVec& recvdispls = psiTemp.RecvDispls();
  const IntNumMat& sendk      = psiTemp.Sendk();
  const IntNumMat& recvk      = psiTemp.Recvk();

  cuDblNumVec cu_sendbuf(height*widthLocal);
  cuDblNumVec cu_recvbuf(heightLocal*width);
  cuIntNumVec cu_sendcounts(mpisize);
  cuIntNumVec cu_recvcounts(mpisize);
  cuIntNumVec cu_senddispls(mpisize);
  cuIntNumVec cu_recvdispls(mpisize);
  cuIntNumMat cu_sendk( height, widthLocal );
  cuIntNumMat cu_recvk( heightLocal, width );
  
  cu_sendcounts.CopyFrom( sendcounts );
  cu_recvcounts.CopyFrom( recvcounts );
  cu_senddispls.CopyFrom( senddispls );
  cu_recvdispls.CopyFrom( recvdispls ); 
  cu_sendk.CopyFrom(sendk);
  cu_recvk.CopyFrom(recvk);

  // S = ( X | W | P ) is a triplet used for LOBPCG.
  // W is the preconditioned residual
  DblNumMat       S( heightLocal, 3*width ),    AS( heightLocal, 3*width );
  cuDblNumMat  cu_S( heightLocal, 3*width ), cu_AS( heightLocal, 3*width );

  // Temporary buffer array.
  // The unpreconditioned residual will also be saved in Xtemp
  DblNumMat  XTX( width, width );
  DblNumMat  XTXtemp1( width, width );
  DblNumMat  Xtemp( heightLocal, width );

  Real  resBlockNormLocal, resBlockNorm; // Frobenius norm of the residual block
  Real  resMax, resMin;
  DblNumVec resNormLocal( width );
  DblNumVec resNorm( width );
  cuDblNumVec cu_resNormLocal ( width );
  
  // For convenience
  DblNumMat  X( heightLocal, width, false, S.VecData(0) );
  DblNumMat  W( heightLocal, width, false, S.VecData(width) );
  DblNumMat  P( heightLocal, width, false, S.VecData(2*width) );
  DblNumMat AX( heightLocal, width, false, AS.VecData(0) );
  DblNumMat AW( heightLocal, width, false, AS.VecData(width) );
  DblNumMat AP( heightLocal, width, false, AS.VecData(2*width) );

  DblNumMat  Xcol( height, widthLocal );
  DblNumMat  Wcol( height, widthLocal );
  DblNumMat AXcol( height, widthLocal );
  DblNumMat AWcol( height, widthLocal );

  // for GPU. please note we need to use copyTo and copyFrom in the GPU matrix
  cuDblNumMat cu_XTX(width, width);
  cuDblNumMat cu_XTXtemp1(width, width);
  cuDblNumMat cu_Xtemp(heightLocal, width);

  cuDblNumMat cu_X ( heightLocal, width, false, cu_S.VecData(0)        );
  cuDblNumMat cu_W ( heightLocal, width, false, cu_S.VecData(width)    );
  cuDblNumMat cu_P ( heightLocal, width, false, cu_S.VecData(2*width)  );
  cuDblNumMat cu_AX( heightLocal, width, false, cu_AS.VecData(0)       );
  cuDblNumMat cu_AW( heightLocal, width, false, cu_AS.VecData(width)   );
  cuDblNumMat cu_AP( heightLocal, width, false, cu_AS.VecData(2*width) );

  cuDblNumMat cu_Xcol ( height, widthLocal );
  cuDblNumMat cu_Wcol ( height, widthLocal );
  cuDblNumMat cu_AXcol( height, widthLocal );
  cuDblNumMat cu_AWcol( height, widthLocal );

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
  cuda_memcpy_CPU2GPU(cu_Xcol.Data(), psiTemp.Wavefun().MatData(spinswitch*noccLocal),
        sizeof(Real)*height*widthLocal);

  cuda_mapping_to_buf( cu_sendbuf.Data(), cu_Xcol.Data(), cu_sendk.Data(), height*widthLocal);
#ifdef GPUDIRECT
    MPI_Alltoallv( &cu_sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE,
        &cu_recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE, mpi_comm );
#else
    cuda_memcpy_GPU2CPU( sendbuf.Data(), cu_sendbuf.Data(), sizeof(Real)*height*widthLocal);
    MPI_Alltoallv( &sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE,
        &recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE, mpi_comm );
    cuda_memcpy_CPU2GPU(cu_recvbuf.Data(), recvbuf.Data(), sizeof(Real)*heightLocal*width);
#endif
    cuda_mapping_from_buf(cu_X.Data(), cu_recvbuf.Data(), cu_recvk.Data(), heightLocal*width);
 
  Real one = 1.0;
  Real minus_one = -1.0;
  Real zero = 0.0;
  
  // *********************************************************************
  // Main loop
  // *********************************************************************
  if( scfIter == 1 ){

    cublas::Gemm( cu_transT, cu_transN, width, width, heightLocal, &one, cu_X.Data(),
        heightLocal, cu_X.Data(), heightLocal, &zero, cu_XTXtemp1.Data(), width );

    cu_XTXtemp1.CopyTo(XTXtemp1);

    MPI_Allreduce( XTXtemp1.Data(), XTX.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );

    if ( mpirank == 0) {
      lapack::Potrf( 'U', width, XTX.Data(), width );
    }
    MPI_Bcast(XTX.Data(), width*width, MPI_DOUBLE, 0, mpi_comm);

    // X <- X * U^{-1} is orthogonal
    cu_XTX.CopyFrom( XTX );
    cublas::Trsm( right, up, cu_transN, nondiag, heightLocal, width, &one, 
        cu_XTX.Data(), width, cu_X.Data(), heightLocal );
    cu_XTX.CopyTo( XTX );

    cuda_mapping_to_buf( cu_recvbuf.Data(), cu_X.Data(), cu_recvk.Data(), heightLocal*width);
#ifdef GPUDIRECT
    MPI_Alltoallv( &cu_recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE,
        &cu_sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE, mpi_comm );
#else
    cuda_memcpy_GPU2CPU( recvbuf.Data(), cu_recvbuf.Data(), sizeof(Real)*heightLocal*width);
    MPI_Alltoallv( &recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE,
        &sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE, mpi_comm );
    cuda_memcpy_CPU2GPU(cu_sendbuf.Data(), sendbuf.Data(), sizeof(Real)*height*widthLocal);
#endif
    cuda_mapping_from_buf(cu_Xcol.Data(), cu_sendbuf.Data(), cu_sendk.Data(), height*widthLocal);
  } // ---- if( scfIter == 1 ) ----

  // Applying the Hamiltonian matrix
  {
    Spinor spnTemp(fftPtr_->domain, ncom, noccTotal, false, cu_Xcol.Data(), true);
    cuNumTns<Real> tnsTemp(ntot, ncom, noccLocal, false, cu_AXcol.Data());

    hamPtr_->MultSpinorGPU( spnTemp, tnsTemp, *fftPtr_ );
  }

  cuda_mapping_to_buf( cu_sendbuf.Data(), cu_AXcol.Data(), cu_sendk.Data(), height*widthLocal);
#ifdef GPUDIRECT
  MPI_Alltoallv( &cu_sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE,
      &cu_recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE, mpi_comm );
#else
  cuda_memcpy_GPU2CPU( sendbuf.Data(), cu_sendbuf.Data(), sizeof(Real)*height*widthLocal);
  MPI_Alltoallv( &sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE,
      &recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE, mpi_comm );
  cuda_memcpy_CPU2GPU(cu_recvbuf.Data(), recvbuf.Data(), sizeof(Real)*heightLocal*width);
#endif
  cuda_mapping_from_buf(cu_AX.Data(), cu_recvbuf.Data(), cu_recvk.Data(), heightLocal*width);

  // Start the main loop
  Int iter = 0;
  do{
    iter++;

    if( iter == 1 || isRestart == true )
      numSet = 2;
    else
      numSet = 3;

    cublas::Gemm( cu_transT, cu_transN, width, width, heightLocal, &one, cu_X.Data(),
        heightLocal, cu_AX.Data(), heightLocal, &zero, cu_XTXtemp1.Data(), width );

    cu_XTXtemp1.CopyTo(XTXtemp1);
    MPI_Allreduce( XTXtemp1.Data(), XTX.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
    
    // Compute the residual.
    // R <- AX - X*(X'*AX)
    cu_Xtemp.CopyFrom ( cu_AX );
    cu_XTX.CopyFrom(XTX);
    cublas::Gemm( cu_transN, cu_transN, heightLocal, width, width, &minus_one, cu_X.Data(),
        heightLocal, cu_XTX.Data(), width, &one, cu_Xtemp.Data(), heightLocal );
   
    // Compute the Frobenius norm of the residual block
    SetValue( resNormLocal, 0.0 );
    cuda_calculate_Energy( cu_Xtemp.Data(), cu_resNormLocal.Data(), width, heightLocal );
    cu_resNormLocal.CopyTo(resNormLocal);

    MPI_Allreduce( resNormLocal.Data(), resNorm.Data(), width, MPI_DOUBLE,
        MPI_SUM, mpi_comm );
 
    if ( mpirank == 0 ){
      for( Int k = 0; k < width; k++ ){
        resNorm(k) = std::sqrt( resNorm(k) ) / std::max( 1.0, std::abs( XTX(k,k) ) );
      }
    }

    MPI_Bcast(resNorm.Data(), width, MPI_DOUBLE, 0, mpi_comm);

    resMax = *(std::max_element( resNorm.Data(), resNorm.Data() + numEig ) );
    resMin = *(std::min_element( resNorm.Data(), resNorm.Data() + numEig ) );

    notconv = 0;
    for( Int i = 0; i < numEig; i++ ){
      if( resNorm[i] > eigTolerance ){
        notconv ++;
      }
    }

    // Convert from row format to column format.
    // MPI_Alltoallv
    // Only convert Xtemp here
    cuda_mapping_to_buf( cu_recvbuf.Data(), cu_Xtemp.Data(), cu_recvk.Data(), heightLocal*width);
#ifdef GPUDIRECT
    MPI_Alltoallv( &cu_recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE,
        &cu_sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE, mpi_comm );
#else
    cuda_memcpy_GPU2CPU( recvbuf.Data(), cu_recvbuf.Data(), sizeof(Real)*heightLocal*width);
    MPI_Alltoallv( &recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE,
        &sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE, mpi_comm );
    cuda_memcpy_CPU2GPU(cu_sendbuf.Data(), sendbuf.Data(), sizeof(Real)*height*widthLocal);
#endif
    cuda_mapping_from_buf(cu_Xcol.Data(), cu_sendbuf.Data(), cu_sendk.Data(), height*widthLocal);

    // Compute W = TW
    {
      Spinor spnTemp(fftPtr_->domain, ncom, noccTotal, false, cu_Xcol.Data(),true);
      cuNumTns<Real> tnsTemp(ntot, ncom, widthLocal-numLockedLocal, false, cu_Wcol.Data());

      spnTemp.AddTeterPrecondGPU( fftPtr_, hamPtr_->Teter(), tnsTemp );
    }

    // Compute AW = A*W
    {
      Spinor spnTemp(fftPtr_->domain, ncom, noccTotal, false, cu_Wcol.Data(), true);
      cuNumTns<Real> tnsTemp(ntot, ncom, widthLocal-numLockedLocal, false, cu_AWcol.Data());

      hamPtr_->MultSpinorGPU( spnTemp, tnsTemp, *fftPtr_ );
    }

    // Convert from column format to row format
    // MPI_Alltoallv
    // Only convert W and AW
    cuda_mapping_to_buf( cu_sendbuf.Data(), cu_Wcol.Data(), cu_sendk.Data(), height*widthLocal);
#ifdef GPUDIRECT
    MPI_Alltoallv( &cu_sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE,
        &cu_recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE, mpi_comm );
#else
    cuda_memcpy_GPU2CPU( sendbuf.Data(), cu_sendbuf.Data(), sizeof(Real)*height*widthLocal);
    MPI_Alltoallv( &sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE,
        &recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE, mpi_comm );
    cuda_memcpy_CPU2GPU(cu_recvbuf.Data(), recvbuf.Data(), sizeof(Real)*heightLocal*width);
#endif
    cuda_mapping_from_buf(cu_W.Data(), cu_recvbuf.Data(), cu_recvk.Data(), heightLocal*width);

    cuda_mapping_to_buf( cu_sendbuf.Data(), cu_AWcol.Data(), cu_sendk.Data(), height*widthLocal);
#ifdef GPUDIRECT
    MPI_Alltoallv( &cu_sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE,
        &cu_recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE, mpi_comm );
#else
    cuda_memcpy_GPU2CPU( sendbuf.Data(), cu_sendbuf.Data(), sizeof(Real)*height*widthLocal);
    MPI_Alltoallv( &sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE,
        &recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE, mpi_comm );
    cuda_memcpy_CPU2GPU(cu_recvbuf.Data(), recvbuf.Data(), sizeof(Real)*heightLocal*width);
#endif
    cuda_mapping_from_buf(cu_AW.Data(), cu_recvbuf.Data(), cu_recvk.Data(), heightLocal*width);

    // W = W - X(X'W), AW = AW - AX(X'W)
    cublas::Gemm( cu_transT, cu_transN, width, width, heightLocal, &one, cu_X.Data(),
        heightLocal, cu_W.Data(), heightLocal, &zero, cu_XTXtemp1.Data(), width );

    cu_XTXtemp1.CopyTo(XTXtemp1);
    MPI_Allreduce( XTXtemp1.Data(), XTX.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
    cu_XTX.CopyFrom(XTX);

    cublas::Gemm( cu_transN, cu_transN, heightLocal, width, width, &minus_one, cu_X.Data(),
        heightLocal, cu_XTX.Data(), width, &one, cu_W.Data(), heightLocal );
    cublas::Gemm( cu_transN, cu_transN, heightLocal, width, width, &minus_one,
        cu_AX.Data(), heightLocal, cu_XTX.Data(), width, &one, cu_AW.Data(), heightLocal );

    // Normalize columns of W
    Real normLocal[width];
    Real normGlobal[width];
    cuDblNumVec cu_normLocal(width);

    cuda_calculate_Energy( cu_W.Data(), cu_normLocal.Data(), width-numLockedLocal, heightLocal );
    cuda_memcpy_GPU2CPU( normLocal, cu_normLocal.Data(), sizeof(Real)*width);
    MPI_Allreduce( &normLocal[0], &normGlobal[0], width, MPI_DOUBLE, MPI_SUM, mpi_comm );

    cuda_memcpy_CPU2GPU(cu_normLocal.Data(), normGlobal, sizeof(Real)*width);
    cuda_batch_Scal( cu_W.Data(),  cu_normLocal.Data(), width, heightLocal);
    cuda_batch_Scal( cu_AW.Data(), cu_normLocal.Data(), width, heightLocal);

    // P = P - X(X'P), AP = AP - AX(X'P)
    if( numSet == 3 ){
      cublas::Gemm( cu_transT, cu_transN, width, width, heightLocal, &one, cu_X.Data(),
          heightLocal, cu_P.Data(), heightLocal, &zero, cu_XTXtemp1.Data(), width );

      cu_XTXtemp1.CopyTo(XTXtemp1);
      MPI_Allreduce( XTXtemp1.Data(), XTX.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
      cu_XTX.CopyFrom( XTX );

      cublas::Gemm( cu_transN, cu_transN, heightLocal, width, width, &minus_one,
          cu_X.Data(), heightLocal, cu_XTX.Data(), width, &one, cu_P.Data(), heightLocal );
      cublas::Gemm( cu_transN, cu_transN, heightLocal, width, width, &minus_one,
          cu_AX.Data(), heightLocal, cu_XTX.Data(), width, &one, cu_AP.Data(), heightLocal );

      // Normalize the conjugate direction
      cuda_calculate_Energy( cu_P.Data(), cu_normLocal.Data(), width-numLockedLocal, heightLocal );
      cuda_memcpy_GPU2CPU( normLocal, cu_normLocal.Data(), sizeof(Real)*width);
      MPI_Allreduce( &normLocal[0], &normGlobal[0], width, MPI_DOUBLE, MPI_SUM, mpi_comm );

      cuda_memcpy_CPU2GPU(cu_normLocal.Data(), normGlobal, sizeof(Real)*width);
      cuda_batch_Scal( cu_P.Data(),  cu_normLocal.Data(), width, heightLocal);
      cuda_batch_Scal( cu_AP.Data(), cu_normLocal.Data(), width, heightLocal);
    }

    // Perform the sweep
    Int sbSize = esdfParam.PPCGsbSize;
    Int nsb = (width + sbSize - 1) / sbSize ;
    bool isDivid = ( width % sbSize == 0 );
    // Leading dimension of buff matrix
    Int sbSize1 = sbSize;
    Int sbSize2 = sbSize * 2;
    Int sbSize3 = sbSize * 3;
    
    DblNumMat AMat, BMat;
    DblNumMat  AMatAll( sbSize3, sbSize3*nsb ), BMatAll( sbSize3, sbSize3*nsb );
    DblNumMat  AMatAllLocal( sbSize3, sbSize3*nsb ), BMatAllLocal( sbSize3, sbSize3*nsb );
    // GPU
    cuDblNumMat cu_AMatAllLocal( sbSize3, sbSize3*nsb );
    cuDblNumMat cu_BMatAllLocal( sbSize3, sbSize3*nsb );

    // LOCKING NOT SUPPORTED, loop over all columns
    for( Int k = 0; k < nsb; k++ ){

      if( (k == nsb - 1) && (!isDivid) )
        sbSize = width % sbSize;
      else
        sbSize = esdfParam.PPCGsbSize;

      // fetch indiviual columns
      DblNumMat  x( heightLocal, sbSize, false, X.VecData(sbSize1*k) );
      DblNumMat  w( heightLocal, sbSize, false, W.VecData(sbSize1*k) );
      DblNumMat ax( heightLocal, sbSize, false, AX.VecData(sbSize1*k) );
      DblNumMat aw( heightLocal, sbSize, false, AW.VecData(sbSize1*k) );

      // gpu data structure.
      cuDblNumMat cu_ax( heightLocal, sbSize, false, cu_AX.VecData(sbSize1*k)  );
      cuDblNumMat cu_x ( heightLocal, sbSize, false, cu_X.VecData(sbSize1*k)  );
      cuDblNumMat cu_w ( heightLocal, sbSize, false, cu_W.VecData(sbSize1*k) );
      cuDblNumMat cu_aw( heightLocal, sbSize, false, cu_AW.VecData(sbSize1*k) );

      // Compute AMatAllLoc and BMatAllLoc
      // AMatAllLoc
      cublas::Gemm( cu_transT, cu_transN, sbSize, sbSize, heightLocal, &one, cu_x.Data(),
          heightLocal, cu_ax.Data(), heightLocal, &zero, &cu_AMatAllLocal(0,sbSize3*k), sbSize3 );
      cublas::Gemm( cu_transT, cu_transN, sbSize, sbSize, heightLocal, &one, cu_w.Data(),
          heightLocal, cu_aw.Data(), heightLocal, &zero, &cu_AMatAllLocal(sbSize1,sbSize3*k+sbSize1), sbSize3 );
      cublas::Gemm( cu_transT, cu_transN, sbSize, sbSize, heightLocal, &one, cu_x.Data(),
          heightLocal, cu_aw.Data(), heightLocal, &zero, &cu_AMatAllLocal(0,sbSize3*k+sbSize1), sbSize3 );

      // BMatAllLoc
      cublas::Gemm( cu_transT, cu_transN, sbSize, sbSize, heightLocal, &one, cu_x.Data(),
          heightLocal, cu_x.Data(), heightLocal, &zero, &cu_BMatAllLocal(0,sbSize3*k), sbSize3 );
      cublas::Gemm( cu_transT, cu_transN, sbSize, sbSize, heightLocal, &one, cu_w.Data(),
          heightLocal, cu_w.Data(), heightLocal, &zero, &cu_BMatAllLocal(sbSize1,sbSize3*k+sbSize1), sbSize3 );
      cublas::Gemm( cu_transT, cu_transN, sbSize, sbSize, heightLocal, &one, cu_x.Data(),
          heightLocal, cu_w.Data(), heightLocal, &zero, &cu_BMatAllLocal(0,sbSize3*k+sbSize1), sbSize3 );

      if ( numSet == 3 ){

        DblNumMat  p( heightLocal, sbSize, false, P.VecData(sbSize1*k) );
        DblNumMat ap( heightLocal, sbSize, false, AP.VecData(sbSize1*k) );

        cuDblNumMat  cu_p (heightLocal, sbSize, false, cu_P.VecData(sbSize1*k)  );
        cuDblNumMat cu_ap (heightLocal, sbSize, false, cu_AP.VecData(sbSize1*k) );

        cublas::Gemm( cu_transT, cu_transN, sbSize, sbSize, heightLocal, &one, cu_p.Data(),
            heightLocal, cu_ap.Data(), heightLocal, &zero, &cu_AMatAllLocal(sbSize2,sbSize3*k+sbSize2), sbSize3);
        cublas::Gemm( cu_transT, cu_transN, sbSize, sbSize, heightLocal, &one, cu_x.Data(),
            heightLocal, cu_ap.Data(), heightLocal, &zero, &cu_AMatAllLocal(0,sbSize3*k+sbSize2), sbSize3 );
        cublas::Gemm( cu_transT, cu_transN, sbSize, sbSize, heightLocal, &one, cu_w.Data(),
            heightLocal, cu_ap.Data(), heightLocal, &zero, &cu_AMatAllLocal(sbSize1,sbSize3*k+sbSize2), sbSize3 );

        cublas::Gemm( cu_transT, cu_transN, sbSize, sbSize, heightLocal, &one, cu_p.Data(),
            heightLocal, cu_p.Data(), heightLocal, &zero, &cu_BMatAllLocal(sbSize2,sbSize3*k+sbSize2), sbSize3 );
        cublas::Gemm( cu_transT, cu_transN, sbSize, sbSize, heightLocal, &one, cu_x.Data(),
            heightLocal, cu_p.Data(), heightLocal, &zero, &cu_BMatAllLocal(0,sbSize3*k+sbSize2), sbSize3 );
        cublas::Gemm( cu_transT, cu_transN, sbSize, sbSize, heightLocal, &one, cu_w.Data(),
            heightLocal, cu_p.Data(), heightLocal, &zero, &cu_BMatAllLocal(sbSize1,sbSize3*k+sbSize2), sbSize3 );
      }
    } // for (k)

    cu_AMatAllLocal.CopyTo( AMatAllLocal );
    cu_BMatAllLocal.CopyTo( BMatAllLocal );

    MPI_Allreduce( AMatAllLocal.Data(), AMatAll.Data(), sbSize3*sbSize3*nsb, MPI_DOUBLE, MPI_SUM, mpi_comm );
    MPI_Allreduce( BMatAllLocal.Data(), BMatAll.Data(), sbSize3*sbSize3*nsb, MPI_DOUBLE, MPI_SUM, mpi_comm );

    // Solve nsb small eigenproblems and update columns of X
    for( Int k = 0; k < nsb; k++ ){
      
      if( (k == nsb - 1) && !isDivid )
        sbSize = width % sbSize;
      else
        sbSize = esdfParam.PPCGsbSize;

      Real eigs[3*sbSize];
      DblNumMat  cx( sbSize, sbSize ), cw( sbSize, sbSize ), cp( sbSize, sbSize);
      DblNumMat tmp( heightLocal, sbSize );

      // GPU      
      cuDblNumMat  cu_cx( sbSize, sbSize ), cu_cw( sbSize, sbSize ), cu_cp( sbSize, sbSize);
      cuDblNumMat cu_tmp( heightLocal, sbSize );

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
      DblNumMat  x( heightLocal, sbSize, false, X.VecData(sbSize1*k) );
      DblNumMat  w( heightLocal, sbSize, false, W.VecData(sbSize1*k) );
      DblNumMat  p( heightLocal, sbSize, false, P.VecData(sbSize1*k) );
      DblNumMat ax( heightLocal, sbSize, false, AX.VecData(sbSize1*k) );
      DblNumMat aw( heightLocal, sbSize, false, AW.VecData(sbSize1*k) );
      DblNumMat ap( heightLocal, sbSize, false, AP.VecData(sbSize1*k) );

      cuDblNumMat  cu_x( heightLocal, sbSize, false, cu_X.VecData(sbSize1*k) );
      cuDblNumMat  cu_w( heightLocal, sbSize, false, cu_W.VecData(sbSize1*k) );
      cuDblNumMat  cu_p( heightLocal, sbSize, false, cu_P.VecData(sbSize1*k) );
      cuDblNumMat cu_ax( heightLocal, sbSize, false, cu_AX.VecData(sbSize1*k) );
      cuDblNumMat cu_aw( heightLocal, sbSize, false, cu_AW.VecData(sbSize1*k) );
      cuDblNumMat cu_ap( heightLocal, sbSize, false, cu_AP.VecData(sbSize1*k) );

      lapack::Lacpy( 'A', sbSize, sbSize, &AMat(0,0), 3*sbSize, cx.Data(), sbSize );
      lapack::Lacpy( 'A', sbSize, sbSize, &AMat(sbSize,0), 3*sbSize, cw.Data(), sbSize );

      cuda_memcpy_CPU2GPU( cu_cx.Data(), cx.Data(), sbSize*sbSize*sizeof(Real));
      cuda_memcpy_CPU2GPU( cu_cw.Data(), cw.Data(), sbSize*sbSize*sizeof(Real));

      // p = w*cw + p*cp; x = x*cx + p; ap = aw*cw + ap*cp; ax = ax*cx + ap;
      if( numSet == 3 ){

        lapack::Lacpy( 'A', sbSize, sbSize, &AMat(2*sbSize,0), 3*sbSize, cp.Data(), sbSize );

        cuda_memcpy_CPU2GPU( cu_cp.Data(), cp.Data(), sbSize*sbSize*sizeof(Real) );

        // tmp <- p*cp 
        cublas::Gemm( cu_transN, cu_transN, heightLocal, sbSize, sbSize, &one,
            cu_p.Data(), heightLocal, cu_cp.Data(), sbSize, &zero, cu_tmp.Data(),heightLocal );
       
        // p <- w*cw + tmp
        cublas::Gemm( cu_transN, cu_transN, heightLocal, sbSize, sbSize, &one,
            cu_w.Data(), heightLocal, cu_cw.Data(), sbSize, &one, cu_tmp.Data(),heightLocal );

        cuda_memcpy_GPU2GPU( cu_p.Data(), cu_tmp.Data(), heightLocal*sbSize*sizeof(Real) );

        // tmp <- ap*cp
        cublas::Gemm( cu_transN, cu_transN, heightLocal, sbSize, sbSize, &one,
            cu_ap.Data(), heightLocal, cu_cp.Data(), sbSize, &zero, cu_tmp.Data(),heightLocal);

        // ap <- aw*cw + tmp
        cublas::Gemm( cu_transN, cu_transN, heightLocal, sbSize, sbSize, &one,
            cu_aw.Data(), heightLocal, cu_cw.Data(), sbSize, &one, cu_tmp.Data(),heightLocal);

        cuda_memcpy_GPU2GPU( cu_ap.Data(), cu_tmp.Data(), heightLocal*sbSize*sizeof(Real));
      }
      else{ 
        // p <- w*cw
        cublas::Gemm( cu_transN, cu_transN, heightLocal, sbSize, sbSize, &one,
            cu_w.Data(), heightLocal, cu_cw.Data(), sbSize, &zero, cu_p.Data(),heightLocal);

        // ap <- aw*cw
        cublas::Gemm( cu_transN, cu_transN, heightLocal, sbSize, sbSize, &one,
            cu_aw.Data(), heightLocal, cu_cw.Data(), sbSize, &zero, cu_ap.Data(),heightLocal);
      }

      // x <- x*cx + p
      cuda_memcpy_GPU2GPU( cu_tmp.Data(), cu_p.Data(), heightLocal*sbSize*sizeof(Real));
      cublas::Gemm( cu_transN, cu_transN, heightLocal, sbSize, sbSize, &one,
          cu_x.Data(), heightLocal, cu_cx.Data(), sbSize, &one, cu_tmp.Data(),heightLocal);
        
      cuda_memcpy_GPU2GPU( cu_x.Data(), cu_tmp.Data(), heightLocal*sbSize*sizeof(Real));

      // ax <- ax*cx + ap
      cuda_memcpy_GPU2GPU( cu_tmp.Data(), cu_ap.Data(), heightLocal*sbSize*sizeof(Real));
      cublas::Gemm( cu_transN, cu_transN, heightLocal, sbSize, sbSize, &one,
          cu_ax.Data(), heightLocal, cu_cx.Data(), sbSize, &one, cu_tmp.Data(),heightLocal);
      cuda_memcpy_GPU2GPU( cu_ax.Data(), cu_tmp.Data(), heightLocal*sbSize*sizeof(Real));
    } // for (k)

    // CholeskyQR of the updated block X
    cublas::Gemm( cu_transT, cu_transN, width, width, heightLocal, &one, cu_X.Data(),
        heightLocal, cu_X.Data(), heightLocal, &zero, cu_XTXtemp1.Data(), width );
    cu_XTXtemp1.CopyTo(XTXtemp1);
    MPI_Allreduce( XTXtemp1.Data(), XTX.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );

    lapack::Potrf( 'U',width, XTX.Data(), width );
    cu_XTX.CopyFrom(XTX);

    // X <- X * U^{-1} is orthogonal
    cublas::Trsm( right, up, cu_transN, nondiag, heightLocal, width, &one, 
        cu_XTX.Data(), width, cu_X.Data(), heightLocal );
    cublas::Trsm( right, up, cu_transN, nondiag, heightLocal, width, &one, 
        cu_XTX.Data(), width, cu_AX.Data(), heightLocal );

    cu_XTX.CopyTo(XTX);

  } while( (iter < eigMaxIter) && (resMax > eigTolerance) );

  // *********************************************************************
  // Post processing
  // *********************************************************************

  // Obtain the eigenvalues and eigenvectors
  // if isConverged == true then XTX should contain the matrix X' * (AX); and X is an
  // orthonormal set

  if (!isConverged){
    cublas::Gemm( cu_transT, cu_transN, width, width, heightLocal, &one, cu_X.Data(),
        heightLocal, cu_AX.Data(), heightLocal, &zero, cu_XTXtemp1.Data(), width);

    cu_XTXtemp1.CopyTo(XTXtemp1);
    MPI_Allreduce( XTXtemp1.Data(), XTX.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
  }

  if(esdfParam.PWSolver == "PPCGScaLAPACK")
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
  else // PWSolver_ == "PPCG"
  {
    if( mpirank == 0 ){
      lapack::Syevd( 'V','U',width, XTX.Data(), width, eigValS.Data() );
    }
    
    cu_XTX.CopyFrom( XTX );
  }

  MPI_Bcast(XTX.Data(), width*width, MPI_DOUBLE, 0, mpi_comm);
  MPI_Bcast(eigValS.Data(), width, MPI_DOUBLE, 0, mpi_comm);

  cu_XTX.CopyFrom( XTX );

  // X <- X*C
  cublas::Gemm( cu_transN, cu_transN, heightLocal, width, width, &one, cu_X.Data(),
      heightLocal, cu_XTX.Data(), width, &zero, cu_Xtemp.Data(), heightLocal);
  cu_Xtemp.CopyTo( cu_X );

  // AX <- AX*C
  cublas::Gemm( cu_transN, cu_transN, heightLocal, width, width, &one, cu_AX.Data(),
      heightLocal, cu_XTX.Data(), width, &zero, cu_Xtemp.Data(), heightLocal);
  cu_Xtemp.CopyTo( cu_AX );

  // Compute norms of individual eigenpairs
  cuDblNumVec cu_eigValS(lda);
  cu_eigValS.CopyFrom(eigValS);

  cu_X_Equal_AX_minus_X_eigVal(cu_Xtemp.Data(), cu_AX.Data(), cu_X.Data(),
      cu_eigValS.Data(), width, heightLocal);

  cuda_calculate_Energy( cu_Xtemp.Data(), cu_resNormLocal.Data(), width, heightLocal);
  cu_resNormLocal.CopyTo(resNormLocal);

  SetValue( resNorm, 0.0 );
  MPI_Allreduce( resNormLocal.Data(), resNorm.Data(), width, MPI_DOUBLE,
      MPI_SUM, mpi_comm );

  if ( mpirank == 0 ){
    for( Int k = 0; k < width; k++ ){
      resNorm(k) = std::sqrt( resNorm(k) ) / std::max( 1.0, std::abs( eigValS(k) ) );
    }
  }

  MPI_Bcast(resNorm.Data(), width, MPI_DOUBLE, 0, mpi_comm);

  resMax = *(std::max_element( resNorm.Data(), resNorm.Data() + numEig ) );
  resMin = *(std::min_element( resNorm.Data(), resNorm.Data() + numEig ) );

  notconv = 0;
  for( Int i = 0; i < numEig; i++ ){
    if( resNorm[i] > eigTolerance ){
      notconv ++;
    }
  }

  // Save the eigenvalues and eigenvectors back to the eigensolver data
  // structure
  blas::Copy( width, eigValS.Data(), 1, &eigVal_[noccTotal*spinswitch], 1 );
  blas::Copy( width, resNorm.Data(), 1, &resVal_[noccTotal*spinswitch], 1 );

  cuda_mapping_to_buf( cu_recvbuf.Data(), cu_X.Data(), cu_recvk.Data(), heightLocal*width);
#ifdef GPUDIRECT
  MPI_Alltoallv( &cu_recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE,
      &cu_sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE, mpi_comm );
#else
  cuda_memcpy_GPU2CPU( recvbuf.Data(), cu_recvbuf.Data(), sizeof(Real)*heightLocal*width);
  MPI_Alltoallv( &recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE,
      &sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE, mpi_comm );
  cuda_memcpy_CPU2GPU(cu_sendbuf.Data(), sendbuf.Data(), sizeof(Real)*height*widthLocal);
#endif
  cuda_mapping_from_buf(cu_Xcol.Data(), cu_sendbuf.Data(), cu_sendk.Data(), height*widthLocal);

  cuda_memcpy_GPU2CPU( psiTemp.Wavefun().MatData(noccLocal*spinswitch),
      cu_Xcol.Data(), sizeof(Real)*height*widthLocal);

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

  cuda_set_vtot_flag(); // set the vtot_flag to false.

  return;
}         // -----  end of method EigenSolver::PPCGSolveRealGPU  -----

} // namespace pwdft





