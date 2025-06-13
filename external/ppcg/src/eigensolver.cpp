/// @file eigensolver.cpp
/// @brief Eigensolver in the global domain.
/// @date 2023-07-01
#include  "ppcg/eigensolver.hpp"
#include  "ppcg/utility.hpp"
#include  "ppcg/blas.hpp"
#include  "ppcg/lapack.hpp"
#include  "ppcg/mpi_interf.hpp"

namespace PPCG {

EigenSolver::EigenSolver() {

}

EigenSolver::~EigenSolver() {

}

void EigenSolver::Setup(
    Hamiltonian& ham,
    Spinor& psi,
    Fourier& fft ) {
  hamPtr_ = &ham;
  psiPtr_ = &psi;
  fftPtr_ = &fft;


  eigVal_.Resize(psiPtr_->NumStateTotal());  SetValue(eigVal_, 0.0);
  resVal_.Resize(psiPtr_->NumStateTotal());  SetValue(resVal_, 0.0);

  return;
}

// Basic version of PPCG with columnwise sweep  
void
EigenSolver::PPCGSolveReal    (
    Int          Iter,
    Int          numEig,
    Int          eigMaxIter,
    Real         eigMinTolerance,
    Real         eigTolerance,
    Int          sbSize_in)
{
  // *********************************************************************
  // Initialization
  // *********************************************************************
  //std::stringstream  ss;
  //ss << "ppcg." << Iter;
  //statusOFS.open( ss.str().c_str() );  

  MPI_Comm mpi_comm = fftPtr_->domain.comm;
  MPI_Barrier(mpi_comm);
  Int mpirank;  MPI_Comm_rank(mpi_comm, &mpirank);
  Int mpisize;  MPI_Comm_size(mpi_comm, &mpisize);

  Int ntot = psiPtr_->NumGridTotal();
  Int ncom = psiPtr_->NumComponent();
  Int noccLocal = psiPtr_->NumState();
  Int noccTotal = psiPtr_->NumStateTotal();


  Int height = ntot * ncom;
  Int width = noccTotal;
  Int lda = 3 * width;

  Int widthBlocksize = width / mpisize;
  Int heightBlocksize = height / mpisize;
  Int widthLocal = widthBlocksize;
  Int heightLocal = heightBlocksize;

  if(mpirank < (width % mpisize)){
    widthLocal = widthBlocksize + 1;
  }

  if(mpirank < (height % mpisize)){
    heightLocal = heightBlocksize + 1;
  }

  if( widthLocal != noccLocal ){
    throw std::logic_error("widthLocal != noccLocal.");
  }

  // Time for GemmT, GemmN, Alltoallv, Spinor, Mpirank0 
  // GemmT: blas::Gemm( 'T', 'N')
  // GemmN: blas::Gemm( 'N', 'N')
  // Alltoallv: row-partition to column partition via MPI_Alltoallv 
  // Spinor: Applying the Hamiltonian matrix 
  // Mpirank0: Serial calculation part

  Real timeSta, timeEnd;
  Real timeSta1, timeEnd1;
  Real timeSta2, timeEnd2;
  Real timeGemmT = 0.0;
  Real timeGemmN = 0.0;
  Real timeBcast = 0.0;
  Real timeAllreduce = 0.0;
  Real timeAlltoallv = 0.0;
  Real timeAlltoallvMap = 0.0;
  Real timeSpinor = 0.0;
  Real timeTrsm = 0.0;
  Real timePotrf = 0.0;
  Real timeSyevd = 0.0;
  Real timeSygvd = 0.0;
  Real timeMpirank0 = 0.0;
  Real timeScaLAPACKFactor = 0.0;
  Real timeScaLAPACK = 0.0;
  Real timeSweepT = 0.0;
  Real timeCopy = 0.0;
  Real timeOther = 0.0;
  Int  iterGemmT = 0;
  Int  iterGemmN = 0;
  Int  iterBcast = 0;
  Int  iterAllreduce = 0;
  Int  iterAlltoallv = 0;
  Int  iterAlltoallvMap = 0;
  Int  iterSpinor = 0;
  Int  iterTrsm = 0;
  Int  iterPotrf = 0;
  Int  iterSyevd = 0;
  Int  iterSygvd = 0;
  Int  iterMpirank0 = 0;
  Int  iterScaLAPACKFactor = 0;
  Int  iterScaLAPACK = 0;
  Int  iterSweepT = 0;
  Int  iterCopy = 0;
  Int  iterOther = 0;

  if( numEig > width ){
    std::ostringstream msg;
    msg 
      << "Number of eigenvalues requested  = " << numEig << std::endl
      << "which is larger than the number of columns in psi = " << width << std::endl;
    throw std::runtime_error( msg.str().c_str() );
  }

  GetTime( timeSta2 );

  // The following codes are not replaced by AlltoallForward /
  // AlltoallBackward since they are repetitively used in the
  // eigensolver.
  //
  DblNumVec sendbuf(height*widthLocal); 
  DblNumVec recvbuf(heightLocal*width);
  IntNumVec sendcounts(mpisize);
  IntNumVec recvcounts(mpisize);
  IntNumVec senddispls(mpisize);
  IntNumVec recvdispls(mpisize);
  IntNumMat  sendk( height, widthLocal );
  IntNumMat  recvk( heightLocal, width );

  GetTime( timeSta );

  for( Int k = 0; k < mpisize; k++ ){ 
    sendcounts[k] = heightBlocksize * widthLocal;
    if( k < (height % mpisize)){
      sendcounts[k] = sendcounts[k] + widthLocal;  
    }
  }

  for( Int k = 0; k < mpisize; k++ ){ 
    recvcounts[k] = heightLocal * widthBlocksize;
    if( k < (width % mpisize)){
      recvcounts[k] = recvcounts[k] + heightLocal;  
    }
  }

  senddispls[0] = 0;
  recvdispls[0] = 0;
  for( Int k = 1; k < mpisize; k++ ){ 
    senddispls[k] = senddispls[k-1] + sendcounts[k-1];
    recvdispls[k] = recvdispls[k-1] + recvcounts[k-1];
  }

  if((height % mpisize) == 0){
    for( Int j = 0; j < widthLocal; j++ ){ 
      for( Int i = 0; i < height; i++ ){
        sendk(i, j) = senddispls[i / heightBlocksize] + j * heightBlocksize + i % heightBlocksize;
      } 
    }
  }
  else{
    for( Int j = 0; j < widthLocal; j++ ){ 
      for( Int i = 0; i < height; i++ ){
        if( i < ((height % mpisize) * (heightBlocksize+1)) ){
          sendk(i, j) = senddispls[i / (heightBlocksize+1)] + j * (heightBlocksize+1) + i % (heightBlocksize+1);
        }
        else {
          sendk(i, j) = senddispls[(height % mpisize) + (i-(height % mpisize)*(heightBlocksize+1))/heightBlocksize]
            + j * heightBlocksize + (i-(height % mpisize)*(heightBlocksize+1)) % heightBlocksize;
        }
      }
    }
  }

  for( Int j = 0; j < width; j++ ){ 
    for( Int i = 0; i < heightLocal; i++ ){
      recvk(i, j) = recvdispls[j % mpisize] + (j / mpisize) * heightLocal + i;
    }
  }
  // end For Alltoall

  GetTime( timeEnd );
  iterAlltoallvMap = iterAlltoallvMap + 1;
  timeAlltoallvMap = timeAlltoallvMap + ( timeEnd - timeSta );

  // S = ( X | W | P ) is a triplet used for LOBPCG.  
  // W is the preconditioned residual
  // DblNumMat  S( height, 3*widthLocal ), AS( height, 3*widthLocal ); 
  DblNumMat  S( heightLocal, 3*width ), AS( heightLocal, 3*width ); 
  // AMat = S' * (AS),  BMat = S' * S
  // 
  // AMat = (X'*AX   X'*AW   X'*AP)
  //      = (  *     W'*AW   W'*AP)
  //      = (  *       *     P'*AP)
  //
  // BMat = (X'*X   X'*W   X'*P)
  //      = (  *    W'*W   W'*P)
  //      = (  *      *    P'*P)
  //


  //    DblNumMat  AMat( 3*width, 3*width ), BMat( 3*width, 3*width );
  //    DblNumMat  AMatT1( 3*width, 3*width );

  // Temporary buffer array.
  // The unpreconditioned residual will also be saved in Xtemp
  DblNumMat  XTX( width, width );
  //?    DblNumMat  XTXtemp( width, width );
  DblNumMat  XTXtemp1( width, width );

  DblNumMat  Xtemp( heightLocal, width );

  Real  resBlockNormLocal, resBlockNorm; // Frobenius norm of the residual block  
  Real  resMax, resMin;

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

  //Int info;
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
  GetTime( timeSta );
  lapack::Lacpy( 'A', height, widthLocal, psiPtr_->Wavefun().Data(), height, 
      Xcol.Data(), height );
  GetTime( timeEnd );
  iterCopy = iterCopy + 1;
  timeCopy = timeCopy + ( timeEnd - timeSta );

  GetTime( timeSta );
  for( Int j = 0; j < widthLocal; j++ ){ 
    for( Int i = 0; i < height; i++ ){
      sendbuf[sendk(i, j)] = Xcol(i, j); 
    }
  }
  MPI_Alltoallv( &sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE, 
      &recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE, mpi_comm );
  for( Int j = 0; j < width; j++ ){ 
    for( Int i = 0; i < heightLocal; i++ ){
      X(i, j) = recvbuf[recvk(i, j)];
    }
  }
  GetTime( timeEnd );
  iterAlltoallv = iterAlltoallv + 1;
  timeAlltoallv = timeAlltoallv + ( timeEnd - timeSta );

  // *********************************************************************
  // Main loop
  // *********************************************************************

  // Orthogonalization through Cholesky factorization
  GetTime( timeSta );
  blas::Gemm( 'T', 'N', width, width, heightLocal, 1.0, X.Data(), 
      heightLocal, X.Data(), heightLocal, 0.0, XTXtemp1.Data(), width );
  GetTime( timeEnd );
  iterGemmT = iterGemmT + 1;
  timeGemmT = timeGemmT + ( timeEnd - timeSta );
  GetTime( timeSta );
  SetValue( XTX, 0.0 );
  MPI_Allreduce( XTXtemp1.Data(), XTX.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
  GetTime( timeEnd );
  iterAllreduce = iterAllreduce + 1;
  timeAllreduce = timeAllreduce + ( timeEnd - timeSta );

  if ( mpirank == 0) {
    GetTime( timeSta );
    lapack::Potrf( 'U', width, XTX.Data(), width );
    GetTime( timeEnd );
    iterMpirank0 = iterMpirank0 + 1;
    timeMpirank0 = timeMpirank0 + ( timeEnd - timeSta );
  }
  GetTime( timeSta );
  MPI_Bcast(XTX.Data(), width*width, MPI_DOUBLE, 0, mpi_comm);
  GetTime( timeEnd );
  iterBcast = iterBcast + 1;
  timeBcast = timeBcast + ( timeEnd - timeSta );

  // X <- X * U^{-1} is orthogonal
  GetTime( timeSta );
  blas::Trsm( 'R', 'U', 'N', 'N', heightLocal, width, 1.0, XTX.Data(), width, 
      X.Data(), heightLocal );
  GetTime( timeEnd );
  iterTrsm = iterTrsm + 1;
  timeTrsm = timeTrsm + ( timeEnd - timeSta );

  GetTime( timeSta );
  for( Int j = 0; j < width; j++ ){ 
    for( Int i = 0; i < heightLocal; i++ ){
      recvbuf[recvk(i, j)] = X(i, j);
    }
  }
  MPI_Alltoallv( &recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE, 
      &sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE, mpi_comm );
  for( Int j = 0; j < widthLocal; j++ ){ 
    for( Int i = 0; i < height; i++ ){
      Xcol(i, j) = sendbuf[sendk(i, j)]; 
    }
  }
  GetTime( timeEnd );
  iterAlltoallv = iterAlltoallv + 1;
  timeAlltoallv = timeAlltoallv + ( timeEnd - timeSta );

  // Applying the Hamiltonian matrix
  {
    GetTime( timeSta );
    Spinor spnTemp(fftPtr_->domain, ncom, noccTotal, noccLocal, false, Xcol.Data());
    NumTns<Real> tnsTemp(ntot, ncom, noccLocal, false, AXcol.Data());

    hamPtr_->MultSpinor( spnTemp, tnsTemp, *fftPtr_ );
    GetTime( timeEnd );
    iterSpinor = iterSpinor + 1;
    timeSpinor = timeSpinor + ( timeEnd - timeSta );
  }

  GetTime( timeSta );
  for( Int j = 0; j < widthLocal; j++ ){ 
    for( Int i = 0; i < height; i++ ){
      sendbuf[sendk(i, j)] = AXcol(i, j); 
    }
  }
  MPI_Alltoallv( &sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE, 
      &recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE, mpi_comm );
  for( Int j = 0; j < width; j++ ){ 
    for( Int i = 0; i < heightLocal; i++ ){
      AX(i, j) = recvbuf[recvk(i, j)];
    }
  }
  GetTime( timeEnd );
  iterAlltoallv = iterAlltoallv + 1;
  timeAlltoallv = timeAlltoallv + ( timeEnd - timeSta );


  // Start the main loop
  Int iter = 0;
  statusOFS << "Minimum tolerance is " << eigMinTolerance << std::endl;

  do{
    iter++;
#if ( _DEBUGlevel_ >= 1 )
    statusOFS << "iter = " << iter << std::endl;
#endif

    if( iter == 1 || isRestart == true )
      numSet = 2;
    else
      numSet = 3;

    // XTX <- X' * (AX)
    GetTime( timeSta );
    blas::Gemm( 'T', 'N', width, width, heightLocal, 1.0, X.Data(),
        heightLocal, AX.Data(), heightLocal, 0.0, XTXtemp1.Data(), width );
    GetTime( timeEnd );
    iterGemmT = iterGemmT + 1;
    timeGemmT = timeGemmT + ( timeEnd - timeSta );
    GetTime( timeSta );
    SetValue( XTX, 0.0 );
    MPI_Allreduce( XTXtemp1.Data(), XTX.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
    GetTime( timeEnd );
    iterAllreduce = iterAllreduce + 1;
    timeAllreduce = timeAllreduce + ( timeEnd - timeSta );

    // Compute the residual.
    // R <- AX - X*(X'*AX)
    GetTime( timeSta );
    lapack::Lacpy( 'A', heightLocal, width, AX.Data(), heightLocal, Xtemp.Data(), heightLocal );
    GetTime( timeEnd );
    iterCopy = iterCopy + 1;
    timeCopy = timeCopy + ( timeEnd - timeSta );

    GetTime( timeSta );
    blas::Gemm( 'N', 'N', heightLocal, width, width, -1.0, 
        X.Data(), heightLocal, XTX.Data(), width, 1.0, Xtemp.Data(), heightLocal );
    GetTime( timeEnd );
    iterGemmN = iterGemmN + 1;
    timeGemmN = timeGemmN + ( timeEnd - timeSta );



    // Compute the Frobenius norm of the residual block

    if(0){

      GetTime( timeSta );

      resBlockNormLocal = 0.0; resBlockNorm = 0.0; 
      for (Int i=0; i < heightLocal; i++){
        for (Int j=0; j < width; j++ ){
          resBlockNormLocal += Xtemp(i,j)*Xtemp(i,j); 
        }
      }

      MPI_Allreduce( &resBlockNormLocal, &resBlockNorm, 1, MPI_DOUBLE,
          MPI_SUM, mpi_comm); 
      resBlockNorm = std::sqrt(resBlockNorm);

      GetTime( timeEnd );
      iterOther = iterOther + 1;
      timeOther = timeOther + ( timeEnd - timeSta );

      statusOFS << "Time for resBlockNorm in PWDFT is " <<  timeEnd - timeSta  << std::endl << std::endl;


      /////////////// UNCOMMENT THIS #if ( _DEBUGlevel_ >= 1 )
      statusOFS << "Frob. norm of the residual block = " << resBlockNorm << std::endl;
      //////////////#endif

      // THIS STOPPING CRITERION LIKELY IRRELEVANT
      if( resBlockNorm < eigTolerance ){
        isConverged = true;
        break;
      }

    } // if(0)

    // LOCKING not supported, PPCG needs Rayleigh--Ritz to lock         
    //        numActiveTotal = width - numLockedTotal;
    //        numActiveLocal = widthLocal - numLockedLocal;

    // Compute the preconditioned residual W = T*R.
    // The residual is saved in Xtemp

    // Convert from row format to column format.
    // MPI_Alltoallv
    // Only convert Xtemp here

    GetTime( timeSta );
    for( Int j = 0; j < width; j++ ){ 
      for( Int i = 0; i < heightLocal; i++ ){
        recvbuf[recvk(i, j)] = Xtemp(i, j);
      }
    }
    MPI_Alltoallv( &recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE, 
        &sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE, mpi_comm );
    for( Int j = 0; j < widthLocal; j++ ){ 
      for( Int i = 0; i < height; i++ ){
        Xcol(i, j) = sendbuf[sendk(i, j)]; 
      }
    }
    GetTime( timeEnd );
    iterAlltoallv = iterAlltoallv + 1;
    timeAlltoallv = timeAlltoallv + ( timeEnd - timeSta );

    // Compute W = TW
    {
      GetTime( timeSta );
      Spinor spnTemp(fftPtr_->domain, ncom, noccTotal, widthLocal-numLockedLocal, false, Xcol.VecData(numLockedLocal));
      NumTns<Real> tnsTemp(ntot, ncom, widthLocal-numLockedLocal, false, Wcol.VecData(numLockedLocal));

      SetValue( tnsTemp, 0.0 );
      spnTemp.AddTeterPrecond( fftPtr_, tnsTemp );
      GetTime( timeEnd );
      iterSpinor = iterSpinor + 1;
      timeSpinor = timeSpinor + ( timeEnd - timeSta );
    }


    // Compute AW = A*W
    {
      GetTime( timeSta );
      Spinor spnTemp(fftPtr_->domain, ncom, noccTotal, widthLocal-numLockedLocal, false, Wcol.VecData(numLockedLocal));
      NumTns<Real> tnsTemp(ntot, ncom, widthLocal-numLockedLocal, false, AWcol.VecData(numLockedLocal));

      hamPtr_->MultSpinor( spnTemp, tnsTemp, *fftPtr_ );
      GetTime( timeEnd );
      iterSpinor = iterSpinor + 1;
      timeSpinor = timeSpinor + ( timeEnd - timeSta );
    }

    // Convert from column format to row format
    // MPI_Alltoallv
    // Only convert W and AW

    GetTime( timeSta );
    for( Int j = 0; j < widthLocal; j++ ){ 
      for( Int i = 0; i < height; i++ ){
        sendbuf[sendk(i, j)] = Wcol(i, j); 
      }
    }
    MPI_Alltoallv( &sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE, 
        &recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE, mpi_comm );
    for( Int j = 0; j < width; j++ ){ 
      for( Int i = 0; i < heightLocal; i++ ){
        W(i, j) = recvbuf[recvk(i, j)];
      }
    }
    GetTime( timeEnd );
    iterAlltoallv = iterAlltoallv + 1;
    timeAlltoallv = timeAlltoallv + ( timeEnd - timeSta );

    GetTime( timeSta );
    for( Int j = 0; j < widthLocal; j++ ){ 
      for( Int i = 0; i < height; i++ ){
        sendbuf[sendk(i, j)] = AWcol(i, j); 
      }
    }
    MPI_Alltoallv( &sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE, 
        &recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE, mpi_comm );
    for( Int j = 0; j < width; j++ ){ 
      for( Int i = 0; i < heightLocal; i++ ){
        AW(i, j) = recvbuf[recvk(i, j)];
      }
    }
    GetTime( timeEnd );
    iterAlltoallv = iterAlltoallv + 1;
    timeAlltoallv = timeAlltoallv + ( timeEnd - timeSta );


    // W = W - X(X'W), AW = AW - AX(X'W)
    GetTime( timeSta );
    blas::Gemm( 'T', 'N', width, width, heightLocal, 1.0, X.Data(),
        heightLocal, W.Data(), heightLocal, 0.0, XTXtemp1.Data(), width );
    GetTime( timeEnd );
    iterGemmT = iterGemmT + 1;
    timeGemmT = timeGemmT + ( timeEnd - timeSta );
    GetTime( timeSta );
    SetValue( XTX, 0.0 );
    MPI_Allreduce( XTXtemp1.Data(), XTX.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
    GetTime( timeEnd );
    iterAllreduce = iterAllreduce + 1;
    timeAllreduce = timeAllreduce + ( timeEnd - timeSta );


    GetTime( timeSta );
    blas::Gemm( 'N', 'N', heightLocal, width, width, -1.0, 
        X.Data(), heightLocal, XTX.Data(), width, 1.0, W.Data(), heightLocal );
    GetTime( timeEnd );
    iterGemmN = iterGemmN + 1;
    timeGemmN = timeGemmN + ( timeEnd - timeSta );


    GetTime( timeSta );
    blas::Gemm( 'N', 'N', heightLocal, width, width, -1.0, 
        AX.Data(), heightLocal, XTX.Data(), width, 1.0, AW.Data(), heightLocal );
    GetTime( timeEnd );
    iterGemmN = iterGemmN + 1;
    timeGemmN = timeGemmN + ( timeEnd - timeSta );

    // Normalize columns of W
    Real normLocal[width]; 
    Real normGlobal[width];

    GetTime( timeSta );
    for( Int k = numLockedLocal; k < width; k++ ){
      normLocal[k] = Energy(DblNumVec(heightLocal, false, W.VecData(k)));
      normGlobal[k] = 0.0;
    }
    MPI_Allreduce( &normLocal[0], &normGlobal[0], width, MPI_DOUBLE, MPI_SUM, mpi_comm );
    for( Int k = numLockedLocal; k < width; k++ ){
      Real norm = std::sqrt( normGlobal[k] );
      blas::Scal( heightLocal, 1.0 / norm, W.VecData(k), 1 );
      blas::Scal( heightLocal, 1.0 / norm, AW.VecData(k), 1 );
    }
    GetTime( timeEnd );
    iterOther = iterOther + 2;
    timeOther = timeOther + ( timeEnd - timeSta );

#if ( _DEBUGlevel_ >= 1 )
    statusOFS << "Time for norm1 in PWDFT is " <<  timeEnd - timeSta  << std::endl << std::endl;
#endif


    // P = P - X(X'P), AP = AP - AX(X'P)
    if( numSet == 3 ){
      GetTime( timeSta );
      blas::Gemm( 'T', 'N', width, width, heightLocal, 1.0, X.Data(),
          heightLocal, P.Data(), heightLocal, 0.0, XTXtemp1.Data(), width );
      GetTime( timeEnd );
      iterGemmT = iterGemmT + 1;
      timeGemmT = timeGemmT + ( timeEnd - timeSta );
      GetTime( timeSta );
      SetValue( XTX, 0.0 );
      MPI_Allreduce( XTXtemp1.Data(), XTX.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
      GetTime( timeEnd );
      iterAllreduce = iterAllreduce + 1;
      timeAllreduce = timeAllreduce + ( timeEnd - timeSta );

      GetTime( timeSta );
      blas::Gemm( 'N', 'N', heightLocal, width, width, -1.0, 
          X.Data(), heightLocal, XTX.Data(), width, 1.0, P.Data(), heightLocal );
      GetTime( timeEnd );
      iterGemmN = iterGemmN + 1;
      timeGemmN = timeGemmN + ( timeEnd - timeSta );

      GetTime( timeSta );
      blas::Gemm( 'N', 'N', heightLocal, width, width, -1.0, 
          AX.Data(), heightLocal, XTX.Data(), width, 1.0, AP.Data(), heightLocal );
      GetTime( timeEnd );
      iterGemmN = iterGemmN + 1;
      timeGemmN = timeGemmN + ( timeEnd - timeSta );

      // Normalize the conjugate direction
      GetTime( timeSta );
      for( Int k = numLockedLocal; k < width; k++ ){
        normLocal[k] = Energy(DblNumVec(heightLocal, false, P.VecData(k)));
        normGlobal[k] = 0.0;
      }
      MPI_Allreduce( &normLocal[0], &normGlobal[0], width, MPI_DOUBLE, MPI_SUM, mpi_comm );
      for( Int k = numLockedLocal; k < width; k++ ){
        Real norm = std::sqrt( normGlobal[k] );
        blas::Scal( heightLocal, 1.0 / norm, P.VecData(k), 1 );
        blas::Scal( heightLocal, 1.0 / norm, AP.VecData(k), 1 );
      }
      GetTime( timeEnd );
      iterOther = iterOther + 2;
      timeOther = timeOther + ( timeEnd - timeSta );

#if ( _DEBUGlevel_ >= 1 )
      statusOFS << "Time for norm2 in PWDFT is " <<  timeEnd - timeSta  << std::endl << std::endl;
#endif

    }

    // Perform the sweep
    GetTime( timeSta );
    Int sbSize = sbSize_in, nsb = width/sbSize; // this should be generalized to subblocks 
    DblNumMat  AMat( 3*sbSize, 3*sbSize ), BMat( 3*sbSize, 3*sbSize );
    DblNumMat  AMatAll( 3*sbSize, 3*sbSize*nsb ), BMatAll( 3*sbSize, 3*sbSize*nsb ); // contains all nsb 3-by-3 matrices
    DblNumMat  AMatAllLocal( 3*sbSize, 3*sbSize*nsb ), BMatAllLocal( 3*sbSize, 3*sbSize*nsb ); // contains local parts of all nsb 3-by-3 matrices

    SetValue( AMat, 0.0 ); SetValue( BMat, 0.0 );
    SetValue( AMatAll, 0.0 ); SetValue( BMatAll, 0.0 );
    SetValue( AMatAllLocal, 0.0 ); SetValue( BMatAllLocal, 0.0 );

    // LOCKING NOT SUPPORTED, loop over all columns 
    for( Int k = 0; k < nsb; k++ ){

      // fetch indiviual columns
      DblNumMat  x( heightLocal, sbSize, false, X.VecData(sbSize*k) );
      DblNumMat  w( heightLocal, sbSize, false, W.VecData(sbSize*k) );
      DblNumMat ax( heightLocal, sbSize, false, AX.VecData(sbSize*k) );
      DblNumMat aw( heightLocal, sbSize, false, AW.VecData(sbSize*k) );

      // Compute AMatAllLoc and BMatAllLoc            
      // AMatAllLoc
      GetTime( timeSta );
      blas::Gemm( 'T', 'N', sbSize, sbSize, heightLocal, 1.0, x.Data(),
          heightLocal, ax.Data(), heightLocal, 
          0.0, &AMatAllLocal(0,3*sbSize*k), 3*sbSize );
      GetTime( timeEnd );
      iterGemmT = iterGemmT + 1;
      timeGemmT = timeGemmT + ( timeEnd - timeSta );

      GetTime( timeSta );
      blas::Gemm( 'T', 'N', sbSize, sbSize, heightLocal, 1.0, w.Data(),
          heightLocal, aw.Data(), heightLocal, 
          0.0, &AMatAllLocal(sbSize,3*sbSize*k+sbSize), 3*sbSize );
      GetTime( timeEnd );
      iterGemmT = iterGemmT + 1;
      timeGemmT = timeGemmT + ( timeEnd - timeSta );

      GetTime( timeSta );
      blas::Gemm( 'T', 'N', sbSize, sbSize, heightLocal, 1.0, x.Data(),
          heightLocal, aw.Data(), heightLocal, 
          0.0, &AMatAllLocal(0,3*sbSize*k+sbSize), 3*sbSize );
      GetTime( timeEnd );
      iterGemmT = iterGemmT + 1;
      timeGemmT = timeGemmT + ( timeEnd - timeSta );

      // BMatAllLoc            
      GetTime( timeSta );
      blas::Gemm( 'T', 'N', sbSize, sbSize, heightLocal, 1.0, x.Data(),
          heightLocal, x.Data(), heightLocal, 
          0.0, &BMatAllLocal(0,3*sbSize*k), 3*sbSize );
      GetTime( timeEnd );
      iterGemmT = iterGemmT + 1;
      timeGemmT = timeGemmT + ( timeEnd - timeSta );

      GetTime( timeSta );
      blas::Gemm( 'T', 'N', sbSize, sbSize, heightLocal, 1.0, w.Data(),
          heightLocal, w.Data(), heightLocal, 
          0.0, &BMatAllLocal(sbSize,3*sbSize*k+sbSize), 3*sbSize );
      GetTime( timeEnd );
      iterGemmT = iterGemmT + 1;
      timeGemmT = timeGemmT + ( timeEnd - timeSta );

      GetTime( timeSta );
      blas::Gemm( 'T', 'N', sbSize, sbSize, heightLocal, 1.0, x.Data(),
          heightLocal, w.Data(), heightLocal, 
          0.0, &BMatAllLocal(0,3*sbSize*k+sbSize), 3*sbSize );
      GetTime( timeEnd );
      iterGemmT = iterGemmT + 1;
      timeGemmT = timeGemmT + ( timeEnd - timeSta );

      if ( numSet == 3 ){

        DblNumMat  p( heightLocal, sbSize, false, P.VecData(k) );
        DblNumMat ap( heightLocal, sbSize, false, AP.VecData(k) );

        // AMatAllLoc
        GetTime( timeSta );
        blas::Gemm( 'T', 'N', sbSize, sbSize, heightLocal, 1.0, p.Data(),
            heightLocal, ap.Data(), heightLocal, 
            0.0, &AMatAllLocal(2*sbSize,3*sbSize*k+2*sbSize), 3*sbSize );
        GetTime( timeEnd );
        iterGemmT = iterGemmT + 1;
        timeGemmT = timeGemmT + ( timeEnd - timeSta );

        GetTime( timeSta );
        blas::Gemm( 'T', 'N', sbSize, sbSize, heightLocal, 1.0, x.Data(),
            heightLocal, ap.Data(), heightLocal, 
            0.0, &AMatAllLocal(0, 3*sbSize*k+2*sbSize), 3*sbSize );
        GetTime( timeEnd );
        iterGemmT = iterGemmT + 1;
        timeGemmT = timeGemmT + ( timeEnd - timeSta );

        GetTime( timeSta );
        blas::Gemm( 'T', 'N', sbSize, sbSize, heightLocal, 1.0, w.Data(),
            heightLocal, ap.Data(), heightLocal, 
            0.0, &AMatAllLocal(sbSize, 3*sbSize*k+2*sbSize), 3*sbSize );
        GetTime( timeEnd );
        iterGemmT = iterGemmT + 1;
        timeGemmT = timeGemmT + ( timeEnd - timeSta );

        // BMatAllLoc
        GetTime( timeSta );
        blas::Gemm( 'T', 'N', sbSize, sbSize, heightLocal, 1.0, p.Data(),
            heightLocal, p.Data(), heightLocal, 
            0.0, &BMatAllLocal(2*sbSize,3*sbSize*k+2*sbSize), 3*sbSize );
        GetTime( timeEnd );
        iterGemmT = iterGemmT + 1;
        timeGemmT = timeGemmT + ( timeEnd - timeSta );

        GetTime( timeSta );
        blas::Gemm( 'T', 'N', sbSize, sbSize, heightLocal, 1.0, x.Data(),
            heightLocal, p.Data(), heightLocal, 
            0.0, &BMatAllLocal(0, 3*sbSize*k+2*sbSize), 3*sbSize );
        GetTime( timeEnd );
        iterGemmT = iterGemmT + 1;
        timeGemmT = timeGemmT + ( timeEnd - timeSta );

        GetTime( timeSta );
        blas::Gemm( 'T', 'N', sbSize, sbSize, heightLocal, 1.0, w.Data(),
            heightLocal, p.Data(), heightLocal, 
            0.0, &BMatAllLocal(sbSize, 3*sbSize*k+2*sbSize), 3*sbSize );
        GetTime( timeEnd );
        iterGemmT = iterGemmT + 1;
        timeGemmT = timeGemmT + ( timeEnd - timeSta );

      }             

    }

    GetTime( timeSta );
    MPI_Allreduce( AMatAllLocal.Data(), AMatAll.Data(), 9*sbSize*sbSize*nsb, MPI_DOUBLE, MPI_SUM, mpi_comm );
    GetTime( timeEnd );
    iterAllreduce = iterAllreduce + 1;
    timeAllreduce = timeAllreduce + ( timeEnd - timeSta );

    GetTime( timeSta );
    MPI_Allreduce( BMatAllLocal.Data(), BMatAll.Data(), 9*sbSize*sbSize*nsb, MPI_DOUBLE, MPI_SUM, mpi_comm );
    GetTime( timeEnd );
    iterAllreduce = iterAllreduce + 1;
    timeAllreduce = timeAllreduce + ( timeEnd - timeSta );

    // Solve nsb small eigenproblems and update columns of X 
    for( Int k = 0; k < nsb; k++ ){

      Real eigs[3*sbSize];
      DblNumMat  cx( sbSize, sbSize ), cw( sbSize, sbSize ), cp( sbSize, sbSize);
      DblNumMat tmp( heightLocal, sbSize );            

      // small eigensolve
      GetTime( timeSta );
      lapack::Lacpy( 'A', 3*sbSize, 3*sbSize, &AMatAll(0,3*sbSize*k), 3*sbSize, AMat.Data(), 3*sbSize );
      lapack::Lacpy( 'A', 3*sbSize, 3*sbSize, &BMatAll(0,3*sbSize*k), 3*sbSize, BMat.Data(), 3*sbSize );
      GetTime( timeEnd );
      iterCopy = iterCopy + 2;
      timeCopy = timeCopy + ( timeEnd - timeSta );

      //if (mpirank==0){
      //    statusOFS << "sweep num = " << k << std::endl;
      //    statusOFS << "AMat = " << AMat << std::endl;
      //    statusOFS << "BMat = " << BMat << std::endl<<std::endl;
      //}

      Int dim = (numSet == 3) ? 3*sbSize : 2*sbSize;
      GetTime( timeSta );
      lapack::Sygvd(1, 'V', 'U', dim, AMat.Data(), 3*sbSize, BMat.Data(), 3*sbSize, eigs);
      GetTime( timeEnd );
      iterSygvd = iterSygvd + 1;
      timeSygvd = timeSygvd + ( timeEnd - timeSta );

      // fetch indiviual columns
      DblNumMat  x( heightLocal, sbSize, false, X.VecData(sbSize*k) );
      DblNumMat  w( heightLocal, sbSize, false, W.VecData(sbSize*k) );
      DblNumMat  p( heightLocal, sbSize, false, P.VecData(sbSize*k) );
      DblNumMat ax( heightLocal, sbSize, false, AX.VecData(sbSize*k) );
      DblNumMat aw( heightLocal, sbSize, false, AW.VecData(sbSize*k) );
      DblNumMat ap( heightLocal, sbSize, false, AP.VecData(sbSize*k) );

      GetTime( timeSta );
      lapack::Lacpy( 'A', sbSize, sbSize, &AMat(0,0), 3*sbSize, cx.Data(), sbSize );
      lapack::Lacpy( 'A', sbSize, sbSize, &AMat(sbSize,0), 3*sbSize, cw.Data(), sbSize );
      GetTime( timeEnd );
      iterCopy = iterCopy + 2;
      timeCopy = timeCopy + ( timeEnd - timeSta );

      //  p = w*cw + p*cp; x = x*cx + p; ap = aw*cw + ap*cp; ax = ax*cx + ap;
      if( numSet == 3 ){

        GetTime( timeSta );
        lapack::Lacpy( 'A', sbSize, sbSize, &AMat(2*sbSize,0), 3*sbSize, cp.Data(), sbSize );
        GetTime( timeEnd );
        iterCopy = iterCopy + 1;
        timeCopy = timeCopy + ( timeEnd - timeSta );

        // tmp <- p*cp 
        GetTime( timeSta );
        blas::Gemm( 'N', 'N', heightLocal, sbSize, sbSize, 1.0,
            p.Data(), heightLocal, cp.Data(), sbSize,
            0.0, tmp.Data(), heightLocal );
        GetTime( timeEnd );
        iterGemmN = iterGemmN + 1;
        timeGemmN = timeGemmN + ( timeEnd - timeSta );

        // p <- w*cw + tmp
        GetTime( timeSta );
        blas::Gemm( 'N', 'N', heightLocal, sbSize, sbSize, 1.0,
            w.Data(), heightLocal, cw.Data(), sbSize,
            1.0, tmp.Data(), heightLocal );
        GetTime( timeEnd );
        iterGemmN = iterGemmN + 1;
        timeGemmN = timeGemmN + ( timeEnd - timeSta );

        GetTime( timeSta );
        lapack::Lacpy( 'A', heightLocal, sbSize, tmp.Data(), heightLocal, p.Data(), heightLocal );
        GetTime( timeEnd );
        iterCopy = iterCopy + 1;
        timeCopy = timeCopy + ( timeEnd - timeSta );

        // tmp <- ap*cp 
        GetTime( timeSta );
        blas::Gemm( 'N', 'N', heightLocal, sbSize, sbSize, 1.0,
            ap.Data(), heightLocal, cp.Data(), sbSize,
            0.0, tmp.Data(), heightLocal );
        GetTime( timeEnd );
        iterGemmN = iterGemmN + 1;
        timeGemmN = timeGemmN + ( timeEnd - timeSta );

        // ap <- aw*cw + tmp
        GetTime( timeSta );
        blas::Gemm( 'N', 'N', heightLocal, sbSize, sbSize, 1.0,
            aw.Data(), heightLocal, cw.Data(), sbSize,
            1.0, tmp.Data(), heightLocal );
        GetTime( timeEnd );
        iterGemmN = iterGemmN + 1;
        timeGemmN = timeGemmN + ( timeEnd - timeSta );
        GetTime( timeSta );
        lapack::Lacpy( 'A', heightLocal, sbSize, tmp.Data(), heightLocal, ap.Data(), heightLocal );
        GetTime( timeEnd );
        iterCopy = iterCopy + 1;
        timeCopy = timeCopy + ( timeEnd - timeSta );

      }else{
        // p <- w*cw
        GetTime( timeSta );
        blas::Gemm( 'N', 'N', heightLocal, sbSize, sbSize, 1.0,
            w.Data(), heightLocal, cw.Data(), sbSize,
            0.0, p.Data(), heightLocal );
        GetTime( timeEnd );
        iterGemmN = iterGemmN + 1;
        timeGemmN = timeGemmN + ( timeEnd - timeSta );
        // ap <- aw*cw
        GetTime( timeSta );
        blas::Gemm( 'N', 'N', heightLocal, sbSize, sbSize, 1.0,
            aw.Data(), heightLocal, cw.Data(), sbSize,
            0.0, ap.Data(), heightLocal );
        GetTime( timeEnd );
        iterGemmN = iterGemmN + 1;
        timeGemmN = timeGemmN + ( timeEnd - timeSta );
      }

      // x <- x*cx + p
      GetTime( timeSta );
      lapack::Lacpy( 'A', heightLocal, sbSize, p.Data(), heightLocal, tmp.Data(), heightLocal );
      GetTime( timeEnd );
      iterCopy = iterCopy + 1;
      timeCopy = timeCopy + ( timeEnd - timeSta );

      GetTime( timeSta );
      blas::Gemm( 'N', 'N', heightLocal, sbSize, sbSize, 1.0,
          x.Data(), heightLocal, cx.Data(), sbSize,
          1.0, tmp.Data(), heightLocal );
      GetTime( timeEnd );
      iterGemmN = iterGemmN + 1;
      timeGemmN = timeGemmN + ( timeEnd - timeSta );

      GetTime( timeSta );
      lapack::Lacpy( 'A', heightLocal, sbSize, tmp.Data(), heightLocal, x.Data(), heightLocal );
      GetTime( timeEnd );
      iterCopy = iterCopy + 1;
      timeCopy = timeCopy + ( timeEnd - timeSta );

      // ax <- ax*cx + ap
      GetTime( timeSta );
      lapack::Lacpy( 'A', heightLocal, sbSize, ap.Data(), heightLocal, tmp.Data(), heightLocal );
      GetTime( timeEnd );
      iterCopy = iterCopy + 1;
      timeCopy = timeCopy + ( timeEnd - timeSta );

      GetTime( timeSta );
      blas::Gemm( 'N', 'N', heightLocal, sbSize, sbSize, 1.0,
          ax.Data(), heightLocal, cx.Data(), sbSize,
          1.0, tmp.Data(), heightLocal );
      GetTime( timeEnd );
      iterGemmN = iterGemmN + 1;
      timeGemmN = timeGemmN + ( timeEnd - timeSta );

      GetTime( timeSta );
      lapack::Lacpy( 'A', heightLocal, sbSize, tmp.Data(), heightLocal, ax.Data(), heightLocal );
      GetTime( timeEnd );
      iterCopy = iterCopy + 1;
      timeCopy = timeCopy + ( timeEnd - timeSta );

    }

    // CholeskyQR of the updated block X
    GetTime( timeSta );
    blas::Gemm( 'T', 'N', width, width, heightLocal, 1.0, X.Data(), 
        heightLocal, X.Data(), heightLocal, 0.0, XTXtemp1.Data(), width );
    GetTime( timeEnd );
    iterGemmT = iterGemmT + 1;
    timeGemmT = timeGemmT + ( timeEnd - timeSta );
    GetTime( timeSta );
    SetValue( XTX, 0.0 );
    MPI_Allreduce( XTXtemp1.Data(), XTX.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
    GetTime( timeEnd );
    iterAllreduce = iterAllreduce + 1;
    timeAllreduce = timeAllreduce + ( timeEnd - timeSta );

    if ( mpirank == 0) {
      GetTime( timeSta );
      GetTime( timeSta1 );
      lapack::Potrf( 'U', width, XTX.Data(), width );
      GetTime( timeEnd1 );
      iterPotrf = iterPotrf + 1;
      timePotrf = timePotrf + ( timeEnd1 - timeSta1 );
      GetTime( timeEnd );
      iterMpirank0 = iterMpirank0 + 1;
      timeMpirank0 = timeMpirank0 + ( timeEnd - timeSta );
    }
    GetTime( timeSta );
    MPI_Bcast(XTX.Data(), width*width, MPI_DOUBLE, 0, mpi_comm);
    GetTime( timeEnd );
    iterBcast = iterBcast + 1;
    timeBcast = timeBcast + ( timeEnd - timeSta );

    // X <- X * U^{-1} is orthogonal
    GetTime( timeSta );
    blas::Trsm( 'R', 'U', 'N', 'N', heightLocal, width, 1.0, XTX.Data(), width, 
        X.Data(), heightLocal );
    blas::Trsm( 'R', 'U', 'N', 'N', heightLocal, width, 1.0, XTX.Data(), width,
        AX.Data(), heightLocal );
    GetTime( timeEnd );
    iterTrsm = iterTrsm + 1;
    timeTrsm = timeTrsm + ( timeEnd - timeSta );


    //            // Copy the eigenvalues
    //            SetValue( eigValS, 0.0 );
    //            for( Int i = 0; i < numKeep; i++ ){
    //                eigValS[i] = eigs[i];
    //            }


    //#if ( _DEBUGlevel_ >= 1 )
    //        statusOFS << "numLocked = " << numLocked << std::endl;
    //        statusOFS << "eigValS   = " << eigValS << std::endl;
    //#endif

  } while( (iter < (10 * eigMaxIter)) && ( (iter < eigMaxIter) || (resMin > eigMinTolerance) ) );



  // *********************************************************************
  // Post processing
  // *********************************************************************

  // Obtain the eigenvalues and eigenvectors
  // if isConverged==true then XTX should contain the matrix X' * (AX); and X is an
  // orthonormal set

  if (!isConverged){
    GetTime( timeSta );
    blas::Gemm( 'T', 'N', width, width, heightLocal, 1.0, X.Data(),
        heightLocal, AX.Data(), heightLocal, 0.0, XTXtemp1.Data(), width );
    GetTime( timeEnd );
    iterGemmT = iterGemmT + 1;
    timeGemmT = timeGemmT + ( timeEnd - timeSta );
    GetTime( timeSta );
    SetValue( XTX, 0.0 );
    MPI_Allreduce( XTXtemp1.Data(), XTX.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
    GetTime( timeEnd );
    iterAllreduce = iterAllreduce + 1;
    timeAllreduce = timeAllreduce + ( timeEnd - timeSta );
  }

  GetTime( timeSta1 );

  {
    if ( mpirank == 0 ){
      GetTime( timeSta );
      lapack::Syevd( 'V', 'U', width, XTX.Data(), width, eigValS.Data() );
      GetTime( timeEnd );
      iterMpirank0 = iterMpirank0 + 1;
      timeMpirank0 = timeMpirank0 + ( timeEnd - timeSta );
    }
  }

  GetTime( timeEnd1 );
  iterSyevd = iterSyevd + 1;
  timeSyevd = timeSyevd + ( timeEnd1 - timeSta1 );

  GetTime( timeSta );
  MPI_Bcast(XTX.Data(), width*width, MPI_DOUBLE, 0, mpi_comm);
  MPI_Bcast(eigValS.Data(), width, MPI_DOUBLE, 0, mpi_comm);
  GetTime( timeEnd );
  iterBcast = iterBcast + 2;
  timeBcast = timeBcast + ( timeEnd - timeSta );

  GetTime( timeSta );
  // X <- X*C
  blas::Gemm( 'N', 'N', heightLocal, width, width, 1.0, X.Data(),
      heightLocal, XTX.Data(), width, 0.0, Xtemp.Data(), heightLocal );
  GetTime( timeEnd );
  iterGemmN = iterGemmN + 1;
  timeGemmN = timeGemmN + ( timeEnd - timeSta );

  GetTime( timeSta );
  lapack::Lacpy( 'A', heightLocal, width, Xtemp.Data(), heightLocal,
      X.Data(), heightLocal );
  GetTime( timeEnd );
  iterCopy = iterCopy + 1;
  timeCopy = timeCopy + ( timeEnd - timeSta );


  GetTime( timeSta );
  // AX <- AX*C
  blas::Gemm( 'N', 'N', heightLocal, width, width, 1.0, AX.Data(),
      heightLocal, XTX.Data(), width, 0.0, Xtemp.Data(), heightLocal );
  GetTime( timeEnd );
  iterGemmN = iterGemmN + 1;
  timeGemmN = timeGemmN + ( timeEnd - timeSta );

  GetTime( timeSta );
  lapack::Lacpy( 'A', heightLocal, width, Xtemp.Data(), heightLocal,
      AX.Data(), heightLocal );
  GetTime( timeEnd );
  iterCopy = iterCopy + 1;
  timeCopy = timeCopy + ( timeEnd - timeSta );

  // Compute norms of individual eigenpairs 
  DblNumVec  resNormLocal ( width ); 
  DblNumVec  resNorm( width );

  GetTime( timeSta );
  for(Int j=0; j < width; j++){
    for(Int i=0; i < heightLocal; i++){
      Xtemp(i,j) = AX(i,j) - X(i,j)*eigValS(j);  
    }
  } 
  GetTime( timeEnd );
  iterOther = iterOther + 1;
  timeOther = timeOther + ( timeEnd - timeSta );

  statusOFS << "Time for Xtemp in PWDFT is " <<  timeEnd - timeSta  << std::endl << std::endl;

  SetValue( resNormLocal, 0.0 );
  GetTime( timeSta );
  for( Int k = 0; k < width; k++ ){
    resNormLocal(k) = Energy(DblNumVec(heightLocal, false, Xtemp.VecData(k)));
  }
  GetTime( timeEnd );
  iterOther = iterOther + 1;
  timeOther = timeOther + ( timeEnd - timeSta );

  statusOFS << "Time for resNorm in PWDFT is " <<  timeEnd - timeSta  << std::endl << std::endl;

  SetValue( resNorm, 0.0 );
  MPI_Allreduce( resNormLocal.Data(), resNorm.Data(), width, MPI_DOUBLE, 
      MPI_SUM, mpi_comm );

  if ( mpirank == 0 ){
    GetTime( timeSta );
    for( Int k = 0; k < width; k++ ){
      //            resNorm(k) = std::sqrt( resNorm(k) ) / std::max( 1.0, std::abs( XTX(k,k) ) );
      resNorm(k) = std::sqrt( resNorm(k) ) / std::max( 1.0, std::abs( eigValS(k) ) );
    }
    GetTime( timeEnd );
    timeMpirank0 = timeMpirank0 + ( timeEnd - timeSta );
  }
  GetTime( timeSta );
  MPI_Bcast(resNorm.Data(), width, MPI_DOUBLE, 0, mpi_comm);
  GetTime( timeEnd );
  iterBcast = iterBcast + 1;
  timeBcast = timeBcast + ( timeEnd - timeSta );

  GetTime( timeSta );
  resMax = *(std::max_element( resNorm.Data(), resNorm.Data() + numEig ) );
  resMin = *(std::min_element( resNorm.Data(), resNorm.Data() + numEig ) );
  GetTime( timeEnd );
  iterOther = iterOther + 2;
  timeOther = timeOther + ( timeEnd - timeSta );

  statusOFS << "Time for resMax and resMin in PWDFT is " <<  timeEnd - timeSta  << std::endl << std::endl;

#if ( _DEBUGlevel_ >= 1 )
  statusOFS << "resNorm = " << resNorm << std::endl;
  statusOFS << "eigValS = " << eigValS << std::endl;
  statusOFS << "maxRes  = " << resMax  << std::endl;
  statusOFS << "minRes  = " << resMin  << std::endl;
#endif



#if ( _DEBUGlevel_ >= 2 )

  GetTime( timeSta );
  blas::Gemm( 'T', 'N', width, width, heightLocal, 1.0, X.Data(), 
      heightLocal, X.Data(), heightLocal, 0.0, XTXtemp1.Data(), width );
  GetTime( timeEnd );
  iterGemmT = iterGemmT + 1;
  timeGemmT = timeGemmT + ( timeEnd - timeSta );
  GetTime( timeSta );
  SetValue( XTX, 0.0 );
  MPI_Allreduce( XTXtemp1.Data(), XTX.Data(), width*width, MPI_DOUBLE, MPI_SUM, mpi_comm );
  GetTime( timeEnd );
  iterAllreduce = iterAllreduce + 1;
  timeAllreduce = timeAllreduce + ( timeEnd - timeSta );

  statusOFS << "After the PPCG, XTX = " << XTX << std::endl;

#endif

  // Save the eigenvalues and eigenvectors back to the eigensolver data
  // structure

  eigVal_ = DblNumVec( width, true, eigValS.Data() );
  resVal_ = resNorm;

  GetTime( timeSta );
  for( Int j = 0; j < width; j++ ){ 
    for( Int i = 0; i < heightLocal; i++ ){
      recvbuf[recvk(i, j)] = X(i, j);
    }
  }
  MPI_Alltoallv( &recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE, 
      &sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE, mpi_comm );
  for( Int j = 0; j < widthLocal; j++ ){ 
    for( Int i = 0; i < height; i++ ){
      Xcol(i, j) = sendbuf[sendk(i, j)]; 
    }
  }
  GetTime( timeEnd );
  iterAlltoallv = iterAlltoallv + 1;
  timeAlltoallv = timeAlltoallv + ( timeEnd - timeSta );

  GetTime( timeSta );
  lapack::Lacpy( 'A', height, widthLocal, Xcol.Data(), height, 
      psiPtr_->Wavefun().Data(), height );
  GetTime( timeEnd );
  iterCopy = iterCopy + 1;
  timeCopy = timeCopy + ( timeEnd - timeSta );

  // REPORT ACTUAL EIGENRESIDUAL NORMS?
  statusOFS << std::endl << "After " << iter 
    << " PPCG iterations the min res norm is " 
    << resMin << ". The max res norm is " << resMax << std::endl << std::endl;

  GetTime( timeEnd2 );

#if ( _DEBUGlevel_ >= 0 )
  statusOFS << "Time for iterGemmT        = " << iterGemmT           << "  timeGemmT        = " << timeGemmT << std::endl;
  statusOFS << "Time for iterGemmN        = " << iterGemmN           << "  timeGemmN        = " << timeGemmN << std::endl;
  statusOFS << "Time for iterBcast        = " << iterBcast           << "  timeBcast        = " << timeBcast << std::endl;
  statusOFS << "Time for iterAllreduce    = " << iterAllreduce       << "  timeAllreduce    = " << timeAllreduce << std::endl;
  statusOFS << "Time for iterAlltoallv    = " << iterAlltoallv       << "  timeAlltoallv    = " << timeAlltoallv << std::endl;
  statusOFS << "Time for iterAlltoallvMap = " << iterAlltoallvMap    << "  timeAlltoallvMap = " << timeAlltoallvMap << std::endl;
  statusOFS << "Time for iterSpinor       = " << iterSpinor          << "  timeSpinor       = " << timeSpinor << std::endl;
  statusOFS << "Time for iterTrsm         = " << iterTrsm            << "  timeTrsm         = " << timeTrsm << std::endl;
  statusOFS << "Time for iterPotrf        = " << iterPotrf           << "  timePotrf        = " << timePotrf << std::endl;
  statusOFS << "Time for iterSyevd        = " << iterSyevd           << "  timeSyevd        = " << timeSyevd << std::endl;
  statusOFS << "Time for iterSygvd        = " << iterSygvd           << "  timeSygvd        = " << timeSygvd << std::endl;
  statusOFS << "Time for iterMpirank0     = " << iterMpirank0        << "  timeMpirank0     = " << timeMpirank0 << std::endl;
  statusOFS << "Time for iterSweepT       = " << iterSweepT          << "  timeSweepT       = " << timeSweepT << std::endl;
  statusOFS << "Time for iterCopy         = " << iterCopy            << "  timeCopy         = " << timeCopy << std::endl;
  statusOFS << "Time for iterOther        = " << iterOther           << "  timeOther        = " << timeOther << std::endl;
  statusOFS << "Time for PPCG in PWDFT is " <<  timeEnd2 - timeSta2  << std::endl << std::endl;
#endif
  // Close record file
  //statusOFS.close( ss.str().c_str() );

  return ;
}         // -----  end of method EigenSolver::PPCGSolveReal  ----- 

} // namespace PPCG

