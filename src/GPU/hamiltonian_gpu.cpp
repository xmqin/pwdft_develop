/// @file hamiltonian_gpu.cpp
/// @brief GPU-ified functions in class Hamiltonian.
/// @date 2024-02-04
#include  "hamiltonian.hpp"
#include  "blas.hpp"
#include  "lapack.hpp"

namespace pwdft{

using namespace pwdft::PseudoComponent;
using namespace pwdft::DensityComponent;
using namespace pwdft::esdf;
using namespace pwdft::SpinTwo;

void
KohnSham::CalculateDensityGPU ( const Spinor &psi, const DblNumVec &occrate, Real &val, Fourier &fft )
{
  SetValue( density_, 0.0 );
  SetValue( spindensity_, 0.0 );

  Real vol  = domain_.Volume();
  Int ntotFine  = fft.domain.NumGridTotalFine();
  Int ntotFine2 = ntotFine * numDensityComponent_;

  MPI_Barrier(domain_.comm);

  DblNumMat   densityLocal;
  densityLocal.Resize( ntotFine, numDensityComponent_ );
  SetValue( densityLocal, 0.0 );

  Int ntot  = psi.NumGridTotal();
  Int ncom  = psi.NumComponent();
  Int nocc  = psi.NumState();
  Int nocc_total = psi.NumStateTotal();

  if( numDensityComponent_ == 4 ){
    ErrorHandling("The real-value version of PWDFT does not support spin-noncollinear calculation !");
  }
  else if( numDensityComponent_ == 2 ){
    nocc /= 2;
    nocc_total /= 2;
  }

  Real fac;
 
  CpxNumVec psi_temp(ntot);

  // Data in GPU device
  int *dev_idxFineGrid = (int*) cuda_malloc(sizeof(int) * ntot); 
  cuda_memcpy_CPU2GPU(dev_idxFineGrid, fft.idxFineGrid.Data(), sizeof(Int) *ntot);

  cuCpxNumVec cu_psi(ntot);
  cuCpxNumVec cu_psi_out(ntot);
  cuCpxNumVec cu_psi_fine_out(ntotFine);
  cuCpxNumVec cu_psi_fine(ntotFine);
  cuDblNumMat cu_density(ntotFine, numDensityComponent_);
  cuDblNumMat cu_den(ntotFine, numDensityComponent_);
  cuDblNumMat cu_spinden(ntotFine, numDensityComponent_);

  cuda_setValue( cu_density.Data(), 0.0, ntotFine2);
  cuDoubleComplex zero; zero.x = 0.0; zero.y = 0.0;

  for( Int k = 0; k < nocc; k++ ){
    for( Int j = 0; j < numDensityComponent_; j++ ){
      for( Int i = 0; i < ntot; i++ ){
        psi_temp(i) = Complex( psi.Wavefun(i,RHO,k+j*nocc), 0.0 );
      }

      cuda_memcpy_CPU2GPU(cu_psi.Data(), psi_temp.Data(), sizeof(cuDoubleComplex)*ntot);
      
      cuFFTExecuteForward2( fft, fft.cuPlanC2C[0], 0, cu_psi, cu_psi_out );

      cuDoubleComplex zero_cucpx; zero_cucpx.x = 0.0; zero_cucpx.y = 0.0;
      cuda_setValue(reinterpret_cast<cuDoubleComplex*>(cu_psi_fine_out.Data()), 
          zero_cucpx , ntotFine);
   
      Real fac = sqrt( double(ntot) / double(ntotFine) );
      cuda_interpolate_wf_C2F( reinterpret_cast<cuDoubleComplex*>(cu_psi_out.Data()),
                               reinterpret_cast<cuDoubleComplex*>(cu_psi_fine_out.Data()),
                               dev_idxFineGrid,
                               ntot,
                               fac);
      cuFFTExecuteInverse(fft, fft.cuPlanC2CFine[0], 1, cu_psi_fine_out, cu_psi_fine);
      fac = numSpin_ * occrate(psi.WavefunIdx(k)+j*nocc_total);
      cuda_XTX( cu_psi_fine.Data(), cu_den.VecData(j), ntotFine);
      cublas::Axpy( ntotFine, &fac, cu_den.VecData(j), 1, cu_density.VecData(j), 1);
    }
  }

  cuda_free(dev_idxFineGrid);
#ifdef GPUDIRECT
  if( numDensityComponent_ == 1 ){ 
    mpi::Allreduce( cu_density.Data(), cu_den.Data(), ntotFine2, MPI_SUM, domain_.comm );
  }
  else{
    mpi::Allreduce( cu_density.Data(), cu_spinden.Data(), ntotFine2, MPI_SUM, domain_.comm );
  }
#else
  cuda_memcpy_GPU2CPU( densityLocal.Data(), cu_density.Data(), ntotFine2*sizeof(double));
  if( numDensityComponent_ == 1 ){
    mpi::Allreduce( densityLocal.Data(), density_.Data(), ntotFine2, MPI_SUM, domain_.comm );
    cuda_memcpy_CPU2GPU( cu_den.Data(), density_.Data(), ntotFine2*sizeof(double));
  }
  else{
    mpi::Allreduce( densityLocal.Data(), spindensity_.Data(), ntotFine2, MPI_SUM, domain_.comm );
    cuda_memcpy_CPU2GPU( cu_spinden.Data(), spindensity_.Data(), ntotFine2*sizeof(double));
  }
#endif

  if( numDensityComponent_ == 2 ){
    cuda_set_vector( cu_den.VecData(0), cu_spinden.VecData(0), ntotFine );
    cuda_set_vector( cu_den.VecData(1), cu_spinden.VecData(0), ntotFine );

    Real one = 1.0, minus_one = -1.0;
    cublas::Axpy( ntotFine, &one, cu_spinden.VecData(1), 1, cu_den.VecData(0), 1);
    cublas::Axpy( ntotFine, &minus_one, cu_spinden.VecData(1), 1, cu_den.VecData(1), 1);
  }

  double * val_dev = (double*) cuda_malloc( sizeof(double) );
  val = 0.0; // sum of density
  cuda_reduce( cu_den.VecData(RHO), val_dev, 1, ntotFine);
  cuda_memcpy_GPU2CPU( &val, val_dev, sizeof(double));
  Real val1 = val;
  Real temp = (numSpin_ * Real(numOccupiedState_) * Real(ntotFine)) / ( vol * val );
  cublas::Scal( ntotFine2, &temp, cu_den.Data(), 1 );
  cuda_memcpy_GPU2CPU( density_.Data(), cu_den.Data(), ntotFine2*sizeof(double));

  if( numDensityComponent_ == 2 ){
    cublas::Scal( ntotFine2, &temp, cu_spinden.Data(), 1 );
    cuda_memcpy_GPU2CPU( spindensity_.Data(), cu_spinden.Data(), ntotFine2*sizeof(double));
  }

  cuda_set_vector( cu_density.Data(), cu_den.Data(), ntotFine2);
  temp = vol / ntotFine;
  cublas::Scal( ntotFine2, &temp, cu_density.Data(), 1 );

  cuda_reduce( cu_density.VecData(RHO), val_dev, 1, ntotFine);
  cuda_memcpy_GPU2CPU( &val, val_dev, sizeof(double));
  Real val2 = val;

  cuda_free(val_dev);

  // Calculate magnetization
  if( numDensityComponent_ == 2 ){
    Real magnet = 0.0;
    Real magabs = 0.0;
    for( Int i = 0; i < ntotFine; i++ ){
      magnet += ( spindensity_(i, UP) - spindensity_(i,DN) ) * vol / ntotFine;
      magabs += ( std::abs( spindensity_(i, UP) - spindensity_(i,DN) ) ) * vol / ntotFine;
    }
    statusOFS << "The net magnetization moment         = " << magnet << std::endl;
    statusOFS << "The absolute magnetization moment    = " << magabs << std::endl;
  }

  return;
}         // -----  end of method KohnSham::CalculateDensityGPU ----- 

void
KohnSham::MultSpinorGPU    ( Spinor& psi, cuNumTns<Real>& a3, Fourier& fft )
{
  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

  Int nspin     = numDensityComponent_;
  Int ntot      = fft.domain.NumGridTotal();
  Int ntotFine  = fft.domain.NumGridTotalFine();
  Int numStateTotal = psi.NumStateTotal();
  Int numStateLocal = psi.NumState();
  NumTns<Real>& wavefun = psi.Wavefun();
  Int ncom = wavefun.n();

  Int ntotR2C = fft.numGridTotalR2C; 

  DblNumVec vtot = DblNumVec( ntotFine, false, vtot_.VecData( spinswitch_ ) );
  psi.AddMultSpinorFineR2CGPU( fft, ekin_, vtot, pseudo_, a3 );

  // adding up the Hybrid part in the GPU
  // Note now, the psi.data is the GPU data. and a3.data is also in GPU.
  if( isHybrid_ && isEXXActive_ ){

    if( esdfParam.isHybridACE ){
      // Convert the column partition to row partition
      Int numStateBlocksize = numStateTotal / mpisize;
      Int ntotBlocksize = ntot / mpisize;

      Int numStateLocal = numStateBlocksize;
      Int ntotLocal = ntotBlocksize;

      if(mpirank < (numStateTotal % mpisize)){
        numStateLocal = numStateBlocksize + 1;
      }

      if(mpirank < (ntot % mpisize)){
        ntotLocal = ntotBlocksize + 1;
      }

      // copy the GPU data to CPU.
      DblNumMat psiCol( ntot, numStateLocal );
      DblNumMat psiRow( ntotLocal, numStateTotal );

      cuda_memcpy_GPU2CPU( psiCol.Data(), psi.cuWavefun().Data(), ntot*numStateLocal*sizeof(Real) );

      // for the Project VexxProj
      DblNumMat vexxProjCol( ntot, numStateLocal );
      DblNumMat vexxProjRow( ntotLocal, numStateTotal );
      if( nspin == 1 ){
        lapack::Lacpy( 'A', ntot, numStateLocal, vexxProj_.Data(), ntot, vexxProjCol.Data(), ntot );
      } 
      else if( nspin == 2 ){
        if( spinswitch_ == 0 )
          lapack::Lacpy( 'A', ntot, numStateLocal, UpvexxProj_.Data(),
            ntot, vexxProjCol.Data(), ntot );
        else
          lapack::Lacpy( 'A', ntot, numStateLocal, DnvexxProj_.Data(),
            ntot, vexxProjCol.Data(), ntot );
      }

      // MPI_Alltoall for the data redistribution.
      AlltoallForward (psiCol, psiRow, domain_.comm);
      AlltoallForward (vexxProjCol, vexxProjRow, domain_.comm);

      // GPU data for the G-para
      cuDblNumMat cu_vexxProjRow ( ntotLocal, numStateTotal );
      cuDblNumMat cu_psiRow ( ntotLocal, numStateTotal );
      cuDblNumMat cu_MTemp( numStateTotal, numStateTotal );
      DblNumMat MTemp( numStateTotal, numStateTotal );

      // Copy data from CPU to GPU.
      cuda_memcpy_CPU2GPU( cu_psiRow.Data(), psiRow.Data(), numStateTotal*ntotLocal*sizeof(Real) );
      cuda_memcpy_CPU2GPU( cu_vexxProjRow.Data(), vexxProjRow.Data(), numStateTotal*ntotLocal*sizeof(Real) );

      Real one = 1.0;
      Real minus_one = -1.0;
      Real zero = 0.0;

      // GPU DGEMM calculation
      cublas::Gemm( CUBLAS_OP_T, CUBLAS_OP_N, numStateTotal, numStateTotal, ntotLocal,
                    &one, cu_vexxProjRow.Data(), ntotLocal,
                    cu_psiRow.Data(), ntotLocal, &zero,
                    cu_MTemp.Data(), numStateTotal );
      cuda_memcpy_GPU2CPU( MTemp.Data(), cu_MTemp.Data(), numStateTotal*numStateTotal*sizeof(Real) );
      DblNumMat M(numStateTotal, numStateTotal);
      MPI_Allreduce( MTemp.Data(), M.Data(), numStateTotal * numStateTotal, MPI_DOUBLE, MPI_SUM, domain_.comm );

      // copy from CPU to GPU
      cuda_memcpy_CPU2GPU(  cu_MTemp.Data(), M.Data(), numStateTotal*numStateTotal*sizeof(Real) );

      cuDblNumMat cu_a3Row( ntotLocal, numStateTotal );
      DblNumMat a3Row( ntotLocal, numStateTotal );

      cublas::Gemm( CUBLAS_OP_N, CUBLAS_OP_N, ntotLocal, numStateTotal, numStateTotal,
                    &minus_one, cu_vexxProjRow.Data(), ntotLocal,
                    cu_MTemp.Data(), numStateTotal, &zero,
                    cu_a3Row.Data(), ntotLocal );

      cuda_memcpy_GPU2CPU( a3Row.Data(), cu_a3Row.Data(), numStateTotal*ntotLocal*sizeof(Real) );

      // a3Row to a3Col
      DblNumMat a3Col( ntot, numStateLocal );
      cuDblNumMat cu_a3Col( ntot, numStateLocal );
      AlltoallBackward (a3Row, a3Col, domain_.comm);

      // Copy a3Col to GPU.
      cuda_memcpy_CPU2GPU( cu_a3Col.Data(), a3Col.Data(), numStateLocal*ntot*sizeof(Real) );

      // do the matrix addition.
      cuda_DMatrix_Add( a3.Data(), cu_a3Col.Data(), ntot, numStateLocal);
    }
    else{
      ErrorHandling(" GPU does not support normal HSE, try ACE");
    }
  } // ---- if( isHybrid_ && isEXXActive_ ) ----

  return;
}         // -----  end of method KohnSham::MultSpinorGPU  ----- 

void
KohnSham::ACEOperatorGPU ( cuDblNumMat& cu_psi, Fourier& fft, cuDblNumMat& cu_Hpsi, const int spinswitch )
{
  // 1. the projector is in a Row Parallel fashion
  // 2. the projector is in GPU.
  // 3. the AX (H*psi) is in the GPU

  // in here we perform: 
  // M = W'*AX
  // reduece M
  // AX = AX + W*M

  Int nspin = numDensityComponent_;

  if( isHybrid_ && isEXXActive_ ){

    if( esdfParam.isHybridACE ){

      int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
      int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

      Int ntot      = fft.domain.NumGridTotal();
      Int ntotFine  = fft.domain.NumGridTotalFine();
      Int numStateTotal = cu_psi.n();

      Int ntotBlocksize = ntot / mpisize;
      Int ntotLocal = ntotBlocksize;
      if(mpirank < (ntot % mpisize)){
        ntotLocal = ntotBlocksize + 1;
      }

      Real one = 1.0;
      Real minus_one = -1.0;
      Real zero = 0.0;

      DblNumMat MTemp( numStateTotal, numStateTotal );
      cuDblNumMat cu_MTemp( numStateTotal, numStateTotal );
     
      cuDblNumMat& cu_vexxProj = cu_vexxProj_;
      if( nspin == 2 ){
        if( spinswitch == 0 )
          cu_vexxProj = cu_UpvexxProj_;
        else
          cu_vexxProj = cu_DnvexxProj_;
      }

      cublas::Gemm( CUBLAS_OP_T, CUBLAS_OP_N, numStateTotal, numStateTotal, ntotLocal,
                    &one, cu_vexxProj.Data(), ntotLocal,
                    cu_psi.Data(), ntotLocal, &zero,
                    cu_MTemp.Data(), numStateTotal );

      cuda_memcpy_GPU2CPU( MTemp.Data(), cu_MTemp.Data(), numStateTotal*numStateTotal*sizeof(Real) );

      DblNumMat M(numStateTotal, numStateTotal);
      MPI_Allreduce( MTemp.Data(), M.Data(), numStateTotal * numStateTotal, MPI_DOUBLE, MPI_SUM, domain_.comm );

      cuda_memcpy_CPU2GPU(  cu_MTemp.Data(), M.Data(), numStateTotal*numStateTotal*sizeof(Real) );

      cublas::Gemm( CUBLAS_OP_N, CUBLAS_OP_N, ntotLocal, numStateTotal, numStateTotal,
                    &minus_one, cu_vexxProj.Data(), ntotLocal,
                    cu_MTemp.Data(), numStateTotal, &one,
                    cu_Hpsi.Data(), ntotLocal );
    }
  }
}         // -----  end of method KohnSham::ACEOperatorGPU  ----- 

void
KohnSham::CalculateVexxACEGPU ( Spinor& psi, Fourier& fft )
{
  MPI_Barrier(domain_.comm); 
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

  bool spherecut = esdfParam.isUseSphereCut;

  Int nspin     = domain_.numSpinComponent;
  Int ncom      = ( nspin == 4 ) ? 2 : 1;
  Int numACE    = ( nspin == 2 ) ? 2 : 1;
  Int ntot;
  if( spherecut )
    ntot = domain_.numGridSphere;
  else
    ntot = domain_.NumGridTotal();

  Int numStateTotal = psi.NumStateTotal();
  Int numStateLocal = psi.NumState();
  if( nspin == 2 )
  {
    numStateTotal /= 2;
    numStateLocal /= 2;
  }
  
  cuNumTns<Real>  cu_vexxPsi( ntot, ncom, numStateLocal );
  NumTns<Real>  vexxPsi( ntot, ncom, numStateLocal );

  DblNumVec occSpin;
  for( Int ispin = 0; ispin < numACE; ispin++ ){
    if( nspin == 2 ){
      occSpin.Resize( numStateTotal );
      blas::Copy( numStateTotal, &(occupationRate_[ispin*numStateTotal]), 1, occSpin.Data(), 1 );
    }

    Spinor psiTemp( fft.domain, ncom, numStateTotal, false,
        psi.Wavefun().VecData(0,ispin*numStateLocal) );

    cuda_setValue( cu_vexxPsi.Data(), 0.0, ntot*ncom*numStateLocal);
    if( nspin == 1 ){
      psiTemp.AddMultSpinorEXXGPU( fft, phiEXX_, exxgkkR2C_,
          exxFraction_, nspin, occupationRate_, cu_vexxPsi );
    }
    else if( nspin == 2 ){
      if( ispin == 0 )
        psiTemp.AddMultSpinorEXXGPU( fft, UpphiEXX_, exxgkkR2C_,
            exxFraction_, nspin, occSpin, cu_vexxPsi );
      else
        psiTemp.AddMultSpinorEXXGPU( fft, DnphiEXX_, exxgkkR2C_,
            exxFraction_, nspin, occSpin, cu_vexxPsi );
    }

    DblNumMat  M(numStateTotal, numStateTotal);

    Int numStateBlocksize = numStateTotal / mpisize;
    Int ntotBlocksize = ntot / mpisize;

    Int numStateLocal = numStateBlocksize;
    Int ntotLocal = ntotBlocksize;

    if(mpirank < (numStateTotal % mpisize)){
      numStateLocal = numStateBlocksize + 1;
    }

    if(mpirank < (ntot % mpisize)){
      ntotLocal = ntotBlocksize + 1;
    }
 
    DblNumMat localPsiRow( ntotLocal, numStateTotal );
    DblNumMat localVexxPsiRow( ntotLocal, numStateTotal );
    DblNumMat localPsiCol( ntot, numStateLocal );

    cuDblNumMat cu_temp( ntot, numStateLocal, false, cu_vexxPsi.Data() );

    cuDblNumMat& cu_vexxProj = cu_vexxProj_; 
    if( nspin == 2 ){
      if( ispin == 0 )
        cu_vexxProj = cu_UpvexxProj_;
      else
        cu_vexxProj = cu_DnvexxProj_;
    }

    cu_vexxProj.Resize( ntotLocal, numStateTotal );
    GPU_AlltoallForward (cu_temp, cu_vexxProj, domain_.comm);

    cuda_memcpy_CPU2GPU( cu_temp.Data(), psiTemp.Wavefun().Data(), ntot*numStateLocal*sizeof(Real));
    cuDblNumMat cu_localPsiRow( ntotLocal, numStateTotal);
    GPU_AlltoallForward (cu_temp, cu_localPsiRow, domain_.comm);

    DblNumMat MTemp( numStateTotal, numStateTotal );
    cuDblNumMat cu_MTemp( numStateTotal, numStateTotal );

    Real minus_one = -1.0;
    Real zero =  0.0;
    Real one  =  1.0;

    cublas::Gemm( CUBLAS_OP_T, CUBLAS_OP_N, numStateTotal, numStateTotal, ntotLocal,
                  &minus_one, cu_localPsiRow.Data(), ntotLocal,
                  cu_vexxProj.Data(), ntotLocal, &zero,
                  cu_MTemp.Data(), numStateTotal );

    cu_MTemp.CopyTo(MTemp);

    MPI_Allreduce( MTemp.Data(), M.Data(), numStateTotal * numStateTotal, MPI_DOUBLE, MPI_SUM, domain_.comm );

    lapack::Potrf( 'L', numStateTotal, M.Data(), numStateTotal );
    cu_MTemp.CopyFrom(M);

    cublas::Trsm( CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                  ntotLocal, numStateTotal, &one, cu_MTemp.Data(), numStateTotal, cu_vexxProj.Data(),
                  ntotLocal);

    cu_localPsiRow.Resize( ntot, numStateLocal ); // use this as a column distribution data.

    GPU_AlltoallBackward (cu_vexxProj, cu_localPsiRow, domain_.comm);

    if( nspin == 1 ){
      vexxProj_.Resize( ntot, numStateLocal );
      cu_localPsiRow.CopyTo( vexxProj_ );
    }
    else{
      if( ispin == 0 ){
        UpvexxProj_.Resize( ntot, numStateLocal );
        cu_localPsiRow.CopyTo( UpvexxProj_ );
      }
      else{
        DnvexxProj_.Resize( ntot, numStateLocal );
        cu_localPsiRow.CopyTo( DnvexxProj_ );
      }
    }
  } // for (ispin)
    
  return;
}         // -----  end of method KohnSham::CalculateVexxACEGPU  ----- 

} // namespace pwdft


