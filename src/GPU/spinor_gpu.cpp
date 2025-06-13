/// @file spinor_gpu.cpp
/// @brief GPU-ified functions in class Spinor.
/// @date 2024-02-04
#include  "spinor.hpp"
#include  "utility.hpp"
#include  "blas.hpp"
#include  "lapack.hpp"
#include  "scalapack.hpp"
#include  "mpi_interf.hpp"

namespace pwdft{

using namespace pwdft::scalapack;
using namespace pwdft::PseudoComponent;
using namespace pwdft::DensityComponent;
using namespace pwdft::SpinTwo;
using namespace pwdft::esdf;

Spinor::Spinor ( const Domain &dm,
    const Int numComponent,
    const Int numStateTotal,
    const bool owndata,
    Real* data,
    bool isGPU )
{
  if(!isGPU || owndata)
    ErrorHandling(" GPU Spinor setup error.");
 
  this->SetupGPU( dm, numComponent, numStateTotal, owndata, data);
}         // -----  end of method Spinor::Spinor  -----

void Spinor::SetupGPU ( const Domain &dm,
    const Int numComponent,
    const Int numStateTotal,
    const bool owndata,
    Real* data )
{
  domain_  = dm;

  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

  if( esdfParam.isUseSphereCut == true )
    numGridTotal_ = domain_.numGridSphere;
  else
    numGridTotal_ = domain_.NumGridTotal();

  numStateTotal_ = numStateTotal;

  nblocksize_ = esdfParam.BlockSizeState;
  Int nblockTotal = ( numStateTotal_ + nblocksize_ - 1 ) / nblocksize_;
  Int nres = numStateTotal_ % nblocksize_;

  Int nblockbase = nblockTotal / mpisize;
  Int nblockres = nblockTotal % mpisize;

  Int nblockLocal = nblockbase;
  if( mpirank < nblockres ){
    nblockLocal = nblockLocal + 1;
  }

  if( nblockLocal == 0 ){
    numStateLocal_ = 0;
  }
  else{
    bool holdres = (mpirank == nblockres - 1 + ((mpisize - nblockres) / mpisize
        * mpisize) ) && (nres > 0);
    Int nlast = ( holdres == true ) ? nres : nblocksize_;
    numStateLocal_ = (nblockLocal - 1) * nblocksize_ + nlast;

    wavefunIdx_.Resize( numStateLocal_ );

    Int *idxPtr = wavefunIdx_.Data();
    for( Int ib = 0; ib < nblockLocal - 1; ib++ ){
      for( Int j = 0; j < nblocksize_; j++ ){
        *(idxPtr++) = (mpirank + ib * mpisize) * nblocksize_ + j;
      }
    }

    for( Int j = 0; j < nlast; j++ ){
      *(idxPtr++) = (mpirank + (nblockLocal - 1) * mpisize) * nblocksize_ + j;
    }
  }

  cu_wavefun_ = cuNumTns<Real>( numGridTotal_, numComponent, numStateLocal_,
      owndata, data );
}         // -----  end of method Spinor::SetupGPU  ----- 

void
Spinor::AddTeterPrecondGPU (Fourier* fftPtr, DblNumVec& teter, cuNumTns<Real>& a3)
{
  Fourier& fft = *fftPtr;
  if( !fftPtr->isInitialized ){
    ErrorHandling("Fourier is not prepared.");
  }
  Int ntot = cu_wavefun_.m();
  Int ncom = cu_wavefun_.n();
  Int nocc = cu_wavefun_.p();

  if( esdfParam.isUseSphereCut == false ){
    if( fft.domain.NumGridTotal() != ntot )
      ErrorHandling("Domain size does not match.");
  }
  else{
    if( fft.domain.numGridSphere != ntot )
      ErrorHandling("Domain size does not match.");
    if( fft.domain.NumGridTotal() < ntot )
      ErrorHandling("numGridSphere is larger than numGridTotal.");
  }

  Int ntothalf = fftPtr->numGridTotalR2C;
  
  cuDblNumVec cu_psi(ntot);
  cuDblNumVec cu_psi_out(2*ntothalf);

  if( !teter_gpu_flag ){
    dev_TeterPrecond = (double*) cuda_malloc( sizeof(Real) * ntothalf );
    cuda_memcpy_CPU2GPU(dev_TeterPrecond, teter.Data(), sizeof(Real)*ntothalf);
    teter_gpu_flag = true;
  }

  for (Int k=0; k<nocc; k++) {
    for (Int j=0; j<ncom; j++) {
      cuda_memcpy_GPU2GPU(cu_psi.Data(), cu_wavefun_.VecData(j,k), sizeof(Real)*ntot);

      cuFFTExecuteForward( fft, fft.cuPlanR2C[0], 0, cu_psi, cu_psi_out);

      cuda_teter( reinterpret_cast<cuDoubleComplex*>(cu_psi_out.Data()), dev_TeterPrecond, ntothalf);

      cuFFTExecuteInverse( fft, fft.cuPlanC2R[0], 0, cu_psi_out, cu_psi);

      cuda_memcpy_GPU2GPU(a3.VecData(j,k), cu_psi.Data(), ntot*sizeof(Real));
    }
  }

  return ;
}         // -----  end of method Spinor::AddTeterPrecondGPU ----- 

void
Spinor::AddTeterPrecondGPU (Fourier* fftPtr, DblNumVec& teter, NumTns<Real>& a3)
{
  Fourier& fft = *fftPtr;
  if( !fftPtr->isInitialized ){
    ErrorHandling("Fourier is not prepared.");
  }
  Int ntot = wavefun_.m();
  Int ncom = wavefun_.n();
  Int nocc = wavefun_.p();

  if( esdfParam.isUseSphereCut == false ){
    if( fft.domain.NumGridTotal() != ntot )
      ErrorHandling("Domain size does not match.");
  }
  else{
    if( fft.domain.numGridSphere != ntot )
      ErrorHandling("Domain size does not match.");
    if( fft.domain.NumGridTotal() < ntot )
      ErrorHandling("numGridSphere is larger than numGridTotal.");
  }

  Int ntothalf = fftPtr->numGridTotalR2C;

  cuDblNumVec cu_psi(ntot); 
  cuDblNumVec cu_psi_out(2*ntothalf);
  cuDblNumVec cu_TeterPrecond(ntothalf);
  cuda_memcpy_CPU2GPU(cu_TeterPrecond.Data(), teter.Data(), sizeof(Real)*ntothalf); 

  for (Int k=0; k<nocc; k++) {
    for (Int j=0; j<ncom; j++) {
      cuda_memcpy_CPU2GPU(cu_psi.Data(), wavefun_.VecData(j,k), sizeof(Real)*ntot);

      cuFFTExecuteForward( fft, fft.cuPlanR2C[0], 0, cu_psi, cu_psi_out);      

      cuda_teter( reinterpret_cast<cuDoubleComplex*>(cu_psi_out.Data()), cu_TeterPrecond.Data(), ntothalf);
      
      cuFFTExecuteInverse( fft, fft.cuPlanC2R[0], 0, cu_psi_out, cu_psi);
      
      cuda_memcpy_GPU2CPU(a3.VecData(j,k), cu_psi.Data(), ntot*sizeof(Real));
    }
  }

  return ;
}         // -----  end of method Spinor::AddTeterPrecondGPU ----- 

void
Spinor::AddMultSpinorFineR2CGPU ( Fourier& fft, DblNumVec& ekin, const DblNumVec& vtot,
    const std::vector<PseudoPot>& pseudo, cuNumTns<Real>& a3 )
{

  if( !fft.isInitialized ){
    ErrorHandling("Fourier is not prepared.");
  }

  Index3& numGrid = domain_.numGrid;
  Index3& numGridFine = domain_.numGridFine;
  Int ntot = cu_wavefun_.m();
  Int ncom = cu_wavefun_.n();
  Int numStateLocal = cu_wavefun_.p();
  Int ntotFine = domain_.NumGridTotalFine();
  Real vol = domain_.Volume();

  Int ntotR2C = fft.numGridTotalR2C;
  Int ntotR2CFine = fft.numGridTotalR2CFine;

  if( fft.domain.NumGridTotal() != ntot ){
    ErrorHandling("Domain size does not match");
  }

  // Temporary variable for saving wavefunction on a fine grid
  DblNumVec psiFine(ntotFine);
  DblNumVec psiUpdateFine(ntotFine);
  cuDblNumVec cu_psi(ntot);
  cuDblNumVec cu_psi_out(2*ntotR2C);
  cuDblNumVec cu_psi_fine(ntotFine);
  cuDblNumVec cu_psi_fineUpdate(ntotFine);
  cuDblNumVec cu_psi_fine_out(2*ntotR2CFine);

  if( NL_gpu_flag == false )
  {
    // get the total number of the nonlocal vector
    Int totNLNum = 0;
     totPart_gpu = 1;
     Int natm = pseudo.size();
     for (Int iatm=0; iatm<natm; iatm++) {
        Int nobt = pseudo[iatm].vnlList.size();
        totPart_gpu += nobt;
        for(Int iobt = 0; iobt < nobt; iobt++)
        {
              const SparseVec &vnlvecFine = pseudo[iatm].vnlList[iobt].first;
              const IntNumVec &ivFine = vnlvecFine.first;
              totNLNum += ivFine.m();
        }
    }
    DblNumVec NLvecFine(totNLNum);
    IntNumVec NLindex(totNLNum);
    IntNumVec NLpart (totPart_gpu);
    DblNumVec atom_weight(totPart_gpu);

    Int index = 0;
    Int ipart = 0;
    for (Int iatm=0; iatm<natm; iatm++) {
      Int nobt = pseudo[iatm].vnlList.size();
      for(Int iobt = 0; iobt < nobt; iobt++)
      {
        const Real       vnlwgt = pseudo[iatm].vnlList[iobt].second;
        const SparseVec &vnlvecFine = pseudo[iatm].vnlList[iobt].first;
        const IntNumVec &ivFine = vnlvecFine.first;
        const DblNumMat &dvFine = vnlvecFine.second;
        const Int    *ivFineptr = ivFine.Data();
        const Real   *dvFineptr = dvFine.VecData(VAL);
        atom_weight(ipart) = vnlwgt *vol/Real(ntotFine);

        NLpart(ipart++) = index;
        for(Int i = 0; i < ivFine.m(); i++)
        {
          NLvecFine(index)  = *(dvFineptr++);
          NLindex(index++)  = *(ivFineptr++);
        }
      }
    }

    NLpart(ipart) = index;
    dev_NLvecFine   = ( double*) cuda_malloc ( sizeof(double) * totNLNum );
    dev_NLindex     = ( int*   ) cuda_malloc ( sizeof(int )   * totNLNum );
    dev_NLpart      = ( int*   ) cuda_malloc ( sizeof(int )   * totPart_gpu );
    dev_atom_weight = ( double*) cuda_malloc ( sizeof(double) * totPart_gpu );
    dev_temp_weight = ( double*) cuda_malloc ( sizeof(double) * totPart_gpu );

    dev_idxFineGridR2C = ( int*) cuda_malloc ( sizeof(int   ) * ntotR2C );
    dev_gkkR2C      = ( double*) cuda_malloc ( sizeof(double) * ntotR2C );
    dev_vtot        = ( double*) cuda_malloc ( sizeof(double) * ntotFine);

    cuda_memcpy_CPU2GPU( dev_NLvecFine,   NLvecFine.Data(),   totNLNum * sizeof(double) );
    cuda_memcpy_CPU2GPU( dev_atom_weight, atom_weight.Data(), totPart_gpu* sizeof(double) );
    cuda_memcpy_CPU2GPU( dev_NLindex,     NLindex.Data(),     totNLNum * sizeof(int) );
    cuda_memcpy_CPU2GPU( dev_NLpart ,     NLpart.Data(),      totPart_gpu  * sizeof(int) );

    cuda_memcpy_CPU2GPU(dev_idxFineGridR2C, fft.idxFineGridR2C.Data(), sizeof(Int) *ntotR2C); 
    cuda_memcpy_CPU2GPU(dev_gkkR2C, ekin.Data(), sizeof(Real) *ntotR2C);
    cuda_memcpy_CPU2GPU(dev_vtot, vtot.Data(), sizeof(Real) *ntotFine);

    NL_gpu_flag = true;
    vtot_gpu_flag = true;
  }

  if( !vtot_gpu_flag) {
    cuda_memcpy_CPU2GPU(dev_vtot, vtot.Data(), sizeof(Real) *ntotFine);
    vtot_gpu_flag = true;
  }

  for (Int k=0; k<numStateLocal; k++) {
    for (Int j=0; j<ncom; j++) {
      cuda_memcpy_GPU2GPU(cu_psi.Data(), cu_wavefun_.VecData(j,k), sizeof(Real)*ntot);
      cuFFTExecuteForward( fft, fft.cuPlanR2C[0], 0, cu_psi, cu_psi_out);
      // 1. coarse to fine
      SetValue(cu_psi_fine_out, 0.0);
      Real fac = sqrt( double(ntot) / double(ntotFine) );
      cuda_interpolate_wf_C2F( reinterpret_cast<cuDoubleComplex*>(cu_psi_out.Data()),
                               reinterpret_cast<cuDoubleComplex*>(cu_psi_fine_out.Data()),
                               dev_idxFineGridR2C,
                               ntotR2C,
                               fac);
                              
      cuFFTExecuteInverse(fft, fft.cuPlanC2RFine[0], 1, cu_psi_fine_out, cu_psi_fine);                         
      cuda_memcpy_GPU2GPU(cu_psi_fineUpdate.Data(), cu_psi_fine.Data(), sizeof(Real)*ntotFine);

      cuda_vtot( cu_psi_fineUpdate.Data(), dev_vtot, ntotFine);

      cuda_calculate_nonlocal(cu_psi_fineUpdate.Data(), cu_psi_fine.Data(), dev_NLvecFine,
        dev_NLindex, dev_NLpart, dev_atom_weight, dev_temp_weight, totPart_gpu-1);

      cuda_laplacian(  reinterpret_cast<cuDoubleComplex*>( cu_psi_out.Data()),
                       dev_gkkR2C,
                       ntotR2C);
      cuda_memcpy_GPU2GPU(cu_psi_fine.Data(), cu_psi_fineUpdate.Data(), sizeof(Real)*ntotFine);
      cuFFTExecuteForward(fft, fft.cuPlanR2CFine[0], 1, cu_psi_fine, cu_psi_fine_out);
      // 2. fine to coarse
      fac = sqrt( double(ntotFine) / double(ntot) );
      cuda_interpolate_wf_F2C( reinterpret_cast<cuDoubleComplex*>(cu_psi_fine_out.Data()),
                               reinterpret_cast<cuDoubleComplex*>(cu_psi_out.Data()),
                               dev_idxFineGridR2C,
                               ntotR2C,
                               fac);
      cuFFTExecuteInverse( fft, fft.cuPlanC2R[0], 0, cu_psi_out, cu_psi);
      cuda_memcpy_GPU2GPU(a3.VecData(j,k), cu_psi.Data(), sizeof(Real)*ntot);
    } // j++
  } // k++

  return;
}         // -----  end of method Spinor::AddMultSpinorFineR2CGPU ----- 

void Spinor::AddMultSpinorEXXGPU ( Fourier& fft,
    const NumTns<Real>& phi,
    const DblNumVec& exxgkkR2C,
    Real  exxFraction,
    Real  numSpin,
    const DblNumVec& occupationRate,
    cuNumTns<Real>& a3 )
{
  if( !fft.isInitialized ){
    ErrorHandling("Fourier is not prepared.");
  }

  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

  Index3& numGrid = domain_.numGrid;
  Index3& numGridFine = domain_.numGridFine;

  Int ntot     = domain_.NumGridTotal();
  Int ntotFine = domain_.NumGridTotalFine();
  Int ntotR2C = fft.numGridTotalR2C;
  Int ntotR2CFine = fft.numGridTotalR2CFine;
  Int ncom = wavefun_.n();
  Int numStateLocal = wavefun_.p();
  Int numStateTotal = numStateTotal_;

  Int ncomPhi = phi.n();

  Real vol = domain_.Volume();

  if( ncomPhi != 1 || ncom != 1 ){
    ErrorHandling("Spin-noncollinear case is not supproted by real-value version PWDFT.");
  }

  if( fft.domain.NumGridTotal() != ntot ){
    ErrorHandling("Domain size does not match.");
  }

  // Temporary variable for saving wavefunction on a fine grid
  DblNumVec phiTemp(ntot);

  Int numStateLocalTemp;

  cuDblNumVec cu_phiTemp(ntot);
  cuDblNumVec cu_psi(ntot);
  cuDblNumVec cu_psi_out(2*ntotR2C);
  cuDblNumVec cu_exxgkkR2C(ntotR2C);
  cuDblNumMat cu_wave(ntot, numStateLocal);

  cuda_memcpy_CPU2GPU(cu_exxgkkR2C.Data(), exxgkkR2C.Data(), sizeof(Real)*ntotR2C);

  cuda_memcpy_CPU2GPU(cu_wave.Data(), wavefun_.Data(), sizeof(Real)* numStateLocal * ntot );

  for( Int iproc = 0; iproc < mpisize; iproc++ ){

    if( iproc == mpirank )
      numStateLocalTemp = numStateLocal;

    MPI_Bcast( &numStateLocalTemp, 1, MPI_INT, iproc, domain_.comm );

    IntNumVec wavefunIdxTemp(numStateLocalTemp);
    if( iproc == mpirank ){
      wavefunIdxTemp = wavefunIdx_;
    }

    MPI_Bcast( wavefunIdxTemp.Data(), numStateLocalTemp, MPI_INT, iproc, domain_.comm );

    for( Int kphi = 0; kphi < numStateLocalTemp; kphi++ ){
      for( Int jphi = 0; jphi < ncomPhi; jphi++ ){

        if( iproc == mpirank )
        {
          Real* phiPtr = phi.VecData(jphi, kphi);
          for( Int ir = 0; ir < ntot; ir++ ){
            phiTemp(ir) = phiPtr[ir];
          }
        }

        MPI_Bcast( phiTemp.Data(), ntot, MPI_DOUBLE, iproc, domain_.comm );

        // version 1: only do the GPU for the inner most part.
        cuda_memcpy_CPU2GPU(cu_phiTemp.Data(), phiTemp.Data(), sizeof(Real)*ntot);
        Real fac = -exxFraction * occupationRate[wavefunIdxTemp(kphi)];

        for (Int k=0; k<numStateLocal; k++) {
          for (Int j=0; j<ncom; j++) {
 
            cuda_set_vector( cu_psi.Data(), &cu_wave(0,k), ntot);
            
            // input vec = psi * phi
            cuda_vtot(cu_psi.Data(), cu_phiTemp.Data(), ntot);
 
            // exec the CUFFT. 
            cuFFTExecuteForward( fft, fft.cuPlanR2C[0], 0, cu_psi, cu_psi_out);

            // Solve the Poisson-like problem for exchange
            // note, exxgkkR2C apply to psi exactly like teter or laplacian
            cuda_teter( reinterpret_cast<cuDoubleComplex*> (cu_psi_out.Data()), cu_exxgkkR2C.Data(), ntotR2C );

            // exec the CUFFT.
            cuFFTExecuteInverse( fft, fft.cuPlanC2R[0], 0, cu_psi_out, cu_psi);

            // multiply by the occupationRate.
            // multiply with fac.
            cuda_Axpyz( a3.VecData(j,k), 1.0, cu_psi.Data(), fac, cu_phiTemp.Data(), ntot);            
          } // for (j)
        } // for (k)

      } // for (jphi)
    } // for (kphi)
  } // for (iproc)

  return;
}        // -----  end of method Spinor::AddMultSpinorEXXGPU  -----

}  // namespace pwdft
