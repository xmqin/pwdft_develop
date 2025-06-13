/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

Author: Lin Lin and Wei Hu

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
/// @file spinor.cpp
/// @brief Spinor (wavefunction) for the global domain or extended
/// element.
/// @date 2012-10-06
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

#ifdef _COMPLEX_
Spinor::Spinor ( 
    const Domain &dm, 
    const Int numComponent,
    const Int numStateTotal,
    const Complex val,
    const Point3 kpoint,
    const Int ik,
    const Int ikLocal ) 
{
  this->Setup( dm, numComponent, numStateTotal, val, kpoint, ik, ikLocal );
}         // -----  end of method Spinor::Spinor ( Complex version )  ----- 

Spinor::Spinor ( const Domain &dm, 
    const Int numComponent, 
    const Int numStateTotal,
    const bool owndata, 
    Complex* data,
    const Point3 kpoint,
    const Int ik,
    const Int ikLocal )
{
  this->Setup( dm, numComponent, numStateTotal, owndata, data, kpoint, ik, ikLocal );
}         // -----  end of method Spinor::Spinor ( Complex version )  ----- 

// Hints:
// 1. the wavefunction partition is only performed at the beginning of the
// SCF calculation when calling the first Spinor::Setup function( assuming 
// that the band number is keeped unchanged )
//
// 2. for spin-polarized calculations, the state number of initial wavefunction 
// is equal to electron number( not divided by 2 ); but in following procedures,
// spin-up and spin-down components are dealed seperately, and the setup of 
// spinor is done by the second Spinor::Setup function with numComponent as 1

void Spinor::Setup ( 
    const Domain &dm, 
    const Int numComponent,
    const Int numStateTotal,
    const Complex val,
    const Point3 kpoint,
    const Int ik,
    const Int ikLocal ) 
{
  domain_  = dm;
  kpoint_  = kpoint;
  ik_      = ik;
  ikLocal_ = ikLocal;

  // The band parallelism and grid parallelism are performed in colComm_kpoint
  MPI_Comm comm = domain_.colComm_kpoint;
  MPI_Barrier(comm);

  numComponent_ = numComponent;

  // Grid and state dimension
  if( esdfParam.isUseSphereCut == true ){
    numGridTotal_ = domain_.numGridSphere[ikLocal_];
  }
  else{
    numGridTotal_ = domain_.NumGridTotal();
  }
  numStateTotal_ = numStateTotal;

  mblocksize_ = esdfParam.BlockSizeGrid;
  nblocksize_ = esdfParam.BlockSizeState;

  // Block cyclic partition for grid and state indices
  CalculateIndexSpinor( numGridTotal_, mblocksize_, numGridLocal_, gridIdx_, comm );
  if( numGridLocal_ == 0 ){
    std::ostringstream msg;
    msg << "There is no grid point in some process,"
        << "reduce mpisize or Block_Size_Grid to continue." 
        <<  std::endl;
    ErrorHandling( msg.str().c_str() );
  }

  if( dm.numSpinComponent == 2 ){
    CalculateIndexSpinor( numStateTotal_/2, nblocksize_, numStateLocal_, wavefunIdx_, comm );
  }
  else{
    CalculateIndexSpinor( numStateTotal_, nblocksize_, numStateLocal_, wavefunIdx_, comm );
  }

  CalculateIndexAlltoall( numGridTotal_, numStateTotal_, mblocksize_, nblocksize_,
      sendcounts, recvcounts,
      senddispls, recvdispls,
      sendk, recvk, comm ); 
  
  if( dm.numSpinComponent == 2 ){
    wavefun_.Resize( numGridTotal_, numComponent_, numStateLocal_*2 );
  }
  else{
    wavefun_.Resize( numGridTotal_, numComponent_, numStateLocal_ );
  }

  SetValue( wavefun_, val );
}         // -----  end of method Spinor::Setup ( Complex version )  ----- 

void Spinor::Setup ( const Domain &dm, 
    const Int numComponent, 
    const Int numStateTotal,
    const bool owndata, 
    Complex* data,
    const Point3 kpoint,
    const Int ik,
    const Int ikLocal )
{
  domain_  = dm;
  kpoint_  = kpoint;
  ik_      = ik;
  ikLocal_ = ikLocal;

  // The band parallelism and grid parallelism are performed in colComm_kpoint
  MPI_Comm comm = domain_.colComm_kpoint;
  MPI_Barrier(comm);

  numComponent_ = numComponent;

  // Grid and state dimension
  if( esdfParam.isUseSphereCut == true ){
    numGridTotal_ = domain_.numGridSphere[ikLocal_];
  }
  else{
    numGridTotal_ = domain_.NumGridTotal();
  }
  numStateTotal_ = numStateTotal;

  mblocksize_ = esdfParam.BlockSizeGrid;
  nblocksize_ = esdfParam.BlockSizeState;

  // Block cyclic partition for grid and state indices
  CalculateIndexSpinor( numGridTotal_, mblocksize_, numGridLocal_, gridIdx_, comm );
  if( numGridLocal_ == 0 ){
    std::ostringstream msg;
    msg << "There is no grid point in some process,"
        << "reduce mpisize or Block_Size_Grid to continue."
        <<  std::endl;
    ErrorHandling( msg.str().c_str() );
  }

  CalculateIndexSpinor( numStateTotal_, nblocksize_, numStateLocal_, wavefunIdx_, comm );

  CalculateIndexAlltoall( numGridTotal_, numStateTotal_, mblocksize_, nblocksize_,
      sendcounts, recvcounts,
      senddispls, recvdispls,
      sendk, recvk, comm ); 

  wavefun_ = NumTns<Complex>( numGridTotal_, numComponent_, numStateLocal_,
      owndata, data );
}         // -----  end of method Spinor::Setup ( Complex version )  ----- 

void
Spinor::Normalize    ( )
{
  Int size = wavefun_.m() * wavefun_.n();
  Int nocc = wavefun_.p();

  for (Int k=0; k<nocc; k++) {
    Complex *ptr = wavefun_.MatData(k);
    Real sum = 0.0;
    for (Int i=0; i<size; i++) {
      sum += pow(abs(*ptr++), 2.0);
    }
    sum = sqrt(sum);
    if (sum != 0.0) {
      ptr = wavefun_.MatData(k);
      for (Int i=0; i<size; i++) *(ptr++) /= sum;
    }
  }

  return ;
}         // -----  end of method Spinor::Normalize ( Complex version )  ----- 

void
Spinor::AddTeterPrecond (Fourier* fftPtr, DblNumVec& teter, NumTns<Complex>& a3)
{
  Fourier& fft = *fftPtr;
  if( !fftPtr->isInitialized ){
    ErrorHandling("Fourier is not prepared.");
  }
  Int ntot = numGridTotal_;
  Int ncom = numComponent_;
  Int nocc = wavefun_.p();

  if( esdfParam.isUseSphereCut == false ){
    if( fft.domain.NumGridTotal() != ntot )
      ErrorHandling("Domain size does not match.");
  }
  else{
    if( fft.domain.numGridSphere[ikLocal_] != ntot )
      ErrorHandling("Domain size does not match.");
    if( fft.domain.NumGridTotal() < ntot )
      ErrorHandling("numGridSphere is larger than numGridTotal.");
  }

  Int numFFTGrid = fftPtr->numGridTotal;
  // These two are private variables in the OpenMP context

  for (Int k=0; k<nocc; k++) {
    for (Int j=0; j<ncom; j++) {
      // For c2r and r2c transforms, the default is to DESTROY the
      // input, therefore a copy of the original matrix is necessary. 
      if( esdfParam.isUseRealSpace == true ){ 
        blas::Copy( ntot, wavefun_.VecData(j,k), 1, 
          fft.inputComplexVec.Data(), 1 );

        FFTWExecute ( fft, fft.forwardPlan );

        Real* ptr1d      = teter.Data();
        Complex* ptr2    = fft.outputComplexVec.Data();
        for (Int i=0; i<numFFTGrid; i++) 
          *(ptr2++) *= *(ptr1d++);

        FFTWExecute ( fft, fft.backwardPlan);

        blas::Axpy( ntot, 1.0, fft.inputComplexVec.Data(), 1, a3.VecData(j,k), 1 );
      }
      else{
        for( Int i = 0; i < ntot; i++ ){
          a3.VecData(j,k)[i] += wavefun_.VecData(j,k)[i] * teter[i];
        }
      }
    }
  }

  return ;
}         // -----  end of method Spinor::AddTeterPrecond ( Complex version ) ----- 

void
Spinor::AddMultSpinorFine ( Fourier& fft, const std::vector<DblNumVec>& ekin,
    const DblNumVec& vtot, const std::vector<PseudoPot>& pseudo, NumTns<Complex>& a3 )
{
  // Complex case -- for TDDFT and k-point calculations ( spin-restricted and spin-unrestricted )

  if( !fft.isInitialized ){
    ErrorHandling("Fourier is not prepared.");
  }

  bool realspace = esdfParam.isUseRealSpace;
  bool spherecut = esdfParam.isUseSphereCut;

  Int ntot = numGridTotal_;
  Int ncom = numComponent_;
  Int numStateLocal = wavefun_.p();
  Int ntotFine = domain_.NumGridTotalFine();
  Real vol = domain_.Volume();

  Index3 &numGrid = fft.domain.numGrid;
  Index3 &numGridFine = fft.domain.numGridFine;
  unsigned plannerFlag = FFTW_MEASURE;

  if( !spherecut ){
    if( fft.domain.NumGridTotal() != ntot )
      ErrorHandling("Domain size does not match.");
  }
  else{
    if( fft.domain.numGridSphere[ikLocal_] != ntot )
      ErrorHandling("Domain size does not match.");
    if( fft.domain.NumGridTotal() < ntot )
      ErrorHandling("numGridSphere is larger than numGridTotal.");
  }

  if( ncom != 1 ){
    ErrorHandling("Component number of spinor should be one in \
        spin-restricted and spin-unrestricted calculations.");
  }

  // Time counter
  Real timeSta, timeEnd, timeSta1, timeEnd1;

  Real timeFFT = 0.0, timeNonlocal = 0.0, timeOther = 0.0;
  Int iterFFT = 0, iterNonlocal = 0;
  
  GetTime( timeSta1 );

  // Temporary variable for saving wavefunction on a fine grid
  CpxNumVec psiFine(ntotFine);
  CpxNumVec psiUpdateFine(ntotFine);

  for( Int k = 0; k < numStateLocal; k++ ){

    SetValue( psiFine, Complex(0.0,0.0) );
    SetValue( psiUpdateFine, Complex(0.0,0.0) );

    // Fourier transform
    if( realspace ){

      GetTime( timeSta );   
      // Fourier transform of wavefunction saved in fft.outputComplexVec
      fftw_execute( fftw_plan_dft_3d(
        numGrid[2], numGrid[1], numGrid[0],
        reinterpret_cast<fftw_complex*>( wavefun_.VecData(VAL,k) ),
        reinterpret_cast<fftw_complex*>( fft.outputComplexVec.Data() ),
        FFTW_FORWARD, plannerFlag ) );
      GetTime( timeEnd );
      timeFFT += ( timeEnd -timeSta );
      iterFFT ++;
    }
    else{
      blas::Copy( ntot, wavefun_.VecData(VAL,k), 1, fft.outputComplexVec.Data(), 1 );
    }
    // Interpolate wavefunction from coarse to fine grid
    {
      SetValue( fft.outputComplexVecFine, Z_ZERO ); 
      Int *idxPtr = NULL;
      if( spherecut )
        idxPtr = fft.idxFineCut[ikLocal_].Data();
      else
        idxPtr = fft.idxFineGrid.Data();
      
      Complex *fftOutFinePtr = fft.outputComplexVecFine.Data();
      Complex *fftOutPtr = fft.outputComplexVec.Data();
      for( Int i = 0; i < ntot; i++ ){
        fftOutFinePtr[*(idxPtr++)] = *(fftOutPtr++);
      }
    }

    GetTime( timeSta ); 
    fftw_execute( fftw_plan_dft_3d(
      numGridFine[2], numGridFine[1], numGridFine[0],
      reinterpret_cast<fftw_complex*>( fft.outputComplexVecFine.Data() ),
      reinterpret_cast<fftw_complex*>( psiFine.Data() ),
      FFTW_BACKWARD, plannerFlag ) );
    GetTime( timeEnd );
    timeFFT += ( timeEnd -timeSta );
    iterFFT ++;   

    Real fac = 1.0 / double(ntotFine);
    blas::Scal( ntotFine, fac, psiFine.Data(), 1 );

    // Add the contribution from local pseudopotential
    {
      Complex *psiUpdateFinePtr = psiUpdateFine.Data();
      Complex *psiFinePtr = psiFine.Data();
      Real *vtotPtr = vtot.Data();
      for( Int i = 0; i < ntotFine; i++ ){
        *(psiUpdateFinePtr++) += *(psiFinePtr++) * *(vtotPtr++);
      }
    }

    // Add the contribution from nonlocal pseudopotential  
    CpxNumVec vnlcX;  
    if( realspace )
    { 
      // Determines the maximum size of the buff used to store psi

      Int natm = pseudo.size();
      Int idxsize_max = 0;
      for( Int iatm = 0; iatm < natm; iatm++ ){
        Int idxsize = pseudo[iatm].vnlList[0].first.first.m();

        if( idxsize > idxsize_max )
          idxsize_max = idxsize;
      }

      CpxNumVec psiTemp( idxsize_max );
      DblNumVec wr( idxsize_max );
      DblNumVec wi( idxsize_max );

      GetTime( timeSta );
      for( Int iatm = 0; iatm < natm; iatm++ ){

        const std::vector<NonlocalPP>& vnlList = pseudo[iatm].vnlList;
        const CpxNumVec &ph = pseudo[iatm].vnlPhase[ik_];  
        const IntNumVec &iv = vnlList[0].first.first;

        Int idxsize = iv.m();
        Int nobt = vnlList.size();

        CpxNumVec weight( nobt );
 
        // Copy corresponding values of wavefunction to psiTemp
        for( Int i = 0; i < idxsize; i++ ){
          psiTemp[i] = psiFine[iv[i]];
        }

        for( Int i = 0; i < idxsize; i++ ){
          Complex wTemp = psiTemp[i] * ph[i];
          wr[i] = wTemp.real();
          wi[i] = wTemp.imag();
        }

        for( Int iobt = 0; iobt < nobt; iobt++ ){

          const Real vnlwgt = vnlList[iobt].second;
          const DblNumMat &dv = vnlList[iobt].first.second;

          Real weightr, weighti;
          weightr = blas::Dot( idxsize, dv.VecData(VAL), 1, wr.Data(), 1);
          weighti = blas::Dot( idxsize, dv.VecData(VAL), 1, wi.Data(), 1);            
          weight(iobt) = Complex( weightr, weighti ) * vol / Real(ntotFine) * vnlwgt;
        }  // for (iobt)
 
        for( Int i = 0; i < idxsize; i++ ){ 

          Complex phase = std::conj(ph[i]);
          Complex val = Z_ZERO;
          Int idx = iv[i];

          for( Int iobt = 0 ; iobt < nobt; iobt++ ){
            const DblNumMat &dv = vnlList[iobt].first.second;
            val += ( weight[iobt] * dv(i,0) );
          }

          psiUpdateFine[idx] += phase * val;
        }
      } // for (iatm)
    }

    GetTime( timeEnd );
    timeNonlocal += ( timeEnd -timeSta );
    iterNonlocal ++;

    // Laplacian operator. Perform inverse Fourier transform in the end
    {
      const DblNumVec& gkk = ekin[ikLocal_];
      for (Int i=0; i<ntot; i++) 
        fft.outputComplexVec(i) *= gkk(i);
    }

    // Restrict psiUpdateFine from fine grid in the real space to
    // coarse grid in the Fourier space. Combine with the Laplacian contribution
    GetTime( timeSta );
    fftw_execute( fftw_plan_dft_3d(
      numGridFine[2], numGridFine[1], numGridFine[0],
      reinterpret_cast<fftw_complex*>( psiUpdateFine.Data() ),
      reinterpret_cast<fftw_complex*>( fft.outputComplexVecFine.Data() ),
      FFTW_FORWARD, plannerFlag ) );
    GetTime( timeEnd );
    timeFFT += ( timeEnd -timeSta );
    iterFFT ++;

    {
      Int *idxPtr = NULL;
      if( spherecut )
        idxPtr = fft.idxFineCut[ikLocal_].Data();
      else
        idxPtr = fft.idxFineGrid.Data();

      Complex *fftOutFinePtr = fft.outputComplexVecFine.Data();
      Complex *fftOutPtr = fft.outputComplexVec.Data();

      for( Int i = 0; i < ntot; i++ ){
        *(fftOutPtr++) += fftOutFinePtr[*(idxPtr++)];
      }
    }

    if( realspace ){
      GetTime( timeSta );
      // Inverse Fourier transform to save back to the output vector
      fftw_execute( fft.backwardPlan );
      GetTime( timeEnd );
      timeFFT += ( timeEnd -timeSta );
      iterFFT ++;

      blas::Axpy( ntot, 1.0 / Real(ntot), 
          fft.inputComplexVec.Data(), 1, a3.VecData(VAL,k), 1 );
    }
    else{
      blas::Copy( ntot, fft.outputComplexVec.Data(), 1, a3.VecData(VAL,k), 1 );
    }
  }  // for (k)

  GetTime( timeEnd1 );

  timeOther = timeEnd1 - timeSta1 - timeFFT - timeNonlocal;

  timeFFT_ += timeFFT; timeNonlocal_ += timeNonlocal;
  timeMultSpinor_ += ( timeEnd1 - timeSta1 ); 

  iterFFT_ += iterFFT; iterNonlocal_ += iterNonlocal;
  iterMultSpinor_ ++;

  //statusOFS << "Time for iterFFT          = " << iterFFT            << "  timeFFT          = " << timeFFT << std::endl;
  //statusOFS << "Time for iterNonlocal     = " << iterNonlocal       << "  timeNonlocal     = " << timeNonlocal << std::endl;
  //statusOFS << "Time for Other procedures = " << timeOther << std::endl;
  //statusOFS << "Time for MultSpinor in PWDFT is " <<  timeEnd1 - timeSta1 << std::endl << std::endl;

  return ;
}        // -----  end of method Spinor::AddMultSpinorFine ( Complex version )  -----      

void
Spinor::AddMultSpinorFine ( Fourier& fft, const std::vector<DblNumVec>& ekin,
    const DblNumMat& vtot, const std::vector<PseudoPot>& pseudo, NumTns<Complex>& a3)
{
  // Complex case -- for spin-noncollinear calculations 

  if( !fft.isInitialized ){
    ErrorHandling("Fourier is not prepared.");
  }
  
  bool lspinorb = fft.domain.SpinOrbitCoupling;
  bool realspace = esdfParam.isUseRealSpace;
  bool spherecut = esdfParam.isUseSphereCut;

  Int ntot = numGridTotal_;
  Int ncom = numComponent_;
  Int numStateLocal = wavefun_.p();
  Int ntotFine = domain_.NumGridTotalFine();
  Real vol = domain_.Volume();

  Index3 &numGrid = fft.domain.numGrid;
  Index3 &numGridFine = fft.domain.numGridFine;
  unsigned plannerFlag = FFTW_MEASURE;

  if( !spherecut ){
    if( fft.domain.NumGridTotal() != ntot )
      ErrorHandling("Domain size does not match.");
  }
  else{
    if( fft.domain.numGridSphere[ikLocal_] != ntot )
      ErrorHandling("Domain size does not match.");
    if( fft.domain.NumGridTotal() < ntot )
      ErrorHandling("numGridSphere is larger than numGridTotal.");
  }

  if( ncom != 2 ){
    ErrorHandling("Component number of spinor should be two in \
        spin-noncollinear calculations.");
  }

  // Time counter
  Real timeSta, timeEnd, timeSta1, timeEnd1;

  Real timeFFT = 0.0, timeNonlocal = 0.0, timeOther = 0.0;
  Int iterFFT = 0, iterNonlocal = 0;

  GetTime( timeSta1 );

  // Temporary variable for saving wavefunction on a fine grid
  CpxNumMat psiFine(ntotFine, ncom);
  CpxNumMat psiUpdateFine(ntotFine, ncom);  

  // Store FFT results of psi in coarse grids
  CpxNumMat Gpsi(ntot, ncom);

  for( Int k = 0; k < numStateLocal; k++ ){

    SetValue( psiFine, Complex(0.0,0.0) );
    SetValue( psiUpdateFine, Complex(0.0,0.0) );

    for( Int j = 0; j < ncom; j++ ){
      // Fourier transform
      if( realspace ){

        GetTime( timeSta );
        // Fourier transform of wavefunction saved in fft.outputComplexVec
        fftw_execute( fftw_plan_dft_3d(
          numGrid[2], numGrid[1], numGrid[0],
          reinterpret_cast<fftw_complex*>( wavefun_.VecData(j,k) ),
          reinterpret_cast<fftw_complex*>( Gpsi.VecData(j) ),
          FFTW_FORWARD, plannerFlag ) );
        GetTime( timeEnd );
        timeFFT += ( timeEnd -timeSta );
        iterFFT ++;
      }
      else{
        blas::Copy( ntot, wavefun_.VecData(j,k), 1, Gpsi.VecData(j), 1 );
      }

      // Interpolate wavefunction from coarse to fine grid
      {
        SetValue( fft.outputComplexVecFine, Z_ZERO );
        Int *idxPtr = NULL;
        if( spherecut )
          idxPtr = fft.idxFineCut[ikLocal_].Data();
        else
          idxPtr = fft.idxFineGrid.Data();

        Complex *fftOutFinePtr = fft.outputComplexVecFine.Data();
        Complex *fftOutPtr = Gpsi.VecData(j);
        for( Int i = 0; i < ntot; i++ ){
          fftOutFinePtr[*(idxPtr++)] = *(fftOutPtr++);
        }
      }

      GetTime( timeSta );
      fftw_execute( fftw_plan_dft_3d(
        numGridFine[2], numGridFine[1], numGridFine[0],
        reinterpret_cast<fftw_complex*>( fft.outputComplexVecFine.Data() ),
        reinterpret_cast<fftw_complex*>( psiFine.VecData(j) ),
        FFTW_BACKWARD, plannerFlag ) );
      GetTime( timeEnd );
      timeFFT += ( timeEnd -timeSta );
      iterFFT ++;

      Real fac = 1.0 / double(domain_.NumGridTotalFine());
      blas::Scal( ntotFine, fac, psiFine.VecData(j), 1 );
    } // for (j)

    // Add the contribution from local pseudopotential
    {
      Complex *psiUpdateFinePtr_up = psiUpdateFine.VecData(UP);
      Complex *psiUpdateFinePtr_dw = psiUpdateFine.VecData(DN);
      Complex *psiFinePtr_up = psiFine.VecData(UP);
      Complex *psiFinePtr_dw = psiFine.VecData(DN);

      Real *vtotPtr_rho = vtot.VecData(RHO);
      Real *vtotPtr_mx = vtot.VecData(MAGX);
      Real *vtotPtr_my = vtot.VecData(MAGY);
      Real *vtotPtr_mz = vtot.VecData(MAGZ);

      for( Int i = 0; i < ntotFine; i++ ){
        *(psiUpdateFinePtr_up++) += ( *(psiFinePtr_up) * ( *(vtotPtr_rho) + *(vtotPtr_mz) )
            + *(psiFinePtr_dw) * ( *(vtotPtr_mx) - *(vtotPtr_my) * Complex(0.0, 1.0) ) );
        *(psiUpdateFinePtr_dw++) += ( *(psiFinePtr_dw) * ( *(vtotPtr_rho) - *(vtotPtr_mz) )
            + *(psiFinePtr_up) * ( *(vtotPtr_mx) + *(vtotPtr_my) * Complex(0.0, 1.0) ) );
        psiFinePtr_up++; psiFinePtr_dw++;
        vtotPtr_rho++; vtotPtr_mx++; vtotPtr_my++; vtotPtr_mz++;         
      }
    }

    // Add the contribution from nonlocal pseudopotential
    if( realspace ) 
    {
      GetTime( timeSta );

      Int natm = pseudo.size();
      Int idxsize_max = 0;
      for( Int iatm = 0; iatm < natm; iatm++ ){
        Int idxsize = pseudo[iatm].vnlList[0].first.first.m();

        if( idxsize > idxsize_max )
          idxsize_max = idxsize;
      }

      CpxNumVec psiTemp( idxsize_max );
      DblNumVec wr( idxsize_max );
      DblNumVec wi( idxsize_max );

      for( Int iatm = 0; iatm < natm; iatm++ ){

        const std::vector<NonlocalPP>& vnlList = pseudo[iatm].vnlList;
        const CpxNumVec &ph = pseudo[iatm].vnlPhase[ik_];
        const IntNumVec &iv = vnlList[0].first.first;

        Int idxsize = iv.m();
        Int nobt = vnlList.size();

        CpxNumMat weight( nobt, ncom );

        for( Int j = 0; j < ncom; j++ ){
          // Copy corresponding values of wavefunction to psiTemp
          for( Int i = 0; i < idxsize; i++ ){
            psiTemp[i] = psiFine(iv[i],j);
          }

          for( Int i = 0; i < idxsize; i++ ){
            Complex wTemp = psiTemp[i] * ph[i];
            wr[i] = wTemp.real();
            wi[i] = wTemp.imag();
          }

          for( Int iobt = 0; iobt < nobt; iobt++ ){

            const DblNumMat &dv = vnlList[iobt].first.second;

            Real weightr, weighti;
            weightr = blas::Dot( idxsize, dv.VecData(VAL), 1, wr.Data(), 1);
            weighti = blas::Dot( idxsize, dv.VecData(VAL), 1, wi.Data(), 1);           
            weight(iobt,j) = Complex( weightr, weighti );
          }  // for (iobt)
        }  // for (j)
         
        CpxNumMat coefw;

        if( lspinorb == false ){
          // For scalar-relativistic pseudopotential, coefMat is a diag matrix
          // which is written as DblNumVec vnlwgt          
          coefw = CpxNumMat( nobt, ncom, false, weight.Data() );
         
          for( Int iobt = 0; iobt < nobt; iobt++ ){

            const Real vnlwgt = vnlList[iobt].second;
            for( Int j = 0 ; j < ncom; j++ ){
              coefw(iobt,j) *= vnlwgt;
            }
          }
        }
        else{
          // Fully relativistic pseudopotential
          const CpxNumTns &coefMat = pseudo[iatm].coefMat;
          coefw.Resize( nobt, ncom );
          SetValue( coefw, Z_ZERO );

          // Matrix-vector multiplications between coefMats and weights
          blas::Gemv( 'N', nobt, nobt, Z_ONE, coefMat.MatData(RHO), nobt,
              weight.VecData(UP), 1, Z_ZERO, coefw.VecData(UP), 1 );
          blas::Gemv( 'N', nobt, nobt, Z_ONE, coefMat.MatData(MAGX), nobt,
              weight.VecData(DN), 1, Z_ONE, coefw.VecData(UP), 1 );

          blas::Gemv( 'N', nobt, nobt, Z_ONE, coefMat.MatData(MAGY), nobt,
              weight.VecData(UP), 1, Z_ZERO, coefw.VecData(DN), 1 );
          blas::Gemv( 'N', nobt, nobt, Z_ONE, coefMat.MatData(MAGZ), nobt,
              weight.VecData(DN), 1, Z_ONE, coefw.VecData(DN), 1 );
        }  // ---- end of if( lspinorb == false ) ----

        Real fac = vol / Real(ntotFine);
        blas::Scal( nobt * ncom, fac, coefw.Data(), 1 );

        for( Int i = 0; i < idxsize; i++ ){

          Complex phase = std::conj(ph[i]);
          Int idx = iv[i];

          for( Int j = 0; j < ncom; j++ ){
            Complex val = Z_ZERO;
            for( Int iobt = 0 ; iobt < nobt; iobt++ ){
              const DblNumMat &dv = vnlList[iobt].first.second;
              val += ( coefw(iobt,j) * dv(i,0) );
            }
            psiUpdateFine(idx,j) += phase * val;
          }
        }

      } // for (iatm)

      GetTime( timeEnd );
      timeNonlocal += ( timeEnd -timeSta );
      iterNonlocal ++;
    } 

    for( Int j = 0; j < ncom; j++ ){

      { 
        // Laplacian operator. Perform inverse Fourier transform in the end
        const DblNumVec& gkk = ekin[ikLocal_];
        for( Int i = 0; i < ntot; i++ ) 
          fft.outputComplexVec(i) = Gpsi(i,j) * gkk(i);
      }
      // Restrict psiUpdateFine from fine grid in the real space to
      // coarse grid in the Fourier space. Combine with the Laplacian contribution
      GetTime( timeSta );
      fftw_execute( fftw_plan_dft_3d(
        numGridFine[2], numGridFine[1], numGridFine[0],
        reinterpret_cast<fftw_complex*>( psiUpdateFine.VecData(j) ),
        reinterpret_cast<fftw_complex*>( fft.outputComplexVecFine.Data() ),
        FFTW_FORWARD, plannerFlag ) );
      GetTime( timeEnd );
      timeFFT += ( timeEnd -timeSta );
      iterFFT ++;

      {
        Int *idxPtr = NULL;
        if( spherecut )
          idxPtr = fft.idxFineCut[ikLocal_].Data();
        else
          idxPtr = fft.idxFineGrid.Data();

        Complex *fftOutFinePtr = fft.outputComplexVecFine.Data();
        Complex *fftOutPtr = fft.outputComplexVec.Data();

        for( Int i = 0; i < ntot; i++ ){
          *(fftOutPtr++) += fftOutFinePtr[*(idxPtr++)];
        }
      }

      if( realspace ){
        GetTime( timeSta );
        // Inverse Fourier transform to save back to the output vector
        fftw_execute( fft.backwardPlan );
        GetTime( timeEnd );
        timeFFT += ( timeEnd -timeSta );
        iterFFT ++;

        blas::Axpy( ntot, 1.0 / Real(ntot), 
            fft.inputComplexVec.Data(), 1, a3.VecData(j,k), 1 );
      }
      else{
        blas::Copy( ntot, fft.outputComplexVec.Data(), 1, a3.VecData(j,k), 1 );
      }
    }  // for (j)
  }  // for (k)
  
  GetTime( timeEnd1 );

  timeOther = timeEnd1 - timeSta1 - timeFFT - timeNonlocal;

  timeFFT_ += timeFFT; timeNonlocal_ += timeNonlocal;
  timeMultSpinor_ += ( timeEnd1 - timeSta1 );

  iterFFT_ += iterFFT; iterNonlocal_ += iterNonlocal;
  iterMultSpinor_ ++;

  //statusOFS << "Time for iterFFT          = " << iterFFT            << "  timeFFT          = " << timeFFT << std::endl;
  //statusOFS << "Time for iterNonlocal     = " << iterNonlocal       << "  timeNonlocal     = " << timeNonlocal << std::endl;
  //statusOFS << "Time for Other procedures = " << timeOther << std::endl;
  //statusOFS << "Time for MultSpinor in PWDFT is " <<  timeEnd1 - timeSta1 << std::endl << std::endl;

  return ;
}        // -----  end of method Spinor::AddMultSpinorFine ( Complex version )  ----- 

void Spinor::AddMultSpinorEXX ( Fourier& fft,
    std::vector<Spinor>& psik,
    const std::vector<CpxNumTns>& phik,
    const DblNumTns& exxgkk,
    Real  exxFraction,
    Int   spinIndex,
    Int   nspin,
    const std::vector<DblNumVec>& occupationRatek,
    std::vector<CpxNumTns>& a3 )
{
  // Calculate in MPI_COMM_WORLD
  if( !fft.isInitialized ){
    ErrorHandling("Fourier is not prepared.");
  }
  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

  bool realspace    = esdfParam.isUseRealSpace;
  bool spherecut    = esdfParam.isUseSphereCut;
  bool energyband   = esdfParam.isCalculateEnergyBand;

  Real vol          = domain_.Volume(); 
  Int ntot          = domain_.NumGridTotal();
  Int npwAll        = ( spherecut == true ) ? fft.idxFineFock.m() : ntot;
  Int nkLocal       = phik.size();
  Int ncom          = numComponent_;
  Int numStateLocal = psik[0].NumState();
  Int numStateTotal = psik[0].NumStateTotal();
  Int ntot2         = ntot * ncom;

  // Check the dimensions of wavefun
  if( ( (nspin == 1 || nspin == 2) && ncom != 1 ) 
      || (nspin == 4 && ncom != 2) ){
    ErrorHandling("The number of spinor components does not match.");
  }

  if( nspin == 2 ){
    numStateLocal /= 2;
    numStateTotal /= 2;
  }

  Real timeSta1, timeEnd1;
  Real timeSta, timeEnd;
  Real timeBcast = 0.0, timeFFT = 0.0, timeCopy = 0.0, timeOther = 0.0, timeRank0 = 0.0;
  Real iterBcast = 0, iterFFT = 0, iterCopy = 0, iterOther = 0, iterRank0 = 0;

  std::vector<CpxNumTns> vexxPsiR( nkLocal );
  if( !realspace ){
    for( Int k = 0; k < nkLocal; k++ ){
      vexxPsiR[k].Resize( ntot, ncom, numStateLocal );
      SetValue( vexxPsiR[k], Z_ZERO );
    }
  }

  GetTime( timeSta1 );

  // Temporary variable for saving wavefunction on a fine grid
  CpxNumVec phiTemp_rs( ntot2 );
  CpxNumVec vpsir( ntot );

  for( Int iproc = 0; iproc < mpisize; iproc++ ){

    // Bcast the number of k points 
    Int numKpointLocalTemp;

    if( iproc == mpirank ){
      numKpointLocalTemp = nkLocal;
    }

    GetTime( timeSta );    
    MPI_Bcast( &numKpointLocalTemp, 1, MPI_INT, iproc, domain_.comm );
    GetTime( timeEnd );
    timeBcast = timeBcast + timeEnd - timeSta;
    iterBcast ++;

    // Bcast the index of k points
    IntNumVec KpointIdxTemp(numKpointLocalTemp);

    if( iproc == mpirank ){
      GetTime( timeSta );
      for( Int q = 0; q < numKpointLocalTemp; q++ ){
        if( !energyband ){
          KpointIdxTemp(q) = domain_.KpointIdx(q);
        }
        else{
          KpointIdxTemp(q) = domain_.KpointIdx_scf(q);
        }
      }
      GetTime( timeEnd );
      timeRank0 = timeRank0 + timeEnd - timeSta;
      iterRank0 ++;
    }

    GetTime( timeSta );
    MPI_Bcast( KpointIdxTemp.Data(), numKpointLocalTemp, MPI_INT, iproc, domain_.comm );
    GetTime( timeEnd );
    timeBcast = timeBcast + timeEnd - timeSta;
    iterBcast ++;

    // loop for k points of index q
    for( Int q = 0; q < numKpointLocalTemp; q++ ){

      Int iq = KpointIdxTemp(q);

      // Bcast the number of non-empty bands
      Int numStateLocalTemp;

      if( iproc == mpirank ){
        numStateLocalTemp = occupationRatek[q].m();
      }

      GetTime( timeSta );
      MPI_Bcast( &numStateLocalTemp, 1, MPI_INT, iproc, domain_.comm );
      GetTime( timeEnd );
      timeBcast = timeBcast + timeEnd - timeSta;
      iterBcast ++;

      // Bcast the non-zero occupation rates
      DblNumVec occupationRateTemp(numStateLocalTemp);

      if( iproc == mpirank ){
        GetTime( timeSta );
        blas::Copy( numStateLocalTemp, occupationRatek[q].Data(), 1, occupationRateTemp.Data(), 1 );
        GetTime( timeEnd );
        timeRank0 = timeRank0 + timeEnd - timeSta;
        iterRank0 ++;
      }

      GetTime( timeSta );
      MPI_Bcast( occupationRateTemp.Data(), numStateLocalTemp, MPI_DOUBLE, iproc, domain_.comm );
      GetTime( timeEnd );
      timeBcast = timeBcast + timeEnd - timeSta;
      iterBcast ++;

      for( Int kphi = 0; kphi < numStateLocalTemp; kphi++ ){

        // Bcast phi in real space
        if( iproc == mpirank )
        {
          GetTime( timeSta );
          blas::Copy( ntot2, phik[q].VecData(0, kphi+spinIndex*numStateLocalTemp), 1, phiTemp_rs.Data(), 1 );       
          GetTime( timeEnd );
          timeRank0 = timeRank0 + timeEnd - timeSta;
          iterRank0 ++;
        }

        GetTime( timeSta );
        MPI_Bcast( phiTemp_rs.Data(), ntot2, MPI_DOUBLE_COMPLEX, iproc, domain_.comm );         
        GetTime( timeEnd );
        timeBcast = timeBcast + timeEnd - timeSta;
        iterBcast ++;

        for( Int k = 0; k < nkLocal; k++ ){

          CpxNumTns& wavefun_rs = psik[k].WavefunR();  

          for( Int kpsi = 0; kpsi < numStateLocal; kpsi++ ) {

            // Pair products nspin = 1 or 2: conj(phi_ki(r)) * psi_qj(r)
            // nspin = 4: conj(phi_ki\alpha(r)) * psi_qj\alpha(r) + conj(phi_ki\beta(r)) * psi_qj\beta(r)
            GetTime( timeSta ); 
            SetValue( fft.inputComplexVec, Z_ZERO );
            for( Int ispin = 0; ispin < ncom; ispin++ ){
              Complex* psiPtr = wavefun_rs.VecData(ispin,kpsi+spinIndex*numStateLocal);
              for( Int ir = 0; ir < ntot; ir++ ){
                fft.inputComplexVec(ir) += psiPtr[ir] * std::conj(phiTemp_rs(ir+ispin*ntot));
              }
            }
            GetTime( timeEnd );
            timeOther = timeOther + timeEnd - timeSta;
            iterOther ++;

            GetTime( timeSta );
            FFTWExecute ( fft, fft.forwardPlan );
            GetTime( timeEnd );
            timeFFT = timeFFT + timeEnd - timeSta;
            iterFFT ++;
  
            CpxNumVec prodG( ntot );
            GetTime( timeSta );
            blas::Copy( ntot, fft.outputComplexVec.Data(), 1, prodG.Data(), 1 );
            GetTime( timeEnd );
            timeCopy = timeCopy + timeEnd - timeSta;
            iterCopy ++;

            GetTime( timeSta );
            // Solve the Poisson-like problem for exchange
            if( spherecut ){
              // Truncation in Fourier space
              SetValue( fft.outputComplexVec, Z_ZERO );
              for( Int ig = 0; ig < npwAll; ig++ ){
                Int idx = fft.idxFineFock[ig];
                fft.outputComplexVec(idx) = prodG(idx) * exxgkk(ig,k,iq);
              }
            }
            else{
              for( Int ig = 0; ig < npwAll; ig++ )
                fft.outputComplexVec(ig) *= exxgkk(ig,k,iq);
            }
            GetTime( timeEnd );
            timeOther = timeOther + timeEnd - timeSta;
            iterOther ++;

            GetTime( timeSta );
            FFTWExecute ( fft, fft.backwardPlan );
            GetTime( timeEnd );
            timeFFT = timeFFT + timeEnd - timeSta;
            iterFFT ++;

            GetTime( timeSta );
            blas::Copy( ntot, fft.inputComplexVec.Data(), 1, vpsir.Data(), 1 );
            GetTime( timeEnd );
            timeCopy = timeCopy + timeEnd - timeSta;
            iterCopy ++;      

            GetTime( timeSta );
            for( Int ispin = 0; ispin < ncom; ispin++ ){

              Real fac = -exxFraction * occupationRateTemp(kphi);

              if( realspace ){
                Complex* a3Ptr = a3[k].VecData(ispin,kpsi);
                for( Int ir = 0; ir < ntot; ir++ )
                  a3Ptr[ir] += vpsir(ir) * phiTemp_rs(ir+ispin*ntot) * fac;
              }
              else{       
                Complex* vexxPsiPtr = vexxPsiR[k].VecData(ispin, kpsi);
                for( Int ir = 0; ir < ntot; ir++ )
                  *(vexxPsiPtr++)  += vpsir(ir) * phiTemp_rs(ir+ispin*ntot) * fac;
              }
            }  // for (ispin)

            GetTime( timeEnd );
            timeOther = timeOther + timeEnd - timeSta;
            iterOther ++;

          } // for (kpsi)
        } // for (k)
        MPI_Barrier(domain_.comm);
      } // for (kphi)
      MPI_Barrier(domain_.comm);   
    } //  for (q)
  } // for (iproc)
  MPI_Barrier(domain_.comm);

  // Transform vexxPsi back to reciprocal space and add to hpsi
  GetTime( timeSta );
  if( !realspace ){
    for( Int k = 0; k < nkLocal; k++ ){

      Int npw_this = ( spherecut == true ) ? domain_.numGridSphere[k] : ntot;
      IntNumVec &idxCoarseCut_this = fft.idxCoarseCut[k];

      for( Int kpsi = 0; kpsi < numStateLocal; kpsi++ ) {
        for( Int ispin = 0; ispin < ncom; ispin++ ){
  
          Complex* a3Ptr = a3[k].VecData(ispin,kpsi);

          blas::Copy( ntot, vexxPsiR[k].VecData(ispin, kpsi), 1, fft.inputComplexVec.Data(), 1 );

          fftw_execute( fft.forwardPlan );

          if( spherecut ){
            for( Int ig = 0; ig < npw_this; ig++ ){
              a3Ptr[ig] += fft.outputComplexVec(idxCoarseCut_this[ig]);
            }
          }
          else{
            blas::Axpy( npw_this, 1.0, fft.outputComplexVec.Data(), 1, a3Ptr, 1 );
          }
        }
      }
    }
  }
  GetTime( timeEnd );
  statusOFS << "Time for transforming vexxPsi to G space = " << timeEnd - timeSta << std::endl;

  GetTime( timeEnd1 );

  statusOFS << std::endl;
  statusOFS << "Total time for AddMultSpinorEXX is " << timeEnd1 - timeSta1 << std::endl;
  statusOFS << "Time for FFT is " << timeFFT << " Number of FFT is " << iterFFT << std::endl;
  statusOFS << "Time for Bcast is " << timeBcast << " Number of Bcast is " << iterBcast << std::endl;
  statusOFS << "Time for Copy is " << timeCopy << " Number of Copy is " << iterCopy << std::endl;
  statusOFS << "Time for Rank0 is " << timeRank0 << " Number of Rank0 is " << iterRank0 << std::endl;
  statusOFS << "Time for Other is " << timeOther << " Number of Other is " << iterOther << std::endl;

  statusOFS << std::endl;

  return ;
}        // -----  end of method Spinor::AddMultSpinorEXX ( Complex version )  ----- 

void Spinor::AddMultSpinorEXXDF ( Fourier& fft,
    const std::vector<Spinor>& psik,
    const std::vector<CpxNumTns>& phik,
    const DblNumTns& exxgkk,
    Real  exxFraction,
    Int   spinIndex,
    Int   nspin,
    const std::vector<DblNumVec>& occupationRatek,
    std::string& hybridDFType,
    std::string hybridDFKmeansWFType,
    const Real hybridDFKmeansWFAlpha,
    Real  hybridDFKmeansTolerance,
    Int   hybridDFKmeansMaxIter,
    const Real numMuFac,
    const Real numGaussianRandomFac,
    const Int numProcScaLAPACK,
    const Real hybridDFTolerance,
    const Int BlockSizeScaLAPACK,
    bool isFixColumnDF,
    std::vector<CpxNumMat>& VxMat,
    std::vector<CpxNumTns>& a3 )
{
  // Calculate in MPI_COMM_WORLD
  if( !fft.isInitialized ){
    ErrorHandling("Fourier is not prepared.");
  }
  MPI_Barrier(domain_.comm);
  int mpirank; MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize; MPI_Comm_size(domain_.comm, &mpisize);

  MPI_Comm& rowComm = domain_.rowComm_kpoint;
  MPI_Comm& colComm = domain_.colComm_kpoint;

  Int mpirankRow, mpisizeRow, mpirankCol, mpisizeCol;

  MPI_Comm_rank(colComm, &mpirankCol);
  MPI_Comm_size(colComm, &mpisizeCol);
  MPI_Comm_rank(rowComm, &mpirankRow);
  MPI_Comm_size(rowComm, &mpisizeRow);

  bool realspace    = esdfParam.isUseRealSpace;
  bool spherecut    = esdfParam.isUseSphereCut;   

  Int mb = esdfParam.BlockSizeGrid;
  Int nb = esdfParam.BlockSizeState;  

  Real vol          = domain_.Volume();
  Int ntot          = domain_.NumGridTotal();
  Int npw           = ( spherecut == true ) ? fft.idxFineFock.m() : ntot;
  Int nkTotal       = domain_.NumKGridTotal();
  Int nkLocal       = psik.size();
  Int ncom          = numComponent_;
  Int numStateLocal = psik[0].NumState();
  Int numStateTotal = psik[0].NumStateTotal();
  Int numOccLocal   = phik[0].p();
  Int numOccTotal   = phik[0].p()*mpisizeCol; // The number of occupied bands is same for each process
  Int ntot2         = ntot * ncom;

  // Check the dimensions of wavefun
  if( ( (nspin == 1 || nspin == 2) && ncom != 1 )
      || (nspin == 4 && ncom != 2) ){
    ErrorHandling("The number of spinor components does not match.");
  }

  if( nspin == 2 ){
    numStateLocal /= 2;
    numStateTotal /= 2;
  }

  Real timeSta1, timeEnd1;
  Real timeSta, timeEnd;
  Real timeAlltoall = 0, timeBcast = 0.0, timeFFT = 0.0, 
       timeCopy = 0.0, timeOther = 0.0, timeRank0 = 0.0,
       timeGemm = 0.0, timeGather = 0.0, timeAllreduce = 0.0;
  Real iterAlltoall = 0, iterBcast = 0, iterFFT = 0, 
       iterCopy = 0, iterOther = 0, iterRank0 = 0,
       iterGemm = 0, iterGather = 0, iterAllreduce = 0;

  // Get the size of matrix Psi and Phi
  Int nbPsiLocal = numStateLocal * nkLocal;
  Int nbPhiLocal = numOccLocal * nkLocal;

  Int nbPsiTotal, nbPhiTotal;
  MPI_Allreduce( &nbPsiLocal, &nbPsiTotal, 1, MPI_INT, MPI_SUM, domain_.comm );
  MPI_Allreduce( &nbPhiLocal, &nbPhiTotal, 1, MPI_INT, MPI_SUM, domain_.comm );

  if( nbPsiTotal != nkTotal*numStateTotal || nbPhiTotal != nkTotal*numOccTotal  ){
    ErrorHandling("The band number of Psi or Phi does not match.");
  }

  // Rearrange Psi and Phi in the form of a 2-D matrix
  CpxNumMat psiCol( ntot, nbPsiLocal );
  CpxNumMat phiCol( ntot, nbPhiLocal );
  DblNumVec occupationRate( nbPhiLocal );

  GetTime( timeSta1 );

  GetTime( timeSta );
  for( Int k = 0; k < nkLocal; k++ ){  

    lapack::Lacpy( 'A', ntot, numStateLocal, psik[k].WavefunR().Data(),  
        ntot, psiCol.VecData(k*numStateLocal), ntot );

    lapack::Lacpy( 'A', ntot, numOccLocal, phik[k].Data(),
        ntot, phiCol.VecData(k*numOccLocal), ntot );

    blas::Copy( occupationRatek[k].m(), occupationRatek[k].Data(), 1, 
        &occupationRate(k*numOccLocal), 1 );
  }
  GetTime( timeEnd );
  statusOFS << "Time for rearranging wavefunction = " << timeEnd - timeSta << std::endl;

  // *********************************************************************
  // Perform interpolative separable density fitting
  // *********************************************************************

  // Determine the rank for decomposing matrix products
  Int numMu = IRound( numMuFac * std::sqrt(nbPsiTotal * nbPhiTotal) ) ;

  // Grid and mu index in local process
  Int ntotTemp;
  IntNumVec idxGridTemp;
  CalculateIndexSpinor( ntot, mb, ntotTemp, idxGridTemp, domain_.comm );

  Int numMuTemp;
  IntNumVec idxMuTemp;
  CalculateIndexSpinor( numMu, I_ONE, numMuTemp, idxMuTemp, domain_.comm );

  Int ntotLocal, numMuLocal;
  GetTime( timeSta );
  MPI_Allreduce(&ntotTemp, &ntotLocal, 1, MPI_INT, MPI_SUM, rowComm);
  MPI_Allreduce(&numMuTemp, &numMuLocal, 1, MPI_INT, MPI_SUM, rowComm);
  GetTime( timeEnd );
  timeAllreduce = timeAllreduce + timeEnd - timeSta;
  iterAllreduce ++;

  std::vector<IntNumVec> idxGridRow( mpisizeCol );

  {
    GetTime( timeSta );

    idxGridRow[mpirankCol].Resize( ntotLocal );

    IntNumVec localSize( mpisizeRow );
    IntNumVec localDispls( mpisizeRow );
    SetValue( localSize, 0 );
    SetValue( localDispls, 0 );

    Int numElem = ntotTemp;
    MPI_Allgather( &numElem, 1, MPI_INT, localSize.Data(), 1, MPI_INT, rowComm );

    for( Int i = 1; i < mpisizeRow; i++ ){
      localDispls[i] = localDispls[i-1] + localSize[i-1];
    }

    MPI_Allgatherv( idxGridTemp.Data(), numElem, MPI_INT, idxGridRow[mpirankCol].Data(),
        localSize.Data(), localDispls.Data(), MPI_INT, rowComm );

    for( Int irank = 0; irank < mpisizeCol; irank++ ){

      Int ntotLocalTemp;

      if( irank == mpirankCol ){
        ntotLocalTemp = ntotLocal;
      }

      MPI_Bcast( &ntotLocalTemp, 1, MPI_INT, irank, colComm );

      if( irank != mpirankCol ){
        idxGridRow[irank].Resize( ntotLocalTemp );
      }

      MPI_Bcast( idxGridRow[irank].Data(), ntotLocalTemp, MPI_INT, irank, colComm );
    }
    GetTime( timeEnd );
    statusOFS << "Time for gathering idxGridRow = " << timeEnd - timeSta << std::endl;
  }
 
  std::vector<IntNumVec> idxMuCol( mpisizeCol );

  {
    GetTime( timeSta );

    idxMuCol[mpirankCol].Resize( numMuLocal );

    IntNumVec localSize( mpisizeRow );
    IntNumVec localDispls( mpisizeRow );
    SetValue( localSize, 0 );
    SetValue( localDispls, 0 );

    Int numElem = numMuTemp;
    MPI_Allgather( &numElem, 1, MPI_INT, localSize.Data(), 1, MPI_INT, rowComm );

    for( Int i = 1; i < mpisizeRow; i++ ){
      localDispls[i] = localDispls[i-1] + localSize[i-1];
    }
    
    // Collect mu index in colComm to make the distribution of index mu 
    // of K_ru and K_vu same as P_ru and P_vu
    MPI_Allgatherv( idxMuTemp.Data(), numElem, MPI_INT, idxMuCol[mpirankCol].Data(),
        localSize.Data(), localDispls.Data(), MPI_INT, rowComm );
  
    // Bcast the mu index inside k group
    for( Int irank = 0; irank < mpisizeCol; irank++ ){

      Int numMuLocalTemp;

      if( irank == mpirankCol ){
        numMuLocalTemp = numMuLocal;
      }

      MPI_Bcast( &numMuLocalTemp, 1, MPI_INT, irank, colComm );

      if( irank != mpirankCol ){    
        idxMuCol[irank].Resize( numMuLocalTemp ); 
      }

      MPI_Bcast( idxMuCol[irank].Data(), numMuLocalTemp, MPI_INT, irank, colComm );
    }
    GetTime( timeEnd );
    statusOFS << "Time for gathering idxMuCol = " << timeEnd - timeSta << std::endl;
  }

  // mu index is rearranged
  IntNumVec ndispls(mpisizeCol);

  GetTime( timeSta );
  ndispls[0] = 0;
  for( Int irank = 1; irank < mpisizeCol; irank++ ){
    ndispls[irank] = ndispls[irank-1] + idxMuCol[irank].m();
  }

  // ISDF is performed globally
  CpxNumTns VXi1D, KMuNu1D;

  GetTime( timeSta );
  if( isFixColumnDF == false ){
    ISDF_SelectIP( psiCol, phiCol, hybridDFType, hybridDFKmeansWFType,
        hybridDFKmeansWFAlpha, hybridDFKmeansTolerance, hybridDFKmeansMaxIter,
        hybridDFTolerance, numMu, numGaussianRandomFac, mb, nb );
  }
  GetTime( timeEnd );
  statusOFS << "Time for selecting IPs for ISDF = " << timeEnd - timeSta << std::endl;

  GetTime( timeSta );
  ISDF_CalculateIV( fft, psiCol, phiCol, occupationRate, exxgkk,
      exxFraction, numMu, mb, nb, idxMuCol, ndispls, VXi1D, KMuNu1D );
  GetTime( timeEnd );

  statusOFS << "Time for calculating IVs for ISDF = " << timeEnd - timeSta << std::endl;
  statusOFS << std::endl;

  if( ntotTemp != VXi1D.m() ){
    ErrorHandling("The partition of grids is not consistant with ScaLAPACK.");
  }

  if( numMuTemp != KMuNu1D.n() ){
    ErrorHandling("The partition of mu is not consistant with ScaLAPACK.");
  }

  IntNumVec pivMu(numMu);

  for( Int irank = 0; irank < mpisizeCol; irank++ ){

    IntNumVec& idxn = idxMuCol[irank];

    for( Int i = 0; i < idxn.m(); i++ ){
      pivMu(ndispls[irank] + i) = pivQR_(idxn[i]);
    }
  }

  // Gather K_uv and VXi to make each k group owns a complete K_uv and VXi 
  // which distributed in colComm
  Int Nkq = VXi1D.p();
  //CpxNumTns VXiRowT( numMu, ntotLocal, Nkq );
  CpxNumTns VXiRow( ntotLocal, numMu, Nkq );
  CpxNumTns KMuNuCol( numMu, numMuLocal, Nkq );  

  {
    GetTime( timeSta );

    CpxNumMat VXi1DT( numMu, ntotTemp );
    CpxNumMat VXiRowT( numMu, ntotLocal );
    
    IntNumVec localSize( mpisizeRow );
    IntNumVec localDispls( mpisizeRow );
    SetValue( localSize, 0 );
    SetValue( localDispls, 0 );

    Int numElem = numMu * ntotTemp;
    MPI_Allgather( &numElem, 1, MPI_INT, localSize.Data(), 1, MPI_INT, rowComm );

    for( Int i = 1; i < mpisizeRow; i++ ){
      localDispls[i] = localDispls[i-1] + localSize[i-1];
    }

    for( Int k = 0; k < Nkq; k++ ){
      // Transpose VXi1D to make data address continuous
      for( Int irank = 0; irank < mpisizeCol; irank++ ){
        IntNumVec& idxn = idxMuCol[irank];
        for( Int j = 0; j < idxn.m(); j++ ){
          for( Int i = 0; i < ntotTemp; i++ ){
            // Rearrange mu index
            VXi1DT(ndispls[irank] + j, i) = VXi1D(i,idxn[j],k);
          }
        }
      }

      MPI_Allgatherv( VXi1DT.Data(), numElem, MPI_DOUBLE_COMPLEX, VXiRowT.Data(),
          localSize.Data(), localDispls.Data(), MPI_DOUBLE_COMPLEX, rowComm );      

      for( Int j = 0; j < numMu; j++ ){
        for( Int i = 0; i < ntotLocal; i++ ){
          VXiRow(i,j,k) = VXiRowT(j,i);
        }
      }
    }

    GetTime( timeEnd );
    statusOFS << "Time for gathering VXiRow = " << timeEnd - timeSta << std::endl;
  }
  
  { 
    GetTime( timeSta );
    IntNumVec localSize( mpisizeRow );
    IntNumVec localDispls( mpisizeRow );
    SetValue( localSize, 0 );
    SetValue( localDispls, 0 );
    
    Int numElem = numMu * numMuTemp;
    MPI_Allgather( &numElem, 1, MPI_INT, localSize.Data(), 1, MPI_INT, rowComm );
    
    for( Int i = 1; i < mpisizeRow; i++ ){
      localDispls[i] = localDispls[i-1] + localSize[i-1];
    }
    
    CpxNumMat KMuNu1DTemp( numMu, numMuLocal );

    for( Int k = 0; k < Nkq; k++ ){
      // Rearrange row index
      for( Int irank = 0; irank < mpisizeCol; irank++ ){
        IntNumVec& idxn = idxMuCol[irank];
        for( Int j = 0; j < numMuLocal; j++ ){
          for( Int i = 0; i < idxn.m(); i++ ){
            KMuNu1DTemp(ndispls[irank] + i, j) = KMuNu1D(idxn[i],j,k);
          }
        }
      }

      MPI_Allgatherv( KMuNu1DTemp.Data(), numElem, MPI_DOUBLE_COMPLEX, KMuNuCol.MatData(k),
          localSize.Data(), localDispls.Data(), MPI_DOUBLE_COMPLEX, rowComm );
    }
    GetTime( timeEnd );
    statusOFS << "Time for gathering KMuNuCol = " << timeEnd - timeSta << std::endl;
  }

  std::vector<CpxNumMat> VMuNuCol( nkLocal );
  std::vector<CpxNumMat> VrNuRow( nkLocal );

  for( Int k = 0; k < nkLocal; k++ ){
    VMuNuCol[k].Resize( numMu, numMuLocal );
    VrNuRow[k].Resize( ntotLocal, numMu );

    SetValue( VMuNuCol[k], Z_ZERO );
    SetValue( VrNuRow[k], Z_ZERO );
  }

  GetTime( timeEnd );
  timeOther = timeOther + timeEnd - timeSta;
  iterOther ++;

  for( Int irank = 0; irank < mpisizeRow; irank++ ){

    // Bcast the number of k points
    Int numKpointLocalTemp;

    if( irank == mpirankRow ){
      numKpointLocalTemp = nkLocal;
    }

    MPI_Bcast( &numKpointLocalTemp, 1, MPI_INT, irank, rowComm );

    IntNumVec KpointIdxTemp(numKpointLocalTemp);

    if( irank == mpirankRow ){
      GetTime( timeSta );
      for( Int q = 0; q < numKpointLocalTemp; q++ ){
        KpointIdxTemp(q) = domain_.KpointIdx(q);
      }
      GetTime( timeEnd );
      timeRank0 = timeRank0 + timeEnd - timeSta;
      iterRank0 ++;
    }

    MPI_Bcast( KpointIdxTemp.Data(), numKpointLocalTemp, MPI_INT, irank, rowComm );

    // loop for k points of index q
    for( Int q = 0; q < numKpointLocalTemp; q++ ){

      Int iq = KpointIdxTemp(q);

      // Bcast the number of non-empty bands
      Int numStateLocalTemp;

      if( irank == mpirankRow ){
        numStateLocalTemp = occupationRatek[q].m();
      }

      MPI_Bcast( &numStateLocalTemp, 1, MPI_INT, irank, rowComm );

      // Bcast the non-zero occupation rates
      DblNumVec occupationRateTemp(numStateLocalTemp);

      if( irank == mpirankRow ){
        GetTime( timeSta );
        blas::Copy( numStateLocalTemp, occupationRatek[q].Data(), 1, occupationRateTemp.Data(), 1 );
        GetTime( timeEnd );
        timeRank0 = timeRank0 + timeEnd - timeSta;
        iterRank0 ++;
      }

      GetTime( timeSta );
      MPI_Bcast( occupationRateTemp.Data(), numStateLocalTemp, MPI_DOUBLE, irank, rowComm );
      GetTime( timeEnd );
      timeBcast = timeBcast + timeEnd - timeSta;
      iterBcast ++;

      // Temporary variable for saving wavefunction on a fine grid
      CpxNumMat phiTempCol( ntot2, numStateLocalTemp );

      if( irank == mpirankRow ){
        GetTime( timeSta );
        lapack::Lacpy('A', ntot2, numStateLocalTemp, phik[q].Data(), ntot2, phiTempCol.Data(), ntot2);
        GetTime( timeEnd );
        timeRank0 = timeRank0 + timeEnd - timeSta;
        iterRank0 ++;
      }

      GetTime( timeSta );
      MPI_Bcast( phiTempCol.Data(), ntot2*numStateLocalTemp, MPI_DOUBLE_COMPLEX, irank, rowComm );
      GetTime( timeEnd );
      timeBcast = timeBcast + timeEnd - timeSta;
      iterBcast ++;
      // Memory consideration:
      // The size of P_ru ,K_ru and Phi_k is large and can not be fully saved in each process
      // The size of P_vu ,K_vu and Phi_k(ind_mu,:) is acceptable
      //
      // Therefore:
      // For P_ru and P_vu, 
      // we combine and save Phi_k(ind_mu,:) in each process and convert Phi_k from
      // column to row format and multiply it with Phi_k(ind_mu,:)'
      //
      // For K_ru and K_vu,
      // K_ru is keeped column divided and perform multiplication directly.
      
      CpxNumMat phiMuCol( numMu, numStateLocalTemp );

      GetTime( timeSta );
      for( Int k = 0; k < numStateLocalTemp; k++ ){
        for( Int mu = 0; mu < numMu; mu++ ){
          phiMuCol(mu, k) = phiTempCol(pivMu(mu),k) * occupationRateTemp[k];
        }
      }
      GetTime( timeEnd );
      timeOther = timeOther + timeEnd - timeSta;
      iterOther ++;

      Int numStateTemp;
      GetTime( timeSta );
      MPI_Allreduce( &numStateLocalTemp, &numStateTemp, 1, MPI_INT, MPI_SUM, colComm );
      GetTime( timeEnd );
      timeAllreduce = timeAllreduce + timeEnd - timeSta;
      iterAllreduce ++;

      // Calculation of P_ru and P_vu
      CpxNumMat phiMu( numMu, numStateTemp );

      {
        GetTime( timeSta );

        IntNumVec localSize( mpisizeCol );
        IntNumVec localDispls( mpisizeCol );
        SetValue( localSize, 0 );
        SetValue( localDispls, 0 );

        Int numElem = numMu * numStateLocalTemp;
        MPI_Allgather( &numElem, 1, MPI_INT, localSize.Data(), 1, MPI_INT, colComm );

        for( Int i = 1; i < mpisizeCol; i++ ){
          localDispls[i] = localDispls[i-1] + localSize[i-1];
        }

        MPI_Allgatherv( phiMuCol.Data(), numElem, MPI_DOUBLE_COMPLEX, phiMu.Data(),
            localSize.Data(), localDispls.Data(), MPI_DOUBLE_COMPLEX, colComm );

        GetTime( timeEnd );
        timeGather = timeGather + timeEnd - timeSta;
        iterGather ++;
      }

      // column to row conversion of phiTemp
      CpxNumMat phiTempRow( ntotLocal, numStateTemp );
      GetTime( timeSta );
      AlltoallForward( idxGridRow, ncom, phiTempCol, phiTempRow, colComm );
      GetTime( timeEnd );
      timeAlltoall = timeAlltoall + timeEnd - timeSta;
      iterAlltoall ++;

      CpxNumMat PphiMuRow( ntotLocal, numMu );
      CpxNumMat PphiMuCol( ntot, numMuLocal );
      CpxNumMat phiMuNuCol( numMu, numMuLocal );

      GetTime( timeSta );
      blas::Gemm( 'N', 'C', ntotLocal, numMu, numStateTemp, 1.0, phiTempRow.Data(), ntotLocal,
          phiMu.Data(), numMu, 0.0, PphiMuRow.Data(), ntotLocal );
      GetTime( timeEnd );
      timeGemm = timeGemm + timeEnd - timeSta;
      iterGemm ++;

      GetTime( timeSta );
      AlltoallBackward( idxGridRow, ncom, PphiMuRow, PphiMuCol, colComm );
      GetTime( timeEnd );
      timeAlltoall = timeAlltoall + timeEnd - timeSta;
      iterAlltoall ++;

      GetTime( timeSta );
      for( Int nu = 0; nu < numMuLocal; nu++ ){
        for( Int mu = 0; mu < numMu; mu++ ){
          phiMuNuCol(mu, nu) = PphiMuCol( pivMu(mu), nu );
        }
      }
      GetTime( timeEnd );
      timeOther = timeOther + timeEnd - timeSta;
      iterOther ++;

      for( Int k = 0; k < nkLocal; k++ ){

        Int ik = domain_.KpointIdx(k);

        Index3 &numKGrid = domain_.numKGrid;
        std::vector<DblNumVec>& kgrid = domain_.kgrid;       

        Index3 idxkq, nkxyz;     
        for( Int d = 0; d < DIM; d++ ){   
          idxkq(d) = int((kgrid[d][ik] - kgrid[d][iq]) * numKGrid(d));
          nkxyz(d) = 2 * numKGrid(d) - 1;
          if( idxkq(d) < 0 ) idxkq(d) += nkxyz(d);
        }
  
        Int ikq = idxkq[2] * nkxyz[0] * nkxyz[1] + idxkq[1] * nkxyz[0] + idxkq[0];

        GetTime( timeSta );
        for( Int nu = 0; nu < numMuLocal; nu++ ){
          for( Int mu = 0; mu < numMu; mu++ ){
            VMuNuCol[k](mu,nu) += phiMuNuCol(mu,nu) * KMuNuCol(mu,nu,ikq);          
          } 
        }

        for( Int mu = 0; mu < numMu; mu++ ){
          for( Int i = 0; i < ntotLocal; i++ ){
            VrNuRow[k](i,mu) += PphiMuRow(i,mu) * VXiRow(i,mu,ikq);
            //VrNuRow[k](i,mu) += PphiMuRow(i,mu) * VXiRowT(mu,i,ikq);
          }
        }

        GetTime( timeEnd );
        timeOther = timeOther + timeEnd - timeSta;
        iterOther = iterOther + 2;
      } // for (k)
    } // for (q)
  } // for (irank)

  // Calculate matrix VxMat and vexxPsi
  for( Int k = 0; k < nkLocal; k++ ){

    CpxNumMat PsiMuCol( numMu, numStateLocal );
    CpxNumMat PsiMuRow( numMuLocal, numStateTotal );
    CpxNumMat PsiMu( numMu, numStateTotal );

    GetTime( timeSta );
    for( Int j = 0; j < numStateLocal; j++ ){
      for( Int mu = 0; mu < numMu; mu++ ){
        PsiMuCol(mu, j) = psik[k].WavefunR()(pivMu(mu),0,j);
      }
    }
    GetTime( timeEnd );
    timeOther = timeOther + timeEnd - timeSta;
    iterOther = iterOther + 2;

    GetTime( timeSta );
    IntNumVec localSize( mpisizeCol );
    IntNumVec localDispls( mpisizeCol );
    SetValue( localSize, 0 );
    SetValue( localDispls, 0 );

    Int numElem = numMu * numStateLocal;
    MPI_Allgather( &numElem, 1, MPI_INT, localSize.Data(), 1, MPI_INT, colComm );

    for( Int i = 1; i < mpisizeCol; i++ ){
      localDispls[i] = localDispls[i-1] + localSize[i-1];
    }
     
    MPI_Allgatherv( PsiMuCol.Data(), numElem, MPI_DOUBLE_COMPLEX, PsiMu.Data(),
        localSize.Data(), localDispls.Data(), MPI_DOUBLE_COMPLEX, colComm );
    GetTime( timeEnd );
    timeGather = timeGather + timeEnd - timeSta;
    iterGather ++;

    Int Ndispls = ndispls[mpirankCol];

    GetTime( timeSta );
    for( Int j = 0; j < numStateTotal; j++ ){
      for( Int i = 0; i < numMuLocal; i++ ){
        PsiMuRow(i, j) = PsiMu(i+Ndispls, j);
      }
    }
    GetTime( timeEnd );
    timeOther = timeOther + timeEnd - timeSta;
    iterOther = iterOther + 1;

    CpxNumMat vexxPsiRow( ntotLocal, numStateTotal );
    CpxNumMat vexxPsiCol( ntot, numStateLocal );    

    GetTime( timeSta );
    blas::Gemm( 'N', 'N', ntotLocal, numStateTotal, numMu, 1.0, VrNuRow[k].Data(), ntotLocal, 
        PsiMu.Data(), numMu, 0.0, vexxPsiRow.Data(), ntotLocal );
    GetTime( timeEnd );
    timeGemm = timeGemm + timeEnd - timeSta;
    iterGemm ++;

    GetTime( timeSta );
    AlltoallBackward( idxGridRow, ncom, vexxPsiRow, vexxPsiCol, colComm ); 
    GetTime( timeEnd );
    timeAlltoall = timeAlltoall + timeEnd - timeSta;
    iterAlltoall ++;

    GetTime( timeSta );
    if( !realspace ){
      Int npw_this = ( spherecut == true ) ? domain_.numGridSphere[k] : ntot;
      IntNumVec &idxCoarseCut_this = fft.idxCoarseCut[k];


      for( Int kpsi = 0; kpsi < numStateLocal; kpsi++ ){
        Complex* a3Ptr = a3[k].VecData(0,kpsi);

        blas::Copy( ntot, vexxPsiCol.VecData(kpsi), 1, fft.inputComplexVec.Data(), 1 );

        fftw_execute( fft.forwardPlan );

        if( spherecut ){
          for( Int ig = 0; ig < npw_this; ig++ ){
            a3Ptr[ig] += fft.outputComplexVec(idxCoarseCut_this[ig]);
          }
        }
        else{
          blas::Axpy( npw_this, 1.0, fft.outputComplexVec.Data(), 1, a3Ptr, 1 );
        }
      }
    }
    GetTime( timeEnd );
    statusOFS << "Time for transforming vexxPsi to G space = " << timeEnd - timeSta << std::endl;

    CpxNumMat VxMatTemp( numStateTotal, numMuLocal );

    GetTime( timeSta );
    blas::Gemm( 'C', 'N', numStateTotal, numMuLocal, numMu, 1.0, PsiMu.Data(), numMu,
        VMuNuCol[k].Data(), numMu, 0.0, VxMatTemp.Data(), numStateTotal );
    GetTime( timeEnd );
    timeGemm = timeGemm + timeEnd - timeSta;
    iterGemm ++;

    CpxNumMat VxMatLocal( numStateTotal, numStateTotal );

    GetTime( timeSta );
    blas::Gemm( 'N', 'N', numStateTotal, numStateTotal, numMuLocal, 1.0, VxMatTemp.Data(), 
        numStateTotal, PsiMuRow.Data(), numMuLocal, 0.0, VxMatLocal.Data(), numStateTotal );
    GetTime( timeEnd );
    timeGemm = timeGemm + timeEnd - timeSta;
    iterGemm ++;
 
    GetTime( timeSta );
    MPI_Allreduce( VxMatLocal.Data(), VxMat[k].Data(), 2*numStateTotal*numStateTotal, 
        MPI_DOUBLE, MPI_SUM, colComm );
    GetTime( timeEnd );
    timeAllreduce = timeAllreduce + timeEnd - timeSta;
    iterAllreduce ++;
  } 

  MPI_Barrier(domain_.comm);

  GetTime( timeEnd1 );

  Real timeAlltoall1 = 0, timeBcast1 = 0.0, timeFFT1 = 0.0,
       timeCopy1 = 0.0, timeOther1 = 0.0, timeRank01 = 0.0,
       timeGemm1 = 0.0, timeGather1 = 0.0, timeAllreduce1 = 0.0;

  MPI_Allreduce( &timeAlltoall, &timeAlltoall1, 1, MPI_DOUBLE, MPI_MAX, domain_.comm );
  MPI_Allreduce( &timeBcast, &timeBcast1, 1, MPI_DOUBLE, MPI_MAX, domain_.comm );
  MPI_Allreduce( &timeFFT, &timeFFT1, 1, MPI_DOUBLE, MPI_MAX, domain_.comm );
  MPI_Allreduce( &timeCopy, &timeCopy1, 1, MPI_DOUBLE, MPI_MAX, domain_.comm );
  MPI_Allreduce( &timeOther, &timeOther1, 1, MPI_DOUBLE, MPI_MAX, domain_.comm );
  MPI_Allreduce( &timeRank0, &timeRank01, 1, MPI_DOUBLE, MPI_MAX, domain_.comm );
  MPI_Allreduce( &timeGemm, &timeGemm1, 1, MPI_DOUBLE, MPI_MAX, domain_.comm );
  MPI_Allreduce( &timeGather, &timeGather1, 1, MPI_DOUBLE, MPI_MAX, domain_.comm );
  MPI_Allreduce( &timeAllreduce, &timeAllreduce1, 1, MPI_DOUBLE, MPI_MAX, domain_.comm );  

  statusOFS << std::endl;
  statusOFS << "Total time for AddMultSpinorEXXDF is " << timeEnd1 - timeSta1 << std::endl;  
  statusOFS << "Time for FFT is " << timeFFT1 << " Number of FFT is " << iterFFT << std::endl;
  statusOFS << "Time for Alltoall is " << timeAlltoall1 << " Number of Alltoall is " << iterAlltoall << std::endl;
  statusOFS << "Time for Bcast is " << timeBcast1 << " Number of Bcast is " << iterBcast << std::endl;
  statusOFS << "Time for Gather is " << timeGather1 << " Number of Gather is " << iterGather << std::endl;
  statusOFS << "Time for Allreduce is " << timeAllreduce1 << " Number of Allreduce is " << iterAllreduce << std::endl;
  statusOFS << "Time for Gemm is " << timeGemm1 << " Number of Gemm is " << iterGemm << std::endl;
  statusOFS << "Time for Copy is " << timeCopy1 << " Number of Copy is " << iterCopy << std::endl;
  statusOFS << "Time for Rank0 is " << timeRank01 << " Number of Rank0 is " << iterRank0 << std::endl;
  statusOFS << "Time for Other is " << timeOther1 << " Number of Other is " << iterOther << std::endl;

  return;
}         // -----  end of method Spinor::AddMultSpinorEXXDF ( Complex version )  -----

void Spinor::AddMultSpinorEXXDFConv ( Fourier& fft,
    const std::vector<Spinor>& psik,
    const std::vector<CpxNumTns>& phik,
    const DblNumTns& exxgkk,
    Real  exxFraction,
    Int   spinIndex,
    Int   nspin,
    const std::vector<DblNumVec>& occupationRatek,
    std::string& hybridDFType,
    std::string hybridDFKmeansWFType,
    const Real hybridDFKmeansWFAlpha,
    Real  hybridDFKmeansTolerance,
    Int   hybridDFKmeansMaxIter,
    const Real numMuFac,
    const Real numGaussianRandomFac,
    const Int numProcScaLAPACK,
    const Real hybridDFTolerance,
    const Int BlockSizeScaLAPACK,
    bool isFixColumnDF,
    std::vector<CpxNumMat>& VxMat,
    std::vector<CpxNumTns>& a3 )
{
  // Calculate in MPI_COMM_WORLD
  if( !fft.isInitialized ){
    ErrorHandling("Fourier is not prepared.");
  }
  MPI_Barrier(domain_.comm);
  int mpirank; MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize; MPI_Comm_size(domain_.comm, &mpisize);

  MPI_Comm& rowComm = domain_.rowComm_kpoint;
  MPI_Comm& colComm = domain_.colComm_kpoint;

  Int mpirankRow, mpisizeRow, mpirankCol, mpisizeCol;

  MPI_Comm_rank(colComm, &mpirankCol);
  MPI_Comm_size(colComm, &mpisizeCol);
  MPI_Comm_rank(rowComm, &mpirankRow);
  MPI_Comm_size(rowComm, &mpisizeRow);

  bool realspace    = esdfParam.isUseRealSpace;
  bool spherecut    = esdfParam.isUseSphereCut;   

  Int mb = esdfParam.BlockSizeGrid;
  Int nb = esdfParam.BlockSizeState;  

  Real vol          = domain_.Volume();
  Int ntot          = domain_.NumGridTotal();
  Int npw           = ( spherecut == true ) ? fft.idxFineFock.m() : ntot;
  Int nkTotal       = domain_.NumKGridTotal();
  Int nkLocal       = psik.size();
  Int ncom          = numComponent_;
  Int numStateLocal = psik[0].NumState();
  Int numStateTotal = psik[0].NumStateTotal();
  Int numOccLocal   = phik[0].p();
  Int numOccTotal   = phik[0].p()*mpisizeCol; // The number of occupied bands is same for each process
  Int ntot2         = ntot * ncom;

  Index3& numKGrid  = domain_.numKGrid;
  std::vector<DblNumVec>& kgrid = domain_.kgrid;
  Index3 nkxyz;
  for( Int d = 0; d < DIM; d++ ){
    nkxyz(d) = 2 * numKGrid(d) - 1;
  }

  // Check the dimensions of wavefun
  if( ( (nspin == 1 || nspin == 2) && ncom != 1 )
      || (nspin == 4 && ncom != 2) ){
    ErrorHandling("The number of spinor components does not match.");
  }

  if( nspin == 2 ){
    numStateLocal /= 2;
    numStateTotal /= 2;
  }

  Real timeSta1, timeEnd1;
  Real timeSta, timeEnd;
  Real timeAlltoall = 0, timeBcast = 0.0, timeFFT = 0.0, 
       timeCopy = 0.0, timeOther = 0.0, timeRank0 = 0.0,
       timeGemm = 0.0, timeGather = 0.0, timeAllreduce = 0.0;
  Real iterAlltoall = 0, iterBcast = 0, iterFFT = 0, 
       iterCopy = 0, iterOther = 0, iterRank0 = 0,
       iterGemm = 0, iterGather = 0, iterAllreduce = 0;

  // Get the size of matrix Psi and Phi
  Int nbPsiLocal = numStateLocal * nkLocal;
  Int nbPhiLocal = numOccLocal * nkLocal;

  Int nbPsiTotal, nbPhiTotal;
  MPI_Allreduce( &nbPsiLocal, &nbPsiTotal, 1, MPI_INT, MPI_SUM, domain_.comm );
  MPI_Allreduce( &nbPhiLocal, &nbPhiTotal, 1, MPI_INT, MPI_SUM, domain_.comm );

  if( nbPsiTotal != nkTotal*numStateTotal || nbPhiTotal != nkTotal*numOccTotal  ){
    ErrorHandling("The band number of Psi or Phi does not match.");
  }

  // Rearrange Psi and Phi in the form of a 2-D matrix
  CpxNumMat psiCol( ntot, nbPsiLocal );
  CpxNumMat phiCol( ntot, nbPhiLocal );
  DblNumVec occupationRate( nbPhiLocal );

  GetTime( timeSta1 );

  GetTime( timeSta );
  for( Int k = 0; k < nkLocal; k++ ){  

    lapack::Lacpy( 'A', ntot, numStateLocal, psik[k].WavefunR().Data(),  
        ntot, psiCol.VecData(k*numStateLocal), ntot );

    lapack::Lacpy( 'A', ntot, numOccLocal, phik[k].Data(),
        ntot, phiCol.VecData(k*numOccLocal), ntot );

    blas::Copy( occupationRatek[k].m(), occupationRatek[k].Data(), 1, 
        &occupationRate(k*numOccLocal), 1 );
  }
  GetTime( timeEnd );
  statusOFS << "Time for rearranging wavefunction = " << timeEnd - timeSta << std::endl;

  // Determine the rank for decomposing matrix products
  Int numMu = IRound( numMuFac * std::sqrt(nbPsiTotal * nbPhiTotal) ) ;

  // Grid and mu index in local process
  Int ntotLocal;
  IntNumVec idxGridLocal;
  CalculateIndexSpinor( ntot, mb, ntotLocal, idxGridLocal, domain_.comm );

  Int numMuLocal;
  IntNumVec idxMuLocal;
  CalculateIndexSpinor( numMu, I_ONE, numMuLocal, idxMuLocal, domain_.comm );

  // Grid and mu index in rowComm
  Int ntotRow, numMuRow;
  GetTime( timeSta );
  MPI_Allreduce(&ntotLocal, &ntotRow, 1, MPI_INT, MPI_SUM, rowComm);
  MPI_Allreduce(&numMuLocal, &numMuRow, 1, MPI_INT, MPI_SUM, rowComm);
  GetTime( timeEnd );
  timeAllreduce = timeAllreduce + timeEnd - timeSta;
  iterAllreduce ++;

  std::vector<IntNumVec> idxGridRow( mpisizeCol );
  IntNumVec mdisplsRow(mpisizeRow);
  IntNumVec mdisplsCol(mpisizeCol);

  {
    GetTime( timeSta );

    idxGridRow[mpirankCol].Resize( ntotRow );

    IntNumVec localSize( mpisizeRow );
    IntNumVec localDispls( mpisizeRow );
    SetValue( localSize, 0 );
    SetValue( localDispls, 0 );

    Int numElem = ntotLocal;
    MPI_Allgather( &numElem, 1, MPI_INT, localSize.Data(), 1, MPI_INT, rowComm );

    for( Int i = 1; i < mpisizeRow; i++ ){
      localDispls[i] = localDispls[i-1] + localSize[i-1];
    }
    mdisplsRow = localDispls;

    MPI_Allgatherv( idxGridLocal.Data(), numElem, MPI_INT, idxGridRow[mpirankCol].Data(),
        localSize.Data(), localDispls.Data(), MPI_INT, rowComm );

    for( Int irank = 0; irank < mpisizeCol; irank++ ){

      Int ntotLocalTemp;

      if( irank == mpirankCol ){
        ntotLocalTemp = ntotRow;
      }

      MPI_Bcast( &ntotLocalTemp, 1, MPI_INT, irank, colComm );

      if( irank != mpirankCol ){
        idxGridRow[irank].Resize( ntotLocalTemp );
      }

      MPI_Bcast( idxGridRow[irank].Data(), ntotLocalTemp, MPI_INT, irank, colComm );
    }
   
    mdisplsCol[0] = 0;
    for( Int irank = 1; irank < mpisizeCol; irank++ ){
      mdisplsCol[irank] = mdisplsCol[irank-1] + idxGridRow[irank].m();
    }

    GetTime( timeEnd );
    statusOFS << "Time for gathering idxGridRow = " << timeEnd - timeSta << std::endl;
  }
 
  std::vector<IntNumVec> idxMuRow( mpisizeCol );
  IntNumVec ndisplsRow(mpisizeRow);
  IntNumVec ndisplsCol(mpisizeCol);

  {
    GetTime( timeSta );

    idxMuRow[mpirankCol].Resize( numMuRow );

    IntNumVec localSize( mpisizeRow );
    IntNumVec localDispls( mpisizeRow );
    SetValue( localSize, 0 );
    SetValue( localDispls, 0 );

    Int numElem = numMuLocal;
    MPI_Allgather( &numElem, 1, MPI_INT, localSize.Data(), 1, MPI_INT, rowComm );

    for( Int i = 1; i < mpisizeRow; i++ ){
      localDispls[i] = localDispls[i-1] + localSize[i-1];
    }   
    ndisplsRow = localDispls;
 
    // Collect mu index in colComm to make the distribution of index mu 
    // of K_ru and K_vu same as P_ru and P_vu
    MPI_Allgatherv( idxMuLocal.Data(), numElem, MPI_INT, idxMuRow[mpirankCol].Data(),
        localSize.Data(), localDispls.Data(), MPI_INT, rowComm );
  
    // Bcast the mu index inside k group
    for( Int irank = 0; irank < mpisizeCol; irank++ ){

      Int numMuLocalTemp;

      if( irank == mpirankCol ){
        numMuLocalTemp = numMuRow;
      }

      MPI_Bcast( &numMuLocalTemp, 1, MPI_INT, irank, colComm );

      if( irank != mpirankCol ){    
        idxMuRow[irank].Resize( numMuLocalTemp ); 
      }

      MPI_Bcast( idxMuRow[irank].Data(), numMuLocalTemp, MPI_INT, irank, colComm );
    }

    ndisplsCol[0] = 0;
    for( Int irank = 1; irank < mpisizeCol; irank++ ){
      ndisplsCol[irank] = ndisplsCol[irank-1] + idxMuRow[irank].m();
    }

    GetTime( timeEnd );
    statusOFS << "Time for gathering idxMuCol = " << timeEnd - timeSta << std::endl;
  }

  // *********************************************************************
  // Perform interpolative separable density fitting
  // *********************************************************************

  // ISDF is performed globally
  CpxNumTns VXi1D, KMuNu1D;

  {
    GetTime( timeSta );
    if( isFixColumnDF == false ){
      ISDF_SelectIP( psiCol, phiCol, hybridDFType, hybridDFKmeansWFType,
          hybridDFKmeansWFAlpha, hybridDFKmeansTolerance, hybridDFKmeansMaxIter,
          hybridDFTolerance, numMu, numGaussianRandomFac, mb, nb );
    }
    GetTime( timeEnd );
    statusOFS << "Time for selecting IPs for ISDF = " << timeEnd - timeSta << std::endl;

    GetTime( timeSta );
    ISDF_CalculateIV( fft, psiCol, phiCol, occupationRate, exxgkk,
        exxFraction, numMu, mb, nb, idxMuRow, ndisplsCol, VXi1D, KMuNu1D );
    GetTime( timeEnd );
    statusOFS << "Time for calculating IVs for ISDF = " << timeEnd - timeSta << std::endl;
    statusOFS << std::endl;

    if( ntotLocal != VXi1D.n() ){
      ErrorHandling("The partition of grids is not consistant with ScaLAPACK.");
    }

    if( numMuLocal != KMuNu1D.p() ){
      ErrorHandling("The partition of mu is not consistant with ScaLAPACK.");
    }
  }

  IntNumVec pivMu(numMu), pivMuLocal(numMuLocal);

  for( Int irank = 0; irank < mpisizeCol; irank++ ){

    IntNumVec& idxn = idxMuRow[irank];

    for( Int i = 0; i < idxn.m(); i++ ){
      pivMu(ndisplsCol[irank] + i) = pivQR_(idxn[i]);
    }
  }

  for( Int i = 0; i < numMuLocal; i++ ){
    pivMuLocal(i) = pivQR_(idxMuLocal[i]);
  }

  // Grid index in colComm which is only used for col to row transformation
  // of part of Phi
  Int ntotCol;
  GetTime( timeSta );
  MPI_Allreduce(&ntotLocal, &ntotCol, 1, MPI_INT, MPI_SUM, colComm);
  GetTime( timeEnd );
  timeAllreduce = timeAllreduce + timeEnd - timeSta;
  iterAllreduce ++;

  std::vector<IntNumVec> idxGridCol( mpisizeCol );

  {
    GetTime( timeSta );

    idxGridCol[mpirankCol] = idxGridLocal;

    for( Int irank = 0; irank < mpisizeCol; irank++ ){

      Int ntotLocalTemp;

      if( irank == mpirankCol ){
        ntotLocalTemp = ntotLocal;
      }

      MPI_Bcast( &ntotLocalTemp, 1, MPI_INT, irank, colComm );

      if( irank != mpirankCol ){
        idxGridCol[irank].Resize( ntotLocalTemp );
      }

      MPI_Bcast( idxGridCol[irank].Data(), ntotLocalTemp, MPI_INT, irank, colComm );
    }
    GetTime( timeEnd );
    statusOFS << "Time for gathering idxGridCol = " << timeEnd - timeSta << std::endl;
  }

  // The calculation of P_ru and P_vu is performed in colComm
  // and stored in a larger k mesh

  Int Nkq = exxgkk.n();

  CpxNumTns PphiMuRow( Nkq, ntotLocal, numMu );
  CpxNumTns phiMuNuCol( Nkq, numMu, numMuLocal );
  SetValue( PphiMuRow, Z_ZERO );
  SetValue( phiMuNuCol, Z_ZERO );

  for( Int irank = 0; irank < mpisizeRow; irank++ ){

    // Bcast the number of k points
    Int numKpointLocalTemp;

    if( irank == mpirankRow ){
      numKpointLocalTemp = nkLocal;
    }

    MPI_Bcast( &numKpointLocalTemp, 1, MPI_INT, irank, rowComm );

    IntNumVec KpointIdxTemp(numKpointLocalTemp);

    if( irank == mpirankRow ){
      GetTime( timeSta );
      for( Int q = 0; q < numKpointLocalTemp; q++ ){
        KpointIdxTemp(q) = domain_.KpointIdx(q);
      }
      GetTime( timeEnd );
      timeRank0 = timeRank0 + timeEnd - timeSta;
      iterRank0 ++;
    }

    MPI_Bcast( KpointIdxTemp.Data(), numKpointLocalTemp, MPI_INT, irank, rowComm );

    // loop for k points of index q
    for( Int q = 0; q < numKpointLocalTemp; q++ ){

      Int iq = KpointIdxTemp(q);

      Index3 idxkq;
      for( Int d = 0; d < DIM; d++ ){
        idxkq(d) = int(kgrid[d][iq] * numKGrid(d));
        if( idxkq(d) < 0 ) idxkq(d) += nkxyz[d];
      }

      Int ikq = idxkq[2] * nkxyz[0] * nkxyz[1] + idxkq[1] * nkxyz[0] + idxkq[0];

      // Bcast the number of non-empty bands
      Int numStateLocalTemp;

      if( irank == mpirankRow ){
        numStateLocalTemp = occupationRatek[q].m();
      }

      MPI_Bcast( &numStateLocalTemp, 1, MPI_INT, irank, rowComm );

      // Bcast the non-zero occupation rates
      DblNumVec occupationRateTemp(numStateLocalTemp);

      if( irank == mpirankRow ){
        GetTime( timeSta );
        blas::Copy( numStateLocalTemp, occupationRatek[q].Data(), 1, occupationRateTemp.Data(), 1 );
        GetTime( timeEnd );
        timeRank0 = timeRank0 + timeEnd - timeSta;
        iterRank0 ++;
      }

      GetTime( timeSta );
      MPI_Bcast( occupationRateTemp.Data(), numStateLocalTemp, MPI_DOUBLE, irank, rowComm );
      GetTime( timeEnd );
      timeBcast = timeBcast + timeEnd - timeSta;
      iterBcast ++;

      // Temporary variable for saving wavefunction on a fine grid
      CpxNumMat phiTempCol( ntot2, numStateLocalTemp );

      if( irank == mpirankRow ){
        GetTime( timeSta );
        lapack::Lacpy('A', ntot2, numStateLocalTemp, phik[q].Data(), ntot2, phiTempCol.Data(), ntot2);
        GetTime( timeEnd );
        timeRank0 = timeRank0 + timeEnd - timeSta;
        iterRank0 ++;
      }

      GetTime( timeSta );
      MPI_Bcast( phiTempCol.Data(), ntot2*numStateLocalTemp, MPI_DOUBLE_COMPLEX, irank, rowComm );
      GetTime( timeEnd );
      timeBcast = timeBcast + timeEnd - timeSta;
      iterBcast ++;
      
      CpxNumMat phiMuCol( numMu, numStateLocalTemp );

      GetTime( timeSta );
      for( Int k = 0; k < numStateLocalTemp; k++ ){
        for( Int mu = 0; mu < numMu; mu++ ){
          phiMuCol(mu, k) = phiTempCol(pivMu(mu),k) * occupationRateTemp[k];
        }
      }
      GetTime( timeEnd );
      timeOther = timeOther + timeEnd - timeSta;
      iterOther ++;

      Int numStateTemp;
      GetTime( timeSta );
      MPI_Allreduce( &numStateLocalTemp, &numStateTemp, 1, MPI_INT, MPI_SUM, colComm );
      GetTime( timeEnd );
      timeAllreduce = timeAllreduce + timeEnd - timeSta;
      iterAllreduce ++;

      // Calculation of P_ru and P_vu
      CpxNumMat phiMu( numMu, numStateTemp );

      {
        GetTime( timeSta );

        IntNumVec localSize( mpisizeCol );
        IntNumVec localDispls( mpisizeCol );
        SetValue( localSize, 0 );
        SetValue( localDispls, 0 );

        Int numElem = numMu * numStateLocalTemp;
        MPI_Allgather( &numElem, 1, MPI_INT, localSize.Data(), 1, MPI_INT, colComm );

        for( Int i = 1; i < mpisizeCol; i++ ){
          localDispls[i] = localDispls[i-1] + localSize[i-1];
        }

        MPI_Allgatherv( phiMuCol.Data(), numElem, MPI_DOUBLE_COMPLEX, phiMu.Data(),
            localSize.Data(), localDispls.Data(), MPI_DOUBLE_COMPLEX, colComm );

        GetTime( timeEnd );
        timeGather = timeGather + timeEnd - timeSta;
        iterGather ++;
      }

      // column to row conversion of part of phiTemp
      // TODO another implementation may be faster
      CpxNumMat phiTempRow( ntotRow, numStateTemp );
      GetTime( timeSta );
      AlltoallForward( idxGridRow, ncom, phiTempCol, phiTempRow, colComm );
      GetTime( timeEnd );
      timeAlltoall = timeAlltoall + timeEnd - timeSta;
      iterAlltoall ++;

      CpxNumMat PphiMuRowTemp( ntotRow, numMu );
      CpxNumMat PphiMuColTemp( ntot, numMuRow );
      CpxNumMat phiMuNuColTemp( numMu, numMuRow );

      GetTime( timeSta );
      blas::Gemm( 'N', 'C', ntotRow, numMu, numStateTemp, 1.0, phiTempRow.Data(), ntotRow,
          phiMu.Data(), numMu, 0.0, PphiMuRowTemp.Data(), ntotRow );
      GetTime( timeEnd );
      timeGemm = timeGemm + timeEnd - timeSta;
      iterGemm ++;

      AlltoallBackward( idxGridRow, ncom, PphiMuRowTemp, PphiMuColTemp, colComm );

      GetTime( timeSta );
      Int mdispls = mdisplsRow[mpirankRow];
      for( Int nu = 0; nu < numMu; nu++ ){
        for( Int i = 0; i < ntotLocal; i++ ){
          PphiMuRow(ikq, i, nu) = PphiMuRowTemp(i+mdispls, nu);
        }
      }

      Int ndispls = ndisplsRow[mpirankRow];
      for( Int nu = 0; nu < numMuLocal; nu++ ){
        for( Int mu = 0; mu < numMu; mu++ ){
          phiMuNuCol(ikq, mu, nu) = PphiMuColTemp( pivMu[mu], nu+ndispls );
        }
      }

      GetTime( timeEnd );
      timeOther = timeOther + timeEnd - timeSta;
      iterOther ++;
    } // for (q)
  } // for (irank)

  // Calculate V_ru and V_vu by Fourier convolution
  std::vector<CpxNumMat> VrNu1D( nkTotal );
  std::vector<CpxNumMat> VMuNu1D( nkTotal );
  
  for( Int k = 0; k < nkTotal; k++ ){
    VrNu1D[k].Resize( ntotLocal, numMu );
    VMuNu1D[k].Resize( numMu, numMuLocal );
    
    SetValue( VrNu1D[k], Z_ZERO );
    SetValue( VMuNu1D[k], Z_ZERO ); 
  }

  // Defination of FFT on k mesh
  Int nkq = nkxyz[0] * nkxyz[1] * nkxyz[2];

  unsigned plannerFlag = FFTW_MEASURE;

  CpxNumVec inputComplexVec( nkq );
  CpxNumVec outputComplexVec( nkq );

  fftw_plan forwardPlan = fftw_plan_dft_3d(
      nkxyz[2], nkxyz[1], nkxyz[0],
      reinterpret_cast<fftw_complex*>( &inputComplexVec[0] ),
      reinterpret_cast<fftw_complex*>( &outputComplexVec[0] ),
      FFTW_FORWARD, plannerFlag );
  
  fftw_plan backwardPlan = fftw_plan_dft_3d(
      nkxyz[2], nkxyz[1], nkxyz[0],
      reinterpret_cast<fftw_complex*>( &outputComplexVec[0] ),
      reinterpret_cast<fftw_complex*>( &inputComplexVec[0] ),
      FFTW_BACKWARD, plannerFlag );

  {
    // V_ru is calculated by numMu * ntot times FFT
    CpxNumVec VXi_fft( nkq );
    CpxNumVec PphiMu_fft( nkq );
    CpxNumVec VrNu( nkq );

    for( Int nu = 0; nu < numMu; nu++ ){
      for( Int i = 0; i < ntotLocal; i++ ){

        GetTime( timeSta );
        blas::Copy( nkq, VXi1D.VecData(i,nu), 1, inputComplexVec.Data(), 1 );
        fftw_execute( forwardPlan );
        blas::Copy( nkq, outputComplexVec.Data(), 1, VXi_fft.Data(), 1 );

        blas::Copy( nkq, PphiMuRow.VecData(i,nu), 1, inputComplexVec.Data(), 1 );
        fftw_execute( forwardPlan );
        blas::Copy( nkq, outputComplexVec.Data(), 1, PphiMu_fft.Data(), 1 );
        GetTime( timeEnd );
   
        timeFFT = timeFFT + timeEnd - timeSta;
        iterFFT = iterFFT + 2;

        for( Int k = 0; k < nkq; k++ ){
          VXi_fft(k) *= PphiMu_fft(k);
        }
 
        GetTime( timeSta );
        blas::Copy( nkq, VXi_fft.Data(), 1, outputComplexVec.Data(), 1 );
        fftw_execute( backwardPlan );
        blas::Copy( nkq, inputComplexVec.Data(), 1, VrNu.Data(), 1 );
        GetTime( timeEnd );
        timeFFT = timeFFT + timeEnd - timeSta;
        iterFFT = iterFFT + 1;        

        for( Int k = 0; k < nkTotal; k++ ){
          Index3 idxkq;
          for( Int d = 0; d < DIM; d++ ){
            idxkq(d) = int(kgrid[d][k] * numKGrid(d));
            if( idxkq(d) < 0 ) idxkq(d) += nkxyz[d];   
          }

          Int ikq = idxkq[2] * nkxyz[0] * nkxyz[1] + idxkq[1] * nkxyz[0] + idxkq[0];
          VrNu1D[k](i, nu) = VrNu(ikq) / double(nkq);
        }
      }
    }
  }

  {
    // V_vu is calculated by numMu * numMu times FFT
    CpxNumVec KMuNu_fft( nkq );
    CpxNumVec phiMuNu_fft( nkq );
    CpxNumVec VMuNu( nkq );

    for( Int nu = 0; nu < numMuLocal; nu++ ){
      for( Int mu = 0; mu < numMu; mu++ ){
        GetTime( timeSta );
        blas::Copy( nkq, KMuNu1D.VecData(mu,nu), 1, inputComplexVec.Data(), 1 );
        fftw_execute( forwardPlan );
        blas::Copy( nkq, outputComplexVec.Data(), 1, KMuNu_fft.Data(), 1 );

        blas::Copy( nkq, phiMuNuCol.VecData(mu,nu), 1, inputComplexVec.Data(), 1 );
        fftw_execute( forwardPlan );
        blas::Copy( nkq, outputComplexVec.Data(), 1, phiMuNu_fft.Data(), 1 );
        GetTime( timeEnd );

        timeFFT = timeFFT + timeEnd - timeSta;
        iterFFT = iterFFT + 2;

        for( Int k = 0; k < nkq; k++ ){
          KMuNu_fft(k) *= phiMuNu_fft(k);
        }

        GetTime( timeSta );
        blas::Copy( nkq, KMuNu_fft.Data(), 1, outputComplexVec.Data(), 1 );
        fftw_execute( backwardPlan );
        blas::Copy( nkq, inputComplexVec.Data(), 1, VMuNu.Data(), 1 );
        GetTime( timeEnd );
        timeFFT = timeFFT + timeEnd - timeSta;
        iterFFT = iterFFT + 1;

        for( Int k = 0; k < nkTotal; k++ ){
          Index3 idxkq;
          for( Int d = 0; d < DIM; d++ ){
            idxkq(d) = int(kgrid[d][k] * numKGrid(d));
            if( idxkq(d) < 0 ) idxkq(d) += nkxyz[d];
          }

          Int ikq = idxkq[2] * nkxyz[0] * nkxyz[1] + idxkq[1] * nkxyz[0] + idxkq[0];
          VMuNu1D[k](mu, nu) = VMuNu(ikq) / double(nkq);
        }
      }
    }
  }

  // Gather and store V_ru and V_vu according to the local k index 
  // the communication is performed in rowComm 
  std::vector<CpxNumMat> VrNuRow( nkLocal );
  std::vector<CpxNumMat> VMuNuCol( nkLocal );

  for( Int k = 0; k < nkLocal; k++ ){
    VrNuRow[k].Resize( ntotRow, numMu );
    VMuNuCol[k].Resize( numMu, numMuRow );
    
    SetValue( VrNuRow[k], Z_ZERO );
    SetValue( VMuNuCol[k], Z_ZERO );
  }

  {
    GetTime( timeSta );

    CpxNumMat VrNu1DT( numMu, ntotLocal );
    CpxNumMat VrNuRowT( numMu, ntotRow );
    
    IntNumVec localSize( mpisizeRow );
    IntNumVec localDispls( mpisizeRow );
    SetValue( localSize, 0 );
    SetValue( localDispls, 0 );

    Int numElem = numMu * ntotLocal;
    MPI_Allgather( &numElem, 1, MPI_INT, localSize.Data(), 1, MPI_INT, rowComm );

    for( Int i = 1; i < mpisizeRow; i++ ){
      localDispls[i] = localDispls[i-1] + localSize[i-1];
    }

    for( Int k = 0; k < nkTotal; k++ ){
      // Transpose VrNu1D to make data address continuous
      for( Int j = 0; j < numMu; j++ ){
        for( Int i = 0; i < ntotLocal; i++ ){
          VrNu1DT(j, i) = VrNu1D[k](i, j);
        }
      }

      MPI_Allgatherv( VrNu1DT.Data(), numElem, MPI_DOUBLE_COMPLEX, VrNuRowT.Data(),
          localSize.Data(), localDispls.Data(), MPI_DOUBLE_COMPLEX, rowComm );      

      for( Int ik = 0; ik < nkLocal; ik++ ){
        if( domain_.KpointIdx[ik] == k ){
          for( Int j = 0; j < numMu; j++ ){
            for( Int i = 0; i < ntotRow; i++ ){
              VrNuRow[ik](i,j) = VrNuRowT(j,i);
            }
          }
        }
      }
    }

    GetTime( timeEnd );
    statusOFS << "Time for gathering VrNuRow = " << timeEnd - timeSta << std::endl;
  }
  
  { 
    GetTime( timeSta );
    IntNumVec localSize( mpisizeRow );
    IntNumVec localDispls( mpisizeRow );
    SetValue( localSize, 0 );
    SetValue( localDispls, 0 );
    
    Int numElem = numMu * numMuLocal;
    MPI_Allgather( &numElem, 1, MPI_INT, localSize.Data(), 1, MPI_INT, rowComm );
    
    for( Int i = 1; i < mpisizeRow; i++ ){
      localDispls[i] = localDispls[i-1] + localSize[i-1];
    }
    
    CpxNumMat VMuNuColTemp( numMu, numMuRow );

    for( Int k = 0; k < nkTotal; k++ ){

      MPI_Allgatherv( VMuNu1D[k].Data(), numElem, MPI_DOUBLE_COMPLEX, VMuNuColTemp.Data(),
          localSize.Data(), localDispls.Data(), MPI_DOUBLE_COMPLEX, rowComm );

      for( Int ik = 0; ik < nkLocal; ik++ ){
        if( domain_.KpointIdx[ik] == k ){
          for( Int j = 0; j < numMuRow; j++ ){
            for( Int i = 0; i < numMu; i++ ){
              VMuNuCol[ik](i,j) = VMuNuColTemp(i,j);
            }
          }
        }
      }
    }
    GetTime( timeEnd );
    statusOFS << "Time for gathering KMuNuCol = " << timeEnd - timeSta << std::endl;
  }

  // Calculate matrix VxMat and vexxPsi
  for( Int k = 0; k < nkLocal; k++ ){

    CpxNumMat PsiMuCol( numMu, numStateLocal );
    CpxNumMat PsiMuRow( numMuRow, numStateTotal );
    CpxNumMat PsiMu( numMu, numStateTotal );

    GetTime( timeSta );
    for( Int j = 0; j < numStateLocal; j++ ){
      for( Int mu = 0; mu < numMu; mu++ ){
        PsiMuCol(mu, j) = psik[k].WavefunR()(pivMu(mu),0,j);
      }
    }
    GetTime( timeEnd );
    timeOther = timeOther + timeEnd - timeSta;
    iterOther = iterOther + 2;

    GetTime( timeSta );
    IntNumVec localSize( mpisizeCol );
    IntNumVec localDispls( mpisizeCol );
    SetValue( localSize, 0 );
    SetValue( localDispls, 0 );

    Int numElem = numMu * numStateLocal;
    MPI_Allgather( &numElem, 1, MPI_INT, localSize.Data(), 1, MPI_INT, colComm );

    for( Int i = 1; i < mpisizeCol; i++ ){
      localDispls[i] = localDispls[i-1] + localSize[i-1];
    }
     
    MPI_Allgatherv( PsiMuCol.Data(), numElem, MPI_DOUBLE_COMPLEX, PsiMu.Data(),
        localSize.Data(), localDispls.Data(), MPI_DOUBLE_COMPLEX, colComm );
    GetTime( timeEnd );
    timeGather = timeGather + timeEnd - timeSta;
    iterGather ++;

    Int Ndispls = ndisplsCol[mpirankCol];

    GetTime( timeSta );
    for( Int j = 0; j < numStateTotal; j++ ){
      for( Int i = 0; i < numMuRow; i++ ){
        PsiMuRow(i, j) = PsiMu(i+Ndispls, j);
      }
    }
    GetTime( timeEnd );
    timeOther = timeOther + timeEnd - timeSta;
    iterOther = iterOther + 1;

    CpxNumMat vexxPsiRow( ntotRow, numStateTotal );
    CpxNumMat vexxPsiCol( ntot, numStateLocal );    

    GetTime( timeSta );
    blas::Gemm( 'N', 'N', ntotRow, numStateTotal, numMu, 1.0, VrNuRow[k].Data(), ntotRow, 
        PsiMu.Data(), numMu, 0.0, vexxPsiRow.Data(), ntotRow );
    GetTime( timeEnd );
    timeGemm = timeGemm + timeEnd - timeSta;
    iterGemm ++;

    GetTime( timeSta );
    AlltoallBackward( idxGridRow, ncom, vexxPsiRow, vexxPsiCol, colComm ); 
    GetTime( timeEnd );
    timeAlltoall = timeAlltoall + timeEnd - timeSta;
    iterAlltoall ++;

    GetTime( timeSta );
    if( !realspace ){
      Int npw_this = ( spherecut == true ) ? domain_.numGridSphere[k] : ntot;
      IntNumVec &idxCoarseCut_this = fft.idxCoarseCut[k];


      for( Int kpsi = 0; kpsi < numStateLocal; kpsi++ ){
        Complex* a3Ptr = a3[k].VecData(0,kpsi);

        blas::Copy( ntot, vexxPsiCol.VecData(kpsi), 1, fft.inputComplexVec.Data(), 1 );

        fftw_execute( fft.forwardPlan );

        if( spherecut ){
          for( Int ig = 0; ig < npw_this; ig++ ){
            a3Ptr[ig] += fft.outputComplexVec(idxCoarseCut_this[ig]);
          }
        }
        else{
          blas::Axpy( npw_this, 1.0, fft.outputComplexVec.Data(), 1, a3Ptr, 1 );
        }
      }
    }
    GetTime( timeEnd );
    statusOFS << "Time for transforming vexxPsi to G space = " << timeEnd - timeSta << std::endl;

    CpxNumMat VxMatTemp( numStateTotal, numMuRow );

    GetTime( timeSta );
    blas::Gemm( 'C', 'N', numStateTotal, numMuRow, numMu, 1.0, PsiMu.Data(), numMu,
        VMuNuCol[k].Data(), numMu, 0.0, VxMatTemp.Data(), numStateTotal );
    GetTime( timeEnd );
    timeGemm = timeGemm + timeEnd - timeSta;
    iterGemm ++;

    CpxNumMat VxMatLocal( numStateTotal, numStateTotal );

    GetTime( timeSta );
    blas::Gemm( 'N', 'N', numStateTotal, numStateTotal, numMuRow, 1.0, VxMatTemp.Data(), 
        numStateTotal, PsiMuRow.Data(), numMuRow, 0.0, VxMatLocal.Data(), numStateTotal );
    GetTime( timeEnd );
    timeGemm = timeGemm + timeEnd - timeSta;
    iterGemm ++;
 
    GetTime( timeSta );
    MPI_Allreduce( VxMatLocal.Data(), VxMat[k].Data(), 2*numStateTotal*numStateTotal, 
        MPI_DOUBLE, MPI_SUM, colComm );

    GetTime( timeEnd );
    timeAllreduce = timeAllreduce + timeEnd - timeSta;
    iterAllreduce ++;
  } 

  MPI_Barrier(domain_.comm);

  GetTime( timeEnd1 );

  Real timeAlltoall1 = 0, timeBcast1 = 0.0, timeFFT1 = 0.0,
       timeCopy1 = 0.0, timeOther1 = 0.0, timeRank01 = 0.0,
       timeGemm1 = 0.0, timeGather1 = 0.0, timeAllreduce1 = 0.0;

  MPI_Allreduce( &timeAlltoall, &timeAlltoall1, 1, MPI_DOUBLE, MPI_MAX, domain_.comm );
  MPI_Allreduce( &timeBcast, &timeBcast1, 1, MPI_DOUBLE, MPI_MAX, domain_.comm );
  MPI_Allreduce( &timeFFT, &timeFFT1, 1, MPI_DOUBLE, MPI_MAX, domain_.comm );
  MPI_Allreduce( &timeCopy, &timeCopy1, 1, MPI_DOUBLE, MPI_MAX, domain_.comm );
  MPI_Allreduce( &timeOther, &timeOther1, 1, MPI_DOUBLE, MPI_MAX, domain_.comm );
  MPI_Allreduce( &timeRank0, &timeRank01, 1, MPI_DOUBLE, MPI_MAX, domain_.comm );
  MPI_Allreduce( &timeGemm, &timeGemm1, 1, MPI_DOUBLE, MPI_MAX, domain_.comm );
  MPI_Allreduce( &timeGather, &timeGather1, 1, MPI_DOUBLE, MPI_MAX, domain_.comm );
  MPI_Allreduce( &timeAllreduce, &timeAllreduce1, 1, MPI_DOUBLE, MPI_MAX, domain_.comm );  

  statusOFS << std::endl;
  statusOFS << "Total time for AddMultSpinorEXXDF is " << timeEnd1 - timeSta1 << std::endl;  
  statusOFS << "Time for FFT is " << timeFFT1 << " Number of FFT is " << iterFFT << std::endl;
  statusOFS << "Time for Alltoall is " << timeAlltoall1 << " Number of Alltoall is " << iterAlltoall << std::endl;
  statusOFS << "Time for Bcast is " << timeBcast1 << " Number of Bcast is " << iterBcast << std::endl;
  statusOFS << "Time for Gather is " << timeGather1 << " Number of Gather is " << iterGather << std::endl;
  statusOFS << "Time for Allreduce is " << timeAllreduce1 << " Number of Allreduce is " << iterAllreduce << std::endl;
  statusOFS << "Time for Gemm is " << timeGemm1 << " Number of Gemm is " << iterGemm << std::endl;
  statusOFS << "Time for Copy is " << timeCopy1 << " Number of Copy is " << iterCopy << std::endl;
  statusOFS << "Time for Rank0 is " << timeRank01 << " Number of Rank0 is " << iterRank0 << std::endl;
  statusOFS << "Time for Other is " << timeOther1 << " Number of Other is " << iterOther << std::endl;

  return;
}         // -----  end of method Spinor::AddMultSpinorEXXDF ( Complex version )  -----

void Spinor::Recip2Real( Fourier& fft )
{
  bool realspace    = esdfParam.isUseRealSpace;  

  Int ntot          = domain_.NumGridTotal();
  Int ncom          = numComponent_;
  Int numStateLocal = wavefun_.p();

  if( !realspace ){

    bool spherecut    = esdfParam.isUseSphereCut;

    Int npw_this      = ( spherecut == true ) ? domain_.numGridSphere[ikLocal_] : ntot;

    IntNumVec &idxCoarseCut_this = fft.idxCoarseCut[ikLocal_];

    wavefunR_.Resize( ntot, ncom, numStateLocal );
    for( Int j = 0; j < numStateLocal; j++ ){
      for( Int i = 0; i < ncom; i++ ){
        SetValue( fft.outputComplexVec, Z_ZERO );
        if( spherecut ){
          for( Int ig = 0; ig < npw_this; ig++ ){
            fft.outputComplexVec[idxCoarseCut_this[ig]] = wavefun_(ig,i,j);
          }
        }
        else{
         blas::Copy( npw_this, wavefun_.VecData(i,j), 1, fft.outputComplexVec.Data(), 1 );
        } 
        fftw_execute( fft.backwardPlan );
        blas::Copy( ntot, fft.inputComplexVec.Data(), 1, wavefunR_.VecData(i,j), 1 );
      }
    }
  }
  else{
    // Point to wavefun_ when real space method is used
    wavefunR_ = CpxNumTns( ntot, ncom, numStateLocal, false, wavefun_.Data() );
  }

  return;
}         // -----  end of method Spinor::Recip2Real ( Complex version )  -----

void Spinor::ISDF_SelectIP( CpxNumMat& psiCol, CpxNumMat& phiCol, 
    std::string& hybridDFType,
    std::string hybridDFKmeansWFType,
    const Real hybridDFKmeansWFAlpha,
    Real  hybridDFKmeansTolerance,
    Int   hybridDFKmeansMaxIter,
    const Real hybridDFTolerance,
    Int   numMu, const Real numGaussianRandomFac, 
    Int mb, Int nb )
{
  MPI_Barrier(domain_.comm);
  int mpirank; MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize; MPI_Comm_size(domain_.comm, &mpisize);

  // Select interpolation points for ISDF
  Int ntot = psiCol.m();
  if( ntot != phiCol.m() ){
    ErrorHandling("The grid number of Psi and Phi are different.");
  }
  Int ntotLocal;

  Int nbPsiLocal = psiCol.n();
  Int nbPhiLocal = phiCol.n();

  Int nbPsiTotal, nbPhiTotal;
  MPI_Allreduce( &nbPsiLocal, &nbPsiTotal, 1, MPI_INT, MPI_SUM, domain_.comm );
  MPI_Allreduce( &nbPhiLocal, &nbPhiTotal, 1, MPI_INT, MPI_SUM, domain_.comm );

  if( hybridDFType == "QRCP" ){
          
    Int numPrePsi = std::min( nbPsiTotal, 
        IRound(std::sqrt(numMu*numGaussianRandomFac*nbPsiTotal/double(nbPhiTotal))) );
    Int numPrePhi = std::min( nbPhiTotal, 
        IRound(std::sqrt(numMu*numGaussianRandomFac*nbPhiTotal/double(nbPsiTotal))) );
    Int numMG = numPrePsi * numPrePhi;
    Int numMGLocal;
    Int nprow2D, npcol2D;

    if( numMG > ntot ){
      std::ostringstream msg;
      msg << "numPre*numPre > ntot. The number of grid points is perhaps too small!" << std::endl;
      ErrorHandling( msg.str().c_str() );
    }

    Int contxt1, contxt11, contxt2D;
    Int desc_NgNe1DCol[9];
    Int desc_NgNo1DCol[9];   
    Int desc_NgNe1DRow[9];
    Int desc_NgNo1DRow[9];      

    Int desc_NpNg1DCol[9];
    Int desc_NpNg2D[9];

    {
      // Initialize ScaLAPACK partition used for QRCP method 
      Int Ng = ntot;
      Int Ne = nbPsiTotal;
      Int No = nbPhiTotal;

      Int I_ONE = 1, I_ZERO = 0;

      // 1D MPI ---- column partition
      Int nprow1, npcol1, myrow1, mycol1, info1;
      Int nrowsNgNe1DCol, ncolsNgNe1DCol, lldNgNe1DCol;
      Int nrowsNgNo1DCol, ncolsNgNo1DCol, lldNgNo1DCol;
      Int nrowsNpNg1DCol, ncolsNpNg1DCol, lldNpNg1DCol;
      
      nprow1 = 1;
      npcol1 = mpisize;

      Cblacs_get(0, 0, &contxt1);
      Cblacs_gridinit(&contxt1, "C", nprow1, npcol1);
      Cblacs_gridinfo(contxt1, &nprow1, &npcol1, &myrow1, &mycol1);

      // desc_NgNe1DCol
      if(contxt1 >= 0){
        nrowsNgNe1DCol = SCALAPACK(numroc)(&Ng, &Ng, &myrow1, &I_ZERO, &nprow1);
        ncolsNgNe1DCol = SCALAPACK(numroc)(&Ne, &I_ONE, &mycol1, &I_ZERO, &npcol1);
        if( ncolsNgNe1DCol != nbPsiLocal ){
          ErrorHandling("Psi does not satisfy scalapack partition.");
        }
        lldNgNe1DCol = std::max( nrowsNgNe1DCol, 1 );
      }

      SCALAPACK(descinit)(desc_NgNe1DCol, &Ng, &Ne, &Ng, &I_ONE, &I_ZERO,
          &I_ZERO, &contxt1, &lldNgNe1DCol, &info1);

      // desc_NgNo1DCol
      if(contxt1 >= 0){
        nrowsNgNo1DCol = SCALAPACK(numroc)(&Ng, &Ng, &myrow1, &I_ZERO, &nprow1);
        ncolsNgNo1DCol = SCALAPACK(numroc)(&No, &I_ONE, &mycol1, &I_ZERO, &npcol1);
        if( ncolsNgNo1DCol != nbPhiLocal ){
          ErrorHandling("Phi does not satisfy scalapack partition.");
        }
        lldNgNo1DCol = std::max( nrowsNgNo1DCol, 1 );
      }

      SCALAPACK(descinit)(desc_NgNo1DCol, &Ng, &No, &Ng, &I_ONE, &I_ZERO,
          &I_ZERO, &contxt1, &lldNgNo1DCol, &info1);

      // desc_NpNg1DCol
      if(contxt1 >= 0){
        nrowsNpNg1DCol = SCALAPACK(numroc)(&numMG, &numMG, &myrow1, &I_ZERO, &nprow1);
        ncolsNpNg1DCol = SCALAPACK(numroc)(&Ng, &mb, &mycol1, &I_ZERO, &npcol1);
        lldNpNg1DCol = std::max( nrowsNpNg1DCol, 1 );
      }

      SCALAPACK(descinit)(desc_NpNg1DCol, &numMG, &Ng, &numMG, &mb, &I_ZERO,
          &I_ZERO, &contxt1, &lldNpNg1DCol, &info1);

      // 1D MPI ---- row partition
      Int nprow11, npcol11, myrow11, mycol11, info11;
      Int nrowsNgNe1DRow, ncolsNgNe1DRow, lldNgNe1DRow;
      Int nrowsNgNo1DRow, ncolsNgNo1DRow, lldNgNo1DRow;

      nprow11 = mpisize;
      npcol11 = 1;

      Cblacs_get(0, 0, &contxt11);
      Cblacs_gridinit(&contxt11, "C", nprow11, npcol11);
      Cblacs_gridinfo(contxt11, &nprow11, &npcol11, &myrow11, &mycol11);

      // desc_NgNe1DRow
      if(contxt11 >= 0){
        nrowsNgNe1DRow = SCALAPACK(numroc)(&Ng, &mb, &myrow11, &I_ZERO, &nprow11);
        ncolsNgNe1DRow = SCALAPACK(numroc)(&Ne, &Ne, &mycol11, &I_ZERO, &npcol11);
        lldNgNe1DRow = std::max( nrowsNgNe1DRow, 1 );
      }

      SCALAPACK(descinit)(desc_NgNe1DRow, &Ng, &Ne, &mb, &Ne, &I_ZERO,
          &I_ZERO, &contxt11, &lldNgNe1DRow, &info11);

      // desc_NgNo1DRow
      if(contxt11 >= 0){
        nrowsNgNo1DRow = SCALAPACK(numroc)(&Ng, &mb, &myrow11, &I_ZERO, &nprow11);
        ncolsNgNo1DRow = SCALAPACK(numroc)(&No, &No, &mycol11, &I_ZERO, &npcol11);
        lldNgNo1DRow = std::max( nrowsNgNo1DRow, 1 );
      }

      SCALAPACK(descinit)(desc_NgNo1DRow, &Ng, &No, &mb, &No, &I_ZERO,
          &I_ZERO, &contxt11, &lldNgNo1DRow, &info11);  
    
      ntotLocal = nrowsNgNo1DRow;

      // 2D MPI
      Int myrow2D, mycol2D, info2D;
      Int ncols2D, nrows2D, lld2D;

      for( Int i = std::min(mpisize, IRound(sqrt(double(mpisize*(ntot/double(numMG))))));
          i <= mpisize; i++ ){
        npcol2D = i; nprow2D = mpisize / npcol2D;
        if( (npcol2D >= nprow2D) && (nprow2D * npcol2D == mpisize) ) break;
      }

      Cblacs_get(0, 0, &contxt2D);

      IntNumVec pmap(mpisize);
      for ( Int i = 0; i < mpisize; i++ ){
        pmap[i] = i;
      }
      Cblacs_gridmap(&contxt2D, &pmap[0], nprow2D, nprow2D, npcol2D);

      if(contxt2D >= 0){
        Cblacs_gridinfo(contxt2D, &nprow2D, &npcol2D, &myrow2D, &mycol2D);
        nrows2D = SCALAPACK(numroc)(&numMG, &mb, &myrow2D, &I_ZERO, &nprow2D);
        ncols2D = SCALAPACK(numroc)(&Ng, &mb, &mycol2D, &I_ZERO, &npcol2D);
        lld2D = std::max( nrows2D, 1 );
      }

      SCALAPACK(descinit)(desc_NpNg2D, &numMG, &Ng, &mb, &mb, &I_ZERO,
          &I_ZERO, &contxt2D, &lld2D, &info2D);

      numMGLocal = nrows2D;
    } // scalapack initialization

    CpxNumMat localpsiGRow( ntotLocal, numPrePsi );
    SetValue( localpsiGRow, Z_ZERO );

    CpxNumMat localphiGRow( ntotLocal, numPrePhi );
    SetValue( localphiGRow, Z_ZERO );
    
    // Gaussian random projection matrix
    CpxNumMat GPsi(nbPsiTotal, numPrePsi);
    SetValue( GPsi, Z_ZERO );

    CpxNumMat GPhi(nbPhiTotal, numPrePhi);
    SetValue( GPhi, Z_ZERO );

    CpxNumMat psiRow( ntotLocal, nbPsiTotal );
    SetValue( psiRow, Z_ZERO );

    CpxNumMat phiRow( ntotLocal, nbPhiTotal );
    SetValue( phiRow, Z_ZERO );

    // Transform Psi and Phi from column divided format to row divided format
    SCALAPACK(pzgemr2d)(&ntot, &nbPsiTotal, psiCol.Data(), &I_ONE, &I_ONE, desc_NgNe1DCol,
        psiRow.Data(), &I_ONE, &I_ONE, desc_NgNe1DRow, &contxt1 );

    SCALAPACK(pzgemr2d)(&ntot, &nbPhiTotal, phiCol.Data(), &I_ONE, &I_ONE, desc_NgNo1DCol,
        phiRow.Data(), &I_ONE, &I_ONE, desc_NgNo1DRow, &contxt1 );

    // Pre-compression of the wavefunctions. 
    // This uses multiplication with orthonormalized random Gaussian matrices
    if ( mpirank == 0 ) {
      GaussianRandom(GPsi);
      lapack::Orth( nbPsiTotal, numPrePsi, GPsi.Data(), nbPsiTotal );
      GaussianRandom(GPhi);
      lapack::Orth( nbPhiTotal, numPrePhi, GPhi.Data(), nbPhiTotal );
    }
    MPI_Bcast(GPsi.Data(), 2*nbPsiTotal*numPrePsi, MPI_DOUBLE, 0, domain_.comm);
    MPI_Bcast(GPhi.Data(), 2*nbPhiTotal*numPrePhi, MPI_DOUBLE, 0, domain_.comm);

    // TODO
    GPhi = GPsi;    

    blas::Gemm( 'N', 'N', ntotLocal, numPrePsi, nbPsiTotal, Z_ONE,
        psiRow.Data(), ntotLocal, GPsi.Data(), nbPsiTotal, Z_ZERO,
        localpsiGRow.Data(), ntotLocal );

    blas::Gemm( 'N', 'N', ntotLocal, numPrePhi, nbPhiTotal, Z_ONE,
        phiRow.Data(), ntotLocal, GPhi.Data(), nbPhiTotal, Z_ZERO,
        localphiGRow.Data(), ntotLocal );

    // Pivoted QR decomposition for the Hadamard product of
    // the compressed matrix. Transpose format for QRCP
    CpxNumMat MGCol( numMG, ntotLocal );

    for( Int j = 0; j < numPrePsi; j++ ){
      for( Int i = 0; i < numPrePhi; i++ ){
        for( Int ir = 0; ir < ntotLocal; ir++ ){
          MGCol(i+j*numPrePhi,ir) = localpsiGRow(ir,j) * std::conj(localphiGRow(ir,i));
        }
      }
    }

    CpxNumVec tau(ntot);
    SetValue( tau, Z_ZERO );

    pivQR_.Resize(ntot);
    SetValue( pivQR_, 0 ); // Important. Otherwise QRCP uses piv as initial guess
  
    {
      // ScaLAPACK QRCP 2D
      MPI_Comm colComm = MPI_COMM_NULL;

      Int mpirankRow, mpisizeRow, mpirankCol, mpisizeCol;

      MPI_Comm_split( domain_.comm, mpirank % nprow2D, mpirank, &colComm );
      MPI_Comm_rank(colComm, &mpirankCol);
      MPI_Comm_size(colComm, &mpisizeCol);

      IntNumVec pivQRTmp(ntot), pivQRLocal(ntot);

      CpxNumMat MG2D( numMGLocal, ntotLocal );

      SCALAPACK(pzgemr2d)(&numMG, &ntot, MGCol.Data(), &I_ONE, &I_ONE, desc_NpNg1DCol,
          MG2D.Data(), &I_ONE, &I_ONE, desc_NpNg2D, &contxt1 );

      if(contxt2D >= 0){
        SetValue( pivQRTmp, 0 );
        scalapack::QRCPF( numMG, ntot, MG2D.Data(), desc_NpNg2D, pivQRTmp.Data(), tau.Data() );
      }

      SetValue( pivQRLocal, 0 );
      for( Int j = 0; j < ntotLocal; j++ ){
        pivQRLocal[(j / mb) * mb * npcol2D + mpirankCol * mb + j % mb] = pivQRTmp[j];
      }

      SetValue( pivQR_, 0 );
      MPI_Allreduce( pivQRLocal.Data(), pivQR_.Data(), ntot, MPI_INT, MPI_SUM, colComm );

      if( colComm != MPI_COMM_NULL ) MPI_Comm_free( & colComm );
    } // ScaLAPACK QRCP

    if(contxt1 >= 0) {
      Cblacs_gridexit( contxt1 );
    }

    if(contxt11 >= 0) {
      Cblacs_gridexit( contxt11 );
    }

    if(contxt2D >= 0) {
      Cblacs_gridexit( contxt2D );
    }

    if (hybridDFType == "Kmeans+QRCP"){
      hybridDFType = "Kmeans";
    }
  } // if( hybridDFType == "QRCP" )
  else if( hybridDFType == "Kmeans" ){
    DblNumVec weight( ntot );
    Real* wp = weight.Data();

    DblNumVec phiW( ntot );
    SetValue( phiW, 0.0 );
    Real* phW = phiW.Data();
    Complex* ps = psiCol.Data();
    Complex* ph = phiCol.Data();

    for( Int j = 0; j < nbPsiLocal; j++ ){
      for( Int i = 0; i < ntot; i++ ){
        Real absps = std::norm(ps[i+j*ntot]);

        if( hybridDFKmeansWFType == "Add" ){
          phW[i] += std::pow(absps, hybridDFKmeansWFAlpha/2.0);
        }
      } 
    }

    for( Int j = 0; j < nbPhiLocal; j++ ){
      for( Int i = 0; i < ntot; i++ ){
        Real absph = std::norm(ph[i+j*ntot]);
        
        if( hybridDFKmeansWFType == "Add" ){
          phW[i] += std::pow(absph, hybridDFKmeansWFAlpha/2.0);
        }
      } 
    } 

    MPI_Barrier( domain_.comm );
    MPI_Reduce( phW, wp, ntot, MPI_DOUBLE, MPI_SUM, 0, domain_.comm );
    MPI_Bcast( wp, ntot, MPI_DOUBLE, 0, domain_.comm );

    int rk = numMu;

    pivQR_.Resize(ntot);
    SetValue( pivQR_, 0 ); // Important. Otherwise QRCP uses piv as initial guess

    KMEAN( ntot, weight, rk, hybridDFKmeansTolerance, hybridDFKmeansMaxIter, 
        hybridDFTolerance, domain_, pivQR_.Data() );
  } // ---- end of if( hybridDFType == "QRCP" ) ----

  return;
}         // -----  end of method Spinor::ISDF_SelectIP ( Complex version )  -----

void Spinor::ISDF_CalculateIV( Fourier& fft,
    CpxNumMat& psiCol, CpxNumMat& phiCol, 
    DblNumVec& occupationRate, const DblNumTns& exxgkk, 
    Real exxFraction, Int numMu, Int mb, Int nb, 
    std::vector<IntNumVec>& idxMu, IntNumVec& ndispls,
    CpxNumTns& VXiRow, CpxNumTns& KMuNuCol )
{
  MPI_Barrier(domain_.comm);
  int mpirank; MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize; MPI_Comm_size(domain_.comm, &mpisize);

  bool spherecut    = esdfParam.isUseSphereCut;
  bool fftconv      = esdfParam.isHybridFourierConv;

  Real timeSta, timeEnd;
  Real timeFFT = 0.0, timePGemm = 0.0, timeOther = 0.0, timePGemr = 0.0;
  Int iterFFT = 0, iterPGemm = 0, iterOther = 0, iterPGemr = 0;

  // Compute the interpolation matrix via the density matrix formulation
  // PhiMu is scaled by the occupation number to reflect the "true" density matrix
  Int ntot = psiCol.m();
  if( ntot != phiCol.m() ){
    ErrorHandling("The grid number of Psi and Phi are different.");
  }
  Int npw = ( spherecut == true ) ? fft.idxFineFock.m() : ntot;

  Int nbPsiLocal = psiCol.n();
  Int nbPhiLocal = phiCol.n();

  Int nbPsiTotal, nbPhiTotal;
  MPI_Allreduce( &nbPsiLocal, &nbPsiTotal, 1, MPI_INT, MPI_SUM, domain_.comm );
  MPI_Allreduce( &nbPhiLocal, &nbPhiTotal, 1, MPI_INT, MPI_SUM, domain_.comm );
 
  // ScaLAPACK initialization
  Int I_ONE = 1, I_ZERO = 0;
  Complex Z_MinusONE = Complex(-1.0, 0.0);

  Int Ng = ntot;
  Int Npw = npw;
  Int Ne = nbPsiTotal;
  Int No = nbPhiTotal;
  Int Nu = numMu;

  // 1D MPI ---- column partition
  Int contxt1;
  Int nprow1, npcol1, myrow1, mycol1, info1;
  Int nrowsNgNe1DCol, ncolsNgNe1DCol, lldNgNe1DCol;
  Int nrowsNgNo1DCol, ncolsNgNo1DCol, lldNgNo1DCol;
  Int nrowsNgNu1DCol, ncolsNgNu1DCol, lldNgNu1DCol;
  Int nrowsNpwNu1DCol, ncolsNpwNu1DCol, lldNpwNu1DCol;
  Int nrowsNuNu1DCol, ncolsNuNu1DCol, lldNuNu1DCol;
  Int desc_NgNe1DCol[9];
  Int desc_NgNo1DCol[9];
  Int desc_NgNu1DCol[9];
  Int desc_NpwNu1DCol[9];
  Int desc_NuNu1DCol[9];

  nprow1 = 1;
  npcol1 = mpisize;

  Cblacs_get(0, 0, &contxt1);
  Cblacs_gridinit(&contxt1, "C", nprow1, npcol1);
  Cblacs_gridinfo(contxt1, &nprow1, &npcol1, &myrow1, &mycol1);

  // desc_NgNe1DCol
  if(contxt1 >= 0){
    nrowsNgNe1DCol = SCALAPACK(numroc)(&Ng, &Ng, &myrow1, &I_ZERO, &nprow1);
    ncolsNgNe1DCol = SCALAPACK(numroc)(&Ne, &I_ONE, &mycol1, &I_ZERO, &npcol1);
    if( ncolsNgNe1DCol != nbPsiLocal ){
      ErrorHandling("Psi does not satisfy scalapack partition.");
    }
    lldNgNe1DCol = std::max( nrowsNgNe1DCol, 1 );
  }

  SCALAPACK(descinit)(desc_NgNe1DCol, &Ng, &Ne, &Ng, &I_ONE, &I_ZERO,
      &I_ZERO, &contxt1, &lldNgNe1DCol, &info1);

  // desc_NgNo1DCol
  if(contxt1 >= 0){
    nrowsNgNo1DCol = SCALAPACK(numroc)(&Ng, &Ng, &myrow1, &I_ZERO, &nprow1);
    ncolsNgNo1DCol = SCALAPACK(numroc)(&No, &I_ONE, &mycol1, &I_ZERO, &npcol1);
    if( ncolsNgNo1DCol != nbPhiLocal ){
      ErrorHandling("Phi does not satisfy scalapack partition.");
    }
    lldNgNo1DCol = std::max( nrowsNgNo1DCol, 1 );
  }

  SCALAPACK(descinit)(desc_NgNo1DCol, &Ng, &No, &Ng, &I_ONE, &I_ZERO,
      &I_ZERO, &contxt1, &lldNgNo1DCol, &info1);

  // desc_NgNu1DCol
  if(contxt1 >= 0){
    nrowsNgNu1DCol = SCALAPACK(numroc)(&Ng, &Ng, &myrow1, &I_ZERO, &nprow1);
    ncolsNgNu1DCol = SCALAPACK(numroc)(&Nu, &I_ONE, &mycol1, &I_ZERO, &npcol1);
    lldNgNu1DCol = std::max( nrowsNgNu1DCol, 1 );
  }

  SCALAPACK(descinit)(desc_NgNu1DCol, &Ng, &Nu, &Ng, &I_ONE, &I_ZERO,
      &I_ZERO, &contxt1, &lldNgNu1DCol, &info1);

  // desc_NpwNu1DCol
  if(contxt1 >= 0){
    nrowsNpwNu1DCol = SCALAPACK(numroc)(&Npw, &Npw, &myrow1, &I_ZERO, &nprow1);
    ncolsNpwNu1DCol = SCALAPACK(numroc)(&Nu, &I_ONE, &mycol1, &I_ZERO, &npcol1);
    lldNpwNu1DCol = std::max( nrowsNpwNu1DCol, 1 );
  }

  SCALAPACK(descinit)(desc_NpwNu1DCol, &Npw, &Nu, &Npw, &I_ONE, &I_ZERO,
      &I_ZERO, &contxt1, &lldNpwNu1DCol, &info1);

  // desc_NuNu1DCol
  if(contxt1 >= 0){
    nrowsNuNu1DCol = SCALAPACK(numroc)(&Nu, &Nu, &myrow1, &I_ZERO, &nprow1);
    ncolsNuNu1DCol = SCALAPACK(numroc)(&Nu, &I_ONE, &mycol1, &I_ZERO, &npcol1);
    lldNuNu1DCol = std::max( nrowsNuNu1DCol, 1 );
  }

  SCALAPACK(descinit)(desc_NuNu1DCol, &Nu, &Nu, &Nu, &I_ONE, &I_ZERO,
      &I_ZERO, &contxt1, &lldNuNu1DCol, &info1);

  // 1D MPI ---- row partition
  Int contxt11;
  Int nprow11, npcol11, myrow11, mycol11, info11;
  Int ncolsNeNu1DRow, nrowsNeNu1DRow, lldNeNu1DRow;
  Int ncolsNoNu1DRow, nrowsNoNu1DRow, lldNoNu1DRow;
  Int ncolsNgNu1DRow, nrowsNgNu1DRow, lldNgNu1DRow;
  Int desc_NeNu1DRow[9];
  Int desc_NoNu1DRow[9];
  Int desc_NgNu1DRow[9];

  nprow11 = mpisize;
  npcol11 = 1;

  Cblacs_get(0, 0, &contxt11);
  Cblacs_gridinit(&contxt11, "C", nprow11, npcol11);
  Cblacs_gridinfo(contxt11, &nprow11, &npcol11, &myrow11, &mycol11);

  // desc_NeNu1DRow
  if(contxt11 >= 0){
    nrowsNeNu1DRow = SCALAPACK(numroc)(&Ne, &I_ONE, &myrow11, &I_ZERO, &nprow11);
    ncolsNeNu1DRow = SCALAPACK(numroc)(&Nu, &Nu, &mycol11, &I_ZERO, &npcol11);
    lldNeNu1DRow = std::max( nrowsNeNu1DRow, 1 );
  }

  SCALAPACK(descinit)(desc_NeNu1DRow, &Ne, &Nu, &I_ONE, &Nu, &I_ZERO,
      &I_ZERO, &contxt11, &lldNeNu1DRow, &info11);

  // desc_NoNu1DRow
  if(contxt11 >= 0){
    nrowsNoNu1DRow = SCALAPACK(numroc)(&No, &I_ONE, &myrow11, &I_ZERO, &nprow11);
    ncolsNoNu1DRow = SCALAPACK(numroc)(&Nu, &Nu, &mycol11, &I_ZERO, &npcol11);
    lldNoNu1DRow = std::max( nrowsNoNu1DRow, 1 );
  }

  SCALAPACK(descinit)(desc_NoNu1DRow, &No, &Nu, &I_ONE, &Nu, &I_ZERO,
      &I_ZERO, &contxt11, &lldNoNu1DRow, &info11);

  // desc_NgNu1DRow
  if(contxt11 >= 0){
    nrowsNgNu1DRow = SCALAPACK(numroc)(&Ng, &mb, &myrow11, &I_ZERO, &nprow11);
    ncolsNgNu1DRow = SCALAPACK(numroc)(&Nu, &Nu, &mycol11, &I_ZERO, &npcol11);
    lldNgNu1DRow = std::max( nrowsNgNu1DRow, 1 );
  }

  SCALAPACK(descinit)(desc_NgNu1DRow, &Ng, &Nu, &mb, &Nu, &I_ZERO,
      &I_ZERO, &contxt11, &lldNgNu1DRow, &info11);

  // 2D MPI
  Int contxt2;
  Int nprow2, npcol2, myrow2, mycol2, info2;
  Int ncolsNeNu2D, nrowsNeNu2D, lldNeNu2D;
  Int ncolsNoNu2D, nrowsNoNu2D, lldNoNu2D;
  Int nrowsNgNe2D, ncolsNgNe2D, lldNgNe2D;
  Int nrowsNgNo2D, ncolsNgNo2D, lldNgNo2D;
  Int nrowsNgNu2D, ncolsNgNu2D, lldNgNu2D;
  Int nrowsNpwNu2D, ncolsNpwNu2D, lldNpwNu2D;
  Int ncolsNuNg2D, nrowsNuNg2D, lldNuNg2D;
  Int ncolsNuNu2D, nrowsNuNu2D, lldNuNu2D;

  Int desc_NeNu2D[9];
  Int desc_NoNu2D[9];
  Int desc_NgNe2D[9];
  Int desc_NgNo2D[9];
  Int desc_NgNu2D[9];
  Int desc_NpwNu2D[9];
  Int desc_NuNg2D[9];
  Int desc_NuNu2D[9];

  for( Int i = IRound(sqrt(double(mpisize))); i <= mpisize; i++){
    nprow2 = i; npcol2 = mpisize / nprow2;
    if( (nprow2 >= npcol2) && (nprow2 * npcol2 == mpisize) ) break;
  }

  Cblacs_get(0, 0, &contxt2);

  IntNumVec pmap2(mpisize);
  for ( Int i = 0; i < mpisize; i++ ){
    pmap2[i] = i;
  }
  Cblacs_gridmap(&contxt2, &pmap2[0], nprow2, nprow2, npcol2);

  Int mb2 = mb;
  Int nb2 = mb;

  // desc_NeNu2D
  if(contxt2 >= 0){
    Cblacs_gridinfo(contxt2, &nprow2, &npcol2, &myrow2, &mycol2);
    nrowsNeNu2D = SCALAPACK(numroc)(&Ne, &mb2, &myrow2, &I_ZERO, &nprow2);
    ncolsNeNu2D = SCALAPACK(numroc)(&Nu, &nb2, &mycol2, &I_ZERO, &npcol2);
    lldNeNu2D = std::max( nrowsNeNu2D, 1 );
  }

  SCALAPACK(descinit)(desc_NeNu2D, &Ne, &Nu, &mb2, &nb2, &I_ZERO,
      &I_ZERO, &contxt2, &lldNeNu2D, &info2);

  // desc_NoNu2D
  if(contxt2 >= 0){
    Cblacs_gridinfo(contxt2, &nprow2, &npcol2, &myrow2, &mycol2);
    nrowsNoNu2D = SCALAPACK(numroc)(&No, &mb2, &myrow2, &I_ZERO, &nprow2);
    ncolsNoNu2D = SCALAPACK(numroc)(&Nu, &nb2, &mycol2, &I_ZERO, &npcol2);
    lldNoNu2D = std::max( nrowsNoNu2D, 1 );
  }

  SCALAPACK(descinit)(desc_NoNu2D, &No, &Nu, &mb2, &nb2, &I_ZERO,
      &I_ZERO, &contxt2, &lldNoNu2D, &info2);

  // desc_NgNe2D
  if(contxt2 >= 0){
    Cblacs_gridinfo(contxt2, &nprow2, &npcol2, &myrow2, &mycol2);
    nrowsNgNe2D = SCALAPACK(numroc)(&Ng, &mb2, &myrow2, &I_ZERO, &nprow2);
    ncolsNgNe2D = SCALAPACK(numroc)(&Ne, &nb2, &mycol2, &I_ZERO, &npcol2);
    lldNgNe2D = std::max( nrowsNgNe2D, 1 );
  }

  SCALAPACK(descinit)(desc_NgNe2D, &Ng, &Ne, &mb2, &nb2, &I_ZERO,
      &I_ZERO, &contxt2, &lldNgNe2D, &info2);

  // desc_NgNo2D
  if(contxt2 >= 0){
    Cblacs_gridinfo(contxt2, &nprow2, &npcol2, &myrow2, &mycol2);
    nrowsNgNo2D = SCALAPACK(numroc)(&Ng, &mb2, &myrow2, &I_ZERO, &nprow2);
    ncolsNgNo2D = SCALAPACK(numroc)(&No, &nb2, &mycol2, &I_ZERO, &npcol2);
    lldNgNo2D = std::max( nrowsNgNo2D, 1 );
  }

  SCALAPACK(descinit)(desc_NgNo2D, &Ng, &No, &mb2, &nb2, &I_ZERO,
      &I_ZERO, &contxt2, &lldNgNo2D, &info2);

  // desc_NgNu2D
  if(contxt2 >= 0){
    Cblacs_gridinfo(contxt2, &nprow2, &npcol2, &myrow2, &mycol2);
    nrowsNgNu2D = SCALAPACK(numroc)(&Ng, &mb2, &myrow2, &I_ZERO, &nprow2);
    ncolsNgNu2D = SCALAPACK(numroc)(&Nu, &nb2, &mycol2, &I_ZERO, &npcol2);
    lldNgNu2D = std::max( nrowsNgNu2D, 1 );
  }

  SCALAPACK(descinit)(desc_NgNu2D, &Ng, &Nu, &mb2, &nb2, &I_ZERO,
      &I_ZERO, &contxt2, &lldNgNu2D, &info2);

  // desc_NpwNu2D
  if(contxt2 >= 0){
    Cblacs_gridinfo(contxt2, &nprow2, &npcol2, &myrow2, &mycol2);
    nrowsNpwNu2D = SCALAPACK(numroc)(&Npw, &mb2, &myrow2, &I_ZERO, &nprow2);
    ncolsNpwNu2D = SCALAPACK(numroc)(&Nu, &nb2, &mycol2, &I_ZERO, &npcol2);
    lldNpwNu2D = std::max( nrowsNpwNu2D, 1 );
  }

  SCALAPACK(descinit)(desc_NpwNu2D, &Npw, &Nu, &mb2, &nb2, &I_ZERO,
      &I_ZERO, &contxt2, &lldNpwNu2D, &info2);

  // desc_NuNg2D
  if(contxt2 >= 0){
    Cblacs_gridinfo(contxt2, &nprow2, &npcol2, &myrow2, &mycol2);
    nrowsNuNg2D = SCALAPACK(numroc)(&Nu, &mb2, &myrow2, &I_ZERO, &nprow2);
    ncolsNuNg2D = SCALAPACK(numroc)(&Ng, &nb2, &mycol2, &I_ZERO, &npcol2);
    lldNuNg2D = std::max( nrowsNuNg2D, 1 );
  }

  SCALAPACK(descinit)(desc_NuNg2D, &Nu, &Ng, &mb2, &nb2, &I_ZERO,
      &I_ZERO, &contxt2, &lldNuNg2D, &info2);

  // desc_NuNu2D
  if(contxt2 >= 0){
    Cblacs_gridinfo(contxt2, &nprow2, &npcol2, &myrow2, &mycol2);
    nrowsNuNu2D = SCALAPACK(numroc)(&Nu, &mb2, &myrow2, &I_ZERO, &nprow2);
    ncolsNuNu2D = SCALAPACK(numroc)(&Nu, &nb2, &mycol2, &I_ZERO, &npcol2);
    lldNuNu2D = std::max( nrowsNuNu2D, 1 );
  }

  SCALAPACK(descinit)(desc_NuNu2D, &Nu, &Nu, &mb2, &nb2, &I_ZERO,
      &I_ZERO, &contxt2, &lldNuNu2D, &info2);

  GetTime( timeSta );

  IntNumVec& pivMu = pivQR_;

  CpxNumMat psiMuCol( nbPsiLocal, numMu );
  CpxNumMat phiMuCol( nbPhiLocal, numMu );
  CpxNumMat phiMuCol1( nbPhiLocal, numMu );
  SetValue( psiMuCol, Z_ZERO );
  SetValue( phiMuCol, Z_ZERO );
  SetValue( phiMuCol1, Z_ZERO );

  for( Int k = 0; k < nbPsiLocal; k++ ){
    for( Int mu = 0; mu < numMu; mu++ ){
      psiMuCol(k, mu) = std::conj(psiCol(pivMu(mu),k));
    }
  }

  for( Int k = 0; k < nbPhiLocal; k++ ){
    for( Int mu = 0; mu < numMu; mu++ ){
      phiMuCol(k, mu) = std::conj(phiCol(pivMu(mu),k));
      phiMuCol1(k, mu) = std::conj(phiCol(pivMu(mu),k)) * occupationRate[k];
    }
  }

  CpxNumMat psiMu2D( nrowsNeNu2D, ncolsNeNu2D );
  CpxNumMat phiMu2D( nrowsNoNu2D, ncolsNoNu2D );
  SetValue( psiMu2D, Z_ZERO );
  SetValue( phiMu2D, Z_ZERO );

  SCALAPACK(pzgemr2d)(&Ne, &Nu, psiMuCol.Data(), &I_ONE, &I_ONE, desc_NeNu1DRow,
      psiMu2D.Data(), &I_ONE, &I_ONE, desc_NeNu2D, &contxt11 );

  SCALAPACK(pzgemr2d)(&No, &Nu, phiMuCol1.Data(), &I_ONE, &I_ONE, desc_NoNu1DRow,
      phiMu2D.Data(), &I_ONE, &I_ONE, desc_NoNu2D, &contxt11 );

  CpxNumMat psi2D( nrowsNgNe2D, ncolsNgNe2D );
  CpxNumMat phi2D( nrowsNgNo2D, ncolsNgNo2D );
  SetValue( psi2D, Z_ZERO );
  SetValue( phi2D, Z_ZERO );

  SCALAPACK(pzgemr2d)(&Ng, &Ne, psiCol.Data(), &I_ONE, &I_ONE, desc_NgNe1DCol,
      psi2D.Data(), &I_ONE, &I_ONE, desc_NgNe2D, &contxt1 );

  SCALAPACK(pzgemr2d)(&Ng, &No, phiCol.Data(), &I_ONE, &I_ONE, desc_NgNo1DCol,
      phi2D.Data(), &I_ONE, &I_ONE, desc_NgNo2D, &contxt1 );

  CpxNumMat PpsiMu2D( nrowsNgNu2D, ncolsNgNu2D );
  CpxNumMat PphiMu2D( nrowsNgNu2D, ncolsNgNu2D );
  SetValue( PpsiMu2D, Z_ZERO );
  SetValue( PphiMu2D, Z_ZERO );

  SCALAPACK(pzgemm)("N", "N", &Ng, &Nu, &Ne,
      &Z_ONE,
      psi2D.Data(), &I_ONE, &I_ONE, desc_NgNe2D,
      psiMu2D.Data(), &I_ONE, &I_ONE, desc_NeNu2D,
      &Z_ZERO,
      PpsiMu2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D);

  SCALAPACK(pzgemm)("N", "N", &Ng, &Nu, &No,
      &Z_ONE,
      phi2D.Data(), &I_ONE, &I_ONE, desc_NgNo2D,
      phiMu2D.Data(), &I_ONE, &I_ONE, desc_NoNu2D,
      &Z_ZERO,
      PphiMu2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D);

  CpxNumMat Xi2D( nrowsNgNu2D, ncolsNgNu2D );
  SetValue( Xi2D, Z_ZERO );

  Complex* Xi2DPtr = Xi2D.Data();
  Complex* PpsiMu2DPtr = PpsiMu2D.Data();
  Complex* PphiMu2DPtr = PphiMu2D.Data();

  for( Int g = 0; g < nrowsNgNu2D * ncolsNgNu2D; g++ ){
    Xi2DPtr[g] = std::conj(PpsiMu2DPtr[g]) * PphiMu2DPtr[g];
  }

  CpxNumMat Xi1D( nrowsNgNu1DCol, ncolsNgNu1DCol );
  SetValue( Xi1D, Z_ZERO );

  SCALAPACK(pzgemr2d)( &Ng, &Nu, Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D,
      Xi1D.Data(), &I_ONE, &I_ONE, desc_NgNu1DCol, &contxt2 );

  CpxNumMat PMuNu1D( nrowsNuNu1DCol, ncolsNuNu1DCol );
  SetValue( PMuNu1D, Z_ZERO );

  for( Int mu = 0; mu < nrowsNuNu1DCol; mu++ ){
    for( Int nu = 0; nu < ncolsNuNu1DCol; nu++ ){
      PMuNu1D(mu, nu) = Xi1D(pivMu(mu),nu);
    }
  }

  CpxNumMat PMuNu2D( nrowsNuNu2D, ncolsNuNu2D );
  SetValue( PMuNu2D, Z_ZERO );

  SCALAPACK(pzgemr2d)( &Nu, &Nu, PMuNu1D.Data(), &I_ONE, &I_ONE, desc_NuNu1DCol,
      PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D, &contxt1 );

  Int LSmethod = esdfParam.hybridDFLSmethod;  

  if( LSmethod == 0 ){
    SCALAPACK(pzpotrf)("L", &Nu, PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D, &info2);

    SCALAPACK(pztrsm)("R", "L", "T", "N", &Ng, &Nu, &Z_ONE,
        PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D,
        Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D);

    SCALAPACK(pztrsm)("R", "L", "N", "N", &Ng, &Nu, &Z_ONE,
        PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D,
        Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D);
  }
  else if( LSmethod == 1 ){ 
    SCALAPACK(pzpotrf)("L", &Nu, PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D, &info2);
    SCALAPACK(pzpotri)("L", &Nu, PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D, &info2);

    CpxNumMat PMuNu2DTemp(nrowsNuNu2D, ncolsNuNu2D);
    SetValue( PMuNu2DTemp, Z_ZERO );

    lapack::Lacpy( 'A', nrowsNuNu2D, ncolsNuNu2D, PMuNu2D.Data(),
        nrowsNuNu2D, PMuNu2DTemp.Data(), nrowsNuNu2D );

    SCALAPACK(pztradd)("U", "C", &Nu, &Nu,
        &Z_ONE,
        PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D,
        &Z_ZERO,
        PMuNu2DTemp.Data(), &I_ONE, &I_ONE, desc_NuNu2D);

    CpxNumMat Xi2DTemp(nrowsNgNu2D, ncolsNgNu2D);
    SetValue( Xi2DTemp, Z_ZERO );

    SCALAPACK(pzgemm)("N", "N", &Ng, &Nu, &Nu,
        &Z_ONE,
        Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D,
        PMuNu2DTemp.Data(), &I_ONE, &I_ONE, desc_NuNu2D,
        &Z_ZERO,
        Xi2DTemp.Data(), &I_ONE, &I_ONE, desc_NgNu2D);

    SetValue( Xi2D, Z_ZERO );
    lapack::Lacpy( 'A', nrowsNgNu2D, ncolsNgNu2D, Xi2DTemp.Data(),
        nrowsNgNu2D, Xi2D.Data(), nrowsNgNu2D );
  }
  else if( LSmethod == 2 ){
    SCALAPACK(pzpotrf)("L", &Nu, PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D, &info2);
    SCALAPACK(pzpotri)("L", &Nu, PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D, &info2);

    CpxNumMat Xi2DTemp(nrowsNgNu2D, ncolsNgNu2D);
    SetValue( Xi2DTemp, Z_ZERO );

    SCALAPACK(pzsymm)("R", "L", &Ng, &Nu,
        &Z_ONE,
        PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D,
        Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D,
        &Z_ZERO,
        Xi2DTemp.Data(), &I_ONE, &I_ONE, desc_NgNu2D);

    SetValue( Xi2D, Z_ZERO );
    lapack::Lacpy( 'A', nrowsNgNu2D, ncolsNgNu2D, Xi2DTemp.Data(), nrowsNgNu2D, Xi2D.Data(), nrowsNgNu2D );
  }
  else if( LSmethod == 3 ){
    CpxNumMat Xi2DTemp(nrowsNuNg2D, ncolsNuNg2D);
    SetValue( Xi2DTemp, Z_ZERO );

    CpxNumMat PMuNu2DTemp(ncolsNuNu2D, nrowsNuNu2D);
    SetValue( PMuNu2DTemp, Z_ZERO );

    SCALAPACK(pzgeadd)("T", &Nu, &Ng,
        &Z_ONE,
        Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D,
        &Z_ZERO,
        Xi2DTemp.Data(), &I_ONE, &I_ONE, desc_NuNg2D);

    SCALAPACK(pzgeadd)("T", &Nu, &Nu,
        &Z_ONE,
        PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D,
        &Z_ZERO,
        PMuNu2DTemp.Data(), &I_ONE, &I_ONE, desc_NuNu2D);

    Int lwork=-1, info;
    Complex dummyWork;

    SCALAPACK(pzgels)("N", &Nu, &Nu, &Ng,
        PMuNu2DTemp.Data(), &I_ONE, &I_ONE, desc_NuNu2D,
        Xi2DTemp.Data(), &I_ONE, &I_ONE, desc_NuNg2D,
        &dummyWork, &lwork, &info);

    lwork = dummyWork.real();
    std::vector<Complex> work(lwork);

    SCALAPACK(pzgels)("N", &Nu, &Nu, &Ng,
        PMuNu2DTemp.Data(), &I_ONE, &I_ONE, desc_NuNu2D,
        Xi2DTemp.Data(), &I_ONE, &I_ONE, desc_NuNg2D,
        &work[0], &lwork, &info);

    SetValue( Xi2D, Z_ZERO );
    SCALAPACK(pzgeadd)("T", &Ng, &Nu,
        &Z_ONE,
        Xi2DTemp.Data(), &I_ONE, &I_ONE, desc_NuNg2D,
        &Z_ZERO,
        Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D);
  }
 
  GetTime( timeEnd );
  statusOFS << std::endl << "Time for calculating IVs for ISDF = " << timeEnd - timeSta << std::endl;

  GetTime( timeSta );
  // Convert Xi from 2D format to 1D column partitioned format
  SCALAPACK(pzgemr2d)(&Ng, &Nu, Xi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D,
      Xi1D.Data(), &I_ONE, &I_ONE, desc_NgNu1DCol, &contxt2 );
  GetTime( timeEnd );
  statusOFS << "Time for transforming Xi from 2D to 1D = " << timeEnd - timeSta << std::endl;

  // Calculate P_vu(k-q) and P_ru(k-q) globally
  Int Nkq = exxgkk.n();
  if( !fftconv ){
    VXiRow.Resize( nrowsNgNu1DRow, ncolsNgNu1DRow, Nkq );
    KMuNuCol.Resize( nrowsNuNu1DCol, ncolsNuNu1DCol, Nkq );
  }
  else{
    VXiRow.Resize( Nkq, nrowsNgNu1DRow, ncolsNgNu1DRow );
    KMuNuCol.Resize( Nkq, nrowsNuNu1DCol, ncolsNuNu1DCol );
  }

  GetTime( timeSta );
  CpxNumMat XiG1D( npw, ncolsNgNu1DCol );
  for( Int mu = 0; mu < ncolsNgNu1DCol; mu++ ){
    blas::Copy( ntot,  Xi1D.VecData(mu), 1, fft.inputComplexVec.Data(), 1 );

    FFTWExecute ( fft, fft.forwardPlan );

    Complex* XiGPtr = XiG1D.VecData(mu);

    for( Int ig = 0; ig < npw; ig++ ){
      Int idx = fft.idxFineFock[ig];
      *(XiGPtr++) = fft.outputComplexVec(idx);
    }
  }
  GetTime( timeEnd );
  statusOFS << "Time for transforming Xi to G space = " << timeEnd - timeSta << std::endl;

  Real eps = -1e-16;

  for( Int k = 0; k < Nkq; k++ ){

    CpxNumMat sqrtVXiGCol( nrowsNpwNu1DCol, ncolsNpwNu1DCol );
    CpxNumMat sqrtVXiG2D( nrowsNpwNu2D, ncolsNpwNu2D );

    CpxNumMat VXiCol( nrowsNgNu1DCol, ncolsNgNu1DCol );
    CpxNumMat VXi2D( nrowsNgNu2D, ncolsNgNu2D );

    SetValue( fft.outputComplexVec, Z_ZERO );
    for( Int mu = 0; mu < ncolsNgNu1DCol; mu++ ){
      GetTime( timeSta );
      for( Int ig = 0; ig < npw; ig++ ){
        Int idx = fft.idxFineFock[ig];
        fft.outputComplexVec(idx) = -exxFraction * XiG1D(ig,mu) * exxgkk(ig, k, 0);
        if( exxgkk(ig, k, 0) < eps ){
          sqrtVXiGCol(ig,mu) = XiG1D(ig,mu) * Complex( 0.0, std::sqrt(-exxFraction * exxgkk(ig, k, 0)) );
        }
        else{
          sqrtVXiGCol(ig,mu) = XiG1D(ig,mu) * std::sqrt(exxFraction * exxgkk(ig, k, 0));
        }
      }
      GetTime( timeEnd );
      timeOther = timeOther + timeEnd - timeSta;
      iterOther ++;

      GetTime( timeSta );
      FFTWExecute ( fft, fft.backwardPlan );
      GetTime( timeEnd );
      timeFFT = timeFFT + timeEnd - timeSta;
      iterFFT ++;

      blas::Copy( ntot, fft.inputComplexVec.Data(), 1, VXiCol.VecData(mu), 1 );
    } // for (mu)

    GetTime( timeSta );
    if( !fftconv ){
      SCALAPACK(pzgemr2d)(&Ng, &Nu, VXiCol.Data(), &I_ONE, &I_ONE, desc_NgNu1DCol,
          VXiRow.MatData(k), &I_ONE, &I_ONE, desc_NgNu1DRow, &contxt1 );
    }
    else{
      CpxNumMat VXiRowTemp( nrowsNgNu1DRow, ncolsNgNu1DRow );
      SCALAPACK(pzgemr2d)(&Ng, &Nu, VXiCol.Data(), &I_ONE, &I_ONE, desc_NgNu1DCol,
          VXiRowTemp.Data(), &I_ONE, &I_ONE, desc_NgNu1DRow, &contxt1 );

      for( Int irank = 0; irank < idxMu.size(); irank++ ){
        IntNumVec& idxn = idxMu[irank];
        for( Int j = 0; j < idxn.m(); j++ ){
          for( Int i = 0; i < nrowsNgNu1DRow; i++ ){
            // Rearrange mu index
            VXiRow(k, i, ndispls[irank] + j) = VXiRowTemp(i, idxn[j]);
          } 
        }
      }
    }
    
    SCALAPACK(pzgemr2d)(&Ng, &Nu, VXiCol.Data(), &I_ONE, &I_ONE, desc_NgNu1DCol,
        VXi2D.Data(), &I_ONE, &I_ONE, desc_NgNu2D, &contxt1 );

    SCALAPACK(pzgemr2d)(&Npw, &Nu, sqrtVXiGCol.Data(), &I_ONE, &I_ONE, desc_NpwNu1DCol,
        sqrtVXiG2D.Data(), &I_ONE, &I_ONE, desc_NpwNu2D, &contxt1 );
    GetTime( timeEnd );
    timePGemr = timePGemr + timeEnd - timeSta;
    iterPGemr = iterPGemr + 2;

    GetTime( timeSta );
    Complex fac = Complex( Ng / (domain_.Volume()*domain_.Volume()), 0.0 );

    if( exxgkk(0, k, 0) < eps ){
      CpxNumMat sqrtVXiG2D1 = sqrtVXiG2D;
      if( myrow2 == 0 ){
        for( Int j = 0; j < ncolsNuNu2D; j++ ){
          sqrtVXiG2D1(0, j) = -sqrtVXiG2D1(0, j);
        }
      }
      SCALAPACK(pzgemm)("C", "N", &Nu, &Nu, &Npw,
          &fac,
          sqrtVXiG2D.Data(), &I_ONE, &I_ONE, desc_NpwNu2D,
          sqrtVXiG2D1.Data(), &I_ONE, &I_ONE, desc_NpwNu2D,
          &Z_ZERO,
          PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D);
    }
    else{
      SCALAPACK(pzgemm)("C", "N", &Nu, &Nu, &Npw,
          &fac,
          sqrtVXiG2D.Data(), &I_ONE, &I_ONE, desc_NpwNu2D,
          sqrtVXiG2D.Data(), &I_ONE, &I_ONE, desc_NpwNu2D,
          &Z_ZERO,
          PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D);
    }
    GetTime( timeEnd );
    timePGemm = timePGemm + timeEnd - timeSta;
    iterPGemm ++;

    GetTime( timeSta );
    if( !fftconv ){
      SCALAPACK(pzgemr2d)(&Nu, &Nu, PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D,
          KMuNuCol.MatData(k), &I_ONE, &I_ONE, desc_NuNu1DCol, &contxt2 );
    }
    else{
      CpxNumMat KMuNuColTemp( nrowsNuNu1DCol, ncolsNuNu1DCol );
      SCALAPACK(pzgemr2d)(&Nu, &Nu, PMuNu2D.Data(), &I_ONE, &I_ONE, desc_NuNu2D,
          KMuNuColTemp.Data(), &I_ONE, &I_ONE, desc_NuNu1DCol, &contxt2 );
     
      for( Int irank = 0; irank < idxMu.size(); irank++ ){
        IntNumVec& idxn = idxMu[irank];
        for( Int j = 0; j < ncolsNuNu1DCol; j++ ){
          for( Int i = 0; i < idxn.m(); i++ ){
            KMuNuCol(k, ndispls[irank] + i, j) = KMuNuColTemp(idxn[i],j);
          }
        }
      }
    }

    GetTime( timeEnd );
    timePGemr = timePGemr + timeEnd - timeSta;
    iterPGemr ++;
  } // for (k)

  statusOFS << std::endl;
  statusOFS << "Time for FFT is " << timeFFT << " Number of FFT is " << iterFFT << std::endl;
  statusOFS << "Time for Other is " << timeOther << " Number of Other is " << iterOther << std::endl;
  statusOFS << "Time for PGemm is " << timePGemm << " Number of PGemm is " << iterPGemm << std::endl;
  statusOFS << "Time for PGemr is " << timePGemr << " Number of PGemr is " << iterPGemr << std::endl;

  if(contxt1 >= 0) {
    Cblacs_gridexit( contxt1 );
  }

  if(contxt11 >= 0) {
    Cblacs_gridexit( contxt11 );
  }

  if(contxt2 >= 0) {
    Cblacs_gridexit( contxt2 );
  }

  return;
}         // -----  end of method Spinor::ISDF_CalculateIV ( Complex version )  -----

#endif

}  // namespace pwdft
