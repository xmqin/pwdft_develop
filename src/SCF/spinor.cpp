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

Spinor::Spinor () {}
Spinor::~Spinor    () {}

#ifndef _COMPLEX_
Spinor::Spinor ( 
    const Domain &dm, 
    const Int numComponent,
    const Int numStateTotal,
    const Real val ) 
{
  this->Setup( dm, numComponent, numStateTotal, val );
}         // -----  end of method Spinor::Spinor ( Real version )  ----- 

Spinor::Spinor ( const Domain &dm, 
    const Int numComponent, 
    const Int numStateTotal,
    const bool owndata, 
    Real* data )
{
  this->Setup( dm, numComponent, numStateTotal, owndata, data );
}         // -----  end of method Spinor::Spinor ( Real version ) ----- 

void Spinor::Setup ( 
    const Domain &dm, 
    const Int numComponent,
    const Int numStateTotal,
    const Real val ) 
{
  domain_  = dm;

  MPI_Comm comm = domain_.comm;
  MPI_Barrier(comm);

  numComponent_ = numComponent;

  // Grid and state dimension
  numGridTotal_ = domain_.NumGridTotal();
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
}         // -----  end of method Spinor::Setup ( Real version )  ----- 

void Spinor::Setup ( const Domain &dm, 
    const Int numComponent, 
    const Int numStateTotal,
    const bool owndata, 
    Real* data )
{
  domain_  = dm;

  MPI_Comm comm = domain_.comm;
  MPI_Barrier(comm);

  numComponent_ = numComponent;

  // Grid and state dimension
  numGridTotal_ = domain_.NumGridTotal();
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

  wavefun_ = NumTns<Real>( numGridTotal_, numComponent_, numStateLocal_,
      owndata, data );
}         // -----  end of method Spinor::Setup ( Real version )  ----- 

Spinor::Spinor ( 
    const Domain &dm, 
    const Int numComponent,
    const Int numStateTotal,
    const Complex val ) 
{
  this->Setup( dm, numComponent, numStateTotal, val );
}         // -----  end of method Spinor::Spinor ( Real version )  ----- 

Spinor::Spinor ( const Domain &dm, 
    const Int numComponent, 
    const Int numStateTotal,
    const bool owndata, 
    Complex* data )
{
  this->Setup( dm, numComponent, numStateTotal, owndata, data );
}         // -----  end of method Spinor::Spinor ( Real version ) ----- 

void Spinor::Setup ( 
    const Domain &dm, 
    const Int numComponent,
    const Int numStateTotal,
    const Complex val ) 
{
  domain_  = dm;

  MPI_Comm comm = domain_.comm;
  MPI_Barrier(comm);

  numComponent_ = numComponent;

  // Grid and state dimension
  numGridTotal_ = domain_.numGridSphere;
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
    wavefunG_.Resize( numGridTotal_, numComponent_, numStateLocal_*2 );
  }
  else{
    wavefunG_.Resize( numGridTotal_, numComponent_, numStateLocal_ );
  }

  SetValue( wavefunG_, val );
}         // -----  end of method Spinor::Setup ( Real version )  ----- 

void Spinor::Setup ( const Domain &dm, 
    const Int numComponent, 
    const Int numStateTotal,
    const bool owndata, 
    Complex* data )
{
  domain_  = dm;

  MPI_Comm comm = domain_.comm;
  MPI_Barrier(comm);

  numComponent_ = numComponent;

  // Grid and state dimension
  numGridTotal_ = domain_.numGridSphere;
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

  wavefunG_ = NumTns<Complex>( numGridTotal_, numComponent_, numStateLocal_,
      owndata, data );
}         // -----  end of method Spinor::Setup ( Real version )  -----

void
Spinor::Normalize    ( )
{
  Int size = wavefun_.m() * wavefun_.n();
  Int nocc = wavefun_.p();

  for (Int k=0; k<nocc; k++) {
    Real *ptr = wavefun_.MatData(k);
    Real   sum = 0.0;
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
}         // -----  end of method Spinor::Normalize ( Real version )  ----- 

void
Spinor::AddTeterPrecond (Fourier* fftPtr, DblNumVec& teter, NumTns<Real>& a3)
{
  Fourier& fft = *fftPtr;
  if( !fftPtr->isInitialized ){
    ErrorHandling("Fourier is not prepared.");
  }
  Int ntot = wavefun_.m();
  Int ncom = wavefun_.n();
  Int nocc = wavefun_.p();

  if( fft.domain.NumGridTotal() != ntot )
    ErrorHandling("Domain size does not match.");

  Int ntothalf = fftPtr->numGridTotalR2C;

  for (Int k=0; k<nocc; k++) {
    for (Int j=0; j<ncom; j++) {
      // For c2r and r2c transforms, the default is to DESTROY the
      // input, therefore a copy of the original matrix is necessary. 
      blas::Copy( ntot, wavefun_.VecData(j,k), 1, 
          reinterpret_cast<Real*>(fft.inputVecR2C.Data()), 1 );

      FFTWExecute ( fft, fft.forwardPlanR2C );

      Real*    ptr1d   = teter.Data();
      Complex* ptr2    = fft.outputVecR2C.Data();
      for (Int i=0; i<ntothalf; i++) 
        *(ptr2++) *= *(ptr1d++);

      FFTWExecute ( fft, fft.backwardPlanR2C);

      blas::Axpy( ntot, 1.0, fft.inputVecR2C.Data(), 1, a3.VecData(j,k), 1 );
    }
  }

  return ;
}         // -----  end of method Spinor::AddTeterPrecond ( Real version ) ----- 

void
Spinor::AddTeterPrecond (Fourier* fftPtr, DblNumVec& teter, NumTns<Complex>& a3)
{
  Fourier& fft = *fftPtr;
  if( !fftPtr->isInitialized ){
    ErrorHandling("Fourier is not prepared.");
  }

  Int ntothalf = numGridTotal_;
  Int ncom = numComponent_;
  Int nocc = wavefunG_.p();

  if( fft.domain.numGridSphere != ntothalf )
    ErrorHandling("Domain size does not match.");
  if( fft.numGridTotalR2C < ntothalf )
    ErrorHandling("numGridSphere is larger than numGridTotalR2C.");

  for( Int k = 0; k < nocc; k++ ){
    for( Int j = 0; j < ncom; j++ ){
      for( Int i = 0; i < ntothalf; i++ ){
        a3.VecData(j,k)[i] += wavefunG_.VecData(j,k)[i] * teter[i];
      }
    }
  }

  return ;
}         // -----  end of method Spinor::AddTeterPrecond ( Real version ) ----- 

void
Spinor::AddMultSpinorFineR2C ( Fourier& fft, const DblNumVec& ekin,
    const DblNumVec& vtot, const std::vector<PseudoPot>& pseudo, NumTns<Real>& a3 )
{
  // Real version -- for Gamma-point calculation ( spin-restricted and spin-unrestricted )

  if( !fft.isInitialized ){
    ErrorHandling("Fourier is not prepared.");
  }

  Index3& numGrid = domain_.numGrid;
  Index3& numGridFine = domain_.numGridFine;
  Int ntot = wavefun_.m();
  Int ncom = wavefun_.n();
  Int numStateLocal = wavefun_.p();
  Int ntotFine = domain_.NumGridTotalFine();
  Real vol = domain_.Volume();

  Int ntotR2C = fft.numGridTotalR2C;
  Int ntotR2CFine = fft.numGridTotalR2CFine;

  if( fft.domain.NumGridTotal() != ntot ){
    ErrorHandling("Domain size does not match3.");
  }

  // Temporary variable for saving wavefunction on a fine grid
  DblNumVec psiFine(ntotFine);
  DblNumVec psiUpdateFine(ntotFine);
  
  for (Int k=0; k<numStateLocal; k++) {
    for (Int j=0; j<ncom; j++) {
      SetValue( psiFine, 0.0 );
      SetValue( psiUpdateFine, 0.0 );

      // R2C version
      SetValue( fft.inputVecR2C, 0.0 ); 
      SetValue( fft.inputVecR2CFine, 0.0 ); 
      SetValue( fft.outputVecR2C, Z_ZERO ); 
      SetValue( fft.outputVecR2CFine, Z_ZERO ); 

      // For c2r and r2c transforms, the default is to DESTROY the
      // input, therefore a copy of the original matrix is necessary. 
      blas::Copy( ntot, wavefun_.VecData(j,k), 1, 
          fft.inputVecR2C.Data(), 1 );

      FFTWExecute ( fft, fft.forwardPlanR2C );

      // Interpolate wavefunction from coarse to fine grid
      // 1. coarse to fine
      // The FFT number is even(esdf.cpp now), change as follows
      if(  esdfParam.FFTtype == "even" || esdfParam.FFTtype == "power")
      {
        Complex *fftOutFinePtr = fft.outputVecR2CFine.Data();
        Complex *fftOutPtr = fft.outputVecR2C.Data();
        IP_c2f(numGrid.Data(),numGridFine.Data(),fftOutPtr,fftOutFinePtr);
      }
      else if( esdfParam.FFTtype == "odd" )
      // odd version
      {                          
        Real fac = sqrt( double(ntot) / double(ntotFine) );    
        Int *idxPtr = fft.idxFineGridR2C.Data();               
        Complex *fftOutFinePtr = fft.outputVecR2CFine.Data();  
        Complex *fftOutPtr = fft.outputVecR2C.Data();          
        for( Int i = 0; i < ntotR2C; i++ ){                    
          fftOutFinePtr[*(idxPtr++)] = *(fftOutPtr++) * fac; 
        }                                                   
      }                             
    
      FFTWExecute ( fft, fft.backwardPlanR2CFine );

      blas::Copy( ntotFine, fft.inputVecR2CFine.Data(), 1, psiFine.Data(), 1 );

      // Add the contribution from local pseudopotential
      {
        Real *psiUpdateFinePtr = psiUpdateFine.Data();
        Real *psiFinePtr = psiFine.Data();
        Real *vtotPtr = vtot.Data();
        for( Int i = 0; i < ntotFine; i++ ){
          *(psiUpdateFinePtr++) += *(psiFinePtr++) * *(vtotPtr++);
        }
      }

      // Add the contribution from nonlocal pseudopotential
      Int natm = pseudo.size();
      for (Int iatm=0; iatm<natm; iatm++) {
        Int nobt = pseudo[iatm].vnlList.size();
        for (Int iobt=0; iobt<nobt; iobt++) {
          const Real       vnlwgt = pseudo[iatm].vnlList[iobt].second;
          const SparseVec &vnlvec = pseudo[iatm].vnlList[iobt].first;
          const IntNumVec &iv = vnlvec.first;
          const DblNumMat &dv = vnlvec.second;

          Real    weight = 0.0; 
          const Int    *ivptr = iv.Data();
          const Real   *dvptr = dv.VecData(VAL);
          for (Int i=0; i<iv.m(); i++) {
            weight += (*(dvptr++)) * psiFine[*(ivptr++)];
          }
          weight *= vol/Real(ntotFine)*vnlwgt;

          ivptr = iv.Data();
          dvptr = dv.VecData(VAL);
          for (Int i=0; i<iv.m(); i++) {
            psiUpdateFine[*(ivptr++)] += (*(dvptr++)) * weight;
          }
        } // for (iobt)
      } // for (iatm)      

      // Laplacian operator. Perform inverse Fourier transform in the end
      {
        for (Int i=0; i<ntotR2C; i++) 
          fft.outputVecR2C(i) *= ekin(i);
      }

      // Restrict psiUpdateFine from fine grid in the real space to
      // coarse grid in the Fourier space. Combine with the Laplacian contribution
      SetValue( fft.inputVecR2CFine, 0.0 );
      blas::Copy( ntotFine, psiUpdateFine.Data(), 1, fft.inputVecR2CFine.Data(), 1 );

      // Fine to coarse grid
      // Note the update is important since the Laplacian contribution is already taken into account.
      // The computation order is also important
      FFTWExecute ( fft, fft.forwardPlanR2CFine );

      // 2. fine to coarse
     if(  esdfParam.FFTtype == "even" || esdfParam.FFTtype == "power")
     {
        Complex *fftOutFinePtr = fft.outputVecR2CFine.Data();
        Complex *fftOutPtr = fft.outputVecR2C.Data();
        IP_f2c(numGrid.Data(),numGridFine.Data(),fftOutPtr,fftOutFinePtr);
     }
     else if(  esdfParam.FFTtype == "odd")
     {
        Real fac = sqrt( double(ntotFine) / double(ntot) );
        Int *idxPtr = fft.idxFineGridR2C.Data();
        Complex *fftOutFinePtr = fft.outputVecR2CFine.Data();
        Complex *fftOutPtr = fft.outputVecR2C.Data();
        for( Int i = 0; i < ntotR2C; i++ ){
          *(fftOutPtr++) += fftOutFinePtr[*(idxPtr++)] * fac;
        }
      }
      FFTWExecute ( fft, fft.backwardPlanR2C );

      blas::Axpy( ntot, 1.0, fft.inputVecR2C.Data(), 1, a3.VecData(j,k), 1 );

    } // j++
  } // k++

  return ;
}        // -----  end of method Spinor::AddMultSpinorFineR2C  -----

void
Spinor::AddMultSpinorFineR2C ( Fourier& fft, const DblNumVec& ekin,
    const DblNumVec& vtot, const std::vector<PseudoPot>& pseudo, NumTns<Complex>& a3 )
{
  // Real case -- for Gamma-point calculation ( spin-restricted and spin-unrestricted )
  
  if( !fft.isInitialized ){
    ErrorHandling("Fourier is not prepared.");
  }

  Int ntot = numGridTotal_;
  Int ncom = numComponent_;
  Int numStateLocal = wavefunG_.p();
  Int ntotFine = domain_.NumGridTotalFine();
  Real vol = domain_.Volume();

  Index3& numGrid = domain_.numGrid;
  Index3& numGridFine = domain_.numGridFine;
  unsigned plannerFlag = FFTW_MEASURE;

  Int ntotR2C = fft.numGridTotalR2C;

  if( fft.domain.numGridSphere != ntot ){
    ErrorHandling("Domain size does not match3.");
  }

  if( ncom != 1 ){
    ErrorHandling("Component number of spinor should be one in \
        spin-restricted and spin-unrestricted calculations.");
  }

  // Temporary variable for saving wavefunction on a fine grid
  DblNumVec psiFine(ntotFine);
  DblNumVec psiUpdateFine(ntotFine);
  
  for( Int k = 0; k < numStateLocal; k++ ){

    SetValue( psiFine, 0.0 );
    SetValue( psiUpdateFine, 0.0 );

    // R2C version
    SetValue( fft.inputVecR2C, 0.0 );
    SetValue( fft.inputVecR2CFine, 0.0 );
    SetValue( fft.outputVecR2C, Z_ZERO );
    SetValue( fft.outputVecR2CFine, Z_ZERO );

    {
      // Copy the wavefun in sphere cutted R2C grids to
      // the normal R2C grids
      Int *idxPtr = fft.idxCoarseCut.Data();

      Complex *fftOutR2CPtr = fft.outputVecR2C.Data();
      Complex *psiPtr = wavefunG_.VecData(0,k);
      fftOutR2CPtr[*(idxPtr++)] = *(psiPtr++);
      for( Int i = 1; i < ntot; i++ ){
        // Note: sqrt(2.0) is due to that the orthogonalization conditions is:
        // |psi(G=0)|^2 + \sum_{G!=0 R2C} |(psi(G)|^2 = 1.0 rather than
        // |psi(G=0)|^2 + \sum_{G!=0} |(psi(G)|^2 = 1.0
        fftOutR2CPtr[*(idxPtr++)] = *(psiPtr++) / std::sqrt(2.0);
      }

      // Padding
      std::pair<IntNumVec, IntNumVec>& idxc = fft.idxCoarsePadding;
      idxPtr = idxc.second.Data();
      Int *idxPtr1 = idxc.first.Data();
      for( Int i = 0; i < idxc.first.m(); i++ ){
        fftOutR2CPtr[*(idxPtr++)] = std::conj(fftOutR2CPtr[*(idxPtr1++)]);
      }
    }

    // Interpolate wavefunction from coarse to fine grid
    if(  esdfParam.FFTtype == "even" || esdfParam.FFTtype == "power")
    {
      Complex *fftOutFinePtr = fft.outputVecR2CFine.Data();
      Complex *fftOutPtr = fft.outputVecR2C.Data();
      IP_c2f(numGrid.Data(),numGridFine.Data(),fftOutPtr,fftOutFinePtr);
    }
    else if( esdfParam.FFTtype == "odd" )
    {
      Int *idxPtr = fft.idxFineGridR2C.Data();
      Complex *fftOutFinePtr = fft.outputVecR2CFine.Data();
      Complex *fftOutPtr = fft.outputVecR2C.Data();
      for( Int i = 0; i < ntotR2C; i++ ){
        fftOutFinePtr[*(idxPtr++)] = *(fftOutPtr++);
      }
    }

    fftw_execute( fftw_plan_dft_c2r_3d(
      numGrid[2], numGrid[1], numGrid[0],
      reinterpret_cast<fftw_complex*>( fft.outputVecR2CFine.Data() ),
      psiFine.Data(),
      plannerFlag ) );

    Real fac = 1.0 / double(ntotFine);
    blas::Scal( ntotFine, fac, psiFine.Data(), 1 );

    // Add the contribution from local pseudopotential
    {
      Real *psiUpdateFinePtr = psiUpdateFine.Data();
      Real *psiFinePtr = psiFine.Data();
      Real *vtotPtr = vtot.Data();
      for( Int i = 0; i < ntotFine; i++ ){
        *(psiUpdateFinePtr++) += *(psiFinePtr++) * *(vtotPtr++);
      }
    }

    // The calculation of contribution from nonlocal pseudopotential 
    // is moved to EigenSolver::NonlocalMultX

    // Laplacian operator. 
    {
      for( Int i = 0; i < ntot; i++ ) 
        a3(i,VAL,k) = wavefunG_(i,VAL,k) * ekin(i);
    }

    fftw_execute( fftw_plan_dft_r2c_3d(
      numGrid[2], numGrid[1], numGrid[0],
      psiUpdateFine.Data(),
      reinterpret_cast<fftw_complex*>( fft.outputVecR2CFine.Data() ),
      plannerFlag ) );

    // Interpolation from fine to coarse
    SetValue( fft.outputVecR2C, Z_ZERO );
    if(  esdfParam.FFTtype == "even" || esdfParam.FFTtype == "power" )
    {
      Complex *fftOutFinePtr = fft.outputVecR2CFine.Data();
      Complex *fftOutPtr = fft.outputVecR2C.Data();
      IP_f2c(numGrid.Data(),numGridFine.Data(),fftOutPtr,fftOutFinePtr);
    }
    else if(  esdfParam.FFTtype == "odd" )
    {
      Int *idxPtr = fft.idxFineGridR2C.Data();
      Complex *fftOutFinePtr = fft.outputVecR2CFine.Data();
      Complex *fftOutPtr = fft.outputVecR2C.Data();
      for( Int i = 0; i < ntotR2C; i++ ){
        *(fftOutPtr++) = fftOutFinePtr[*(idxPtr++)];
      }
    }

    {
      // Normal R2C grids -> Cutted R2C grids
      Int *idxPtr = fft.idxCoarseCut.Data();

      Complex *fftOutR2CPtr = fft.outputVecR2C.Data();
      Complex *psiPtr = a3.VecData(VAL,k);
      *(psiPtr++) += fftOutR2CPtr[*(idxPtr++)];
      for( Int i = 1; i < ntot; i++ ){
        *(psiPtr++) += fftOutR2CPtr[*(idxPtr++)] * std::sqrt(2.0);
      }
    }
  } // for (k)

  return ;
}        // -----  end of method Spinor::AddMultSpinorFineR2C  -----

void Spinor::AddMultSpinorEXX ( Fourier& fft, 
    const NumTns<Real>& phi,
    const DblNumVec& exxgkkR2C,
    Real  exxFraction,
    Int  nspin,
    const DblNumVec& occupationRate,
    NumTns<Real>& a3 )
{
  if( !fft.isInitialized ){
    ErrorHandling("Fourier is not prepared.");
  }

  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

  Int ntot     = domain_.NumGridTotal();
  Int ntotR2C = fft.numGridTotalR2C;
  Int ncom = numComponent_;
  Int ncomPhi = phi.n();
  Int numStateLocal = wavefun_.p();
  Int numStateTotal = numStateTotal_;

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

  MPI_Barrier(domain_.comm);

  for( Int iproc = 0; iproc < mpisize; iproc++ ){

    if( iproc == mpirank )
      numStateLocalTemp = numStateLocal;

    MPI_Bcast( &numStateLocalTemp, 1, MPI_INT, iproc, domain_.comm );

    IntNumVec wavefunIdxTemp(numStateLocalTemp);
    if( iproc == mpirank ){
      wavefunIdxTemp = wavefunIdx_;
    }

    MPI_Bcast( wavefunIdxTemp.Data(), numStateLocalTemp, MPI_INT, iproc, domain_.comm );

    // FIXME OpenMP does not work since all variables are shared
    for( Int kphi = 0; kphi < numStateLocalTemp; kphi++ ){

      SetValue( phiTemp, D_ZERO );

      if( iproc == mpirank )
      { 
        Real* phiPtr = phi.VecData(VAL, kphi);
        for( Int ir = 0; ir < ntot; ir++ ){
          phiTemp(ir) = phiPtr[ir];
        }
      }

      MPI_Bcast( phiTemp.Data(), ntot, MPI_DOUBLE, iproc, domain_.comm );

      for (Int k=0; k<numStateLocal; k++) {

        Real* psiPtr = wavefun_.VecData(VAL,k);
        for( Int ir = 0; ir < ntot; ir++ ){
          fft.inputVecR2C(ir) = psiPtr[ir] * phiTemp(ir);
        }

        FFTWExecute ( fft, fft.forwardPlanR2C );

        // Solve the Poisson-like problem for exchange
        for( Int ig = 0; ig < ntotR2C; ig++ ){
          fft.outputVecR2C(ig) *= exxgkkR2C(ig);
        }

        FFTWExecute ( fft, fft.backwardPlanR2C );

        Real* a3Ptr = a3.VecData(VAL,k);
        Real fac = -exxFraction * occupationRate[wavefunIdxTemp(kphi)];  
        for( Int ir = 0; ir < ntot; ir++ ) {
          a3Ptr[ir] += fft.inputVecR2C(ir) * phiTemp(ir) * fac;
        }
      } // for (k)

      MPI_Barrier(domain_.comm);
    } // for (kphi)
  } //iproc

  MPI_Barrier(domain_.comm);

  return ;
}        // -----  end of method Spinor::AddMultSpinorEXX ( Real version)  ----- 

void Spinor::AddMultSpinorEXX ( Fourier& fft, 
    const NumTns<Complex>& phi,
    const DblNumVec& exxgkkR2C,
    Real  exxFraction,
    Int  nspin,
    const DblNumVec& occupationRate,
    NumTns<Complex>& a3 )
{
  if( !fft.isInitialized ){
    ErrorHandling("Fourier is not prepared.");
  }

  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

  bool realspace    = esdfParam.isUseRealSpace;
  bool spherecut    = esdfParam.isUseSphereCut;

  Int ntot     = domain_.NumGridTotal();
  Int ntotR2C  = fft.numGridTotalR2C;
  Int nPW      = ( spherecut == true ) ? domain_.numGridSphere : ntot;
  Int ntotFock = ( spherecut == true ) ? fft.idxFineFock.m() : ntotR2C;
  Int ncom = numComponent_;
  Int ncomPhi = phi.n();
  Int numStateLocal = wavefunG_.p();
  Int numStateTotal = numStateTotal_;

  Real vol = domain_.Volume();
  IntNumVec &idxCoarseCut = fft.idxCoarseCut;

  if( ncomPhi != 1 || ncom != 1 ){
    ErrorHandling("Spin-noncollinear case is not supproted by real-value version PWDFT.");
  }

  if( numGridTotal_ != nPW ){
    ErrorHandling("Domain size does not match.");
  }

  // Wavefunction in real space
  DblNumMat wavefun_rs;
  DblNumMat phi_rs;

  {
    // Transform psi to real space
    wavefun_rs.Resize( ntot, numStateLocal );
    for( Int j = 0; j < numStateLocal; j++ ){
      SetValue( fft.outputVecR2C, Z_ZERO );
      fft.outputVecR2C[idxCoarseCut[0]] = wavefunG_(0,VAL,j);
      for( Int i = 1; i < nPW; i++ ){
        fft.outputVecR2C[idxCoarseCut[i]] = wavefunG_(i,VAL,j) / std::sqrt(2.0);
      }
      // Padding
      std::pair<IntNumVec, IntNumVec>& idxc = fft.idxCoarsePadding;
      Int *idxPtr = idxc.second.Data();
      Int *idxPtr1 = idxc.first.Data();
      for( Int i = 0; i < idxc.first.m(); i++ ){
        fft.outputVecR2C[*(idxPtr++)] = std::conj(fft.outputVecR2C[*(idxPtr1++)]);
      }

      fftw_execute( fft.backwardPlanR2C );
      blas::Copy( ntot, fft.inputVecR2C.Data(), 1, wavefun_rs.VecData(j), 1 );
    }
    // Transform phi to real space
    phi_rs.Resize( ntot, numStateLocal );
    for( Int j = 0; j < numStateLocal; j++ ){
      SetValue( fft.outputVecR2C, Z_ZERO );
      fft.outputVecR2C[idxCoarseCut[0]] = phi(0,VAL,j);
      for( Int i = 1; i < nPW; i++ ){
        fft.outputVecR2C[idxCoarseCut[i]] = phi(i,VAL,j) / std::sqrt(2.0);
      }
      std::pair<IntNumVec, IntNumVec>& idxc = fft.idxCoarsePadding;
      Int *idxPtr = idxc.second.Data();
      Int *idxPtr1 = idxc.first.Data();
      for( Int i = 0; i < idxc.first.m(); i++ ){
        fft.outputVecR2C[*(idxPtr++)] = std::conj(fft.outputVecR2C[*(idxPtr1++)]);
      }

      fftw_execute( fft.backwardPlanR2C );
      blas::Copy( ntot, fft.inputVecR2C.Data(), 1, phi_rs.VecData(j), 1 );
    }
  }  // ---- end of if( realspace ) ----

  // Temporary variable for saving wavefunction on a fine grid
  DblNumVec phiTemp_rs(ntot);

  Int numStateLocalTemp;

  MPI_Barrier(domain_.comm);

  for( Int iproc = 0; iproc < mpisize; iproc++ ){

    if( iproc == mpirank )
      numStateLocalTemp = numStateLocal;

    MPI_Bcast( &numStateLocalTemp, 1, MPI_INT, iproc, domain_.comm );

    IntNumVec wavefunIdxTemp(numStateLocalTemp);
    if( iproc == mpirank ){
      wavefunIdxTemp = wavefunIdx_;
    }

    MPI_Bcast( wavefunIdxTemp.Data(), numStateLocalTemp, MPI_INT, iproc, domain_.comm );

    // FIXME OpenMP does not work since all variables are shared
    for( Int kphi = 0; kphi < numStateLocalTemp; kphi++ ){

      SetValue( phiTemp_rs, D_ZERO );

      if( iproc == mpirank )
      { 
        Real* phiPtr = phi_rs.VecData(kphi);
        for( Int ir = 0; ir < ntot; ir++ ){
          phiTemp_rs(ir) = phiPtr[ir];
        }
      }

      MPI_Bcast( phiTemp_rs.Data(), ntot, MPI_DOUBLE, iproc, domain_.comm );

      for (Int k=0; k<numStateLocal; k++) {

        Real* psiPtr = wavefun_rs.VecData(k);
        for( Int ir = 0; ir < ntot; ir++ ){
          fft.inputVecR2C(ir) = psiPtr[ir] * phiTemp_rs(ir);
        }

        FFTWExecute ( fft, fft.forwardPlanR2C );

        CpxNumVec prodG( ntotR2C );
        blas::Copy( ntotR2C, fft.outputVecR2C.Data(), 1, prodG.Data(), 1 );

        // Solve the Poisson-like problem for exchange
        SetValue( fft.outputVecR2C, Z_ZERO );  
        for( Int ig = 0; ig < ntotFock; ig++ ){
          Int idx = fft.idxFineFock[ig];
          fft.outputVecR2C(idx) = prodG(idx) * exxgkkR2C(ig);
        }

        FFTWExecute ( fft, fft.backwardPlanR2C );

        Complex* a3Ptr = a3.VecData(VAL,k);
        Real fac = -exxFraction * occupationRate[wavefunIdxTemp(kphi)];  

        for( Int ir = 0; ir < ntot; ir++ ){
          fft.inputVecR2C(ir) *= phiTemp_rs(ir) * fac;
        }

        fftw_execute( fft.forwardPlanR2C );

        a3Ptr[0] += fft.outputVecR2C(idxCoarseCut[0]);
        for( Int ig = 1; ig < nPW; ig++ ){
          a3Ptr[ig] += fft.outputVecR2C(idxCoarseCut[ig]) * std::sqrt(2.0);
        }
      } // for (k)

      MPI_Barrier(domain_.comm);
    } // for (kphi)
  } //iproc

  MPI_Barrier(domain_.comm);

  return ;
}        // -----  end of method Spinor::AddMultSpinorEXX ( Real version)  ----- 
#endif

} // namespace pwdft
