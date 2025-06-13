/// @file spinor.cpp
/// @brief Spinor (wavefunction) for the global domain.
/// @date 2023-07-01
#include  "ppcg/spinor.hpp"
#include  "ppcg/utility.hpp"
#include  "ppcg/blas.hpp"
#include  "ppcg/lapack.hpp"
#include  "ppcg/mpi_interf.hpp"

namespace PPCG {

Spinor::Spinor () { }         
Spinor::~Spinor    () {}

#ifdef _COMPLEX_
Spinor::Spinor ( const Domain &dm, 
    const Int numComponent, 
    const Int numStateTotal,
    Int numStateLocal,
    const bool owndata, 
    Complex* data )
{
  this->Setup( dm, numComponent, numStateTotal, numStateLocal, owndata, data );

}         // -----  end of method Spinor::Spinor  ----- 

void Spinor::Setup ( const Domain &dm, 
    const Int numComponent, 
    const Int numStateTotal,
    Int numStateLocal,
    const bool owndata, 
    Complex* data )
{

  domain_       = dm;
  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

  wavefun_      = NumTns<Complex>( dm.NumGridTotal(), numComponent, numStateLocal,
      owndata, data );

  Int blocksize;

  if ( numStateTotal <=  mpisize ) {
    blocksize = 1;
  }
  else {  // numStateTotal >  mpisize
    if ( numStateTotal % mpisize == 0 ){
      blocksize = numStateTotal / mpisize;
    }
    else {
      blocksize = ((numStateTotal - 1) / mpisize) + 1;
    }    
  }

  numStateTotal_ = numStateTotal;
  blocksize_ = blocksize;

  wavefunIdx_.Resize( numStateLocal );
  SetValue( wavefunIdx_, 0 );
  for (Int i = 0; i < numStateLocal; i++){
    wavefunIdx_[i] = i * mpisize + mpirank ;
  }

}         // -----  end of method Spinor::Setup  ----- 

void
Spinor::AddTeterPrecond (Fourier* fftPtr, NumTns<Complex>& a3)
{
  Fourier& fft = *fftPtr;
  if( !fftPtr->isInitialized ){
    ErrorHandling("Fourier is not prepared.");
  }
  Int ntot = wavefun_.m();
  Int ncom = wavefun_.n();
  Int nocc = wavefun_.p();

  if( fftPtr->domain.NumGridTotal() != ntot ){
    ErrorHandling("Domain size does not match.");
  }

  Int numFFTGrid = fftPtr->numGridTotal;
  // These two are private variables in the OpenMP context

  for (Int k=0; k<nocc; k++) {
    for (Int j=0; j<ncom; j++) {
      // For c2r and r2c transforms, the default is to DESTROY the
      // input, therefore a copy of the original matrix is necessary. 
      blas::Copy( ntot, wavefun_.VecData(j,k), 1, 
          fft.inputComplexVec.Data(), 1 );

      FFTWExecute ( fft, fft.forwardPlan );

      Real* ptr1d      = fftPtr->TeterPrecond.Data();
      Complex* ptr2    = fft.outputComplexVec.Data();
      for (Int i=0; i<numFFTGrid; i++) 
        *(ptr2++) *= *(ptr1d++);

      FFTWExecute ( fft, fft.backwardPlan);

      blas::Axpy( ntot, 1.0, fft.inputComplexVec.Data(), 1, a3.VecData(j,k), 1 );
    }
  }


  return ;
}         // -----  end of method Spinor::AddTeterPrecond ----- 

void
Spinor::AddMultSpinorFine ( Fourier& fft, const DblNumVec& vtot, const DblNumVec& vnlc,
    const IntNumVec& index, const IntNumVec& nGproj, const DblNumVec& coef, NumTns<Complex>& a3 )
{
  if( !fft.isInitialized ){
    ErrorHandling("Fourier is not prepared.");
  }
  Int ntot = wavefun_.m();
  Int ncom = wavefun_.n();
  Int numStateLocal = wavefun_.p();
  Int ntotFine = domain_.NumGridTotalFine();
  Real vol = domain_.Volume();

  if( fft.domain.NumGridTotal() != ntot ){
    ErrorHandling("Domain size does not match.");
  }

  // Temporary variable for saving wavefunction on a fine grid
  CpxNumVec psiFine(ntotFine);
  CpxNumVec psiUpdateFine(ntotFine);

  for (Int k=0; k<numStateLocal; k++) {
    for (Int j=0; j<ncom; j++) {

      SetValue( psiFine, Complex(0.0,0.0) );
      SetValue( psiUpdateFine, Complex(0.0,0.0) );

      // Fourier transform
      SetValue( fft.inputComplexVec, Z_ZERO ); // no need to set to zero
      blas::Copy( ntot, wavefun_.VecData(j,k), 1, fft.inputComplexVec.Data(), 1 );

      // Fourier transform of wavefunction saved in fft.outputComplexVec
      fftw_execute( fft.forwardPlan );

      // Interpolate wavefunction from coarse to fine grid
      {
        SetValue( fft.outputComplexVecFine, Z_ZERO ); 
        Int *idxPtr = fft.idxFineGrid.Data();
        Complex *fftOutFinePtr = fft.outputComplexVecFine.Data();
        Complex *fftOutPtr = fft.outputComplexVec.Data();
        for( Int i = 0; i < ntot; i++ ){
          //fft.outputComplexVecFine(fft.idxFineGrid(i)) = fft.outputComplexVec(i);
          fftOutFinePtr[*(idxPtr++)] = *(fftOutPtr++);
        }
      }
      fftw_execute( fft.backwardPlanFine );
      Real fac = 1.0 / std::sqrt( double(domain_.NumGridTotal())  *
          double(domain_.NumGridTotalFine()) ); 
      blas::Copy( ntotFine, fft.inputComplexVecFine.Data(),
          1, psiFine.Data(), 1 );
      blas::Scal( ntotFine, fac, psiFine.Data(), 1 );
      {
        Complex *psiUpdateFinePtr = psiUpdateFine.Data();
        Complex *psiFinePtr = psiFine.Data();
        Real *vtotPtr = vtot.Data();
        for( Int i = 0; i < ntotFine; i++ ){
          *(psiUpdateFinePtr++) += *(psiFinePtr++) * *(vtotPtr++);
        }
      }

      // Add the contribution from nonlocal pseudopotential
      Int nbeta = coef.m();      
      Int start = 0;
      for (Int iobt=0; iobt<nbeta; iobt++) {
        const Real vnlwgt = coef[iobt];
        const Int tempnGproj = nGproj[iobt];
        Complex weight = (0.0,0.0); 
        const Real *dvptr = &vnlc[start];
        const Int  *ivptr = &index[start];

        for (Int i=0; i<tempnGproj; i++) {
          weight += (*(dvptr++)) * psiFine[*(ivptr++)];
        }
        weight *= vol/Real(ntotFine)*vnlwgt;

        dvptr = &vnlc[start];
        ivptr = &index[start];
        for (Int i=0; i<tempnGproj; i++) {
          psiUpdateFine[*(ivptr++)] += (*(dvptr++)) * weight;
        }

        start += tempnGproj;
      } // for (iobt)
      
      // Laplacian operator. Perform inverse Fourier transform in the end
      {
        for (Int i=0; i<ntot; i++) 
          fft.outputComplexVec(i) *= fft.gkk(i);
      }

      // Restrict psiUpdateFine from fine grid in the real space to
      // coarse grid in the Fourier space. Combine with the Laplacian contribution
      SetValue( fft.inputComplexVecFine, Z_ZERO );
      blas::Copy( ntotFine, psiUpdateFine.Data(), 1,
          fft.inputComplexVecFine.Data(), 1 );

      // Fine to coarse grid
      // Note the update is important since the Laplacian contribution is already taken into account.
      // The computation order is also important
      fftw_execute( fft.forwardPlanFine );
      {
        Real fac = std::sqrt(Real(ntot) / (Real(ntotFine)));
        Int* idxPtr = fft.idxFineGrid.Data();
        Complex *fftOutFinePtr = fft.outputComplexVecFine.Data();
        Complex *fftOutPtr = fft.outputComplexVec.Data();

        for( Int i = 0; i < ntot; i++ ){
          *(fftOutPtr++) += fftOutFinePtr[*(idxPtr++)] * fac;
        }
      }

      // Inverse Fourier transform to save back to the output vector
      fftw_execute( fft.backwardPlan );
      blas::Axpy( ntot, 1.0 / Real(ntot), 
          fft.inputComplexVec.Data(), 1, a3.VecData(j,k), 1 );
    }
  }

  return ;
}        // -----  end of method Spinor::AddMultSpinorFine  ----- 

#else
// Real version including R2C method
Spinor::Spinor ( const Domain &dm, 
    const Int numComponent, 
    const Int numStateTotal,
    Int numStateLocal,
    const bool owndata, 
    Real* data )
{
  this->Setup( dm, numComponent, numStateTotal, numStateLocal, owndata, data );

}         // -----  end of method Spinor::Spinor  ----- 

void Spinor::Setup ( const Domain &dm, 
    const Int numComponent, 
    const Int numStateTotal,
    Int numStateLocal,
    const bool owndata, 
    Real* data )
{

  domain_       = dm;
  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

  wavefun_      = NumTns<Real>( dm.NumGridTotal(), numComponent, numStateLocal,
      owndata, data );

  Int blocksize;

  if ( numStateTotal <=  mpisize ) {
    blocksize = 1;
  }
  else {  // numStateTotal >  mpisize
    if ( numStateTotal % mpisize == 0 ){
      blocksize = numStateTotal / mpisize;
    }
    else {
      blocksize = ((numStateTotal - 1) / mpisize) + 1;
    }    
  }

  numStateTotal_ = numStateTotal;
  blocksize_ = blocksize;

  wavefunIdx_.Resize( numStateLocal );
  SetValue( wavefunIdx_, 0 );
  for (Int i = 0; i < numStateLocal; i++){
    wavefunIdx_[i] = i * mpisize + mpirank ;
  }

}         // -----  end of method Spinor::Setup  ----- 

void
Spinor::AddTeterPrecond (Fourier* fftPtr, NumTns<Real>& a3)
{
  Fourier& fft = *fftPtr;
  if( !fftPtr->isInitialized ){
    ErrorHandling("Fourier is not prepared.");
  }
  Int ntot = wavefun_.m();
  Int ncom = wavefun_.n();
  Int nocc = wavefun_.p();

  if( fftPtr->domain.NumGridTotal() != ntot ){
    ErrorHandling("Domain size does not match1.");
  }

  Int ntothalf = fftPtr->numGridTotalR2C;

  for (Int k=0; k<nocc; k++) {
    for (Int j=0; j<ncom; j++) {
      // For c2r and r2c transforms, the default is to DESTROY the
      // input, therefore a copy of the original matrix is necessary. 
      blas::Copy( ntot, wavefun_.VecData(j,k), 1, 
          reinterpret_cast<Real*>(fft.inputVecR2C.Data()), 1 );
      FFTWExecute ( fft, fft.forwardPlanR2C );

      Real*    ptr1d   = fftPtr->TeterPrecondR2C.Data();
      Complex* ptr2    = fft.outputVecR2C.Data();
      for (Int i=0; i<ntothalf; i++) 
        *(ptr2++) *= *(ptr1d++);

      FFTWExecute ( fft, fft.backwardPlanR2C);
      blas::Axpy( ntot, 1.0, fft.inputVecR2C.Data(), 1, a3.VecData(j,k), 1 );
    }
  }

  return ;
}         // -----  end of method Spinor::AddTeterPrecond ----- 

void
Spinor::AddMultSpinorFine ( Fourier& fft, const DblNumVec& vtot, const DblNumVec& vnlc,
    const IntNumVec& index, const IntNumVec& nGproj, const DblNumVec& coef, NumTns<Real>& a3 )
{
  // TODO Complex case

  if( !fft.isInitialized ){
    ErrorHandling("Fourier is not prepared.");
  }
  Int ntot = wavefun_.m();
  Int ncom = wavefun_.n();
  Int numStateLocal = wavefun_.p();
  Int ntotFine = domain_.NumGridTotalFine();
  Real vol = domain_.Volume();

  if( fft.domain.NumGridTotal() != ntot ){
    ErrorHandling("Domain size does not match2.");
  }

  // Temporary variable for saving wavefunction on a fine grid
  DblNumVec psiFine(ntotFine);
  DblNumVec psiUpdateFine(ntotFine);

  for (Int k=0; k<numStateLocal; k++) {
    for (Int j=0; j<ncom; j++) {

      SetValue( psiFine, 0.0 );
      SetValue( psiUpdateFine, 0.0 );

      SetValue( fft.inputComplexVec, Z_ZERO );
      blas::Copy( ntot, wavefun_.VecData(j,k), 1,
          reinterpret_cast<Real*>(fft.inputComplexVec.Data()), 2 );

      // Fourier transform of wavefunction saved in fft.outputComplexVec
      fftw_execute( fft.forwardPlan );

      // Interpolate wavefunction from coarse to fine grid
      {
        SetValue( fft.outputComplexVecFine, Z_ZERO ); 
        Int *idxPtr = fft.idxFineGrid.Data();
        Complex *fftOutFinePtr = fft.outputComplexVecFine.Data();
        Complex *fftOutPtr = fft.outputComplexVec.Data();
        for( Int i = 0; i < ntot; i++ ){
          fftOutFinePtr[*(idxPtr++)] = *(fftOutPtr++);
        }
      }
      fftw_execute( fft.backwardPlanFine );
      Real fac = 1.0 / std::sqrt( double(domain_.NumGridTotal())  *
          double(domain_.NumGridTotalFine()) ); 

      blas::Copy( ntotFine, reinterpret_cast<Real*>(fft.inputComplexVecFine.Data()),
          2, psiFine.Data(), 1 );
      blas::Scal( ntotFine, fac, psiFine.Data(), 1 );
      {
        Real *psiUpdateFinePtr = psiUpdateFine.Data();
        Real *psiFinePtr = psiFine.Data();
        Real *vtotPtr = vtot.Data();
        for( Int i = 0; i < ntotFine; i++ ){
          *(psiUpdateFinePtr++) += *(psiFinePtr++) * *(vtotPtr++);
        }
      }

      // Add the contribution from nonlocal pseudopotential
      Int nbeta = coef.m();
      Int start = 0;
      for (Int iobt=0; iobt<nbeta; iobt++) {
        const Real vnlwgt = coef[iobt];
        const Int tempnGproj = nGproj[iobt];
        Real weight = 0.0; 
        const Real *dvptr = &vnlc[start];
        const Int  *ivptr = &index[start];

        for (Int i=0; i<tempnGproj; i++) {
          weight += (*(dvptr++)) * psiFine[*(ivptr++)];
        }
        weight *= vol/Real(ntotFine)*vnlwgt;

        dvptr = &vnlc[start];
        ivptr = &index[start];
        for (Int i=0; i<tempnGproj; i++) {
          psiUpdateFine[*(ivptr++)] += (*(dvptr++)) * weight;
        }

        start += tempnGproj;
      } // for (iobt)
    
      // Laplacian operator. Perform inverse Fourier transform in the end
      {
        for (Int i=0; i<ntot; i++) 
          fft.outputComplexVec(i) *= fft.gkk(i);
      }

      // Restrict psiUpdateFine from fine grid in the real space to
      // coarse grid in the Fourier space. Combine with the Laplacian contribution
      SetValue( fft.inputComplexVecFine, Z_ZERO );
      blas::Copy( ntotFine, psiUpdateFine.Data(), 1,
          reinterpret_cast<Real*>(fft.inputComplexVecFine.Data()), 2 );

      // Fine to coarse grid
      // Note the update is important since the Laplacian contribution is already taken into account.
      // The computation order is also important
      fftw_execute( fft.forwardPlanFine );
      {
        Real fac = std::sqrt(Real(ntot) / (Real(ntotFine)));
        Int* idxPtr = fft.idxFineGrid.Data();
        Complex *fftOutFinePtr = fft.outputComplexVecFine.Data();
        Complex *fftOutPtr = fft.outputComplexVec.Data();

        for( Int i = 0; i < ntot; i++ ){
          *(fftOutPtr++) += fftOutFinePtr[*(idxPtr++)] * fac;
        }
      }

      // Inverse Fourier transform to save back to the output vector
      fftw_execute( fft.backwardPlan );
      blas::Axpy( ntot, 1.0 / Real(ntot), 
          reinterpret_cast<Real*>(fft.inputComplexVec.Data()), 2,
          a3.VecData(j,k), 1 );
    }
  }

  return ;
}        // -----  end of method Spinor::AddMultSpinorFine  -----

void
Spinor::AddMultSpinorFineR2C ( Fourier& fft, const DblNumVec& vtot, const DblNumVec& vnlc,
    const IntNumVec& index, const IntNumVec& nGproj, const DblNumVec& coef, NumTns<Real>& a3 )
{

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

  Real timeSta, timeEnd;
  Real timeSta1, timeEnd1;
  
  Real timeFFTCoarse = 0.0;
  Real timeFFTFine = 0.0;
  Real timeNonlocal = 0.0;
  Real timeOther = 0.0;
  Int  iterFFTCoarse = 0;
  Int  iterFFTFine = 0;
  Int  iterNonlocal = 0;
  Int  iterOther = 0;

  GetTime( timeSta1 );
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

      GetTime( timeSta );
      FFTWExecute ( fft, fft.forwardPlanR2C );
      GetTime( timeEnd );
      iterFFTCoarse = iterFFTCoarse + 1;
      timeFFTCoarse = timeFFTCoarse + ( timeEnd - timeSta );

      // Interpolate wavefunction from coarse to fine grid
//1. coarse to fine
// The FFT number is even(esdf.cpp now), change as follows

      if(  fft.FFTtype == 0 )
      {
          Complex *fftOutFinePtr = fft.outputVecR2CFine.Data();
          Complex *fftOutPtr = fft.outputVecR2C.Data();
          IP_c2f(numGrid.Data(),numGridFine.Data(),fftOutPtr,fftOutFinePtr);
      }
      else if( fft.FFTtype == 1 )
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
    
      GetTime( timeSta );
      FFTWExecute ( fft, fft.backwardPlanR2CFine );
      GetTime( timeEnd );
      iterFFTFine = iterFFTFine + 1;
      timeFFTFine = timeFFTFine + ( timeEnd - timeSta );

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
      GetTime( timeSta );
      Int nbeta = coef.m();
      Int start = 0;
      for (Int iobt=0; iobt<nbeta; iobt++) {
        const Real vnlwgt = coef[iobt];
        const Int tempnGproj = nGproj[iobt];
        Real weight = 0.0; 
        const Real *dvptr = &vnlc[start];
        const Int  *ivptr = &index[start];

        for (Int i=0; i<tempnGproj; i++) {
          weight += (*(dvptr++)) * psiFine[*(ivptr++)];
        }
        weight *= vol/Real(ntotFine)*vnlwgt;

        dvptr = &vnlc[start];
        ivptr = &index[start];
        for (Int i=0; i<tempnGproj; i++) {
          psiUpdateFine[*(ivptr++)] += (*(dvptr++)) * weight;
        }

        start += tempnGproj;
      } // for (iobt)

      GetTime( timeEnd );
      iterNonlocal = iterNonlocal + 1;
      timeNonlocal = timeNonlocal + ( timeEnd - timeSta );

      // Laplacian operator. Perform inverse Fourier transform in the end
      {
        for (Int i=0; i<ntotR2C; i++) 
          fft.outputVecR2C(i) *= fft.gkkR2C(i);
      }

      // Restrict psiUpdateFine from fine grid in the real space to
      // coarse grid in the Fourier space. Combine with the Laplacian contribution
      SetValue( fft.inputVecR2CFine, 0.0 );
      blas::Copy( ntotFine, psiUpdateFine.Data(), 1, fft.inputVecR2CFine.Data(), 1 );

      // Fine to coarse grid
      // Note the update is important since the Laplacian contribution is already taken into account.
      // The computation order is also important
      GetTime( timeSta );
      FFTWExecute ( fft, fft.forwardPlanR2CFine );
      GetTime( timeEnd );
      iterFFTFine = iterFFTFine + 1;
      timeFFTFine = timeFFTFine + ( timeEnd - timeSta );
//2. fine to coarse
     if(  fft.FFTtype == 0 )
     {
        Complex *fftOutFinePtr = fft.outputVecR2CFine.Data();
        Complex *fftOutPtr = fft.outputVecR2C.Data();
        IP_f2c(numGrid.Data(),numGridFine.Data(),fftOutPtr,fftOutFinePtr);
     }
     else if(  fft.FFTtype == 1)
     {
        Real fac = sqrt( double(ntotFine) / double(ntot) );
        Int *idxPtr = fft.idxFineGridR2C.Data();
        Complex *fftOutFinePtr = fft.outputVecR2CFine.Data();
        Complex *fftOutPtr = fft.outputVecR2C.Data();
        for( Int i = 0; i < ntotR2C; i++ ){
          *(fftOutPtr++) += fftOutFinePtr[*(idxPtr++)] * fac;
        }
      }
      GetTime( timeSta );
      FFTWExecute ( fft, fft.backwardPlanR2C );
      GetTime( timeEnd );
      iterFFTCoarse = iterFFTCoarse + 1;
      timeFFTCoarse = timeFFTCoarse + ( timeEnd - timeSta );

      // Inverse Fourier transform to save back to the output vector
      //fftw_execute( fft.backwardPlan );

      blas::Axpy( ntot, 1.0, fft.inputVecR2C.Data(), 1, a3.VecData(j,k), 1 );
    } // j++
  } // k++

  GetTime( timeEnd1 );
  iterOther = iterOther + 1;
  timeOther = timeOther + ( timeEnd1 - timeSta1 ) - timeFFTCoarse - timeFFTFine - timeNonlocal;

#if ( _DEBUGlevel_ >= 0 )
    statusOFS << "Time for iterFFTCoarse    = " << iterFFTCoarse       << "  timeFFTCoarse    = " << timeFFTCoarse << std::endl;
    statusOFS << "Time for iterFFTFine      = " << iterFFTFine         << "  timeFFTFine      = " << timeFFTFine << std::endl;
    statusOFS << "Time for iterNonlocal     = " << iterNonlocal        << "  timeNonlocal     = " << timeNonlocal << std::endl;
    statusOFS << "Time for iterOther        = " << iterOther           << "  timeOther        = " << timeOther << std::endl;
#endif

  return ;
}        // -----  end of method Spinor::AddMultSpinorFineR2C  ----- 

#endif

} // namespace PPCG


