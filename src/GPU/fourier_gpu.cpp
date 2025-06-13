/// @file fourier_gpu.cpp
/// @brief GPU-ified sequential Fourier wrapper.
/// @date 2024-02-04
#include  "fourier.hpp"
#include  "blas.hpp"

#ifdef GPU
#include "cublas.hpp"
#include "mpi_interf.hpp"
#endif 

namespace pwdft{

void cuFFTExecuteForward2( Fourier& fft, cufftHandle &plan, int fft_type, cuCpxNumVec &cu_psi_in, cuCpxNumVec &cu_psi_out )
{
  Index3& numGrid = fft.domain.numGrid;
  Index3& numGridFine = fft.domain.numGridFine;
  Real vol      = fft.domain.Volume();
  Int ntot      = fft.domain.NumGridTotal();
  Int ntotFine  = fft.domain.NumGridTotalFine();
  Real factor;
  
  if(fft_type > 0) // fine grid FFT.
  {
    factor = vol/ntotFine;
    assert( cufftExecZ2Z(plan, cu_psi_in.Data(), cu_psi_out.Data(), CUFFT_FORWARD)  == CUFFT_SUCCESS );
    cublas::Scal( ntotFine, &factor, cu_psi_out.Data(),1);
  }
  else // coarse grid FFT.
  {
    factor = vol/ntot;
    assert( cufftExecZ2Z(plan, cu_psi_in.Data(), cu_psi_out.Data(), CUFFT_FORWARD)  == CUFFT_SUCCESS );
    cublas::Scal(ntot, &factor, cu_psi_out.Data(), 1);
  }
}        // -----  end of function Fourier::cuFFTExecuteForward2  -----

void cuFFTExecuteForward( Fourier& fft, cufftHandle &plan, int fft_type, cuCpxNumVec &cu_psi_in, cuCpxNumVec &cu_psi_out )
{
  Index3& numGrid = fft.domain.numGrid;
  Index3& numGridFine = fft.domain.numGridFine;
  Real vol      = fft.domain.Volume();
  Int ntot      = fft.domain.NumGridTotal();
  Int ntotFine  = fft.domain.NumGridTotalFine();
  Real factor;

  if(fft_type > 0) // fine grid FFT.
  {
    factor = vol/ntotFine;
    assert( cufftExecZ2Z(plan, cu_psi_in.Data(), cu_psi_out.Data(), CUFFT_FORWARD)  == CUFFT_SUCCESS );
    cublas::Scal( ntotFine, &factor, cu_psi_out.Data(),1);
  }
  else // coarse grid FFT.   
  {
    assert( cufftExecZ2Z(plan, cu_psi_in.Data(), cu_psi_out.Data(), CUFFT_FORWARD)  == CUFFT_SUCCESS );
  }
}        // -----  end of function Fourier::cuFFTExecuteForward  -----

void cuFFTExecuteInverse2( Fourier& fft, cufftHandle &plan, int fft_type, cuCpxNumVec &cu_psi_in, cuCpxNumVec &cu_psi_out )
{
  Index3& numGrid = fft.domain.numGrid;
  Index3& numGridFine = fft.domain.numGridFine;
  Real vol      = fft.domain.Volume();
  Int ntot      = fft.domain.NumGridTotal();
  Int ntotFine  = fft.domain.NumGridTotalFine();
  Real factor;

  if(fft_type > 0) // fine grid FFT.
  {
    factor = 1.0 / vol;
    assert( cufftExecZ2Z(plan, reinterpret_cast<cuDoubleComplex*> (cu_psi_in.Data()), cu_psi_out.Data(), CUFFT_INVERSE) == CUFFT_SUCCESS );
    cublas::Scal(ntotFine, &factor, cu_psi_out.Data(),1);
  }
  else // coarse grid FFT.
  {
    assert( cufftExecZ2Z(plan, reinterpret_cast<cuDoubleComplex*> (cu_psi_in.Data()), cu_psi_out.Data(), CUFFT_INVERSE) == CUFFT_SUCCESS );
  }
}        // -----  end of function Fourier::cuFFTExecuteInverse2

void cuFFTExecuteInverse( Fourier& fft, cufftHandle &plan, int fft_type, cuCpxNumVec &cu_psi_in, cuCpxNumVec &cu_psi_out )
{
  Index3& numGrid = fft.domain.numGrid;
  Index3& numGridFine = fft.domain.numGridFine;
  Real vol      = fft.domain.Volume();
  Int ntot      = fft.domain.NumGridTotal();
  Int ntotFine  = fft.domain.NumGridTotalFine();
  Real factor;

  if(fft_type > 0) // fine grid FFT.
  {
    factor = 1.0 / vol;
    assert( cufftExecZ2Z(plan, reinterpret_cast<cuDoubleComplex*> (cu_psi_in.Data()), cu_psi_out.Data(), CUFFT_INVERSE) == CUFFT_SUCCESS );
    cublas::Scal(ntotFine, &factor, cu_psi_out.Data(),1);
  }
  else // coarse grid FFT.
  {
    factor = 1.0 / Real(ntot);
    assert( cufftExecZ2Z(plan, reinterpret_cast<cuDoubleComplex*> (cu_psi_in.Data()), cu_psi_out.Data(), CUFFT_INVERSE) == CUFFT_SUCCESS );
    cublas::Scal(ntot, &factor, cu_psi_out.Data(), 1);
  }
}        // -----  end of function Fourier::cuFFTExecuteInverse

void cuFFTExecuteInverse( Fourier& fft, cufftHandle &plan, int fft_type, cuCpxNumVec &cu_psi_in, cuCpxNumVec &cu_psi_out , int nbands)
{
   Index3& numGrid = fft.domain.numGrid;
   Index3& numGridFine = fft.domain.numGridFine;
   Real vol      = fft.domain.Volume();
   Int ntot      = fft.domain.NumGridTotal();
   Int ntotFine  = fft.domain.NumGridTotalFine();
   Real factor;

   if(fft_type > 0) // fine grid FFT.
   {
      factor = 1.0 / vol;
      assert( cufftExecZ2Z(plan, reinterpret_cast<cuDoubleComplex*> (cu_psi_in.Data()), cu_psi_out.Data(), CUFFT_INVERSE) == CUFFT_SUCCESS );
      cublas::Scal(ntotFine, &factor, cu_psi_out.Data(),1);
   }
   else // coarse grid FFT.
   {
      factor = 1.0 / Real(ntot*nbands);
      assert( cufftExecZ2Z(plan, reinterpret_cast<cuDoubleComplex*> (cu_psi_in.Data()), cu_psi_out.Data(), CUFFT_INVERSE) == CUFFT_SUCCESS );
      cublas::Scal(ntot, &factor, cu_psi_out.Data(), 1);  
   }
}        // -----  end of function Fourier::cuFFTExecuteInverse

void cuFFTExecuteForward( Fourier& fft, cufftHandle &plan, int fft_type, cuDblNumVec &cu_psi_in, cuDblNumVec &cu_psi_out )
{
   Index3& numGrid = fft.domain.numGrid;
   Index3& numGridFine = fft.domain.numGridFine;
   Real vol      = fft.domain.Volume();
   Int ntot      = fft.domain.NumGridTotal();
   Int ntotFine  = fft.domain.NumGridTotalFine();
   Int ntotR2C = (numGrid[0]/2+1) * numGrid[1] * numGrid[2];
   Int ntotR2CFine = (numGridFine[0]/2+1) * numGridFine[1] * numGridFine[2];

   Real factor;
   if(fft_type > 0) // fine grid FFT.
   {
      factor = vol/ntotFine;
      assert( cufftExecD2Z(plan, cu_psi_in.Data(), reinterpret_cast<cuDoubleComplex*> (cu_psi_out.Data())) == CUFFT_SUCCESS );
      cublas::Scal(2*ntotR2CFine, &factor, cu_psi_out.Data(),1);
   }
   else // coarse grid FFT.
   {
      factor = vol/ntot;
      assert( cufftExecD2Z(plan, cu_psi_in.Data(), reinterpret_cast<cuDoubleComplex*> (cu_psi_out.Data())) == CUFFT_SUCCESS );
      cublas::Scal(2*ntotR2C, &factor, cu_psi_out.Data(), 1);
   }
}        // -----  end of function Fourier::cuFFTExecuteForward

void cuFFTExecuteInverse( Fourier& fft, cufftHandle &plan, int fft_type, cuDblNumVec &cu_psi_in, cuDblNumVec &cu_psi_out )
{
   Index3& numGrid = fft.domain.numGrid;
   Index3& numGridFine = fft.domain.numGridFine;
   Real vol      = fft.domain.Volume();
   Int ntot      = fft.domain.NumGridTotal();
   Int ntotFine  = fft.domain.NumGridTotalFine();
   Int ntotR2C = (numGrid[0]/2+1) * numGrid[1] * numGrid[2];
   Int ntotR2CFine = (numGridFine[0]/2+1) * numGridFine[1] * numGridFine[2];

   Real factor;
   if(fft_type > 0) // fine grid FFT.
   {
      factor = 1.0 / vol;
      assert( cufftExecZ2D(plan, reinterpret_cast<cuDoubleComplex*> (cu_psi_in.Data()), cu_psi_out.Data()) == CUFFT_SUCCESS );
      cublas::Scal(ntotFine, &factor, cu_psi_out.Data(),1);
   }
   else // coarse grid FFT.
   {
      factor = 1.0 / vol;
      assert( cufftExecZ2D(plan, reinterpret_cast<cuDoubleComplex*> (cu_psi_in.Data()), cu_psi_out.Data()) == CUFFT_SUCCESS );
      cublas::Scal(ntot, &factor, cu_psi_out.Data(), 1);
   }
}        // -----  end of function Fourier::cuFFTExecuteInverse

} // namespace pwdft
















