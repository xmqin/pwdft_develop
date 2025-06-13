/// @file fourier.cpp
/// @brief Sequential Fourier wrapper.
/// @brief Eigensolver in the global domain.
#include  "ppcg/fourier.hpp"
#include  "ppcg/blas.hpp"

// *********************************************************************
// Sequential FFTW
// *********************************************************************
namespace PPCG {

Fourier::Fourier () : 
  isInitialized(false),
  numGridTotal(0),
  numGridTotalFine(0),
  plannerFlag(FFTW_MEASURE | FFTW_UNALIGNED )
  {
    backwardPlan  = NULL;
    forwardPlan   = NULL;
    backwardPlanR2C  = NULL;
    forwardPlanR2C   = NULL;
    backwardPlanFine  = NULL;
    forwardPlanFine   = NULL;
    backwardPlanR2CFine  = NULL;
    forwardPlanR2CFine   = NULL;
  }

Fourier::~Fourier () 
{
  if( backwardPlan ) fftw_destroy_plan( backwardPlan );
  if( forwardPlan  ) fftw_destroy_plan( forwardPlan );
  if( backwardPlanR2C  ) fftw_destroy_plan( backwardPlanR2C );
  if( forwardPlanR2C   ) fftw_destroy_plan( forwardPlanR2C );
  if( backwardPlanFine ) fftw_destroy_plan( backwardPlanFine );
  if( forwardPlanFine  ) fftw_destroy_plan( forwardPlanFine );
  if( backwardPlanR2CFine  ) fftw_destroy_plan( backwardPlanR2CFine );
  if( forwardPlanR2CFine   ) fftw_destroy_plan( forwardPlanR2CFine );
}

void Fourier::Initialize ( const Domain& dm )
{
  if( isInitialized ) {
    ErrorHandling("Fourier has been initialized.");
  }

  domain = dm;
  Index3& numGrid = domain.numGrid;

  FFTtype = ( (numGrid[0]%2 == 1) || (numGrid[1]%2 == 1) || (numGrid[2]%2 == 1) );
  
  numGridTotal = domain.NumGridTotal();

  inputComplexVec.Resize( numGridTotal );
  outputComplexVec.Resize( numGridTotal );

  forwardPlan = fftw_plan_dft_3d( 
      numGrid[2], numGrid[1], numGrid[0], 
      reinterpret_cast<fftw_complex*>( &inputComplexVec[0] ), 
      reinterpret_cast<fftw_complex*>( &outputComplexVec[0] ),
      FFTW_FORWARD, plannerFlag );

  backwardPlan = fftw_plan_dft_3d(
      numGrid[2], numGrid[1], numGrid[0],
      reinterpret_cast<fftw_complex*>( &outputComplexVec[0] ),
      reinterpret_cast<fftw_complex*>( &inputComplexVec[0] ),
      FFTW_BACKWARD, plannerFlag);

  std::vector<DblNumVec>  KGrid(DIM);                // Fourier grid
  for( Int idim = 0; idim < DIM; idim++ ){
    KGrid[idim].Resize( numGrid[idim] );
    for( Int i = 0; i <= numGrid[idim] / 2; i++ ){
      KGrid[idim](i) = i;
    }
    for( Int i = numGrid[idim] / 2 + 1; i < numGrid[idim]; i++ ){
      KGrid[idim](i) = i - numGrid[idim];
    }
  }

  gkk.Resize( dm.NumGridTotal() ); SetValue( gkk, D_ZERO );
  TeterPrecond.Resize( dm.NumGridTotal() );
  ik.resize(DIM);
  ik[0].Resize( dm.NumGridTotal() );
  ik[1].Resize( dm.NumGridTotal() );
  ik[2].Resize( dm.NumGridTotal() );

  Real*     gkkPtr = gkk.Data();
  Complex*  ikXPtr = ik[0].Data();
  Complex*  ikYPtr = ik[1].Data();
  Complex*  ikZPtr = ik[2].Data();

  Point3 gmesh, gmesh_car;
  for( Int k = 0; k < numGrid[2]; k++ ){
    for( Int j = 0; j < numGrid[1]; j++ ){
      for( Int i = 0; i < numGrid[0]; i++ ){
        gmesh = Point3( KGrid[0](i), KGrid[1](j), KGrid[2](k) );
        gmesh_car = Point3( 0.0, 0.0, 0.0 );
        for( Int ip = 0; ip < DIM; ip++ ){
          for( Int jp = 0; jp < DIM; jp++ ){
            gmesh_car[ip] += dm.recipcell(jp,ip) * gmesh[jp];
          }
        }
        *(gkkPtr++) += ( gmesh_car[0]*gmesh_car[0] + 
            gmesh_car[1]*gmesh_car[1] + gmesh_car[2]*gmesh_car[2] ) / 2;

        *(ikXPtr++) = Complex( 0.0, gmesh_car[0] );
        *(ikYPtr++) = Complex( 0.0, gmesh_car[1] );
        *(ikZPtr++) = Complex( 0.0, gmesh_car[2] );        
      }
    }
  }

  // TeterPreconditioner
  Real  a, b;
  for( Int i = 0; i < domain.NumGridTotal(); i++ ){
    a = gkk[i] * 2.0;
    b = 27.0 + a * (18.0 + a * (12.0 + a * 8.0) );
    TeterPrecond[i] = b / ( b + 16.0 * pow(a, 4.0) );
  }

  // R2C transform
  numGridTotalR2C = (numGrid[0]/2+1) * numGrid[1] * numGrid[2];

  inputVecR2C.Resize( numGridTotal );
  outputVecR2C.Resize( numGridTotalR2C );

  forwardPlanR2C = fftw_plan_dft_r2c_3d( 
      numGrid[2], numGrid[1], numGrid[0], 
      ( &inputVecR2C[0] ), 
      reinterpret_cast<fftw_complex*>( &outputVecR2C[0] ),
      plannerFlag );

  backwardPlanR2C = fftw_plan_dft_c2r_3d(
      numGrid[2], numGrid[1], numGrid[0],
      reinterpret_cast<fftw_complex*>( &outputVecR2C[0] ),
      &inputVecR2C[0],
      plannerFlag);

  // -1/2 \Delta  and Teter preconditioner in R2C
  gkkR2C.Resize( numGridTotalR2C ); SetValue( gkkR2C, D_ZERO );
  TeterPrecondR2C.Resize( numGridTotalR2C );
  ikR2C.resize(DIM);
  ikR2C[0].Resize( numGridTotalR2C );
  ikR2C[1].Resize( numGridTotalR2C );
  ikR2C[2].Resize( numGridTotalR2C );


  Real*  gkkR2CPtr = gkkR2C.Data();
  Complex*  ikXR2CPtr = ikR2C[0].Data();
  Complex*  ikYR2CPtr = ikR2C[1].Data();
  Complex*  ikZR2CPtr = ikR2C[2].Data();

  for( Int k = 0; k < numGrid[2]; k++ ){
    for( Int j = 0; j < numGrid[1]; j++ ){
      for( Int i = 0; i < numGrid[0]/2+1; i++ ){
        gmesh = Point3( KGrid[0](i), KGrid[1](j), KGrid[2](k) );
        gmesh_car = Point3( 0.0, 0.0, 0.0 );
        for( Int ip = 0; ip < DIM; ip++ ){
          for( Int jp = 0; jp < DIM; jp++ ){
            gmesh_car[ip] += dm.recipcell(jp,ip) * gmesh[jp];
          }
        }
        *(gkkR2CPtr++) += ( gmesh_car[0]*gmesh_car[0] +
            gmesh_car[1]*gmesh_car[1] + gmesh_car[2]*gmesh_car[2] ) / 2;

        *(ikXR2CPtr++) = Complex( 0.0, gmesh_car[0] );
        *(ikYR2CPtr++) = Complex( 0.0, gmesh_car[1] );
        *(ikZR2CPtr++) = Complex( 0.0, gmesh_car[2] );
      }
    }
  }


  // TeterPreconditioner
  for( Int i = 0; i < numGridTotalR2C; i++ ){
    a = gkkR2C[i] * 2.0;
    b = 27.0 + a * (18.0 + a * (12.0 + a * 8.0) );
    TeterPrecondR2C[i] = b / ( b + 16.0 * pow(a, 4.0) );
  }

  // Mark Fourier to be initialized
  isInitialized = true;
  return ;
}        // -----  end of function Fourier::Initialize  ----- 


void Fourier::InitializeFine ( const Domain& dm )
{
  domain = dm;
  // FIXME Problematic definition
  Index3& numGrid = domain.numGridFine;

  numGridTotalFine = domain.NumGridTotalFine();

  inputComplexVecFine.Resize( numGridTotalFine );
  outputComplexVecFine.Resize( numGridTotalFine );

  forwardPlanFine = fftw_plan_dft_3d( 
      numGrid[2], numGrid[1], numGrid[0], 
      reinterpret_cast<fftw_complex*>( &inputComplexVecFine[0] ), 
      reinterpret_cast<fftw_complex*>( &outputComplexVecFine[0] ),
      FFTW_FORWARD, plannerFlag );

  backwardPlanFine = fftw_plan_dft_3d(
      numGrid[2], numGrid[1], numGrid[0],
      reinterpret_cast<fftw_complex*>( &outputComplexVecFine[0] ),
      reinterpret_cast<fftw_complex*>( &inputComplexVecFine[0] ),
      FFTW_BACKWARD, plannerFlag);

  std::vector<DblNumVec>  KGrid(DIM);                // Fourier grid
  for( Int idim = 0; idim < DIM; idim++ ){
    KGrid[idim].Resize( numGrid[idim] );
    for( Int i = 0; i <= numGrid[idim] / 2; i++ ){     
      KGrid[idim](i) = i;
    }
    for( Int i = numGrid[idim] / 2 + 1; i < numGrid[idim]; i++ ){
      KGrid[idim](i) =  i - numGrid[idim]; 
    }
  }

  gkkFine.Resize( dm.NumGridTotalFine() ); SetValue( gkkFine, D_ZERO );
  TeterPrecondFine.Resize( dm.NumGridTotalFine() );
  ikFine.resize(DIM);
  ikFine[0].Resize( dm.NumGridTotalFine() );
  ikFine[1].Resize( dm.NumGridTotalFine() );
  ikFine[2].Resize( dm.NumGridTotalFine() );

  Real*     gkkPtr = gkkFine.Data();
  Complex*  ikXPtr = ikFine[0].Data();
  Complex*  ikYPtr = ikFine[1].Data();
  Complex*  ikZPtr = ikFine[2].Data();

  Point3 gmesh, gmesh_car;
  for( Int k = 0; k < numGrid[2]; k++ ){
    for( Int j = 0; j < numGrid[1]; j++ ){
      for( Int i = 0; i < numGrid[0]; i++ ){
        gmesh = Point3( KGrid[0](i), KGrid[1](j), KGrid[2](k) );
        gmesh_car = Point3( 0.0, 0.0, 0.0 );
        for( Int ip = 0; ip < DIM; ip++ ){
          for( Int jp = 0; jp < DIM; jp++ ){
            gmesh_car[ip] += dm.recipcell(jp,ip) * gmesh[jp];
          }
        }
        *(gkkPtr++) += ( gmesh_car[0]*gmesh_car[0] +
            gmesh_car[1]*gmesh_car[1] + gmesh_car[2]*gmesh_car[2] ) / 2;

        *(ikXPtr++) = Complex( 0.0, gmesh_car[0] );
        *(ikYPtr++) = Complex( 0.0, gmesh_car[1] );
        *(ikZPtr++) = Complex( 0.0, gmesh_car[2] );
      }
    }
  }


  // TeterPreconditioner
  Real  a, b;
  for( Int i = 0; i < domain.NumGridTotalFine(); i++ ){
    a = gkkFine[i] * 2.0;
    b = 27.0 + a * (18.0 + a * (12.0 + a * 8.0) );
    TeterPrecondFine[i] = b / ( b + 16.0 * pow(a, 4.0) );
  }

  // Compute the index for mapping coarse to find grid
  idxFineGrid.Resize(domain.NumGridTotal());
  SetValue( idxFineGrid, 0 );
  {
    Int PtrC, PtrF, iF, jF, kF;
    for( Int kk = 0; kk < domain.numGrid[2]; kk++ ){
      for( Int jj = 0; jj < domain.numGrid[1]; jj++ ){
        for( Int ii = 0; ii < domain.numGrid[0]; ii++ ){

          PtrC = ii + jj * domain.numGrid[0] + kk * domain.numGrid[0] * domain.numGrid[1];

          if ( (0 <= ii) && (ii < domain.numGrid[0] / 2) ) { iF = ii; } 
          else if ( (ii == domain.numGrid[0] / 2) ) { iF = domain.numGridFine[0] / 2; } 
          else { iF = domain.numGridFine[0] - domain.numGrid[0] + ii; } 

          if ( (0 <= jj) && (jj < domain.numGrid[1] / 2) ) { jF = jj; } 
          else if ( (jj == domain.numGrid[1] / 2) ) { jF = domain.numGridFine[1] / 2; } 
          else { jF = domain.numGridFine[1] - domain.numGrid[1] + jj; } 

          if ( (0 <= kk) && (kk < domain.numGrid[2] / 2) ) { kF = kk; } 
          else if ( (kk == domain.numGrid[2] / 2) ) { kF = domain.numGridFine[2] / 2; } 
          else { kF = domain.numGridFine[2] - domain.numGrid[2] + kk; } 

          PtrF = iF + jF * domain.numGridFine[0] + kF * domain.numGridFine[0] * domain.numGridFine[1];

          idxFineGrid[PtrC] = PtrF;
        } 
      }
    }
  }

  numGridTotalR2C = (domain.numGrid[0]/2+1) * domain.numGrid[1] * domain.numGrid[2];
  numGridTotalR2CFine = (domain.numGridFine[0]/2+1) * domain.numGridFine[1] * domain.numGridFine[2];

  inputVecR2CFine.Resize( numGridTotalFine );
  outputVecR2CFine.Resize( numGridTotalR2CFine );

  forwardPlanR2CFine = fftw_plan_dft_r2c_3d( 
      numGrid[2], numGrid[1], numGrid[0], 
      ( &inputVecR2CFine[0] ), 
      reinterpret_cast<fftw_complex*>( &outputVecR2CFine[0] ),
      plannerFlag );

  backwardPlanR2CFine = fftw_plan_dft_c2r_3d(
      numGrid[2], numGrid[1], numGrid[0],
      reinterpret_cast<fftw_complex*>( &outputVecR2CFine[0] ),
      &inputVecR2CFine[0],
      plannerFlag);

  // -1/2 \Delta  and Teter preconditioner in R2C
  gkkR2CFine.Resize( numGridTotalR2CFine ); SetValue( gkkR2CFine, D_ZERO );
  TeterPrecondR2CFine.Resize( numGridTotalR2CFine );

  Real*  gkkR2CPtr = gkkR2CFine.Data();

  for( Int k = 0; k < numGrid[2]; k++ ){
    for( Int j = 0; j < numGrid[1]; j++ ){
      for( Int i = 0; i < numGrid[0]/2+1; i++ ){
        gmesh = Point3( KGrid[0](i), KGrid[1](j), KGrid[2](k) );
        gmesh_car = Point3( 0.0, 0.0, 0.0 );
        for( Int ip = 0; ip < DIM; ip++ ){
          for( Int jp = 0; jp < DIM; jp++ ){
            gmesh_car[ip] += dm.recipcell(jp,ip) * gmesh[jp];
          }
        }
        *(gkkR2CPtr++) += ( gmesh_car[0]*gmesh_car[0] +
            gmesh_car[1]*gmesh_car[1] + gmesh_car[2]*gmesh_car[2] ) / 2;
      }
    }
  }

  // TeterPreconditioner
  for( Int i = 0; i < numGridTotalR2CFine; i++ ){
    a = gkkR2CFine[i] * 2.0;
    b = 27.0 + a * (18.0 + a * (12.0 + a * 8.0) );
    TeterPrecondR2CFine[i] = b / ( b + 16.0 * pow(a, 4.0) );
  }

  // Compute the index for mapping coarse to find grid
  idxFineGridR2C.Resize(numGridTotalR2C);
  SetValue( idxFineGridR2C, 0 );
  {
    Int PtrC, PtrF, iF, jF, kF;
    for( Int kk = 0; kk < domain.numGrid[2]; kk++ ){
      for( Int jj = 0; jj < domain.numGrid[1]; jj++ ){
        for( Int ii = 0; ii < (domain.numGrid[0]/2+1); ii++ ){

          PtrC = ii + jj * (domain.numGrid[0]/2+1) + kk * (domain.numGrid[0]/2+1) * domain.numGrid[1];

          if ( (0 <= ii) && (ii < domain.numGrid[0] / 2) ) { iF = ii; } 
          else if ( (ii == domain.numGrid[0] / 2) ) { iF = domain.numGridFine[0] / 2; } 
          else { iF = (domain.numGridFine[0]/2+1) - (domain.numGrid[0]/2+1) + ii; } 

          if ( (0 <= jj) && (jj < domain.numGrid[1] / 2) ) { jF = jj; } 
          else if ( (jj == domain.numGrid[1] / 2) ) { jF = domain.numGridFine[1] / 2; } 
          else { jF = domain.numGridFine[1] - domain.numGrid[1] + jj; } 

          if ( (0 <= kk) && (kk < domain.numGrid[2] / 2) ) { kF = kk; } 
          else if ( (kk == domain.numGrid[2] / 2) ) { kF = domain.numGridFine[2] / 2; } 
          else { kF = domain.numGridFine[2] - domain.numGrid[2] + kk; } 

          PtrF = iF + jF * (domain.numGridFine[0]/2+1) + kF * (domain.numGridFine[0]/2+1) * domain.numGridFine[1];

          idxFineGridR2C[PtrC] = PtrF;
        } 
      }
    }
  }

  return ;
}        // -----  end of function Fourier::InitializeFine  ----- 

void FFTWExecute ( Fourier& fft, fftw_plan& plan ){

  Index3& numGrid = fft.domain.numGrid;
  Index3& numGridFine = fft.domain.numGridFine;
  Int ntot      = fft.domain.NumGridTotal();
  Int ntotFine  = fft.domain.NumGridTotalFine();
  Real vol      = fft.domain.Volume();
  Real fac;

  Int ntotR2C = (numGrid[0]/2+1) * numGrid[1] * numGrid[2];
  Int ntotR2CFine = (numGridFine[0]/2+1) * numGridFine[1] * numGridFine[2];

  if ( plan == fft.backwardPlan )
  {
    fftw_execute( fft.backwardPlan );
    fac = 1.0 / vol;
    blas::Scal( ntot, fac, fft.inputComplexVec.Data(), 1);
  }

  if ( plan == fft.forwardPlan )
  {
    fftw_execute( fft.forwardPlan );
    fac = vol / double(ntot);
    blas::Scal( ntot, fac, fft.outputComplexVec.Data(), 1);
  }

  if ( plan == fft.backwardPlanR2C )
  {
    fftw_execute( fft.backwardPlanR2C );
    fac = 1.0 / vol;
    blas::Scal( ntot, fac, fft.inputVecR2C.Data(), 1);
  }

  if ( plan == fft.forwardPlanR2C )
  {
    fftw_execute( fft.forwardPlanR2C );
    fac = vol / double(ntot);
    blas::Scal( ntotR2C, fac, fft.outputVecR2C.Data(), 1);
  }

  if ( plan == fft.backwardPlanFine )
  {
    fftw_execute( fft.backwardPlanFine );
    fac = 1.0 / vol;
    blas::Scal( ntotFine, fac, fft.inputComplexVecFine.Data(), 1);
  }

  if ( plan == fft.forwardPlanFine )
  {
    fftw_execute( fft.forwardPlanFine );
    fac = vol / double(ntotFine);
    blas::Scal( ntotFine, fac, fft.outputComplexVecFine.Data(), 1);
  }

  if ( plan == fft.backwardPlanR2CFine )
  {
    fftw_execute( fft.backwardPlanR2CFine );
    fac = 1.0 / vol;
    blas::Scal( ntotFine, fac, fft.inputVecR2CFine.Data(), 1);
  }

  if ( plan == fft.forwardPlanR2CFine )
  {
    fftw_execute( fft.forwardPlanR2CFine );
    fac = vol / double(ntotFine);
    blas::Scal( ntotR2CFine, fac, fft.outputVecR2CFine.Data(), 1);
  }

  return ;
}        // -----  end of function Fourier::FFTWExecute  ----- 

} // namespace PPCG




