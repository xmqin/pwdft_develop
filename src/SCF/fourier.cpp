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
/// @file fourier.cpp
/// @brief Sequential and Distributed Fourier wrapper.
/// @date 2011-11-01
/// @date 2014-02-01 Dual grid implementation.
#include  "fourier.hpp"
#include  "blas.hpp"
#include  "utility.hpp"
namespace pwdft{

using namespace pwdft::esdf;

// *********************************************************************
// Sequential FFTW
// *********************************************************************

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
#ifdef GPU
  for( Int i = 0; i < NSTREAM; i++ )
  {
    cufftDestroy(cuPlanR2C[i]);
    cufftDestroy(cuPlanR2CFine[i]);
    cufftDestroy(cuPlanC2R[i]);
    cufftDestroy(cuPlanC2RFine[i]);
    cufftDestroy(cuPlanC2C[i]);
    cufftDestroy(cuPlanC2CFine[i]);
  }
#endif
}

void Fourier::Initialize ( const Domain& dm )
{
  if( isInitialized ) {
    ErrorHandling("Fourier has been initialized.");
  }

  domain = dm;
  Index3& numGrid = domain.numGrid;
  Point3& length  = domain.length;

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

  std::vector<DblNumVec>  KGrid(DIM); // Fourier grid
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
#ifdef GPU
  mpi::cuda_setDevice(MPI_COMM_WORLD);
  for( Int i = 0; i < NSTREAM; i++ )
  {
    cufftPlan3d(&cuPlanR2C[i], numGrid[2], numGrid[1], numGrid[0], CUFFT_D2Z);
    cufftPlan3d(&cuPlanC2R[i], numGrid[2], numGrid[1], numGrid[0], CUFFT_Z2D);
    cufftPlan3d(&cuPlanC2C[i], numGrid[2], numGrid[1], numGrid[0], CUFFT_Z2Z);
  }
#endif
  // -1/2 \Delta in R2C
  gkkR2C.Resize( numGridTotalR2C ); SetValue( gkkR2C, D_ZERO );
  ikR2C.resize(DIM);
  ikR2C[0].Resize( numGridTotalR2C );
  ikR2C[1].Resize( numGridTotalR2C );
  ikR2C[2].Resize( numGridTotalR2C );

  iKR2C.resize(DIM);
  iKR2C[0].Resize( numGridTotalR2C );
  iKR2C[1].Resize( numGridTotalR2C );
  iKR2C[2].Resize( numGridTotalR2C );

  Real*  gkkR2CPtr = gkkR2C.Data();
  Complex*  ikXR2CPtr = ikR2C[0].Data();
  Complex*  ikYR2CPtr = ikR2C[1].Data();
  Complex*  ikZR2CPtr = ikR2C[2].Data();

  Int*      iKXR2CPtr = iKR2C[0].Data();
  Int*      iKYR2CPtr = iKR2C[1].Data();
  Int*      iKZR2CPtr = iKR2C[2].Data();

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

        *(iKXR2CPtr++) = gmesh[0];
        *(iKYR2CPtr++) = gmesh[1];
        *(iKZR2CPtr++) = gmesh[2];
      }
    }
  }

  // Mark Fourier to be initialized
  isInitialized = true;

  return ;
}        // -----  end of function Fourier::Initialize  ----- 

void Fourier::InitializeFine ( const Domain& dm )
{
  domain = dm;
  Index3& numGrid = domain.numGridFine;
  Point3& length  = domain.length;

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

  std::vector<DblNumVec>  KGrid(DIM); // Fourier grid
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

  {
    // Compute the index for mapping coarse grid to fine grid
    idxFineGrid.Resize(domain.NumGridTotal());
    SetValue( idxFineGrid, 0 );
  
    Int PtrC, PtrF, iF, jF, kF;
    for( Int kk = 0; kk < domain.numGrid[2]; kk++ ){
      for( Int jj = 0; jj < domain.numGrid[1]; jj++ ){
        for( Int ii = 0; ii < domain.numGrid[0]; ii++ ){

          PtrC = ii + jj * domain.numGrid[0] + kk * domain.numGrid[0] * domain.numGrid[1];
          
          if ( (0 <= ii) && (ii <= domain.numGrid[0] / 2) ) { iF = ii; } 
          else { iF = domain.numGridFine[0] - domain.numGrid[0] + ii; } 

          if ( (0 <= jj) && (jj <= domain.numGrid[1] / 2) ) { jF = jj; } 
          else { jF = domain.numGridFine[1] - domain.numGrid[1] + jj; } 

          if ( (0 <= kk) && (kk <= domain.numGrid[2] / 2) ) { kF = kk; } 
          else { kF = domain.numGridFine[2] - domain.numGrid[2] + kk; } 

          PtrF = iF + jF * domain.numGridFine[0] + kF * domain.numGridFine[0] * domain.numGridFine[1];

          idxFineGrid[PtrC] = PtrF;
        } 
      }
    }
  }
  
  // R2C transform
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
#ifdef GPU
  for( Int i = 0; i < NSTREAM; i++ )
  {
    cufftPlan3d(&cuPlanR2CFine[i], numGrid[2], numGrid[1], numGrid[0], CUFFT_D2Z);
    cufftPlan3d(&cuPlanC2RFine[i], numGrid[2], numGrid[1], numGrid[0], CUFFT_Z2D);
    cufftPlan3d(&cuPlanC2CFine[i], numGrid[2], numGrid[1], numGrid[0], CUFFT_Z2Z);
  }
#endif
  // -1/2 \Delta in R2C
  gkkR2CFine.Resize( numGridTotalR2CFine ); SetValue( gkkR2CFine, D_ZERO );

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

  // Compute the index for mapping coarse to find grid
  idxFineGridR2C.Resize(numGridTotalR2C);
  SetValue( idxFineGridR2C, 0 );
  {
    Int PtrC, PtrF, iF, jF, kF;
    for( Int kk = 0; kk < domain.numGrid[2]; kk++ ){
      for( Int jj = 0; jj < domain.numGrid[1]; jj++ ){
        for( Int ii = 0; ii < (domain.numGrid[0]/2+1); ii++ ){

          PtrC = ii + jj * (domain.numGrid[0]/2+1) + kk * (domain.numGrid[0]/2+1) * domain.numGrid[1];

          iF = ii; 

          if ( (0 <= jj) && (jj <= domain.numGrid[1] / 2) ) { jF = jj; } 
          else { jF = domain.numGridFine[1] - domain.numGrid[1] + jj; } 

          if ( (0 <= kk) && (kk <= domain.numGrid[2] / 2) ) { kF = kk; } 
          else { kF = domain.numGridFine[2] - domain.numGrid[2] + kk; } 
          
          PtrF = iF + jF * (domain.numGridFine[0]/2+1) + kF * (domain.numGridFine[0]/2+1) * domain.numGridFine[1];

          idxFineGridR2C[PtrC] = PtrF;
        } 
      }
    }
  }

  return ;
}        // -----  end of function Fourier::InitializeFine  ----- 

void Fourier::InitializeSphere ( Domain& dm ){

  bool spherecut = esdfParam.isUseSphereCut;

#ifdef _COMPLEX_
  if( spherecut ){    
    // Compute the index for mapping the sphere cutted wavefunction grid to coarse/fine grid
    IntNumVec& KpointIdx = domain.KpointIdx;
    Int nkLocal = KpointIdx.Size();
    Int ntot = domain.NumGridTotal();  

    idxCoarseCut.resize( nkLocal );
    idxFineCut.resize( nkLocal );

    domain.numGridSphere.Resize( nkLocal );
    dm.numGridSphere.Resize( nkLocal );

    for( Int k = 0; k < nkLocal; k++ ){
      std::vector<Int> idxGrid;
     
      Int idk = KpointIdx(k); 
      Point3 kpoint = Point3( domain.klist[0][idk], domain.klist[1][idk], domain.klist[2][idk] );
      
      for( Int i = 0; i < ntot; i++ ){

        Point3 kG = ( kpoint + Point3( ik[0][i].imag(), ik[1][i].imag(), ik[2][i].imag() ) );

        Real gkk = ( kG[0]*kG[0] + kG[1]*kG[1] + kG[2]*kG[2] ) / 2.0;

        if( gkk < esdfParam.ecutWavefunction ){
          idxGrid.push_back(i);
        }
      }

      Int idxsize = idxGrid.size();

      idxCoarseCut[k].Resize( idxsize );
      for( Int i = 0; i < idxsize; i++ ) idxCoarseCut[k][i] = idxGrid[i];

      idxFineCut[k].Resize( idxsize );
      for( Int i = 0; i < idxsize; i++ ) idxFineCut[k][i] = idxFineGrid[idxCoarseCut[k][i]];  

      domain.numGridSphere[k] = idxsize;
      dm.numGridSphere[k] = idxsize;
    }

    // Compute the index for mapping the Fock grid to coarse/fine grid
    // here we select Fock grid as the cutted density grid
    // FIXME a parameter EcutFock should be added 
    {
      std::set<Int> idxGridFock;
      Point3 gamma = Point3( 0.0, 0.0, 0.0 );
      for( Int i = 0; i < ntot; i++ ){
        Point3 kG = ( gamma + Point3( ik[0][i].imag(), ik[1][i].imag(), ik[2][i].imag() ) );

        Real gkk = ( kG[0]*kG[0] + kG[1]*kG[1] + kG[2]*kG[2] ) / 2.0;

        if( gkk < esdfParam.ecutWavefunction * 4.0 ){
          idxGridFock.insert(i);
        }
      }

      Int idxsizeFock = idxGridFock.size();

      idxCoarseFock.Resize( idxsizeFock );

      Int i = 0;
      for( std::set<Int>::iterator igrid = idxGridFock.begin();
          igrid != idxGridFock.end(); igrid++ ){
        idxCoarseFock[i++] = *igrid;
      }

      idxFineFock.Resize( idxsizeFock );
      for( Int i = 0; i < idxsizeFock; i++ ) idxFineFock[i] = idxFineGrid[idxCoarseFock[i]];

      domain.numGridFock = idxsizeFock;
      dm.numGridFock = idxsizeFock;
    }

    // Calculate the SCF grids for hybrid energy band calculation
    // with fake-SCF method
    if( esdfParam.isCalculateEnergyBand &&
        esdfParam.XCType == "XC_HYB_GGA_XC_HSE06"  ){

      IntNumVec& KpointIdx_scf = domain.KpointIdx_scf;
      nkLocal = KpointIdx_scf.Size();
      idxCoarseCutSCF.resize( nkLocal );
      idxFineCutSCF.resize( nkLocal );
    
      domain.numGridSphereSCF.Resize( nkLocal );
      dm.numGridSphereSCF.Resize( nkLocal );

      for( Int k = 0; k < nkLocal; k++ ){
        std::vector<Int> idxGrid;

        Int idk = KpointIdx_scf(k);
        Point3 kpoint = Point3( domain.klist_scf[0][idk], domain.klist_scf[1][idk], 
            domain.klist_scf[2][idk] );

        for( Int i = 0; i < ntot; i++ ){

          Point3 kG = ( kpoint + Point3( ik[0][i].imag(), ik[1][i].imag(), ik[2][i].imag() ) );

          Real gkk = ( kG[0]*kG[0] + kG[1]*kG[1] + kG[2]*kG[2] ) / 2.0;

          if( gkk < esdfParam.ecutWavefunction ){
            idxGrid.push_back(i);
          }
        }

        Int idxsize = idxGrid.size();
        idxCoarseCutSCF[k].Resize( idxsize );
        for( Int i = 0; i < idxsize; i++ ) idxCoarseCutSCF[k][i] = idxGrid[i];
        idxFineCutSCF[k].Resize( idxsize );
        for( Int i = 0; i < idxsize; i++ ) idxFineCutSCF[k][i] = idxFineGrid[idxCoarseCutSCF[k][i]];

        domain.numGridSphereSCF[k] = idxsize;
        dm.numGridSphereSCF[k] = idxsize;
      }
    }
  }
#else
  if( spherecut ){ 
    Index3& numGrid = domain.numGrid;

    Int ntotR2C = numGridTotalR2C;

    std::vector<Int> idxGrid;
 
    // Debug
    std::vector<Int> idxR2CGrid;

    std::vector<Int> idxSource, idxDest;

    for( Int i = 0; i < ntotR2C; i++ ){

      Index3 G = Index3( iKR2C[0][i], iKR2C[1][i], iKR2C[2][i] );

      Real gkk = gkkR2C[i];

      Real eps = 1e-16;

      bool inGrid = false;
      bool doPadding = false;
      bool inSphere = ( gkk < esdfParam.ecutWavefunction );
 
      if( G[0] > eps ){
        inGrid = true;
      }
      else{ 
        if( G[1] > eps ){
          inGrid = true;
        }
        else if( G[1] > -eps ){
          if( G[2] >= -eps ){
            inGrid = true;
          }
        }
      }

      if( G[0] < eps ){
        if( G[1] < -eps ){
          doPadding = true;
        }
        else if( G[1] < eps ){
          if( G[2] < -eps ){
            doPadding = true;
          }
        }
      }

      if( inGrid && inSphere ){
        idxGrid.push_back(i);
        Int n1 = G[0], n2 = G[1], n3 = G[2];
        if( n1 < 0 ) n1 = n1 + numGrid[0];
        if( n2 < 0 ) n2 = n2 + numGrid[1];
        if( n3 < 0 ) n3 = n3 + numGrid[2];
        Int idx = n1 + numGrid[0]*n2
            + numGrid[0]*numGrid[1]*n3;
        idxR2CGrid.push_back(idx);  
      }
      
      if( doPadding && inSphere ){
        Int n1 = G[0], n2 = -G[1], n3 = -G[2];
        if( n2 < 0 ) n2 = n2 + numGrid[1];
        if( n3 < 0 ) n3 = n3 + numGrid[2];
        Int idx = n1 + (numGrid[0]/2+1)*n2 
            + (numGrid[0]/2+1)*numGrid[1]*n3;   
        if( idx < ntotR2C ){
          idxSource.push_back(idx);
          idxDest.push_back(i);
        }
      }
    }    

    Int idxsize = idxGrid.size();

    idxCoarseCut.Resize( idxsize );
    for( Int i = 0; i < idxsize; i++ ) idxCoarseCut[i] = idxGrid[i];

    idxFineCut.Resize( idxsize );
    for( Int i = 0; i < idxsize; i++ ) idxFineCut[i] = idxFineGridR2C[idxCoarseCut[i]];

    idxR2C.Resize( idxsize );
    for( Int i = 0; i < idxsize; i++ ) idxR2C[i] = idxR2CGrid[i];

    domain.numGridSphere = idxsize;
    dm.numGridSphere = idxsize;

    idxsize = idxSource.size();
    idxCoarsePadding.first.Resize( idxsize );
    idxCoarsePadding.second.Resize( idxsize );
    idxFinePadding.first.Resize( idxsize );
    idxFinePadding.second.Resize( idxsize ); 

    for( Int i = 0; i < idxsize; i++ ){
      idxCoarsePadding.first[i] = idxSource[i];
      idxCoarsePadding.second[i] = idxDest[i];
      idxFinePadding.first[i] = idxFineGridR2C[idxSource[i]];
      idxFinePadding.second[i] = idxFineGridR2C[idxDest[i]];     
    }

    { 
      std::set<Int> idxGridFock;
      Point3 gamma = Point3( 0.0, 0.0, 0.0 );
      for( Int i = 0; i < ntotR2C; i++ ){
        Point3 kG = ( gamma + Point3( ikR2C[0][i].imag(), ikR2C[1][i].imag(), ikR2C[2][i].imag() ) );
        
        Real gkk = ( kG[0]*kG[0] + kG[1]*kG[1] + kG[2]*kG[2] ) / 2.0;
        
        if( gkk < esdfParam.ecutWavefunction * 4.0 ){
          idxGridFock.insert(i);
        }
      }
      
      Int idxsizeFock = idxGridFock.size();
      
      idxCoarseFock.Resize( idxsizeFock );
      
      Int i = 0;
      for( std::set<Int>::iterator igrid = idxGridFock.begin();
          igrid != idxGridFock.end(); igrid++ ){
        idxCoarseFock[i++] = *igrid;
      }
      
      idxFineFock.Resize( idxsizeFock );
      for( Int i = 0; i < idxsizeFock; i++ ) idxFineFock[i] = idxFineGrid[idxCoarseFock[i]];
      
      domain.numGridFock = idxsizeFock;
      dm.numGridFock = idxsizeFock;
    }
  }
#endif

  // Compute the index for mapping the sphere cutted density grid to fine grid 
  {
    std::set<Int> idxGridDensity;
    Int ntotFine = domain.NumGridTotalFine();
    for( Int i = 0; i < ntotFine; i++ ){
      if( gkkFine(i) < esdfParam.ecutWavefunction * 4.0 )
        idxGridDensity.insert(i);
    }

    Int idxsizeDensity = idxGridDensity.size();

    idxFineCutDensity.Resize( idxsizeDensity );

    Int i = 0;
    for( std::set<Int>::iterator igrid = idxGridDensity.begin();
        igrid != idxGridDensity.end(); igrid++ ){
      idxFineCutDensity[i++] = *igrid;
    }
  }
}        // -----  end of function Fourier::InitializeSphere  ----- 

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

} // namespace pwdft
