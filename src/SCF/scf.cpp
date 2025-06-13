/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

Author: Lin Lin, Wei Hu and Amartya Banerjee

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
/// @file scf.cpp
/// @brief SCF class for the global domain or extended element.
/// @date 2012-10-25 Initial version
/// @date 2014-02-01 Dual grid implementation
/// @date 2014-08-07 Parallelization for PWDFT
/// @date 2016-01-19 Add hybrid functional
/// @date 2016-04-08 Update mixing
/// @date 2023-11-01 Add spin and k-points
#include  "scf.hpp"
#include  "blas.hpp"
#include  "lapack.hpp"
#include  "scalapack.hpp"
#include  "mpi_interf.hpp"
#include  "utility.hpp"
#include  "spinor.hpp"
#include  "periodtable.hpp"

#ifdef GPU
#include "cuda_utils.h"
#include  "cublas.hpp"
#endif

namespace  pwdft{

using namespace pwdft::DensityComponent;
using namespace pwdft::SpinTwo;
using namespace pwdft::esdf;
using namespace pwdft::scalapack;

SCF::SCF    (  )
{
  eigSolPtr_ = NULL;
  ptablePtr_ = NULL;
}         // -----  end of method SCF::SCF  ----- 

SCF::~SCF    (  )
{

}         // -----  end of method SCF::~SCF  ----- 

void
SCF::Setup    ( EigenSolver& eigSol, PeriodTable& ptable )
{
  int mpirank;  MPI_Comm_rank(esdfParam.domain.comm, &mpirank);
  int mpisize;  MPI_Comm_size(esdfParam.domain.comm, &mpisize);
  Real timeSta, timeEnd;

  // esdf parameters
  {
    spinType_      = esdfParam.spinType;      
    mixMaxDim_     = esdfParam.mixMaxDim;
    mixType_       = esdfParam.mixType;
    mixVariable_   = esdfParam.mixVariable;
    mixStepLength_ = esdfParam.mixStepLength;
    // Note: for PW SCF there is no inner loop. Use the parameter value
    // for the outer SCF loop only.
    eigTolerance_  = esdfParam.eigTolerance;
    eigMinTolerance_  = esdfParam.eigMinTolerance;
    eigMaxIter_    = esdfParam.eigMaxIter;
    scfTolerance_  = esdfParam.scfInnerTolerance;
    scfMaxIter_    = esdfParam.scfInnerMaxIter;
    scfPhiMaxIter_ = esdfParam.scfPhiMaxIter;
    scfPhiTolerance_ = esdfParam.scfPhiTolerance;
    isEigToleranceDynamic_ = esdfParam.isEigToleranceDynamic;
    Tbeta_         = esdfParam.Tbeta;
    BlockSizeScaLAPACK_      = esdfParam.BlockSizeScaLAPACK;
    // Chebyshev Filtering related parameters
    if(esdfParam.PWSolver == "CheFSI")
      Diag_SCF_PWDFT_by_Cheby_ = 1;
    else
      Diag_SCF_PWDFT_by_Cheby_ = 0;

    First_SCF_PWDFT_ChebyFilterOrder_ = esdfParam.First_SCF_PWDFT_ChebyFilterOrder;
    First_SCF_PWDFT_ChebyCycleNum_ = esdfParam.First_SCF_PWDFT_ChebyCycleNum;
    General_SCF_PWDFT_ChebyFilterOrder_ = esdfParam.General_SCF_PWDFT_ChebyFilterOrder;
    PWDFT_Cheby_use_scala_ = esdfParam.PWDFT_Cheby_use_scala;
    PWDFT_Cheby_apply_wfn_ecut_filt_ =  esdfParam.PWDFT_Cheby_apply_wfn_ecut_filt;
    Cheby_iondynamics_schedule_flag_ = 0;
  }

  // other SCF parameters
  {
    eigSolPtr_ = &eigSol;

    ptablePtr_ = &ptable;

    Int ntotFine = esdfParam.domain.NumGridTotalFine();
    Int nspin = esdfParam.domain.numSpinComponent;

    vtotNew_.Resize(ntotFine, nspin); SetValue(vtotNew_, 0.0);
    densityOld_.Resize(ntotFine, nspin); SetValue(densityOld_, 0.0);
    // Anderson mixing is performed in real space, while Broyden mixing
    // is performed in reciprocal space
    if( mixType_ == "anderson"|| mixType_ == "kerker+anderson" ){
      dfMat_.Resize( ntotFine, nspin, mixMaxDim_ ); SetValue( dfMat_, D_ZERO );
      dvMat_.Resize( ntotFine, nspin, mixMaxDim_ ); SetValue( dvMat_, D_ZERO );
    }
    else if( mixType_ == "broyden"){
      // Cut saved density or potential in Fourier space to save memory
      Int ntot = eigSol.FFT().idxFineCutDensity.Size();

      GdfMat_.Resize( ntot, nspin, mixMaxDim_ ); SetValue( GdfMat_, Z_ZERO );
      GdvMat_.Resize( ntot, nspin, mixMaxDim_ ); SetValue( GdvMat_, Z_ZERO );
      GcdfMat_.Resize( ntot, nspin, 2 ); SetValue( GcdfMat_, Z_ZERO );
    }
    else{
       ErrorHandling("Invalid mixing type.");
    }
    
    restartDensityFileName_   = "DEN";
    restartPotentialFileName_ = "POT";
    restartWfnFileName_       = "WFN";
    restartIdxFileName_       = "IDX";
    eigValFileName_           = "EIG";
  }

  // Density
  {
    Hamiltonian& ham = eigSolPtr_->Ham();
    DblNumMat&  density = ham.Density();

    if( esdfParam.isRestartDensity ){
      std::istringstream rhoStream;      
      SharedRead( restartDensityFileName_, rhoStream);
      // Read the grid
      std::vector<DblNumVec> gridpos(DIM);
      for( Int d = 0; d < DIM; d++ ){
        deserialize( gridpos[d], rhoStream, NO_MASK );
      }
      DblNumVec densityVec;
      // only for restricted spin case
      deserialize( densityVec, rhoStream, NO_MASK );    
      blas::Copy( densityVec.m(), densityVec.Data(), 1, 
          density.VecData(RHO), 1 );
      statusOFS << "Density restarted from file " 
        << restartDensityFileName_ << std::endl;
    } // else using the zero initial guess
    else{
      if( esdfParam.isUseAtomDensity ){

        ham.CalculateAtomDensity( *ptablePtr_, eigSolPtr_->FFT() );
        // Use the superposition of atomic density as the initial guess for density
        const Domain& dm = esdfParam.domain;
        Int ntotFine = dm.NumGridTotalFine();
        Int numDensityComponent = dm.numSpinComponent;

        SetValue( density, 0.0 );
        blas::Copy( ntotFine*numDensityComponent, ham.AtomDensity().Data(), 1, 
            density.Data(), 1 );
        if( numDensityComponent >= 2 ){
          DblNumMat& spindensity = ham.SpinDensity();

          if( numDensityComponent == 2 ){
            blas::Copy( ntotFine, density.VecData(0), 1, spindensity.VecData(0), 1 );
            blas::Copy( ntotFine, density.VecData(0), 1, spindensity.VecData(1), 1 );

            // rhoup = (arho + drho)/2
            // rhodw = (arho - drho)/2
            blas::Axpy( ntotFine, 1.0, density.VecData(1), 1, spindensity.VecData(0), 1);
            blas::Axpy( ntotFine, -1.0, density.VecData(1), 1, spindensity.VecData(1), 1);
            blas::Scal( ntotFine*numDensityComponent, 0.5, spindensity.Data(), 1 );
          }
          else{
            Real amag, segniTemp;
            DblNumVec& segni = ham.Segni();
            Point3 magTemp;
            Point3& spinaxis = ham.SpinAxis();
            bool isParallel = esdfParam.isParallel;

            for( Int i = 0; i < ntotFine; i++){

              magTemp = Point3(  density(i,1), density(i,2), density(i,3) );

              if( isParallel ){
                segniTemp = signx( magTemp[0]*spinaxis[0] + magTemp[1]*spinaxis[1] 
                    + magTemp[2]*spinaxis[2] );
              }
              else{
                segniTemp = 1.0;
              }

              amag = magTemp.l2();
              spindensity(i,UP) = 0.5 * ( density(i,RHO) + segniTemp*amag );
              spindensity(i,DN) = 0.5 * ( density(i,RHO) - segniTemp*amag );
              segni[i] = segniTemp;
            }
          }
        }  // ---- end of if( numDensityComponent >= 2 ) ----
      }  // ---- if( esdfParam.isUseAtomDensity ) ----
      else{
        // Start from pseudocharge, usually this is not a very good idea
        // make sure the pseudocharge is initialized
        DblNumVec&  pseudoCharge = ham.PseudoCharge();
        const Domain& dm = esdfParam.domain;

        SetValue( density, 0.0 );

        Int ntotFine = dm.NumGridTotalFine();

        Real sum0 = 0.0, sum1 = 0.0;
        Real EPS = 1e-6;

        // make sure that the electron density is positive
        for (Int i=0; i<ntotFine; i++){
          density(i, RHO) = ( pseudoCharge(i) > EPS ) ? pseudoCharge(i) : EPS;
          //                density(i, RHO) = pseudoCharge(i);
          sum0 += density(i, RHO);
          sum1 += pseudoCharge(i);
        }

        Print( statusOFS, "Initial density. Sum of density      = ", 
            sum0 * dm.Volume() / dm.NumGridTotalFine() );

        // Rescale the density
        for (int i=0; i <ntotFine; i++){
          density(i, RHO) *= sum1 / sum0;
        } 

        Print( statusOFS, "Rescaled density. Sum of density      = ", 
            sum1 * dm.Volume() / dm.NumGridTotalFine() );
      }
    }
  }

  if( !esdfParam.isRestartWfn ) {
    // Randomized input from outside
    // Setup the occupation rate by aufbau principle (needed for hybrid functional calculation)
    Int nocc = eigSolPtr_->Ham().NumOccupiedState();
#ifdef _COMPLEX_
    std::vector<DblNumVec>& occ = eigSolPtr_->Ham().OccupationRate();
    for( Int i = 0; i < occ.size(); i++ ){
      Int npsi = eigSolPtr_->Psi(i).NumStateTotal();
      Int ik   = eigSolPtr_->Psi(i).Ik();
      Real weight = esdfParam.domain.weight[ik];
      occ[i].Resize( npsi );
      
      SetValue( occ[i] , 0.0 );
      for( Int k = 0; k < nocc; k++ ){
        occ[i][k] = 1.0 * weight;
      }
    }
#else
    DblNumVec& occ = eigSolPtr_->Ham().OccupationRate();
    Int npsi = eigSolPtr_->Psi().NumStateTotal();
    occ.Resize( npsi );
    SetValue( occ, 0.0 );
    for( Int k = 0; k < nocc; k++ ){
      occ[k] = 1.0;
    }
#endif
  }
  else {
    std::istringstream wfnStream;
    SeparateRead( restartWfnFileName_, wfnStream, mpirank );
#ifdef _COMPLEX_
    const Domain& dm =  eigSolPtr_->FFT().domain;
    Int nkLocal = dm.KpointIdx.Size();
    Int nspin = dm.numSpinComponent;
    
    if( nspin == 1 || nspin == 4 ){
      for( Int k = 0; k < nkLocal; k++ ){
        deserialize( eigSolPtr_->Psi(k).Wavefun(), wfnStream, NO_MASK );
        deserialize( eigSolPtr_->Ham().OccupationRate(k), wfnStream, NO_MASK );
      }
    }
    else{
      for( Int k = 0; k < nkLocal; k++ ){
        CpxNumTns& Psitemp = eigSolPtr_->Psi(k).Wavefun();
        Int npw = Psitemp.m();
        Int nbandLocal = Psitemp.p() / 2;

        CpxNumTns Uppsi = CpxNumTns( npw, 1, nbandLocal, false,
            Psitemp.MatData(0) );
        CpxNumTns Dnpsi = CpxNumTns( npw, 1, nbandLocal, false,
            Psitemp.MatData(nbandLocal) );
        deserialize( Uppsi, wfnStream, NO_MASK );
        deserialize( Dnpsi, wfnStream, NO_MASK );
        deserialize( eigSolPtr_->Ham().OccupationRate(k), wfnStream, NO_MASK );     
      }
    }
#else
    deserialize( eigSolPtr_->Psi().Wavefun(), wfnStream, NO_MASK );
    deserialize( eigSolPtr_->Ham().OccupationRate(), wfnStream, NO_MASK );
#endif
    statusOFS << "Wavefunction restarted from file "
      << restartWfnFileName_ << std::endl;
  }

  // XC functional
  {
    isCalculateGradRho_ = false;
    if( esdfParam.XCType == "XC_GGA_XC_PBE" || 
        esdfParam.XCType == "XC_HYB_GGA_XC_HSE06" ||
        esdfParam.XCType == "XC_HYB_GGA_XC_PBEH" ) {
      isCalculateGradRho_ = true;
    }
  }

  return ;
}         // -----  end of method SCF::Setup  ----- 

void
SCF::Update    ( )
{
  Int ntotFine  = eigSolPtr_->FFT().domain.NumGridTotalFine();
  Int nspin = esdfParam.domain.numSpinComponent;

  vtotNew_.Resize(ntotFine,nspin); SetValue(vtotNew_, 0.0);
  densityOld_.Resize(ntotFine, nspin); SetValue(densityOld_, 0.0);

  if( mixType_ == "anderson"|| mixType_ == "kerker+anderson" ){
    dfMat_.Resize( ntotFine, nspin, mixMaxDim_ ); SetValue( dfMat_, D_ZERO );
    dvMat_.Resize( ntotFine, nspin, mixMaxDim_ ); SetValue( dvMat_, D_ZERO );
  }
  else if( mixType_ == "broyden"){
    Int ntot = eigSolPtr_->FFT().idxFineCutDensity.Size();
   
    GdfMat_.Resize( ntot, nspin, mixMaxDim_ ); SetValue( GdfMat_, Z_ZERO );
    GdvMat_.Resize( ntot, nspin, mixMaxDim_ ); SetValue( GdvMat_, Z_ZERO );
    GcdfMat_.Resize( ntot, nspin, 2 ); SetValue( GcdfMat_, Z_ZERO );
  }
  else{
    ErrorHandling("Invalid mixing type.");
  }

  return ;
}         // -----  end of method SCF::Update  ----- 

void
SCF::Iterate (  )
{
  int mpirank;  MPI_Comm_rank(eigSolPtr_->FFT().domain.comm, &mpirank);
  int mpisize;  MPI_Comm_size(eigSolPtr_->FFT().domain.comm, &mpisize);

  Real timeSta, timeEnd;
  Real timeIterStart(0), timeIterEnd(0);

  // Only works for KohnSham class
  Hamiltonian& ham = eigSolPtr_->Ham();
  Fourier&     fft = eigSolPtr_->FFT();
#ifdef _COMPLEX_
  std::vector<Spinor>& psi = eigSolPtr_->Psi();
#else
  Spinor&              psi = eigSolPtr_->Psi();
#endif

  // Compute the kinetic energy and precondition
  ham.CalculateEkin( fft );

  // Compute the exchange-correlation potential and energy
  if( isCalculateGradRho_ ){
    ham.CalculateGradDensity( fft );
  }

  // Compute the Hartree energy
  ham.CalculateXC( Exc_, fft ); 
  ham.CalculateHartree( fft );

  // Compute the total potential
  ham.CalculateVtot( ham.Vtot() );

#ifdef GPU
  cuda_init_vtot();
  cublas::Init();
#endif

  // Perform non-hybrid functional calculation first
  if( !ham.IsHybrid() || !ham.IsEXXActive() ){
    std::ostringstream msg;
    msg << "Starting regular SCF iteration.";
    PrintBlock( statusOFS, msg.str() );
    bool isSCFConverged = false;

    if( !ham.IsEXXActive() && ham.IsHybrid() ) {
      ham.Setup_XC( "XC_GGA_XC_PBE");

      statusOFS << "Re-calculate XC " << std::endl;
      if( isCalculateGradRho_ ){
        ham.CalculateGradDensity( fft );
      }
      ham.CalculateXC( Exc_, fft ); 
      ham.CalculateHartree( fft );
      // Compute the total potential
      ham.CalculateVtot( ham.Vtot() );
    }
    
    for (Int iter=1; iter <= scfMaxIter_; iter++) {
      if ( isSCFConverged ) break;
      // *********************************************************************
      // Performing each iteartion
      // *********************************************************************
      {
        std::ostringstream msg;
        msg << "SCF iteration # " << iter;
        PrintBlock( statusOFS, msg.str() );
      }

      GetTime( timeIterStart );

      // Solve eigenvalue problem
      // Update density, gradDensity, potential (stored in vtotNew_)
      InnerSolve( iter );

      // Use potential as convergence criteria when potential is mixed
#if 0
      if( mixVariable_ == "potential" ){

        DblNumMat& vtotOld_ = ham.Vtot();
        Int ntot = vtotOld_.m();
        Int nspin = vtotOld_.n();

        DblNumMat dVtot( ntot, nspin );
        blas::Copy( ntot * nspin, vtotNew_.Data(), 1, dVtot.Data(), 1 );
        blas::Axpy( ntot * nspin, -1.0, vtotOld_.Data(), 1, dVtot.Data(), 1 );

        scfNorm_ = this->CalculateError( dVtot );
      }
#endif

      CalculateEnergy();

      PrintState( iter );
      
      Int numAtom = ham.AtomList().size();
      efreeDifPerAtom_ = std::abs(Efree_ - EfreeHarris_) / numAtom;


      Print(statusOFS, "norm(out-in)/norm(in) = ", scfNorm_ );
      Print(statusOFS, "Efree diff per atom   = ", efreeDifPerAtom_ ); 

      if( scfNorm_ < scfTolerance_ ){
        /* converged */
        statusOFS << "SCF is converged in " << iter << " steps !" << std::endl;
        isSCFConverged = true;
      }
 
      if ( mixVariable_ == "density" ){
        ham.Vtot() = vtotNew_;
      }

      // Potential mixing
      if( !isSCFConverged && mixVariable_ == "potential" ){
        DblNumMat& vtotOld_ = ham.Vtot();
        if( mixType_ == "anderson" || mixType_ == "kerker+anderson" ){
            Int ntot = vtotOld_.m();
            Int nspin = vtotOld_.n();
            if( nspin == 1 || nspin == 4 ){
              andersonMix(
                  iter,
                  mixStepLength_,
                  mixType_,
                  ham.Vtot(),
                  vtotOld_,
                  vtotNew_,
                  dvMat_,
                  dfMat_ );
            }   
            else if( nspin == 2 ){ 
              // Transform up-down potentials to total-diff potentials for mixing
              DblNumMat vtotOld_alldf( ntot, 2 );
              blas::Copy( ntot, vtotOld_.VecData(UP), 1, vtotOld_alldf.VecData(UP), 1 );
              blas::Copy( ntot, vtotOld_.VecData(UP), 1, vtotOld_alldf.VecData(DN), 1 );
              blas::Axpy( ntot, 1.0, vtotOld_.VecData(DN), 1, vtotOld_alldf.VecData(UP), 1 );
              blas::Axpy( ntot, -1.0, vtotOld_.VecData(DN), 1, vtotOld_alldf.VecData(DN), 1 );

              DblNumMat vtotNew_alldf( ntot, 2 );
              blas::Copy( ntot, vtotNew_.VecData(UP), 1, vtotNew_alldf.VecData(UP), 1 );
              blas::Copy( ntot, vtotNew_.VecData(UP), 1, vtotNew_alldf.VecData(DN), 1 );
              blas::Axpy( ntot, 1.0, vtotNew_.VecData(DN), 1, vtotNew_alldf.VecData(UP), 1 );
              blas::Axpy( ntot, -1.0, vtotNew_.VecData(DN), 1, vtotNew_alldf.VecData(DN), 1 );

              DblNumMat vtotMix_alldf( ntot, 2 );
              DblNumMat& vtotMix = ham.Vtot();
              andersonMix(
                  iter,
                  mixStepLength_,
                  mixType_,
                  vtotMix_alldf,
                  vtotOld_alldf,
                  vtotNew_alldf,
                  dvMat_,
                  dfMat_ );
              // Transform mixed potentials to up-down format
              blas::Copy( ntot, vtotMix_alldf.VecData(UP), 1, vtotMix.VecData(UP), 1 );
              blas::Copy( ntot, vtotMix_alldf.VecData(UP), 1, vtotMix.VecData(DN), 1 );
              blas::Axpy( ntot, 1.0, vtotMix_alldf.VecData(DN), 1, vtotMix.VecData(UP), 1 );
              blas::Axpy( ntot, -1.0, vtotMix_alldf.VecData(DN), 1, vtotMix.VecData(DN), 1 );
              blas::Scal( ntot*2, 0.5, vtotMix.Data(), 1 );
            }
        }
        else if( mixType_ == "broyden"){
          Int ntot = vtotOld_.m();
          Int nspin = vtotOld_.n();
          if( nspin == 1 || nspin == 4 ){
            BroydenMix(
                iter,
                mixStepLength_,
                mixType_,
                ham.Vtot(),
                vtotOld_,
                vtotNew_,
                GdfMat_,
                GdvMat_,
                GcdfMat_);
          }   
          else if( nspin == 2 ){ 
            // Transform up-down potentials to total-diff potentials for mixing
            DblNumMat vtotOld_alldf( ntot, 2 );
            blas::Copy( ntot, vtotOld_.VecData(UP), 1, vtotOld_alldf.VecData(UP), 1 );
            blas::Copy( ntot, vtotOld_.VecData(UP), 1, vtotOld_alldf.VecData(DN), 1 );
            blas::Axpy( ntot, 1.0, vtotOld_.VecData(DN), 1, vtotOld_alldf.VecData(UP), 1 );
            blas::Axpy( ntot, -1.0, vtotOld_.VecData(DN), 1, vtotOld_alldf.VecData(DN), 1 );

            DblNumMat vtotNew_alldf( ntot, 2 );
            blas::Copy( ntot, vtotNew_.VecData(UP), 1, vtotNew_alldf.VecData(UP), 1 );
            blas::Copy( ntot, vtotNew_.VecData(UP), 1, vtotNew_alldf.VecData(DN), 1 );
            blas::Axpy( ntot, 1.0, vtotNew_.VecData(DN), 1, vtotNew_alldf.VecData(UP), 1 );
            blas::Axpy( ntot, -1.0, vtotNew_.VecData(DN), 1, vtotNew_alldf.VecData(DN), 1 );

            DblNumMat vtotMix_alldf( ntot, 2 );
            DblNumMat& vtotMix = ham.Vtot();
            BroydenMix(
                iter,
                mixStepLength_,
                mixType_,
                vtotMix_alldf,
                vtotOld_alldf,
                vtotNew_alldf,
                GdfMat_,
                GdvMat_,
                GcdfMat_);
            // Transform mixed potentials to up-down format
            blas::Copy( ntot, vtotMix_alldf.VecData(UP), 1, vtotMix.VecData(UP), 1 );
            blas::Copy( ntot, vtotMix_alldf.VecData(UP), 1, vtotMix.VecData(DN), 1 );
            blas::Axpy( ntot, 1.0, vtotMix_alldf.VecData(DN), 1, vtotMix.VecData(UP), 1 );
            blas::Axpy( ntot, -1.0, vtotMix_alldf.VecData(DN), 1, vtotMix.VecData(DN), 1 );
            blas::Scal( ntot*2, 0.5, vtotMix.Data(), 1 );
          }
        }
        else{
          ErrorHandling("Invalid mixing type.");
        }
      }  // ---- end of if( !isSCFConverged && mixVariable_ == "potential" ) ----

      GetTime( timeIterEnd );
      statusOFS << "Total wall clock time for this SCF iteration = " << timeIterEnd - timeIterStart
         << " [s]" << std::endl;

    }  // ---- for (iter) ----
#if 0    
    // Output info about calculation time for local or semi-local functional calculations
    statusOFS << "*********************************************************************" << std::endl;
    statusOFS << "Time for non-hybrid calculations" << std::endl;
    statusOFS << "*********************************************************************" << std::endl;
    statusOFS << "MultSpinor:" << std::endl;
    statusOFS << "Time for iterPhase        = " << iterPhase_          << "  timePhase        = " << timePhase_ << std::endl;
    statusOFS << "Time for iterFFT          = " << iterFFT_            << "  timeFFT          = " << timeFFT_ << std::endl;
    statusOFS << "Time for iterNonlocal     = " << iterNonlocal_       << "  timeNonlocal     = " << timeNonlocal_ << std::endl;
    statusOFS << "Time for iterEkin         = " << iterEkin_           << "  timeEkin         = " << timeEkin_ << std::endl;
    statusOFS << "Time for iterAssign       = " << iterAssign_         << "  timeAssign       = " << timeAssign_ << std::endl;
    statusOFS << "Time for MultSpinor       = " << iterMultSpinor_     << "  timeMultSpinor   = " << timeMultSpinor_ << std::endl << std::endl;    

    statusOFS << "EigenSolver:" << std::endl;
    statusOFS << "Time for iterGemmT        = " << iterGemmT_          << "  timeGemmT        = " << timeGemmT_ << std::endl;
    statusOFS << "Time for iterGemmN        = " << iterGemmN_          << "  timeGemmN        = " << timeGemmN_ << std::endl;
    statusOFS << "Time for iterBcast        = " << iterBcast_          << "  timeBcast        = " << timeBcast_ << std::endl;
    statusOFS << "Time for iterAllreduce    = " << iterAllreduce_      << "  timeAllreduce    = " << timeAllreduce_ << std::endl;
    statusOFS << "Time for iterAlltoallv    = " << iterAlltoallv_      << "  timeAlltoallv    = " << timeAlltoallv_ << std::endl;
    statusOFS << "Time for iterAlltoallvMap = " << iterAlltoallvMap_   << "  timeAlltoallvMap = " << timeAlltoallvMap_ << std::endl;
    statusOFS << "Time for iterSpinor       = " << iterSpinor_         << "  timeSpinor       = " << timeSpinor_ << std::endl;
    statusOFS << "Time for iterTrsm         = " << iterTrsm_           << "  timeTrsm         = " << timeTrsm_ << std::endl;
    statusOFS << "Time for iterPotrf        = " << iterPotrf_          << "  timePotrf        = " << timePotrf_ << std::endl;
    statusOFS << "Time for iterSyevd        = " << iterSyevd_          << "  timeSyevd        = " << timeSyevd_ << std::endl;
    statusOFS << "Time for iterSygvd        = " << iterSygvd_          << "  timeSygvd        = " << timeSygvd_ << std::endl;
    statusOFS << "Time for iterMpirank0     = " << iterMpirank0_       << "  timeMpirank0     = " << timeMpirank0_ << std::endl;
    statusOFS << "Time for iterTrtri        = " << iterTrtri_          << "  timeTrtri        = " << timeTrtri_ << std::endl;
    statusOFS << "Time for iterCopy         = " << iterCopy_           << "  timeCopy         = " << timeCopy_ << std::endl;
    statusOFS << "Time for iterOther        = " << iterOther_          << "  timeOther        = " << timeOther_ << std::endl;
    statusOFS << "Time for EigenSolver in PWDFT is " <<  timeEigenSol_  << std::endl << std::endl;
#endif
  }  // ---- end of if( !ham.IsHybrid() || !ham.IsEXXActive()) ----

  // The following is the hybrid functional part
  // NOTE: The different mixing mode of hybrid functional calculations 
  // are not compatible with each other. So each requires its own code
  if( ham.IsHybrid() ){
    {
      ham.Setup_XC( "XC_HYB_GGA_XC_HSE06");
      statusOFS << "Re-calculate XC " << std::endl;

      if( isCalculateGradRho_ ){
        ham.CalculateGradDensity( fft );
      }
      ham.CalculateXC( Exc_, fft );
      ham.CalculateHartree( fft );
      // Compute the total potential
      ham.CalculateVtot( ham.Vtot() );
    }

    // Fock energies
    Real fock0 = 0.0, fock1 = 0.0, fock2 = 0.0;

    // EXX: Run SCF::Iterate here
    bool isPhiIterConverged = false;

    bool isFixColumnDF = false;
    Real timePhiIterStart(0), timePhiIterEnd(0);
    Real dExx;

    if( ham.IsEXXActive() == false )
      ham.SetEXXActive(true);
    
    // Evaluate the Fock energy
    // Update Phi <- Psi
    GetTime( timeSta );
    ham.SetPhiEXX( psi, fft );

    // Update the ACE if needed
    if( esdfParam.isHybridACE ){
      if( esdfParam.isHybridDF ){
        ham.CalculateVexxACEDF ( psi, fft, isFixColumnDF );
        // Fix the column after the first iteraiton 
        isFixColumnDF = true;
      }
      else{
#ifdef GPU
        ham.CalculateVexxACEGPU ( psi, fft );
#else
        ham.CalculateVexxACE ( psi, fft );
#endif
      }
    }
        
    GetTime( timeEnd );
    //statusOFS << "Time for updating Phi related variable is " <<
      //timeEnd - timeSta << " [s]" << std::endl << std::endl;

    GetTime( timeSta );
    fock0 = ham.CalculateEXXEnergy( psi, fft );
    GetTime( timeEnd );

    statusOFS << "fock0 = " << fock0 << std::endl;

    //statusOFS << "Time for computing the EXX energy is " <<
      //timeEnd - timeSta << " [s]" << std::endl << std::endl;
  
    GetTime( timeSta );
    
    if( esdfParam.hybridMixType == "nested" ){

      for( Int phiIter = 1; phiIter <= scfPhiMaxIter_; phiIter++ ){

        GetTime( timePhiIterStart );

        std::ostringstream msg;
        msg << "Phi iteration # " << phiIter;
        PrintBlock( statusOFS, msg.str() );

        // Nested SCF iteration
        bool isSCFConverged = false;
        for (Int iter=1; iter <= scfMaxIter_; iter++) {
          if ( isSCFConverged ) break;
          // *********************************************************************
          // Performing each iteartion
          // *********************************************************************
          {
            std::ostringstream msg;
            msg << "SCF iteration # " << iter;
            PrintBlock( statusOFS, msg.str() );
          }

          GetTime( timeIterStart );

          // Solve eigenvalue problem
          // Update density, gradDensity, potential (stored in vtotNew_)
          InnerSolve( iter );

          // Use potential as convergence criteria when potential is mixed
          if( mixVariable_ == "potential" ){

            DblNumMat& vtotOld_ = ham.Vtot();
            Int ntot = vtotOld_.m();
            Int nspin = vtotOld_.n();

            DblNumMat dVtot( ntot, nspin );
            blas::Copy( ntot * nspin, vtotNew_.Data(), 1, dVtot.Data(), 1 );
            blas::Axpy( ntot * nspin, -1.0, vtotOld_.Data(), 1, dVtot.Data(), 1 );

            scfNorm_ = this->CalculateError( dVtot );
          }

          GetTime( timeSta );
          CalculateEnergy();
          GetTime( timeEnd );

          PrintState( iter );
          
          Int numAtom = ham.AtomList().size();
          efreeDifPerAtom_ = std::abs(Efree_ - EfreeHarris_) / numAtom;

          Print(statusOFS, "norm(out-in)/norm(in) = ", scfNorm_ );
          Print(statusOFS, "Efree diff per atom   = ", efreeDifPerAtom_ ); 

          if( scfNorm_ < scfTolerance_ ){
            /* converged */
            statusOFS << "SCF is converged in " << iter << " steps !" << std::endl;
            isSCFConverged = true;
          }

          if ( mixVariable_ == "density" ){
            ham.Vtot() = vtotNew_;
          }

          // Potential mixing         
          if( !isSCFConverged && mixVariable_ == "potential" ){
            DblNumMat& vtotOld_ = ham.Vtot();

            if( mixType_ == "anderson" || mixType_ == "kerker+anderson" ){
              Int ntot = vtotOld_.m();
              Int nspin = vtotOld_.n();
              if( nspin == 1 || nspin == 4 ){
                andersonMix(
                    iter,
                    mixStepLength_,
                    mixType_,
                    ham.Vtot(),
                    vtotOld_,
                    vtotNew_,
                    dvMat_,
                    dfMat_ );
              }   
              else if( nspin == 2 ){ 
                // Transform up-down potentials to total-diff potentials for mixing
                DblNumMat vtotOld_alldf( ntot, 2 );
                blas::Copy( ntot, vtotOld_.VecData(UP), 1, vtotOld_alldf.VecData(UP), 1 );
                blas::Copy( ntot, vtotOld_.VecData(UP), 1, vtotOld_alldf.VecData(DN), 1 );
                blas::Axpy( ntot, 1.0, vtotOld_.VecData(DN), 1, vtotOld_alldf.VecData(UP), 1 );
                blas::Axpy( ntot, -1.0, vtotOld_.VecData(DN), 1, vtotOld_alldf.VecData(DN), 1 );

                DblNumMat vtotNew_alldf( ntot, 2 );
                blas::Copy( ntot, vtotNew_.VecData(UP), 1, vtotNew_alldf.VecData(UP), 1 );
                blas::Copy( ntot, vtotNew_.VecData(UP), 1, vtotNew_alldf.VecData(DN), 1 );
                blas::Axpy( ntot, 1.0, vtotNew_.VecData(DN), 1, vtotNew_alldf.VecData(UP), 1 );
                blas::Axpy( ntot, -1.0, vtotNew_.VecData(DN), 1, vtotNew_alldf.VecData(DN), 1 );

                DblNumMat vtotMix_alldf( ntot, 2 );
                DblNumMat& vtotMix = ham.Vtot();
                andersonMix(
                    iter,
                    mixStepLength_,
                    mixType_,
                    vtotMix_alldf,
                    vtotOld_alldf,
                    vtotNew_alldf,
                    dvMat_,
                    dfMat_ );
                // Transform mixed potentials to up-down format
                blas::Copy( ntot, vtotMix_alldf.VecData(UP), 1, vtotMix.VecData(UP), 1 );
                blas::Copy( ntot, vtotMix_alldf.VecData(UP), 1, vtotMix.VecData(DN), 1 );
                blas::Axpy( ntot, 1.0, vtotMix_alldf.VecData(DN), 1, vtotMix.VecData(UP), 1 );
                blas::Axpy( ntot, -1.0, vtotMix_alldf.VecData(DN), 1, vtotMix.VecData(DN), 1 );
                blas::Scal( ntot*2, 0.5, vtotMix.Data(), 1 );
              }
            }
            else if( mixType_ == "broyden"){
              Int ntot = vtotOld_.m();
              Int nspin = vtotOld_.n();
              if( nspin == 1 || nspin == 4 ){
                BroydenMix(
                    iter,
                    mixStepLength_,
                    mixType_,
                    ham.Vtot(),
                    vtotOld_,
                    vtotNew_,
                    GdfMat_,
                    GdvMat_,
                    GcdfMat_);
              }   
              else if( nspin == 2 ){ 
                // Transform up-down potentials to total-diff potentials for mixing
                DblNumMat vtotOld_alldf( ntot, 2 );
                blas::Copy( ntot, vtotOld_.VecData(UP), 1, vtotOld_alldf.VecData(UP), 1 );
                blas::Copy( ntot, vtotOld_.VecData(UP), 1, vtotOld_alldf.VecData(DN), 1 );
                blas::Axpy( ntot, 1.0, vtotOld_.VecData(DN), 1, vtotOld_alldf.VecData(UP), 1 );
                blas::Axpy( ntot, -1.0, vtotOld_.VecData(DN), 1, vtotOld_alldf.VecData(DN), 1 );

                DblNumMat vtotNew_alldf( ntot, 2 );
                blas::Copy( ntot, vtotNew_.VecData(UP), 1, vtotNew_alldf.VecData(UP), 1 );
                blas::Copy( ntot, vtotNew_.VecData(UP), 1, vtotNew_alldf.VecData(DN), 1 );
                blas::Axpy( ntot, 1.0, vtotNew_.VecData(DN), 1, vtotNew_alldf.VecData(UP), 1 );
                blas::Axpy( ntot, -1.0, vtotNew_.VecData(DN), 1, vtotNew_alldf.VecData(DN), 1 );

                DblNumMat vtotMix_alldf( ntot, 2 );
                DblNumMat& vtotMix = ham.Vtot();
                BroydenMix(
                    iter,
                    mixStepLength_,
                    mixType_,
                    vtotMix_alldf,
                    vtotOld_alldf,
                    vtotNew_alldf,
                    GdfMat_,
                    GdvMat_,
                    GcdfMat_);
                // Transform mixed potentials to up-down format
                blas::Copy( ntot, vtotMix_alldf.VecData(UP), 1, vtotMix.VecData(UP), 1 );
                blas::Copy( ntot, vtotMix_alldf.VecData(UP), 1, vtotMix.VecData(DN), 1 );
                blas::Axpy( ntot, 1.0, vtotMix_alldf.VecData(DN), 1, vtotMix.VecData(UP), 1 );
                blas::Axpy( ntot, -1.0, vtotMix_alldf.VecData(DN), 1, vtotMix.VecData(DN), 1 );
                blas::Scal( ntot*2, 0.5, vtotMix.Data(), 1 );
              }
            }
            else{
              ErrorHandling("Invalid mixing type.");
            }
          }  // ---- end of if( !isSCFConverged && mixVariable_ == "potential" ) ----

          GetTime( timeIterEnd );

          statusOFS << "Total wall clock time for this SCF iteration = " << timeIterEnd - timeIterStart
            << " [s]" << std::endl;
        }  // for (iter)

        GetTime( timePhiIterEnd );

        statusOFS << "Total wall clock time for this Phi iteration = " << 
          timePhiIterEnd - timePhiIterStart << " [s]" << std::endl;

        fock1 = ham.CalculateEXXEnergy( psi, fft );
      
        // Update Phi <- Psi
        GetTime( timeSta );
        ham.SetPhiEXX( psi, fft ); 

        // Update the ACE if needed
        if( esdfParam.isHybridACE ){
          if( esdfParam.isHybridDF ){
            ham.CalculateVexxACEDF ( psi, fft, isFixColumnDF );  
          }
          else{
            ham.CalculateVexxACE ( psi, fft );
          }
        }

        GetTime( timeEnd );
        statusOFS << "Time for updating Phi related variable is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;

        GetTime( timeSta );
        fock2 = ham.CalculateEXXEnergy( psi, fft ); 
        GetTime( timeEnd );
        statusOFS << "Time for computing the EXX energy is " <<
          timeEnd - timeSta << " [s]" << std::endl << std::endl;

        // Note: initially fock1 = 0.0. So it should at least run for 1 iteration.
        //dExx = std::abs(fock2 - fock1) / std::abs(fock2);
        dExx = 4.0 * std::abs( fock1 - 0.5 * (fock0 + fock2) );

        fock0 = fock2;
        Efock_ = fock2;

        Etot_ = Etot_ - 2.0 * fock1 + Efock_;
        Efree_ = Efree_ - 2.0 * fock1 + Efock_;

        statusOFS << std::endl;
        Print(statusOFS, "Fock energy       = ",  Efock_, "[au]");
        Print(statusOFS, "Etot(with fock)   = ",  Etot_, "[au]");
        Print(statusOFS, "Efree(with fock)  = ",  Efree_, "[au]");
        Print(statusOFS, "dExx              = ",  dExx, "[au]");
        if( dExx < scfPhiTolerance_ ){
          statusOFS << "SCF for hybrid functional is converged in " 
            << phiIter << " steps !" << std::endl;
          isPhiIterConverged = true;
        }
        if ( isPhiIterConverged ) break;
      } // for(phiIter)
    } // hybridMixType == "nested"

    GetTime( timeEnd );
    statusOFS << "Time for using nested method is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;
  }  // ---- end of if( ham.IsHybrid() ) ---

#ifdef GPU
    cublas::Destroy();
    cuda_clean_vtot();
#endif

#ifdef _COMPLEX_
  //ham.CalculateForce( *ptablePtr_, psi, fft );
#else
  // FIXME
  //ham.CalculateForce( psi, fft );
#endif

  // Output the eigenvalues 
#ifdef _COMPLEX_
  if( esdfParam.isOutputEigvals ){
    statusOFS << "Output the eigenvalues." << std::endl;
    Domain& dm = fft.domain;
    MPI_Barrier(dm.comm);
    Int mpirank;  MPI_Comm_rank(dm.comm, &mpirank);
    Int colrank;  MPI_Comm_rank(dm.colComm_kpoint, &colrank);
    Int rowsize;  MPI_Comm_size(dm.rowComm_kpoint, &rowsize);
    // Global variables 
    if( colrank == 0 ){
      Int nkTotal = dm.NumKGridTotal();
      Int nbTotal = psi[0].NumStateTotal();
      DblNumMat eigValS( nbTotal, nkTotal );

      Int nkLocal = psi.size();
      DblNumMat eigValSLocal( nbTotal, nkLocal );
      for( Int k = 0; k < nkLocal; k++ ){
        blas::Copy( nbTotal, eigSolPtr_->EigVal(k).Data(), 1, eigValSLocal.VecData(k), 1 );
      }

      IntNumVec localSize( rowsize );
      IntNumVec localDispls( rowsize );
      SetValue( localSize, 0 );
      SetValue( localDispls, 0 );

      Int numEig = nbTotal * nkLocal;
      MPI_Allgather( &numEig, 1, MPI_INT, localSize.Data(), 1, MPI_INT, dm.rowComm_kpoint );

      for( Int i = 1; i < rowsize; i++ ){
        localDispls[i] = localDispls[i-1] + localSize[i-1];
      }

      MPI_Allgatherv( eigValSLocal.Data(), numEig, MPI_DOUBLE, eigValS.Data(),
          localSize.Data(), localDispls.Data(), MPI_DOUBLE, dm.rowComm_kpoint );

      if( mpirank == 0 ){
        std::ofstream eigStream(eigValFileName_.c_str());
        if( !eigStream.good() ){
          ErrorHandling( "Eigenvalue file cannot be opened." );
        }

        DblNumVec eigVec(eigValS.Size(), false, eigValS.Data());
        serialize( eigVec, eigStream, NO_MASK );
        eigStream.close();
      }
    }  // ---- end of if( colrank == 0 ) ---- 
  }
#endif
  // Output the information after SCF
  {
    // Energy
    Real HOMO, LUMO;
#ifdef _COMPLEX_
    // To be added: HOMO, LUMO and fermi energy for crystals
    HOMO = 0.0;
    LUMO = 0.0;
#else
    HOMO = eigSolPtr_->EigVal()(eigSolPtr_->Ham().NumOccupiedState()-1);
    if( eigSolPtr_->Ham().NumExtraState() > 0 )
      LUMO = eigSolPtr_->EigVal()(eigSolPtr_->Ham().NumOccupiedState());
#endif
    // Print out the energy
    PrintBlock( statusOFS, "Energy" );
    statusOFS 
      << "NOTE:  Ecor  = Exc - EVxc - Ehart - Eself + EIonSR + EVdw + Eext" << std::endl
      << "       Etot  = Ekin + Ecor" << std::endl
      << "       Efree = Etot + Entropy" << std::endl << std::endl;
    Print(statusOFS, "! Etot            = ",  Etot_, "[au]");
    Print(statusOFS, "! Efree           = ",  Efree_, "[au]");
    Print(statusOFS, "! EfreeHarris     = ",  EfreeHarris_, "[au]");
    Print(statusOFS, "! EVdw            = ",  EVdw_, "[au]"); 
    Print(statusOFS, "! Eext            = ",  Eext_, "[au]");
    Print(statusOFS, "! Fermi           = ",  fermi_, "[au]");
    Print(statusOFS, "! HOMO            = ",  HOMO*au2ev, "[ev]");
    if( ham.NumExtraState() > 0 ){
      Print(statusOFS, "! LUMO            = ",  LUMO*au2ev, "[eV]");
    }
  }

  {
    // Print out the force
    PrintBlock( statusOFS, "Atomic Force" );

    Point3 forceCM(0.0, 0.0, 0.0);
    std::vector<Atom>& atomList = ham.AtomList();
    Int numAtom = atomList.size();

    for( Int a = 0; a < numAtom; a++ ){
      Print( statusOFS, "atom", a, "force", atomList[a].force );
      forceCM += atomList[a].force;
    }
    statusOFS << std::endl;
    Print( statusOFS, "force for centroid  : ", forceCM );
    Print( statusOFS, "Max force magnitude : ", MaxForce(atomList) );
    statusOFS << std::endl;
  }

  // Output the structure information
  {
    if( mpirank == 0 ){
      std::ostringstream structStream;
#if ( _DEBUGlevel_ >= 0 )
      statusOFS << std::endl 
        << "Output the structure information" 
        << std::endl;
#endif
      // Domain
      const Domain& dm =  eigSolPtr_->FFT().domain;
      serialize( dm.length, structStream, NO_MASK );
      serialize( dm.numGrid, structStream, NO_MASK );
      serialize( dm.numGridFine, structStream, NO_MASK );
      serialize( dm.posStart, structStream, NO_MASK );

      // Atomic information
      serialize( ham.AtomList(), structStream, NO_MASK );
      std::string structFileName = "STRUCTURE";

      std::ofstream fout(structFileName.c_str());
      if( !fout.good() ){
        std::ostringstream msg;
        msg 
          << "File " << structFileName.c_str() << " cannot be open." 
          << std::endl;
        ErrorHandling( msg.str().c_str() );
      }
      fout << structStream.str();
      fout.close();
    }
  }

  // Output restarting information
  if( esdfParam.isOutputDensity ){
    if( mpirank == 0 ){
      std::ofstream rhoStream(restartDensityFileName_.c_str());
      if( !rhoStream.good() ){
        ErrorHandling( "Density file cannot be opened." );
      }

      const Domain& dm =  eigSolPtr_->FFT().domain;
      std::vector<DblNumVec>   gridpos(DIM);
      UniformMeshFine ( dm, gridpos );
      for( Int d = 0; d < DIM; d++ ){
        serialize( gridpos[d], rhoStream, NO_MASK );
      }

      DblNumMat& densityMat = eigSolPtr_->Ham().Density();
      DblNumVec densityVec(densityMat.Size(), false, densityMat.Data());
      serialize( densityVec, rhoStream, NO_MASK );
      rhoStream.close();
    }
  }    

  // Output the total potential
  if( esdfParam.isOutputPotential ){
    if( mpirank == 0 ){
      std::ofstream vtotStream(restartPotentialFileName_.c_str());
      if( !vtotStream.good() ){
        ErrorHandling( "Potential file cannot be opened." );
      }

      const Domain& dm =  eigSolPtr_->FFT().domain;
      std::vector<DblNumVec>   gridpos(DIM);
      UniformMeshFine ( dm, gridpos );
      for( Int d = 0; d < DIM; d++ ){
        serialize( gridpos[d], vtotStream, NO_MASK );
      }

      serialize( eigSolPtr_->Ham().Vtot(), vtotStream, NO_MASK );
      vtotStream.close();
    }
  }

  if( esdfParam.isOutputWfn ){
    std::ostringstream wfnStream;
    std::ostringstream idxStream;
#ifdef _COMPLEX_
    const Domain& dm =  eigSolPtr_->FFT().domain;
    Int nkLocal = dm.KpointIdx.Size();
    Int nspin = dm.numSpinComponent;

    if( nspin == 1 || nspin == 4 ){
      for( Int k = 0; k < nkLocal; k++ ){
        serialize( eigSolPtr_->Psi(k).Wavefun(), wfnStream, NO_MASK );
        serialize( eigSolPtr_->Ham().OccupationRate(k), wfnStream, NO_MASK );
      }
    }
    else{
      for( Int k = 0; k < nkLocal; k++ ){
        CpxNumTns Psitemp = eigSolPtr_->Psi(k).Wavefun();
        Int npw = Psitemp.m();
        Int nbandLocal = Psitemp.p() / 2;
 
        CpxNumTns Uppsi = CpxNumTns( npw, 1, nbandLocal, false, 
            Psitemp.MatData(0) );
        CpxNumTns Dnpsi = CpxNumTns( npw, 1, nbandLocal, false, 
            Psitemp.MatData(nbandLocal) );
        serialize( Uppsi, wfnStream, NO_MASK );
        serialize( Dnpsi, wfnStream, NO_MASK );
        serialize( eigSolPtr_->Ham().OccupationRate(k), wfnStream, NO_MASK );
      }
    }

    for( Int k = 0; k < nkLocal; k++ ){
      serialize( eigSolPtr_->FFT().idxCoarseCut[k], idxStream, NO_MASK );      
    }
#else
    serialize( eigSolPtr_->Psi().Wavefun(), wfnStream, NO_MASK );
    serialize( eigSolPtr_->Ham().OccupationRate(), wfnStream, NO_MASK );
#endif
    SeparateWrite( restartWfnFileName_, wfnStream, mpirank );
    SeparateWrite( restartIdxFileName_, idxStream, mpirank );
  }   

  return ;
}         // -----  end of method SCF::Iterate  ----- 

#ifdef _COMPLEX_
void
SCF::NonSCF (  )
{
  int mpirank;  MPI_Comm_rank(eigSolPtr_->FFT().domain.comm, &mpirank);
  int mpisize;  MPI_Comm_size(eigSolPtr_->FFT().domain.comm, &mpisize);

  // Only works for KohnSham class
  Hamiltonian& ham = eigSolPtr_->Ham();
  Fourier&     fft = eigSolPtr_->FFT();
  std::vector<Spinor>& psi = eigSolPtr_->Psi();
  // Compute the k-point dependent kinetic energy
  ham.CalculateEkin( fft );

  // Compute the exchange-correlation potential and energy
  if( isCalculateGradRho_ ){
    ham.CalculateGradDensity( fft );
  }

  // Compute the Hartree energy
  ham.CalculateXC( Exc_, fft ); 
  ham.CalculateHartree( fft );

  // Compute the total potential
  ham.CalculateVtot( ham.Vtot() );

  if( ham.IsHybrid() ){
    if( ham.IsEXXActive() == false )
      ham.SetEXXActive(true);

    ham.SetPhiEXX( restartWfnFileName_, fft );
    // Update the ACE if needed
    if( esdfParam.isHybridACE ){
      if( esdfParam.isHybridDF ){
        // FIXME ACE-ISDF
      }
      else{
        ham.CalculateVexxACE ( psi, fft );
      }
    }
  }

  // Solve eigenvalue problem
  InnerSolve( 1 );

  statusOFS << "The hybrid energy band calculation is finished !" << std::endl;

  // Output the eigenvalues 
  statusOFS << "Output the eigenvalues." << std::endl << std::endl;
  Domain& dm = fft.domain;
  MPI_Barrier(dm.comm);
  Int colrank;  MPI_Comm_rank(dm.colComm_kpoint, &colrank);
  Int rowsize;  MPI_Comm_size(dm.rowComm_kpoint, &rowsize);
  // Global variables 
  if( colrank == 0 ){
    Int nkTotal = dm.NumKGridTotal();
    Int nbTotal = psi[0].NumStateTotal();
    DblNumMat eigValS( nbTotal, nkTotal );

    Int nkLocal = psi.size();
    DblNumMat eigValSLocal( nbTotal, nkLocal );
    for( Int k = 0; k < nkLocal; k++ ){
      blas::Copy( nbTotal, eigSolPtr_->EigVal(k).Data(), 1, eigValSLocal.VecData(k), 1 );
    }

    IntNumVec localSize( rowsize );
    IntNumVec localDispls( rowsize );
    SetValue( localSize, 0 );
    SetValue( localDispls, 0 );

    Int numEig = nbTotal * nkLocal;
    MPI_Allgather( &numEig, 1, MPI_INT, localSize.Data(), 1, MPI_INT, dm.rowComm_kpoint );

    for( Int i = 1; i < rowsize; i++ ){
      localDispls[i] = localDispls[i-1] + localSize[i-1];
    }

    MPI_Allgatherv( eigValSLocal.Data(), numEig, MPI_DOUBLE, eigValS.Data(),
        localSize.Data(), localDispls.Data(), MPI_DOUBLE, dm.rowComm_kpoint );

    if( mpirank == 0 ){

      for( Int k = 0; k < nkTotal ; k++ ){
        statusOFS << "#k point " << k << " :" << std::endl;
        for( Int i = 0; i < nbTotal; i++ ){
          Print( statusOFS,
           "band#    = ", i,
           "eigval   = ", eigValS(i,k) );
        }
      }

      std::ofstream eigStream(eigValFileName_.c_str());
      if( !eigStream.good() ){
        ErrorHandling( "Eigenvalue file cannot be opened." );
      }

      DblNumVec eigVec(eigValS.Size(), false, eigValS.Data());
      serialize( eigVec, eigStream, NO_MASK );
      eigStream.close();
    }
  }  // ---- end of if( colrank == 0 ) ---- 
}         // -----  end of method SCF::NonSCF  ----- 
#endif

void
SCF::InnerSolve	( Int iter )
{
  Real timeSta, timeEnd;
  // Only works for KohnSham class
  Hamiltonian& ham = eigSolPtr_->Ham();
  Fourier&     fft = eigSolPtr_->FFT();
#ifdef _COMPLEX_
  std::vector<Spinor>& psi = eigSolPtr_->Psi();
#else
  Spinor&              psi = eigSolPtr_->Psi();
#endif
  // Solve the eigenvalue problem
  Real eigTolNow;
  if( isEigToleranceDynamic_ ){
    // Dynamic strategy to control the tolerance
    if( iter == 1 )
      eigTolNow = 1e-2;
    else
      eigTolNow = std::max( std::min( scfNorm_*1e-4, 1e-2 ) , eigTolerance_ );
  }
  else{
    // Static strategy to control the tolerance
    eigTolNow = eigTolerance_;
  }
#ifdef _COMPLEX_
  Int numEig = (psi[0].NumStateTotal());
  Int nkLocal = psi.size();
#else
  Int numEig = (psi.NumStateTotal());
#endif
  if(Diag_SCF_PWDFT_by_Cheby_ == 0)
  {  
    statusOFS << "The current tolerance used by the eigensolver is " 
      << eigTolNow << std::endl;
    statusOFS << "The target number of converged eigenvectors is " 
      << numEig << std::endl;
  }

  if(Diag_SCF_PWDFT_by_Cheby_ == 1)
  {
    if(Cheby_iondynamics_schedule_flag_ == 0)
    {
      // Use static schedule
      statusOFS << std::endl << " CheFSI in PWDFT working on static schedule." << std::endl;
      // Use CheFSI or LOBPCG on first step 
#if 0
      if(iter <= 1){
        if(First_SCF_PWDFT_ChebyCycleNum_ <= 0)
          eigSolPtr_->LOBPCGSolveReal(numEig, eigMaxIter_, eigMinTolerance_, eigTolNow );    
        else
          eigSolPtr_->FirstChebyStep(numEig, First_SCF_PWDFT_ChebyCycleNum_, First_SCF_PWDFT_ChebyFilterOrder_);
      }
      else{
        eigSolPtr_->GeneralChebyStep(numEig, General_SCF_PWDFT_ChebyFilterOrder_);
      }
#endif
    }
    else
    {
      // Use ion-dynamics schedule
#if 0
      statusOFS << std::endl << " CheFSI in PWDFT working on ion-dynamics schedule." << std::endl;
      if( iter <= 1)
      {
        for (int cheby_iter = 1; cheby_iter <= eigMaxIter_; cheby_iter ++)
          eigSolPtr_->GeneralChebyStep(numEig, General_SCF_PWDFT_ChebyFilterOrder_);
      }
      else
      {
        eigSolPtr_->GeneralChebyStep(numEig, General_SCF_PWDFT_ChebyFilterOrder_);
      }
#endif
    }
  }
  else
  {
    // More iteration steps for diagonalization are needed for first SCF cycle
    // to ensure the convergence rate of SCF
    Int eigMaxIter = eigMaxIter_;
#ifdef _COMPLEX_
    // Use LOBPCG
    if( esdfParam.PWSolver == "LOBPCG" || esdfParam.PWSolver == "LOBPCGScaLAPACK" ){
      if( spinType_ == 1 || spinType_ == 4 ){
        for( Int k = 0 ; k < nkLocal; k++ ){
          eigSolPtr_->SetpsiId( k ); 
          eigSolPtr_->LOBPCGSolveComplex(numEig, iter, eigMaxIter, eigMinTolerance_, eigTolNow );
        }
      }  
      else{
        for( Int k = 0 ; k < nkLocal; k++ ){
          eigSolPtr_->SetpsiId( k );

          ham.SetSpinSwitch(0);
          eigSolPtr_->LOBPCGSolveComplex(numEig/2, iter, eigMaxIter, eigMinTolerance_, eigTolNow );

          ham.SetSpinSwitch(1);
          eigSolPtr_->LOBPCGSolveComplex(numEig/2, iter, eigMaxIter, eigMinTolerance_, eigTolNow );
        }
      } 
    }
    else if( esdfParam.PWSolver == "PPCG" || esdfParam.PWSolver == "PPCGScaLAPACK" ){
      if( spinType_ == 1 || spinType_ == 4 ){
        for( Int k = 0 ; k < nkLocal; k++ ){
          eigSolPtr_->SetpsiId( k );
          eigSolPtr_->PPCGSolveComplex(numEig, iter, eigMaxIter, eigMinTolerance_, eigTolNow );
        }
      }
      else {
        for( Int k = 0 ; k < nkLocal; k++ ){
          eigSolPtr_->SetpsiId( k );

          ham.SetSpinSwitch(0);
          eigSolPtr_->PPCGSolveComplex(numEig/2, iter, eigMaxIter, eigMinTolerance_, eigTolNow );

          ham.SetSpinSwitch(1);
          eigSolPtr_->PPCGSolveComplex(numEig/2, iter, eigMaxIter, eigMinTolerance_, eigTolNow );
        }
      }  
    }
    else if( esdfParam.PWSolver == "Davidson" || esdfParam.PWSolver == "DavidsonScaLAPACK" ){
      if( spinType_ == 1 || spinType_ == 4 ){
        for( Int k = 0 ; k < nkLocal; k++ ){
          eigSolPtr_->SetpsiId( k );
          eigSolPtr_->DavidsonSolveComplex(numEig, iter, 2, eigMaxIter, eigMinTolerance_, eigTolNow );
        }
      }
      else {
        for( Int k = 0 ; k < nkLocal; k++ ){
          eigSolPtr_->SetpsiId( k );

          ham.SetSpinSwitch(0);
          eigSolPtr_->DavidsonSolveComplex(numEig/2, iter, 2, eigMaxIter, eigMinTolerance_, eigTolNow );

          ham.SetSpinSwitch(1);
          eigSolPtr_->DavidsonSolveComplex(numEig/2, iter, 2, eigMaxIter, eigMinTolerance_, eigTolNow );
        }
      }
    }
    else{
      // FIXME Merge the Chebyshev into an option of PWSolver
      ErrorHandling("Not supported PWSolver for complex type.");
    }
#else
    if( esdfParam.PWSolver == "LOBPCG" || esdfParam.PWSolver == "LOBPCGScaLAPACK"){
      if( spinType_ == 1 || spinType_ == 4 ){ 
        eigSolPtr_->LOBPCGSolveReal(numEig, iter, eigMaxIter_, eigMinTolerance_, eigTolNow );    
      }
      else{
        ham.SetSpinSwitch(0);
        eigSolPtr_->LOBPCGSolveReal(numEig/2, iter, eigMaxIter, eigMinTolerance_, eigTolNow );

        ham.SetSpinSwitch(1);
        eigSolPtr_->LOBPCGSolveReal(numEig/2, iter, eigMaxIter, eigMinTolerance_, eigTolNow );
      }
    } // Use PPCG
    else if( esdfParam.PWSolver == "PPCG" || esdfParam.PWSolver == "PPCGScaLAPACK" ){
      if( spinType_ == 1 || spinType_ == 4 ){
#ifdef GPU
        eigSolPtr_->PPCGSolveRealGPU(numEig, iter, eigMaxIter_, eigMinTolerance_, eigTolNow );
#else
        eigSolPtr_->PPCGSolveReal(numEig, iter, eigMaxIter_, eigMinTolerance_, eigTolNow );
#endif
      }
      else{ 
        ham.SetSpinSwitch(0);     
#ifdef GPU
        eigSolPtr_->PPCGSolveRealGPU(numEig/2, iter, eigMaxIter_, eigMinTolerance_, eigTolNow );
#else
        eigSolPtr_->PPCGSolveReal(numEig/2, iter, eigMaxIter_, eigMinTolerance_, eigTolNow );
#endif
        ham.SetSpinSwitch(1);
#ifdef GPU
        eigSolPtr_->PPCGSolveRealGPU(numEig/2, iter, eigMaxIter_, eigMinTolerance_, eigTolNow );
#else
        eigSolPtr_->PPCGSolveReal(numEig/2, iter, eigMaxIter_, eigMinTolerance_, eigTolNow );
#endif
      }
    }
    else if( esdfParam.PWSolver == "Davidson" || esdfParam.PWSolver == "DavidsonScaLAPACK" ){
      if( spinType_ == 1 || spinType_ == 4 ){
        eigSolPtr_->DavidsonSolveReal(numEig, iter, 2, eigMaxIter, eigMinTolerance_, eigTolNow );
      }
      else {
        ham.SetSpinSwitch(0);
        eigSolPtr_->DavidsonSolveReal(numEig/2, iter, 2, eigMaxIter, eigMinTolerance_, eigTolNow );

        ham.SetSpinSwitch(1);
        eigSolPtr_->DavidsonSolveReal(numEig/2, iter, 2, eigMaxIter, eigMinTolerance_, eigTolNow );
      }
    }
    else{
      // FIXME Merge the Chebyshev into an option of PWSolver
      ErrorHandling("Not supported PWSolver type.");
    }
#endif
  }

  ham.EigVal() = eigSolPtr_->EigVal();

  if( esdfParam.isCalculateEnergyBand ){
    statusOFS << "The Non-SCF calculation is finished." << std::endl;
    return;
  }

  // Compute the occupation rate
#ifdef _COMPLEX_
  CalculateOccupationRate( ham.EigVal(), ham.OccupationRate(), fft.domain.weight );
#else
  CalculateOccupationRate( ham.EigVal(), ham.OccupationRate() );
#endif

  // Calculate the Harris energy before updating the density
  CalculateHarrisEnergy ();

  // Compute the electron density

  // Store the old density in densityOld_ before updating it
  Int ntotFine = fft.domain.NumGridTotalFine();
  Int nspin = fft.domain.numSpinComponent;
  blas::Copy( ntotFine*nspin, ham.Density().Data(), 1,  densityOld_.Data(), 1 );

#ifdef GPU
  ham.CalculateDensityGPU( psi, ham.OccupationRate(), totalCharge_, fft);
#else
  ham.CalculateDensity( psi, ham.OccupationRate(), totalCharge_, fft);
#endif

    // Use density as convergence criteria when density is mixed
    
    DblNumMat& densityNew_ = ham.Density();

    DblNumMat dRho( ntotFine, nspin );
    blas::Copy( ntotFine * nspin, densityNew_.Data(), 1, dRho.Data(), 1 );
    blas::Axpy( ntotFine * nspin, -1.0, densityOld_.Data(), 1, dRho.Data(), 1 );

    scfNorm_ = this->CalculateError( dRho );

if( mixVariable_ == "density" ){
    // Use density as convergence criteria when density is mixed
    if( scfNorm_ >= scfTolerance_ ){
      if( mixType_ == "anderson" || mixType_ == "kerker+anderson" ){
        andersonMix(
            iter,
            mixStepLength_,
            mixType_,
            ham.Density(),
            densityOld_,
            densityNew_,
            dvMat_,
            dfMat_ );
      }
      else if( mixType_ == "broyden"){
        BroydenMix(
            iter,
            mixStepLength_,
            mixType_,
            ham.Density(),
            densityOld_,
            densityNew_,
            GdfMat_,
            GdvMat_,
            GcdfMat_);   
      }
      else{
        ErrorHandling("Invalid mixing type.");
      }
      // Additional processing is required if spin is considered
      if( nspin == 2 ){
        // The spindensity should be recalculated
        DblNumMat& density = ham.Density();
        DblNumMat& spindensity = ham.SpinDensity();

        blas::Copy( ntotFine, density.VecData(0), 1, spindensity.VecData(0), 1 );
        blas::Copy( ntotFine, density.VecData(0), 1, spindensity.VecData(1), 1 );
        blas::Axpy( ntotFine, 1.0, density.VecData(1), 1, spindensity.VecData(0), 1);
        blas::Axpy( ntotFine, -1.0, density.VecData(1), 1, spindensity.VecData(1), 1);
        blas::Scal( ntotFine*nspin, 0.5, spindensity.Data(), 1 );
      }
      else if( nspin == 4 ){
        DblNumMat& density = ham.Density();
        DblNumMat& spindensity = ham.SpinDensity();
        DblNumVec& segni = ham.Segni();
      
        Real dotmag, amag, vecdot, Tempsegni;
        Real epsDot = 1e-16;
        Point3 tempmag;
        Point3& spinaxis = ham.SpinAxis();
        bool isParallel = esdfParam.isParallel;

        for( Int i = 0; i < ntotFine; i++){
          tempmag = Point3(  density(i,1), density(i,2), density(i,3) );
          if ( isParallel ){
            vecdot = tempmag[0]*spinaxis[0] + tempmag[1]*spinaxis[1] 
                + tempmag[2]*spinaxis[2];
            if( vecdot > epsDot )
              Tempsegni = 1.0;
            else if( vecdot < -epsDot )
              Tempsegni = -1.0;
            else
              Tempsegni = 0.0;
          }
          else{
            Tempsegni = 1.0;
          }

          amag = tempmag.l2();
          spindensity(i,UP) = 0.5 * ( density(i,RHO) + Tempsegni*amag );
          spindensity(i,DN) = 0.5 * ( density(i,RHO) - Tempsegni*amag );
          segni[i] = Tempsegni;
        }
      }  // ---- end of if( nspin == 2 ) ----
    }  // ---- end of if( scfNorm_ >= scfTolerance_ ) ----
  }  // ---- end of if( mixType_ == "density" ) ----

  // Compute the exchange-correlation potential and energy
  if( isCalculateGradRho_ ){
    ham.CalculateGradDensity( fft );
  }

  ham.CalculateXC( Exc_, fft ); 
  // Compute the Hartree energy
  ham.CalculateHartree( fft );
  // No external potential

  // Compute the total potential
  ham.CalculateVtot( vtotNew_ );

  return ;
} 		// -----  end of method SCF::InnerSolve  ----- 

#ifdef _COMPLEX_
void
SCF::CalculateOccupationRate    ( std::vector<DblNumVec>& eigVal, std::vector<DblNumVec>& occupationRate, DblNumVec& weight )
{
  // For a given finite temperature, update the occupation number 

  // Multiple k-points case : calculate in first rowComm_kpoint 
  Domain& dm = eigSolPtr_->FFT().domain;

  MPI_Barrier( dm.comm );

  int colrank;
  MPI_Comm_rank( dm.colComm_kpoint, &colrank );

  std::string smearing_scheme = esdfParam.smearing_scheme;

  Int npsi       = eigSolPtr_->Ham().NumStateTotal();
  Int nOccStates = eigSolPtr_->Ham().NumOccupiedState();
  Int nkLocal    = eigVal.size();

  if( colrank == 0 ){

    MPI_Comm mpi_comm = dm.rowComm_kpoint;
    int rowsize;
    MPI_Comm_size( mpi_comm, &rowsize );

    IntNumVec& KpointIdx = dm.KpointIdx;

    for( Int k = 0 ; k < nkLocal; k++ ){
      if( eigVal[k].m() != npsi ){
        std::ostringstream msg;
        msg 
          << "The number of eigenstates do not match."  << std::endl
          << "eigVal         ~ " << eigVal[k].m() << std::endl
          << "numStateTotal  ~ " << npsi << std::endl;
        ErrorHandling( msg.str().c_str() );
      }
    }

    for( Int k = 0; k < nkLocal; k++ ){
      if( occupationRate[k].m() != npsi ) occupationRate[k].Resize( npsi );
    }

    // The k points are distributed uniformly in each group
    Int nkTotal = nkLocal * rowsize;

    // Sort all eigenvalues in non-descending order
    DblNumVec eigValLocal( nkLocal * npsi );
    DblNumVec eigValTotal( nkTotal * npsi );
    for( Int k = 0; k < nkLocal; k++ ){
      blas::Copy( npsi, eigVal[k].Data(), 1, &eigValLocal[k*npsi], 1 );
    }

    Int sendsize = nkLocal * npsi;
    IntNumVec localSize(rowsize);
    IntNumVec localSizeDispls(rowsize);
    for( Int i = 0; i < rowsize; i++ ){
      localSize(i) = sendsize;
      localSizeDispls(i) = i * sendsize; 
    }
    // The eigenvalues in eigValTotal is in order of ascendent k index
    // but the k index for eigValLocal should be find in KpointIdx
    MPI_Allgatherv( eigValLocal.Data(), sendsize, MPI_DOUBLE, eigValTotal.Data(),
        localSize.Data(), localSizeDispls.Data(), MPI_DOUBLE, mpi_comm );
 
    Sort( eigValTotal );
   
    if( npsi == nOccStates ){
      // No empty band
      for( Int k = 0; k < nkLocal; k++ ){
        Int ik = KpointIdx(k);
        for( Int j = 0; j < npsi; j++ )
          occupationRate[k](j) = weight(ik);
      }
      fermi_ = eigValTotal(npsi*nkTotal-1);
    }
    else if( npsi > nOccStates ){    

      if( esdfParam.temperature == 0.0 ){
        // No smearing in zero-temperature case
        fermi_ = eigValTotal(nOccStates*nkTotal-1);
        for( Int k = 0; k < nkLocal; k++ ){
          Int ik = KpointIdx(k);
          for( Int j = 0; j < npsi; j++ ){
            if( eigVal[k][j] <= fermi_ ){
              occupationRate[k](j) = weight(ik);
            }
            else{
              occupationRate[k](j) = 0.0;
            }
          }
        }
      } 
      else{    
        // Enhance the convergence condition because the fermi energy is 
        // very influenced by error
        Real tol = 1e-16;
        Int maxiter = 200;

        Real lb, ub, flb, fub, occsum;
        Int ilb, iub, iter;

        // Use bisection to find efermi such that 
        // sum_i,k fermidirac(ev(i,k))*weight(k) = nocc
        ilb = 1;
        iub = nkTotal * npsi;
        lb = eigValTotal(ilb-1);
        ub = eigValTotal(iub-1);

        fermi_ = (lb+ub)*0.5;
        occsum = 0.0;
        for(Int k = 0; k < nkTotal; k++){
          for(Int j = 0; j < npsi; j++){
            occsum += weight(k) * wgauss( eigValTotal(j+k*npsi), 
                fermi_, Tbeta_, smearing_scheme );
          }
        }
     
        // Start bisection iteration 
        iter = 1;
        while( (fabs(occsum - nOccStates) > tol) && (iter < maxiter) ) {
          if( occsum < nOccStates ) {lb = fermi_;}
          else {ub = fermi_;}

          fermi_ = (lb+ub)*0.5;
          occsum = 0.0;
          for(Int k = 0; k < nkTotal; k++){
            for(Int j = 0; j < npsi; j++){
              occsum += weight(k) * wgauss( eigValTotal(j+k*npsi), 
                fermi_, Tbeta_, smearing_scheme );
            }
          }
          iter++;
        }

        for(Int k = 0; k < nkLocal; k++){
          Int ik = KpointIdx(k);  
          for(Int j = 0; j < npsi; j++){
            occupationRate[k](j) = weight(ik) * wgauss( eigVal[k](j),
                fermi_, Tbeta_, smearing_scheme );
          }
        }
      }  // ---- end of if( esdfParam.temperature == 0.0 ) ----
    }
    else {
      ErrorHandling( "The number of eigenvalues in ev should be larger than nocc" );
    }
  }

  for( Int k = 0; k < nkLocal; k++ ){ 
    MPI_Bcast(occupationRate[k].Data(), npsi, MPI_DOUBLE, 0, dm.colComm_kpoint);
  }

  return ;
}         // -----  end of method SCF::CalculateOccupationRate ( Complex version )  ----- 
#else
void
SCF::CalculateOccupationRate    ( DblNumVec& eigVal, DblNumVec& occupationRate )
{
  Int npsi       = eigSolPtr_->Ham().NumStateTotal();
  Int nOccStates = eigSolPtr_->Ham().NumOccupiedState();

  std::string smearing_scheme = esdfParam.smearing_scheme;

  if( eigVal.m() != npsi ){
    std::ostringstream msg;
    msg 
      << "The number of eigenstates do not match."  << std::endl
      << "eigVal         ~ " << eigVal.m() << std::endl
      << "numStateTotal  ~ " << npsi << std::endl;
    ErrorHandling( msg.str().c_str() );
  }

  if( occupationRate.m() != npsi ) occupationRate.Resize( npsi );
  
  DblNumVec eigValTotal( npsi );
  blas::Copy( npsi, eigVal.Data(), 1, eigValTotal.Data(), 1 );

  Sort( eigValTotal );
  
  if( npsi == nOccStates ){
    for( Int j = 0; j < npsi; j++ ){
      occupationRate(j) = 1.0;
    }
    fermi_ = eigValTotal(npsi-1);
  }  
  else if( npsi > nOccStates ){
    if( esdfParam.temperature == 0.0 ){
      fermi_ = eigValTotal(nOccStates-1);
      for( Int j = 0; j < npsi; j++ ){
        if( eigVal[j] <= fermi_ ){
          occupationRate(j) = 1.0;
        }
        else{
          occupationRate(j) = 0.0;
        }
      }
    }
    else{
      Real tol = 1e-16;
      Int maxiter = 200;

      Real lb, ub, flb, fub, occsum;
      Int ilb, iub, iter;

      ilb = 1;
      iub = npsi;
      lb = eigValTotal(ilb-1);
      ub = eigValTotal(iub-1);

      fermi_ = (lb+ub)*0.5;
      occsum = 0.0;
      for(Int j = 0; j < npsi; j++){
        occsum += wgauss( eigValTotal(j), fermi_, Tbeta_, smearing_scheme );
      }

      iter = 1;
      while( (fabs(occsum - nOccStates) > tol) && (iter < maxiter) ) {
        if( occsum < nOccStates ) {lb = fermi_;}
        else {ub = fermi_;}

        fermi_ = (lb+ub)*0.5;
        occsum = 0.0;
        for(Int j = 0; j < npsi; j++){
          occsum += wgauss( eigValTotal(j), fermi_, Tbeta_, smearing_scheme );
        }
        iter++;
      }

      for(Int j = 0; j < npsi; j++){
        occupationRate(j) = wgauss( eigVal(j), fermi_, Tbeta_, smearing_scheme );
      }
    }
  }
  else{
    ErrorHandling( "The number of eigenvalues in ev should be larger than nocc" );
  }

  return;
}         // ----- end of method SCF::CalculateOccupationRate ( Real version ) ----- 
#endif

void
SCF::CalculateEnergy    (  )
{
  std::string smearing_scheme = esdfParam.smearing_scheme;
  // Kinetic energy
#ifdef _COMPLEX_
  Domain& dm = eigSolPtr_->FFT().domain;
  
  MPI_Barrier( dm.comm );

  int colrank;
  MPI_Comm_rank( dm.colComm_kpoint, &colrank );

  Int numSpin = eigSolPtr_->Ham().NumSpin();
  std::vector<DblNumVec>& eigVal         = eigSolPtr_->Ham().EigVal();
  std::vector<DblNumVec>& occupationRate = eigSolPtr_->Ham().OccupationRate();

  if( colrank == 0 ){

    Real EkinLocal = 0.0;
    Ekin_ = 0.0;
  
    for( Int k = 0; k < eigVal.size(); k++ ){
      for( Int i = 0; i < eigVal[0].m(); i++ ){
        EkinLocal += numSpin * eigVal[k](i) * occupationRate[k](i);
      }
    }

    MPI_Allreduce( &EkinLocal, &Ekin_, 1, MPI_DOUBLE, MPI_SUM, dm.rowComm_kpoint );
  }

  MPI_Bcast( &Ekin_, 1, MPI_DOUBLE, 0, dm.colComm_kpoint  );
#else
  Ekin_ = 0.0;
  DblNumVec&  eigVal         = eigSolPtr_->Ham().EigVal();
  DblNumVec&  occupationRate = eigSolPtr_->Ham().OccupationRate();

  Int numSpin = eigSolPtr_->Ham().NumSpin();
  for (Int i=0; i < eigVal.m(); i++) {
    Ekin_  += numSpin * eigVal(i) * occupationRate(i);
  }
#endif
  // Hartree and xc part
  Int  ntot  = eigSolPtr_->FFT().domain.NumGridTotalFine();
  Int  nspin = eigSolPtr_->FFT().domain.numSpinComponent;
  Real vol   = eigSolPtr_->FFT().domain.Volume();
  DblNumMat&  density      = eigSolPtr_->Ham().Density();
  DblNumMat&  spindensity  = eigSolPtr_->Ham().SpinDensity();
  DblNumVec&  pseudoCharge = eigSolPtr_->Ham().PseudoCharge();
  DblNumVec&  vhart        = eigSolPtr_->Ham().Vhart();

  // Correct the incorrect Ecoul and Exc included in Ekin
  DblNumMat&  vtotOld      = eigSolPtr_->Ham().Vtot();
  DblNumVec&  vLocalSR     = eigSolPtr_->Ham().VLocalSR();
  DblNumVec&  vext         = eigSolPtr_->Ham().Vext();

  Ecor_ = 0.0;
  if( nspin == 1 ){
    for( Int i = 0; i < ntot; i++ ){
      Ecor_ += ( vLocalSR(i) + vext(i) - vtotOld(i,RHO) ) * density(i,RHO);
    }
  }
  else if( nspin == 2 ){
    for( Int is = 0; is < nspin; is++ ){
      for( Int i = 0; i < ntot; i++ ){
        Ecor_ += ( vLocalSR(i) + vext(i) - vtotOld(i,is) ) * spindensity(i,is);
      }
    }
  }
  else if( nspin == 4 ){
    for( Int i = 0; i < ntot; i++ ){
      Ecor_ += ( vLocalSR(i) + vext(i) - vtotOld(i,RHO) ) * density(i,RHO);
    }
    for( Int is = 1; is < nspin; is++ ){
      for( Int i = 0; i < ntot; i++ ){
        Ecor_ += ( vext(i) - vtotOld(i,is) ) * density(i,is);
      }
    }
  }
  Ecor_ *= vol/Real(ntot);

  Ehart_ = 0.0;

  for( Int i = 0; i < ntot; i++ ){
    Ehart_ += 0.5 * vhart(i) * ( density(i,RHO) - pseudoCharge(i) );
  }

  Ehart_ *= vol/Real(ntot);
  // Ionic repulsion related energy
  Eself_ = eigSolPtr_->Ham().Eself();

  Ecor_ = Ecor_ + Exc_ + Ehart_ - Eself_;
  if( esdfParam.isUseVLocal == true ){
    EIonSR_ = eigSolPtr_->Ham().EIonSR();
    Ecor_ += EIonSR_;
  }

  // Van der Waals energy
  EVdw_ = eigSolPtr_->Ham().EVdw();
  Ecor_ += EVdw_;

  // External energy
  Eext_ = eigSolPtr_->Ham().Eext();
  Ecor_ += Eext_;

  // Total energy
  Etot_ = Ekin_ + Ecor_;

  // Helmholtz fre energy
  if( eigSolPtr_->Ham().NumOccupiedState() == eigSolPtr_->Ham().NumStateTotal()
      || esdfParam.temperature == 0.0 ){
    // Zero temperature
    Efree_ = Etot_;
  }
  else{
    // Finite temperature
    Real fermi = fermi_;
    Real Tbeta = Tbeta_;
#ifdef _COMPLEX_
    if( colrank == 0 ){
      Real EfreeLocal = 0.0;
      Efree_ = 0.0;

      IntNumVec& KpointIdx = dm.KpointIdx;
      DblNumVec &weight = dm.weight;

      for( Int k = 0; k < eigVal.size(); k++ ){
        Int ik = KpointIdx(k);
        for( Int i = 0; i < eigVal[k].m(); i++ ){
          Real eig = eigVal[k](i);
          
          EfreeLocal += numSpin * weight(ik) * getEntropy( eig, fermi, Tbeta, smearing_scheme );
        }
      }

      MPI_Allreduce( &EfreeLocal, &Efree_, 1, MPI_DOUBLE, MPI_SUM, dm.rowComm_kpoint );
    }

    MPI_Bcast( &Efree_, 1, MPI_DOUBLE, 0, dm.colComm_kpoint );
#else
    Efree_ = 0.0;
    for( Int k = 0; k < eigVal.m(); k++){
      Real eig = eigVal(k);

      Efree_ += numSpin * getEntropy( eig, fermi, Tbeta, smearing_scheme );
    }
#endif
    Efree_ += Etot_;
  }

  return ;
}         // -----  end of method SCF::CalculateEnergy  ----- 

void
SCF::CalculateHarrisEnergy ( )
{
  // These variables are temporary variables only used in this routine
  Real Ekin, Eself, Ehart, EVxc, Exc, Ecor, Efree, EIonSR, EVdw, Eext;

  // Kinetic energy
#ifdef _COMPLEX_
  Domain& dm = eigSolPtr_->FFT().domain;

  MPI_Barrier( dm.comm );

  int colrank;
  MPI_Comm_rank( dm.colComm_kpoint, &colrank );

  Int numSpin = eigSolPtr_->Ham().NumSpin();
  std::vector<DblNumVec>& eigVal         = eigSolPtr_->Ham().EigVal();
  std::vector<DblNumVec>& occupationRate = eigSolPtr_->Ham().OccupationRate();
  if( colrank == 0 ){

    Real EkinLocal = 0.0; 
    Ekin = 0.0;

    for( Int k = 0; k < eigVal.size(); k++ ){
      for( Int i = 0; i < eigVal[0].m(); i++ ){
        EkinLocal += numSpin * eigVal[k](i) * occupationRate[k](i);
      }
    }

    MPI_Allreduce( &EkinLocal, &Ekin, 1, MPI_DOUBLE, MPI_SUM, dm.rowComm_kpoint );
  }
  
  MPI_Bcast( &Ekin, 1, MPI_DOUBLE, 0, dm.colComm_kpoint  );
#else
  Ekin = 0.0;
  DblNumVec&  eigVal         = eigSolPtr_->Ham().EigVal();
  DblNumVec&  occupationRate = eigSolPtr_->Ham().OccupationRate();

  Int numSpin = eigSolPtr_->Ham().NumSpin();
  for (Int i=0; i < eigVal.m(); i++) {
    Ekin  += numSpin * eigVal(i) * occupationRate(i);
  }
#endif
  // Self energy part
  Eself = 0;
  std::vector<Atom>&  atomList = eigSolPtr_->Ham().AtomList();
  for(Int a=0; a< atomList.size() ; a++) {
    Int type = atomList[a].type;
    Eself +=  ptablePtr_->SelfIonInteraction(type);
  }

  // Ionic repulsion related energy
  Eself = eigSolPtr_->Ham().Eself();

  EIonSR = eigSolPtr_->Ham().EIonSR();

  // Van der Waals energy
  EVdw = eigSolPtr_->Ham().EVdw();

  // External energy
  Eext = eigSolPtr_->Ham().Eext();

  // Nonlinear correction part.  This part uses the Hartree energy and
  // XC correlation energy from the old electron density.
  Int  ntot = eigSolPtr_->FFT().domain.NumGridTotalFine();
  Int  nspin = eigSolPtr_->FFT().domain.numSpinComponent;
  Real vol  = eigSolPtr_->FFT().domain.Volume();
  DblNumMat&  density      = eigSolPtr_->Ham().Density();
  DblNumMat&  spindensity      = eigSolPtr_->Ham().SpinDensity();
  DblNumMat&  vxc          = eigSolPtr_->Ham().Vxc();
  DblNumVec&  pseudoCharge = eigSolPtr_->Ham().PseudoCharge();
  DblNumVec&  vhart        = eigSolPtr_->Ham().Vhart();
  Ehart = 0.0;
  EVxc  = 0.0;

  if( nspin == 1 || nspin == 4 ){
    for( Int is = 0; is < nspin; is++ ){
      for( Int i = 0; i < ntot; i++ ){
        EVxc  += vxc(i,is) * density(i,is);
      }
    }
  }
  else if( nspin == 2 ){
    for( Int i = 0; i < ntot; i++ ){
      EVxc  += vxc(i,UP) * spindensity(i,UP);
      EVxc  += vxc(i,DN) * spindensity(i,DN);
    }
  }

  for( Int i = 0; i < ntot; i++ ){
    Ehart += 0.5 * vhart(i) * ( density(i,RHO) + pseudoCharge(i) );
  }
  Ehart *= vol/Real(ntot);
  EVxc  *= vol/Real(ntot);
  Exc    = Exc_;

  // Correction energy
  Ecor = (Exc - EVxc) - Ehart - Eself + EIonSR + EVdw + Eext;

  // Helmholtz free energy
  if( eigSolPtr_->Ham().NumOccupiedState() == eigSolPtr_->Ham().NumStateTotal()
      || esdfParam.temperature == 0.0 ){
    // Zero temperature
    Efree = Ekin + Ecor;
  }
  else{
    // Finite temperature
    Real fermi = fermi_;
    Real Tbeta = Tbeta_;
#ifdef _COMPLEX_
    if( colrank == 0 ){
      Real EfreeLocal = 0.0; 
      Efree = 0.0;

      IntNumVec& KpointIdx = dm.KpointIdx;
      DblNumVec& weight = dm.weight;

      for( Int k = 0; k < eigVal.size(); k++ ){
        Int ik = KpointIdx(k);
        for( Int i = 0; i < eigVal[k].m(); i++ ){
          Real eig = eigVal[k](i);
          if( eig - fermi >= 0){
            EfreeLocal += -numSpin * weight(ik) / Tbeta*log(1.0+exp(-Tbeta*(eig - fermi))); 
          }
          else{
            EfreeLocal += numSpin * weight(ik) * (eig - fermi) - numSpin * weight(ik) / 
                Tbeta*log(1.0+exp(Tbeta*(eig-fermi)));
          }
        }
      }

      MPI_Allreduce( &EfreeLocal, &Efree, 1, MPI_DOUBLE, MPI_SUM, dm.rowComm_kpoint );
    }

    MPI_Bcast( &Efree, 1, MPI_DOUBLE, 0, dm.colComm_kpoint );
#else
    Efree = 0.0;
    for( Int k = 0; k < eigVal.m(); k++ ){
      Real eig = eigVal(k);
      if( eig - fermi >= 0){
        Efree += -numSpin / Tbeta * log(1.0 + exp(-Tbeta * (eig - fermi)));
      }
      else{
        Efree += numSpin * (eig - fermi) - numSpin
            / Tbeta * log( 1.0 + exp(Tbeta*(eig-fermi)));
      }
    }
#endif
    Efree += Ecor + fermi * eigSolPtr_->Ham().NumOccupiedState() * numSpin; 
  }

  EfreeHarris_ = Efree;

  return ;
}         // -----  end of method SCF::CalculateHarrisEnergy  ----- 

// To be added : The density or potential mixing in coarse grids should be tried
//               to reduce memory burden
void
SCF::BroydenMix    (
    Int iter,
    Real            mixStepLength,
    std::string     mixType,
    DblNumMat&      vMix,
    DblNumMat&      vOld,
    DblNumMat&      vNew,
    CpxNumTns&      dfMat,
    CpxNumTns&      dvMat,
    CpxNumTns&      cdfMat ) {

  Fourier& fft = eigSolPtr_->FFT();

  IntNumVec& idxDensity = fft.idxFineCutDensity;

  Int ntotFine = fft.domain.NumGridTotalFine();
  Int ntot = idxDensity.Size();
  Int nspin = fft.domain.numSpinComponent;
  // Transform vOld and vNew from real space to reciprocal space
  CpxNumMat GvOld, GvNew, GvMix;
  GvOld.Resize(ntot, nspin); GvNew.Resize(ntot, nspin); GvMix.Resize(ntot, nspin);
  SetValue( GvOld, Z_ZERO ); 
  SetValue( GvNew, Z_ZERO );
  SetValue( GvMix, Z_ZERO );

  for( Int is = 0; is < nspin; is++ ){       
    for( Int i = 0; i < ntotFine; i++ ){
      fft.inputComplexVecFine(i) = Complex(vOld(i,is), D_ZERO); 
    }
    FFTWExecute( fft, fft.forwardPlanFine );

    for( Int i = 0; i < ntot; i++ ){
      GvOld(i,is) = fft.outputComplexVecFine(idxDensity(i));
    }
  }
   
  for( Int is = 0; is < nspin; is++ ){ 
    for( Int i = 0; i < ntotFine; i++ ){
      fft.inputComplexVecFine(i) = Complex(vNew(i,is), D_ZERO);
    }
    FFTWExecute( fft, fft.forwardPlanFine );

    for( Int i = 0; i < ntot; i++ ){
      GvNew(i,is) = fft.outputComplexVecFine(idxDensity(i));
    }
  }

  blas::Axpy( ntot*nspin, -1.0, GvOld.Data(), 1, GvNew.Data(), 1);

  Int iter_used = std::min( iter-1, mixMaxDim_ );
  Int ipos = iter - 1 - ((iter-2)/ mixMaxDim_ ) * mixMaxDim_;   

  if( iter > 1 ){
    blas::Copy( ntot*nspin, cdfMat.MatData(0), 1, dfMat.MatData(ipos-1), 1 );  
    blas::Axpy( ntot*nspin, -1.0, GvNew.Data(), 1, dfMat.MatData(ipos-1), 1);
    blas::Copy( ntot*nspin, cdfMat.MatData(1), 1, dvMat.MatData(ipos-1), 1 ); 
    blas::Axpy( ntot*nspin, -1.0, GvOld.Data(), 1, dvMat.MatData(ipos-1), 1);
  }

  blas::Copy( ntot*nspin, GvNew.Data(), 1, cdfMat.MatData(0), 1 );
  blas::Copy( ntot*nspin, GvOld.Data(), 1, cdfMat.MatData(1), 1 );
 
  if( iter_used > 0 ){
    DblNumMat betamix;
    betamix.Resize( iter_used, iter_used ); SetValue( betamix, D_ZERO );
    // Calculate matrix betamix   
    for (Int i=0; i<iter_used; i++) {
      for (Int j=i; j<iter_used; j++) {
        betamix(i,j) = this->RhoDdot( dfMat.MatData(j), dfMat.MatData(i) );
        betamix(j,i) = betamix(i,j);
      }
    }
    // Inverse betamix using the Bunch-Kaufman diagonal pivoting method
    IntNumVec iwork;
    iwork.Resize( iter_used ); SetValue( iwork, I_ZERO );
    
    lapack::Sytrf( 'U', iter_used, betamix.Data(), iter_used, iwork.Data() );
    lapack::Sytri( 'U', iter_used, betamix.Data(), iter_used, iwork.Data() );
    for (Int i=0; i<iter_used; i++) {
      for (Int j=i+1; j<iter_used; j++) {
        betamix(j,i) = betamix(i,j);
      }
    }
    
    DblNumVec work;
    Real gamma0 = D_ZERO;
    work.Resize( iter_used ); SetValue( work, D_ZERO );

    for (Int i=0; i<iter_used; i++){
      work(i) = this->RhoDdot( dfMat.MatData(i), GvNew.Data() );
    }

    for (Int i=0; i<iter_used; i++){
      gamma0 = blas::Dot( iter_used, betamix.VecData(i), 1, work.Data(), 1 );
      blas::Axpy( ntot*nspin, -gamma0, dvMat.MatData(i), 1, GvOld.Data(), 1);     
      blas::Axpy( ntot*nspin, -gamma0, dfMat.MatData(i), 1, GvNew.Data(), 1);
    }
  } // end of if( iter_used > 0 )
  
  // Transform vMix to real space
  GvMix = GvOld;
  blas::Axpy( ntot*nspin, mixStepLength, GvNew.Data(), 1, GvMix.Data(), 1 );

  for( Int is = 0; is < nspin; is++ ){
    SetValue( fft.outputComplexVecFine, Z_ZERO);
    for( Int i = 0; i < ntot; i++ ){
      fft.outputComplexVecFine(idxDensity(i)) = GvMix(i,is);
    }

    FFTWExecute( fft, fft.backwardPlanFine );

    for( Int i = 0; i < ntotFine; i++ ){
      vMix(i,is) = fft.inputComplexVecFine(i).real();
    }
  }
}  // ----- end of method SCF::BroydenMix -----
  
Real
SCF::RhoDdot    ( Complex* rho1_in, Complex* rho2_in )
{
  // Estimate the self-consistency error on the energy.
  Fourier& fft = eigSolPtr_->FFT();  

  IntNumVec& idxDensity = fft.idxFineCutDensity;

  Int ntot = idxDensity.Size();
  Int nspin = fft.domain.numSpinComponent;

  CpxNumMat rho1(ntot, nspin, false, rho1_in);
  CpxNumMat rho2(ntot, nspin, false, rho2_in);

  Real rhodot = D_ZERO; Real spindot = D_ZERO;
  Real fac = PI * fft.domain.Volume();
  
  for( Int i = 0; i < ntot; i++ ){ 
    Int ig = idxDensity(i);
    if( fft.gkkFine(ig) != 0 ){
      rhodot += (std::conj(rho1(i,RHO)) * rho2(i,RHO)).real() / fft.gkkFine(ig);
    }
  }
  rhodot *= fac;

  fac = fft.domain.Volume() / 2 / PI;
  for(Int i = 1; i < nspin; i++){
    for( Int j = 0; j < ntot; j++){
      spindot += (std::conj(rho1(j,i)) * rho2(j,i)).real();
    }
  }
  spindot *= fac;

  return( rhodot + spindot );
}  // ---- end of method SCF::RhoDdot ---- 

void
SCF::andersonMix    ( 
    Int iter,
    Real            mixStepLength,
    std::string     mixType,
    DblNumMat&      vMix,
    DblNumMat&      vOld,
    DblNumMat&      vNew,
    DblNumTns&      dvMat,
    DblNumTns&      dfMat )
{ // Can be transformed to Broyden if Broyden adpots <f|g> = \sum_i f(ri)g(ri) in real space
  // See JOURNAL OF COMPUTATIONAL PHYSICS 124, 271–285 (1996)
  // "A Comparative Study on Methods for Convergence Acceleration
  // of Iterative Vector Sequences"
  // Here we still use the original form instead of the transformed one
  Fourier& fft = eigSolPtr_->FFT();
  Int ntotFine = fft.domain.NumGridTotalFine();
  Int ntotFineR2C = fft.numGridTotalR2CFine;
  Int nspin = fft.domain.numSpinComponent;
  int pos=((iter-1)%mixMaxDim_-1+mixMaxDim_)%mixMaxDim_;
  int next=(pos+1)%mixMaxDim_;
  blas::Axpy(ntotFine*nspin,-1.0,vOld.Data(),1,vNew.Data(),1); // F=Xout-Xin
  DblNumMat tempMat(ntotFine*nspin,2);
  blas::Copy(ntotFine*nspin,vOld.Data(),1,tempMat.VecData(0),1); // save Xin,F in end
  blas::Copy(ntotFine*nspin,vNew.Data(),1,tempMat.VecData(1),1);
  if(iter>1){
    int dim=std::min(iter-1,mixMaxDim_);
    DblNumMat overlap(dim,dim);
    DblNumMat diff(ntotFine*nspin,dim);
    DblNumMat difv(ntotFine*nspin,dim);
    DblNumVec work(dim);
    blas::Copy(ntotFine*nspin*dim,dvMat.Data(),1,difv.Data(),1);
    blas::Copy(ntotFine*nspin*dim,dfMat.Data(),1,diff.Data(),1);
    for(int i=0;i<dim;i++){
      blas::Axpy(ntotFine*nspin,-1.0,vOld.Data(),1,difv.VecData(i),1); // X(l)-X(l-j)
      blas::Axpy(ntotFine*nspin,-1.0,vNew.Data(),1,diff.VecData(i),1); // F(l)-F(l-j)
    }
    for(int i=0;i<dim;i++){
      work(i)=blas::Dot(ntotFine*nspin,diff.VecData(i),1,vNew.Data(),1); // <F(l)-F(l-i)|F(l)>
      for(int j=i;j<dim;j++){
        overlap(i,j)=blas::Dot(ntotFine*nspin,diff.VecData(i),1,diff.VecData(j),1);
        overlap(j,i)=overlap(i,j); // <F(l)-F(l-i)|F(l)-F(l-j)>
      }
    }
    lapack::Potrf('L',dim,overlap.Data(),dim); // overlap * \theta = work
    lapack::Potrs('L',dim,overlap.Data(),dim,I_ONE,work.Data(),dim); // now work is \theta
    for(int i=0;i<dim;i++){
      blas::Axpy(ntotFine*nspin,-work(i),difv.VecData(i),1,vOld.Data(),1); // \overline{X}
      blas::Axpy(ntotFine*nspin,-work(i),diff.VecData(i),1,vNew.Data(),1); // \overline{F}
    }
  }
  if(mixType_ == "kerker+anderson"){ // may be better for metals
    double k2=2.0*PI*0.08;
    double e0=10.0;
    blas::Copy(ntotFine,vNew.VecData(0),1,fft.inputVecR2CFine.Data(),1);
    FFTWExecute(fft,fft.forwardPlanR2CFine);
    for(int ig=0;ig<ntotFineR2C;ig++){
      fft.outputVecR2CFine[ig]*=fft.gkkR2CFine[ig]/(fft.gkkR2CFine[ig]+k2);
    }
    FFTWExecute(fft,fft.backwardPlanR2CFine);
    blas::Copy(ntotFine,fft.inputVecR2CFine.Data(),1,vNew.VecData(0),1);
  }
  for(int ir=0;ir<ntotFine*nspin;ir++){
    vMix.Data()[ir]=vOld.Data()[ir]+mixStepLength*vNew.Data()[ir]; // X=\overline{X}+\beta*\overline{F}
  }
  blas::Copy(ntotFine*nspin,tempMat.VecData(0),1,dvMat.MatData(next),1); // update history of Xin and F
  blas::Copy(ntotFine*nspin,tempMat.VecData(1),1,dfMat.MatData(next),1);
}

Real
SCF::CalculateError   ( DblNumMat& drho )
{
  Fourier& fft = eigSolPtr_->FFT();
  IntNumVec& idxDensity = fft.idxFineCutDensity;
  
  Int ntotFine = fft.domain.NumGridTotalFine();    
  Int ntot = idxDensity.Size();
  Int nspin = fft.domain.numSpinComponent;
  
  CpxNumMat Gdrho;
  Gdrho.Resize(ntot, nspin); SetValue( Gdrho, Z_ZERO );

  for( Int is = 0; is < nspin; is++ ){
    for( Int i = 0; i < ntotFine; i++ ){
      fft.inputComplexVecFine(i) = Complex(drho(i,is), D_ZERO);
    }
    FFTWExecute( fft, fft.forwardPlanFine );

    for( Int i = 0; i < ntot; i++ ){
      Int ig = idxDensity(i);
      Gdrho(i,is) = fft.outputComplexVecFine(ig);
    }
  }

  Real fac = 1.0 / fft.domain.Volume();
  blas::Scal( ntot*nspin, fac, Gdrho.Data(), 1 ); 

  return ( this->RhoDdot(Gdrho.Data(), Gdrho.Data()) );
}  // ---- end of method SCF::CalculateError ---- 

void
SCF::PrintState    ( const Int iter  )
{
#ifdef _COMPLEX_
if( esdfParam.isCalculateEnergyBand ){
  for( Int k = 0; k < eigSolPtr_->Psi().size(); k++ ){
    Point3 kpoint = eigSolPtr_->Psi(k).Kpoint() ;
    Print(statusOFS, "kpoint#  = ", kpoint );

    for( Int i = 0; i < eigSolPtr_->EigVal(k).m(); i++ ){
      Print(statusOFS,
        "band#    = ", i,
        "eigval   = ", eigSolPtr_->EigVal(k)(i),
        "resval   = ", eigSolPtr_->ResVal(k)(i),
        "occrate  = ", eigSolPtr_->Ham().OccupationRate()[k](i));
    }
    statusOFS << std::endl;
  }
}
#else
#if 0
  for( Int i = 0; i < eigSolPtr_->EigVal().m(); i++ ){
    Print(statusOFS,
        "band#    = ", i,
        "eigval   = ", eigSolPtr_->EigVal()(i),
        "resval   = ", eigSolPtr_->ResVal()(i),
        "occrate  = ", eigSolPtr_->Ham().OccupationRate()(i));
  }
#endif
#endif
if( !esdfParam.isCalculateEnergyBand ){
  statusOFS << std::endl;
  //statusOFS 
    //<< "NOTE:  Ecor  = Exc - EVxc - Ehart - Eself + EIonSR + EVdw + Eext" << std::endl
    //<< "       Etot  = Ekin + Ecor" << std::endl
    //<< "       Efree = Etot    + Entropy" << std::endl << std::endl;
  Print(statusOFS, "Etot              = ",  Etot_, "[au]");
  Print(statusOFS, "Efree             = ",  Efree_, "[au]");
  Print(statusOFS, "EfreeHarris       = ",  EfreeHarris_, "[au]");
  Print(statusOFS, "Ekin              = ",  Ekin_, "[au]");
  Print(statusOFS, "Ehart             = ",  Ehart_, "[au]");
  Print(statusOFS, "EVxc              = ",  EVxc_, "[au]");
  Print(statusOFS, "Exc               = ",  Exc_, "[au]"); 
  Print(statusOFS, "EVdw              = ",  EVdw_, "[au]"); 
  Print(statusOFS, "Eself             = ",  Eself_, "[au]");
  Print(statusOFS, "EIonSR            = ",  EIonSR_, "[au]");
  Print(statusOFS, "Eext              = ",  Eext_, "[au]");
  Print(statusOFS, "Ecor              = ",  Ecor_, "[au]");
  Print(statusOFS, "Fermi             = ",  fermi_, "[au]");
  Print(statusOFS, "Total charge      = ",  totalCharge_, "[au]");
}
#ifdef _COMPLEX_

#else
    Real HOMO, LUMO;
    HOMO = eigSolPtr_->EigVal()(eigSolPtr_->Ham().NumOccupiedState()-1);
    if( eigSolPtr_->Ham().NumExtraState() > 0 )
      LUMO = eigSolPtr_->EigVal()(eigSolPtr_->Ham().NumOccupiedState());

    Print(statusOFS, "HOMO              = ",  HOMO*au2ev, "[eV]");
    if( eigSolPtr_->Ham().NumExtraState() > 0 ){
      Print(statusOFS, "LUMO              = ",  LUMO*au2ev, "[eV]");
    }
#endif

  return ;
}         // -----  end of method SCF::PrintState  ----- 

void
SCF::UpdateMDParameters    ( )
{
  scfMaxIter_ = esdfParam.MDscfInnerMaxIter;
  scfPhiMaxIter_ = esdfParam.MDscfPhiMaxIter;
  return ;
}         // -----  end of method SCF::UpdateMDParameters  ----- 

void
SCF::UpdateTDDFTParameters    ( )
{
  //scfMaxIter_    = esdfParam.TDDFTscfOuterMaxIter;
  //scfPhiMaxIter_ = esdfParam.TDDFTscfPhiMaxIter;
  return ;
}         // -----  end of method SCF::UpdateTDDFTParameters  ----- 

} // namespace pwdft
