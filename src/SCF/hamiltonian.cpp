/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

Author: Lin Lin, Wei Hu, Weile Jia

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
/// @file hamiltonian.cpp
/// @brief Hamiltonian class for planewave basis diagonalization method.
/// @date 2012-09-16
#include  "hamiltonian.hpp"
#include  "blas.hpp"
#include  "lapack.hpp"

namespace pwdft{

using namespace pwdft::PseudoComponent;
using namespace pwdft::DensityComponent;
using namespace pwdft::SpinTwo;
using namespace pwdft::GradThree;
using namespace pwdft::esdf;

// *********************************************************************
// KohnSham class
// *********************************************************************

KohnSham::KohnSham() {
  XCInitialized_ = false;
}

KohnSham::~KohnSham() {
  if( XCInitialized_ ){
    if( XCId_ == XC_LDA_XC_TETER93 )
    {
      xc_func_end(&XCFuncType_);
    }    
    else if( XId_ == XC_LDA_X && CId_ == XC_LDA_C_PZ )
    {
      xc_func_end(&XFuncType_);
      xc_func_end(&CFuncType_);
    }
    else if( ( XId_ == XC_GGA_X_PBE ) && ( CId_ == XC_GGA_C_PBE ) )
    {
      xc_func_end(&XFuncType_);
      xc_func_end(&CFuncType_);
    }
    else if( XCId_ == XC_HYB_GGA_XC_HSE06 ){
      xc_func_end(&XCFuncType_);
    }
    else if( XCId_ == XC_HYB_GGA_XC_PBEH ){
      xc_func_end(&XCFuncType_);
    }
    else
      ErrorHandling("Unrecognized exchange-correlation type");
  }
}

void
KohnSham::Setup    (
    const Domain&               dm,
    const PeriodTable&          ptable,
    const std::vector<Atom>&    atomList )
{
  domain_              = dm;
  atomList_            = atomList;
  numExtraState_       = esdfParam.numExtraState;
  XCType_              = esdfParam.XCType;
  
  hybridDFType_                    = esdfParam.hybridDFType;
  hybridDFKmeansWFType_            = esdfParam.hybridDFKmeansWFType;
  hybridDFKmeansWFAlpha_           = esdfParam.hybridDFKmeansWFAlpha;
  hybridDFKmeansTolerance_         = esdfParam.hybridDFKmeansTolerance;
  hybridDFKmeansMaxIter_           = esdfParam.hybridDFKmeansMaxIter;
  hybridDFNumMu_                   = esdfParam.hybridDFNumMu;
  hybridDFNumGaussianRandom_       = esdfParam.hybridDFNumGaussianRandom;
  hybridDFNumProcScaLAPACK_        = esdfParam.hybridDFNumProcScaLAPACK;
  hybridDFTolerance_               = esdfParam.hybridDFTolerance;
  BlockSizeScaLAPACK_              = esdfParam.BlockSizeScaLAPACK;
  exxDivergenceType_               = esdfParam.exxDivergenceType;

  // The number of density components
  numDensityComponent_ = dm.numSpinComponent;

  // The number of spinor components ( up and down if spin is considered )
  numSpinorComponent_ = ( numDensityComponent_ == 1 ? 1 : 2 );

  // default for spin-restricted case
  spinswitch_ = 0;

  // numSpin = 2 only for spin-restricted case  
  numSpin_ = ( numDensityComponent_ == 1 ? 2 : 1 );

  // Calculate the number of occupied states
  // need to distinguish the number of charges carried by the ion and that
  // carried by the electron
  Int numAtom = atomList_.size();
  Int nZion = 0, nelec = 0;
  for (Int a=0; a<numAtom; a++) {
    Int atype  = atomList_[a].type;
    if( ptable.ptemap().find(atype) == ptable.ptemap().end() ){
      ErrorHandling( "Cannot find the atom type." );
    }
    nZion = nZion + ptable.Zion(atype);
  }
  // Deal with the case when this is a spin-restricted calculation and the
  // number of electrons is not an even number.
  nelec = nZion + esdfParam.extraElectron;
  if( (nelec % 2 != 0) && (numSpin_ == 2) ){
    ErrorHandling( "This is spin-restricted calculation. nelec should be even." );
  }
  numOccupiedState_ = nelec / numSpin_;

  Int ntotCoarse = domain_.NumGridTotal();
  Int ntotFine = domain_.NumGridTotalFine();

  density_.Resize( ntotFine, numDensityComponent_ );   
  SetValue( density_, 0.0 );

  spindensity_.Resize( ntotFine, numSpinorComponent_ );
  SetValue( spindensity_, 0.0 );

  densityold_.Resize( ntotFine, numDensityComponent_ );
  SetValue( densityold_, 0.0 );

  if( numDensityComponent_ == 4 ){
    spinaxis_ = dm.spinaxis;
    segni_.Resize( ntotFine );
    SetValue( segni_, 0.0 );
  }

  gradDensity_.resize( DIM );
  for( Int d = 0; d < DIM; d++ ){
    gradDensity_[d].Resize( ntotFine, numSpinorComponent_ );
    SetValue (gradDensity_[d], 0.0);
  }

  pseudoCharge_.Resize( ntotFine );
  SetValue( pseudoCharge_, 0.0 );

  if( esdfParam.isUseVLocal == true ){
    vLocalSR_.Resize( ntotFine );
    SetValue( vLocalSR_, 0.0 );
    atomDensity_.Resize( ntotFine, numDensityComponent_ );
    SetValue( atomDensity_, 0.0 );
  }

  vext_.Resize( ntotFine );
  SetValue( vext_, 0.0 );

  vhart_.Resize( ntotFine );
  SetValue( vhart_, 0.0 );

  vtot_.Resize( ntotFine, numDensityComponent_ );
  SetValue( vtot_, 0.0 );

  epsxc_.Resize( ntotFine );
  SetValue( epsxc_, 0.0 );

  exc_.Resize( ntotFine );
  SetValue( exc_, 0.0 );

  vxc_.Resize( ntotFine, numDensityComponent_ );
  SetValue( vxc_, 0.0 );

#ifdef _COMPLEX_
  Int nk = domain_.KpointIdx.Size();
  ekin_.resize( nk );
  teter_.resize( nk );
  eigVal_.resize( nk );
  occupationRate_.resize( nk );
#endif

  // MPI communication 
  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);
  int dmCol = DIM;
  int dmRow = mpisize / dmCol;
  
  rowComm_ = MPI_COMM_NULL;
  colComm_ = MPI_COMM_NULL;

  if(mpisize >= DIM){

    IntNumVec mpiRowMap(mpisize);
    IntNumVec mpiColMap(mpisize);

    for( Int i = 0; i < mpisize; i++ ){
      mpiRowMap(i) = i / dmCol;
      mpiColMap(i) = i % dmCol;
    } 

    if( mpisize > dmRow * dmCol ){
      for( Int k = dmRow * dmCol; k < mpisize; k++ ){
        mpiRowMap(k) = dmRow - 1;
      }
    } 

    MPI_Comm_split( domain_.comm, mpiRowMap(mpirank), mpirank, &rowComm_ );
  }

  // Initialize the XC functionals, only spin-unpolarized case
  // Obtain the exchange-correlation id

  // Chens: spin-polarized case is added in 2023/6
  {
    PrintBlock(statusOFS, "XC functional information");

    isHybrid_ = false;
    Int nspin = XC_UNPOLARIZED;
    if( numDensityComponent_ >= 2 ){
      nspin = XC_POLARIZED;
    } 

    if( XCType_ == "XC_LDA_XC_TETER93" )
    { 
      XCId_ = XC_LDA_XC_TETER93;
      statusOFS << "XC_LDA_XC_TETER93  XCId = " << XCId_  << std::endl << std::endl;
      if( xc_func_init(&XCFuncType_, XCId_, nspin) != 0 ){
        ErrorHandling( "XC functional initialization error." );
      } 
      // Teter 93
      // S Goedecker, M Teter, J Hutter, Phys. Rev B 54, 1703 (1996) 
    }    
    else if( XCType_ == "XC_LDA_XC_PZ" )
    {
      XId_ = XC_LDA_X;
      CId_ = XC_LDA_C_PZ;
      statusOFS << "XC_LDA_XC_PZ  XId_ CId_ = " << XId_ << " " << CId_  << std::endl << std::endl;
      if( xc_func_init(&XFuncType_, XId_, nspin) != 0 ){
        ErrorHandling( "X functional initialization error." );
      }
      if( xc_func_init(&CFuncType_, CId_, nspin) != 0 ){
        ErrorHandling( "C functional initialization error." );
      } 
    }
    else if( XCType_ == "XC_GGA_XC_PBE" )
    {
      XId_ = XC_GGA_X_PBE;
      CId_ = XC_GGA_C_PBE;
      XCId_ = XC_GGA_X_PBE;
      statusOFS << "XC_GGA_XC_PBE  XId_ CId_ = " << XId_ << " " << CId_  << std::endl << std::endl;
      // Perdew, Burke & Ernzerhof correlation
      // JP Perdew, K Burke, and M Ernzerhof, Phys. Rev. Lett. 77, 3865 (1996)
      // JP Perdew, K Burke, and M Ernzerhof, Phys. Rev. Lett. 78, 1396(E) (1997)
      if( xc_func_init(&XFuncType_, XId_, nspin) != 0 ){
        ErrorHandling( "X functional initialization error." );
      }
      if( xc_func_init(&CFuncType_, CId_, nspin) != 0 ){
        ErrorHandling( "C functional initialization error." );
      }
    }
    else if( XCType_ == "XC_HYB_GGA_XC_HSE06" )
    {
      XCId_ = XC_HYB_GGA_XC_HSE06;
      XId_ = XC_GGA_X_PBE;
      CId_ = XC_GGA_C_PBE;
      statusOFS << "XC_HYB_GGA_XC_HSE06  XCId = " << XCId_  << std::endl << std::endl;
      if( xc_func_init(&XCFuncType_, XCId_, nspin) != 0 ){
        ErrorHandling( "XC functional initialization error." );
      }
      if( xc_func_init(&XFuncType_, XId_, nspin) != 0 ){
        ErrorHandling( "X functional initialization error." );
      }
      if( xc_func_init(&CFuncType_, CId_, nspin) != 0 ){
        ErrorHandling( "C functional initialization error." );
      }

      isHybrid_ = true;
      // J. Heyd, G. E. Scuseria, and M. Ernzerhof, J. Chem. Phys. 118, 8207 (2003) (doi: 10.1063/1.1564060)
      // J. Heyd, G. E. Scuseria, and M. Ernzerhof, J. Chem. Phys. 124, 219906 (2006) (doi: 10.1063/1.2204597)
      // A. V. Krukau, O. A. Vydrov, A. F. Izmaylov, and G. E. Scuseria, J. Chem. Phys. 125, 224106 (2006) (doi: 10.1063/1.2404663)
      //
      // This is the same as the "hse" functional in QE 5.1
    }
    else {
      ErrorHandling("Unrecognized exchange-correlation type");
    }
  }

  // Set up wavefunction filter options: useful for CheFSI in PWDFT, for example
  // Affects the MATVEC operations in MultSpinor
  if(esdfParam.PWSolver == "CheFSI")
    set_wfn_filter(esdfParam.PWDFT_Cheby_apply_wfn_ecut_filt, 1, esdfParam.ecutWavefunction);
  else
    set_wfn_filter(0, 0, esdfParam.ecutWavefunction);

  return ;
}         // -----  end of method KohnSham::Setup  ----- 

void
KohnSham::CalculatePseudoPotential ( PeriodTable &ptable, Fourier &fft, std::vector<IntNumVec> &grididx ){

  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

  bool usevloc = esdfParam.isUseVLocal;
  bool realspace = esdfParam.isUseRealSpace;

  Int ntotFine = domain_.NumGridTotalFine();
  Int numAtom = atomList_.size();
  Real vol = domain_.Volume();

  pseudo_.clear();
  if( realspace ) pseudo_.resize( numAtom );

  Real timeSta, timeEnd, timeSta1, timeEnd1;

  // Parallelism on atoms
  int numAtomBlocksize = numAtom  / mpisize;
  int numAtomLocal = numAtomBlocksize;
  if(mpirank < (numAtom % mpisize)){
    numAtomLocal = numAtomBlocksize + 1;
  }
  IntNumVec numAtomIdx( numAtomLocal );

  if (numAtomBlocksize == 0 ){
    for (Int i = 0; i < numAtomLocal; i++){
      numAtomIdx[i] = mpirank;
    }
  }
  else {
    if ( (numAtom % mpisize) == 0 ){
      for (Int i = 0; i < numAtomLocal; i++){
        numAtomIdx[i] = numAtomBlocksize * mpirank + i;
      }
    }
    else{
      for (Int i = 0; i < numAtomLocal; i++){
        if ( mpirank < (numAtom % mpisize) ){
          numAtomIdx[i] = (numAtomBlocksize + 1) * mpirank + i;
        }
        else{
          numAtomIdx[i] = (numAtomBlocksize + 1) * (numAtom % mpisize) + numAtomBlocksize * (mpirank - (numAtom % mpisize)) + i;
        }
      }
    }
  }

  IntNumVec numAtomMpirank( numAtom );

  if (numAtomBlocksize == 0 ){
    for (Int i = 0; i < numAtom; i++){
      numAtomMpirank[i] = i % mpisize;
    }
  }
  else {
    if ( (numAtom % mpisize) == 0 ){
      for (Int i = 0; i < numAtom; i++){
        numAtomMpirank[i] = i / numAtomBlocksize;
      }
    }
    else{
      for (Int i = 0; i < numAtom; i++){
        if ( i < (numAtom % mpisize) * (numAtomBlocksize + 1) ){
          numAtomMpirank[i] = i / (numAtomBlocksize + 1);
        }
        else{
          numAtomMpirank[i] = numAtom % mpisize + (i - (numAtom % mpisize) * (numAtomBlocksize + 1)) / numAtomBlocksize;
        }
      }
    }
  }

  GetTime( timeSta );

  Int nZion = 0;
  for (Int a=0; a<numAtom; a++) {
    Int atype  = atomList_[a].type;
    if( ptable.ptemap().find(atype) == ptable.ptemap().end() ){
      ErrorHandling( "Cannot find the atom type." );
    }
    nZion = nZion + ptable.Zion(atype);
  }

  if( !usevloc )
  {
    // For HGH PP, only real space method is implemented by now
    std::vector<DblNumVec> gridpos;
    UniformMeshFine ( domain_, gridpos );

    DblNumVec pseudoChargeLocal(ntotFine);
    SetValue( pseudoChargeLocal, 0.0 );

    for (Int i=0; i<numAtomLocal; i++) {
      int a = numAtomIdx[i];
      ptable.CalculatePseudoCharge( atomList_[a], domain_, 
          gridpos, pseudo_[a].pseudoCharge );

      // accumulate to the global vector
      IntNumVec &idx = pseudo_[a].pseudoCharge.first;
      DblNumMat &val = pseudo_[a].pseudoCharge.second;
      for (Int k=0; k<idx.m(); k++) 
        pseudoChargeLocal[idx(k)] += val(k, VAL);
    }

    SetValue( pseudoCharge_, 0.0 );
    MPI_Allreduce( pseudoChargeLocal.Data(), pseudoCharge_.Data(), ntotFine, MPI_DOUBLE, MPI_SUM, domain_.comm );

    for (Int a=0; a<numAtom; a++) {

      std::stringstream vStream;
      std::stringstream vStreamTemp;
      int vStreamSize;

      PseudoPot& pseudott = pseudo_[a]; 

      serialize( pseudott, vStream, NO_MASK );

      if (numAtomMpirank[a] == mpirank){
        vStreamSize = Size( vStream );
      }

      MPI_Bcast( &vStreamSize, 1, MPI_INT, numAtomMpirank[a], domain_.comm );

      std::vector<char> sstr;
      sstr.resize( vStreamSize );

      if (numAtomMpirank[a] == mpirank){
        vStream.read( &sstr[0], vStreamSize );
      }

      MPI_Bcast( &sstr[0], vStreamSize, MPI_BYTE, numAtomMpirank[a], domain_.comm );

      vStreamTemp.write( &sstr[0], vStreamSize );

      deserialize( pseudott, vStreamTemp, NO_MASK );

    }

    GetTime( timeEnd );

    Real sumrho = 0.0;
    for (Int i=0; i<ntotFine; i++) 
      sumrho += pseudoCharge_[i]; 

    sumrho *= vol / Real(ntotFine);

    // adjustment should be multiplicative
    Real fac = nZion / sumrho;
    for (Int i=0; i<ntotFine; i++) 
      pseudoCharge_(i) *= fac; 
  }    // ---- if( !usevloc ) ----
  else{
    if( realspace ){

      std::vector<DblNumVec> gridpos;
      UniformMeshFine ( domain_, gridpos );

      // Use the pseudocharge formulation
      DblNumVec pseudoChargeLocal( ntotFine );
      DblNumVec vLocalSRLocal( ntotFine );
      SetValue( pseudoChargeLocal, 0.0 );
      SetValue( vLocalSRLocal, 0.0 );
     
      for (Int i=0; i<numAtomLocal; i++) {
        int a = numAtomIdx[i];
        ptable.CalculateVLocal( atomList_[a], domain_, 
            gridpos, pseudo_[a].vLocalSR, pseudo_[a].pseudoCharge );

        // accumulate to the global vector
        {
          IntNumVec &idx = pseudo_[a].pseudoCharge.first;
          DblNumMat &val = pseudo_[a].pseudoCharge.second;
          for (Int k=0; k<idx.m(); k++) 
            pseudoChargeLocal[idx(k)] += val(k, VAL);
        }

        {
          IntNumVec &idx = pseudo_[a].vLocalSR.first;
          DblNumMat &val = pseudo_[a].vLocalSR.second;
          for (Int k=0; k<idx.m(); k++) 
            vLocalSRLocal[idx(k)] += val(k, VAL);
        }
      } //  for (i)

      SetValue( pseudoCharge_, 0.0 );
      SetValue( vLocalSR_, 0.0 );
      MPI_Allreduce( pseudoChargeLocal.Data(), pseudoCharge_.Data(), ntotFine, MPI_DOUBLE, MPI_SUM, domain_.comm );
      MPI_Allreduce( vLocalSRLocal.Data(), vLocalSR_.Data(), ntotFine, MPI_DOUBLE, MPI_SUM, domain_.comm );

      for (Int a=0; a<numAtom; a++) {

        std::stringstream vStream;
        std::stringstream vStreamTemp;
        int vStreamSize;

        PseudoPot& pseudott = pseudo_[a]; 

        serialize( pseudott, vStream, NO_MASK );

        if (numAtomMpirank[a] == mpirank){
          vStreamSize = Size( vStream );
        }

        MPI_Bcast( &vStreamSize, 1, MPI_INT, numAtomMpirank[a], domain_.comm );

        std::vector<char> sstr;
        sstr.resize( vStreamSize );

        if (numAtomMpirank[a] == mpirank){
          vStream.read( &sstr[0], vStreamSize );
        }

        MPI_Bcast( &sstr[0], vStreamSize, MPI_BYTE, numAtomMpirank[a], domain_.comm );

        vStreamTemp.write( &sstr[0], vStreamSize );

        deserialize( pseudott, vStreamTemp, NO_MASK );

      } // for (a)

      GetTime( timeEnd );

      Real sumrho = 0.0;
      for (Int i=0; i<ntotFine; i++) 
        sumrho += pseudoCharge_[i]; 
      sumrho *= vol / Real(ntotFine);

      // adjustment should be multiplicative
      Real fac = nZion / sumrho;
      for (Int i=0; i<ntotFine; i++) 
        pseudoCharge_(i) *= fac; 
    }    // ---- if( realspace ) ----
    else{
      // Search for the number of atom types and build a list of atom types
      SetValue( pseudoCharge_, 0.0 );
      SetValue( vLocalSR_, 0.0 );

      std::set<Int> atomTypeSet;
      for( Int a = 0; a < numAtom; a++ ){
        atomTypeSet.insert( atomList_[a].type );
      } 

      IntNumVec& idxDensity = fft.idxFineCutDensity;
      Int ntot = idxDensity.Size();

      DblNumVec vlocR( ntot );
      CpxNumVec vlocG( ntotFine );
      CpxNumVec ccvecLocal( ntot );
      CpxNumVec ccvec( ntot );
  
      SetValue( vlocG, Z_ZERO );

      for( std::set<Int>::iterator itype = atomTypeSet.begin();
        itype != atomTypeSet.end(); itype++ ){
        Int atype = *itype;
        Atom fakeAtom;
        fakeAtom.type = atype;
        fakeAtom.pos = domain_.posStart;

        SetValue( vlocR, 0.0 );
        SetValue( ccvecLocal, Z_ZERO );
        SetValue( ccvec, Z_ZERO );

        GetTime( timeSta1 );

        ptable.CalculateVLocal( fakeAtom, domain_, fft, vlocR );

        GetTime( timeEnd1 );

        // Compute the structure factor
        Complex* ikxPtr = fft.ikFine[0].Data();
        Complex* ikyPtr = fft.ikFine[1].Data();
        Complex* ikzPtr = fft.ikFine[2].Data();
        Real xx, yy, zz;
        Complex phase;

        GetTime( timeSta1 );
        for( Int k = 0; k < numAtomLocal; k++ ){
          int a = numAtomIdx[k];
          if( atomList_[a].type == atype ){
            xx = atomList_[a].pos[0];
            yy = atomList_[a].pos[1];
            zz = atomList_[a].pos[2];
            for( Int i = 0; i < ntot; i++ ){
              Int ig = idxDensity(i);
              phase = -(ikxPtr[ig] * xx + ikyPtr[ig] * yy + ikzPtr[ig] * zz);
              ccvecLocal(i) += std::exp( phase );
            } 
          } 
        } // for (a)
        MPI_Allreduce( ccvecLocal.Data(), ccvec.Data(), ntot, MPI_DOUBLE_COMPLEX,
            MPI_SUM, domain_.comm ); 
        GetTime( timeEnd1 );

        for( Int i = 0; i < ntot; i++ ){
          vlocG[idxDensity(i)] += vlocR[i] * ccvec(i);
        }
      } // for (itype)

      // Transfer back to the real space and add to vLocalSR_
      blas::Copy( ntotFine, vlocG.Data(), 1, fft.outputComplexVecFine.Data(), 1 );
      FFTWExecute ( fft, fft.backwardPlanFine );
      for( Int i = 0; i < ntotFine; i++ ){
        vLocalSR_[i] = fft.inputComplexVecFine[i].real() * 4 * PI;
      }
    }    // ---- if( realspace ) ----

    GetTime( timeEnd );
  }    // ---- if( !usevloc ) ----

  // Nonlocal projectors
  GetTime( timeSta );

  if( realspace ){

    std::vector<DblNumVec> gridpos;
    UniformMeshFine ( domain_, gridpos );

    Int cnt = 0; // the total number of PS used
    Int cntLocal = 0; // the total number of PS used

    for (Int i=0; i<numAtomLocal; i++) {
      int a = numAtomIdx[i];
      // Introduce the nonlocal pseudopotential on the fine grid.
      ptable.CalculateNonlocalPP( atomList_[a], domain_, gridpos,
          pseudo_[a].vnlList, pseudo_[a].vnlPhase ); 
      cntLocal = cntLocal + pseudo_[a].vnlList.size();
    }

    cnt = 0; // the total number of PS used
    MPI_Allreduce( &cntLocal, &cnt, 1, MPI_INT, MPI_SUM, domain_.comm );
    // Bcast vnlList
    for (Int a=0; a<numAtom; a++) {

      std::stringstream vStream1;
      std::stringstream vStream2;
      std::stringstream vStream1Temp;
      std::stringstream vStream2Temp;
      int vStream1Size, vStream2Size;

      std::vector<NonlocalPP>& vnlList = pseudo_[a].vnlList;

      serialize( vnlList, vStream1, NO_MASK );

      if (numAtomMpirank[a] == mpirank){
        vStream1Size = Size( vStream1 );
      }

      MPI_Bcast( &vStream1Size, 1, MPI_INT, numAtomMpirank[a], domain_.comm );

      std::vector<char> sstr1;
      sstr1.resize( vStream1Size );

      if (numAtomMpirank[a] == mpirank){
        vStream1.read( &sstr1[0], vStream1Size );
      }

      MPI_Bcast( &sstr1[0], vStream1Size, MPI_BYTE, numAtomMpirank[a], domain_.comm );

      vStream1Temp.write( &sstr1[0], vStream1Size );

      deserialize( vnlList, vStream1Temp, NO_MASK );
    }
  }    // ---- if( realspace ) ----
  else{
    // Use parallelism in Fourier grids rather than atoms
    std::set<Int> atomTypeSet;
    for( Int a = 0; a < numAtom; a++ ){
      atomTypeSet.insert( atomList_[a].type );
    }

    Domain& dm = fft.domain;
#ifdef _COMPLEX_
    IntNumVec& KpointIdx = dm.KpointIdx;
    Int nkLocal = KpointIdx.Size();
#else
    Int nkLocal = 1;
#endif

    Int ntotG;
    Point3 kpoint = Point3( 0.0, 0.0, 0.0 );
    Point3 kG     = Point3( 0.0, 0.0, 0.0 );

    vnlc_.resize( nkLocal );
    for( Int k = 0; k < nkLocal; k++ ){
#ifdef _COMPLEX_
      Int ik = KpointIdx(k);
      IntNumVec &idxc = fft.idxCoarseCut[k];
      kpoint = Point3( dm.klist[0][ik], dm.klist[1][ik], dm.klist[2][ik] );
#else
      IntNumVec &idxc = fft.idxCoarseCut;
      kpoint = Point3( 0.0, 0.0, 0.0 );
#endif 
      // Determine the matrix dimension to store PP data
      ntotG = grididx[k].Size();    

      Int numppTotal = 0;
      for( Int a = 0; a < numAtom; a++ ){
        numppTotal += ptable.CountNonlocalPP( atomList_[a] );
      }

      vnlc_[k].first.Resize( ntotG, numppTotal );
      vnlc_[k].second.Resize( numppTotal );
 
      Int cntpp = 0;
      for( std::set<Int>::iterator itype = atomTypeSet.begin();
        itype != atomTypeSet.end(); itype++ ){
        Int atype = *itype;
        Atom fakeAtom;
        fakeAtom.type = atype;

        DblNumVec weight;
        CpxNumMat vkb;
        CpxNumVec ccvec( ntotG );

        Int numpp = ptable.CountNonlocalPP( fakeAtom );
        ptable.CalculateNonlocalPP( fakeAtom, domain_, fft, grididx[k], k, kpoint, weight, vkb );
        // Compute the structure factor
#ifdef _COMPLEX_
        Complex* ikxPtr = fft.ik[0].Data();
        Complex* ikyPtr = fft.ik[1].Data();
        Complex* ikzPtr = fft.ik[2].Data();
#else
        Complex* ikxPtr = fft.ikR2C[0].Data();
        Complex* ikyPtr = fft.ikR2C[1].Data();
        Complex* ikzPtr = fft.ikR2C[2].Data();
#endif
        Real xx, yy, zz;
        Complex phase;

        for( Int a = 0; a < numAtom; a++ ){
          if( atomList_[a].type == atype ){
            xx = atomList_[a].pos[0];
            yy = atomList_[a].pos[1];
            zz = atomList_[a].pos[2];
            for( Int i = 0; i < ntotG; i++ ){
              Int ig = idxc(grididx[k](i));      
              kG = ( kpoint + Point3( ikxPtr[ig].imag(), ikyPtr[ig].imag(),
                  ikzPtr[ig].imag() ) );      
              phase = Complex(0.0, -(kG[0] * xx + kG[1] * yy + kG[2] * zz));
              ccvec(i) = std::exp( phase );
            }

            for( Int np = 0; np < numpp; np++, cntpp++ ){
#ifdef _COMPLEX_
              for( Int i = 0; i < ntotG; i++ ){
                vnlc_[k].first(i, cntpp) = vkb(i, np) * ccvec(i);
              }
#else
              if( grididx[k].m() > 0 ){
                if( grididx[k][0] == 0 ){           
                  vnlc_[k].first(0, cntpp) = vkb(0, np) * ccvec(0);
                }
                else{
                  vnlc_[k].first(0, cntpp) = vkb(0, np) * ccvec(0) * std::sqrt(2.0);
                }
              }

              for( Int i = 1; i < ntotG; i++ ){
                vnlc_[k].first(i, cntpp) = vkb(i, np) * ccvec(i) * std::sqrt(2.0);
              }
#endif            
              vnlc_[k].second(cntpp) = weight(np);
            }
          }
        } // for (a)
      } // for (k)
    } // for (itype)
  }    // ---- if( realspace ) ----
#ifdef _COMPLEX_
  // Bcast vnlPhase if the crystal is calculated with k-ponits
  if( realspace ){
    for (Int a=0; a<numAtom; a++) {

      std::stringstream vStream1;
      std::stringstream vStream2;
      std::stringstream vStream1Temp;
      std::stringstream vStream2Temp;
      int vStream1Size, vStream2Size;

      std::vector<CpxNumVec>& vnlPhase = pseudo_[a].vnlPhase;

      serialize( vnlPhase, vStream1, NO_MASK );

      if (numAtomMpirank[a] == mpirank){
        vStream1Size = Size( vStream1 );
      }

      MPI_Bcast( &vStream1Size, 1, MPI_INT, numAtomMpirank[a], domain_.comm );

      std::vector<char> sstr1;
      sstr1.resize( vStream1Size );

      if (numAtomMpirank[a] == mpirank){
        vStream1.read( &sstr1[0], vStream1Size );
      }

      MPI_Bcast( &sstr1[0], vStream1Size, MPI_BYTE, numAtomMpirank[a], domain_.comm );

      vStream1Temp.write( &sstr1[0], vStream1Size );

      deserialize( vnlPhase, vStream1Temp, NO_MASK );
    }
  }

  // Calculate coefMat if SOC is included
  if( domain_.SpinOrbitCoupling == true ){
    if( realspace ){
      for (Int i=0; i<numAtomLocal; i++) {
        int a = numAtomIdx[i];
        ptable.CalculateCoefSOC( atomList_[a], pseudo_[a].coefMat );
      }

      for (Int a=0; a<numAtom; a++) {

        std::stringstream vStream1;
        std::stringstream vStream2;
        std::stringstream vStream1Temp;
        std::stringstream vStream2Temp;
        int vStream1Size, vStream2Size;

        CpxNumTns& coefMat = pseudo_[a].coefMat;

        serialize( coefMat, vStream1, NO_MASK );

        if (numAtomMpirank[a] == mpirank){
          vStream1Size = Size( vStream1 );
        }

        MPI_Bcast( &vStream1Size, 1, MPI_INT, numAtomMpirank[a], domain_.comm );

        std::vector<char> sstr1;
        sstr1.resize( vStream1Size );

        if (numAtomMpirank[a] == mpirank){
          vStream1.read( &sstr1[0], vStream1Size );
        }

        MPI_Bcast( &sstr1[0], vStream1Size, MPI_BYTE, numAtomMpirank[a], domain_.comm );

        vStream1Temp.write( &sstr1[0], vStream1Size );

        deserialize( coefMat, vStream1Temp, NO_MASK );
      }
    }
    else{
      std::set<Int> atomTypeSet;
      for( Int a = 0; a < numAtom; a++ ){
        atomTypeSet.insert( atomList_[a].type );
      }

      coef_.resize( numAtom );

      Int na = 0;
      for( std::set<Int>::iterator itype = atomTypeSet.begin();
        itype != atomTypeSet.end(); itype++ ){
        Int atype = *itype;
        Atom fakeAtom;
        fakeAtom.type = atype;

        CpxNumTns coefTemp;
        ptable.CalculateCoefSOC( fakeAtom, coefTemp );

        Real fac = std::pow(4*PI, 2) / fft.domain.Volume();
        blas::Scal( coefTemp.Size(), fac, coefTemp.Data(), 1 );

        for( Int a = 0; a < numAtom; a++ ){
          if( atomList_[a].type == atype ){
            coef_[na++] = coefTemp;
          }
        }
      }
    }
  } // ------ End of if( domain_.SpinOrbitCoupling == true ) ------
#endif
  GetTime( timeEnd );
  
  // Calculate other atomic related energies and forces, such as self
  // energy, short range repulsion energy and VdW energies.
  
  this->CalculateIonSelfEnergyAndForce( ptable, fft );

  this->CalculateVdwEnergyAndForce();

  Eext_ = 0.0;
  forceext_.Resize( atomList_.size(), DIM );
  SetValue( forceext_, 0.0 );

  return ;
}         // -----  end of method KohnSham::CalculatePseudoPotential ----- 

void KohnSham::CalculateAtomDensity ( PeriodTable &ptable, Fourier &fft ){

  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

  if( esdfParam.pseudoType == "HGH" ){
    ErrorHandling("HGH pseudopotential does not yet support the computation of atomic density!");
  }
  
  bool realspace = esdfParam.isUseRealSpace;
 
  IntNumVec& idxDensity = fft.idxFineCutDensity;

  Int ntotFine = domain_.NumGridTotalFine();
  Int ntot = (realspace == true) ? ntotFine : idxDensity.Size();
  Int numAtom = atomList_.size();
  Real vol = domain_.Volume();

  std::vector<DblNumVec> gridpos;
  UniformMeshFine ( domain_, gridpos );

  // The number of electrons for normalization purpose. 
  Int nelec = 0;
  for (Int a=0; a<numAtom; a++) {
    Int atype  = atomList_[a].type;
    if( ptable.ptemap().find(atype) == ptable.ptemap().end() ){
      ErrorHandling( "Cannot find the atom type." );
    }
    nelec = nelec + ptable.Zion(atype);
  }
  // add the extra electron
  nelec = nelec + esdfParam.extraElectron;
  if( nelec % 2 != 0 && esdfParam.spinType == 1 ){
    ErrorHandling( "This is spin-restricted calculation. nelec should be even." );
  }
  
  int numAtomBlocksize = numAtom  / mpisize;
  int numAtomLocal = numAtomBlocksize;
  if(mpirank < (numAtom % mpisize)){
    numAtomLocal = numAtomBlocksize + 1;
  }
  IntNumVec numAtomIdx( numAtomLocal );

  if (numAtomBlocksize == 0 ){
    for (Int i = 0; i < numAtomLocal; i++){
      numAtomIdx[i] = mpirank;
    }
  }
  else {
    if ( (numAtom % mpisize) == 0 ){
      for (Int i = 0; i < numAtomLocal; i++){
        numAtomIdx[i] = numAtomBlocksize * mpirank + i;
      }
    }
    else{
      for (Int i = 0; i < numAtomLocal; i++){
        if ( mpirank < (numAtom % mpisize) ){
          numAtomIdx[i] = (numAtomBlocksize + 1) * mpirank + i;
        }
        else{
          numAtomIdx[i] = (numAtomBlocksize + 1) * (numAtom % mpisize) + numAtomBlocksize * (mpirank - (numAtom % mpisize)) + i;
        }
      }
    }
  }

  Real timeSta, timeEnd, timeSta1, timeEnd1;
  GetTime( timeSta );

  // Search for the number of atom types and build a list of atom types
  std::set<Int> atomTypeSet;
  for( Int a = 0; a < numAtom; a++ ){
    atomTypeSet.insert( atomList_[a].type );
  } // for (a)

  // For each atom type, construct the atomic pseudocharge within the
  // cutoff radius starting from the origin in the real space, and
  // construct the structure factor

  // Origin-centered atomDensity in the real space and Fourier space

  DblNumVec atomDensityR( ntot );
  CpxNumMat atomDensityG( ntotFine, numDensityComponent_ );
  CpxNumMat ccvecLocal( ntot, numDensityComponent_ );
  CpxNumMat ccvec( ntot, numDensityComponent_ );

  SetValue( atomDensityG, Z_ZERO );

  for( std::set<Int>::iterator itype = atomTypeSet.begin();
    itype != atomTypeSet.end(); itype++ ){
    Int atype = *itype;
    Atom fakeAtom;
    fakeAtom.type = atype;
    fakeAtom.pos = domain_.posStart;

    SetValue( atomDensityR, 0.0 );
    SetValue( ccvecLocal, Z_ZERO );
    SetValue( ccvec, Z_ZERO );

    GetTime( timeSta1 );    

    if( realspace )
      ptable.CalculateAtomDensity( fakeAtom, domain_, gridpos, atomDensityR );
    else
      ptable.CalculateAtomDensity( fakeAtom, domain_, fft, atomDensityR );

    GetTime( timeEnd1 );

    // Compute the structure factor
    Complex* ikxPtr = fft.ikFine[0].Data();
    Complex* ikyPtr = fft.ikFine[1].Data();
    Complex* ikzPtr = fft.ikFine[2].Data();
    Real xx, yy, zz;
    Complex phase;

    Point3 angle( 0.0, 0.0, 0.0 );
    Real amag, theta, phi;

    GetTime( timeSta1 );
    for( Int k = 0; k < numAtomLocal; k++ ){
      int a = numAtomIdx[k];
      if( atomList_[a].type == atype ){
        xx = atomList_[a].pos[0];
        yy = atomList_[a].pos[1];
        zz = atomList_[a].pos[2];
        for( Int i = 0; i < ntot; i++ ){
          Int ig = (realspace == true ) ? i : idxDensity(i);
          phase = -(ikxPtr[ig] * xx + ikyPtr[ig] * yy + ikzPtr[ig] * zz);
          // total density
          ccvecLocal(i,RHO) += std::exp( phase );

          if( numDensityComponent_ == 2 ){
            // spin density
            amag = atomList_[a].mag[0];
            ccvecLocal(i,1) += std::exp( phase ) * amag;
          }
          
          if( numDensityComponent_ == 4 ){
            // The norm and direction of atomic magnetic moments
            amag = atomList_[a].mag[0];
            if( esdfParam.isParallel ){
              angle = domain_.spinaxis;
            }
            else{
              theta = atomList_[a].mag[1] * PI / 180;
              phi = atomList_[a].mag[2] *PI / 180;

              angle[0] = sin( theta ) * cos( phi );
              angle[1] = sin( theta ) * sin( phi );
              angle[2] = cos( theta );
            }
            
            for( Int is = 1; is < numDensityComponent_; is++ ){
              ccvecLocal(i,is) += std::exp( phase ) * amag * angle[is-1];
            }
          }  // ---- end of if( numDensityComponent_ == 4 ) ----
        }  // for (i)
      }  // ---- end of if( atomList_[a].type == atype ) ----
    }  // for (a)
    MPI_Allreduce( ccvecLocal.Data(), ccvec.Data(), ntot*numDensityComponent_, 
        MPI_DOUBLE_COMPLEX, MPI_SUM, domain_.comm );
    GetTime( timeEnd1 );

    if( realspace ){
      // Transfer the atomic charge from real space to Fourier space, and
      // multiply with the structure factor
      for( Int i = 0; i < ntotFine; i++ ){
        fft.inputComplexVecFine[i] = Complex( atomDensityR[i], 0.0 ); 
      }
 
      FFTWExecute ( fft, fft.forwardPlanFine );      

      for( Int i = 0; i < ntotFine; i++ ){
        // Make it smoother: AGGREESIVELY truncate components beyond EcutWavefunction
        if( fft.gkkFine[i] < esdfParam.ecutWavefunction * 4.0 ){
          for( Int is = 0; is < numDensityComponent_; is++ ){
            atomDensityG(i,is) += fft.outputComplexVecFine[i] * ccvec(i,is);
          }
        }
      }
    }  // realspace
    else{
      for( Int i = 0; i < ntot; i++ ){
        Int ig = idxDensity(i);
        for( Int is = 0; is < numDensityComponent_; is++ ){
          atomDensityG(ig,is) += atomDensityR[i] * ccvec(i,is); 
        }
      }
    }
  } // for (itype)

  // Transfer back to the real space and add to atomDensity_ 
  {
    for( Int is = 0; is < numDensityComponent_; is++ ){ 
      for(Int i = 0; i < ntotFine; i++){
        fft.outputComplexVecFine[i] = atomDensityG(i,is);
      }
  
      FFTWExecute ( fft, fft.backwardPlanFine );

      for( Int i = 0; i < ntotFine; i++ ){
        atomDensity_(i,is) = fft.inputComplexVecFine[i].real();
      }
    }
  }

  GetTime( timeEnd );

  Real sumrho = 0.0;
  Real neg_sumrho = 0.0;

  for (Int i=0; i<ntotFine; i++){ 
    sumrho += atomDensity_(i,RHO); 
    if( atomDensity_(i,RHO) < 0.0 )
      neg_sumrho += std::abs(atomDensity_(i,RHO));
  }

  sumrho *= vol / Real(ntotFine);
  neg_sumrho *= vol / Real(ntotFine);  

  // adjustment should be multiplicative
  Real fac = nelec / sumrho;
  for (Int i=0; i<ntotFine; i++){
    for (Int is = 0; is < numDensityComponent_; is++ ){
      atomDensity_(i,is) *= fac; 
    }
  }

  return ;
}         // -----  end of method KohnSham::CalculateAtomDensity  ----- 

void
KohnSham::CalculateGradDensity ( Fourier& fft )
{
  Int ntotFine  = fft.domain.NumGridTotalFine();
  Real vol  = domain_.Volume();
  
  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);
  int dmCol = DIM;
  int dmRow = mpisize / dmCol;

  // Compute the derivative of the Density via Fourier
  CpxNumVec cpxVec( ntotFine );

  // The rotated density, the density needs to be rotated to spin-up and spin-down
  // components only for spin-noncollinear case
  DblNumMat rotDensity( ntotFine, numSpinorComponent_ ); SetValue( rotDensity, D_ZERO );
  if( numDensityComponent_ == 1 ){
    blas::Copy( ntotFine*numSpinorComponent_, density_.Data(), 1, rotDensity.Data(), 1 ); 
  }
  else{
    blas::Copy( ntotFine*numSpinorComponent_, spindensity_.Data(), 1, rotDensity.Data(), 1 );
  }

  Int d;
  for( Int is = 0; is < numSpinorComponent_; is++ ){ 
    // Transform rotDensity to reciprocal space first
    for( Int i = 0; i < ntotFine; i++ ){
      fft.inputComplexVecFine(i) = Complex( rotDensity(i,is), 0.0 ); 
    }

    FFTWExecute ( fft, fft.forwardPlanFine );

    blas::Copy( ntotFine, fft.outputComplexVecFine.Data(), 1,
        cpxVec.Data(), 1 );
  
    if( mpisize < DIM ){ // mpisize < 3
      for( d = 0; d < DIM; d++ ){
        DblNumMat& gradDensity = gradDensity_[d];
        CpxNumVec& ik = fft.ikFine[d];
        for( Int i = 0; i < ntotFine; i++ ){
          if( fft.gkkFine(i) == 0 || fft.gkkFine(i) > esdfParam.ecutWavefunction * 4.0 ){
            fft.outputComplexVecFine(i) = Z_ZERO;
          }
          else{
            fft.outputComplexVecFine(i) = cpxVec(i) * ik(i); 
          }
        }

        FFTWExecute ( fft, fft.backwardPlanFine );

        for( Int i = 0; i < ntotFine; i++ ){
          gradDensity(i, is) = fft.inputComplexVecFine(i).real();
        }
      } // for (d)
    } // mpisize < 3
    else { // mpisize > 3
      for( d = 0; d < DIM; d++ ){
        DblNumMat& gradDensity = gradDensity_[d];
        if ( d == mpirank % dmCol ){ 
          CpxNumVec& ik = fft.ikFine[d];
          for( Int i = 0; i < ntotFine; i++ ){
            if( fft.gkkFine(i) == 0 || fft.gkkFine(i) > esdfParam.ecutWavefunction * 4.0 ){
              fft.outputComplexVecFine(i) = Z_ZERO;
            }
            else{
              fft.outputComplexVecFine(i) = cpxVec(i) * ik(i); 
            }
          }

          FFTWExecute ( fft, fft.backwardPlanFine );

          for( Int i = 0; i < ntotFine; i++ ){
            gradDensity(i, is) = fft.inputComplexVecFine(i).real();
          }
        } // d == mpirank
      } // for (d)

      for( d = 0; d < DIM; d++ ){
        DblNumMat& gradDensity = gradDensity_[d];
        MPI_Bcast( gradDensity.VecData(is), ntotFine, MPI_DOUBLE, d, rowComm_ );
      } // for (d)
    } // mpisize > 3
  } // for (is)  

  return ;
}         // -----  end of method KohnSham::CalculateGradDensity  ----- 

void
KohnSham::CalculateXC    ( Real &val, Fourier& fft )
{
  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);
  int dmCol = DIM;
  int dmRow = mpisize / dmCol;
  
  bool uselibxc = esdfParam.isUseLibxc;

  Int ntot = domain_.NumGridTotalFine();
  Real vol = domain_.Volume();

  Int ntotBlocksize = ntot / mpisize;
  Int ntotLocal = ntotBlocksize;
  if(mpirank < (ntot % mpisize)){
    ntotLocal = ntotBlocksize + 1;
  } 
  IntNumVec localSize(mpisize);
  IntNumVec localSizeDispls(mpisize);
  SetValue( localSize, 0 );
  SetValue( localSizeDispls, 0 );
  MPI_Allgather( &ntotLocal, 1, MPI_INT, localSize.Data(), 1, MPI_INT, domain_.comm );

  for (Int i = 1; i < mpisize; i++ ){
    localSizeDispls[i] = localSizeDispls[i-1] + localSize[i-1];
  }

  Real epsRho = 1e-10, epsGRho = 1e-10;
  bool badpoint;

  Real timeSta, timeEnd;

  Real timeFFT = 0.00;
  Real timeOther = 0.00;

  if( XCId_ == XC_LDA_XC_TETER93 ) 
  {
    if( numSpinorComponent_ == 1 ){
      xc_lda_exc_vxc( &XCFuncType_, ntot, density_.Data(), 
        epsxc_.Data(), vxc_.Data() );
    }
    else{
      // Transform density format to use libxc
      DblNumVec density_lxc( numSpinorComponent_*ntot );
      DblNumVec vxc_lxc( numSpinorComponent_*ntot );

      SetValue( density_lxc, 0.0 ); SetValue( vxc_lxc, 0.0 );

      blas::Copy( ntot, spindensity_.VecData(UP), 1, &density_lxc[0], 2);
      blas::Copy( ntot, spindensity_.VecData(DN), 1, &density_lxc[1], 2);
      xc_lda_exc_vxc( &XCFuncType_, ntot, density_lxc.Data(),
        epsxc_.Data(), vxc_lxc.Data() );
      blas::Copy( ntot, &vxc_lxc[0], 2, vxc_.VecData(UP), 1 );
      blas::Copy( ntot, &vxc_lxc[1], 2, vxc_.VecData(DN), 1 );
    }
    // Modify "bad points"   
    for( Int i = 0; i < ntot; i++ ){
      badpoint = ( density_(i,RHO) < epsRho );
      if( badpoint ){
        epsxc_(i) = 0.0;
        for( Int is = 0; is < numDensityComponent_; is++ ){
          vxc_( i, is ) = 0.0;
        }
      }
    }
  }// XC_FAMILY_LDA: Teter93
  else if( XId_ == XC_LDA_X && CId_ == XC_LDA_C_PZ ){
    if( numSpinorComponent_ == 1 ){
      DblNumVec vx_lxc( ntot );
      DblNumVec vc_lxc( ntot );
      DblNumVec epsx( ntot );
      DblNumVec epsc( ntot );

      SetValue( vx_lxc, 0.0 );
      SetValue( vc_lxc, 0.0 );
      SetValue( epsx, 0.0 );
      SetValue( epsc, 0.0 );
  
      xc_func_set_dens_threshold( &XFuncType_, epsRho );
      xc_lda_exc_vxc( &XFuncType_, ntot, density_.Data(),
        epsx.Data(), vx_lxc.Data() );

      xc_func_set_dens_threshold( &CFuncType_, epsRho );
      xc_lda_exc_vxc( &CFuncType_, ntot, density_.Data(),
        epsc.Data(), vc_lxc.Data() );

      blas::Copy( ntot, &vx_lxc[0], 1, vxc_.Data(), 1 );
      blas::Copy( ntot, epsx.Data(), 1, epsxc_.Data(), 1 );
      blas::Axpy( ntot, 1.0, &vc_lxc[0], 1, vxc_.Data(), 1 );
      blas::Axpy( ntot, 1.0, epsc.Data(), 1, epsxc_.Data(), 1 );
    }
    else{ 
      DblNumVec density_lxc( numSpinorComponent_*ntot );
      DblNumVec vx_lxc( numSpinorComponent_*ntot );
      DblNumVec vc_lxc( numSpinorComponent_*ntot );
      DblNumVec epsx( ntot );
      DblNumVec epsc( ntot );

      SetValue( density_lxc, 0.0 );
      SetValue( vx_lxc, 0.0 );
      SetValue( vc_lxc, 0.0 );
      SetValue( epsx, 0.0 );
      SetValue( epsc, 0.0 );

      blas::Copy( ntot, spindensity_.VecData(UP), 1, &density_lxc[0], 2);
      blas::Copy( ntot, spindensity_.VecData(DN), 1, &density_lxc[1], 2);

      xc_func_set_dens_threshold( &XFuncType_, epsRho );
      xc_lda_exc_vxc( &XFuncType_, ntot, density_lxc.Data(),
        epsx.Data(), vx_lxc.Data() );

      xc_func_set_dens_threshold( &CFuncType_, epsRho );
      xc_lda_exc_vxc( &CFuncType_, ntot, density_lxc.Data(),
        epsc.Data(), vc_lxc.Data() );

      blas::Copy( ntot, &vx_lxc[0], 2, vxc_.VecData(UP), 1 );
      blas::Copy( ntot, &vx_lxc[1], 2, vxc_.VecData(DN), 1 );
      blas::Copy( ntot, epsx.Data(), 1, epsxc_.Data(), 1 );
      blas::Axpy( ntot, 1.0, &vc_lxc[0], 2, vxc_.VecData(UP), 1 );
      blas::Axpy( ntot, 1.0, &vc_lxc[1], 2, vxc_.VecData(DN), 1 );
      blas::Axpy( ntot, 1.0, epsc.Data(), 1, epsxc_.Data(), 1 );
    }
 
    for( Int i = 0; i < ntot; i++ ){
      badpoint = ( density_(i,RHO) < epsRho );
      if( badpoint ){
        epsxc_(i) = 0.0;
        for( Int is = 0; is < numDensityComponent_; is++ ){
          vxc_( i, is ) = 0.0;
        }
      }
    }

    SetValue( exc_, 0.0 );
    for( Int i = 0; i < ntot; i++ )
      exc_[i] = epsxc_[i] * density_(i,RHO);
    
  }// XC_FAMILY_LDA: PZ
  else if( ( XId_ == XC_GGA_X_PBE ) && ( CId_ == XC_GGA_C_PBE ) || ( XCId_ == XC_HYB_GGA_XC_HSE06 ) ){
    Int numGrad2 = ( numDensityComponent_ > 1 ? 3:1 );

    DblNumVec gradDensity( ntotLocal*numGrad2 );
    DblNumMat& gradDensity0 = gradDensity_[0];
    DblNumMat& gradDensity1 = gradDensity_[1];
    DblNumMat& gradDensity2 = gradDensity_[2];

    GetTime( timeSta );
    for( Int i = 0; i < ntotLocal; i++ ){
      Int ii = i + localSizeDispls(mpirank);

      if( numSpinorComponent_ == 1 ){
        gradDensity[i] = gradDensity0(ii, RHO) * gradDensity0(ii, RHO)
            + gradDensity1(ii, RHO) * gradDensity1(ii, RHO)
                + gradDensity2(ii, RHO) * gradDensity2(ii, RHO);          
      }
      else{
        // For spin-unrestricted and spin-noncollinear case, the gradDensity
        // dots between each kind of spin are needed
        gradDensity[3*i+UPUP] = gradDensity0(ii, UP) * gradDensity0(ii, UP)
            + gradDensity1(ii, UP) * gradDensity1(ii, UP)
                + gradDensity2(ii, UP) * gradDensity2(ii, UP);

        gradDensity[3*i+UPDN] = gradDensity0(ii, UP) * gradDensity0(ii, DN)
            + gradDensity1(ii, UP) * gradDensity1(ii, DN)
              + gradDensity2(ii, UP) * gradDensity2(ii, DN);

        gradDensity[3*i+DNDN] = gradDensity0(ii, DN) * gradDensity0(ii, DN)
            + gradDensity1(ii, DN) * gradDensity1(ii, DN)
              + gradDensity2(ii, DN) * gradDensity2(ii, DN); 
      }
    }

    GetTime( timeEnd );   

    DblNumVec densityTemp;
    densityTemp.Resize( ntotLocal*numSpinorComponent_ );
    
    if( numSpinorComponent_ == 1 ){
      for( Int i = 0; i < ntotLocal; i++ ){
        densityTemp[i] = density_(i + localSizeDispls(mpirank), RHO);
      }
    }
    else{
      for( Int i = 0; i < ntotLocal; i++ ){
        densityTemp[2*i+UP] = spindensity_(i + localSizeDispls(mpirank), UP);
	      densityTemp[2*i+DN] = spindensity_(i + localSizeDispls(mpirank), DN);
      }
    }
 
    DblNumVec excTemp( ntotLocal );
    DblNumVec vxc1Temp( ntotLocal*numSpinorComponent_ ); 
    DblNumVec vxc2Temp( ntotLocal*numGrad2 );
    SetValue( excTemp, 0.0 );
    SetValue( vxc1Temp, 0.0 );
    SetValue( vxc2Temp, 0.0 );

    Real epsRhoGGA = 1.0e-6;
    Real epsGrhoGGA = 1.0e-10;

    if( uselibxc )
    {
      if( ( XId_ == XC_GGA_X_PBE ) && ( CId_ == XC_GGA_C_PBE ) && (XCId_ == XC_GGA_X_PBE) ){
        DblNumVec vx1( ntotLocal*numSpinorComponent_ );
        DblNumVec vx2( ntotLocal*numGrad2 );
        DblNumVec vc1( ntotLocal*numSpinorComponent_ );
        DblNumVec vc2( ntotLocal*numGrad2 );
        DblNumVec epsx( ntotLocal );
        DblNumVec epsc( ntotLocal );
        DblNumVec ex( ntotLocal );
        DblNumVec ec( ntotLocal );

        SetValue( vx1, 0.0 );
        SetValue( vx2, 0.0 );
        SetValue( vc1, 0.0 );
        SetValue( vc2, 0.0 );
        SetValue( epsx, 0.0 );
        SetValue( epsc, 0.0 );
        SetValue( ex, 0.0 );
        SetValue( ec, 0.0 );

        xc_func_set_dens_threshold( &XFuncType_, epsRhoGGA );
        xc_gga_exc_vxc( &XFuncType_, ntotLocal, densityTemp.Data(),
            gradDensity.Data(), epsx.Data(), vx1.Data(), vx2.Data() );

        xc_func_set_dens_threshold( &CFuncType_, epsGrhoGGA );
        xc_gga_exc_vxc( &CFuncType_, ntotLocal, densityTemp.Data(),
            gradDensity.Data(), epsc.Data(), vc1.Data(), vc2.Data() );

        for( Int i = 0; i < ntotLocal; i++ ){
          if( numSpinorComponent_ == 1 ){
            ex[i] = epsx[i] * densityTemp[i];
          }
          else{
            ex[i] = epsx[i] * ( densityTemp[2*i+UP] + densityTemp[2*i+DN] );
          }
        }

        // Delete bad points
        if( numSpinorComponent_ == 1 ){
          for( Int i = 0; i < ntotLocal; i++ ){
            ec[i] = epsc[i] * densityTemp[i];
            if( abs(densityTemp[i]) > epsRhoGGA && abs(gradDensity[i]) > epsGrhoGGA ){

            }
            else{
              vx2[i] = 0.0;
              vc2[i] = 0.0;
            }
          }
        }
        else{
          Real sign_up, sign_dw;
          for( Int i = 0; i < ntotLocal; i++ ){
            sign_up = 1.0; sign_dw = 1.0;
            if( densityTemp[2*i+UP] < epsRhoGGA || std::sqrt( gradDensity(3*i+UPUP) )
                < epsGrhoGGA )
              sign_up = 0.0;

            if( densityTemp[2*i+DN] < epsRhoGGA || std::sqrt( gradDensity(3*i+DNDN) )
                < epsGrhoGGA )
              sign_dw = 0.0;

            ec[i] = epsc[i] * ( densityTemp[2*i+UP]*sign_up + densityTemp[2*i+DN]*sign_dw );
            vc1[2*i+UP] *= sign_up;
            vc1[2*i+DN] *= sign_dw;
            vc2[3*i+UPUP] *= sign_up;
            vc2[3*i+UPDN] *= (sign_up*sign_dw);
            vc2[3*i+DNDN] *= sign_dw;
          }
        }

        // exc = ex + ec
        blas::Copy( ntotLocal, ex.Data(), 1, excTemp.Data(), 1 );
        blas::Axpy( ntotLocal, 1.0, ec.Data(), 1, excTemp.Data(), 1 );
        // vxc = vx + vc
        blas::Copy( ntotLocal*numSpinorComponent_, vx1.Data(), 1, vxc1Temp.Data(), 1 );
        blas::Axpy( ntotLocal*numSpinorComponent_, 1.0, vc1.Data(), 1, vxc1Temp.Data(), 1 );
        blas::Copy( ntotLocal*numGrad2, vx2.Data(), 1, vxc2Temp.Data(), 1 );
        blas::Axpy( ntotLocal*numGrad2, 1.0, vc2.Data(), 1, vxc2Temp.Data(), 1 );
      }    // ---- if( ( XId_ == XC_GGA_X_PBE ) && ( CId_ == XC_GGA_C_PBE ) && (XId_ == XC_GGA_X_PBE) ) ----
      else{
        // For hybrid functional, exchange and correlation parts are calculated together
        // The benchmark results compared to QE are worse if the libxc library is 
        // called directly with id = 428 
        DblNumVec epsxcTemp( ntotLocal );
        SetValue( epsxcTemp, 0.0 );

        xc_func_set_dens_threshold( &XCFuncType_, epsGrhoGGA );
        xc_gga_exc_vxc( &XCFuncType_, ntotLocal, densityTemp.Data(),
            gradDensity.Data(), epsxcTemp.Data(), vxc1Temp.Data(), vxc2Temp.Data() );
        // Delete bad points
        if( numSpinorComponent_ == 1 ){
        for( Int i = 0; i < ntotLocal; i++ ){
          excTemp[i] = epsxcTemp[i] * densityTemp[i];
        } 
        } 
        else{
          Real sign_up, sign_dw;
          for( Int i = 0; i < ntotLocal; i++ ){
            sign_up = 1.0; sign_dw = 1.0;
            if( densityTemp[2*i+UP] < epsRhoGGA || std::sqrt( gradDensity(3*i+UPUP) )
                < epsGrhoGGA )
              sign_up = 0.0; 
              
            if( densityTemp[2*i+DN] < epsRhoGGA || std::sqrt( gradDensity(3*i+DNDN) )
                < epsGrhoGGA )
              sign_dw = 0.0; 
              
            excTemp[i] = epsxcTemp[i] * ( densityTemp[2*i+UP]*sign_up + densityTemp[2*i+DN]*sign_dw );
            vxc1Temp[2*i+UP] *= sign_up;
            vxc1Temp[2*i+DN] *= sign_dw;
            vxc2Temp[3*i+UPUP] *= sign_up;
            vxc2Temp[3*i+UPDN] *= (sign_up*sign_dw);
            vxc2Temp[3*i+DNDN] *= sign_dw;
          } 
        } 
      }
    }    // ---- if( uselibxc ) ----
    else
    {
      bool isHybrid = ( XCId_ == XC_HYB_GGA_XC_HSE06 );

      // Call internal functions to calculate XC potential
      DblNumVec vx1( ntotLocal*numSpinorComponent_ ); 
      DblNumVec vx2( ntotLocal*numGrad2 );
      DblNumVec vc1( ntotLocal*numSpinorComponent_ ); 
      DblNumVec vc2( ntotLocal*numGrad2 );
      DblNumVec epsx( ntotLocal ); 
      DblNumVec epsc( ntotLocal ); 
      DblNumVec ex( ntotLocal );
      DblNumVec ec( ntotLocal );

      SetValue( vx1, 0.0 );
      SetValue( vx2, 0.0 );
      SetValue( vc1, 0.0 );
      SetValue( vc2, 0.0 );
      SetValue( epsx, 0.0 );
      SetValue( epsc, 0.0 );
      SetValue( ex, 0.0 );
      SetValue( ec, 0.0 );
    
      if( numSpinorComponent_ == 1 ){
        Real rho, absrho, grho2, rs, vx, ux, vc, uc;
        Real v1gcx, v2gcx, ugcx;
        Real v1gcc, v2gcc, ugcc;

        for( Int i = 0; i < ntotLocal; i++ ){
          rho = densityTemp(i);
          absrho = std::abs( rho );
          grho2 = gradDensity(i);

          if( absrho > 1e-10 ){
            rs = std::pow(3.0 / 4.0 / PI / absrho, 1.0 / 3.0);
            VExchange_sla(rs, ux, vx);
            VCorrelation_pw(rs, uc, vc);
            
            epsx(i) = epsx(i) + ux;
            epsc(i) = epsc(i) + uc;
            vxc1Temp(i) = vxc1Temp(i) + vx + vc;     
          }

          if( absrho > 1e-6 & grho2 > 1e-10 ){
            VGCExchange_pbx(absrho, grho2, ugcx, v1gcx, v2gcx);
            VGCCorrelation_pbc(absrho, grho2, ugcc, v1gcc, v2gcc);
            
            epsx(i) = epsx(i) + ugcx;
            epsc(i) = epsc(i) + ugcc;
            vxc1Temp(i) = vxc1Temp(i) + v1gcx + v1gcc;
            vxc2Temp(i) = vxc2Temp(i) + 0.5 * v2gcx + 0.5 * v2gcc;
          }

          ex[i] = epsx[i] * densityTemp[i];
          ec[i] = epsc[i] * densityTemp[i];

          if( isHybrid ){
            Real omega = 0.106;
            Real frac = 0.25;
            Real v1xsr, v2xsr, epxsr;

            if( absrho > 1e-6 && std::abs(grho2) > 1e-10 ){
              pbexsr( rho, grho2, omega, epxsr, v1xsr, v2xsr );

              vxc1Temp[i] -= frac * v1xsr;
              vxc2Temp[i] -= frac * v2xsr / 2.0;
              ex[i]  -= frac * epxsr;
            }
          }
        }
      } // if( numSpinorComponent_ == 1 )
      else{
        Real rho_up, rho_dn, absrho, grho2_up, grho2_dn, grho2_ud;
        Real rs, zeta, vx_up, vx_dn, ux, vc_up, vc_dn, uc;
        Real v1gcx_up, v1gcx_dn, v2gcx_up, v2gcx_dn, egcx;
        Real v1gcc_up, v1gcc_dn, v2gcc_up, v2gcc_dn, v2gcc_ud, ugcc;

        for( Int i = 0; i < ntotLocal; i++ ){
          Int ii = i + localSizeDispls(mpirank);

          rho_up = densityTemp(2*i+UP);
          rho_dn = densityTemp(2*i+DN);

          absrho = std::abs( rho_up + rho_dn );
          grho2_up = gradDensity(3*i+UPUP);
          grho2_dn = gradDensity(3*i+DNDN);

          if( absrho > 1e-10 ){
            rs = std::pow(3.0 / 4.0 / PI / absrho, 1.0 / 3.0);
            zeta = ( rho_up - rho_dn ) / absrho;
            if( std::abs(zeta) > 1.0 ) zeta = signx( zeta );

            VExchange_sla_spin( absrho, zeta, ux, vx_up, vx_dn );
            VCorrelation_pw_spin( rs, zeta, uc, vc_up, vc_dn );

            epsx(i) = epsx(i) + ux;
            epsc(i) = epsc(i) + uc;
            vxc1Temp(2*i+UP) = vxc1Temp(2*i+UP) + vx_up + vc_up;
            vxc1Temp(2*i+DN) = vxc1Temp(2*i+DN) + vx_dn + vc_dn;
          }

          VGCExchange_pbx_spin(rho_up, rho_dn, grho2_up, grho2_dn, 
              egcx, v1gcx_up, v1gcx_dn, v2gcx_up, v2gcx_dn, isHybrid);

          if( absrho > 1e-6 ) zeta = ( rho_up - rho_dn ) / absrho;
          if( std::abs(zeta) <= 1.0 ){
            zeta = signx(zeta) * std::min(std::abs(zeta), 1-1e-6);
          }

          grho2_ud = (gradDensity0(ii, UP) + gradDensity0(ii, DN))*
              (gradDensity0(ii, UP) + gradDensity0(ii, DN)) +
              (gradDensity1(ii, UP) + gradDensity1(ii, DN))*
              (gradDensity1(ii, UP) + gradDensity1(ii, DN)) +
              (gradDensity2(ii, UP) + gradDensity2(ii, DN))*
              (gradDensity2(ii, UP) + gradDensity2(ii, DN));

          if( absrho > 1e-6 && std::sqrt(grho2_ud) > 1e-6 
            && std::abs(zeta) <= 1.0 ){
            VGCCorrelation_pbc_spin( absrho, zeta, grho2_ud, ugcc, 
                v1gcc_up, v1gcc_dn, v2gcc_up );
          }
          else{
            ugcc = 0.0;
            v1gcc_up = 0.0;
            v1gcc_dn = 0.0;
            v2gcc_up = 0.0;
          }
          v2gcc_dn = v2gcc_up;
          v2gcc_ud = v2gcc_up;

          epsx(i) = epsx(i) + egcx / ( rho_up + rho_dn );
          epsc(i) = epsc(i) + ugcc;

          vxc1Temp(2*i+UP) = vxc1Temp(2*i+UP) + v1gcx_up + v1gcc_up;
          vxc1Temp(2*i+DN) = vxc1Temp(2*i+DN) + v1gcx_dn + v1gcc_dn;
          vxc2Temp(3*i+UPUP) = vxc2Temp(3*i+UPUP) + v2gcx_up + 0.5 * v2gcc_up;
          vxc2Temp(3*i+DNDN) = vxc2Temp(3*i+DNDN) + v2gcx_dn + 0.5 * v2gcc_dn;
          vxc2Temp(3*i+UPDN) = vxc2Temp(3*i+UPDN) + v2gcc_ud;

          ex[i] = epsx[i] * ( densityTemp[2*i+UP] + densityTemp[2*i+DN] );
          ec[i] = epsc[i] * ( densityTemp[2*i+UP] + densityTemp[2*i+DN] );
        }
      }

      // exc = ex + ec
      blas::Copy( ntotLocal, ex.Data(), 1, excTemp.Data(), 1 );
      blas::Axpy( ntotLocal, 1.0, ec.Data(), 1, excTemp.Data(), 1 );
    }
    
    // vxc1, vxc2 and exc_ are global arrays 
    DblNumVec vxc1( ntot*numSpinorComponent_ ); 
    DblNumVec vxc2( ntot*numGrad2 );
    SetValue( exc_, 0.0 );
    SetValue( vxc1, 0.0 );
    SetValue( vxc2, 0.0 );

    GetTime( timeSta );

    IntNumVec localSizeDispls_vxc1( mpisize );
    IntNumVec localSizeDispls_vxc2( mpisize );
    IntNumVec localSize_vxc1( mpisize );
    IntNumVec localSize_vxc2( mpisize );
    for( Int i = 0; i < mpisize; i++ ){
      localSizeDispls_vxc1[i] = numSpinorComponent_ * localSizeDispls[i];
      localSize_vxc1[i] = numSpinorComponent_ * localSize[i];
      localSizeDispls_vxc2[i] = numGrad2 * localSizeDispls[i];
      localSize_vxc2[i] = numGrad2 * localSize[i];
    }
    // Local to global transform
    MPI_Allgatherv( excTemp.Data(), ntotLocal, MPI_DOUBLE, exc_.Data(),
      localSize.Data(), localSizeDispls.Data(), MPI_DOUBLE, domain_.comm );

    MPI_Allgatherv( vxc1Temp.Data(), ntotLocal*numSpinorComponent_, MPI_DOUBLE, vxc1.Data(), 
      localSize_vxc1.Data(), localSizeDispls_vxc1.Data(), MPI_DOUBLE, domain_.comm );

    MPI_Allgatherv( vxc2Temp.Data(), ntotLocal*numGrad2, MPI_DOUBLE, vxc2.Data(), 
      localSize_vxc2.Data(), localSizeDispls_vxc2.Data(), MPI_DOUBLE, domain_.comm );

    GetTime( timeEnd );
    // Put the first-order term of exchange-correlation potential to vxc_
    if( numSpinorComponent_ == 1){
      blas::Copy( ntot, vxc1.Data(), 1, vxc_.Data(), 1 );
    }
    else{
      blas::Copy( ntot, &vxc1[0], 2, vxc_.VecData(UP), 1 );
      blas::Copy( ntot, &vxc1[1], 2, vxc_.VecData(DN), 1 );
    }  

    Int d;
    // Add the two-order term to vxc_
    if( numSpinorComponent_ == 1){
      if( mpisize < DIM ){ // mpisize < 3
        for( Int d = 0; d < DIM; d++ ){
          DblNumMat& gradDensityd = gradDensity_[d];
          for(Int i = 0; i < ntot; i++){
            fft.inputComplexVecFine(i) = Complex( gradDensityd( i, RHO ) * 2.0 * vxc2[i], 0.0 ); 
          }

          FFTWExecute ( fft, fft.forwardPlanFine );
          CpxNumVec& ik = fft.ikFine[d];

          for( Int i = 0; i < ntot; i++ ){
            if( fft.gkkFine(i) == 0 || fft.gkkFine(i) > esdfParam.ecutWavefunction * 4.0 ){
              fft.outputComplexVecFine(i) = Z_ZERO;
            }
            else{
              fft.outputComplexVecFine(i) *= ik(i);
            }  
          }

          FFTWExecute ( fft, fft.backwardPlanFine );

          for( Int i = 0; i < ntot; i++ ){
            vxc_( i, RHO ) -= fft.inputComplexVecFine(i).real();
          }
        } // for (d)
      } // mpisize < 3
      else { // mpisize > 3
        std::vector<DblNumVec>  vxcTemp3d;
        vxcTemp3d.resize( DIM );
        for( Int d = 0; d < DIM; d++ ){
          vxcTemp3d[d].Resize(ntot);
          SetValue (vxcTemp3d[d], 0.0);
        }

        for( d = 0; d < DIM; d++ ){
          DblNumMat& gradDensityd = gradDensity_[d];
          DblNumVec& vxcTemp3 = vxcTemp3d[d]; 
          if ( d == mpirank % dmCol ){ 
            for(Int i = 0; i < ntot; i++){
              fft.inputComplexVecFine(i) = Complex( gradDensityd( i, RHO ) * 2.0 * vxc2[i], 0.0 ); 
            }

            FFTWExecute ( fft, fft.forwardPlanFine );

            CpxNumVec& ik = fft.ikFine[d];

            for( Int i = 0; i < ntot; i++ ){
              if( fft.gkkFine(i) == 0 || fft.gkkFine(i) > esdfParam.ecutWavefunction * 4.0 ){
                fft.outputComplexVecFine(i) = Z_ZERO;
              }
              else{
                fft.outputComplexVecFine(i) *= ik(i);
              }
            }

            FFTWExecute ( fft, fft.backwardPlanFine );

            for( Int i = 0; i < ntot; i++ ){
              vxcTemp3(i) = fft.inputComplexVecFine(i).real();
            }
          } // d == mpirank
        } // for (d)

        for( d = 0; d < DIM; d++ ){
          DblNumVec& vxcTemp3 = vxcTemp3d[d]; 
          MPI_Bcast( vxcTemp3.Data(), ntot, MPI_DOUBLE, d, rowComm_ );
          for( Int i = 0; i < ntot; i++ ){
            vxc_( i, RHO ) -= vxcTemp3(i);
          }
        } // for (d)
      } // mpisize > 3
    } // numDensityComponent_ == 1
    else{
      if( mpisize < DIM ){ // mpisize < 3
        for( Int d = 0; d < DIM; d++ ){
          DblNumMat& gradDensityd = gradDensity_[d];
          // For spin-up component
          for(Int i = 0; i < ntot; i++){
            fft.inputComplexVecFine(i) = Complex( gradDensityd( i, UP ) * 2.0 * vxc2[3*i + UPUP] 
                + gradDensityd( i, DN ) * 1.0 * vxc2[3*i + UPDN], 0.0 ); 
          }

          FFTWExecute ( fft, fft.forwardPlanFine );

          CpxNumVec& ik = fft.ikFine[d];

          for( Int i = 0; i < ntot; i++ ){
            if( fft.gkkFine(i) == 0 || fft.gkkFine(i) > esdfParam.ecutWavefunction * 4.0 ){
              fft.outputComplexVecFine(i) = Z_ZERO;
            }
            else{
              fft.outputComplexVecFine(i) *= ik(i);
            }
          }

          FFTWExecute ( fft, fft.backwardPlanFine );

          for( Int i = 0; i < ntot; i++ ){
            vxc_( i, UP ) -= fft.inputComplexVecFine(i).real();
          }
          // For spin-dw component
          for(Int i = 0; i < ntot; i++){
            fft.inputComplexVecFine(i) = Complex( gradDensityd( i, DN ) * 2.0 * vxc2[3*i + DNDN]
                + gradDensityd( i, UP ) * 1.0 * vxc2[3*i + UPDN], 0.0 );
          }

          FFTWExecute ( fft, fft.forwardPlanFine );

          for( Int i = 0; i < ntot; i++ ){
            if( fft.gkkFine(i) == 0 || fft.gkkFine(i) > esdfParam.ecutWavefunction * 4.0 ){
              fft.outputComplexVecFine(i) = Z_ZERO;
            }
            else{
              fft.outputComplexVecFine(i) *= ik(i);
            }
          }

          FFTWExecute ( fft, fft.backwardPlanFine );

          for( Int i = 0; i < ntot; i++ ){
            vxc_( i, DN ) -= fft.inputComplexVecFine(i).real();
          }
        } // for (d)
      } // mpisize < 3
      else { // mpisize > 3
        std::vector<DblNumMat>  vxcTemp3d;
        vxcTemp3d.resize( DIM );
        for( Int d = 0; d < DIM; d++ ){
          vxcTemp3d[d].Resize(ntot,numSpinorComponent_);
          SetValue (vxcTemp3d[d], 0.0);
        }

        for( d = 0; d < DIM; d++ ){
          DblNumMat& gradDensityd = gradDensity_[d];
          DblNumMat& vxcTemp3 = vxcTemp3d[d]; 
          if ( d == mpirank % dmCol ){ 
            // For spin-up component
            for(Int i = 0; i < ntot; i++){
              fft.inputComplexVecFine(i) = Complex( gradDensityd( i, UP ) * 2.0 * vxc2[3*i + UPUP]
                + gradDensityd( i, DN ) * 1.0 * vxc2[3*i + UPDN], 0.0 );
            }

            FFTWExecute ( fft, fft.forwardPlanFine );

            CpxNumVec& ik = fft.ikFine[d];

            for( Int i = 0; i < ntot; i++ ){
              if( fft.gkkFine(i) == 0 || fft.gkkFine(i) > esdfParam.ecutWavefunction * 4.0 ){
                fft.outputComplexVecFine(i) = Z_ZERO;
              }
              else{
                fft.outputComplexVecFine(i) *= ik(i);
              }
            }

            FFTWExecute ( fft, fft.backwardPlanFine );

            for( Int i = 0; i < ntot; i++ ){
              vxcTemp3(i,UP) = fft.inputComplexVecFine(i).real();
            }
            // For spin-dw component
            for(Int i = 0; i < ntot; i++){
              fft.inputComplexVecFine(i) = Complex( gradDensityd( i, DN ) * 2.0 * vxc2[3*i + DNDN]
                + gradDensityd( i, UP ) * 1.0 * vxc2[3*i + UPDN], 0.0 );
            }

            FFTWExecute ( fft, fft.forwardPlanFine );

            for( Int i = 0; i < ntot; i++ ){
              if( fft.gkkFine(i) == 0 || fft.gkkFine(i) > esdfParam.ecutWavefunction * 4.0 ){
                fft.outputComplexVecFine(i) = Z_ZERO;
              }
              else{
                fft.outputComplexVecFine(i) *= ik(i);
              }
            }

            FFTWExecute ( fft, fft.backwardPlanFine );

            for( Int i = 0; i < ntot; i++ ){
              vxcTemp3(i,DN) = fft.inputComplexVecFine(i).real();
            }
          } // d == mpirank
        } // for (d)

        for( d = 0; d < DIM; d++ ){
          DblNumMat& vxcTemp3 = vxcTemp3d[d]; 
          MPI_Bcast( vxcTemp3.Data(), ntot*numSpinorComponent_, MPI_DOUBLE, d, rowComm_ );
          for( Int i = 0; i < ntot; i++ ){
            vxc_( i, UP ) -= vxcTemp3(i,UP);
            vxc_( i, DN ) -= vxcTemp3(i,DN);
          }
        } // for (d)

      } // mpisize > 3
    } // ---- end of if( numSpinorComponent_ == 1 ) ----
  } // XC_FAMILY_GGA and XC_FAMILY Hybrid

  // For spin-noncollinear case, the vxc(UP,DN) should be rotated back to
  // four-component format 
  if( numDensityComponent_ == 4 ){
    Real aVxc, dVxc, amag;
    Real epsMag = 1e-12;
    Point3 tempmag;

    for( Int i = 0; i < ntot; i++){
      aVxc = 0.5 * ( vxc_( i, UP ) + vxc_( i, DN ) );
      dVxc = 0.5 * ( vxc_( i, UP ) - vxc_( i, DN ) );

      tempmag = Point3( density_(i,1), density_(i,2), density_(i,3) );   
      amag = tempmag.l2();
      
      vxc_( i, RHO ) =  aVxc;
      if( amag > epsMag ){
        for( Int d = 1; d < 4 ; d++){
          vxc_( i, d ) = dVxc * segni_[i] * density_(i,d) / amag;
        }
      }
    } // for (i)
  }
  
  Real sumvxc_up = 0.0, sumvxc_dn = 0.0;
  for( Int i = 0; i < ntot; i++){
    sumvxc_up += vxc_(i,0);
    sumvxc_dn += vxc_(i,1);
  }

  val = 0.0;
  GetTime( timeSta );
  for(Int i = 0; i < ntot; i++){
    val += exc_(i) * vol / (Real) ntot;
  }

  GetTime( timeEnd );

  return ;
}         // -----  end of method KohnSham::CalculateXC  ----- 

void KohnSham::CalculateHartree( Fourier& fft ) {
  if( !fft.isInitialized ){
    ErrorHandling("Fourier is not prepared.");
  }

  Int ntot = domain_.NumGridTotalFine();
  if( fft.domain.NumGridTotalFine() != ntot ){
    ErrorHandling( "Grid size does not match!" );
  }

  // The contribution of the pseudoCharge is subtracted. So the Poisson
  // equation is well defined for neutral system.
  for( Int i = 0; i < ntot; i++ ){
    fft.inputComplexVecFine(i) = Complex( 
        density_(i,RHO) - pseudoCharge_(i), 0.0 );
  }

  FFTWExecute ( fft, fft.forwardPlanFine );

  Real EPS = 1e-16;
  for( Int i = 0; i < ntot; i++ ){
    if( fft.gkkFine(i) <= EPS || fft.gkkFine(i) > esdfParam.ecutWavefunction * 4.0 ){
      fft.outputComplexVecFine(i) = Z_ZERO;
    }
    else{
      fft.outputComplexVecFine(i) *= 2.0 * PI / fft.gkkFine(i);
    }
  }

  FFTWExecute ( fft, fft.backwardPlanFine );

  for( Int i = 0; i < ntot; i++ ){
    vhart_(i) = fft.inputComplexVecFine(i).real();
  }

  return; 
}  // -----  end of method KohnSham::CalculateHartree ----- 

void
KohnSham::CalculateVtot    ( DblNumMat& vtot )
{
  Int ntot = domain_.NumGridTotalFine();
  if( esdfParam.isUseVLocal == false ){
    for (int i=0; i<ntot; i++) {
      if( numDensityComponent_ == 1 ){
        vtot(i,RHO) = vext_(i) + vhart_(i) + vxc_(i, RHO);
      }
      else if( numDensityComponent_ == 2 ){
        vtot(i,UP) = vext_(i) + vhart_(i) + vxc_(i, UP);
        vtot(i,DN) = vext_(i) + vhart_(i) + vxc_(i, DN);
      }
      else if( numDensityComponent_ == 4 ){
        vtot(i,RHO) = vext_(i) + vhart_(i) + vxc_(i, RHO);
        vtot(i,MAGX) = vxc_(i, MAGX);
        vtot(i,MAGY) = vxc_(i, MAGY);
        vtot(i,MAGZ) = vxc_(i, MAGZ);
      }
    }
  }
  else
  {
    for (int i=0; i<ntot; i++) {
      if( numDensityComponent_ == 1 ){
        vtot(i,RHO) = vext_(i) + vLocalSR_(i) + vhart_(i) + vxc_(i, RHO);
      }
      else if( numDensityComponent_ == 2 ){
        vtot(i,UP) = vext_(i) + vLocalSR_(i) + vhart_(i) + vxc_(i, UP);
        vtot(i,DN) = vext_(i) + vLocalSR_(i) + vhart_(i) + vxc_(i, DN);
      }
      else if( numDensityComponent_ == 4 ){
        vtot(i,RHO) = vext_(i) + vLocalSR_(i) + vhart_(i) + vxc_(i, RHO);
        vtot(i,MAGX) = vxc_(i, MAGX);
        vtot(i,MAGY) = vxc_(i, MAGY);
        vtot(i,MAGZ) = vxc_(i, MAGZ);
      }
    }
  }

  return ;
}         // -----  end of method KohnSham::CalculateVtot  ----- 

void
KohnSham::CalculateIonSelfEnergyAndForce    ( PeriodTable &ptable, Fourier &fft )
{
  std::vector<Atom>&  atomList = this->AtomList();
  EVdw_ = 0.0;
  forceVdw_.Resize( atomList.size(), DIM );
  SetValue( forceVdw_, 0.0 );
  
  // Self energy part. 
  Eself_ = 0.0;
  for(Int a=0; a< atomList.size() ; a++) {
    Int type = atomList[a].type;
    Eself_ +=  ptable.SelfIonInteraction(type);
  }

  if( esdfParam.isUseRealSpace == false ){
    // The long-range part of Ewald energy is calculated in Fourier space
    Real EewaldG = 0.0;

    IntNumVec& idxDensity = fft.idxFineCutDensity;
    Int ntot = idxDensity.Size();
    Int ntotFine = fft.domain.NumGridTotalFine();

    Int numAtom = atomList_.size();
    Real vol = domain_.Volume();
    Int nelec = 0;
    Real alpha = 2.9;
    Real upperbound = 1.0;
    Real EPS = 1e-16;

    Complex phase, val;
    Real arg, sumnb, fac;

    Complex* ikxPtr = fft.ikFine[0].Data();
    Complex* ikyPtr = fft.ikFine[1].Data();
    Complex* ikzPtr = fft.ikFine[2].Data();
    Real*    gkkPtr = fft.gkkFine.Data();

    for( Int a = 0; a < numAtom; a++ ){
      Int atype  = atomList_[a].type;
      if( ptable.ptemap().find(atype) == ptable.ptemap().end() ){
        ErrorHandling( "Cannot find the atom type." );
      }
      nelec = nelec + ptable.Zion(atype);
    }
    // Determine the Gaussian radius where Rgauss = sqrt( 1 / alpha )
    // Gaussian distribution is exp( - (r / Rgauss) ^ 2 )
    while( upperbound > 1e-7 ){
      alpha = alpha - 0.1;
      // ecutDensity = 4 * ecutWavefunction
      upperbound = 2.0 * nelec * nelec * std::sqrt( alpha / PI )
          * erfc( sqrt( esdfParam.ecutWavefunction * 4.0 / 2.0 / alpha ) );
    }

    // Zero point correction of divergent term ( |G| = 0 )
    EewaldG += - nelec * nelec / alpha / 4.0;
   
    forceIonLR_.Resize( atomList.size(), DIM );
    SetValue(forceIonLR_, 0.0);
 
    for( Int i = 0; i < ntot; i++ ){

      Int ig = idxDensity[i];

      if( gkkPtr[ig] > EPS ){
        val = Z_ZERO;
        for( Int a = 0; a < numAtom; a++ ){
          Int atype  = atomList_[a].type;

          phase = ikxPtr[ig] * atomList_[a].pos[0] +
                  ikyPtr[ig] * atomList_[a].pos[1] + 
                  ikzPtr[ig] * atomList_[a].pos[2];
        
          val += ptable.Zion(atype) * std::exp(phase);
        }

        for( Int a = 0; a < numAtom; a++ ){
          Int atype  = atomList_[a].type;

          arg = ikxPtr[ig].imag() * atomList_[a].pos[0] +
                ikyPtr[ig].imag() * atomList_[a].pos[1] + 
                ikzPtr[ig].imag() * atomList_[a].pos[2];

          sumnb = ( std::cos(arg) * val.imag() - std::sin(arg) * val.real() )
              * std::exp( - gkkPtr[ig] / alpha / 2.0 ) / ( gkkPtr[ig] * 2.0 );

          fac = - ptable.Zion(atype) * 4.0 * PI / vol;

          forceIonLR_( a, 0 ) += fac * ikxPtr[ig].imag() * sumnb;
          forceIonLR_( a, 1 ) += fac * ikyPtr[ig].imag() * sumnb;
          forceIonLR_( a, 2 ) += fac * ikzPtr[ig].imag() * sumnb;
        }

        EewaldG += pow( abs(val), 2.0 ) * std::exp( - gkkPtr[ig] / alpha / 2 ) 
            / ( gkkPtr[ig] * 2.0 );
      }   
    }

    EewaldG *= 2 * PI / vol;
    Eself_ -= EewaldG;
  }

  // Short range repulsion part
  EIonSR_ = 0.0;
  forceIonSR_.Resize( atomList.size(), DIM );
  SetValue(forceIonSR_, 0.0);
  if( esdfParam.isUseVLocal == true ){
    const Domain& dm = domain_;
    std::vector<Point3> vec;
    vec.resize(DIM);
    for( Int d = 0; d < DIM; d++ ){
      vec[d] = Point3( dm.supercell(d,0), dm.supercell(d,1), dm.supercell(d,2) );
    }

    for(Int a=0; a< atomList.size() ; a++) {
      Int type_a = atomList[a].type;
      Real Zion_a = ptable.Zion(type_a);
      Real RGaussian_a = ptable.RGaussian(type_a);

      for(Int b=a; b< atomList.size() ; b++) {
        // Need to consider the interaction between the same atom and
        // its periodic image. Be sure not to double ocunt
        bool same_atom = (a==b);

        Int type_b = atomList[b].type;
        Real Zion_b = ptable.Zion(type_b);
        Real RGaussian_b = ptable.RGaussian(type_b);

        Real radius_ab = std::sqrt ( RGaussian_a*RGaussian_a + RGaussian_b*RGaussian_b );
        //Real radius_ab = RGaussian_a + RGaussian_b;
        Point3 pos_ab = atomList[a].pos - atomList[b].pos;

        const Real facNbr = 8.0;        
        // convergence criterion for lattice sums:
        // |Pos_a - Pos_b + ncell_i * L_i| > facNbr * ( RGaussian_a + RGaussian_b )
        Index3 ncell;
        Point3 pos_ab_image;
        for( Int d = 0; d < DIM; d++ ){
          pos_ab_image[0] = pos_ab[0] + ncell[d] * vec[d][0];
          pos_ab_image[1] = pos_ab[1] + ncell[d] * vec[d][1];
          pos_ab_image[2] = pos_ab[2] + ncell[d] * vec[d][2];

          while( pos_ab_image.l2() < facNbr * (radius_ab) ){
            ncell[d] ++;
            pos_ab_image[0] = pos_ab[0] + ncell[d] * vec[d][0];
            pos_ab_image[1] = pos_ab[1] + ncell[d] * vec[d][1];
            pos_ab_image[2] = pos_ab[2] + ncell[d] * vec[d][2];
          }
        }

        // loop over neighboring cells
        Real fac;
        for ( Int ic0 = -ncell[0]; ic0 <= ncell[0]; ic0++ )
          for ( Int ic1 = -ncell[1]; ic1 <= ncell[1]; ic1++ )
            for ( Int ic2 = -ncell[2]; ic2 <= ncell[2]; ic2++ )
            {
              if ( !same_atom || ic0!=0 || ic1!=0 || ic2!=0 )
              {
                if ( same_atom )
                  fac = 0.5;
                else
                  fac = 1.0;
                
                Point3 pos_ab_image;
                
                for( Int d = 0; d < DIM; d++ )
                  pos_ab_image[d] = pos_ab[d] + vec[0][d] * ic0 
                                        + vec[1][d] * ic1 + vec[2][d] * ic2;

                Real r_ab = pos_ab_image.l2();
                Real esr_term = Zion_a * Zion_b * std::erfc(r_ab / radius_ab) / r_ab;
                Real desr_erfc = 2.0 * Zion_a * Zion_b *
                  std::exp(-(r_ab / radius_ab)*(r_ab / radius_ab))/(radius_ab*std::sqrt(PI));
                // desrdr = (1/r) d Esr / dr
                Real desrdr = - fac * (esr_term+desr_erfc) / ( r_ab*r_ab );
                
                EIonSR_ += fac * esr_term;

                forceIonSR_(a,0) -= desrdr * pos_ab_image[0];
                forceIonSR_(b,0) += desrdr * pos_ab_image[0];
                forceIonSR_(a,1) -= desrdr * pos_ab_image[1];
                forceIonSR_(b,1) += desrdr * pos_ab_image[1];
                forceIonSR_(a,2) -= desrdr * pos_ab_image[2];
                forceIonSR_(b,2) += desrdr * pos_ab_image[2];
              }
            }
      } // for (b)
    } // for (a)
  } // if esdfParam.isUseVLocal == true

  return ;
}         // -----  end of method KohnSham::CalculateIonSelfEnergyAndForce  ----- 

void
KohnSham::CalculateVdwEnergyAndForce    ()
{
  std::vector<Atom>&  atomList = this->AtomList();
  EVdw_ = 0.0;
  forceVdw_.Resize( atomList.size(), DIM );
  SetValue( forceVdw_, 0.0 );

  Int numAtom = atomList.size();

  const Domain& dm = domain_;

  if( esdfParam.VDWType == "DFT-D2"){

    const Int vdw_nspecies = 55;
    Int ia,is1,is2,is3,itypat,ja,jtypat,npairs,nshell;
    bool need_gradient,newshell;
    const Real vdw_d = 20.0;
    const Real vdw_tol_default = 1e-10;
    const Real vdw_s_pbe = 0.75, vdw_s_blyp = 1.2, vdw_s_b3lyp = 1.05;
    const Real vdw_s_hse = 0.75, vdw_s_pbe0 = 0.60;
    //Thin Solid Films 535 (2013) 387-389
    //J. Chem. Theory Comput. 2011, 7, 88–96

    Real c6,c6r6,ex,fr,fred1,fred2,fred3,gr,grad,r0,r1,r2,r3,rcart1,rcart2,rcart3;
    //real(dp) :: rcut,rcut2,rsq,rr,sfact,ucvol,vdw_s
    //character(len=500) :: msg
    //type(atomdata_t) :: atom
    //integer,allocatable :: ivdw(:)
    //real(dp) :: gmet(3,3),gprimd(3,3),rmet(3,3)
    //real(dp),allocatable :: vdw_c6(:,:),vdw_r0(:,:),xred01(:,:)
    //DblNumVec vdw_c6_dftd2(vdw_nspecies);

    double vdw_c6_dftd2[vdw_nspecies] = 
    { 0.14, 0.08, 1.61, 1.61, 3.13, 1.75, 1.23, 0.70, 0.75, 0.63,
      5.71, 5.71,10.79, 9.23, 7.84, 5.57, 5.07, 4.61,10.80,10.80,
      10.80,10.80,10.80,10.80,10.80,10.80,10.80,10.80,10.80,10.80,
      16.99,17.10,16.37,12.64,12.47,12.01,24.67,24.67,24.67,24.67,
      24.67,24.67,24.67,24.67,24.67,24.67,24.67,24.67,37.32,38.71,
      38.44,31.74,31.50,29.99, 0.00 };

    // DblNumVec vdw_r0_dftd2(vdw_nspecies);
    double vdw_r0_dftd2[vdw_nspecies] =
    { 1.001,1.012,0.825,1.408,1.485,1.452,1.397,1.342,1.287,1.243,
      1.144,1.364,1.639,1.716,1.705,1.683,1.639,1.595,1.485,1.474,
      1.562,1.562,1.562,1.562,1.562,1.562,1.562,1.562,1.562,1.562,
      1.650,1.727,1.760,1.771,1.749,1.727,1.628,1.606,1.639,1.639,
      1.639,1.639,1.639,1.639,1.639,1.639,1.639,1.639,1.672,1.804,
      1.881,1.892,1.892,1.881,1.000 };

    for(Int i=0; i<vdw_nspecies; i++) {
      vdw_c6_dftd2[i] = vdw_c6_dftd2[i] / 2625499.62 * pow(10/0.52917706, 6);
      vdw_r0_dftd2[i] = vdw_r0_dftd2[i] / 0.52917706;
    }

    DblNumMat vdw_c6(vdw_nspecies, vdw_nspecies);
    DblNumMat vdw_r0(vdw_nspecies, vdw_nspecies);
    SetValue( vdw_c6, 0.0 );
    SetValue( vdw_r0, 0.0 );

    for(Int i=0; i<vdw_nspecies; i++) {
      for(Int j=0; j<vdw_nspecies; j++) {
        vdw_c6(i, j) = std::sqrt( vdw_c6_dftd2[i] * vdw_c6_dftd2[j] );
        vdw_r0(i, j) = vdw_r0_dftd2[i] + vdw_r0_dftd2[j];
      }
    }

    Real vdw_s;

    if (XCType_ == "XC_GGA_XC_PBE") {
      vdw_s = vdw_s_pbe;
    }
    else if (XCType_ == "XC_HYB_GGA_XC_HSE06") {
      vdw_s = vdw_s_hse;
    }
    else if (XCType_ == "XC_HYB_GGA_XC_PBEH") {
      vdw_s = vdw_s_pbe0;
    }
    else {
      ErrorHandling( "Van der Waals DFT-D2 correction in only compatible with GGA-PBE, HSE06, and PBE0!" );
    }

    Index3 shift; 
    Point3 shift_car;
    for(Int ii=-1; ii<2; ii++) {
      for(Int jj=-1; jj<2; jj++) {
        for(Int kk=-1; kk<2; kk++) {

          for(Int i=0; i<atomList.size(); i++) {
            Int iType = atomList[i].type;
            for(Int j=0; j<(i+1); j++) {
              Int jType = atomList[j].type;
              shift = Index3( ii, jj, kk );
              shift_car = Point3( 0.0, 0.0, 0.0 );
              for( Int ip = 0; i < DIM; i++ ){
                for( Int jp = 0; j < DIM; j++ ){
                  shift_car[i] += dm.supercell(j,i) * shift[j];
                }
              }
              Real rx = atomList[i].pos[0] - atomList[j].pos[0] + shift_car[0];
              Real ry = atomList[i].pos[1] - atomList[j].pos[1] + shift_car[1];
              Real rz = atomList[i].pos[2] - atomList[j].pos[2] + shift_car[2];

              Real rr = std::sqrt( rx * rx + ry * ry + rz * rz );

              if ( ( rr > 0.0001 ) && ( rr < 75.0 ) ) {

                Real sfact = vdw_s;
                if ( i == j ) sfact = sfact * 0.5;

                Real c6 = vdw_c6(iType-1, jType-1);
                Real r0 = vdw_r0(iType-1, jType-1);

                Real ex = exp( -vdw_d * ( rr / r0 - 1 ));
                Real fr = 1.0 / ( 1.0 + ex );
                Real c6r6 = c6 / pow(rr, 6.0);

                // Contribution to energy
                EVdw_ = EVdw_ - sfact * fr * c6r6;

                // Contribution to force
                if( i != j ) {

                  Real gr = ( vdw_d / r0 ) * ( fr * fr ) * ex;
                  Real grad = sfact * ( gr - 6.0 * fr / rr ) * c6r6 / rr; 

                  Real fx = grad * rx;
                  Real fy = grad * ry;
                  Real fz = grad * rz;

                  forceVdw_( i, 0 ) = forceVdw_( i, 0 ) + fx; 
                  forceVdw_( i, 1 ) = forceVdw_( i, 1 ) + fy; 
                  forceVdw_( i, 2 ) = forceVdw_( i, 2 ) + fz; 
                  forceVdw_( j, 0 ) = forceVdw_( j, 0 ) - fx; 
                  forceVdw_( j, 1 ) = forceVdw_( j, 1 ) - fy; 
                  forceVdw_( j, 2 ) = forceVdw_( j, 2 ) - fz; 

                } // end for i != j

              } // end if

            } // end for j
          } // end for i

        } // end for ii
      } // end for jj
    } // end for kk

  } // If DFT-D2

  return ;
}         // -----  end of method KohnSham::CalculateVdwEnergyAndForce  ----- 

void KohnSham::Setup_XC( std::string xc_functional)
{
  Int nspin = XC_UNPOLARIZED;
  if( numDensityComponent_ >= 2 ){
    nspin = XC_POLARIZED;
  }

  if( xc_functional == "XC_GGA_XC_PBE" )
  {
    XId_  = XC_GGA_X_PBE;
    CId_  = XC_GGA_C_PBE;
    XCId_ = XC_GGA_X_PBE;
    statusOFS << "XC_GGA_XC_PBE  XId_ CId_ = " << XId_ << " " << CId_  << std::endl << std::endl;
    // Perdew, Burke & Ernzerhof correlation
    // JP Perdew, K Burke, and M Ernzerhof, Phys. Rev. Lett. 77, 3865 (1996)
    // JP Perdew, K Burke, and M Ernzerhof, Phys. Rev. Lett. 78, 1396(E) (1997)
    if( xc_func_init(&XFuncType_, XId_, nspin) != 0 ){
      ErrorHandling( "X functional initialization error." );
    }
    if( xc_func_init(&CFuncType_, CId_, nspin) != 0 ){
      ErrorHandling( "C functional initialization error." );
    }
  }
  else if( xc_functional == "XC_HYB_GGA_XC_HSE06" )
  {
    XCId_ = XC_HYB_GGA_XC_HSE06;
    XId_ = XC_GGA_X_PBE;
    CId_ = XC_GGA_X_PBE;
    statusOFS << "XC_HYB_GGA_XC_HSE06  XCId = " << XCId_  << std::endl << std::endl;
    if( xc_func_init(&XCFuncType_, XCId_, nspin) != 0 ){
      ErrorHandling( "XC functional initialization error." );
    } 
    isHybrid_ = true;
  }
}

#ifndef _COMPLEX_
void
KohnSham::CalculateDensity ( const Spinor &psi, const DblNumVec &occrate, Real &val, Fourier &fft)
{
  bool realspace = esdfParam.isUseRealSpace;
 
  SetValue( density_, 0.0 ); 
  SetValue( spindensity_, 0.0 );   

  Index3& numGrid = domain_.numGrid;
  Index3& numGridFine = domain_.numGridFine;
  unsigned plannerFlag = FFTW_MEASURE;

  Real vol  = domain_.Volume();
  Int ntotFine  = fft.domain.NumGridTotalFine();
  Int ntot  = fft.domain.NumGridTotal();
  Int ntotR2C = fft.numGridTotalR2C;

  MPI_Barrier(domain_.comm);

  DblNumMat   densityLocal;
  densityLocal.Resize( ntotFine, numDensityComponent_ );
  SetValue( densityLocal, 0.0 );

  Int ntotG  = psi.NumGridTotal();
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
 
  DblNumVec psiFine(ntotFine);
 
  for( Int k = 0; k < nocc; k++ ){
    for( Int j = 0; j < numDensityComponent_; j++ ){

      SetValue( psiFine, 0.0 );

      SetValue(fft.outputVecR2C, Z_ZERO );
      SetValue( fft.outputComplexVecFine, Z_ZERO );

      if( realspace ){
        for( Int i = 0; i < ntotG; i++ ){
          fft.inputComplexVec(i) = Complex( psi.Wavefun(i,VAL,k+j*nocc), 0.0 ); 
        }

        FFTWExecute ( fft, fft.forwardPlan );
      }
      else{
        Int *idxPtr = fft.idxCoarseCut.Data();

        Complex *fftOutR2CPtr = fft.outputVecR2C.Data();
        Complex *psiPtr = psi.WavefunG().VecData(VAL,k+j*nocc);
        fftOutR2CPtr[*(idxPtr++)] = *(psiPtr++);
        for( Int i = 1; i < ntotG; i++ ){
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

      if( realspace ){
        SetValue( fft.outputComplexVecFine, Z_ZERO );
        for( Int i = 0; i < ntot; i++ ){
          fft.outputComplexVecFine(fft.idxFineGrid(i)) = fft.outputComplexVec(i) *
            sqrt( double(ntot) / double(ntotFine) );
        }

        FFTWExecute ( fft, fft.backwardPlanFine );

        fac = numSpin_ * occrate(psi.WavefunIdx(k)+j*nocc_total);
        for( Int i = 0; i < ntotFine; i++ ){
          densityLocal(i,j) +=  pow( std::abs(fft.inputComplexVecFine(i).real()), 2.0 ) * fac;
        }
      }
      else{
        // fft Coarse to Fine 
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

        Real fac = 1.0 / std::sqrt(double(ntotFine));
        blas::Scal( ntotFine, fac, psiFine.Data(), 1 );

        fac = numSpin_ * occrate(psi.WavefunIdx(k)+j*nocc_total);
        for( Int i = 0; i < ntotFine; i++ ){
          densityLocal(i,j) +=  pow( psiFine(i), 2.0 ) * fac;
        }
      }
    } // for (j)
  } // for (k)

  if( numDensityComponent_ == 1 ){
    mpi::Allreduce( densityLocal.Data(), density_.Data(), ntotFine*numDensityComponent_, MPI_SUM, domain_.comm );
  }
  else{
    mpi::Allreduce( densityLocal.Data(), spindensity_.Data(), ntotFine*numDensityComponent_, MPI_SUM, domain_.comm );
    blas::Copy( ntotFine, spindensity_.VecData(0), 1, density_.VecData(0), 1 );      
    blas::Copy( ntotFine, spindensity_.VecData(0), 1, density_.VecData(1), 1 );
    // arho = rhoup + rhodw
    // drho = rhoup - rhodw
    blas::Axpy( ntotFine, 1.0, spindensity_.VecData(1), 1, density_.VecData(0), 1);
    blas::Axpy( ntotFine, -1.0, spindensity_.VecData(1), 1, density_.VecData(1), 1);
  }

  val = 0.0; // sum of density

  for( Int i = 0; i < ntotFine; i++ ){
    val  += density_(i, RHO);
  }

  Real val1 = val;

  // Scale the density 
  blas::Scal( ntotFine*numDensityComponent_, (numSpin_ * Real(numOccupiedState_) * Real(ntotFine)) / ( vol * val ), 
      density_.Data(), 1 );

  if( numDensityComponent_ == 2 ){
    // Scale the spindensity
    blas::Scal( ntotFine*numDensityComponent_, (numSpin_ * Real(numOccupiedState_) * Real(ntotFine)) / ( vol * val ),
        spindensity_.Data(), 1 );
  }
  // Double check (can be neglected)
  val = 0.0; // sum of density
  for (Int i=0; i<ntotFine; i++) {
    val  += density_(i, RHO) * vol / ntotFine;
  }

  Real val2 = val;
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

  return ;
}         // -----  end of method KohnSham::CalculateDensity ( Real version )  ----- 

void
KohnSham::CalculateEkin( Fourier& fft )
{
  Domain& dm = fft.domain;

  bool spherecut = esdfParam.isUseSphereCut;

  Int ntothalf;
  Point3 kG = Point3( 0.0, 0.0, 0.0 );
  Real a, b;

  if( spherecut )
    ntothalf = dm.numGridSphere;
  else
    ntothalf = fft.numGridTotalR2C;

  IntNumVec& idxc = fft.idxCoarseCut;
  ekin_.Resize( ntothalf ); teter_.Resize( ntothalf );

  for( Int i = 0; i < ntothalf; i++ ){
    if( spherecut ){
      Int ig = idxc[i];
      kG = Point3( fft.ikR2C[0][ig].imag(), fft.ikR2C[1][ig].imag(),
          fft.ikR2C[2][ig].imag() );
    }
    else{
      kG = Point3( fft.ikR2C[0][i].imag(), fft.ikR2C[1][i].imag(),
          fft.ikR2C[2][i].imag() );
    }

    ekin_[i] = ( kG[0]*kG[0] + kG[1]*kG[1] + kG[2]*kG[2] ) / 2;

    a = ekin_[i] * 2.0;
    b = 27.0 + a * (18.0 + a * (12.0 + a * 8.0) );
    teter_[i] = b / ( b + 16.0 * pow(a, 4.0) );
  }

  return;
}         // -----  end of method KohnSham::CalculateEkin ( Real version )  ----- 

void
KohnSham::CalculateForce    ( Spinor& psi, Fourier& fft  )
{
  Real timeSta, timeEnd;

  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

  bool usevlocal = esdfParam.isUseVLocal;
  bool realspace = esdfParam.isUseRealSpace;

  Int ntotFine  = fft.domain.NumGridTotalFine();
  Int ncom  = psi.NumComponent();
  Int numStateLocal = psi.NumState(); // Local number of states
  Int numStateTotal = psi.NumStateTotal();
  Real vol = fft.domain.Volume();

  Int numAtom   = atomList_.size();

  DblNumMat  force( numAtom, DIM );
  SetValue( force, 0.0 );
  DblNumMat  forceLocal( numAtom, DIM );
  SetValue( forceLocal, 0.0 );

  int numAtomBlocksize = numAtom  / mpisize;
  int numAtomLocal = numAtomBlocksize;
  if(mpirank < (numAtom % mpisize)){
    numAtomLocal = numAtomBlocksize + 1;
  }
  IntNumVec numAtomIdx( numAtomLocal );

  if (numAtomBlocksize == 0 ){
    for (Int i = 0; i < numAtomLocal; i++){
      numAtomIdx[i] = mpirank;
    }
  }
  else {
    if ( (numAtom % mpisize) == 0 ){
      for (Int i = 0; i < numAtomLocal; i++){
        numAtomIdx[i] = numAtomBlocksize * mpirank + i;
      }
    }
    else{
      for (Int i = 0; i < numAtomLocal; i++){
        if ( mpirank < (numAtom % mpisize) ){
          numAtomIdx[i] = (numAtomBlocksize + 1) * mpirank + i;
        }
        else{
          numAtomIdx[i] = (numAtomBlocksize + 1) * (numAtom % mpisize) + numAtomBlocksize * (mpirank - (numAtom % mpisize)) + i;
        }
      }
    }
  }
  // *********************************************************************
  // Compute the force from local pseudopotential
  // *********************************************************************
  // Using integration by parts for local pseudopotential.
  // No need to evaluate the derivative of the local pseudopotential.
  // This could potentially save some coding effort, and perhaps better for other 
  // pseudopotential such as Troullier-Martins

  if( !usevlocal )
  {
    std::vector<DblNumVec>  vhartDrv(DIM);

    DblNumVec  totalCharge(ntotFine);
    SetValue( totalCharge, 0.0 );

    // totalCharge = density_ - pseudoCharge_
    blas::Copy( ntotFine, density_.VecData(0), 1, totalCharge.Data(), 1 );
    blas::Axpy( ntotFine, -1.0, pseudoCharge_.Data(),1,
        totalCharge.Data(), 1 );

    // Total charge in the Fourier space
    CpxNumVec  totalChargeFourier( ntotFine );

    for( Int i = 0; i < ntotFine; i++ ){
      fft.inputComplexVecFine(i) = Complex( totalCharge(i), 0.0 );
    }

    FFTWExecute ( fft, fft.forwardPlanFine );

    blas::Copy( ntotFine, fft.outputComplexVecFine.Data(), 1,
        totalChargeFourier.Data(), 1 );

    // Compute the derivative of the Hartree potential via Fourier
    // transform 
    for( Int d = 0; d < DIM; d++ ){
      CpxNumVec& ikFine = fft.ikFine[d];
      for( Int i = 0; i < ntotFine; i++ ){
        if( fft.gkkFine(i) == 0 ){
          fft.outputComplexVecFine(i) = Z_ZERO;
        }
        else{
          // NOTE: gkk already contains the factor 1/2.
          fft.outputComplexVecFine(i) = totalChargeFourier(i) *
            2.0 * PI / fft.gkkFine(i) * ikFine(i);
        }
      }

      FFTWExecute ( fft, fft.backwardPlanFine );

      // vhartDrv saves the derivative of the Hartree potential
      vhartDrv[d].Resize( ntotFine );

      for( Int i = 0; i < ntotFine; i++ ){
        vhartDrv[d](i) = fft.inputComplexVecFine(i).real();
      }
    } // for (d)

    for( Int i = 0; i < numAtomLocal; i++ ){
      int a = numAtomIdx[i];
      PseudoPot& pp = pseudo_[a];
      SparseVec& sp = pp.pseudoCharge;
      IntNumVec& idx = sp.first;
      DblNumMat& val = sp.second;

      Real wgt = domain_.Volume() / domain_.NumGridTotalFine();
      Real resX = 0.0;
      Real resY = 0.0;
      Real resZ = 0.0;
      for( Int l = 0; l < idx.m(); l++ ){
        resX += val(l, VAL) * vhartDrv[0][idx(l)] * wgt;
        resY += val(l, VAL) * vhartDrv[1][idx(l)] * wgt;
        resZ += val(l, VAL) * vhartDrv[2][idx(l)] * wgt;
      }
      force( a, 0 ) += resX;
      force( a, 1 ) += resY;
      force( a, 2 ) += resZ;
    } // for (a)
  } // pseudocharge formulation of the local contribution to the force
  else{
    if( realspace ){
      // First contribution from the pseudocharge
      std::vector<DblNumVec>  vhartDrv(DIM);

      DblNumVec  totalCharge(ntotFine);
      SetValue( totalCharge, 0.0 );
      // totalCharge = density_ - pseudoCharge_
      blas::Copy( ntotFine, density_.VecData(0), 1, totalCharge.Data(), 1 );
      blas::Axpy( ntotFine, -1.0, pseudoCharge_.Data(),1,
          totalCharge.Data(), 1 );

      // Total charge in the Fourier space
      CpxNumVec  totalChargeFourier( ntotFine );

      for( Int i = 0; i < ntotFine; i++ ){
        fft.inputComplexVecFine(i) = Complex( totalCharge(i), 0.0 );
      }

      FFTWExecute ( fft, fft.forwardPlanFine );

      blas::Copy( ntotFine, fft.outputComplexVecFine.Data(), 1,
          totalChargeFourier.Data(), 1 );

      // Compute the derivative of the Hartree potential via Fourier
      // transform 
      for( Int d = 0; d < DIM; d++ ){
        CpxNumVec& ikFine = fft.ikFine[d];
        for( Int i = 0; i < ntotFine; i++ ){
          if( fft.gkkFine(i) == 0 ){
            fft.outputComplexVecFine(i) = Z_ZERO;
          }
          else{
            // NOTE: gkk already contains the factor 1/2.
            fft.outputComplexVecFine(i) = totalChargeFourier(i) *
              2.0 * PI / fft.gkkFine(i) * ikFine(i);
          }
        }
 
        FFTWExecute ( fft, fft.backwardPlanFine );

        // vhartDrv saves the derivative of the Hartree potential
        vhartDrv[d].Resize( ntotFine );

        for( Int i = 0; i < ntotFine; i++ ){
          vhartDrv[d](i) = fft.inputComplexVecFine(i).real();
        }
      } // for (d)

      for( Int i = 0; i < numAtomLocal; i++ ){
        int a = numAtomIdx[i];
        PseudoPot& pp = pseudo_[a];
        SparseVec& sp = pp.pseudoCharge;
        IntNumVec& idx = sp.first;
        DblNumMat& val = sp.second;

        Real wgt = domain_.Volume() / domain_.NumGridTotalFine();
        Real resX = 0.0;
        Real resY = 0.0;
        Real resZ = 0.0;
        for( Int l = 0; l < idx.m(); l++ ){
          resX += val(l, VAL) * vhartDrv[0][idx(l)] * wgt;
          resY += val(l, VAL) * vhartDrv[1][idx(l)] * wgt;
          resZ += val(l, VAL) * vhartDrv[2][idx(l)] * wgt;
        }
        force( a, 0 ) += resX;
        force( a, 1 ) += resY;
        force( a, 2 ) += resZ;
      } // for (a)
    } // ---- end of if( realspace ) ----

    // Second, contribution from the vLocalSR.  
    // The integration by parts formula requires the calculation of the grad density
    if( realspace ){

      this->CalculateGradDensity( fft );

      for( Int i = 0; i < numAtomLocal; i++ ){
        int a = numAtomIdx[i];
        PseudoPot& pp = pseudo_[a];
        SparseVec& sp = pp.vLocalSR;
        IntNumVec& idx = sp.first;
        DblNumMat& val = sp.second;

        Real wgt = domain_.Volume() / domain_.NumGridTotalFine();
        Real resX = 0.0;
        Real resY = 0.0;
        Real resZ = 0.0;
        for( Int l = 0; l < idx.m(); l++ ){
          resX -= val(l, VAL) * gradDensity_[0](idx(l),0) * wgt;
          resY -= val(l, VAL) * gradDensity_[1](idx(l),0) * wgt;
          resZ -= val(l, VAL) * gradDensity_[2](idx(l),0) * wgt;
        }
        force( a, 0 ) += resX;
        force( a, 1 ) += resY;
        force( a, 2 ) += resZ;
      } // for (a)
    } // ---- end of if ( realspace ) ----
  } // ---- end of if( !usevlocal ) ----
  // *********************************************************************
  // Compute the force from nonlocal pseudopotential
  // *********************************************************************
  // Method 4: Using integration by parts, and throw the derivative to the wavefunctions
  // No need to evaluate the derivative of the non-local pseudopotential.
  // This could potentially save some coding effort, and perhaps better for other 
  // pseudopotential such as Troullier-Martins
  {
    DblNumVec                psiFine( ntotFine );
    CpxNumVec                psiFourier(ntotFine);
    std::vector<DblNumVec>   psiDrvFine(DIM);

    for( Int d = 0; d < DIM; d++ ){
      psiDrvFine[d].Resize( ntotFine );
    }

    Int ntot = psi.NumGridTotal();

    // Loop over atoms and pseudopotentials
    for( Int g = 0; g < numStateLocal; g++ ){
      // Compute the derivative of the wavefunctions on a fine grid
      Real* psiPtr = psi.Wavefun().VecData(0, g);

      for( Int i = 0; i < domain_.NumGridTotal(); i++ ){
        fft.inputComplexVec(i) = Complex( psiPtr[i], 0.0 ); 
      }

      FFTWExecute ( fft, fft.forwardPlan );

      // Interpolate wavefunction from coarse to fine grid
      SetValue( psiFourier, Z_ZERO );
      for( Int i = 0; i < ntot; i++ ){
        psiFourier(fft.idxFineGrid(i)) = fft.outputComplexVec(i);
      }

      for( Int i = 0; i < ntotFine; i++ ){
        fft.outputComplexVecFine(i) = psiFourier(i);
      }

      FFTWExecute ( fft, fft.backwardPlanFine );

      Real fac = sqrt(double(domain_.NumGridTotal())) / 
          sqrt( double(domain_.NumGridTotalFine()) ); 

      blas::Copy( ntotFine, reinterpret_cast<Real*>(fft.inputComplexVecFine.Data()),
          2, psiFine.Data(), 1 );
      blas::Scal( ntotFine, fac, psiFine.Data(), 1 );

      // derivative of psi on a fine grid
      for( Int d = 0; d < DIM; d++ ){
        Complex* ikFinePtr = fft.ikFine[d].Data();
        Complex* psiFourierPtr    = psiFourier.Data();
        Complex* fftOutFinePtr = fft.outputComplexVecFine.Data();
        for( Int i = 0; i < ntotFine; i++ ){
          *(fftOutFinePtr++) = *(psiFourierPtr++) * *(ikFinePtr++);
        }

        FFTWExecute ( fft, fft.backwardPlanFine );

        blas::Copy( ntotFine, reinterpret_cast<Real*>(fft.inputComplexVecFine.Data()),
            2, psiDrvFine[d].Data(), 1 );
        blas::Scal( ntotFine, fac, psiDrvFine[d].Data(), 1 );
      } // for (d)

      // Evaluate the contribution to the atomic force
      for( Int a = 0; a < numAtom; a++ ){
        std::vector<NonlocalPP>& vnlList = pseudo_[a].vnlList;
        for( Int l = 0; l < vnlList.size(); l++ ){
          SparseVec& bl = vnlList[l].first;
          Real  gamma   = vnlList[l].second;
          Real  wgt = domain_.Volume() / domain_.NumGridTotalFine();
          IntNumVec& idx = bl.first;
          DblNumMat& val = bl.second;

          DblNumVec res(4);
          SetValue( res, 0.0 );
          Real* psiPtr = psiFine.Data();
          Real* DpsiXPtr = psiDrvFine[0].Data();
          Real* DpsiYPtr = psiDrvFine[1].Data();
          Real* DpsiZPtr = psiDrvFine[2].Data();
          Real* valPtr   = val.VecData(VAL);
          Int*  idxPtr = idx.Data();
          for( Int i = 0; i < idx.Size(); i++ ){
            res(VAL) += *valPtr * psiPtr[ *idxPtr ] * sqrt(wgt);
            res(DX)  += *valPtr * DpsiXPtr[ *idxPtr ] * sqrt(wgt);
            res(DY)  += *valPtr * DpsiYPtr[ *idxPtr ] * sqrt(wgt);
            res(DZ)  += *valPtr * DpsiZPtr[ *idxPtr ] * sqrt(wgt);
            valPtr++;
            idxPtr++;
          }

          Real occrate;
          if( numDensityComponent_ == 2 ){
            occrate = occupationRate_( psi.WavefunIdx(g%(numStateLocal/2)) +
                (g/(numStateLocal/2)) * numStateTotal / 2 );
          }
          else{
            occrate = occupationRate_(psi.WavefunIdx(g));
          }

          forceLocal( a, 0 ) += -2.0 * occrate * gamma * res[VAL] * res[DX] * numSpin_;
          forceLocal( a, 1 ) += -2.0 * occrate * gamma * res[VAL] * res[DY] * numSpin_;
          forceLocal( a, 2 ) += -2.0 * occrate * gamma * res[VAL] * res[DZ] * numSpin_;
        } // for (l)
      } // for (a)
    } // for (g)
  }
  // *********************************************************************
  // Compute the total force and give the value to atomList
  // *********************************************************************

  // Sum over the force
  DblNumMat  forceTmp( numAtom, DIM );
  DblNumMat  forceTmp1( numAtom, DIM );
  SetValue( forceTmp, 0.0 );
  SetValue( forceTmp1, 0.0 );

  mpi::Allreduce( forceLocal.Data(), forceTmp.Data(), numAtom * DIM, MPI_SUM, domain_.comm );
  mpi::Allreduce( force.Data(), forceTmp1.Data(), numAtom * DIM, MPI_SUM, domain_.comm );

  for( Int a = 0; a < numAtom; a++ ){
    force( a, 0 ) = forceTmp( a, 0 ) + forceTmp1( a, 0 );
    force( a, 1 ) = forceTmp( a, 1 ) + forceTmp1( a, 1 );
    force( a, 2 ) = forceTmp( a, 2 ) + forceTmp1( a, 2 );
  }

  for( Int a = 0; a < numAtom; a++ ){
    atomList_[a].force = Point3( force(a,0), force(a,1), force(a,2) );
  }

  // Add extra contribution to the force
  if( esdfParam.VDWType == "DFT-D2"){
    // Update force
    std::vector<Atom>& atomList = this->AtomList();
    for( Int a = 0; a < atomList.size(); a++ ){
      atomList[a].force += Point3( forceVdw_(a,0), forceVdw_(a,1), forceVdw_(a,2) );
    }
  }

  // Add the contribution from short range interaction
  if( usevlocal ){
    std::vector<Atom>& atomList = this->AtomList();
    for( Int a = 0; a < atomList.size(); a++ ){
      atomList[a].force += Point3( forceIonSR_(a,0), forceIonSR_(a,1), forceIonSR_(a,2) );
    }
  }

  // Add the contribution from external force
  {
    std::vector<Atom>& atomList = this->AtomList();
    for( Int a = 0; a < atomList.size(); a++ ){
      atomList[a].force += Point3( forceext_(a,0), forceext_(a,1), forceext_(a,2) );
    }
  }

  return ;
}         // -----  end of method KohnSham::CalculateForce ( Real version )  ----- 

void
KohnSham::MultSpinor    ( Spinor& psi, NumTns<Real>& a3, Fourier& fft )
{
  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

  bool useace = esdfParam.isHybridACE;

  Int ntotFine  = fft.domain.NumGridTotalFine();
  Int nspin     = numDensityComponent_;
  Int ncom      = ( nspin == 4 ) ? 2 : 1;

  Int ntot      = psi.NumGridTotal();
  Int ntotLocal = psi.NumGrid();

  Int numStateTotal = psi.NumStateTotal();
  Int numStateLocal = psi.NumState();

  NumTns<Real>& wavefun = psi.Wavefun();

  Int ntotR2C = fft.numGridTotalR2C;

  Real timeSta, timeEnd;
  Real timeSta1, timeEnd1;

  Real timeGemm = 0.0;
  Real timeAlltoallv = 0.0;
  Real timeAllreduce = 0.0;

  SetValue( a3, D_ZERO );

  // Apply an initial filter on the wavefunctions, if required
  if((apply_filter_ == 1 && apply_first_ == 1))
  {
    apply_first_ = 0;

    for (Int k=0; k<numStateLocal; k++) {
      for (Int j=0; j<ncom; j++) {

        SetValue( fft.inputVecR2C, 0.0 );
        SetValue( fft.outputVecR2C, Z_ZERO );

        blas::Copy( ntot, wavefun.VecData(j,k), 1,
            fft.inputVecR2C.Data(), 1 );
        FFTWExecute ( fft, fft.forwardPlanR2C ); // So outputVecR2C contains the FFT result now

        for (Int i=0; i<ntotR2C; i++)
        {
          if(fft.gkkR2C(i) > wfn_cutoff_)
            fft.outputVecR2C(i) = Z_ZERO;
        }

        FFTWExecute ( fft, fft.backwardPlanR2C );
        blas::Copy( ntot,  fft.inputVecR2C.Data(), 1,
            wavefun.VecData(j,k), 1 );
      }
    }
  }

  GetTime( timeSta );
  // The vtot can be inputed as DblNumVec here as only one component 
  // of local potential is needed for spin-restricted or spin-unrestricted case
  DblNumVec vtot( ntotFine );
  blas::Copy( ntotFine, vtot_.VecData( spinswitch_ ), 1, vtot.Data(), 1 );
  psi.AddMultSpinorFineR2C( fft, ekin_, vtot, pseudo_, a3 );
  
  GetTime( timeEnd );
#if 0
  statusOFS << "Time for psi.AddMultSpinorFineR2C is " <<
    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

  if( isHybrid_ && isEXXActive_ ){

    GetTime( timeSta );

    if( useace ){
      // Convert the column partition to row partition
      Int numOccLocal, numOccTotal;
 
      if( nspin == 1 ){
        numOccLocal = vexxProj_.n();
      }
      else if( nspin ==2 ){
        if( spinswitch_ == 0 ){
          numOccLocal = UpvexxProj_.n();
        }
        else{
          numOccLocal = DnvexxProj_.n();
        }
      }

      MPI_Allreduce( &numOccLocal, &numOccTotal, 1, MPI_INT, MPI_SUM, domain_.comm );

      DblNumMat psiCol( ntot, numStateLocal );
      SetValue( psiCol, D_ZERO );

      DblNumMat vexxProjCol( ntot, numOccLocal );
      SetValue( vexxProjCol, D_ZERO );

      DblNumMat psiRow( ntotLocal, numStateTotal );
      SetValue( psiRow, D_ZERO );

      DblNumMat vexxProjRow( ntotLocal, numOccTotal );
      SetValue( vexxProjRow, D_ZERO );
        
      lapack::Lacpy( 'A', ntot, numStateLocal, psi.Wavefun().Data(), ntot, psiCol.Data(), ntot );

      if( nspin == 1 )
        lapack::Lacpy( 'A', ntot, numOccLocal, vexxProj_.Data(), 
            ntot, vexxProjCol.Data(), ntot );
      else if ( nspin == 2 ){
        if( spinswitch_ == 0 )
          lapack::Lacpy( 'A', ntot, numOccLocal, UpvexxProj_.Data(), 
            ntot, vexxProjCol.Data(), ntot );
        else
          lapack::Lacpy( 'A', ntot, numOccLocal, DnvexxProj_.Data(), 
            ntot, vexxProjCol.Data(), ntot );
      }
       
      Int mb = esdfParam.BlockSizeGrid;
      Int nb = esdfParam.BlockSizeState;

      AlltoallForward( mb, nb, psiCol, psiRow, domain_.comm );

      AlltoallForward( mb, nb, vexxProjCol, vexxProjRow, domain_.comm );

      DblNumMat MTemp( numOccTotal, numStateTotal );
      SetValue( MTemp, D_ZERO );

      blas::Gemm( 'T', 'N', numOccTotal, numStateTotal, ntotLocal,
          1.0, vexxProjRow.Data(), ntotLocal,
          psiRow.Data(), ntotLocal, 0.0,
          MTemp.Data(), numOccTotal );

      DblNumMat M(numOccTotal, numStateTotal);
      SetValue( M, D_ZERO );
      MPI_Allreduce( MTemp.Data(), M.Data(), numOccTotal*numStateTotal, MPI_DOUBLE, MPI_SUM, domain_.comm );

      DblNumMat a3Col( ntot, numStateLocal );
      SetValue( a3Col, D_ZERO );

      DblNumMat a3Row( ntotLocal, numStateTotal );
      SetValue( a3Row, D_ZERO );

      blas::Gemm( 'N', 'N', ntotLocal, numStateTotal, numOccTotal,
          -1.0, vexxProjRow.Data(), ntotLocal,
          M.Data(), numOccTotal, 0.0,
          a3Row.Data(), ntotLocal );

      AlltoallBackward( mb, nb, a3Row, a3Col, domain_.comm );

      for( Int k = 0; k < numStateLocal; k++ ){
        Real *p1 = a3Col.VecData(k);
        Real *p2 = a3.VecData(0, k);
        for( Int i = 0; i < ntot; i++ ){
          *(p2++) += *(p1++);
        }
      }
    }
    else{
      if( nspin == 1 )
        psi.AddMultSpinorEXX( fft, phiEXX_, exxgkkR2C_,
          exxFraction_, nspin, occupationRate_, a3 );
      else{
        DblNumVec occSpin;
        Int numStateTotal = occupationRate_.Size() / 2;
        occSpin.Resize( numStateTotal );
        blas::Copy( numStateTotal, &(occupationRate_[spinswitch_*numStateTotal]), 1, occSpin.Data(), 1 );

        if( spinswitch_ == 0 )
          psi.AddMultSpinorEXX( fft, UpphiEXX_, exxgkkR2C_,
              exxFraction_, nspin, occSpin, a3 );  
        else
          psi.AddMultSpinorEXX( fft, DnphiEXX_, exxgkkR2C_,
              exxFraction_, nspin, occSpin, a3 );         
      }
      GetTime( timeEnd );
    }
  } // ---- if( isHybrid_ && isEXXActive_ ) ----

  // Apply filter on the wavefunctions before exit, if required
  if((apply_filter_ == 1))
  {
    for (Int k=0; k<numStateLocal; k++) {
      for (Int j=0; j<ncom; j++) {

        SetValue( fft.inputVecR2C, 0.0 );
        SetValue( fft.outputVecR2C, Z_ZERO );

        blas::Copy( ntot, a3.VecData(j,k), 1,
            fft.inputVecR2C.Data(), 1 );
        FFTWExecute ( fft, fft.forwardPlanR2C ); // So outputVecR2C contains the FFT result now


        for (Int i=0; i<ntotR2C; i++)
        {
          if(fft.gkkR2C(i) > wfn_cutoff_)
            fft.outputVecR2C(i) = Z_ZERO;
        }

        FFTWExecute ( fft, fft.backwardPlanR2C );
        blas::Copy( ntot,  fft.inputVecR2C.Data(), 1,
            a3.VecData(j,k), 1 );
      }
    }
  }

  return ;
}         // -----  end of method KohnSham::MultSpinor ( Real version )  ----- 

void
KohnSham::MultSpinor    ( Spinor& psi, NumTns<Complex>& a3, Fourier& fft )
{
  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

  bool useace = esdfParam.isHybridACE;

  Int ntotFine  = fft.domain.NumGridTotalFine();
  Int nspin     = numDensityComponent_;
  Int ncom      = ( nspin == 4 ) ? 2 : 1;

  Int ntot      = psi.NumGridTotal();
  Int ntotLocal = psi.NumGrid();

  Int numStateTotal = psi.NumStateTotal();
  Int numStateLocal = psi.NumState();

  NumTns<Real>& wavefun = psi.Wavefun();

  Int ntotR2C = fft.numGridTotalR2C;

  Real timeSta, timeEnd;
  Real timeSta1, timeEnd1;

  Real timeGemm = 0.0;
  Real timeAlltoallv = 0.0;
  Real timeAllreduce = 0.0;

  SetValue( a3, Z_ZERO );

  // Apply an initial filter on the wavefunctions, if required
  if((apply_filter_ == 1 && apply_first_ == 1))
  {
    apply_first_ = 0;

    for (Int k=0; k<numStateLocal; k++) {
      for (Int j=0; j<ncom; j++) {

        SetValue( fft.inputVecR2C, 0.0 );
        SetValue( fft.outputVecR2C, Z_ZERO );

        blas::Copy( ntot, wavefun.VecData(j,k), 1,
            fft.inputVecR2C.Data(), 1 );
        FFTWExecute ( fft, fft.forwardPlanR2C ); // So outputVecR2C contains the FFT result now

        for (Int i=0; i<ntotR2C; i++)
        {
          if(fft.gkkR2C(i) > wfn_cutoff_)
            fft.outputVecR2C(i) = Z_ZERO;
        }

        FFTWExecute ( fft, fft.backwardPlanR2C );
        blas::Copy( ntot,  fft.inputVecR2C.Data(), 1,
            wavefun.VecData(j,k), 1 );
      }
    }
  }

  GetTime( timeSta );
  // The vtot can be inputed as DblNumVec here as only one component 
  // of local potential is needed for spin-restricted or spin-unrestricted case
  DblNumVec vtot( ntotFine );
  blas::Copy( ntotFine, vtot_.VecData( spinswitch_ ), 1, vtot.Data(), 1 );
  psi.AddMultSpinorFineR2C( fft, ekin_, vtot, pseudo_, a3 );
  
  GetTime( timeEnd );
#if 0
  statusOFS << "Time for psi.AddMultSpinorFineR2C is " <<
    timeEnd - timeSta << " [s]" << std::endl << std::endl;
#endif

  if( isHybrid_ && isEXXActive_ ){

    GetTime( timeSta );

    if( useace ){
      // Convert the column partition to row partition
      Int numOccLocal, numOccTotal;

      if( nspin == 1 ){
        numOccLocal = vexxProjG_.n();
      }
      else if( nspin ==2 ){
        if( spinswitch_ == 0 ){
          numOccLocal = UpvexxProjG_.n();
        }
        else{
          numOccLocal = DnvexxProjG_.n();
        }
      }

      MPI_Allreduce( &numOccLocal, &numOccTotal, 1, MPI_INT, MPI_SUM, domain_.comm );

      CpxNumMat psiCol( ntot, numStateLocal );
      SetValue( psiCol, Z_ZERO );

      CpxNumMat vexxProjCol( ntot, numOccLocal );
      SetValue( vexxProjCol, Z_ZERO );

      CpxNumMat psiRow( ntotLocal, numStateTotal );
      SetValue( psiRow, Z_ZERO );

      CpxNumMat vexxProjRow( ntotLocal, numOccTotal );
      SetValue( vexxProjRow, Z_ZERO );
        
      lapack::Lacpy( 'A', ntot, numStateLocal, psi.WavefunG().Data(), ntot, psiCol.Data(), ntot );

      if( nspin == 1 )
        lapack::Lacpy( 'A', ntot, numOccLocal, vexxProjG_.Data(), 
            ntot, vexxProjCol.Data(), ntot );
      else if ( nspin == 2 ){
        if( spinswitch_ == 0 )
          lapack::Lacpy( 'A', ntot, numOccLocal, UpvexxProjG_.Data(), 
            ntot, vexxProjCol.Data(), ntot );
        else
          lapack::Lacpy( 'A', ntot, numOccLocal, DnvexxProjG_.Data(), 
            ntot, vexxProjCol.Data(), ntot );
      }

      Int mb = esdfParam.BlockSizeGrid;
      Int nb = esdfParam.BlockSizeState;
        
      AlltoallForward( mb, nb, ncom, psiCol, psiRow, domain_.comm );

      AlltoallForward( mb, nb, ncom, vexxProjCol, vexxProjRow, domain_.comm );

      DblNumMat MTemp( numOccTotal, numStateTotal );
      SetValue( MTemp, D_ZERO );

      blas::Gemm( 'C', 'N', numOccTotal, numStateTotal, ntotLocal,
          1.0, vexxProjRow.Data(), ntotLocal,
          psiRow.Data(), ntotLocal, 0.0,
          MTemp.Data(), numOccTotal );

      DblNumMat M(numOccTotal, numStateTotal);
      SetValue( M, D_ZERO );
      MPI_Allreduce( MTemp.Data(), M.Data(), numOccTotal*numStateTotal, MPI_DOUBLE, MPI_SUM, domain_.comm );

      CpxNumMat a3Col( ntot, numStateLocal );
      SetValue( a3Col, Z_ZERO );

      CpxNumMat a3Row( ntotLocal, numStateTotal );
      SetValue( a3Row, Z_ZERO );

      blas::Gemm( 'N', 'N', ntotLocal, numStateTotal, numOccTotal,
          -1.0, vexxProjRow.Data(), ntotLocal,
          M.Data(), numOccTotal, 0.0,
          a3Row.Data(), ntotLocal );

      AlltoallBackward( mb, nb, ncom, a3Row, a3Col, domain_.comm );

      for( Int k = 0; k < numStateLocal; k++ ){
        Complex *p1 = a3Col.VecData(k);
        Complex *p2 = a3.VecData(0, k);
        for( Int i = 0; i < ntot; i++ ){
          *(p2++) += *(p1++);
        }
      }
    }
    else{
      if( nspin == 1 )
        psi.AddMultSpinorEXX( fft, phiEXXG_, exxgkkR2C_,
          exxFraction_, nspin, occupationRate_, a3 );
      else{
        DblNumVec occSpin;
        Int numStateTotal = occupationRate_.Size() / 2;
        occSpin.Resize( numStateTotal );
        blas::Copy( numStateTotal, &(occupationRate_[spinswitch_*numStateTotal]), 1, occSpin.Data(), 1 );

        if( spinswitch_ == 0 )
          psi.AddMultSpinorEXX( fft, UpphiEXXG_, exxgkkR2C_,
              exxFraction_, nspin, occSpin, a3 );  
        else
          psi.AddMultSpinorEXX( fft, DnphiEXXG_, exxgkkR2C_,
              exxFraction_, nspin, occSpin, a3 );         
      }
    }
  } // ---- if( isHybrid_ && isEXXActive_ ) ----

  return ;
}         // -----  end of method KohnSham::MultSpinor ( Real version )  ----- 

void KohnSham::InitializeEXX ( Real ecutWavefunction, Fourier& fft )
{
  bool spherecut  = esdfParam.isUseSphereCut;

  const Real epsDiv = 1e-8;
  isEXXActive_ = false;

  Int ntotR2C = fft.numGridTotalR2C;
  Int npw;
  
  IntNumVec &idxc = fft.idxFineFock;
  if( spherecut )
    npw = idxc.m();
  else
    npw = ntotR2C;

  exxgkkR2C_.Resize(npw);
  SetValue( exxgkkR2C_, 0.0 );

  Real exxAlpha = 10.0 / (ecutWavefunction * 2.0);

  Real gkk2;
  if(exxDivergenceType_ == 0){
    exxDiv_ = 0.0;
  }
  else if (exxDivergenceType_ == 1){
    exxDiv_ = 0.0;
   
    for( Int ig = 0; ig < fft.numGridTotal; ig++ ){
      gkk2 = fft.gkk(ig) * 2.0;
       
      bool inGrid = ((!spherecut) && (gkk2 > epsDiv)) ||
          (spherecut && gkk2 > epsDiv && gkk2 < esdfParam.ecutWavefunction * 4.0 );

      if( inGrid ){
        if( screenMu_ > 0.0 ){
          exxDiv_ += std::exp(-exxAlpha * gkk2) / gkk2 *
            (1.0 - std::exp(-gkk2 / (4.0*screenMu_*screenMu_)));
        }
        else{
          exxDiv_ += std::exp(-exxAlpha * gkk2) / gkk2;
        }
      }
    } // for (ig)

    if( screenMu_ > 0.0 ){
      exxDiv_ += 1.0 / (4.0*screenMu_*screenMu_);
    }
    else{
      exxDiv_ -= exxAlpha;
    }
    exxDiv_ *= 4.0 * PI;

    Real aa = 0.0;

    if( screenMu_ > 0.0 ){
      aa = + 1.0 / std::sqrt(exxAlpha*PI) -
          1.0 / std::sqrt(PI*(exxAlpha + 1.0 / (4.0*screenMu_*screenMu_ )));
    }
    else{
      aa = + 1.0 / std::sqrt(exxAlpha*PI);
    }
    exxDiv_ -= domain_.Volume()*aa;
  }

  statusOFS << "computed exxDiv_ = " << exxDiv_ << std::endl;

  for( Int ig = 0; ig < npw; ig++ ){

    if( spherecut ){
      gkk2 = fft.gkkR2C(idxc[ig]) * 2.0;    
    }
    else{
      gkk2 = fft.gkkR2C(ig) * 2.0;
    }

    if( gkk2 > epsDiv ){
      if( screenMu_ > 0 ){
        exxgkkR2C_[ig] = 4.0 * PI / gkk2 * (1.0 -
          std::exp( -gkk2 / (4.0*screenMu_*screenMu_) ));
      }
      else{
        exxgkkR2C_[ig] = 4.0 * PI / gkk2;
      }
    }
    else{
      exxgkkR2C_[ig] = -exxDiv_;
      if( screenMu_ > 0 ){
        exxgkkR2C_[ig] += PI / (screenMu_*screenMu_);
      }
    }
  } // for (ig)

  statusOFS << "Hybrid mixing parameter  = " << exxFraction_ << std::endl;
  statusOFS << "Hybrid screening length = " << screenMu_ << std::endl;

  return ;
}        // -----  end of function KohnSham::InitializeEXX ( Real version )  ----- 

void
KohnSham::SetPhiEXX    (Spinor& psi, Fourier& fft)
{
  // Collect Psi into a globally shared array in the MPI context.
  Domain &dm = fft.domain;

  bool realspace = esdfParam.isUseRealSpace;
  bool spherecut = esdfParam.isUseSphereCut;

  const NumTns<Real>& wavefun = psi.Wavefun();
  const NumTns<Complex>& wavefunG = psi.WavefunG();

  Int nspin = dm.numSpinComponent;
  Int ntot = dm.NumGridTotal();
  Real vol = fft.domain.Volume();

  if( spherecut ){
    Int npw = dm.numGridSphere;
    Int ncom = wavefunG.n();
    Int numStateLocal = wavefunG.p(); 
    Real fac = std::sqrt( 1.0 / vol / double(ntot) );

    if( nspin == 1 ){
      phiEXXG_.Resize( npw, ncom, numStateLocal );
      SetValue( phiEXXG_, Z_ZERO );

      for( Int j = 0; j < numStateLocal; j++ ){
        for( Int i = 0; i < ncom; i++ ){
          blas::Copy( npw, wavefunG.VecData(i,j), 1, phiEXXG_.VecData(i,j), 1 );
          blas::Scal( npw, fac, phiEXXG_.VecData(i,j), 1 );
        } // for (i)
      } // for (j)   
    }
    else{
      numStateLocal /= 2;

      UpphiEXXG_.Resize( npw, 1, numStateLocal );
      DnphiEXXG_.Resize( npw, 1, numStateLocal );
      SetValue( UpphiEXXG_, Z_ZERO );
      SetValue( DnphiEXXG_, Z_ZERO );

      for( Int j = 0; j < numStateLocal; j++){
        blas::Copy( npw, wavefunG.VecData(0,j+UP*numStateLocal),
            1, UpphiEXXG_.VecData(0,j), 1 );
        blas::Copy( npw, wavefunG.VecData(0,j+DN*numStateLocal),
            1, DnphiEXXG_.VecData(0,j), 1 );
        blas::Scal( npw, fac, UpphiEXXG_.VecData(0,j), 1 );
        blas::Scal( npw, fac, DnphiEXXG_.VecData(0,j), 1 );
      } // for (j)   
    }
  }
  else{
    Int npw = ntot;
    Int ncom = wavefun.n();
    Int numStateLocal = wavefun.p();
    Real fac = std::sqrt( double(ntot) / vol );

    if( nspin == 1 ){
      phiEXX_.Resize( npw, ncom, numStateLocal );
      SetValue( phiEXX_, D_ZERO );

      for( Int j = 0; j < numStateLocal; j++ ){
        for( Int i = 0; i < ncom; i++ ){
          blas::Copy( npw, wavefun.VecData(i,j), 1, phiEXX_.VecData(i,j), 1 );
          blas::Scal( npw, fac, phiEXX_.VecData(i,j), 1 );
        } // for (i)
      } // for (j)   
    }
    else{
      numStateLocal /= 2;

      UpphiEXX_.Resize( npw, 1, numStateLocal );
      DnphiEXX_.Resize( npw, 1, numStateLocal );
      SetValue( UpphiEXX_, D_ZERO );
      SetValue( DnphiEXX_, D_ZERO );

      for( Int j = 0; j < numStateLocal; j++){
        blas::Copy( npw, wavefun.VecData(0,j+UP*numStateLocal),
            1, UpphiEXX_.VecData(0,j), 1 );
        blas::Copy( npw, wavefun.VecData(0,j+DN*numStateLocal),
            1, DnphiEXX_.VecData(0,j), 1 );
        blas::Scal( npw, fac, UpphiEXX_.VecData(0,j), 1 );
        blas::Scal( npw, fac, DnphiEXX_.VecData(0,j), 1 );
      } // for (j)   
    }
  }
      
  return ;
}    // -----  end of method KohnSham::SetPhiEXX ( Real version )  -----

void
KohnSham::SetPhiEXX    (std::string WfnFileName, Fourier& fft)
{
  // Read wavefunction and occupationRate from files to build
  // Fock exchange operator
  Domain &dm = fft.domain;
  Int mpirank;
  MPI_Comm_rank( dm.comm, &mpirank ); 

  bool realspace = esdfParam.isUseRealSpace;

  Int nspin = dm.numSpinComponent;
  Int ntot = dm.NumGridTotal();

  Real vol = fft.domain.Volume();
  Real fac;
  if( realspace )
    fac = std::sqrt( double(ntot) / vol );
  else
    fac = std::sqrt( 1.0 / vol / double(ntot) );

  std::istringstream wfnStream; 
  SeparateRead( WfnFileName, wfnStream, mpirank );

  if( nspin == 1 ){ 
    deserialize( phiEXX_, wfnStream, NO_MASK );
    deserialize( occupationRate_, wfnStream, NO_MASK );

    blas::Scal( phiEXX_.Size(), fac, phiEXX_.Data(), 1 );
  }
  else if( nspin == 2 ){
    deserialize( UpphiEXX_, wfnStream, NO_MASK );
    deserialize( DnphiEXX_, wfnStream, NO_MASK );
    deserialize( occupationRate_, wfnStream, NO_MASK );

    blas::Scal( UpphiEXX_.Size(), fac, UpphiEXX_.Data(), 1 );
    blas::Scal( DnphiEXX_.Size(), fac, DnphiEXX_.Data(), 1 );
  }  
  
  return ;
}    // -----  end of method KohnSham::SetPhiEXX ( Real version )  -----

void
KohnSham::CalculateVexxACE ( Spinor& psi, Fourier& fft )
{
  // This assumes SetPhiEXX has been called so that phiEXX and psi
  // contain the same information. 

  // Since this is a projector, it should be done on the COARSE grid,
  // i.e. to the wavefunction directly

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

  // The band number for each spin
  Int numStateTotal = psi.NumStateTotal();
  Int numStateLocal = psi.NumState();
  if( nspin == 2 )
  {
    numStateTotal /= 2;
    numStateLocal /= 2;
  }

  if( !spherecut ){
    NumTns<Real>  vexxPsi( ntot, ncom, numStateLocal );;
    // The ACE operator are constructed separately for spin-up and spin-down spinors
    // in the case of spin-unrestricted and spin-restricted calculations
    DblNumVec occSpin;
    for( Int ispin = 0; ispin < numACE; ispin++ ){    
      if( nspin == 2 ){
        occSpin.Resize( numStateTotal ); 
        blas::Copy( numStateTotal, &(occupationRate_[ispin*numStateTotal]), 1, occSpin.Data(), 1 );
      }
      
      Spinor psiTemp( fft.domain, ncom, numStateTotal, false, 
          psi.Wavefun().VecData(0,ispin*numStateLocal) );   

      // VexxPsi = V_{exx}*Psi.
      SetValue( vexxPsi, D_ZERO );
      if( nspin == 1 ){
        psiTemp.AddMultSpinorEXX( fft, phiEXX_, exxgkkR2C_,
            exxFraction_, nspin, occupationRate_, vexxPsi );
      }
      else if( nspin == 2 ){
        if( ispin == 0 )
          psiTemp.AddMultSpinorEXX( fft, UpphiEXX_, exxgkkR2C_,
            exxFraction_, nspin, occSpin, vexxPsi ); 
        else
          psiTemp.AddMultSpinorEXX( fft, DnphiEXX_, exxgkkR2C_,
            exxFraction_, nspin, occSpin, vexxPsi ); 
      }

      // Implementation based on SVD
      DblNumMat  M(numStateTotal, numStateTotal);
      // Convert the column partition to row partition
      Int numStateBlocksize = numStateTotal / mpisize;
      Int ntotBlocksize = ntot / mpisize;

      numStateLocal = numStateBlocksize;
      Int ntotLocal = ntotBlocksize;

      if(mpirank < (numStateTotal % mpisize)){
        numStateLocal = numStateBlocksize + 1;
      }

      if(mpirank < (ntot % mpisize)){
        ntotLocal = ntotBlocksize + 1;
      }

      DblNumMat localPsiCol( ntot, numStateLocal );
      SetValue( localPsiCol, D_ZERO );

      DblNumMat localVexxPsiCol( ntot, numStateLocal );
      SetValue( localVexxPsiCol, D_ZERO );

      DblNumMat localPsiRow( ntotLocal, numStateTotal );
      SetValue( localPsiRow, D_ZERO );

      DblNumMat localVexxPsiRow( ntotLocal, numStateTotal );
      SetValue( localVexxPsiRow, D_ZERO );

      // Initialize     
      lapack::Lacpy( 'A', ntot, numStateLocal, psiTemp.Wavefun().Data(), ntot, localPsiCol.Data(), ntot );
      lapack::Lacpy( 'A', ntot, numStateLocal, vexxPsi.Data(), ntot, localVexxPsiCol.Data(), ntot );

      AlltoallForward (localPsiCol, localPsiRow, domain_.comm);
      AlltoallForward (localVexxPsiCol, localVexxPsiRow, domain_.comm);

      DblNumMat MTemp( numStateTotal, numStateTotal );
      SetValue( MTemp, D_ZERO );

      blas::Gemm( 'T', 'N', numStateTotal, numStateTotal, ntotLocal,
          -1.0, localPsiRow.Data(), ntotLocal,
          localVexxPsiRow.Data(), ntotLocal, 0.0,
          MTemp.Data(), numStateTotal );

      SetValue( M, D_ZERO );
      MPI_Allreduce( MTemp.Data(), M.Data(), numStateTotal * numStateTotal, 
          MPI_DOUBLE, MPI_SUM, domain_.colComm_kpoint );

      if ( mpirank == 0) {
        lapack::Potrf('L', numStateTotal, M.Data(), numStateTotal);
      }

      MPI_Bcast(M.Data(), numStateTotal * numStateTotal, MPI_DOUBLE, 0, domain_.comm);

      blas::Trsm( 'R', 'L', 'T', 'N', ntotLocal, numStateTotal, 1.0,
          M.Data(), numStateTotal, localVexxPsiRow.Data(), ntotLocal );   

      if( nspin == 1 ){
        vexxProj_.Resize( ntot, numStateLocal );
        AlltoallBackward (localVexxPsiRow, vexxProj_, domain_.comm);
      }
      else if( nspin == 2 ){
        if( ispin == 0 ){
          UpvexxProj_.Resize( ntot, numStateLocal );
          AlltoallBackward (localVexxPsiRow, UpvexxProj_, domain_.comm);  
        }
        else{
          DnvexxProj_.Resize( ntot, numStateLocal );
          AlltoallBackward (localVexxPsiRow, DnvexxProj_, domain_.comm);
        }
      }       
    }
  }
  else{
    NumTns<Complex>  vexxPsi( ntot, ncom, numStateLocal );;
    // The ACE operator are constructed separately for spin-up and spin-down spinors
    // in the case of spin-unrestricted and spin-restricted calculations
    DblNumVec occSpin;
    for( Int ispin = 0; ispin < numACE; ispin++ ){    
      if( nspin == 2 ){
        occSpin.Resize( numStateTotal ); 
        blas::Copy( numStateTotal, &(occupationRate_[ispin*numStateTotal]), 1, occSpin.Data(), 1 );
      }
      
      Spinor psiTemp( fft.domain, ncom, numStateTotal, false, 
          psi.WavefunG().VecData(0,ispin*numStateLocal) );   

      // VexxPsi = V_{exx}*Psi.
      SetValue( vexxPsi, Z_ZERO );
      if( nspin == 1 ){
        psiTemp.AddMultSpinorEXX( fft, phiEXXG_, exxgkkR2C_,
            exxFraction_, nspin, occupationRate_, vexxPsi );
      }
      else if( nspin == 2 ){
        if( ispin == 0 )
          psiTemp.AddMultSpinorEXX( fft, UpphiEXXG_, exxgkkR2C_,
            exxFraction_, nspin, occSpin, vexxPsi ); 
        else
          psiTemp.AddMultSpinorEXX( fft, DnphiEXXG_, exxgkkR2C_,
            exxFraction_, nspin, occSpin, vexxPsi ); 
      }

      // Implementation based on SVD
      DblNumMat  M(numStateTotal, numStateTotal);
      // Convert the column partition to row partition
      Int numStateBlocksize = numStateTotal / mpisize;
      Int ntotBlocksize = ntot / mpisize;

      numStateLocal = numStateBlocksize;
      Int ntotLocal = ntotBlocksize;

      if(mpirank < (numStateTotal % mpisize)){
        numStateLocal = numStateBlocksize + 1;
      }

      if(mpirank < (ntot % mpisize)){
        ntotLocal = ntotBlocksize + 1;
      }

      CpxNumMat localPsiCol( ntot, numStateLocal );
      SetValue( localPsiCol, Z_ZERO );

      CpxNumMat localVexxPsiCol( ntot, numStateLocal );
      SetValue( localVexxPsiCol, Z_ZERO );

      CpxNumMat localPsiRow( ntotLocal, numStateTotal );
      SetValue( localPsiRow, Z_ZERO );

      CpxNumMat localVexxPsiRow( ntotLocal, numStateTotal );
      SetValue( localVexxPsiRow, Z_ZERO );

      // Initialize     
      lapack::Lacpy( 'A', ntot, numStateLocal, psiTemp.WavefunG().Data(), ntot, localPsiCol.Data(), ntot );
      lapack::Lacpy( 'A', ntot, numStateLocal, vexxPsi.Data(), ntot, localVexxPsiCol.Data(), ntot );

      AlltoallForward (localPsiCol, localPsiRow, domain_.comm);
      AlltoallForward (localVexxPsiCol, localVexxPsiRow, domain_.comm);

      DblNumMat MTemp( numStateTotal, numStateTotal );
      SetValue( MTemp, D_ZERO );

      blas::Gemm( 'C', 'N', numStateTotal, numStateTotal, ntotLocal,
          -1.0, localPsiRow.Data(), ntotLocal,
          localVexxPsiRow.Data(), ntotLocal, 0.0,
          MTemp.Data(), numStateTotal );

      SetValue( M, D_ZERO );
      MPI_Allreduce( MTemp.Data(), M.Data(), numStateTotal * numStateTotal, 
          MPI_DOUBLE, MPI_SUM, domain_.comm );

      if ( mpirank == 0) {
        lapack::Potrf('L', numStateTotal, M.Data(), numStateTotal);
      }

      MPI_Bcast(M.Data(), numStateTotal * numStateTotal, MPI_DOUBLE, 0, domain_.comm);

      blas::Trsm( 'R', 'L', 'T', 'N', ntotLocal, numStateTotal, 1.0,
          M.Data(), numStateTotal, localVexxPsiRow.Data(), ntotLocal );   

      if( nspin == 1 ){
        vexxProjG_.Resize( ntot, numStateLocal );
        AlltoallBackward (localVexxPsiRow, vexxProjG_, domain_.comm);
      }
      else if( nspin == 2 ){
        if( ispin == 0 ){
          UpvexxProjG_.Resize( ntot, numStateLocal );
          AlltoallBackward (localVexxPsiRow, UpvexxProjG_, domain_.comm);  
        }
        else{
          DnvexxProjG_.Resize( ntot, numStateLocal );
          AlltoallBackward (localVexxPsiRow, DnvexxProjG_, domain_.comm);
        }
      }       
    }
  }

  return ;
}         // -----  end of method KohnSham::CalculateVexxACE ( Real version )  -----

void
KohnSham::CalculateVexxACEDF ( Spinor& psi, Fourier& fft, bool isFixColumnDF )
{
  // TODO to be added here

  return;
}

// This comes from exxenergy2() function in exx.f90 in QE.
Real
KohnSham::CalculateEXXEnergy    ( Spinor& psi, Fourier& fft )
{
  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);
  
  Real fockEnergy = 0.0;
  Real fockEnergyLocal = 0.0;

  if( !fft.isInitialized ){
    ErrorHandling("Fourier is not prepared.");
  }

  bool spherecut = esdfParam.isUseSphereCut;
  bool useace    = esdfParam.isHybridACE;

  Real vol      = domain_.Volume();
  Int nspin     = domain_.numSpinComponent;
  Int ncom      = ( nspin == 4 ) ? 2 : 1;
  Int numACE    = ( nspin == 2 ) ? 2 : 1;
  Int ntot;
  if( spherecut )
    ntot = domain_.numGridSphere;
  else
    ntot = domain_.NumGridTotal();

  // The band number for each spin
  Int numStateTotal = psi.NumStateTotal();
  Int numStateLocal = psi.NumState();
  if( nspin == 2 )
  {
    numStateTotal /= 2;
    numStateLocal /= 2;
  }

  DblNumVec occSpin;
  for( Int ispin = 0; ispin < numACE; ispin++ ){
    if( nspin == 2 && !esdfParam.isHybridACE  ){
      occSpin.Resize( numStateTotal ); 
      blas::Copy( numStateTotal, &(occupationRate_[ispin*numStateTotal]), 1, occSpin.Data(), 1 );
    }

    if( !spherecut ){
      Spinor psiTemp( fft.domain, ncom, numStateTotal, false, 
          psi.Wavefun().VecData(0,ispin*numStateLocal) );  
      NumTns<Real>& wavefun = psiTemp.Wavefun();

      // Directly use the phiEXX_ and vexxProj_ to calculate the exchange energy
      if( useace ){
        // temporarily just implement here
        // Directly use projector   
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

        DblNumMat psiCol( ntot, numStateLocal );
        SetValue( psiCol, D_ZERO );

        DblNumMat psiRow( ntotLocal, numStateTotal );
        SetValue( psiRow, D_ZERO );

        DblNumMat vexxProjCol( ntot, numStateLocal );
        SetValue( vexxProjCol, D_ZERO );

        DblNumMat vexxProjRow( ntotLocal, numStateTotal );
        SetValue( vexxProjRow, D_ZERO );

        DblNumMat vexxPsiCol( ntot, numStateLocal );
        SetValue( vexxPsiCol, D_ZERO );

        DblNumMat vexxPsiRow( ntotLocal, numStateTotal );
        SetValue( vexxPsiRow, D_ZERO );
        lapack::Lacpy( 'A', ntot, numStateLocal, psiTemp.Wavefun().Data(), ntot, psiCol.Data(), ntot );
        if( numACE == 1 )    
          lapack::Lacpy( 'A', ntot, numStateLocal, vexxProj_.Data(), ntot, vexxProjCol.Data(), ntot );
        else{
          if( ispin == 0)
            lapack::Lacpy( 'A', ntot, numStateLocal, UpvexxProj_.Data(), ntot, vexxProjCol.Data(), ntot );
          else
            lapack::Lacpy( 'A', ntot, numStateLocal, DnvexxProj_.Data(), ntot, vexxProjCol.Data(), ntot ); 
        }
          
        AlltoallForward (psiCol, psiRow, domain_.comm);
        AlltoallForward (vexxProjCol, vexxProjRow, domain_.comm);

        DblNumMat MTemp( numStateTotal, numStateTotal );
        SetValue( MTemp, D_ZERO );

        blas::Gemm( 'T', 'N', numStateTotal, numStateTotal, ntotLocal,
            1.0, vexxProjRow.Data(), ntotLocal,
            psiRow.Data(), ntotLocal, 0.0,
            MTemp.Data(), numStateTotal );

        DblNumMat M(numStateTotal, numStateTotal);
        SetValue( M, D_ZERO );

        MPI_Allreduce( MTemp.Data(), M.Data(), numStateTotal * numStateTotal, MPI_DOUBLE, MPI_SUM, domain_.comm );

        blas::Gemm( 'N', 'N', ntotLocal, numStateTotal, numStateTotal, -1.0,
            vexxProjRow.Data(), ntotLocal, M.Data(), numStateTotal,
            0.0, vexxPsiRow.Data(), ntotLocal );

        AlltoallBackward (vexxPsiRow, vexxPsiCol, domain_.comm);

        for( Int j = 0; j < numStateLocal; j++ ){
          for( Int i = 0; i < ncom; i++ ){
            for( Int ir = 0; ir < ntot; ir++ ){
              fockEnergyLocal += (vexxPsiCol(ir+i*ntot,j) * wavefun(ir,i,j))
                  * occupationRate_[psi.WavefunIdx(j)+ispin*numStateTotal];
            }
          }
        }
      }  // ---- if( esdfParam.isHybridACE ) ----
      else{
        NumTns<Real>  vexxPsi( ntot, ncom, numStateLocal );
        SetValue( vexxPsi, D_ZERO );

        if( nspin == 1 ){
          psiTemp.AddMultSpinorEXX( fft, phiEXX_, exxgkkR2C_,
              exxFraction_, nspin, occupationRate_, vexxPsi );
        }
        else if ( nspin == 2 ){
          if( ispin == 0 )
            psiTemp.AddMultSpinorEXX( fft, UpphiEXX_, exxgkkR2C_,
              exxFraction_, nspin, occSpin, vexxPsi ); 
          else
            psiTemp.AddMultSpinorEXX( fft, DnphiEXX_, exxgkkR2C_,
              exxFraction_, nspin, occSpin, vexxPsi ); 
        }

        for( Int j = 0; j < numStateLocal; j++ ){
          for( Int i = 0; i < ncom; i++ ){
            for( Int ir = 0; ir < ntot; ir++ ){
              fockEnergyLocal += (vexxPsi(ir,i,j) * wavefun(ir,i,j))
                  * occupationRate_[psi.WavefunIdx(j)+ispin*numStateTotal];
            }
          }
        }
      }   
    }  
    else{
      Spinor psiTemp( fft.domain, ncom, numStateTotal, false, 
          psi.WavefunG().VecData(0,ispin*numStateLocal) );  
      NumTns<Complex>& wavefun = psiTemp.WavefunG();

      CpxNumMat vexxPsiCol; 
      // Directly use the phiEXX_ and vexxProj_ to calculate the exchange energy
      if( useace ){
        // temporarily just implement here
        // Directly use projector   
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

        CpxNumMat psiCol( ntot, numStateLocal );
        SetValue( psiCol, Z_ZERO );

        CpxNumMat psiRow( ntotLocal, numStateTotal );
        SetValue( psiRow, Z_ZERO );

        CpxNumMat vexxProjCol( ntot, numStateLocal );
        SetValue( vexxProjCol, Z_ZERO );

        CpxNumMat vexxProjRow( ntotLocal, numStateTotal );
        SetValue( vexxProjRow, Z_ZERO );

        vexxPsiCol.Resize( ntot, numStateLocal );
        SetValue( vexxPsiCol, Z_ZERO );

        CpxNumMat vexxPsiRow( ntotLocal, numStateTotal );
        SetValue( vexxPsiRow, Z_ZERO );
        lapack::Lacpy( 'A', ntot, numStateLocal, psiTemp.WavefunG().Data(), ntot, psiCol.Data(), ntot );
        if( numACE == 1 )    
          lapack::Lacpy( 'A', ntot, numStateLocal, vexxProjG_.Data(), ntot, vexxProjCol.Data(), ntot );
        else{
          if( ispin == 0)
            lapack::Lacpy( 'A', ntot, numStateLocal, UpvexxProjG_.Data(), ntot, vexxProjCol.Data(), ntot );
          else
            lapack::Lacpy( 'A', ntot, numStateLocal, DnvexxProjG_.Data(), ntot, vexxProjCol.Data(), ntot ); 
        }
          
        AlltoallForward (psiCol, psiRow, domain_.comm);
        AlltoallForward (vexxProjCol, vexxProjRow, domain_.comm);

        DblNumMat MTemp( numStateTotal, numStateTotal );
        SetValue( MTemp, D_ZERO );

        blas::Gemm( 'C', 'N', numStateTotal, numStateTotal, ntotLocal,
            1.0, vexxProjRow.Data(), ntotLocal,
            psiRow.Data(), ntotLocal, 0.0,
            MTemp.Data(), numStateTotal );

        DblNumMat M(numStateTotal, numStateTotal);
        SetValue( M, D_ZERO );

        MPI_Allreduce( MTemp.Data(), M.Data(), numStateTotal * numStateTotal, MPI_DOUBLE, MPI_SUM, domain_.comm );

        blas::Gemm( 'N', 'N', ntotLocal, numStateTotal, numStateTotal, -1.0,
            vexxProjRow.Data(), ntotLocal, M.Data(), numStateTotal,
            0.0, vexxPsiRow.Data(), ntotLocal );

        AlltoallBackward (vexxPsiRow, vexxPsiCol, domain_.comm);
      }  // ---- if( esdfParam.isHybridACE ) ----
      else{
        NumTns<Complex>  vexxPsi( ntot, ncom, numStateLocal );
        SetValue( vexxPsi, Z_ZERO );

        if( nspin == 1 ){
          psiTemp.AddMultSpinorEXX( fft, phiEXXG_, exxgkkR2C_,
              exxFraction_, nspin, occupationRate_, vexxPsi );
        }
        else if ( nspin == 2 ){
          if( ispin == 0 )
            psiTemp.AddMultSpinorEXX( fft, UpphiEXXG_, exxgkkR2C_,
              exxFraction_, nspin, occSpin, vexxPsi ); 
          else
            psiTemp.AddMultSpinorEXX( fft, DnphiEXXG_, exxgkkR2C_,
              exxFraction_, nspin, occSpin, vexxPsi ); 
        }

        vexxPsiCol = CpxNumMat( ntot, numStateLocal, true, vexxPsi.Data() );
      }
      
      for( Int j = 0; j < numStateLocal; j++ ){
        for( Int ir = 0; ir < ntot; ir++ ){
          fockEnergyLocal += (vexxPsiCol(ir,j) * std::conj(wavefun(ir,VAL,j))).real()
              * occupationRate_[psi.WavefunIdx(j)+ispin*numStateTotal];
        }
      }
    }  // ---- end of if(!spherecut)
  }  // for (ispin)

  MPI_Barrier( domain_.comm );
  mpi::Allreduce( &fockEnergyLocal, &fockEnergy, 1, MPI_SUM, domain_.comm );

  return( fockEnergy * numSpin_ / 2.0 );
}         // -----  end of method KohnSham::CalculateEXXEnergy ( Real version ) ----
#endif

} // namespace pwdft
