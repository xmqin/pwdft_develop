#include  "hamiltonian.hpp"
#include  "blas.hpp"
#include  "lapack.hpp"

namespace pwdft{

using namespace pwdft::PseudoComponent;
using namespace pwdft::DensityComponent;
using namespace pwdft::SpinTwo;
using namespace pwdft::GradThree;
using namespace pwdft::esdf;

#ifdef _COMPLEX_
void
KohnSham::CalculateDensity ( const std::vector<Spinor> &psi, const std::vector<DblNumVec> &occrate, 
    Real &val, Fourier &fft )
{
  SetValue( density_, 0.0 );
  SetValue( spindensity_, 0.0 );

  Real vol  = domain_.Volume();
  Int ntotFine  = fft.domain.NumGridTotalFine();

  MPI_Barrier(domain_.comm);

  DblNumMat densityLocal;
  densityLocal.Resize( ntotFine, numDensityComponent_ );
  SetValue( densityLocal, 0.0 );

  for( Int ispinor = 0; ispinor < psi.size(); ispinor++ ){

    const Spinor &psiTemp = psi[ispinor]; 
    const DblNumVec &occTemp = occrate[ispinor];

    Int ikLocal    = psiTemp.IkLocal();
    Int ntot  = psiTemp.NumGridTotal();
    Int ncom  = psiTemp.NumComponent();
    Int nocc  = psiTemp.NumState();
    Int nocc_total = psiTemp.NumStateTotal();

    if( numDensityComponent_ == 2 ){
      nocc /= 2;
      nocc_total /= 2;
    }

    Real fac;
    
    for (Int k=0; k<nocc; k++) {
      if( numDensityComponent_ <= 2 ){
        for (Int j=0; j<numDensityComponent_; j++) {

          if( esdfParam.isUseRealSpace == true ){
            for( Int i = 0; i < ntot; i++ ){
              fft.inputComplexVec(i) = psiTemp.Wavefun(i,RHO,k+j*nocc);
            }

            FFTWExecute ( fft, fft.forwardPlan );
          }
          else{
            fac = vol / std::sqrt( ntot );
            blas::Copy( ntot, psiTemp.Wavefun().VecData(RHO,k+j*nocc), 1, fft.outputComplexVec.Data(), 1 );
            blas::Scal( ntot, fac, fft.outputComplexVec.Data(), 1 ); 
          }

          SetValue( fft.outputComplexVecFine, Z_ZERO );
          for( Int i = 0; i < ntot; i++ ){
            if( esdfParam.isUseSphereCut == true )
              fft.outputComplexVecFine(fft.idxFineCut[ikLocal](i)) = fft.outputComplexVec(i) *
                sqrt( double(ntot) / double(ntotFine) );
            else
              fft.outputComplexVecFine(fft.idxFineGrid(i)) = fft.outputComplexVec(i) * 
                sqrt( double(ntot) / double(ntotFine) );
          } 

          FFTWExecute ( fft, fft.backwardPlanFine );

          fac = numSpin_ * occTemp(psiTemp.WavefunIdx(k)+j*nocc_total);
          for( Int i = 0; i < ntotFine; i++ ){
            densityLocal(i,j) +=  (fft.inputComplexVecFine(i) * std::conj(fft.inputComplexVecFine(i))).real()* fac;
          }
        }
      }
      else{
        Real Rpsiup,Rpsidw,Ipsiup,Ipsidw;
        CpxNumMat psiFine( ntotFine, numSpinorComponent_ );
        // Transform two-component spinor from coarse grids to dense grids
        for( Int is = 0; is < numSpinorComponent_; is++ ){
          if( esdfParam.isUseRealSpace == true ){
            for( Int i = 0; i < ntot; i++ ){
              fft.inputComplexVec(i) = psiTemp.Wavefun(i,is,k);
            }

            FFTWExecute ( fft, fft.forwardPlan );
          }
          else{
            fac = vol / std::sqrt( ntot );
            blas::Copy( ntot, psiTemp.Wavefun().VecData(is,k), 1, fft.outputComplexVec.Data(), 1 );
            blas::Scal( ntot, fac, fft.outputComplexVec.Data(), 1 );
          }

          SetValue( fft.outputComplexVecFine, Z_ZERO );
          for( Int i = 0; i < ntot; i++ ){
            if( esdfParam.isUseSphereCut == true )
              fft.outputComplexVecFine(fft.idxFineCut[ikLocal](i)) = fft.outputComplexVec(i) *
                sqrt( double(ntot) / double(ntotFine) );
            else
              fft.outputComplexVecFine(fft.idxFineGrid(i)) = fft.outputComplexVec(i) *
                sqrt( double(ntot) / double(ntotFine) );
          }

          FFTWExecute ( fft, fft.backwardPlanFine );
          blas::Copy( ntotFine, fft.inputComplexVecFine.Data(), 1, psiFine.VecData(is), 1 );
        } // for (is)

        for( Int i = 0; i < ntotFine; i++ ){
          Rpsiup = psiFine(i,UP).real();
          Ipsiup = psiFine(i,UP).imag();
          Rpsidw = psiFine(i,DN).real();
          Ipsidw = psiFine(i,DN).imag();

          fac = numSpin_ * occTemp(psiTemp.WavefunIdx(k));

          densityLocal(i,0) += fac * ( pow( Rpsiup, 2.0 ) + pow( Rpsidw, 2.0 ) +
                                pow( Ipsiup, 2.0 ) + pow( Ipsidw, 2.0 ) );
          densityLocal(i,1) += 2 * fac * ( Rpsiup * Rpsidw + Ipsiup * Ipsidw );
          densityLocal(i,2) += 2 * fac * ( Rpsiup * Ipsidw - Rpsidw * Ipsiup );
          densityLocal(i,3) += fac * ( pow( Rpsiup, 2.0 ) - pow( Rpsidw, 2.0 ) +
                                pow( Ipsiup, 2.0 ) - pow( Ipsidw, 2.0 ) );
        }
      }  // ---- end of if ( numDensityComponent <= 2 ) ----
    }  // for (k)
  }  // for (ispinor)

  if( numDensityComponent_ == 1 || numDensityComponent_ == 4 ){
    mpi::Allreduce( densityLocal.Data(), density_.Data(), ntotFine*numDensityComponent_, MPI_SUM, domain_.comm );
  }
  else{
    mpi::Allreduce( densityLocal.Data(), spindensity_.Data(), ntotFine*numDensityComponent_, MPI_SUM, domain_.comm );
  }

  if( numDensityComponent_ == 2 ){
    // Transform spindensity to density(arho, drho) 
    blas::Copy( ntotFine, spindensity_.VecData(0), 1, density_.VecData(0), 1 );
    blas::Copy( ntotFine, spindensity_.VecData(0), 1, density_.VecData(1), 1 );
    // arho = rhoup + rhodw
    // drho = rhoup - rhodw
    blas::Axpy( ntotFine, 1.0, spindensity_.VecData(1), 1, density_.VecData(0), 1);
    blas::Axpy( ntotFine, -1.0, spindensity_.VecData(1), 1, density_.VecData(1), 1);
  }

  val = 0.0; // sum of density

  for (Int i=0; i<ntotFine; i++) {
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
  else if( numDensityComponent_ == 4 ){
    Point3 Vecmag( 0.0, 0.0, 0.0 );
    for( Int is = 1; is < numDensityComponent_; is++ ){
      for( Int i = 0; i < ntotFine; i++ ){
        Vecmag[is-1] += density_(i, is) * vol / ntotFine;
      }
    }
    statusOFS << "The vector magnetization moment      = " << Vecmag << std::endl;
  }

  if( numDensityComponent_ == 4 ){
    // Rotate four-component density through a unitary matrix to up and down format     
    // such that we can use LSDA to calculate functional
    Real dotmag, amag, vecdot, Tempsegni;
    Real epsDot = 1e-16;
    Point3 tempmag;
    bool isParallel = esdfParam.isParallel;

    for( Int i = 0; i < ntotFine; i++){
      tempmag = Point3(  density_(i,1), density_(i,2), density_(i,3) );
      
      if( isParallel ){
        vecdot = tempmag[0]*spinaxis_[0] + tempmag[1]*spinaxis_[1] 
            + tempmag[2]*spinaxis_[2];

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
      spindensity_(i,UP) = 0.5 * ( density_(i,RHO) + Tempsegni*amag );
      spindensity_(i,DN) = 0.5 * ( density_(i,RHO) - Tempsegni*amag );

      segni_[i] = Tempsegni;
    }
  }

  return ;
}         // -----  end of method KohnSham::CalculateDensity ( Complex version )  ----- 

void
KohnSham::CalculateEkin( Fourier& fft )
{
  Domain& dm = fft.domain;
 
  IntNumVec& KpointIdx = dm.KpointIdx;
  Int nkLocal = KpointIdx.Size();
  Int ntot;
  Point3 kpoint = Point3( 0.0, 0.0, 0.0 );
  Point3 kG     = Point3( 0.0, 0.0, 0.0 );
  Real a, b;

  for( Int k = 0; k < nkLocal; k++ ){
    Int ik = KpointIdx(k);

    if( esdfParam.isUseSphereCut == true )
      ntot = dm.numGridSphere[k];
    else
      ntot = dm.NumGridTotal();

    IntNumVec &idxFineCut = fft.idxFineCut[k];
    kpoint = Point3( dm.klist[0][ik], dm.klist[1][ik], dm.klist[2][ik] );
    ekin_[k].Resize( ntot ); teter_[k].Resize( ntot );
    for( Int i = 0; i < ntot; i++ ){

      if( esdfParam.isUseSphereCut == true ){
        Int ig = idxFineCut[i]; 
        kG = ( kpoint + Point3( fft.ikFine[0][ig].imag(), fft.ikFine[1][ig].imag(),
          fft.ikFine[2][ig].imag() ) );
      }
      else{
        kG = ( kpoint + Point3( fft.ik[0][i].imag(), fft.ik[1][i].imag(),
            fft.ik[2][i].imag() ) );
      }

      ekin_[k][i] = ( kG[0]*kG[0] + kG[1]*kG[1] + kG[2]*kG[2] ) / 2;
    
      a = ekin_[k][i] * 2.0;
      b = 27.0 + a * (18.0 + a * (12.0 + a * 8.0) );
      teter_[k][i] = b / ( b + 16.0 * pow(a, 4.0) );
    }
  }

  return;
}         // -----  end of method KohnSham::CalculateEkin ( Complex version )  ----- 

void
KohnSham::CalculateForce    ( PeriodTable &ptable, std::vector<Spinor>& psi, Fourier& fft  )
{
  Real timeSta, timeEnd;

  MPI_Barrier(domain_.comm);
  int mpirank;  MPI_Comm_rank(domain_.comm, &mpirank);
  int mpisize;  MPI_Comm_size(domain_.comm, &mpisize);

  bool usevlocal = esdfParam.isUseVLocal;
  bool realspace = esdfParam.isUseRealSpace;
  bool spherecut = esdfParam.isUseSphereCut;
  bool lspinorb = fft.domain.SpinOrbitCoupling;

  Int nkLocal = psi.size();
  Int ntotFine  = fft.domain.NumGridTotalFine();
  Int ncom  = psi[0].NumComponent();
  Int numStateLocal = psi[0].NumState(); // Local number of states
  Int numStateTotal = psi[0].NumStateTotal();
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

      DblNumVec totalCharge( ntotFine );
      SetValue( totalCharge, 0.0 );
      // totalCharge = density_ - pseudoCharge
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
    }
    else{
      std::set<Int> atomTypeSet;
      for( Int a = 0; a < numAtom; a++ ){
        atomTypeSet.insert( atomList_[a].type );
      }

      IntNumVec& idxDensity = fft.idxFineCutDensity;
      Int ntotDensity = idxDensity.Size();

      CpxNumVec densityG( ntotDensity );
      DblNumVec vlocR( ntotDensity );
      
      for( Int i = 0; i < ntotFine; i++ ){
        fft.inputComplexVecFine(i) = Complex( density_(i,RHO), 0.0 );
      }

      FFTWExecute ( fft, fft.forwardPlanFine );    

      for( Int i = 0; i < ntotDensity; i++ ){
        densityG(i) = fft.outputComplexVecFine[idxDensity(i)];
      }

      for( std::set<Int>::iterator itype = atomTypeSet.begin();
        itype != atomTypeSet.end(); itype++ ){
        Int atype = *itype;
        Atom fakeAtom;
        fakeAtom.type = atype;
        fakeAtom.pos = domain_.posStart;

        SetValue( vlocR, 0.0 );
        ptable.CalculateVLocal( fakeAtom, domain_, fft, vlocR );   
        Real fac_vlocal = 4 * PI / vol;
        blas::Scal( ntotDensity, fac_vlocal, vlocR.Data(), 1 );

        Complex* ikxPtr = fft.ikFine[0].Data();
        Complex* ikyPtr = fft.ikFine[1].Data();
        Complex* ikzPtr = fft.ikFine[2].Data();
        Real xx, yy, zz;
        Real arg, argrhog;

        for( Int i = 0; i < numAtomLocal; i++ ){
          int a = numAtomIdx[i];
          if( atomList_[a].type == atype ){
            xx = atomList_[a].pos[0];
            yy = atomList_[a].pos[1];
            zz = atomList_[a].pos[2];
            for( Int i = 0; i < ntotDensity; i++ ){
              Int ig = idxDensity(i);
              arg = ikxPtr[ig].imag() * xx + ikyPtr[ig].imag() * yy 
                  + ikzPtr[ig].imag() * zz;
              argrhog = std::sin(arg) * densityG[i].real() 
                  + std::cos(arg) * densityG[i].imag();
              force( a, 0 ) += ikxPtr[ig].imag() * vlocR[i] * argrhog;
              force( a, 1 ) += ikyPtr[ig].imag() * vlocR[i] * argrhog;
              force( a, 2 ) += ikzPtr[ig].imag() * vlocR[i] * argrhog;
            }
          }
        } // for (a)
      }
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
    std::vector<DblNumVec>& klist = fft.domain.klist;
    IntNumVec& KpointIdx = fft.domain.KpointIdx; 

    CpxNumMat                psiFine( ntotFine, ncom );
    CpxNumMat                psiFourier( ntotFine, ncom );
    std::vector<CpxNumMat>   psiDrvFine(DIM);

    for( Int d = 0; d < DIM; d++ ){
      psiDrvFine[d].Resize( ntotFine, ncom );
    }

    for( Int k = 0; k < nkLocal; k++ ){
      Int ik = KpointIdx(k);
      Real kx = klist[0][ik];
      Real ky = klist[1][ik];
      Real kz = klist[2][ik];
      Int ntot = psi[k].NumGridTotal();

      // Loop over atoms and pseudopotentials     
      for( Int g = 0; g < numStateLocal; g++ ){
        // Compute the derivative of the wavefunctions on a fine grid
        for( Int is = 0; is < ncom; is++ ){         
          Complex* psiPtr = psi[k].Wavefun().VecData(is, g);

          if( realspace ){
            blas::Copy( ntot, psiPtr, 1, fft.inputComplexVec.Data(), 1 );
            FFTWExecute ( fft, fft.forwardPlan );
          }
          else{
            SetValue( fft.outputComplexVec, Z_ZERO );
            blas::Copy( ntot, psiPtr, 1, fft.outputComplexVec.Data(), 1 );
            Real fac_psi = vol / sqrt(double(domain_.NumGridTotal()));
            blas::Scal( ntot, fac_psi, fft.outputComplexVec.Data(), 1 );
          } 
          // Interpolate wavefunction from coarse to fine grid        
          SetValue( fft.outputComplexVecFine, Z_ZERO );
          Int *idxPtr = NULL;
          if( spherecut )
            idxPtr = fft.idxFineCut[k].Data();
          else
            idxPtr = fft.idxFineGrid.Data();

          Complex *fftOutFinePtr = fft.outputComplexVecFine.Data();
          Complex *fftOutPtr = fft.outputComplexVec.Data();
          for( Int i = 0; i < ntot; i++ ){
            fftOutFinePtr[*(idxPtr++)] = *(fftOutPtr++);
          }

          blas::Copy( ntotFine, fft.outputComplexVecFine.Data(), 1, 
              psiFourier.VecData(is), 1 );

          FFTWExecute ( fft, fft.backwardPlanFine );

          Real fac = sqrt(double(domain_.NumGridTotal())) /
              sqrt( double(domain_.NumGridTotalFine()) );

          blas::Copy( ntotFine, fft.inputComplexVecFine.Data(),
              1, psiFine.VecData(is), 1 );
          blas::Scal( ntotFine, fac, psiFine.VecData(is), 1 );

          // derivative of psi on a fine grid
          for( Int d = 0; d < DIM; d++ ){
            Complex* ikFinePtr     = fft.ikFine[d].Data();
            Complex* psiFourierPtr = psiFourier.VecData(is);
            Complex* fftOutFinePtr = fft.outputComplexVecFine.Data();
            for( Int i = 0; i < ntotFine; i++ ){
              *(fftOutFinePtr++) = *(psiFourierPtr++) * *(ikFinePtr++);
            }

            FFTWExecute ( fft, fft.backwardPlanFine );

            blas::Copy( ntotFine, fft.inputComplexVecFine.Data(),
                1, psiDrvFine[d].VecData(is), 1 );
            blas::Scal( ntotFine, fac, psiDrvFine[d].VecData(is), 1 );
          } // for (d)
        } // for (is)

        // Evaluate the contribution to the atomic force
        for( Int a = 0; a < numAtom; a++ ){
          std::vector<NonlocalPP>& vnlList = pseudo_[a].vnlList;
          const CpxNumTns &coefMat = pseudo_[a].coefMat;
          const CpxNumVec &phase = pseudo_[a].vnlPhase[k];  
          Int nbeta = vnlList.size();
          // Save the inner product <\beta|psi> and <\beta|dpsi> if SOC is included
          std::vector<CpxNumMat> res_save;
          if( lspinorb ){
            res_save.resize(4);
            for(Int i = 0; i < 4; i++ ){
              res_save[i].Resize(nbeta, ncom);
            }
          }
                 
          for( Int l = 0; l < vnlList.size(); l++ ){
            SparseVec& bl = vnlList[l].first;             
            Real  wgt = domain_.Volume() / domain_.NumGridTotalFine();
            IntNumVec& idx = bl.first;
            DblNumMat& val = bl.second;

            CpxNumMat res(4, ncom);
            SetValue( res, Complex(0.0,0.0) );
            for( Int is = 0; is < ncom; is++ ){               
              Complex* psiPtr = psiFine.VecData(is);
              Complex* DpsiXPtr = psiDrvFine[0].VecData(is);
              Complex* DpsiYPtr = psiDrvFine[1].VecData(is);
              Complex* DpsiZPtr = psiDrvFine[2].VecData(is);
              Real* valPtr   = val.VecData(VAL);
              Int*  idxPtr = idx.Data();
              for( Int i = 0; i < idx.Size(); i++ ){
                res(VAL,is) += psiPtr[ *idxPtr ] * phase[i] * sqrt(wgt) * (*valPtr);
                res(DX, is)  += ( DpsiXPtr[ *idxPtr ] + psiPtr[ *idxPtr ] * Complex(0.0, kx) )
                    * (*valPtr) * phase[i] * sqrt(wgt);
                res(DY, is)  += ( DpsiYPtr[ *idxPtr ] + psiPtr[ *idxPtr ] * Complex(0.0, ky) )
                    * (*valPtr) * phase[i] * sqrt(wgt);
                res(DZ, is)  += ( DpsiZPtr[ *idxPtr ] + psiPtr[ *idxPtr ] * Complex(0.0, kz) )
                    * (*valPtr) * phase[i] * sqrt(wgt);
                valPtr++;
                idxPtr++;
              }
            }

            if( lspinorb ){
              for( Int is = 0; is < ncom; is++ ){
                for( Int d = 0; d < 4; d++ ){
                  res_save[d](l,is) = res(d,is);
                }
              }
            }

            if( !lspinorb ){
              for( Int is = 0; is < ncom; is++ ){              
                Real gamma = vnlList[l].second;
                Real occrate;
                if( numDensityComponent_ == 2 ){
                  occrate = occupationRate_[k]( psi[k].WavefunIdx(g%(numStateLocal/2)) + 
                      (g/(numStateLocal/2)) * numStateTotal / 2 );
                }
                else{
                  occrate = occupationRate_[k](psi[k].WavefunIdx(g));
                }
 
                forceLocal( a, 0 ) += -2.0 * occrate
                    * gamma * (res(VAL, is) * std::conj(res(DX, is))).real() * numSpin_;
                forceLocal( a, 1 ) += -2.0 * occrate
                    * gamma * (res(VAL, is) * std::conj(res(DY, is))).real() * numSpin_;
                forceLocal( a, 2 ) += -2.0 * occrate
                    * gamma * (res(VAL, is) * std::conj(res(DZ, is))).real() * numSpin_;
              }
            }
          } // for (l)

          if( lspinorb ){   
            CpxNumMat coefw( nbeta, ncom ); 
            Complex sum;

            for( Int d = 0; d < DIM; d++ ){      
              Int ijs = 0;
              for( Int is = 0; is < ncom; is++ ){ 
                for( Int js = 0; js < ncom; js++ ){
                  CpxNumMat gamma = CpxNumMat( nbeta, nbeta, true, coefMat.MatData(ijs++));
  
                  SetValue( coefw, Z_ZERO );
                  sum = Z_ZERO;

		              for( Int i = 0; i < nbeta; i++ ){
                    for( Int j = 0; j < nbeta; j++ ){
                      coefw(i,0) += gamma(i,j) * res_save[0](j,js);
                      coefw(i,1) += gamma(i,j) * res_save[d+1](j,js);
                    } 
                  }

                  for( Int i = 0; i < nbeta; i++ ){
                    sum += std::conj(res_save[0](i, is)) * coefw(i,1) + std::conj(res_save[d+1](i, is)) * coefw(i,0);
                  }

                  forceLocal( a, d ) += -1.0 * occupationRate_[k](psi[k].WavefunIdx(g)) * sum.real();
                }
              }  
            }                         
          }
        } // for (a)
      } // for (g)
    } // for (k)
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
    
    if( !realspace ){
      for( Int a = 0; a < atomList.size(); a++ ){
        atomList[a].force += Point3( forceIonLR_(a,0), forceIonLR_(a,1), forceIonLR_(a,2) );
      }
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
}         // -----  end of method KohnSham::CalculateForce ( Complex version )  ----- 

void
KohnSham::MultSpinor    ( Spinor& psi, NumTns<Complex>& a3, Fourier& fft )
{
  MPI_Comm comm = domain_.colComm_kpoint;

  MPI_Barrier(comm);
  int mpirank;  MPI_Comm_rank(comm, &mpirank);
  int mpisize;  MPI_Comm_size(comm, &mpisize);

  bool spherecut = esdfParam.isUseSphereCut;
  bool useace = esdfParam.isHybridACE;

  Int ntotFine  = fft.domain.NumGridTotalFine();
  Int nkLocal   = fft.domain.KpointIdx.Size(); 
  Int nspin     = numDensityComponent_;
  Int ncom      = ( nspin == 4 ) ? 2 : 1;

  Int ntot = psi.NumGridTotal();
  Int ntot2 = ntot * ncom;
  Int ntotLocal = psi.NumGrid();  
  Int ntotLocal2 = ntotLocal * ncom;

  Int numStateTotal = psi.NumStateTotal();
  Int numStateLocal = psi.NumState();

  NumTns<Complex>& wavefun = psi.Wavefun();

  Int ntotR2C = fft.numGridTotal;

  Real timeSta, timeEnd;
  Real timeSta1, timeEnd1;

  Real timeGemm = 0.0;
  Real timeAlltoallv = 0.0;
  Real timeAllreduce = 0.0;

  SetValue( a3, Complex(0.0,0.0) );

  // Apply an initial filter on the wavefunctions, if required
  if((apply_filter_ == 1 && apply_first_ == 1))
  {
    apply_first_ = 0;

    for (Int k=0; k<numStateLocal; k++) {
      for (Int j=0; j<ncom; j++) {

        SetValue( fft.inputComplexVec,  Z_ZERO);
        SetValue( fft.outputComplexVec, Z_ZERO );

        blas::Copy( ntot, wavefun.VecData(j,k), 1,
            fft.inputComplexVec.Data(), 1 );
        FFTWExecute ( fft, fft.forwardPlan ); // So outputComplexVec contains the FFT result now

        for (Int i=0; i<ntotR2C; i++)
        {
          if(fft.gkk(i) > wfn_cutoff_)
            fft.outputComplexVec(i) = Z_ZERO;
        }

        FFTWExecute ( fft, fft.backwardPlan);
        blas::Copy( ntot,  fft.inputComplexVec.Data(), 1,
            wavefun.VecData(j,k), 1 );
      }
    }
  }

  GetTime( timeSta );
  // For spin-restricted and spin-unrestricted calculations, vtot is inputed
  // as DblNumVec just as real version; But for spin-noncollinear case, vtot_
  // is inputed as a DblNumMat 
  if( nspin == 4 ){
    psi.AddMultSpinorFine( fft, ekin_, vtot_, pseudo_, a3 );
  }
  else{
    DblNumVec vtot( ntotFine ); 
    blas::Copy( ntotFine, vtot_.VecData( spinswitch_ ), 1, vtot.Data(), 1 );
    psi.AddMultSpinorFine( fft, ekin_, vtot, pseudo_, a3 );
  }
  
  if( isHybrid_ && isEXXActive_ ){

    GetTime( timeSta );

    if( esdfParam.isHybridACE ){
      // Convert the column partition to row partition
      
      Int numOccLocal, numOccTotal;

      if( nspin == 1 || nspin == 4 ){
        numOccLocal = vexxProj_[psi.IkLocal()].n();
      }
      else{
        if( spinswitch_ == 0 ){
          numOccLocal = UpvexxProj_[psi.IkLocal()].n();
        }
        else{
          numOccLocal = DnvexxProj_[psi.IkLocal()].n();
        }
      }

      MPI_Allreduce( &numOccLocal, &numOccTotal, 1, MPI_INT, MPI_SUM, comm );

      CpxNumMat psiCol( ntot2, numStateLocal );
      SetValue( psiCol, Z_ZERO );

      CpxNumMat vexxProjCol( ntot2, numOccLocal );
      SetValue( vexxProjCol, Z_ZERO );

      CpxNumMat psiRow( ntotLocal2, numStateTotal );
      SetValue( psiRow, Z_ZERO );

      CpxNumMat vexxProjRow( ntotLocal2, numOccTotal );
      SetValue( vexxProjRow, Z_ZERO );
        
      lapack::Lacpy( 'A', ntot2, numStateLocal, psi.Wavefun().Data(), ntot2, psiCol.Data(), ntot2 );

      if( nspin == 1 || nspin == 4 )
        lapack::Lacpy( 'A', ntot2, numOccLocal, vexxProj_[psi.IkLocal()].Data(), 
            ntot2, vexxProjCol.Data(), ntot2 );
      else{
        if( spinswitch_ == 0 )
          lapack::Lacpy( 'A', ntot2, numOccLocal, UpvexxProj_[psi.IkLocal()].Data(), 
            ntot2, vexxProjCol.Data(), ntot2 );
        else
          lapack::Lacpy( 'A', ntot2, numOccLocal, DnvexxProj_[psi.IkLocal()].Data(), 
            ntot2, vexxProjCol.Data(), ntot2 );
      }
       
      Int mb = esdfParam.BlockSizeGrid;
      Int nb = esdfParam.BlockSizeState;

      AlltoallForward( mb, nb, ncom, psiCol, psiRow, comm );

      AlltoallForward( mb, nb, ncom, vexxProjCol, vexxProjRow, comm );

      GetTime( timeEnd1 );
      timeAlltoallv = timeAlltoallv + ( timeEnd1 - timeSta1 );

      CpxNumMat MTemp( numOccTotal, numStateTotal );
      SetValue( MTemp, Z_ZERO );

      GetTime( timeSta1 );
      blas::Gemm( 'C', 'N', numOccTotal, numStateTotal, ntotLocal2,
          1.0, vexxProjRow.Data(), ntotLocal2,
          psiRow.Data(), ntotLocal2, 0.0,
          MTemp.Data(), numOccTotal );
      GetTime( timeEnd1 );
      timeGemm = timeGemm + ( timeEnd1 - timeSta1 );

      CpxNumMat M(numOccTotal, numStateTotal);
      SetValue( M, Z_ZERO );
      GetTime( timeSta1 );
      MPI_Allreduce( MTemp.Data(), M.Data(), 2*numOccTotal*numStateTotal, MPI_DOUBLE, MPI_SUM, comm );
      GetTime( timeEnd1 );
      timeAllreduce = timeAllreduce + ( timeEnd1 - timeSta1 );

      CpxNumMat a3Col( ntot2, numStateLocal );
      SetValue( a3Col, Z_ZERO );

      CpxNumMat a3Row( ntotLocal2, numStateTotal );
      SetValue( a3Row, Z_ZERO );

      GetTime( timeSta1 );
      blas::Gemm( 'N', 'N', ntotLocal2, numStateTotal, numOccTotal,
          -1.0, vexxProjRow.Data(), ntotLocal2,
          M.Data(), numOccTotal, 0.0,
          a3Row.Data(), ntotLocal2 );
      GetTime( timeEnd1 );
      timeGemm = timeGemm + ( timeEnd1 - timeSta1 );

      AlltoallBackward( mb, nb, ncom, a3Row, a3Col, comm );

      GetTime( timeEnd1 );
      timeAlltoallv = timeAlltoallv + ( timeEnd1 - timeSta1 );

      GetTime( timeSta1 );
      for( Int k = 0; k < numStateLocal; k++ ){
        Complex *p1 = a3Col.VecData(k);
        Complex *p2 = a3.VecData(0, k);
        for( Int i = 0; i < ntot2; i++ ){
          *(p2++) += *(p1++);
        }
      }

      GetTime( timeEnd1 );
      timeGemm = timeGemm + ( timeEnd1 - timeSta1 );
    }
    else{
      // TODO this part should be calculated as the calculation of ACE 
#if 0
      if( nspin == 1 || nspin == 4 )
        psi.AddMultSpinorEXX( fft, phiEXX_, exxgkk_,
          exxFraction_, nspin, occupationRate_, a3 );
      else{
        std::vector<DblNumVec> occSpin( nkLocal );
        for( Int k = 0; k < nkLocal; k++ ){
          Int numStateTotal = occupationRate_[k].Size() / 2;
          occSpin[k].Resize( numStateTotal );
          blas::Copy( numStateTotal, &(occupationRate_[k][spinswitch_*numStateTotal]), 1, occSpin[k].Data(), 1 );
        }

        if( spinswitch_ == 0 )
          psi.AddMultSpinorEXX( fft, UpphiEXX_, exxgkk_,
              exxFraction_, nspin, occSpin, a3 );  
        else
          psi.AddMultSpinorEXX( fft, DnphiEXX_, exxgkk_,
              exxFraction_, nspin, occSpin, a3 );         
      }
      GetTime( timeEnd );
#endif
    }
  } // ---- if( isHybrid_ && isEXXActive_ ) ----

  // Apply filter on the wavefunctions before exit, if required
  if((apply_filter_ == 1))
  {
    for (Int k=0; k<numStateLocal; k++) {
      for (Int j=0; j<ncom; j++) {
        SetValue( fft.inputComplexVec, Z_ZERO );
        SetValue( fft.outputComplexVec, Z_ZERO );

        blas::Copy( ntot, a3.VecData(j,k), 1,
            fft.inputComplexVec.Data(), 1 );
        FFTWExecute ( fft, fft.forwardPlan ); // So outputVecR2C contains the FFT result now


        for (Int i=0; i<ntotR2C; i++)
        {
          if(fft.gkk(i) > wfn_cutoff_)
            fft.outputComplexVec(i) = Z_ZERO;
        }

        FFTWExecute ( fft, fft.backwardPlan );
        blas::Copy( ntot,  fft.inputComplexVec.Data(), 1,
            a3.VecData(j,k), 1 );
      }
    }
  }

  return ;
}         // -----  end of method KohnSham::MultSpinor ( Complex version )  ----- 

void KohnSham::InitializeEXX ( Real ecutWavefunction, Fourier& fft )
{
  Domain &dm = fft.domain; 

  bool spherecut  = esdfParam.isUseSphereCut;
  bool energyband = esdfParam.isCalculateEnergyBand;
  bool useisdf    = esdfParam.isHybridDF;

  const Real epsDiv = 1e-8;
  isEXXActive_      = false;

  IntNumVec& KpointIdx = dm.KpointIdx;

  Int nkTotal = (energyband == true) ? dm.NumKGridSCFTotal() : dm.NumKGridTotal();
  Int nkLocal = KpointIdx.Size();

  Int ntot = dm.NumGridTotal();
  Int npw;
  Int nkTemp;

  IntNumVec &idxc = fft.idxFineFock; // The mapping index for truncated Fock grids
  if( spherecut )
    npw = idxc.m();
  else
    npw = ntot;

  if( useisdf )
    nkTemp = nkTotal;
  else
    nkTemp = nkLocal;

  // extra 2.0 factor for ecutWavefunction compared to QE due to unit difference
  // tpiba2 in QE is just a unit for G^2. Do not include it here
  Real exxAlpha = 10.0 / (ecutWavefunction * 2.0);

  // Gygi-Baldereschi regularization. Currently set to zero and compare
  // with QE without the regularization 
  // Set exxdiv_treatment to "none"
  //
  // Now I have completed the part of 'Gygi-Baldereschi' regularization.
  // The inner numerical integration formula is still in doubt.
  // However, compared with QE, I got a nice payoff.
  //
  // Compute the divergent term for |k-q+G|=0
  Real gkk2;
  Int  iftruncated;
  Point3 kpoint = Point3( 0.0, 0.0, 0.0 );
  Point3 qpoint = Point3( 0.0, 0.0, 0.0 );
  Point3 kG     = Point3( 0.0, 0.0, 0.0 );

  // If the truncation method is adopted, the corresponding reciprocal space 
  // Coulomb potential will be different, so it is necessary to make a judgment.  
  iftruncated = 0;    

  // For HSE06 and PBE0, the Coulumb kernel is different.
  // Through judging 'screenMu_', it can take different forms.
  if(exxDivergenceType_ == 0){
    // None
    exxDiv_ = 0.0;
  }
  else if (exxDivergenceType_ == 1){
    // Gygi-Baldereschi regularization
    exxDiv_ = 0.0;

    for( Int k = 0; k < nkTotal; k++ ){
      kpoint = Point3( dm.klist[0][k], dm.klist[1][k], dm.klist[2][k] );
        
      for( Int i = 0; i < npw; i++ ){
        if( spherecut ){
          Int index = idxc[i];
          kG = ( kpoint + Point3( fft.ik[0][index].imag(), fft.ik[1][index].imag(),
              fft.ik[2][index].imag() ) );
        }
        else{
          kG = ( kpoint + Point3( fft.ik[0][i].imag(), fft.ik[1][i].imag(),
              fft.ik[2][i].imag() ) );
        }
        gkk2 = kG[0] * kG[0] + kG[1] * kG[1] + kG[2] * kG[2];
        if( gkk2 > epsDiv ){
          if( screenMu_ > 0 ){
            exxDiv_ += std::exp(-exxAlpha * gkk2) / gkk2 * 
                (1.0 - std::exp( -gkk2 / (4.0*screenMu_*screenMu_) ));     
          }
          else{
            exxDiv_ += std::exp(-exxAlpha * gkk2) / gkk2;
          }
        } // for if
      } // for i
    } // for k

    if( screenMu_ > 0.0 ){
      exxDiv_ += 1.0 / (4.0*screenMu_*screenMu_);
    }
    else{
      exxDiv_ += 0.0;
    }

    exxDiv_ *= 4.0 * PI;
    
    // This program runs the range of non-singularities for 'nk*nk' times,
    // but for the singularity only 'nk' times. Now it is for "nkTotal*nkLoca"l times.
    // I think it maybe cause some Minimal errors after the Auxiliary Function.
    // According to the test results, this averaging algorithm is more profitable than 
    // finding the corresponding singularity for patching.

    // Following is the analytic integral solution of Auxiliary Function.
    Real aa = 0.0;

    if( screenMu_ > 0.0 ){
      aa = + 1.0 / std::sqrt(exxAlpha*PI) - 
          1.0 / std::sqrt(PI*(exxAlpha + 1.0 / (4.0*screenMu_*screenMu_ )));
    }
    else{
      aa = + 1.0 / std::sqrt(exxAlpha*PI);	
    }
    exxDiv_ -= domain_.Volume()*aa*nkTotal;  
    // The results of the analytical solution written in the annotation
    // are exactly the same as the results of the numerical integral solution.
  } 
  else if (exxDivergenceType_ == 2){
    // Sphere truncation 
    if( screenMu_ > 0.0 ){
      ErrorHandling( "For HSE06 -- the short-range Coulomb kernels, the formula is too complex to be implemented here, we just set it to be 0.");
      exxDiv_ = 0.0;
    }
    else{
      exxDiv_ = 0.0;
      iftruncated = 1;
    }   
  }
  else if (exxDivergenceType_ == 3){
    // Wigner-Seize truncation
    if( screenMu_ > 0.0 ){
      ErrorHandling( "For HSE06 -- the short-range Coulomb kernels, the formula is not available,we just set it to be 0."); 
      exxDiv_ = 0.0;
    }
    else{
      exxDiv_ = 0.0;
    }
  }  
  else if (exxDivergenceType_ == 4){
    ErrorHandling( "Extrapolation for exxDiv with q-point has not been implemented" );
  }

  statusOFS << "computed exxDiv_ = " << exxDiv_ << std::endl;
  // In QE, k+q=kq, i is index of k, j is index of kq.
  // The input argument of the formulation of exxgkk is k-kq=-q.
  // See function g2_convolution of exx_base.f90 in QE.  
 
  if (iftruncated == 0){     
    if( !useisdf ){        
      // For "none" and "gygi-method" 
      exxgkk_.Resize( npw, nkTemp, nkTotal );
      SetValue( exxgkk_, 0.0 );

      for( Int k = 0; k < nkTemp; k++ ){
        Int ik = (useisdf == true) ? k : KpointIdx(k);
        kpoint = Point3( dm.klist[0][ik], dm.klist[1][ik], dm.klist[2][ik] );
        for( Int q = 0; q < nkTotal; q++ ){
          if( !energyband ){
            qpoint = Point3( dm.klist[0][q], dm.klist[1][q], dm.klist[2][q] );
          }
          else{
            qpoint = Point3( dm.klist_scf[0][q], dm.klist_scf[1][q], dm.klist_scf[2][q] );
          }
          for( Int i = 0; i < npw; i++ ){
  
            if( spherecut ){
              Int ig = idxc[i];
              kG = ( kpoint - qpoint + Point3( fft.ik[0][ig].imag(), fft.ik[1][ig].imag(),
                  fft.ik[2][ig].imag() ) );
            } 
            else{                
              kG = ( kpoint - qpoint + Point3( fft.ik[0][i].imag(), fft.ik[1][i].imag(),
                  fft.ik[2][i].imag() ) );
            }

            gkk2 = kG[0] * kG[0] + kG[1] * kG[1] + kG[2] * kG[2];

            if( gkk2 > epsDiv ){
              if( screenMu_ > 0 ){
                // 2.0*pi instead 4.0*pi due to gkk includes a factor of 2
                exxgkk_(i,k,q) = 4.0 * PI / gkk2 * (1.0 - 
                    std::exp( -gkk2 / (4.0*screenMu_*screenMu_) ));
              }
              else{
                exxgkk_(i,k,q) = 4.0 * PI / gkk2;
              }
            }
            else{
              exxgkk_(i,k,q) = -exxDiv_;
              if( screenMu_ > 0 ){
                exxgkk_(i,k,q) += PI / (screenMu_*screenMu_);
              }
              else{
                exxgkk_(i,k,q) += 0.0;
              }
            }
          } // for (i)
        } // for (q)
      } // for (k)
    }
    else{
      Index3 &numKGrid = domain_.numKGrid; 
      // The size of grid for saving k-q is (2N1-1)*(2N2-1)*(2N3-1)
      Int nkx = 2 * numKGrid[0] - 1;
      Int nky = 2 * numKGrid[1] - 1;
      Int nkz = 2 * numKGrid[2] - 1;
      Int Nkq = nkx * nky * nkz;
      Real kx, ky, kz;

      exxgkk_.Resize( npw, Nkq, 1 );
      SetValue( exxgkk_, 0.0 );

      Point3 gmesh, gmesh_car;
      Real* exxgkkPtr = exxgkk_.Data();

      for( Int k = 0; k < nkz; k++ ){
        for( Int j = 0; j < nky; j++ ){
          for( Int i = 0; i < nkx; i++ ){
            if( i < numKGrid[0] ) kx = i / double(numKGrid[0]);
            else kx = ( i - nkx ) / double(numKGrid[0]);

            if( j < numKGrid[1] ) ky = j / double(numKGrid[1]);
            else ky = ( j - nky ) / double(numKGrid[1]);

            if( k < numKGrid[2] ) kz = k / double(numKGrid[2]);
            else kz = ( k - nkz ) / double(numKGrid[2]);

            gmesh = Point3( kx, ky, kz );
            gmesh_car = Point3( 0.0, 0.0, 0.0 );
            for( Int ip = 0; ip < DIM; ip++ ){
              for( Int jp = 0; jp < DIM; jp++ ){
                gmesh_car[ip] += domain_.recipcell(jp,ip) * gmesh[jp];
              }
            }

            for( Int g = 0; g < npw; g++ ){

              if( spherecut ){
                Int ig = idxc[g];
                kG = ( gmesh_car + Point3( fft.ik[0][ig].imag(), fft.ik[1][ig].imag(),
                    fft.ik[2][ig].imag() ) );
              }
              else{
                kG = ( gmesh_car + Point3( fft.ik[0][g].imag(), fft.ik[1][g].imag(),
                    fft.ik[2][g].imag() ) );
              }

              gkk2 = kG[0] * kG[0] + kG[1] * kG[1] + kG[2] * kG[2];

              if( gkk2 > epsDiv ){
                if( screenMu_ > 0 ){
                  *(exxgkkPtr++) = 4.0 * PI / gkk2 * (1.0 -
                      std::exp( -gkk2 / (4.0*screenMu_*screenMu_) ));
                }
                else{
                  *(exxgkkPtr++) = 4.0 * PI / gkk2;
                }
              }
              else{
                *(exxgkkPtr) = -exxDiv_;
                if( screenMu_ > 0 ){
                  *(exxgkkPtr++) += PI / (screenMu_*screenMu_);
                }
                else{
                  *(exxgkkPtr++) += 0.0;
                }
              }
            } // for (g)
          } // for (i)
        } // for (j)
      } // for (k)
    }
  } // for iftruncated
  else if(iftruncated == 1){  
    // For sphere truncation
    Real Rc = dm.length[0];

    // implementation in QE
    // Rc = std::cbrt( 3.0 * domain_.Volume() / 4.0 / PI);
    for( Int d = 1; d < DIM; d++ ){
      Real RTemp = dm.length[d];
      if( RTemp < Rc ) Rc = RTemp;
    }
    Rc *= 0.5;
    Rc = Rc - Rc / 50.0;

    statusOFS << "Cutoff radius for coulomb potential is " << Rc << std::endl;

    exxDiv_ = 2 * PI * Rc * Rc;

    if( !useisdf ){
      exxgkk_.Resize( npw, nkTemp, nkTotal );
      SetValue( exxgkk_, 0.0 );

      for( Int k = 0; k < nkTemp; k++ ){
        Int ik = (useisdf == true) ? k : KpointIdx(k);
        kpoint = Point3( dm.klist[0][ik], dm.klist[1][ik], dm.klist[2][ik] );
        for( Int q = 0; q < nkTotal; q++ ){ 
          if( !energyband ){
            qpoint = Point3( dm.klist[0][q], dm.klist[1][q], dm.klist[2][q] );
          }
          else{
            qpoint = Point3( dm.klist_scf[0][q], dm.klist_scf[1][q], dm.klist_scf[2][q] );
          }
          for( Int i = 0; i < npw; i++ ){

            if( spherecut ){
              Int index = idxc[i];
              kG = ( kpoint - qpoint + Point3( fft.ik[0][index].imag(), fft.ik[1][index].imag(),
                  fft.ik[2][index].imag() ) );
            }
            else{
              kG = ( kpoint - qpoint + Point3( fft.ik[0][i].imag(), fft.ik[1][i].imag(),
                  fft.ik[2][i].imag() ) );
            }

            gkk2 = kG[0] * kG[0] + kG[1] * kG[1] + kG[2] * kG[2];

            if( gkk2 > epsDiv ){
              exxgkk_(i,k,q) = 4 * PI /gkk2 * ( 1.0 - std::cos( std::sqrt(gkk2) * Rc)) ;
            }
            else{
              exxgkk_(i,k,q) = -exxDiv_;
              exxgkk_(i,k,q) += 2 * PI  * Rc * Rc;
            }
          } // for i
        } // for q
      } // for k     
    }
    else{
      Index3 &numKGrid = domain_.numKGrid;
      Int nkx = 2 * numKGrid[0] - 1;
      Int nky = 2 * numKGrid[1] - 1;
      Int nkz = 2 * numKGrid[2] - 1;
      Int Nkq = nkx * nky * nkz;

      exxgkk_.Resize( npw, Nkq, 1 );
      SetValue( exxgkk_, 0.0 );

      Point3 gmesh, gmesh_car;
      Real* exxgkkPtr = exxgkk_.Data();

      for( Int k = 0; k < nkz; k++ ){
        for( Int j = 0; j < nky; j++ ){
          for( Int i = 0; i < nkx; i++ ){
            Real kx = ( i + 1 - numKGrid[0] ) / double(numKGrid[0]);
            Real ky = ( j + 1 - numKGrid[1] ) / double(numKGrid[1]);
            Real kz = ( k + 1 - numKGrid[2] ) / double(numKGrid[2]);

            gmesh = Point3( kx, ky, kz );
            gmesh_car = Point3( 0.0, 0.0, 0.0 );
            for( Int ip = 0; ip < DIM; ip++ ){
              for( Int jp = 0; jp < DIM; jp++ ){
                gmesh_car[ip] += domain_.recipcell(jp,ip) * gmesh[jp];
              }
            }

            for( Int g = 0; g < npw; g++ ){

              if( spherecut ){
                Int ig = idxc[g];
                kG = ( gmesh_car + Point3( fft.ik[0][ig].imag(), fft.ik[1][ig].imag(),
                    fft.ik[2][ig].imag() ) );
              }
              else{
                kG = ( gmesh_car + Point3( fft.ik[0][g].imag(), fft.ik[1][g].imag(),
                    fft.ik[2][g].imag() ) );
              }

              gkk2 = kG[0] * kG[0] + kG[1] * kG[1] + kG[2] * kG[2];
              
              if( gkk2 > epsDiv ){
                *(exxgkkPtr++) = 4 * PI /gkk2 * ( 1.0 - std::cos( std::sqrt(gkk2) * Rc)) ;
              }
              else{
                *(exxgkkPtr) = -exxDiv_;
                *(exxgkkPtr++) += 2 * PI  * Rc * Rc;
              }
            } // for (g)
          } // for (i)
        } // for (j)
      } // for (k)
    }
  } // for iftruncated
  statusOFS << "Hybrid mixing parameter  = " << exxFraction_ << std::endl; 
  statusOFS << "Hybrid screening length = " << screenMu_ << std::endl;

  return ;
}        // -----  end of function KohnSham::InitializeEXX ( Complex version )  ----- 

void
KohnSham::SetPhiEXX    (std::vector<Spinor>& psi, Fourier& fft)
{
  // Collect Psi into a globally shared array in the MPI context.
  Domain &dm = fft.domain;

  bool realspace = esdfParam.isUseRealSpace;
  bool useisdf   = esdfParam.isHybridDF;

  Int nkLocal = psi.size();
  Int nspin = dm.numSpinComponent;
  Int ncom = psi[0].Wavefun().n();
  Int ntot = dm.NumGridTotal();

  Real vol = fft.domain.Volume();
  Real fac;
  if( realspace )
    fac = std::sqrt( double(ntot) / vol );
  else
    fac = std::sqrt( 1.0 / vol / double(ntot) );

  Real timeSta, timeEnd;

  GetTime( timeSta );
  // Transform psi_k to real space and store them in wavefunR_
  for( Int k = 0; k < nkLocal; k++ ){
    psi[k].Recip2Real( fft );
  }
  GetTime( timeEnd );
  
  statusOFS << "Time for transforming psi to real space = " << timeEnd - timeSta << std::endl;

  // Collect non-zero occupation rates
  if( nspin == 1 || nspin == 4 ){
    occLocal_.resize( nkLocal );

    Real eps = 1e-16;
    Int nbPhimax = 0, nbPhimaxLocal = 0;

    for( Int k = 0; k < nkLocal; k++ ){

      Int nbPsi = psi[k].WavefunIdx().m();    
      Int nbPhi = 0;
      Int* wfcIdx = psi[k].WavefunIdx().Data();

      for( Int i = 0; i < nbPsi; i++ ){
        Int ib = *(wfcIdx++);
        if( occupationRate_[k][ib] > eps ){
          nbPhi = nbPhi + 1;
        }
      }

      if( !useisdf ){
        occLocal_[k].Resize( nbPhi );    
        wfcIdx = psi[k].WavefunIdx().Data();

        for( Int i = 0; i < nbPhi; i++ ){
          Int ib = *(wfcIdx++);
          occLocal_[k][i] = occupationRate_[k][ib];
        }
      }
      else{
        nbPhimaxLocal = std::max( nbPhi, nbPhimaxLocal );
      }
    } // for (k) 

    if( useisdf ){
      // Make the shape of Phi_k regular to call ScaLAPACK conveniently
      MPI_Allreduce( &nbPhimaxLocal, &nbPhimax, 1, MPI_INT, MPI_MAX, dm.comm ); 
      statusOFS << "The band number of Phi is set to " << nbPhimax << std::endl;
      for( Int k = 0; k < nkLocal; k++ ){
        Int* wfcIdx = psi[k].WavefunIdx().Data();
        occLocal_[k].Resize( nbPhimax );
        for( Int i = 0; i < nbPhimax; i++ ){
          Int ib = *(wfcIdx++);
          occLocal_[k][i] = occupationRate_[k][ib];
        }
      } // for (k)         
    }
  }
  else{
    UpoccLocal_.resize( nkLocal );
    DnoccLocal_.resize( nkLocal );

    Real eps = 1e-16;
    for( Int k = 0; k < nkLocal; k++ ){

      Int nbPsi = psi[k].WavefunIdx().m();
      Int nbPhi_up = 0;
      Int nbPhi_dn = 0;
      Int* wfcIdx = psi[k].WavefunIdx().Data();

      for( Int i = 0; i < nbPsi; i++ ){
        Int ib = *(wfcIdx++);
        if( occupationRate_[k][ib] > eps ){
          nbPhi_up = nbPhi_up + 1;
        }
        if( occupationRate_[k][ib+nbPsi] > eps ){
          nbPhi_dn = nbPhi_dn + 1;
        }
      }

      UpoccLocal_[k].Resize( nbPhi_up );
      DnoccLocal_[k].Resize( nbPhi_dn );
      wfcIdx = psi[k].WavefunIdx().Data();

      for( Int i = 0; i < nbPsi; i++ ){
        Int ib = *(wfcIdx++);
        if( occupationRate_[k][ib] > eps ){
          UpoccLocal_[k][i] = occupationRate_[k][ib];
        }
        if( occupationRate_[k][ib+nbPsi] > eps ){
          DnoccLocal_[k][i] = occupationRate_[k][ib+nbPsi];
        }
      }
    } // for (k) 
  }

  if( nspin == 1 || nspin == 4 ){
    phiEXX_.resize( nkLocal );

    for( Int k = 0; k < nkLocal; k++ ){

      const NumTns<Complex>& wavefun = psi[k].WavefunR();
      
      Int nbPhi = occLocal_[k].m();

      phiEXX_[k].Resize( ntot, ncom, nbPhi );
      SetValue( phiEXX_[k], Z_ZERO );

      for( Int j = 0; j < nbPhi; j++ ){
        for( Int i = 0; i < ncom; i++ ){
          blas::Copy( ntot, wavefun.VecData(i,j), 1, phiEXX_[k].VecData(i,j), 1 );
          blas::Scal( ntot, fac, phiEXX_[k].VecData(i,j), 1 );
        } // for (i)
      } // for (j)   
    } // for (k)
  }
  else{
    UpphiEXX_.resize( nkLocal ); 
    DnphiEXX_.resize( nkLocal );

    for( Int k = 0; k < nkLocal; k++ ){

      const NumTns<Complex>& wavefun = psi[k].WavefunR();

      Int nbPsi = wavefun.p() / 2;
      Int nbPhi_up = UpoccLocal_[k].m();
      Int nbPhi_dn = DnoccLocal_[k].m();

      UpphiEXX_[k].Resize( ntot, 1, nbPhi_up );
      DnphiEXX_[k].Resize( ntot, 1, nbPhi_dn );
      SetValue( UpphiEXX_[k], Z_ZERO );
      SetValue( DnphiEXX_[k], Z_ZERO );

      for( Int j = 0; j < nbPhi_up; j++){
        blas::Copy( ntot, wavefun.VecData(0,j+UP*nbPsi),
            1, UpphiEXX_[k].VecData(0,j), 1 );
        blas::Scal( ntot, fac, UpphiEXX_[k].VecData(0,j), 1 );
      }
      for( Int j = 0; j < nbPhi_dn; j++){
        blas::Copy( ntot, wavefun.VecData(0,j+DN*nbPsi),
            1, DnphiEXX_[k].VecData(0,j), 1 );
        blas::Scal( ntot, fac, DnphiEXX_[k].VecData(0,j), 1 );
      } // for (j)   
    } // for (k)
  }
    
  return ;
}    // -----  end of method KohnSham::SetPhiEXX ( Complex version )  -----

void
KohnSham::SetPhiEXX    (std::string WfnFileName, Fourier& fft)
{
  // Read wavefunction and occupationRate from files to build
  // Fock exchange operator
  Domain &dm = fft.domain;
  Int mpirank;
  MPI_Comm_rank( dm.comm, &mpirank ); 

  bool realspace = esdfParam.isUseRealSpace;

  Int nkLocal = dm.KpointIdx_scf.Size();
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

  occupationRate_.resize( nkLocal );
  if( nspin == 1 || nspin == 4 ){ 
    phiEXX_.resize( nkLocal );
    for( Int k = 0; k < nkLocal; k++ ){
      deserialize( phiEXX_[k], wfnStream, NO_MASK );
      deserialize( occupationRate_[k], wfnStream, NO_MASK );

      blas::Scal( phiEXX_[k].Size(), fac, phiEXX_[k].Data(), 1 );
    }
  }
  else{
    UpphiEXX_.resize( nkLocal );
    DnphiEXX_.resize( nkLocal );
    for( Int k = 0; k < nkLocal; k++ ){
      deserialize( UpphiEXX_[k], wfnStream, NO_MASK );
      deserialize( DnphiEXX_[k], wfnStream, NO_MASK );
      deserialize( occupationRate_[k], wfnStream, NO_MASK );

      blas::Scal( UpphiEXX_[k].Size(), fac, UpphiEXX_[k].Data(), 1 );
      blas::Scal( DnphiEXX_[k].Size(), fac, DnphiEXX_[k].Data(), 1 );
    }
  }  
  
  return ;
}    // -----  end of method KohnSham::SetPhiEXX ( Complex version )  -----

void
KohnSham::CalculateVexxACE ( std::vector<Spinor>& psik, Fourier& fft )
{
  // This assumes SetPhiEXX has been called so that phiEXX and psi
  // contain the same information. 

  // Since this is a projector, it should be done on the COARSE grid,
  // i.e. to the wavefunction directly

  MPI_Barrier(domain_.comm);

  int colrank;  MPI_Comm_rank(domain_.colComm_kpoint, &colrank);
  int colsize;  MPI_Comm_size(domain_.colComm_kpoint, &colsize);

  bool spherecut    = esdfParam.isUseSphereCut;

  Int nkLocal       = psik.size();
  Int nspin         = domain_.numSpinComponent;
  Int numStateLocal = psik[0].NumState();
  Int numStateTotal = psik[0].NumStateTotal();
  Int ncom          = ( nspin == 4 ) ? 2 : 1;
  Int numACE        = ( nspin == 2 ) ? 2 : 1;
  Int ntot;
  Int ntot2;

  if( nspin == 1 || nspin == 4 ){
    vexxProj_.resize( nkLocal );
  }
  else{
    UpvexxProj_.resize( nkLocal ); DnvexxProj_.resize( nkLocal );
    numStateTotal /= 2;
    numStateLocal /= 2;
  }
 
  Real timeSta, timeEnd;
  
  GetTime( timeSta );

  // The ACE operator are constructed separately for spin-up and spin-down spinors
  // in the case of spin-unrestricted and spin-restricted calculations
  for( Int ispin = 0; ispin < numACE; ispin++ ){    

    // Allocate space for saving VexxPsi
    std::vector<CpxNumTns> vexxPsi( nkLocal );
    for( Int k = 0; k < nkLocal; k++ ){ 
      if( spherecut )
        ntot = domain_.numGridSphere[k];
      else
        ntot = domain_.NumGridTotal();

      vexxPsi[k].Resize( ntot, ncom, numStateLocal );
      SetValue( vexxPsi[k], Z_ZERO );
    }

    CpxNumTns emptyWavefun;
    Spinor psi( fft.domain, ncom, 0, false, emptyWavefun.Data() );

    psi.AddMultSpinorEXX( fft, psik, phiEXX_, exxgkk_,
        exxFraction_, ispin, nspin, occLocal_, vexxPsi );

    for( Int k = 0; k < nkLocal; k++ ){ 

      if( spherecut )
        ntot = domain_.numGridSphere[k];
      else
        ntot = domain_.NumGridTotal();

      ntot2 = ntot * ncom; 

      // Implementation based on SVD
      CpxNumMat  M(numStateTotal, numStateTotal);
      // Convert the column partition to row partition
      Int numStateBlocksize = numStateTotal / colsize;
      Int ntotBlocksize = ntot2 / colsize;

      numStateLocal = numStateBlocksize;
      Int ntotLocal = ntotBlocksize;

      if(colrank < (numStateTotal % colsize)){
        numStateLocal = numStateBlocksize + 1;
      }

      if(colrank < (ntot2 % colsize)){
        ntotLocal = ntotBlocksize + 1;
      }

      CpxNumMat localPsiCol( ntot2, numStateLocal );
      SetValue( localPsiCol, Z_ZERO );

      CpxNumMat localVexxPsiCol( ntot2, numStateLocal );
      SetValue( localVexxPsiCol, Z_ZERO );

      CpxNumMat localPsiRow( ntotLocal, numStateTotal );
      SetValue( localPsiRow, Z_ZERO );

      CpxNumMat localVexxPsiRow( ntotLocal, numStateTotal );
      SetValue( localVexxPsiRow, Z_ZERO );

      // Initialize     
      lapack::Lacpy( 'A', ntot2, numStateLocal, psik[k].Wavefun().Data(), ntot2, localPsiCol.Data(), ntot2 );
      lapack::Lacpy( 'A', ntot2, numStateLocal, vexxPsi[k].Data(), ntot2, localVexxPsiCol.Data(), ntot2 );

      AlltoallForward (localPsiCol, localPsiRow, domain_.colComm_kpoint);
      AlltoallForward (localVexxPsiCol, localVexxPsiRow, domain_.colComm_kpoint);

      CpxNumMat MTemp( numStateTotal, numStateTotal );
      SetValue( MTemp, Z_ZERO );

      blas::Gemm( 'C', 'N', numStateTotal, numStateTotal, ntotLocal,
          -1.0, localPsiRow.Data(), ntotLocal,
          localVexxPsiRow.Data(), ntotLocal, 0.0,
          MTemp.Data(), numStateTotal );

      SetValue( M, Z_ZERO );
      MPI_Allreduce( MTemp.Data(), M.Data(), numStateTotal * numStateTotal*2, 
          MPI_DOUBLE, MPI_SUM, domain_.colComm_kpoint );

      if ( colrank == 0) {
        lapack::Potrf('L', numStateTotal, M.Data(), numStateTotal);
      }

      MPI_Bcast(M.Data(), 2*numStateTotal * numStateTotal, MPI_DOUBLE, 0, domain_.colComm_kpoint);

      blas::Trsm( 'R', 'L', 'C', 'N', ntotLocal, numStateTotal, 1.0,
          M.Data(), numStateTotal, localVexxPsiRow.Data(), ntotLocal );   

      if( nspin == 1 || nspin == 4 ){
        vexxProj_[k].Resize( ntot2, numStateLocal );
        AlltoallBackward (localVexxPsiRow, vexxProj_[k], domain_.colComm_kpoint);
      }
      else{
        if( ispin == 0 ){
          UpvexxProj_[k].Resize( ntot2, numStateLocal );
          AlltoallBackward (localVexxPsiRow, UpvexxProj_[k], domain_.colComm_kpoint);  
        }
        else{
          DnvexxProj_[k].Resize( ntot2, numStateLocal );
          AlltoallBackward (localVexxPsiRow, DnvexxProj_[k], domain_.colComm_kpoint);
        }
      }       
    } // for (k)
  } // for (ispin)

  GetTime( timeEnd );
  statusOFS << "Total time for calculation of ACE operator is " << timeEnd - timeSta << std::endl;
 
  return ;
}         // -----  end of method KohnSham::CalculateVexxACE ( Complex version )  -----

void
KohnSham::CalculateVexxACEDF ( std::vector<Spinor>& psik, Fourier& fft, bool isFixColumnDF )
{
  MPI_Barrier(domain_.comm);

  int colrank;  MPI_Comm_rank(domain_.colComm_kpoint, &colrank);
  int colsize;  MPI_Comm_size(domain_.colComm_kpoint, &colsize);

  bool spherecut = esdfParam.isUseSphereCut;

  Int mb = esdfParam.BlockSizeGrid;

  Int nkLocal       = psik.size();
  Int nspin         = domain_.numSpinComponent;
  Int numStateLocal = psik[0].NumState();
  Int numStateTotal = psik[0].NumStateTotal();
  Int ncom          = ( nspin == 4 ) ? 2 : 1;
  Int numACE        = ( nspin == 2 ) ? 2 : 1;
  Int ntot;
  Int ntot2;

  if( nspin == 1 || nspin == 4 ){
    vexxProj_.resize( nkLocal );
  }
  else{
    UpvexxProj_.resize( nkLocal ); DnvexxProj_.resize( nkLocal );
    numStateTotal /= 2;
    numStateLocal /= 2;
  }

  Real timeSta, timeEnd;
  
  GetTime( timeSta );

  // The ACE operator are constructed separately for spin-up and spin-down spinors
  // in the case of spin-unrestricted and spin-restricted calculations
  for( Int ispin = 0; ispin < numACE; ispin++ ){    

    // Allocate space for saving VexxPsi and M
    std::vector<CpxNumTns> vexxPsi( nkLocal );
    std::vector<CpxNumMat> M( nkLocal );
    for( Int k = 0; k < nkLocal; k++ ){ 
      if( spherecut )
        ntot = domain_.numGridSphere[k];
      else
        ntot = domain_.NumGridTotal();

      vexxPsi[k].Resize( ntot, ncom, numStateLocal );
      SetValue( vexxPsi[k], Z_ZERO );

      M[k].Resize( numStateTotal, numStateTotal );
      SetValue( M[k], Z_ZERO );
    }

    CpxNumTns emptyWavefun;
    Spinor psi( fft.domain, ncom, 0, false, emptyWavefun.Data() );  

    // Read the old index of interpolation points
    if( isFixColumnDF == true ){ 
      psi.PivQR() = psik[0].PivQR(); 
    }

    if( !esdfParam.isHybridFourierConv ){
      psi.AddMultSpinorEXXDF( fft, psik, phiEXX_, exxgkk_, 
          exxFraction_, ispin, nspin,
          occLocal_, hybridDFType_, hybridDFKmeansWFType_,
          hybridDFKmeansWFAlpha_, hybridDFKmeansTolerance_,
          hybridDFKmeansMaxIter_, hybridDFNumMu_, hybridDFNumGaussianRandom_,
          hybridDFNumProcScaLAPACK_, hybridDFTolerance_, BlockSizeScaLAPACK_,
          isFixColumnDF, M, vexxPsi );
    }
    else{
      psi.AddMultSpinorEXXDFConv( fft, psik, phiEXX_, exxgkk_,
          exxFraction_, ispin, nspin,
          occLocal_, hybridDFType_, hybridDFKmeansWFType_,
          hybridDFKmeansWFAlpha_, hybridDFKmeansTolerance_,
          hybridDFKmeansMaxIter_, hybridDFNumMu_, hybridDFNumGaussianRandom_,
          hybridDFNumProcScaLAPACK_, hybridDFTolerance_, BlockSizeScaLAPACK_,
          isFixColumnDF, M, vexxPsi );
    }

    // Store the new index of interpolation points 
    if( isFixColumnDF == false ){
      for( Int k = 0; k < nkLocal; k++ ){
        psik[k].PivQR() = psi.PivQR();
      }
    }
    // Implementation based on Cholesky
    for( Int k = 0; k < nkLocal; k++ ){  

      ntot = domain_.numGridSphere[k];
      ntot2 = ntot * ncom;

      Int ntotLocal;
      IntNumVec idxGridTemp;
      CalculateIndexSpinor( ntot2, mb, ntotLocal, idxGridTemp, domain_.colComm_kpoint );

      CpxNumMat localVexxPsiCol( ntot2, numStateLocal );
      SetValue( localVexxPsiCol, Z_ZERO );

      CpxNumMat localVexxPsiRow( ntotLocal, numStateTotal );
      SetValue( localVexxPsiRow, Z_ZERO );
  
      // Initialize
      lapack::Lacpy( 'A', ntot2, numStateLocal, vexxPsi[k].Data(), ntot2, 
          localVexxPsiCol.Data(), ntot2 );

      AlltoallForward( mb, ncom, localVexxPsiCol, localVexxPsiRow, domain_.colComm_kpoint );

      if ( colrank == 0) {
        lapack::Potrf('L', numStateTotal, M[k].Data(), numStateTotal);
      }

      MPI_Bcast(M[k].Data(), 2*numStateTotal*numStateTotal, MPI_DOUBLE, 0, domain_.colComm_kpoint);

      blas::Trsm( 'R', 'L', 'C', 'N', ntotLocal, numStateTotal, 1.0,
          M[k].Data(), numStateTotal, localVexxPsiRow.Data(), ntotLocal );

      vexxProj_[k].Resize( ntot2, numStateLocal );

      AlltoallBackward( mb, ncom, localVexxPsiRow, vexxProj_[k], domain_.colComm_kpoint );
    } // for (k)
  } // for (ispin)
  
  GetTime( timeEnd );
  statusOFS << "Total time for calculation of ACE operator is " << timeEnd - timeSta << std::endl;

  return;
}         // -----  end of method KohnSham::CalculateVexxACEDF ( Complex version )  -----

// This comes from exxenergy2() function in exx.f90 in QE.
Real
KohnSham::CalculateEXXEnergy    ( std::vector<Spinor>& psik, Fourier& fft )
{
  MPI_Barrier(domain_.comm);
  int colrank;  MPI_Comm_rank(domain_.colComm_kpoint, &colrank);
  int colsize;  MPI_Comm_size(domain_.colComm_kpoint, &colsize);
  
  Real fockEnergy = 0.0;
  Real fockEnergyLocal = 0.0;

  if( !fft.isInitialized ){
    ErrorHandling("Fourier is not prepared.");
  }

  bool spherecut = esdfParam.isUseSphereCut;
  bool useace    = esdfParam.isHybridACE;

  Real vol      = domain_.Volume();
  Int nkLocal   = psik.size();
  Int nspin     = domain_.numSpinComponent;
  Int ncom      = ( nspin == 4 ) ? 2 : 1;
  Int numACE    = ( nspin == 2 ) ? 2 : 1;
  Int ntot;
  Int ntot2;

  std::vector<DblNumVec> occSpin;
  for( Int ispin = 0; ispin < numACE; ispin++ ){
    if( nspin == 2 && !esdfParam.isHybridACE  ){
      occSpin.resize( nkLocal );
      for( Int k = 0; k < nkLocal; k++ ){
        Int numStateTotal = occupationRate_[k].Size() / 2; 
        occSpin[k].Resize( numStateTotal ); 
        blas::Copy( numStateTotal, &(occupationRate_[k][ispin*numStateTotal]), 1, occSpin[k].Data(), 1 );
      }
    }
    for( Int k = 0; k < nkLocal; k++ ){

      if( spherecut )
        ntot = domain_.numGridSphere[k];
      else
        ntot = domain_.NumGridTotal();

      ntot2 = ntot * ncom;
      // The band number for each spin
      Spinor &psiTemp = psik[k];
      Int numStateTotal = psiTemp.NumStateTotal();
      Int numStateLocal = psiTemp.NumState();

      if( nspin == 2 )
      {
        numStateTotal /= 2;
        numStateLocal /= 2;
      } 

      Spinor psi( fft.domain, ncom, numStateTotal, false, 
          psiTemp.Wavefun().VecData(0,ispin*numStateLocal), psiTemp.Kpoint(), 
          psiTemp.Ik(), psiTemp.IkLocal() );  
      NumTns<Complex>& wavefun = psi.Wavefun();

      CpxNumMat vexxPsiCol;
      // Directly use the phiEXX_ and vexxProj_ to calculate the exchange energy
      if( useace ){
        // temporarily just implement here
        // Directly use projector   
        Int numStateBlocksize = numStateTotal / colsize;
        Int ntotBlocksize = ntot2 / colsize;

        Int numStateLocal = numStateBlocksize;
        Int ntotLocal = ntotBlocksize;

        if(colrank < (numStateTotal % colsize)){
          numStateLocal = numStateBlocksize + 1;
        }

        if(colrank < (ntot2 % colsize)){
          ntotLocal = ntotBlocksize + 1;
        }

        CpxNumMat psiCol( ntot2, numStateLocal );
        SetValue( psiCol, Z_ZERO );

        CpxNumMat psiRow( ntotLocal, numStateTotal );
        SetValue( psiRow, Z_ZERO );

        CpxNumMat vexxProjCol( ntot2, numStateLocal );
        SetValue( vexxProjCol, Z_ZERO );

        CpxNumMat vexxProjRow( ntotLocal, numStateTotal );
        SetValue( vexxProjRow, Z_ZERO );

        vexxPsiCol.Resize( ntot2, numStateLocal );
        SetValue( vexxPsiCol, Z_ZERO );

        CpxNumMat vexxPsiRow( ntotLocal, numStateTotal );
        SetValue( vexxPsiRow, Z_ZERO );
        lapack::Lacpy( 'A', ntot2, numStateLocal, psi.Wavefun().Data(), ntot2, psiCol.Data(), ntot2 );
        if( numACE == 1 )    
          lapack::Lacpy( 'A', ntot2, numStateLocal, vexxProj_[k].Data(), ntot2, vexxProjCol.Data(), ntot2 );
        else{
          if( ispin == 0)
            lapack::Lacpy( 'A', ntot2, numStateLocal, UpvexxProj_[k].Data(), ntot2, vexxProjCol.Data(), ntot2 );
          else
            lapack::Lacpy( 'A', ntot2, numStateLocal, DnvexxProj_[k].Data(), ntot2, vexxProjCol.Data(), ntot2 ); 
        }
          
        AlltoallForward (psiCol, psiRow, domain_.colComm_kpoint);
        AlltoallForward (vexxProjCol, vexxProjRow, domain_.colComm_kpoint);

        CpxNumMat MTemp( numStateTotal, numStateTotal );
        SetValue( MTemp, Z_ZERO );

        blas::Gemm( 'C', 'N', numStateTotal, numStateTotal, ntotLocal,
            1.0, vexxProjRow.Data(), ntotLocal,
            psiRow.Data(), ntotLocal, 0.0,
            MTemp.Data(), numStateTotal );

        CpxNumMat M(numStateTotal, numStateTotal);
        SetValue( M, Z_ZERO );

        MPI_Allreduce( MTemp.Data(), M.Data(), 2*numStateTotal * numStateTotal, MPI_DOUBLE, MPI_SUM, domain_.colComm_kpoint );

        blas::Gemm( 'N', 'N', ntotLocal, numStateTotal, numStateTotal, -1.0,
            vexxProjRow.Data(), ntotLocal, M.Data(), numStateTotal,
            0.0, vexxPsiRow.Data(), ntotLocal );

        AlltoallBackward (vexxPsiRow, vexxPsiCol, domain_.colComm_kpoint);
      }  // ---- if( esdfParam.isHybridACE ) ----
      else{
#if 0
        NumTns<Complex>  vexxPsi( ntot, ncom, numStateLocal );
        SetValue( vexxPsi, Z_ZERO );

        if( nspin == 1 || nspin == 4 ){
          psi.AddMultSpinorEXX( fft, phiEXX_, exxgkk_,
              exxFraction_, nspin, occupationRate_, vexxPsi );
        }
        else{
          if( ispin == 0 )
            psi.AddMultSpinorEXX( fft, UpphiEXX_, exxgkk_,
              exxFraction_, nspin, occSpin, vexxPsi ); 
          else
            psi.AddMultSpinorEXX( fft, DnphiEXX_, exxgkk_,
              exxFraction_, nspin, occSpin, vexxPsi ); 
        }

        vexxPsiCol = CpxNumMat( ntot2, numStateLocal, true, vexxPsi.Data() );
#endif
      }

      for( Int j = 0; j < numStateLocal; j++ ){
        for( Int i = 0; i < ncom; i++ ){
          for( Int ir = 0; ir < ntot; ir++ ){
            fockEnergyLocal += (vexxPsiCol(ir+i*ntot,j) * std::conj(wavefun(ir,i,j))).real()
                * occupationRate_[k][psi.WavefunIdx(j)+ispin*numStateTotal];
          }
        }
      }
    }  // for (k)
  }

  MPI_Barrier( domain_.comm );
  mpi::Allreduce( &fockEnergyLocal, &fockEnergy, 1, MPI_SUM, domain_.comm );

  return( fockEnergy * numSpin_ / 2.0 );
}         // -----  end of method KohnSham::CalculateEXXEnergy ( Complex version ) ----
#endif

} // namespace pwdft
