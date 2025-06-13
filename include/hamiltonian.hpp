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
/// @file hamiltonian.hpp
/// @brief Hamiltonian class for planewave basis diagonalization method.
/// @date 2012-09-16
#ifndef _HAMILTONIAN_HPP_
#define _HAMILTONIAN_HPP_

#include  "environment.hpp"
#include  "numvec_impl.hpp"
#include  "domain.hpp"
#include  "periodtable.hpp"
#include  "spinor.hpp"
#include  "utility.hpp"
#include  "esdf.hpp"
#include  <xc.h>

#ifdef GPU
#include "cublas.hpp"
#include "cusolver.hpp"
#endif

namespace pwdft{

// *********************************************************************
// Base Hamiltonian class 
// *********************************************************************

/// @brief Pure virtual class for handling different types of
/// Hamiltonian.
class Hamiltonian {
protected:
  Domain                      domain_;
  // List of atoms
  std::vector<Atom>           atomList_;
  Int                         numSpin_; 
  Int                         numExtraState_;
  Int                         numOccupiedState_;
  Int                         numDensityComponent_;
  Int                         numSpinorComponent_;
  // Used to indicate spin-up or spin-down Hamiltonian
  // in spin-unrestrited calculations
  Int                         spinswitch_;
  // Type of pseudopotential, default HGH
  std::string                 pseudoType_;
  // Id of the exchange-correlation potential
  std::string                 XCType_;
  Int                         XCId_;
  Int                         XId_;
  Int                         CId_;
  // Exchange-correlation potential using libxc package.
  xc_func_type                XCFuncType_; 
  xc_func_type                XFuncType_; 
  xc_func_type                CFuncType_; 
  bool                        XCInitialized_;
  // MPI communication 
  MPI_Comm rowComm_, colComm_;
  // Pseudocharge to represent the local pseudopotential or the Gaussian
  // compensation pseudocharge
  DblNumVec                   pseudoCharge_;
  // Short range part of the the local pseudopotential
  DblNumVec                   vLocalSR_;
  // density_(:,1)    electron density
  // density_(:,2-4)  magnetization along x,y,z directions
  DblNumMat                   density_;  
  // spin-up and spin-down density
  DblNumMat                   spindensity_; 
  // Old density for density mixing
  DblNumMat                   densityold_;
  // Fixed spin axis for spin-noncollinear calculations
  bool                        isParallel_;
  Point3                      spinaxis_;
  DblNumVec                   segni_;
  // Gradient of the density
  std::vector<DblNumMat>      gradDensity_;
  // atomic charge densities
  DblNumMat                   atomDensity_;
  // External potential. TODO not implemented
  DblNumVec                   vext_;            
  // Hartree potential
  DblNumVec                   vhart_;
  // vxc_(:,1)        exchange-correlation potential
  // vxc_(:,2-4)      vector exchange-correlation potential along x,y,z
  // directions
  // The same dimension as the dimension of density_
  DblNumMat                   vxc_;             
  // Total potential
  DblNumMat                   vtot_;        
  // the exchange-correlation energy density
  DblNumVec                   epsxc_; 
  // the exchange-correlation energy in each grid
  // exc_ = epsxc_ * density_
  DblNumVec                   exc_;
#ifdef _COMPLEX_
  // Kinetic energy
  std::vector<DblNumVec>      ekin_;
  // Theter precondition
  std::vector<DblNumVec>      teter_;
  // Coulomb term 
  DblNumTns                   exxgkk_;
#else
  DblNumVec                   ekin_;
  DblNumVec                   teter_;
  DblNumVec                   exxgkkR2C_;
#endif
  // Pseudopotential for each atom
  // If isUsePseudoCharge == true, then pseudo_ contains the information
  // from the pseudocharge. Otherwise it contains the information of the
  // local part of pseudocharge.
  std::vector<PseudoPot>      pseudo_;
  std::vector<NonlocalGPP>    vnlc_;
  std::vector<CpxNumTns>      coef_;
  // Eigenvalues
#ifdef _COMPLEX_
  std::vector<DblNumVec>      eigVal_;
  std::vector<DblNumVec>      occupationRate_;
  std::vector<DblNumVec>      occLocal_; // non-zero occupation rates in local process
  std::vector<DblNumVec>      UpoccLocal_;
  std::vector<DblNumVec>      DnoccLocal_;
#else
  DblNumVec                   eigVal_;
  DblNumVec                   occupationRate_;
#endif
  // EXX variables default as non-hybrid functional
  bool                        isHybrid_ = false;
  bool                        isEXXActive_ = false;
  Int                         exxDivergenceType_;
  Real                        exxDiv_;
  /// @brief Screening parameter mu for range separated hybrid functional. Currently hard coded
  const Real                  screenMu_ = 0.106;
  /// @brief Mixing parameter for hybrid functional calculation. Currently hard coded
  const Real                  exxFraction_ = 0.25;
  // ISDF parameters
  std::string                 hybridDFType_;
  std::string                 hybridDFKmeansWFType_;
  Real                        hybridDFKmeansWFAlpha_;
  Real                        hybridDFKmeansTolerance_;
  Int                         hybridDFKmeansMaxIter_;
  Real                        hybridDFNumMu_;
  Real                        hybridDFNumGaussianRandom_;
  Int                         hybridDFNumProcScaLAPACK_;
  Int                         BlockSizeScaLAPACK_;
  Real                        hybridDFTolerance_;
  // Energy
  Real                EVdw_;                     // Van der Waals energy
  Real                Eself_;                    // Self energy due to the pseudopotential
  Real                EIonSR_;                   // Short range repulsion energy for Gaussian charge
  Real                Eext_;                     // Energy due to external potential
  // Van der Waals force
  DblNumMat           forceVdw_;                 
  // Ion short range repulsion energy due to the use of // Gaussian
  // compensation charge formulation 
  DblNumMat           forceIonSR_;
  // Ion long range repulsion force
  DblNumMat           forceIonLR_;
  // Force due to external potential
  DblNumMat           forceext_;
  // Internal variables related to wavefunction filter in CheFSI for PWDFT 
  bool apply_filter_;
  bool apply_first_;                           
  Real wfn_cutoff_;

public:

  // *********************************************************************
  // Lifecycle
  // *********************************************************************
  Hamiltonian() {}
  virtual ~Hamiltonian() {}

  virtual void Setup (
      const Domain&              dm,
      const PeriodTable&         ptable,
      const std::vector<Atom>&   atomList ) = 0;

  // *********************************************************************
  // Operations
  // *********************************************************************

  /// @brief Calculate pseudopotential, as well as other atomic related
  /// energies and forces, such as self energy, short range repulsion
  /// energy and VdW energies.
  virtual void CalculatePseudoPotential( PeriodTable &ptable, Fourier &fft, 
      std::vector<IntNumVec>& grididx ) = 0;

  /// @brief Atomic density is implemented using the structure factor
  /// method due to the large cutoff radius
  virtual void CalculateAtomDensity( PeriodTable &ptable, Fourier &fft ) = 0;
#ifdef _COMPLEX_
  virtual void CalculateDensity( const std::vector<Spinor> &psi, const std::vector<DblNumVec> &occrate, 
               Real &val, Fourier &fft ) = 0;
#else
  virtual void CalculateDensity( const Spinor &psi, const DblNumVec &occrate, Real &val, Fourier &fft ) = 0;
#endif
  virtual void CalculateGradDensity( Fourier &fft ) = 0;

  virtual void CalculateXC (Real &val, Fourier& fft) = 0;

  virtual void CalculateHartree( Fourier& fft ) = 0;

  virtual void CalculateVtot( DblNumMat& vtot ) = 0;
  
  virtual void CalculateEkin( Fourier& fft ) = 0;
#ifdef _COMPLEX_
  /// @brief Calculate the Hellmann-Feynman force for each atom.
  /// This is a clean version for computing the force.
  ///
  /// In particular it is very important to calculate the nonlocal
  /// contribution of the force on the fine grid.
  virtual void CalculateForce ( PeriodTable &ptable, std::vector<Spinor>& psi, Fourier& fft ) = 0;
  virtual void MultSpinor(Spinor& psi, NumTns<Complex>& a3, Fourier& fft) = 0;
  // Hybrid functional
  virtual void CalculateVexxACE( std::vector<Spinor>& psi, Fourier& fft ) = 0;
  virtual void CalculateVexxACEDF( std::vector<Spinor>& psi, Fourier& fft, bool isFixColumnDF ) = 0;
  virtual std::vector<CpxNumTns>& PhiEXX() = 0;
  virtual void SetPhiEXX( std::vector<Spinor>& psi, Fourier& fft ) = 0;
  virtual void SetPhiEXX( std::string WfnFileName, Fourier& fft ) = 0;
  virtual Real CalculateEXXEnergy( std::vector<Spinor>& psi, Fourier& fft ) = 0;
#else
  virtual void CalculateForce ( Spinor& psi, Fourier& fft ) = 0;
  virtual void MultSpinor(Spinor& psi, NumTns<Real>& a3, Fourier& fft) = 0;
  virtual void MultSpinor(Spinor& psi, NumTns<Complex>& a3, Fourier& fft) = 0;

  virtual void CalculateVexxACE( Spinor& psi, Fourier& fft ) = 0;
  virtual void CalculateVexxACEDF( Spinor& psi, Fourier& fft, bool isFixColumnDF ) = 0;
  virtual DblNumTns& PhiEXX() = 0;
  virtual void SetPhiEXX( Spinor& psi, Fourier& fft ) = 0;
  virtual void SetPhiEXX( std::string WfnFileName, Fourier& fft ) = 0;
  virtual Real CalculateEXXEnergy( Spinor& psi, Fourier& fft ) = 0;
#endif
  virtual void InitializeEXX( Real ecutWavefunction, Fourier& fft ) = 0;

  virtual void  Setup_XC( std::string xc) = 0;

  void UpdateHamiltonian ( std::vector<Atom>&  atomList ) { atomList_ = atomList; }

#ifdef GPU
  // GPU-ified functions
  virtual void CalculateDensityGPU( const Spinor &psi, const DblNumVec &occrate, Real &val, Fourier& fft ) = 0;
  virtual void MultSpinorGPU(Spinor& psi, cuNumTns<Real>& a3, Fourier& fft) = 0;
  virtual void ACEOperatorGPU( cuDblNumMat& cu_psi, Fourier& fft, cuDblNumMat& cu_Hpsi, const int spinswitch = 0 ) = 0;
  virtual void CalculateVexxACEGPU( Spinor& psi, Fourier& fft ) = 0;
#endif

  // *********************************************************************
  // Access
  // *********************************************************************
  DblNumMat&  Vtot() { return vtot_; }
  DblNumVec&  Vext() { return vext_; }
  DblNumMat&  Vxc()  { return vxc_; }  
  DblNumVec&  Vhart() { return vhart_; }
  DblNumVec&  VLocalSR() { return vLocalSR_; }

#ifdef _COMPLEX_
  DblNumVec&  Ekin( Int k ) { return ekin_[k]; }
  DblNumVec&  Teter( Int k ) { return teter_[k]; };
  DblNumTns&  Exxgkk() { return exxgkk_; }
#else
  DblNumVec&  Ekin() { return ekin_; }
  DblNumVec&  Teter() { return teter_; };
  DblNumVec&  ExxgkkR2C() { return exxgkkR2C_; }
#endif
  DblNumMat&  Density() { return density_; }
  DblNumMat&  SpinDensity() { return spindensity_; }
  DblNumMat&  DensityOld() { return densityold_; }
  DblNumVec&  Segni() { return segni_; }
  Point3&     SpinAxis() { return spinaxis_; };

  std::vector<DblNumMat>  GradDensity() { return gradDensity_; }
  DblNumMat&  AtomDensity() { return atomDensity_; }
  DblNumVec&  PseudoCharge() { return pseudoCharge_; }
  std::vector<PseudoPot>& Pseudo() { return pseudo_; };
  std::vector<NonlocalGPP>& Vnlc() { return vnlc_; };
  std::vector<CpxNumTns>& Coef() { return coef_; };
#ifdef _COMPLEX_
  std::vector<DblNumVec>&  EigVal() { return eigVal_; }
  DblNumVec&  EigVal(Int i) { return eigVal_[i]; }
  std::vector<DblNumVec>&  OccupationRate() { return occupationRate_; }
  DblNumVec&  OccupationRate(Int i) { return occupationRate_[i]; }
#else
  DblNumVec&  EigVal() { return eigVal_; }
  DblNumVec&  OccupationRate() { return occupationRate_; }
#endif
  std::vector<Atom>&  AtomList() { return atomList_; }

  bool        IsEXXActive() { return isEXXActive_; }
  bool        IsHybrid() { return isHybrid_; }

  void        SetEXXActive(bool flag) { isEXXActive_ = flag; }

  Real        ScreenMu() { return screenMu_;}
  Real        EXXFraction() { return exxFraction_;}

  Real        EVdw() { return EVdw_; }
  Real        Eself() {return Eself_;}
  Real        EIonSR() {return EIonSR_;}
  Real        Eext() {return Eext_;}
  DblNumMat&  ForceVdw() { return forceVdw_;}
  DblNumMat&  ForceIonSR() {return forceIonSR_;}
  DblNumMat&  ForceIonLR() {return forceIonLR_;}
  DblNumMat&  ForceExt() {return forceext_;}

  void        SetVext(const DblNumVec& vext) {vext_ = vext;}
  void        SetEext(Real Eext) {Eext_=Eext;}
  void        SetForceExt(const DblNumMat& forceext) {forceext_ = forceext;}
  // Functions to set and toggle state of filter application
  void set_wfn_filter(int apply_filter, int apply_first, Real wfn_cutoff)
  {
    apply_filter_ = apply_filter;
    apply_first_ = apply_first;
    wfn_cutoff_ = wfn_cutoff;
  }
  void set_wfn_filter(int apply_first)
  {
    apply_first_ = apply_first;
  }
  // *********************************************************************
  // Inquiry
  // *********************************************************************
  Int  NumSpin() const { return numSpin_; }
  // For spin-unrestricted case, we set the same state number NumStateTotal/2
  // for spin-up and spin-down electrons
  // when nelec is odd, we set NumStateTotal as the minimal even number larger
  // than nelec
  Int  NumStateTotal() const { if( numDensityComponent_ == 2 )
                                 return ( numExtraState_ + numOccupiedState_ + 1 ) / 2 * 2;
                               else
                                 return ( numExtraState_ + numOccupiedState_);  
                             }
  Int  NumOccupiedState() const { return numOccupiedState_; }
  Int  NumExtraState() const { return numExtraState_; }
  Int  NumDensityComponent() const { return density_.n(); }
  Int  NumSpinorComponent() const { return numSpinorComponent_ ; }
  Int  SpinSwitch() const { return spinswitch_ ; }
  void SetSpinSwitch(Int i) { spinswitch_ = i ; return; }
};

// *********************************************************************
// Kohn-Sham class
// *********************************************************************
/// @brief Detailed implementation of spin-restricted, spin-unrestricted
/// and spin-noncollinear Kohn-Sham calculations.
class KohnSham: public Hamiltonian {
private: 
  /// @brief Store all the orbitals for exact exchange calculation
  /// FIXME: This might impose serious memory constraint for relatively
  /// large systems.
#ifdef _COMPLEX_
  std::vector<CpxNumTns>      phiEXX_;
  std::vector<CpxNumMat>      vexxProj_;
  // For spin-unrestricted case
  std::vector<CpxNumTns>      UpphiEXX_;
  std::vector<CpxNumTns>      DnphiEXX_;
  std::vector<CpxNumMat>      UpvexxProj_;
  std::vector<CpxNumMat>      DnvexxProj_;
#else
  DblNumTns                   phiEXX_; 
  DblNumMat                   vexxProj_;
  
  DblNumTns                   UpphiEXX_;
  DblNumTns                   DnphiEXX_;
  DblNumMat                   UpvexxProj_;
  DblNumMat                   DnvexxProj_;

  CpxNumTns                   phiEXXG_;
  CpxNumMat                   vexxProjG_;

  CpxNumTns                   UpphiEXXG_;
  CpxNumTns                   DnphiEXXG_;
  CpxNumMat                   UpvexxProjG_;
  CpxNumMat                   DnvexxProjG_;
#endif

#ifdef GPU
  cuDblNumMat                 cu_vexxProj_;
  cuDblNumMat                 cu_UpvexxProj_;
  cuDblNumMat                 cu_DnvexxProj_;
#endif

public:

  // *********************************************************************
  // Lifecycle
  // *********************************************************************
  KohnSham();
  ~KohnSham();

  virtual void Setup (
      const Domain&              dm,
      const PeriodTable&         ptable,
      const std::vector<Atom>&   atomList );

  // *********************************************************************
  // Operations
  // *********************************************************************

  virtual void CalculatePseudoPotential( PeriodTable &ptable, Fourier &fft, std::vector<IntNumVec> &grididx );

  virtual void CalculateAtomDensity( PeriodTable &ptable, Fourier &fft );
#ifdef _COMPLEX_
  virtual void CalculateDensity( const std::vector<Spinor> &psi, const std::vector<DblNumVec> &occrate,
               Real &val, Fourier& fft );
#else 
  virtual void CalculateDensity( const Spinor &psi, const DblNumVec &occrate, Real &val, Fourier& fft );
#endif
  virtual void CalculateGradDensity( Fourier& fft );

  virtual void CalculateXC ( Real &val, Fourier& fft );

  virtual void CalculateHartree( Fourier& fft );

  virtual void CalculateVtot( DblNumMat& vtot );

  virtual void CalculateEkin( Fourier& fft );

#ifdef _COMPLEX_
  /// @brief Calculate the Hellmann-Feynman force for each atom.
  virtual void CalculateForce ( PeriodTable &ptable, std::vector<Spinor>& psi, Fourier& fft );
  // Matrix vector multiplication
  virtual void MultSpinor(Spinor& psi, NumTns<Complex>& a3, Fourier& fft);

  virtual void InitializeEXX( Real ecutWavefunction, Fourier& fft );
  /// @brief Update phiEXX by the spinor psi.
  virtual void SetPhiEXX( std::vector<Spinor>& psi, Fourier& fft );
  /// @brief Read phiEXX from files
  virtual void SetPhiEXX( std::string WfnFileName, Fourier& fft );
  virtual std::vector<CpxNumTns>& PhiEXX() {return phiEXX_;}
  virtual void CalculateVexxACE( std::vector<Spinor>& psi, Fourier& fft );
  virtual void CalculateVexxACEDF( std::vector<Spinor>& psi, Fourier& fft, bool isFixColumnDF );
  virtual Real CalculateEXXEnergy( std::vector<Spinor>& psi, Fourier& fft );
#else
  virtual void CalculateForce ( Spinor& psi, Fourier& fft );
  virtual void MultSpinor(Spinor& psi, NumTns<Real>& a3, Fourier& fft);
  virtual void MultSpinor(Spinor& psi, NumTns<Complex>& a3, Fourier& fft);

  virtual void InitializeEXX( Real ecutWavefunction, Fourier& fft );
  virtual void SetPhiEXX( Spinor& psi, Fourier& fft );
  virtual void SetPhiEXX( std::string WfnFileName, Fourier& fft );
  virtual DblNumTns& PhiEXX() {return phiEXX_;}
  virtual void CalculateVexxACE( Spinor& psi, Fourier& fft );
  virtual void CalculateVexxACEDF( Spinor& psi, Fourier& fft, bool isFixColumnDF );
  virtual Real CalculateEXXEnergy( Spinor& psi, Fourier& fft );
#endif
  /// @brief Calculate ionic self energy and short range repulsion
  /// energy and force
  void  CalculateIonSelfEnergyAndForce( PeriodTable &ptable, Fourier &fft );

  /// @brief Calculate Van der Waals energy and force (which only depends on the
  /// atomic position)
  void  CalculateVdwEnergyAndForce();

  void  Setup_XC( std::string xc);

#ifdef GPU
  virtual void CalculateDensityGPU( const Spinor &psi, const DblNumVec &occrate, Real &val, Fourier& fft );
  virtual void MultSpinorGPU(Spinor& psi, cuNumTns<Real>& a3, Fourier& fft);
  virtual void ACEOperatorGPU( cuDblNumMat& cu_psi, Fourier& fft, cuDblNumMat& cu_Hpsi, const int spinswitch = 0 );
  virtual void CalculateVexxACEGPU( Spinor& psi, Fourier& fft );
#endif

};

} // namespace pwdft

#endif // _HAMILTONIAN_HPP_
