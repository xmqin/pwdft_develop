/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

Authors: Chris J. Pickard and Lin Lin

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
/// @file esdf.hpp
/// @brief Electronic structure data format for reading the input data.
/// @date 2012-08-10
#ifndef _ESDF_HPP_
#define _ESDF_HPP_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <limits.h>
#include "mpi.h"
#include "domain.hpp"
#include "numvec_impl.hpp"
#include "nummat_impl.hpp"
#include "numtns_impl.hpp"
#include "tinyvec_impl.hpp"

namespace pwdft{

// Forward declaration of Atom structure in periodtable.hpp 
struct Atom;

// *********************************************************************
// Electronic structure data format
// *********************************************************************
namespace esdf{

/************************************************************ 
 * Main routines
 ************************************************************/
void esdf_bcast();
void esdf_key();
void esdf_init(const char *);
void esdf_string(const char *, const char *, char *);
int esdf_integer(const char *, int);
float esdf_single(const char *, float);
double esdf_double(const char *, double);
double esdf_physical(const char *, double, char *);
bool esdf_defined(const char *);
bool esdf_boolean(const char *, bool);
bool esdf_block(const char *, int *);
char *esdf_reduce(char *);
double esdf_convfac(char *, char *);
int esdf_unit(int *);
void esdf_file(int *, char *, int *);
void esdf_lablchk(char *, char *, int *);
void esdf_die(char *);
void esdf_warn(char *);
void esdf_close();

/************************************************************ 
 * Utilities
 ************************************************************/
void getaline(FILE *, char *);
void getlines(FILE *fp, char **);
char *trim(char *);
void adjustl(char *,char *);
int len_trim(char *);
int indexstr(char *, char *);
int indexch(char *, char);
int countw(char *, char **, int);
char *strlwr(char *);
char *strupr(char *);

// *********************************************************************
// Input interface
// *********************************************************************

/// @struct ESDFInputParam
/// @brief Main structure containing input parameters for the
/// electronic structure calculation.
struct ESDFInputParam{

  /// @brief Global computational domain.
  ///
  /// Not an input parameter by the user.
  Domain              domain;

  /// @brief Whether to print detailed calculation time or not
  ///
  /// Default: 0
  bool                isPrintTime;

  /// @brief Type of the exchange-correlation functional.
  ///
  /// Default: "XC_LDA_XC_TETER93"
  ///
  /// The exchange-correlation functional is implemented using the
  /// libxc package. Currently only the LDA and GGA xc functionals is
  /// supported.
  std::string         XCType;

  /// @brief Whether to call Libxc to calculate XC functional
  ///
  /// Default: 1
  bool                isUseLibxc;

  /// @brief Type of the van der Waals correction.
  ///
  /// Default: "DFT-D2"
  ///
  /// Currently only the DFT-D2 correction is supported.
  std::string         VDWType;

  /// @brief Calculation type related to spin
  ///
  /// Default: 1
  ///
  /// -= 1                     : spin-restricted Kohn-Sham
  /// -= 2                     : spin-unrestricted Kohn-Sham
  /// -= 4                     : spin-noncollinear Kohn-Sham
  Int                 spinType;

  /// @brief Whether to add spin-orbit coupling to Hamiltonian
  ///
  /// Default: 0
  bool                SpinOrbitCoupling;

  /// @brief Whether the spin axis is fixed and magnetization is parallel
  ///
  /// Default: 0
  bool                isParallel;  

  /// @brief Whether a energy band calculation or not
  /// 
  /// Default: 0
  bool                isCalculateEnergyBand;

  /// @brief Type of the pseudopotential
  ///
  /// Default: "HGH"
  ///
  /// Currently HGH and ONCV are the only supported pseudopotential format.
  std::string         pseudoType;

  /// @brief File for storing the information of the HGH pseudopotential.
  ///
  /// Default: "HGH.bin"
  ///
  /// HGH pseudopotential is generated by the utility subroutine HGH.m.
  ///
  /// @note Only the master processor (mpirank == 0) reads this table,
  /// and the information is broadcast to other processors.
  std::string         periodTableFile;

  /// @brief File for storing the information of the ONCV pseudopotential.
  ///
  /// Default: None
  ///
  /// In the current PWDFT code, we use UPF file pseudopotential, which 
  /// is downloaded from QE website.
  ///
  /// @note Only the master processor (mpirank == 0) reads this table,
  /// and the information is broadcast to other processors.
  std::vector<std::string>    
                      pspFile;

  /// @brief Number of atom types
  Int                 numAtomType;

  /// @brief Types and positions of all atoms in the global
  /// computational domain.
  ///
  /// Not an input parameter by the user.
  std::vector<Atom>   atomList;

  /// @brief Mixing maximum dimension.
  ///
  /// Default: 8
  ///
  /// This parameter is relevant for Anderson mixing.
  Int                 mixMaxDim;

  /// @brief Coefficient in front of the preconditioned residual to be
  /// mixed with the mixing variable in the previous step.
  ///
  /// Default: 0.8
  ///
  /// For metallic systems or small gapped semiconductors,
  /// mixStepLength is often needed to be smaller than 0.1.  In such
  /// case, a better preconditioner such as Kerker preconditioner can
  /// be helpful.
  Real                mixStepLength;     

  /// @brief Mixing type for self-consistent field iteration.
  ///
  /// Default: anderson
  ///
  /// - = "anderson"           : Anderson mixing
  /// - = "kerker+anderson"    : Anderson mixing with Kerker
  std::string         mixType;

  /// @brief Which variable to mix
  ///
  /// Default: potential
  ///
  /// - = "density"            : Density mixing
  /// - = "potential"          : Potential mixing
  std::string         mixVariable;        

  /// @brief Tolerance for SCF iteration with non-hybrid functionals and
  /// the inner iteration for hybrid functional
  ///
  /// Default: 1e-9
  Real                scfInnerTolerance;

  /// @brief Energy tolerance for SCF iteration with non-hybrid functionals
  /// and the inner iteration for hybrid functional
  ///
  /// Default: 1e-6
  Real                scfInnerEnergyTolerance;
  
  /// @brief Minimum number of SCF calculation with non-hybrid functionals and
  /// the inner iteration for hybrid functional
  ///
  /// Default: 3
  Int                 scfInnerMinIter;

  /// @brief Maximum number of SCF calculation with non-hybrid functionals and
  /// the inner iteration for hybrid functional
  ///
  /// Default: 30
  Int                 scfInnerMaxIter; 

  /// @brief Maximum number of iterations for hybrid functional
  /// iterations.
  /// 
  /// Default: 20
  Int                 scfPhiMaxIter;

  /// @brief Tolerance for hybrid functional iterations using Fock
  /// energy
  ///
  /// Default: 1e-9
  Real                scfPhiTolerance;

  /// @brief Treatment of the divergence term in hybrid functional
  /// calculation.
  ///
  /// Default: 1
  ///
  /// - = 0                    : No regularization
  /// - = 1                    : Gygi-Baldereschi regularization
  Int                 exxDivergenceType;

  /// @brief Mixing type for performing hybrid functional calculations.
  ///
  /// Default: nested
  ///
  /// - = "nested"             : Standard nested two loop procedure
  /// - = "scdiis"             : Selected column DIIS
  /// - = "pcdiis"             : Projected commutator DIIS
  std::string         hybridMixType;

  /// @brief Whether to update the ACE operator twice in PCDIIS.
  ///
  /// Default: 1
  bool                isHybridACETwicePCDIIS;

  /// @brief Whether to use the adaptively compressed exchange (ACE)
  /// formulation for hybrid functional.
  ///
  /// Default: 1
  bool                isHybridACE;

  /// @brief Whether to use Fourier convolution to calculate the summation
  /// about k point index, only useful when ACE and ISDF are used together.
  ///
  /// Default: 0
  bool                isHybridFourierConv;

  /// @brief Whether to initialize hybrid functional in the initial
  /// step.
  ///
  /// Default: 0
  bool                isHybridActiveInit;
  
  /// @brief Whether to use the density fitting formalism for
  /// hybrid functional. Currently this must be used with the ACE
  /// formulation.
  ///
  /// Default: 0
  bool                isHybridDF;

  /// @brief Type for selecting points in density fitting
  ///
  /// Default: "QRCP"
  ///
  /// Currently two types (QRCP and Kmeans) are supported.
  std::string         hybridDFType;

  /// @brief The method to calculate the weight function for complex kmeans
  ///
  /// Default: "Add" 
  ///
  /// - = "Add"                : psi^2 + phi^2
  /// - = "Multi"              : (psi^2)(phi^2)
  std::string         hybridDFKmeansWFType;

  /// @brief when hybridDFKmeansWFType == Add, weight function = psi^alpha + phi^alpha
  ///
  /// Default: "2.0"
  Real                hybridDFKmeansWFAlpha;
 
  /// @brief The tolerance used in K-means method of density fitting
  ///
  /// Default: 1e-3
  Real                hybridDFKmeansTolerance;
  
  /// @brief The max number of iterations used in K-means method of density fitting
  ///
  /// Default: 99
  Int                 hybridDFKmeansMaxIter;
  
  /// @brief Density fitting uses numMu * numStateTotal number of
  /// states for hybrid calculations with density fitting.
  ///
  /// Default: 6.0
  Real                hybridDFNumMu;
  
  /// @brief Density fitting uses numGaussianRandom * numMu * numStateTotal number of
  /// states for GaussianRandom.
  ///
  /// Default: 1.5
  Real                hybridDFNumGaussianRandom;
  
  /// @brief Density fitting uses this number of cores in ScaLAPACAL
  ///
  /// Default: mpisize
  Int                 hybridDFNumProcScaLAPACK;
  
  /// @brief Density fitting uses the tolerance to remove the matrix element
  ///
  /// Default: 1e-20
  Real                hybridDFTolerance;
  
  /// @brief Calculation method for the least square 
  ///
  /// Default: 1 
  Int                 hybridDFLSmethod;

  /// @brief Solver for the planewave problem.  
  ///
  /// Default: "LOBPCG"
  ///
  /// - = "LOBPCG"       
  /// - = "LOBPCGScaLAPACK"       
  /// - = "PPCG"       
  /// - = "PPCGScaLAPACK"       
  /// - = "CheFSI"       
  std::string         PWSolver;  

  /// @brief Whether to control the tolerance of the eigensolver
  /// dynamically.
  ///
  /// Default: 1
  ///
  /// When isEigToleranceDynamic = 1, the tolerance for the
  /// eigensolver is controlled dynamically and is related to the
  /// error in the current %SCF iteration.  The lower limit of the
  /// tolerance is controlled by
  /// @ref pwdft::esdf::ESDFInputParam::eigTolerance "eigTolerance".
  bool                isEigToleranceDynamic;

  /// @brief Tolerance for the eigenvalue solver
  ///
  /// Default: 1e-6
  ///
  /// Currently the LOBPCG method is used as the eigenvalue solver for
  /// obtaining the adaptive local basis functions, and eigTolerance
  /// controls the tolerance for the LOBPCG solver.  
  ///
  /// In the case when the eigensolver tolerance is tunned dynamically
  /// (see 
  /// @ref pwdft::esdf::ESDFInputParam::isEigToleranceDynamic "isEigToleranceDynamic"), the tolerance for
  /// the eigensolver is controlled dynamically and can be larger than
  /// eigTolerance.
  Real                eigTolerance;

  /// @brief Tolerance for minimum of the residual that should be
  /// reached by the eigensolver
  ///
  /// Default: 1e-3
  /// 
  /// Currently if eigMinTolerance is not reached, the LOBPCG
  /// iterations continue regardless of whether eigMaxIter is reached
  Real                eigMinTolerance;

  /// @brief Minimum number of iterations for the eigensolver.
  ///
  /// Default: 2
  Int                 eigMinIter;

  /// @brief Maximum number of iterations for the eigensolver.
  ///
  /// Default: 3
  Int                 eigMaxIter;

  // Filter Order for the first phase of Chebyshev Filtered SCF iterations 
  ///
  /// Default: 40
  Int                 First_SCF_PWDFT_ChebyFilterOrder;

  ///
  ///
  /// Default: 5
  Int                 First_SCF_PWDFT_ChebyCycleNum; 

  /// Filter Order for general phase of Chebyshev Filtered SCF iterations
  ///
  /// Default: 35
  Int                 General_SCF_PWDFT_ChebyFilterOrder;

  /// Whether to use ScaLAPACK in CheFSI
  ///
  /// Default: 1
  bool                PWDFT_Cheby_use_scala; 

  /// Whether to
  ///
  /// Default: 1
  bool                PWDFT_Cheby_apply_wfn_ecut_filt; 

  /// @brief Subblock size used in PPCG in PWDFT.  
  ///
  /// Default: 1
  ///
  /// - = "1"       
  /// - = "Nband+Extra_States" : Equivalent to LOBPCG      
  Int                 PPCGsbSize;  

  /// @brief Whether to use the saved electron density as the start.
  ///
  /// Default: 0
  bool                isRestartDensity;

  /// @brief Whether to use the saved basis functions in extended
  /// element as the start.
  ///
  /// Default: 0
  bool                isRestartWfn;

  /// @brief Whether to output the electron density.
  ///
  /// Default: 0
  ///
  /// When isOutputDensity = 1, files DEN_xxx_yyy will be generated,
  /// where by default xxx is the mpirank (starting from 0), and yyy
  /// is mpisize.
  ///
  /// This option is needed to restart the electron density using 
  /// @ref pwdft::esdf::ESDFInputParam::isRestartDensity "isRestartDensity".
  bool                isOutputDensity;

  /// @brief Whether to output the total potential.
  ///
  /// Default: 0
  ///
  /// When isOutputPotential = 1, files POT_xxx_yyy will be generated,
  /// where by default xxx is the mpirank (starting from 0), and yyy
  /// is mpisize.
  bool                isOutputPotential;
  
  /// @brief Whether to output the wavefunctions in the extended
  /// element.
  ///
  /// Default: 0
  ///
  /// When isOutputWfn = 1, the approximate eigenvectors in the
  /// extended element are given in the output, in the form
  /// WFN_xxx_yyy, where by default xxx is the mpirank (starting
  /// from 0), and yyy is mpisize.
  bool                isOutputWfn;

  /// @brief Whether to output the eigenvalues
  ///
  /// Default: 0
  bool                isOutputEigvals;

  /// @brief Whether to use the sum of atomic density as the initial
  /// guess for electron density.
  ///
  /// Default: 1
  ///
  /// Currently only works for ONCV pseudopotential and not for HGH
  /// pseudopotential.
  bool                isUseAtomDensity;

  /// @brief Whether to use the VLocal generated from the
  /// pseudopotential, and use Gaussian pseudocharge as compensation
  /// charge
  ///
  /// Default: 1
  ///
  /// @todo This option will become obsolete in the future when
  /// pseudopotential is read from text files.
  bool                isUseVLocal;

  /// @brief Whether to use the real space method 
  ///
  /// Default: 0
  bool                isUseRealSpace;

  /// @brief Whether to perform sphere truncation on Fourier space
  ///
  /// determined automatically according to "isUseRealSpace"
  bool                isUseSphereCut;  

  /// @brief Temperature in the unit of Kelvin
  Real                temperature;

  /// @brief Inverse of temperature.
  ///
  /// Default: 1.0 / (100 K * k_b)
  ///
  /// This parameter is not controlled directly, but through 
  /// "Temperature" in the input file, in the unit of Kelvin.
  Real                Tbeta;    

  /// @brief smearing scheme for fractional occupations 
  ///
  /// Default: "FD"
  /// - = "FD"                 : Fermi-Dirac distribution
  /// - = "GB"                 : Gaussian Broadening or Methfessel / Paxton of order 0
  /// - = "MP"                 : Methfessel-Paxton smearing - order 2 default, up to order 3 supported 
  std::string         smearing_scheme;

  /// @brief Number of empty states for finite temperature
  /// calculation.
  ///
  /// Default: 0
  ///
  /// This parameter must be larger than 0 for small gapped systems or
  /// relatively high temperature calculations.
  Int                 numExtraState;

  /// @brief Some states for the planewave solver are unused in order
  /// to accelerate the convergence rate of the eigensolver.
  ///
  /// Default: 0
  Int                 numUnusedState;
  
  /// @brief Number of extra electrons
  ///
  /// Default: 0
  Int                 extraElectron; 

  /// @brief The blocksize of cores in ScaLAPACAL
  ///
  /// Default: 32
  Int                 BlockSizeScaLAPACK;

  /// @brief Number of processors used by ScaLAPACK in the PW part
  ///
  /// Default: mpisize for PWDFT. 
  Int                 numProcScaLAPACKPW;

  /// @brief Block size for ScaLAPACK.
  ///
  /// Default: 32
  ///
  /// Only used when ScaLAPACK is invoked.
  Int                 scaBlockSize;
 
  /// @brief The number of k-point group
  ///
  /// Default: 1
  Int                 NumGroupKpoint;

  /// @brief The blocksize of grids for dividing wavefunction
  ///
  /// Default: 1
  Int                 BlockSizeGrid;

  /// @brief The blocksize of states for dividing wavefunction
  ///
  /// Default: 32
  Int                 BlockSizeState;

  /// @brief Kinetic energy cutoff for the wavefunction on the uniform
  /// grid.
  ///
  /// Default: 20.0 Ha
  ///
  /// The number of uniform grids for the wavefunction along each
  /// direction i (i=x,y,z) is given by the formula
  /// \f[
  ///    N_i = \sqrt{2*ecutWavefunction}*L_i
  /// \f]
  /// where \f$L_i\f$ is the dimension of the domain
  /// along the direction i. The domain can be the global domain,
  /// extended element or element.
  Real                ecutWavefunction;

  /// @brief The ratio between the number of grids for the density and
  /// the wavefunction in the uniform grid along each dimension.
  ///
  /// Default: 2.0
  ///
  /// The number of uniform grids for the density and potential along
  /// each direction i (i=x,y,z) is given by the formula
  /// \f[
  ///    N_i = densityGridFactor * \sqrt{2*ecutWavefunction}*L_i
  /// \f]
  /// where \f$L_i\f$ is the dimension of the domain
  /// along the direction i. The domain can be the global domain,
  /// extended element or element.
  Real                densityGridFactor;

  /// @brief The type of FFT grids
  ///
  /// Default: "even"
  ///
  /// - = "even"              
  /// - = "odd"
  /// - = "power"               
  std::string         FFTtype;

  /// @brief Maximum number of steps for geometry optimization /
  /// molecular dynamics
  ///
  /// Default: 0 
  Int                 ionMaxIter;

  /// @brief Mode for geometry optimization and molecular dynamics
  ///
  /// Default: NONE
  ///
  /// Geometry optimization routine:
  ///
  /// ionMove = bb               : BarzilaiBorwein 
  ///           cg               : conjugate gradient
  ///           bfgs             : BFGS
  ///           fire             : FIRE optimization method
  ///
  /// Molecular dynamics routine:
  ///
  /// ionMove = verlet           : Velocity-Verlet (NVE)
  ///           nosehoover1      : Nose-Hoover method with chain
  ///                              level 1
  ///           langevin         : Langevin dynamics
  std::string         ionMove;

  /// @brief Maximum force for geometry optimization
  ///
  /// Default: 0.001 
  Real                geoOptMaxForce;
  
  /// @brief Line search step length parameter in NLCG for Geo Opt 
  ///
  /// Default = 0.02
  Real                geoOpt_NLCG_sigma; 
  
  // These variables are related to the FIRE optimizer
  Int  FIRE_Nmin; // Set to 10 by default
  Real FIRE_dt; // Set to 40.0 a.u. (= 1 femtosecond) by default
  Real FIRE_atomicmass; // Set to 4.0 by default

  /// @brief Temperature for ion.
  ///
  /// Default: 300 K
  Real                ionTemperature;

  /// @brief Inverse of ionTemperature.
  ///
  /// Default: 1.0 / (100 K * k_b)
  ///
  /// This parameter is not controlled directly, but through 
  /// "ionTemperature" in the input file, in the unit of Kelvin.
  Real                TbetaIonTemperature; 

  /// @brief Time step for MD simulation.
  ///
  /// Default: 40.0
  Int                 MDTimeStep; 
  /// @brief Extrapolation type for updating the density or wavefunction
  ///
  /// Default: "linear"
  /// 
  /// = "linear"        : 1-step extrapolation 
  /// = "quadratic"     : 2-step extrapolation
  /// = "aspc2"         : 2-step ASPC extrapolation
  /// = "aspc3"         : 3-step ASPC extrapolation
  /// = "xlbomd"        : wavefunction extrapolation by using XLBOMD
  
  std::string         MDExtrapolationType;
  /// @brief Extrapolation variable
  ///
  /// Default: "density"
  ///
  /// = "density"        : density extrapolation
  /// = "wavefun"        : wavefunction extrapolation
  ///                      currently only available in PWDFT. The
  ///                      density is constructed from the
  ///                      wavefunctions
  std::string         MDExtrapolationVariable;
             
  /// @brief Mass for Nose-Hoover thermostat
  ///
  /// Default: 85000.0
  Real                qMass;   

  /// @brief Dampling factor for Langevin theromostat
  ///
  /// Default: 0.01
  Real                langevinDamping;

  /// @brief Kappa value of XL-BOMD
  ///
  /// Default: 1.7
  Real                kappaXLBOMD;

  /// @brief Whether to use the previous position
  ///
  /// Default: 0
  bool                isRestartPosition;

  /// @brief Whether to use the previous velocity and thermostat
  ///
  /// Default: 0
  bool                isRestartVelocity;

  /// @brief Whether to output position information stored in
  /// lastPos.out
  ///
  /// Default: 1
  bool                isOutputPosition;

  /// @brief Whether to output velocity and thermostat information
  /// stored in lastVel.out
  ///
  /// Default: 1
  bool                isOutputVelocity;

  /// @brief Output the atomic position in XYZ format. Used in MD
  /// simulation and geometry optimization. Outputs MD.xyz. This
  /// only stores the finished configuration.
  ///
  /// Default: 1
  bool                isOutputXYZ;

  /// @brief From which ion iteration to engage energy based convergence in MD
  /// 
  /// Default: ionMaxIter + 1
  Int                 MDscfEnergyCriteriaEngageIonIter;

  /// @brief Etot tolerance in Energy based convergence
  /// The difference in Etot should be less than this in energy based SCf convergence
  Real                MDscfEtotdiff;

  /// @brief Eband tolerance in Energy based convergence
  /// The difference in Eband should be less than this in energy based SCf convergence
  Real                MDscfEbanddiff;

  /// @brief Maximum number of iterations for hybrid functional
  /// iterations in MD
  /// 
  /// Default: the same as scfPhiMaxIter
  Int                 MDscfPhiMaxIter;

  /// @brief Maximum number of inner SCF iterations in MD
  ///
  /// Default: the same as scfInnerMaxIter
  Int                 MDscfInnerMaxIter;

  /// RTTDDFT parameters

  /// @brief use TDDFT or not
  /// 
  /// Default: 0
  bool                isTDDFT;

  /// @brief Maximum number of outer SCF iterations in TDDFT
  /// This is usually used in PWDFT  
  ///
  /// Default: 0
  Int                 restartTDDFTStep;

  /// @brief auto save TDDFT WFN and DEN 
  /// 
  /// Default: 20
  Int                 TDDFTautoSaveSteps;

  /// @brief use TDDFT ehrenfest dynamics
  /// 
  /// Default: 1
  bool                isTDDFTEhrenfest;

  /// @brief use TDDFT  Vexternal or not
  /// 
  /// Default: 1
  bool                isTDDFTVext;

  /// @brief calculate TDDFT dipole or not
  /// 
  /// Default: 1
  bool                isTDDFTDipole;

  /// @brief use TDDFT  Vexternal polorization in the X direction.
  /// 
  /// Default: 1.0
  Real                TDDFTVextPolx;

  /// @brief use TDDFT  Vexternal polorization in the X direction.
  /// 
  /// Default: 0.0
  Real                TDDFTVextPoly;

  /// @brief use TDDFT  Vexternal polorization in the X direction.
  /// 
  /// Default: 0.0
  Real                TDDFTVextPolz;

  /// @brief use TDDFT Vexternal Frequency.
  /// 
  /// Default: 18.0/27.211385
  Real                TDDFTVextFreq;

  /// @brief use TDDFT Vexternal Phase
  /// 
  /// Default: 0.0
  Real                TDDFTVextPhase;

  /// @brief use TDDFT Vexternal Amplitude 
  /// 
  /// Default: 0.0194
  Real                TDDFTVextAmp;

  /// @brief use TDDFT Vexternal T0
  /// 
  /// Default: 13.6056925
  Real                TDDFTVextT0;

  /// @brief use TDDFT Vexternal Tau
  /// 
  /// Default: 13.6056925
  Real                TDDFTVextTau;

  /// @brief types of external fields
  ///
  /// Default: gaussian 
  ///
  /// - = "guassian"             : Gaussian type of External field
  /// - = "constant"             : Constant type of External field
  /// - = "sinsq"                : sin type of the external field
  /// - = "erf"                  : Exp type of the external field
  /// - = "kick "                : Kick type of the external field                     
  ///
  std::string         TDDFTVextEnv;

  /// @brief types of TDDFT methods
  ///
  /// Default: PTTRAP
  ///
  /// - = "PTTRAP"       : implicit Parallel transport evolution with trapezoidal rule
  /// - = "RK4"          :  Explicit Runge-Kutta 4th order method
  ///
  std::string         TDDFTMethod;

  /// @brief TDDFT delta T
  /// 
  /// Default: 1.0 atomic unit
  Real                TDDFTDeltaT;

  /// @brief TDDFT total T
  /// 
  /// Default: 40.0 atomic unit
  Real                TDDFTTotalT;

  /// @brief TDDFT Krylov Max iteration number
  /// 
  /// Default: 30
  Int                 TDDFTKrylovMax;

  /// @brief TDDFT Krylov Tolerance
  /// 
  /// Default: 1.0E-7
  Real                TDDFTKrylovTol;

  /// @brief TDDFT Phi Tolerance
  /// 
  /// Default: 1.0E-8
  Real                TDDFTPhiTol;

  /// @brief TDDFT Diis Tolerance
  /// 
  /// Default: 1.0E-6
  Real                TDDFTDiisTol;

  /// @brief Maximum number of iterations for hybrid functional
  /// iterations in TDDFT
  /// 
  /// Default: the same as PhiMaxIter
  Int                 TDDFTPhiMaxIter;

  /// @brief Maximum number of outer SCF iterations in TDDFT
  /// This is usually used in PWDFT  
  Int                 TDDFTDiisMaxIter;

  // Inputs related to LRTDDFT  
  bool isLRTDDFT;                 // Default 0
  bool isLRTDDFTISDF;             // Default 0
  bool isOutputExcitationEnergy;  // Default 1
  bool isOutputExcitationWfn;     // Default 0
  Int  nvband;                    // Default 1
  Int  ncband;                    // Default 1
  Int  nkband;                    // LOBPCG
  Int  startband;                 // LOBPCG
  Int  endband;                   // LOBPCG
  Int  numProcEigenSolverLRTDDFT;
  bool isdouble;                  // Default 1
  bool isfloat;                   // Default 0
  Real numMuFacLRTDDFTISDF;
  Real numGaussianRandomFacLRTDDFTISDF;
  Real toleranceLRTDDFT; 
  Int  maxIterKmeansLRTDDFTISDF;
  Real toleranceKmeansLRTDDFTISDF; 
  Int  eigMaxIterLRTDDFT;
  Real eigMinToleranceLRTDDFT;
  Real eigToleranceLRTDDFT; 
  std::string         ipTypeLRTDDFTISDF; 
  std::string         eigenSolverLRTDDFT;

  // Inputs related to Spectrum
  bool isOutputExcitationSpectrum; 
  Real LRTDDFTVextPolx;
  Real LRTDDFTVextPoly;
  Real LRTDDFTVextPolz;
  Real LRTDDFTOmegagrid;
  Real LRTDDFTSigma;

  // Inputs related to GW-BSE
  /// @brief use GW or not
  ///
  /// Default: 0
  bool isGW;

  /// @brief Number of occupied states to calculate epsilon
  ///
  /// Default: 1
  Int nv_oper;   

  /// @brief Number of unoccupied states to calculate epsilon
  ///
  /// Default: 1
  Int nc_oper; 

  /// @brief Cutoff of dielectric matrix
  ///
  /// Default: equal to cutoff of wave
  Int nv_ener;
  Int nc_ener;

  bool isGWISDF;
  bool isRestartGW;
  std::string isdf_method;
  bool iscauchy;
  Int vcrank_ratio;
  Int vsrank_ratio;
  Int ssrank_ratio;
  Real epsilon_cutoff;

  Int  maxiterkmeans_GW_ISDF;
  Real tolerancekmeans_GW_ISDF;

  /// @brief frequency dependence of the inverse dielectric matrix
  ///
  /// Default: COHSEX
  std::string ipTypeFrequency_Dep;

  /// @brief Type of Coulomb screening
  ///
  /// Default: semiconductor
  std::string ipTypeCoulomb_screen;

  /// @brief Type of Coulomb truncation
  ///
  /// Default: spherical_truncation
  std::string ipTypeCoulomb_trunc;

  /// @brief Radius of spherical truncation
  ///
  /// Default: 0.0
  Real spherical_radius;

  /// @brief Whether to write Coulomb interaction
  ///
  /// Default: 0
  bool iswriteCoulomb;

  /// @brief Number of selected occupied states to calculate quasiparticle energy
  ///
  /// Default: 1
  Int nvbands_sigma;

  /// @brief Number of selected unoccupied states to calculate quasiparticle energy 
  ///
  /// Default: 1
  Int ncbands_sigma;

  /// @brief use BSE or not
  ///
  /// Default: 0
  bool isBSE;

  /// @brief Number of selected valence bands to construct BSE Hamiltonian
  ///
  /// Default: 1
  Int BSE_valence;

  /// @brief Number of selected conduction bands to construct BSE Hamiltonian
  ///
  /// Default: 1
  Int BSE_conduction;

  /// @brief Method to construct BSE Hamiltonian
  ///
  /// Default: TDA
  std::string ipTypeBSEmethod;

  /// @brief Method to construct optical transition probabilities
  ///
  /// default: momentum
  std:: string ipTypedipole;

  /// @brief Direction of optical dipole
  ///
  /// Default: 0 0 0
  Real BSEVextPolX;
  Real BSEVextPolY;
  Real BSEVextPolZ;

  /// @brief Eigensolver of BSE Hamiltonian
  ///
  /// Default: LAPACK
  std::string EigenSolver_BSE;

  /// @brief Type for generating absorption spectrum with broadening method
  /// Default: Gaussian
  std::string ipTypeBroadening_method;

  /// @brief Numerical broadening width
  ///
  /// Default: 0.1
  Real broadening_width;

  /// @brief Optical spectrum step
  ///  
  /// Default: 0.0
  Real optical_step;
  
  /// @brief Whether to write eigenvectors of BSE hamiltonion
  ///
  /// Default: 0
  bool write_eigenvectors;

  // Inputs related to RPA
  bool isRPA;
  std::string freq_int_method;
  Int numFrequencyRPA;  
  bool isRestartRPA;
  bool isUseSphereCutRPA;
};

void ESDFReadInput( const std::string filename );

void ESDFReadInput( const char* filename );

void ESDFPrintInput( );

// Global input parameters
extern ESDFInputParam  esdfParam;

} // namespace esdf
} // namespace pwdft
#endif // _ESDF_HPP_
