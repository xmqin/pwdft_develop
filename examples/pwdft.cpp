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
/// @file pwdft.cpp
/// @brief Main driver for self-consistent field iteration using plane
/// wave basis set.  
///
/// The current version of pwdft is a sequential code and is used for
/// testing purpose, both for energy and for force.
/// @date 2013-10-16 Original implementation
/// @date 2014-02-01 Dual grid implementation
/// @date 2014-07-15 Parallelization of PWDFT.
/// @date 2016-03-07 Refactoring PWDFT to include geometry optimization
/// and molecular dynamics.
#include "pwdft.hpp"

using namespace pwdft;
using namespace std;
using namespace pwdft::esdf;
using namespace pwdft::scalapack;

void Usage(){
  std::cout 
    << "pwdft -in [inFile]" << std::endl
    << "in:             Input file (default: config.yaml)" << std::endl;
}

int main(int argc, char **argv) 
{
  MPI_Init(&argc, &argv);
  int mpirank, mpisize;
  MPI_Comm_rank( MPI_COMM_WORLD, &mpirank );
  MPI_Comm_size( MPI_COMM_WORLD, &mpisize );

  Real timeSta, timeEnd, timeSta1, timeEnd1;

  GetTime( timeSta1 );

  if( mpirank == 0 )
    Usage();

  try
  {
    // *********************************************************************
    // Input parameter
    // *********************************************************************

    // Initialize input parameters
    std::map<std::string,std::string> options;
    OptionsCreate(argc, argv, options);

    std::string inFile;                   
    if( options.find("-in") != options.end() ){ 
      inFile = options["-in"];
    }
    else{
      inFile = "config.yaml";
    }

    std::string outFile;                   
    if( options.find("-out") != options.end() ){ 
      outFile = options["-out"];
    }
    else{
      outFile = "statfile";
    }

    // Initialize log file
#ifdef _RELEASE_
    // In the release mode, only the master processor outputs information
    if( mpirank == 0 ){
      stringstream  ss;
      ss << outFile;
      statusOFS.open( ss.str().c_str() );
    }
#else
    // Every processor outputs information
    {
      stringstream  ss;
      ss << outFile << "." << mpirank;
      statusOFS.open( ss.str().c_str() );
    }
#endif

    Print( statusOFS, "mpirank in MPI_COMM_WORLD = ", mpirank );
    Print( statusOFS, "mpisize of MPI_COMM_WORLD = ", mpisize );
  
    // Read ESDF input file. Note: esdfParam is a global variable (11/25/2016)
    ESDFReadInput( inFile.c_str() );

    // Print the initial state
    ESDFPrintInput( );

    // Initialize multithreaded version of FFTW
#ifdef _USE_FFTW_OPENMP_
#ifndef _USE_OPENMP_
    ErrorHandling("Threaded FFTW must use OpenMP.");
#endif
    statusOFS << "FFTW uses " << omp_get_max_threads() << " threads." << std::endl;
    fftw_init_threads();
    fftw_plan_with_nthreads(omp_get_max_threads());
#endif

    // *********************************************************************
    // Preparation
    // *********************************************************************
    SetRandomSeed(mpirank);

    Domain&  dm = esdfParam.domain;

    PeriodTable ptable;
    Fourier fft;
    Spinor  psi;

    EigenSolver eigSol;
    KohnSham hamKS;
    SCF  scf;

    ptable.Setup( );

    fft.Initialize( dm );

    fft.InitializeFine( dm );

    fft.InitializeSphere( dm );

    // Hamiltonian
    hamKS.Setup( dm, ptable, esdfParam.atomList );

    DblNumVec& vext = hamKS.Vext();
    SetValue( vext, 0.0 );

    // Wavefunctions
    int numStateTotal = hamKS.NumStateTotal();
    int numComponent = hamKS.NumSpinorComponent();
    if( hamKS.NumDensityComponent() == 2 ){
      numComponent /= 2;
    }

    // Safeguard for Chebyshev Filtering
    if(esdfParam.PWSolver == "CheFSI")
    { 
      if(numStateTotal % mpisize != 0)
      {
        MPI_Barrier(MPI_COMM_WORLD);  
        statusOFS << std::endl << std::endl 
            << " Input Error ! Currently CheFSI within PWDFT requires total number of bands \
            to be divisble by mpisize. " << std::endl << " Total No. of states = " << 
            numStateTotal << " , mpisize = " << mpisize << " ." << std::endl <<  
            " Use a different value of extrastates." << endl << " Aborting ..." <<
            std::endl << std::endl;

        MPI_Barrier(MPI_COMM_WORLD);
        exit(-1);  
      }    
    }

    if( esdfParam.isUseSphereCut == false ){ 
      psi.Setup( dm, numComponent, numStateTotal, D_ZERO );
      UniformRandom( psi.Wavefun() );
    }
    else{
      psi.Setup( dm, numComponent, numStateTotal, Z_ZERO );
      CpxNumTns& wavefun = psi.WavefunG();
      UniformRandom( wavefun );

      // psi(G = 0) is limited to be real, this is important
      for( Int j = 0; j < wavefun.p(); j++ ){
        wavefun(0,0,j) = Complex( wavefun(0,0,j).real(), 0.0 );
      }
    }

    std::vector<IntNumVec> grididx( 1 );
    grididx[0] = psi.GridIdx();
    
    hamKS.CalculatePseudoPotential( ptable, fft, grididx );   

    if( hamKS.IsHybrid() ){
      GetTime( timeSta );
      hamKS.InitializeEXX( esdfParam.ecutWavefunction, fft );
      GetTime( timeEnd );
      statusOFS << "Time for setting up the exchange for the Hamiltonian part = "
        << timeEnd - timeSta << " [s]" << std::endl;
      if( esdfParam.isHybridActiveInit || esdfParam.isCalculateEnergyBand )
        hamKS.SetEXXActive(true);
    }

    // Eigensolver class
    eigSol.Setup( hamKS, psi, fft );

    scf.Setup( eigSol, ptable );

    // *********************************************************************
    // Single shot calculation first
    // *********************************************************************
    scf.Iterate();
    GetTime( timeEnd1 );
    statusOFS << "! Total time for the SCF iteration = " << timeEnd1 - timeSta1
      << " [s]" << std::endl;

    // *********************************************************************
    // Geometry optimization or Molecular dynamics
    // *********************************************************************
    if( esdfParam.ionMaxIter >= 1 ){
      IonDynamics ionDyn;

      ionDyn.Setup( hamKS.AtomList(), ptable ); 

      // Change the SCF parameters if necessary
      scf.UpdateMDParameters( );

      Int maxHist = ionDyn.MaxHist();
      // Need to define both but one of them may be empty
      std::vector<DblNumMat>    densityHist(maxHist);

      std::vector<DblNumTns>    wavefunHist(maxHist);
      DblNumTns                 wavefunPre;           // predictor

      if( esdfParam.MDExtrapolationVariable == "density" ){
        // densityHist[0] is the lastest density
        for( Int l = 0; l < maxHist; l++ ){
          densityHist[l] = hamKS.Density();
        } // for (l)
      }
      if( esdfParam.MDExtrapolationVariable == "wavefun" ){
        // wavefunHist[0] is the lastest density
        for( Int l = 0; l < maxHist; l++ ){
          wavefunHist[l] = psi.Wavefun();
        } // for (l)
        wavefunPre = psi.Wavefun();
      }

      // Main loop for geometry optimization or molecular dynamics
      // If ionMaxIter == 1, it is equivalent to single shot calculation
      Int ionMaxIter = esdfParam.ionMaxIter;

      for( Int ionIter = 1; ionIter <= ionMaxIter; ionIter++ ){
        {
          std::ostringstream msg;
          msg << "Ion move step # " << ionIter;
          PrintBlock( statusOFS, msg.str() );
        }

        if(ionIter >= 1)
          scf.set_Cheby_iondynamics_schedule_flag(1);

        // Get the new atomic coordinates
        // NOTE: ionDyn directly updates the coordinates in Hamiltonian
        ionDyn.SetEpot( scf.Efree() );
        ionDyn.MoveIons(ionIter);

        GetTime( timeSta );
        hamKS.UpdateHamiltonian( hamKS.AtomList() );

        std::vector<IntNumVec> grididx( 1 );
        grididx[0] = psi.GridIdx();

        hamKS.CalculatePseudoPotential( ptable, fft, grididx );

        // Reset wavefunctions to random values for geometry optimization
        // Except for CheFSI
        if((ionDyn.IsGeoOpt() == true) && (esdfParam.PWSolver != "CheFSI")){
          statusOFS << std::endl << " Resetting to random wavefunctions ... \n" << std::endl ; 
          UniformRandom( psi.Wavefun() );
        }

        scf.Update( ); 
        GetTime( timeEnd );
        statusOFS << "Time for updating the Hamiltonian = " << timeEnd - timeSta
          << " [s]" << std::endl;        

        ionDyn.Extrapolate( ionIter, maxHist, hamKS, psi, fft,
            densityHist, wavefunHist, wavefunPre );

        GetTime( timeSta );
        scf.Iterate( );
        GetTime( timeEnd );
        statusOFS << "! Total time for the SCF iteration = " << timeEnd - timeSta
          << " [s]" << std::endl;

        // Geometry optimization
        if( ionDyn.IsGeoOpt() ){
          if( MaxForce( hamKS.AtomList() ) < esdfParam.geoOptMaxForce ){
            statusOFS << "Stopping criterion for geometry optimization has been reached." << std::endl
              << "Exit the loops for ions." << std::endl;
            break;
          }
        }
      } // ionIter
    }

    // *********************************************************************
    // LRTDDFT
    // *********************************************************************   
    if( esdfParam.isLRTDDFT ){
      statusOFS << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ " << std::endl;
      statusOFS << " ! Begin the LRTDDFT simulation now " << std::endl;
      statusOFS << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ " << std::endl;

      LRTDDFT lrtddft; 

      GetTime( timeSta );
      lrtddft.Setup(hamKS, psi, fft, dm);
      if( esdfParam.isLRTDDFTISDF ){
        lrtddft.CalculateLRTDDFT_ISDF(hamKS, psi, fft, dm);
      }
      else{
        lrtddft.CalculateLRTDDFT(hamKS, psi, fft, dm);
      }

      GetTime( timeEnd );
      statusOFS << "! LRTDDFT used time: " << timeEnd - timeSta << " [s]" <<std::endl;
      statusOFS << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ " << std::endl;
    }

    // *********************************************************************
    // GW
    // ********************************************************************* 
  }
  catch( std::exception& e )
  {
    std::cerr << " caught exception with message: "
      << e.what() << std::endl;
  }

  // Finalize 
#ifdef _USE_FFTW_OPENMP
  fftw_cleanup_threads();
#endif
  MPI_Finalize();

  return 0;
}
