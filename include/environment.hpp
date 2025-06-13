/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

Author: Lin Lin

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
/// @file environment.hpp
/// @brief Environment variables for DGDFT.
/// @date 2013-09-06
#ifndef _ENVIRONMENT_DECL_HPP_
#define _ENVIRONMENT_DECL_HPP_

// STL libraries
#include <iostream> 
#include <iomanip> 
#include <fstream>
#include <sstream>
#include <unistd.h>

#include <cfloat>
#include <complex>
#include <string>
#include <cstring>

#include <set>
#include <map>
#include <stack>
#include <vector>

#include <algorithm>
#include <cmath>

#include <cassert>
#include <stdexcept>

// FFTW libraries
#include <fftw3.h>
#include <fftw3-mpi.h>

// MPI
#include "mpi.h"

// OpenMP
#ifdef OPENMP
#define _USE_OPENMP_ 
#include <omp.h>
#endif

#ifdef FFTWOPENMP
#define _USE_FFTW_OPENMP_
#endif


// Google coredumper for debugging
#ifdef COREDUMPER
#define _COREDUMPER_
#endif

// *********************************************************************
// Redefine the global macros
// *********************************************************************

// The verbose level of debugging information
#ifdef  DEBUG
#define _DEBUGlevel_ DEBUG
#endif

// Release mode. For speed up the calculation and reduce verbose level.
// Note that RELEASE overwrites DEBUG level.
#ifdef RELEASE
#define _RELEASE_
#define _DEBUGlevel -1
#endif


// Usage of the PEXSI package
#ifdef PEXSI
#define _USE_PEXSI_
#include  "c_pexsi_interface.h"
#endif

#ifdef CPX
#define _COMPLEX_    // complex psi, just for TDDFT now. kind of a hack.
#endif


/***********************************************************************
 *  Data types and constants
 **********************************************************************/

namespace pwdft{

// Basic data types

#define BLAS(name)      name##_
#define LAPACK(name)    name##_
#define SCALAPACK(name) name##_
#define F2C(name)       name##_

typedef    int                   Int;
typedef    double                Real;
typedef    std::complex<double>  Complex; 

// IO
extern  std::ofstream  statusOFS;

// Time Counter
extern Real timePhase_, timeFFT_, timeNonlocal_, timeEkin_, timeAssign_,
       timeMultSpinor_;

extern Int iterPhase_, iterFFT_, iterNonlocal_, iterEkin_, iterAssign_,
       iterMultSpinor_;

extern Real timeGemmT_, timeGemmN_, timeBcast_, timeAllreduce_, timeAlltoallv_,
       timeAlltoallvMap_, timeSpinor_, timeTrsm_, timePotrf_, timeSyevd_,
       timeSygvd_, timeMpirank0_, timeTrtri_, timeCopy_, timeOther_,
       timeEigenSol_;

extern Int iterGemmT_, iterGemmN_, iterBcast_, iterAllreduce_, iterAlltoallv_,
       iterAlltoallvMap_, iterSpinor_, iterTrsm_, iterPotrf_, iterSyevd_,
       iterSygvd_, iterMpirank0_, iterTrtri_, iterCopy_, iterOther_;

// *********************************************************************
// Define constants
// *********************************************************************
// Commonly used
const Int I_ZERO = 0;
const Int I_ONE = 1;
const Real D_ZERO = 0.0;
const Real D_ONE  = 1.0;
const Complex Z_ZERO = Complex(0.0, 0.0);
const Complex Z_ONE  = Complex(1.0, 0.0);
const char UPPER = 'U';
const char LOWER = 'L';

// Physical constants

const Int DIM = 3;                            // Always in 3D
const Real au2K = 315774.67;
const Real au2ev = 27.211385;
const Real au2ang = 0.52917721;
const Real amu2au = 1822.8885;
const Real SPEED_OF_LIGHT = 137.0359895; 
const Real PI = 3.141592653589793;
const Real au2as = 24.188843;
const Real au2fs = 0.024188843;


/// @namespace DensityComponent
/// 
/// @brief Four-component RHO and MAGnetization
namespace DensityComponent{
enum {RHO, MAGX, MAGY, MAGZ};  
}

/// @namespace SpinTwo
///
/// @brief Two-component spin, spin-UP and spin-DowN
namespace SpinTwo{
enum {UP, DN};                 
}

/// @namespace SpinFour
///
/// @brief Four-component spin, LarGe/SMall spin-UP/DowN
namespace SpinFour{
enum {LGUP, LGDN, SMUP, SMDN}; 
}

/// @namespace PseudoComponent
///
/// @brief Pseudopotential component, VALue and Derivatives along the X,
/// Y, Z directions
namespace PseudoComponent{
enum {VAL, DX, DY, DZ};
}

/// @namespace GradThree
///
/// @brief Dots between gradients of spin-up and spin-down densities
namespace GradThree{
enum {UPUP, UPDN, DNDN};
}

/// @brief Default argument for most serialization/deserialization process.
const std::vector<Int> NO_MASK(1);


// Write format control parameters 
const int LENGTH_VAR_NAME = 8;
const int LENGTH_DBL_DATA = 16;
const int LENGTH_INT_DATA = 5;
const int LENGTH_VAR_UNIT = 6;
const int LENGTH_DBL_PREC = 12;
const int LENGTH_FULL_PREC = 16;
const int LENGTH_VAR_DATA = 16;


} // namespace pwdft

/***********************************************************************
 *  Error handling
 **********************************************************************/

namespace pwdft{


void ErrorHandling( const char * msg );

inline void ErrorHandling( const std::string& msg ){ ErrorHandling( msg.c_str() ); }

inline void ErrorHandling( const std::ostringstream& msg ) {ErrorHandling( msg.str().c_str() );}

// We define an output stream that does nothing. This is done so that the 
// root process can be used to print data to a file's ostream while all other 
// processes use a null ostream. 
struct NullStream : std::ostream
{            
  struct NullStreamBuffer : std::streambuf
  {
    Int overflow( Int c ) { return traits_type::not_eof(c); }
  } nullStreamBuffer_;

  NullStream() 
    : std::ios(&nullStreamBuffer_), std::ostream(&nullStreamBuffer_)
    { }
};  

// iA / iC macros.  Not often used.
#define iC(fun)  { int ierr=fun; if(ierr!=0) exit(1); }
#define iA(expr) { if((expr)==0) { std::cerr<<"wrong "<<__LINE__<<" in " <<__FILE__<<std::endl; std::cerr.flush(); exit(1); } }

} // namespace pwdft


#endif // _ENVIRONMENT_DECL_HPP_
