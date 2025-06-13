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
/// @file domain.hpp
/// @brief Computational domain.
/// @date 2012-08-01
#ifndef _DOMAIN_HPP_
#define _DOMAIN_HPP_

#include  "environment.hpp"
#include  "tinyvec_impl.hpp"
#include  "numvec_impl.hpp"
#include  "nummat_impl.hpp"

namespace pwdft{

struct Domain
{
  Point3       length;                          // length
  DblNumMat    supercell;                       // lattice vectors of supercell in each direction
  DblNumMat    recipcell;                       // reciprocal lattice vectors in each direction
  Point3       posStart;                        // starting position
  Index3       numGrid;                         // number of coarse grid points in each direction
  Index3       numGridFine;                     // number of fine grid points in each direction
#ifdef _COMPLEX_
  IntNumVec    numGridSphere;                   // total number of fine grids cutted in a sphere
#else
  Int          numGridSphere; 
#endif
  Int          numGridFock;                     // the number of Fock grids

  IntNumVec    numGridSphereSCF; 
  // For spin calculations
  Int          numSpinComponent;                // number of density and potential components
  bool         SpinOrbitCoupling;               // Whether to include spin-orbit coupling
  Point3       spinaxis;                        // The fixed spin axis ( only used in spin-noncollinear calculations )
#ifdef _COMPLEX_
  // For k-point calculations
  Index3       numKGrid;                        // number of k grid points for Brillouin zone sampling
  Point3       kshift;                          // Shift for k grid points
  std::vector<DblNumVec>  klist;                // Cartesian coordinates of k-points 
  std::vector<DblNumVec>  kgrid;                // Direct coordinates of k-points with respect to recipcell
  DblNumVec    weight;                          // Weights of each k-point for calculating energy and density
  IntNumVec    KpointIdx;                       // Global k index for local k points 

  // Used for hybrid energy band calculation
  // the k points for SCF calculation to obtain density and wavefunctions
  // in uniform and coarse k mesh 
  std::vector<DblNumVec>  klist_scf;            
  std::vector<DblNumVec>  kgrid_scf;            
  IntNumVec    KpointIdx_scf;                      
#endif
  MPI_Comm     comm;                            // MPI Communicator
  MPI_Comm     rowComm;
  MPI_Comm     colComm;
  MPI_Comm     rowComm_kpoint;                  
  MPI_Comm     colComm_kpoint;

  Domain()
  { 
    length        = Point3( 0.0, 0.0, 0.0 );
    supercell.Resize( DIM, DIM ); SetValue( supercell, D_ZERO);
    recipcell.Resize( DIM, DIM ); SetValue( recipcell, D_ZERO);

    posStart      = Point3( 0.0, 0.0, 0.0 );
    numGrid       = Index3( 0, 0, 0 );
    numGridFine   = Index3( 0, 0, 0 );

    numSpinComponent  = I_ONE;
    SpinOrbitCoupling = false;
    spinaxis          = Point3( 0.0, 0.0, 1.0 );
#ifdef _COMPLEX_
    numKGrid             = Index3( 0, 0, 0 );
    kshift            = Point3( 0.0, 0.0, 0.0 );
    klist.resize( DIM );
    kgrid.resize( DIM );
    klist_scf.resize( DIM );
    kgrid_scf.resize( DIM );
#endif
    comm    = MPI_COMM_WORLD; 
    rowComm = MPI_COMM_WORLD;
    colComm = MPI_COMM_WORLD;
    rowComm_kpoint = MPI_COMM_WORLD;
    colComm_kpoint = MPI_COMM_WORLD;
  }

  ~Domain(){}

  Real Volume() const { return std::abs( supercell(0,0) * ( supercell(1,1) * supercell(2,2) -
                                           supercell(2,1) * supercell(1,2) ) -
                                         supercell(0,1) * ( supercell(1,0) * supercell(2,2) - 
                                           supercell(2,0) * supercell(1,2) ) + 
                                         supercell(0,2) * ( supercell(1,0) * supercell(2,1) - 
                                           supercell(2,0) * supercell(1,1) ) ); }
  Int  NumGridTotal() const { return numGrid[0] * numGrid[1] * numGrid[2]; }
  Int  NumGridTotalFine() const { return numGridFine[0] * numGridFine[1] * numGridFine[2]; }
#ifdef _COMPLEX_
  Int  NumKGridTotal() const { return klist[0].Size(); }
  Int  NumKGridSCFTotal() const { return klist_scf[0].Size(); }
#endif
};

} // namespace pwdft

#endif // _DOMAIN_HPP
