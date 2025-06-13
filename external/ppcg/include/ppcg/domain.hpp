/// @file domain.hpp
/// @brief Computational domain.
/// @date 2023-07-01
#ifndef _PPCG_DOMAIN_HPP_
#define _PPCG_DOMAIN_HPP_

#include  "ppcg/environment.hpp"
#include  "ppcg/TinyVec.hpp"
#include  "ppcg/NumVec.hpp"
#include  "ppcg/NumMat.hpp"

namespace PPCG {

struct Domain
{
  DblNumMat    supercell;                       // lattice vectors of supercell in each direction
  DblNumMat    recipcell;                       // reciprocal lattice vectors in each direction
  Index3       numGrid;                         // number of coarse grids points in each direction
  Index3       numGridFine;                     // number of fine grids points in each direction
  MPI_Comm     comm;                            // MPI Communicator
  // FIXME new MPI Communicator for rowComm and colComm
  MPI_Comm     rowComm;
  MPI_Comm     colComm;

  Domain()
  { 
    supercell.Resize( DIM, DIM ); SetValue( supercell, D_ZERO);
    recipcell.Resize( DIM, DIM ); SetValue( recipcell, D_ZERO);

    numGrid       = Index3( 0, 0, 0 );
    numGridFine   = Index3( 0, 0, 0 );

    comm    = MPI_COMM_WORLD; 
    rowComm = MPI_COMM_WORLD;
    colComm = MPI_COMM_WORLD;
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

};

} // namespace PPCG

#endif // _PPCG_DOMAIN_HPP_
