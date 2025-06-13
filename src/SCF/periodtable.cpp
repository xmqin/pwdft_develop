/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

Author: Lin Lin and Weile Jia

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
/// @file periodtable.hpp
/// @brief Periodic table and its entries.
/// @date 2012-08-10
#include "periodtable.hpp"
#include "esdf.hpp"
#include "utility.hpp"
#include "blas.hpp"

namespace  pwdft{

using namespace pwdft::PseudoComponent;
using namespace pwdft::esdf;

// *********************************************************************
// PTEntry
// *********************************************************************

Int serialize(const PTEntry& val, std::ostream& os, const std::vector<Int>& mask)
{
  serialize(val.params,  os, mask);
  serialize(val.samples, os, mask);
  serialize(val.weights, os, mask);
  serialize(val.types,   os, mask);
  serialize(val.soctypes, os, mask); 
  serialize(val.cutoffs, os, mask);
  return 0;
}

Int deserialize(PTEntry& val, std::istream& is, const std::vector<Int>& mask)
{
  deserialize(val.params,  is, mask);
  deserialize(val.samples, is, mask);
  deserialize(val.weights, is, mask);
  deserialize(val.types,   is, mask);
  deserialize(val.soctypes, is, mask);
  deserialize(val.cutoffs, is, mask);
  return 0;
}

Int combine(PTEntry& val, PTEntry& ext)
{
  ErrorHandling( "Combine operation is not implemented" );
  return 0;
}

// *********************************************************************
// PeriodTable
// *********************************************************************

// PTEntry / PTSample is the old format for reading the binary file, but works
// in the new format too, as long as the pseudopotential is associated
// with atoms rather than species.
// 
// There is also room for optimizing the rcps parameter for each species 
// (maybe solving a least squares problems by matlab and store the default value in a table?)
void PeriodTable::Setup( )
{
  std::vector<Int> all(1,1);

  std::istringstream iss;  
  if( esdfParam.periodTableFile.empty() )
  {
    // all the readins are in the samples in the old version, 
    // now in the new version, I should readin something else. 
    // the readin are reading in a sequence of numbers, which
    // is used to construct the ptemap_ struct.
    // Read from UPF file
    MPI_Barrier(MPI_COMM_WORLD);
    int mpirank;  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    int mpisize;  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);

    PTEntry tempEntry;
    if(mpirank==0) {
      for( int i = 0; i < esdfParam.pspFile.size(); i++){
        int atom;
        ReadUPF(esdfParam.pspFile[i], &tempEntry, &atom);
        std::map <Int,PTEntry> :: iterator it = ptemap_.end();
        ptemap_.insert(it, std::pair<Int, PTEntry>(atom, tempEntry));
      }
    }

    // implement the MPI Bcast of the ptemap_, now we are doing all processors readin
    std::stringstream vStream;
    std::stringstream vStreamTemp;
    int vStreamSize;

    serialize( ptemap_, vStream, all);

    if( mpirank == 0)
      vStreamSize = Size( vStream );

    MPI_Bcast( &vStreamSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<char> sstr;
    sstr.resize( vStreamSize );

    if( mpirank == 0)
      vStream.read( &sstr[0], vStreamSize );

    MPI_Bcast( &sstr[0], vStreamSize, MPI_BYTE, 0, MPI_COMM_WORLD );
    vStreamTemp.write( &sstr[0], vStreamSize );
    deserialize( ptemap_, vStreamTemp, all);
  }
  else{
    // Read from periodTableFile, old format.
    // This will become deprecated in the future
    SharedRead( esdfParam.periodTableFile, iss );
    deserialize(ptemap_, iss, all);
  }

  // Setup constant private variable parameters
  if( esdfParam.pseudoType == "HGH" ){
    ptsample_.RADIAL_GRID       = 0;
    ptsample_.PSEUDO_CHARGE     = 1;
    ptsample_.DRV_PSEUDO_CHARGE = 2;
    ptsample_.RHOATOM           = -999;
    ptsample_.DRV_RHOATOM       = -999;
    ptsample_.NONLOCAL          = 3;
  }
  if( esdfParam.pseudoType == "ONCV" ){
    ptsample_.RADIAL_GRID       = 0;
    // If isUseVLocal == false, then ptsample stores the pseudocharge of
    // the local part of the pseudopotential 
    ptsample_.PSEUDO_CHARGE     = 1;
    ptsample_.DRV_PSEUDO_CHARGE = 2;
    // If isUseVLocal == true, then ptsample stores the local part of
    // the pseudopotential 
    ptsample_.VLOCAL            = 1;
    ptsample_.DRV_VLOCAL        = 2;

    ptsample_.RHOATOM           = 3;
    ptsample_.DRV_RHOATOM       = 4;
    ptsample_.NONLOCAL          = 5;
  }

  // Common so far for all pseudopotential
  {
    pttype_.RADIAL            = 9;
    pttype_.PSEUDO_CHARGE     = 99;
    pttype_.RHOATOM           = 999;
    pttype_.VLOCAL            = 9999;
    pttype_.L0                = 0;
    pttype_.L1                = 1;
    pttype_.L2                = 2;
    pttype_.L3                = 3;
    pttype_.SPINORBIT_L1      = -1;
    pttype_.SPINORBIT_L2      = -2;
    pttype_.SPINORBIT_L3      = -3;
  }

  {
    ptparam_.ZNUC   = 0;
    ptparam_.MASS   = 1;
    ptparam_.ZION   = 2;
    // If isUseVLocal == false, the position ESELF stores the self energy
    // If isUseVLocal == true, the same position stores the radius of the
    // Gaussian pseudocharge
    ptparam_.ESELF  = 3;
    ptparam_.RGAUSSIAN = 3;
  }

  // Extra processing of Vlocal data
  if( esdfParam.isUseVLocal == true ){
    // Determine the Gaussian radius where Rgauss = sqrt( 1 / alpha )
    // Gaussian distribution is exp( - (r / Rgauss) ^ 2 )
    Int nelec = 0;
    Real alpha = 2.9;
    Real upperbound = 1.0;
    std::vector<Atom> &atomList = esdfParam.atomList;
    Int numAtom = atomList.size();    

    for( Int a = 0; a < numAtom; a++ ){
      Int atype  = atomList[a].type;
      if( ptemap_.find(atype) == ptemap_.end() ){
        ErrorHandling( "Cannot find the atom type." );
      }
      nelec = nelec + this->Zion(atype);
    }

    while( upperbound > 1e-7 ){
      alpha = alpha - 0.1;
      // ecutDensity = 4 * ecutWavefunction
      upperbound = 2.0 * nelec * nelec * std::sqrt( alpha / PI )
          * erfc( sqrt( esdfParam.ecutWavefunction * 4.0 / 2.0 / alpha ) );
    }

    for(std::map<Int,PTEntry>::iterator mi=ptemap_.begin(); mi!=ptemap_.end(); mi++){
      Int type = (*mi).first;    
      PTEntry& ptcur = (*mi).second;
      DblNumVec& params = ptcur.params;
      // Gaussian radius is changed here
      Real &RGaussian = params(ptparam_.RGAUSSIAN);     
      RGaussian = 1 / std::sqrt( 2.0 * alpha );

      if( esdfParam.isUseRealSpace == true ){
        Real Zion = params(ptparam_.ZION);
        DblNumMat& samples = ptcur.samples;
        Int nspl = samples.m();

        DblNumVec rad(nspl, false, samples.VecData(ptsample_.RADIAL_GRID));
        DblNumVec vlocal(nspl, false, samples.VecData(ptsample_.VLOCAL));
        // Remove the pseudocharge contribution
        for(Int i = 0; i < rad.m(); i++){
          if( rad[i] == 0 ){
            vlocal[i] += Zion / RGaussian * 2.0 / std::sqrt(PI);
          }
          else{
            vlocal[i] += Zion / rad[i] * std::erf(rad[i] / RGaussian);
          }
        }
      }
    }
  }

  // create splines to improve accuracy when the real space method is used
  if( esdfParam.isUseRealSpace == true ){
    for(std::map<Int,PTEntry>::iterator mi=ptemap_.begin(); mi!=ptemap_.end(); mi++){
      Int type = (*mi).first;    
      PTEntry& ptcur = (*mi).second;
      DblNumVec& params = ptcur.params;
      DblNumMat& samples = ptcur.samples;
      std::map< Int, std::vector<DblNumVec> > spltmp;
      for(Int g=1; g<samples.n(); g++) {
        Int nspl = samples.m();
        DblNumVec rad(nspl, true, samples.VecData(0));
        DblNumVec a(nspl, true, samples.VecData(g));
        DblNumVec b(nspl), c(nspl), d(nspl);
        
        spline(nspl, rad.Data(), a.Data(), b.Data(), c.Data(), d.Data());
        std::vector<DblNumVec> aux(5);
        aux[0] = rad;      aux[1] = a;      aux[2] = b;      aux[3] = c;      aux[4] = d;
        spltmp[g] = aux;
      }
      splmap_[type] = spltmp;
    }
  }
}         // -----  end of method PeriodTable::Setup  ----- 

void
PeriodTable::CalculatePseudoCharge    (
    const Atom& atom, 
    const Domain& dm,
    const std::vector<DblNumVec>& gridpos,        
    SparseVec& res )
{
  Int type   = atom.type;
  Point3 pos = atom.pos;
  Point3 Ls  = dm.length;
  Point3 posStart = dm.posStart;
  Index3 Ns  = dm.numGridFine;

  //get entry data and spline data
  PTEntry& ptentry = ptemap_[type];
  std::map< Int, std::vector<DblNumVec> >& spldata = splmap_[type];

  Real Rzero = this->RcutPseudoCharge( type );

  // Initialize
  {
    SparseVec empty;
    res = empty;
  }
  // Compute the minimal distance of the atom to this set of grid points
  // and determine whether to continue 
  std::vector<DblNumVec>  dist(DIM);

  Point3 minDist;
  for( Int d = 0; d < DIM; d++ ){
    dist[d].Resize( gridpos[d].m() );

    minDist[d] = Rzero;
    for( Int i = 0; i < gridpos[d].m(); i++ ){
      dist[d](i) = gridpos[d](i) - pos[d];
      dist[d](i) = dist[d](i) - IRound( dist[d](i) / Ls[d] ) * Ls[d];
      if( std::abs( dist[d](i) ) < minDist[d] )
        minDist[d] = std::abs( dist[d](i) );
    }
  }
  if( std::sqrt( dot(minDist, minDist) ) <= Rzero ){
    // At least one grid point is within Rzero
    Int irad = 0;
    std::vector<Int>  idx;
    std::vector<Real> rad;
    std::vector<Real> xx, yy, zz;
    for(Int k = 0; k < gridpos[2].m(); k++)
      for(Int j = 0; j < gridpos[1].m(); j++)
        for(Int i = 0; i < gridpos[0].m(); i++){
          Real dtmp = std::sqrt( 
              dist[0](i) * dist[0](i) +
              dist[1](j) * dist[1](j) +
              dist[2](k) * dist[2](k) );

          if( dtmp <= Rzero ) {
            idx.push_back(irad);
            rad.push_back(dtmp);
            xx.push_back(dist[0](i));        
            yy.push_back(dist[1](j));        
            zz.push_back(dist[2](k));
          }
          irad++;
        } // for (i)

    Int idxsize = idx.size();
    
    std::vector<DblNumVec>& valspl = spldata[ptsample_.PSEUDO_CHARGE]; 
    std::vector<Real> val(idxsize,0.0);
    seval(&(val[0]), idxsize, &(rad[0]), valspl[0].m(), valspl[0].Data(), 
        valspl[1].Data(), valspl[2].Data(), valspl[3].Data(), valspl[4].Data());
    
    std::vector<DblNumVec>& derspl = spldata[ptsample_.DRV_PSEUDO_CHARGE];
    std::vector<Real> der(idxsize,0.0);

    seval(&(der[0]), idxsize, &(rad[0]), derspl[0].m(), derspl[0].Data(), 
        derspl[1].Data(), derspl[2].Data(), derspl[3].Data(), derspl[4].Data());
    
    IntNumVec iv(idx.size(), true, &(idx[0])); 
    DblNumMat dv( idx.size(), DIM+1 );  // Value and its three derivatives
    
    for(Int g=0; g<idx.size(); g++) {
      dv(g, VAL) = val[g];
      if( rad[g]> MIN_RADIAL ) {
        dv(g, DX) = der[g] * xx[g]/rad[g];
        dv(g, DY) = der[g] * yy[g]/rad[g];
        dv(g, DZ) = der[g] * zz[g]/rad[g];
      } else {
        dv(g, DX) = 0;
        dv(g, DY) = 0;
        dv(g, DZ) = 0;
      }
    }
    res = SparseVec(iv,dv);
  } // if (norm(minDist) <= Rzero )

  return ;
}         // -----  end of method PeriodTable::CalculatePseudoCharge ( Real space method )  ----- 

void PeriodTable::CalculateAtomDensity( 
    const Atom& atom, 
    const Domain& dm, 
    const std::vector<DblNumVec>& gridpos, 
    DblNumVec& atomDensity )
{
  Int type   = atom.type;
  Point3 pos = atom.pos;

  //get entry data and spline data
  PTEntry& ptentry = ptemap_[type];
  std::map< Int, std::vector<DblNumVec> >& spldata = splmap_[type];

  Real Rzero = this->RcutRhoAtom( type );

  SetValue(atomDensity, 0.0);

  DblNumTns dist;
  dist.Resize( dm.numGridFine[0], dm.numGridFine[1], dm.numGridFine[2] );
  SetValue( dist, D_ZERO );
  
  Point3 gridpos_, gridpos_car, shift, difpos_min;

  // Only the nearest neighbor cell is considered when getting the initial density
  const Int Nneighbor = 1; 
  
  Int irad = 0;
  std::vector<Int>  idx;
  std::vector<Real> rad;
  std::vector<Real> xx, yy, zz;

  for( Int k = 0; k < dm.numGridFine[2]; k++ ){
    for( Int j = 0; j < dm.numGridFine[1]; j++ ){
      for( Int i = 0; i < dm.numGridFine[0]; i++ ){

        for( Int d2 = -Nneighbor; d2 <= Nneighbor; d2 ++ ){
          for( Int d1 = -Nneighbor; d1 <= Nneighbor; d1 ++ ){
            for( Int d0 = -Nneighbor; d0 <= Nneighbor; d0 ++ ){
              gridpos_ = Point3( gridpos[0](i), gridpos[1](j), gridpos[2](k) );
              shift = Point3( d0, d1, d2 );
              gridpos_ += shift;
               
              gridpos_car = Point3( 0.0, 0.0, 0.0 );
              for( Int ip = 0; ip < DIM; ip++ ){
                for( Int jp = 0; jp < DIM; jp++ ){
                  gridpos_car[ip] += dm.supercell(jp,ip) * gridpos_[jp];
                }
              }

              gridpos_car -= pos;
              if( d0 + d1 + d2 == -3 ){
                dist(i,j,k) = std::sqrt( dot( gridpos_car, gridpos_car ) );
                difpos_min = gridpos_car;
              }
              else{
                if( std::sqrt( dot( gridpos_car, gridpos_car ) ) < dist(i,j,k) ){
                  dist(i,j,k) = std::sqrt( dot( gridpos_car, gridpos_car ) );
                  difpos_min = gridpos_car;
                  // Store the Cartesian coordinate difference difpos_min
                  // with minimal norm dist(i,j,k)
                }
              }
            }  //  for (d0)
          }  //  for (d1)
        }  //  for (d2)
           
        if( dist(i,j,k) <= Rzero ) {
          idx.push_back(irad);
          rad.push_back(dist(i,j,k));
          xx.push_back(difpos_min[0]);
          yy.push_back(difpos_min[1]);
          zz.push_back(difpos_min[2]);
        }
        irad++;
      }  //  for (i)
    }  //  for (j)
  }  // for (k)

  Int idxsize = idx.size();
  if( idxsize > 0 ){
    std::vector<DblNumVec>& valspl = spldata[ptsample_.RHOATOM]; 
    std::vector<Real> val(idxsize,0.0);
    seval(&(val[0]), idxsize, &(rad[0]), valspl[0].m(), valspl[0].Data(), 
        valspl[1].Data(), valspl[2].Data(), valspl[3].Data(), valspl[4].Data());

    for(Int g=0; g<idx.size(); g++) {
      atomDensity[idx[g]] = val[g];
    }
  }  // if( idxsize > 0 )

  return ;
}         // -----  end of method PeriodTable::CalculateAtomDensity ( Real space method )  ----- 

void
PeriodTable::CalculateAtomDensity(
    const Atom& atom,
    const Domain& dm,
    Fourier& fft,
    DblNumVec& atomDensity )
{
  MPI_Comm mpi_comm = dm.comm;
  MPI_Barrier(mpi_comm);
  Int mpirank;  MPI_Comm_rank(mpi_comm, &mpirank);
  Int mpisize;  MPI_Comm_size(mpi_comm, &mpisize);

  Int type   = atom.type;

  PTEntry& ptentry = ptemap_[type];
  Int upf_mesh_size = ptentry.samples.m();
  Real upf_rab = 0.01;

  DblNumVec upf_r( upf_mesh_size, false,
      ptentry.samples.VecData(ptsample_.RADIAL_GRID) );

  DblNumVec upf_rho( upf_mesh_size, false,
      ptentry.samples.VecData(ptsample_.RHOATOM) );

  IntNumVec& idxDensity = fft.idxFineCutDensity;
  Int ntot = idxDensity.Size();

  atomDensity.Resize( ntot );
  SetValue( atomDensity, 0.0 );

  Int ig0;
  DblNumVec aux( upf_mesh_size );
  DblNumVec aux1( upf_mesh_size );

  if( fft.gkkFine[0] < 1e-8 ){
    for( Int i = 0; i < upf_mesh_size; i++ ){
      aux[i] = upf_rho[i];
    }
    atomDensity[0] = Simpson( upf_mesh_size, aux, upf_rab );
    ig0 = 1;
  }
  else{
    ig0 = 0;
  }

  Real timeSta, timeEnd;
  GetTime( timeSta );
  ntot = ntot - ig0;

  Int ntotLocal = ntot / mpisize;
  if( mpirank < ntot % mpisize ){
    ntotLocal = ntotLocal + 1;
  }

  IntNumVec localSizeGrid( mpisize );
  IntNumVec localDisplsGrid( mpisize );
  SetValue( localSizeGrid, 0 );
  SetValue( localDisplsGrid, 0 );

  MPI_Allgather( &ntotLocal, 1, MPI_INT, localSizeGrid.Data(), 1, MPI_INT, mpi_comm );
  for( Int i = 1; i < mpisize; i++ ){
    localDisplsGrid[i] = localDisplsGrid[i-1] + localSizeGrid[i-1];
  }
  Int istart = ig0 + localDisplsGrid( mpirank );

  DblNumVec atomDensityLocal( ntotLocal );
  DblNumVec gkk( ntotLocal );

  for( Int i = 0; i < ntotLocal; i++ ){
    gkk(i) = fft.gkkFine(idxDensity(i+istart)) * 2.0;
  }

  for( Int i = 0; i < ntotLocal; i++ ){
    Real gx =  std::sqrt( gkk[i] );
    for( Int j = 0; j < upf_mesh_size; j++ ){
      if( upf_r[j] < 1e-8 )
        aux[j] = upf_rho[j];
      else
        aux[j] = upf_rho[j] * std::sin( gx * upf_r[j] ) / ( gx * upf_r[j] );
    }
    atomDensityLocal[i] = Simpson( upf_mesh_size, aux, upf_rab );
  }

  MPI_Allgatherv( atomDensityLocal.Data(), ntotLocal, MPI_DOUBLE, &atomDensity(ig0),
      localSizeGrid.Data(), localDisplsGrid.Data(), MPI_DOUBLE, mpi_comm );
  GetTime( timeEnd );

  return;
}         // ----  end of method PeriodTable::CalculateAtomDensity ( Fourier space method )  ----- 

void
PeriodTable::CalculateVLocal(
    const Atom& atom, 
    const Domain& dm,
    const std::vector<DblNumVec>& gridpos,        
    SparseVec& resVLocalSR, 
    SparseVec& resGaussianPseudoCharge )
{
  Int type   = atom.type;
  //Point3 pos = atom.pos_frac;
  Point3 pos = atom.pos;

  //get entry data and spline data
  PTEntry& ptentry = ptemap_[type];
  std::map< Int, std::vector<DblNumVec> >& spldata = splmap_[type];

  // Use the pseudocharge cutoff for Gaussian compensation charge and
  // short range potential
  Real Rzero = this->RcutPseudoCharge( type );
  Real RGaussian = this->RGaussian( type );
  Real Zion = this->Zion( type );
  // Initialize
  {
    SparseVec empty;
    resVLocalSR = empty;
    resGaussianPseudoCharge = empty;
  }

#if 0
  Real dist;
  Point3 gridpos_, gridpos_car;

  Int irad = 0;
  std::vector<Int>  idx;
  std::vector<Real> rad;
  std::vector<Real> xx, yy, zz;

  const Index3 &nG = dm.numGridFine;
  const DblNumMat &rec = dm.recipcell;
  Index3 nmin = Index3( 0, 0, 0 );
  Index3 nmax = Index3( 0, 0, 0 );
  Int ii, jj, kk;

  for( Int dim = 0; dim < DIM; dim++ ){
    nmin[dim] = round( ( pos[dim] - Rzero / (2 * PI) * sqrt( rec(dim,0)*rec(dim,0) + 
        rec(dim,1)*rec(dim,1) + rec(dim,2)*rec(dim,2) ) ) * nG[dim] );
    nmax[dim] = round( ( pos[dim] + Rzero / (2 * PI) * sqrt( rec(dim,0)*rec(dim,0) +
        rec(dim,1)*rec(dim,1) + rec(dim,2)*rec(dim,2) ) ) * nG[dim] );
    if( nmin[dim] < 0 && nmax[dim] > 0 ){
      
    }
  }
  
  for( Int k = nmin[2]; k <= nmax[2]; k++ ){
    kk = k % nG[2];
    if( kk < 0 ) kk += nG[2];
    if( kk < 0 || kk >= nG[2] ) continue;

    for( Int j = nmin[1]; j <= nmax[1]; j++ ){
      jj = j % nG[1];
      if( jj < 0 ) jj += nG[1];
      if( jj < 0 || jj >= nG[1] ) continue;

      for( Int i = nmin[0]; i <= nmax[0]; i++ ){
        ii = i % nG[0];
        if( ii < 0) ii += nG[0];

        gridpos_ = Point3( i / double( nG[0] ), j / double( nG[1] ), k / double( nG[2] ) );
        gridpos_ = ( gridpos_ - pos );
        // Transform to Cartesian coordinates
        gridpos_car = Point3( 0.0, 0.0, 0.0 );
        for( Int ip = 0; ip < DIM; ip++ ){
          for( Int jp = 0; jp < DIM; jp++ ){
            gridpos_car[ip] += dm.supercell(jp,ip) * gridpos_[jp];
          }
        }

        dist = std::sqrt( dot( gridpos_car, gridpos_car ) );
        irad = ii + jj * nG[0] + kk * nG[0] * nG[1];

        if( dist < Rzero ){
            //std::find( idx.begin(), idx.end(), irad ) == idx.end() ){
          idx.push_back(irad);
          rad.push_back(dist);
          xx.push_back(gridpos_car[0]);
          yy.push_back(gridpos_car[1]);
          zz.push_back(gridpos_car[2]);
        }
      }  // for (i)
    }  // for (j)
  }  // for (k)
#else

  DblNumTns dist;
  dist.Resize( dm.numGridFine[0], dm.numGridFine[1], dm.numGridFine[2] );
  SetValue( dist, D_ZERO );

  Point3 gridpos_, gridpos_car, shift, difpos_min;

  // Only the nearest neighbor cell is considered when getting the initial density
  const Int Nneighbor = 1;

  Int irad = 0;
  std::vector<Int>  idx;
  std::vector<Real> rad;
  std::vector<Real> xx, yy, zz;

  for( Int k = 0; k < dm.numGridFine[2]; k++ ){
    for( Int j = 0; j < dm.numGridFine[1]; j++ ){
      for( Int i = 0; i < dm.numGridFine[0]; i++ ){

        for( Int d2 = -Nneighbor; d2 <= Nneighbor; d2 ++ ){
          for( Int d1 = -Nneighbor; d1 <= Nneighbor; d1 ++ ){
            for( Int d0 = -Nneighbor; d0 <= Nneighbor; d0 ++ ){
              gridpos_ = Point3( gridpos[0](i), gridpos[1](j), gridpos[2](k) );
              shift = Point3( d0, d1, d2 );
              gridpos_ += shift;

              gridpos_car = Point3( 0.0, 0.0, 0.0 );
              for( Int ip = 0; ip < DIM; ip++ ){
                for( Int jp = 0; jp < DIM; jp++ ){
                  gridpos_car[ip] += dm.supercell(jp,ip) * gridpos_[jp];
                }
              }

              gridpos_car -= pos;
              if( d0 + d1 + d2 == -3 ){
                dist(i,j,k) = std::sqrt( dot( gridpos_car, gridpos_car ) );
                difpos_min = gridpos_car;
              }
              else{
                if( std::sqrt( dot( gridpos_car, gridpos_car ) ) < dist(i,j,k) ){
                  dist(i,j,k) = std::sqrt( dot( gridpos_car, gridpos_car ) );
                  difpos_min = gridpos_car;
                  // Store the Cartesian coordinate difference difpos_min
                  // with minimal norm dist(i,j,k)
                  }
              }
            }  //  for (d0)
          }  //  for (d1)
        }  //  for (d2)

        if( dist(i,j,k) <= Rzero ) {
          idx.push_back(irad);
          rad.push_back(dist(i,j,k));
          xx.push_back(difpos_min[0]);
          yy.push_back(difpos_min[1]);
          zz.push_back(difpos_min[2]);
        }
        irad++;
      }  //  for (i)
    }  //  for (j)
  }  // for (k)
#endif
  Int idxsize = idx.size();
  // Short range pseudopotential

  // Interpolate values of grids rad  inside the ball according to
  // vlocal(r) read from pseudopotential
  std::vector<DblNumVec>& valspl = spldata[ptsample_.VLOCAL]; 
  std::vector<Real> val(idxsize,0.0);
  seval(&(val[0]), idxsize, &(rad[0]), valspl[0].m(), valspl[0].Data(), 
      valspl[1].Data(), valspl[2].Data(), valspl[3].Data(), valspl[4].Data());

  std::vector<DblNumVec>& derspl = spldata[ptsample_.DRV_VLOCAL];
  std::vector<Real> der(idxsize,0.0);
  seval(&(der[0]), idxsize, &(rad[0]), derspl[0].m(), derspl[0].Data(), 
      derspl[1].Data(), derspl[2].Data(), derspl[3].Data(), derspl[4].Data());

  IntNumVec iv(idx.size(), true, &(idx[0])); 
  DblNumMat dv( idx.size(), DIM+1 );  // Value and its three derivatives
  for(Int g=0; g<idx.size(); g++) {
    dv(g, VAL) = val[g];
  }
  resVLocalSR = SparseVec(iv,dv);

  // Gaussian pseudocharge
  SetValue(dv, D_ZERO);
  Real fac = Zion / std::pow(std::sqrt(PI) * RGaussian,3);
  for(Int g=0; g<idx.size(); g++) {
    dv(g, VAL) = fac * std::exp(-(rad[g]/RGaussian)*(rad[g]/RGaussian)) ;
  }
  resGaussianPseudoCharge = SparseVec(iv,dv);

  return ;
}         // -----  end of method PeriodTable::CalculateVLocal ( Real space method )  ----- 

void
PeriodTable::CalculateVLocal(
    const Atom& atom,
    const Domain& dm,
    Fourier& fft,
    DblNumVec& resVLocalSR )
{
  MPI_Comm mpi_comm = dm.comm;
  MPI_Barrier(mpi_comm);
  Int mpirank;  MPI_Comm_rank(mpi_comm, &mpirank);
  Int mpisize;  MPI_Comm_size(mpi_comm, &mpisize);
 
  Int type = atom.type;

  PTEntry& ptentry = ptemap_[type];
  Int upf_mesh_size = ptentry.samples.m();
  Real upf_rab = 0.01;

  DblNumVec upf_r( upf_mesh_size, false, 
      ptentry.samples.VecData(ptsample_.RADIAL_GRID) );

  DblNumVec upf_vloc( upf_mesh_size, false, 
      ptentry.samples.VecData(ptsample_.VLOCAL) );

  Real Zion = this->Zion( type );

  IntNumVec& idxDensity = fft.idxFineCutDensity;
  Int ntot = idxDensity.Size();

  resVLocalSR.Resize( ntot );
  SetValue( resVLocalSR, 0.0 );

  Int ig0;
  DblNumVec aux( upf_mesh_size ); 
  DblNumVec aux1( upf_mesh_size ); 

  if( fft.gkkFine[0] < 1e-8 ){
    for( Int i = 0; i < upf_mesh_size; i++ ){
      aux[i] = upf_r[i] * ( upf_r[i] * upf_vloc[i] + Zion );
    }
    resVLocalSR[0] = Simpson( upf_mesh_size, aux, upf_rab );
    ig0 = 1;
  }
  else{
    ig0 = 0;
  }

  for( Int i = 0; i < upf_mesh_size; i++ ){
    aux1[i] = upf_r[i] * upf_vloc[i] + Zion * erf( upf_r[i] );
  }
  
  // Parallel processing for FFT grids
  Real timeSta, timeEnd;
  GetTime( timeSta );
  ntot = ntot - ig0;

  Int ntotLocal = ntot / mpisize;
  if( mpirank < ntot % mpisize ){
    ntotLocal = ntotLocal + 1;
  }  

  IntNumVec localSizeGrid( mpisize );
  IntNumVec localDisplsGrid( mpisize );
  SetValue( localSizeGrid, 0 );
  SetValue( localDisplsGrid, 0 );

  MPI_Allgather( &ntotLocal, 1, MPI_INT, localSizeGrid.Data(), 1, MPI_INT, mpi_comm );
  for( Int i = 1; i < mpisize; i++ ){
    localDisplsGrid[i] = localDisplsGrid[i-1] + localSizeGrid[i-1];
  }
  Int istart = ig0 + localDisplsGrid( mpirank );

  DblNumVec resVLocalSRLocal( ntotLocal );
  DblNumVec gkk( ntotLocal );

  for( Int i = 0; i < ntotLocal; i++ ){
    gkk(i) = fft.gkkFine(idxDensity(i+istart)) * 2.0;
  }

  for( Int i = 0; i < ntotLocal; i++ ){
    Real gx =  std::sqrt( gkk[i] );
    for( Int j = 0; j < upf_mesh_size; j++ ){
      aux[j] = aux1[j] * std::sin( gx * upf_r[j] ) / gx;
    }
    resVLocalSRLocal[i] = Simpson( upf_mesh_size, aux, upf_rab ) -
          Zion * std::exp( - gkk[i] / 4 ) / gkk[i];
  } 

  MPI_Allgatherv( resVLocalSRLocal.Data(), ntotLocal, MPI_DOUBLE, &resVLocalSR(ig0),
      localSizeGrid.Data(), localDisplsGrid.Data(), MPI_DOUBLE, mpi_comm );
  GetTime( timeEnd );
}         // ----  end of method PeriodTable::CalculateVLocal ( Fourier space method )  ----- 

Int PeriodTable::CountNonlocalPP    ( const Atom& atom )
{
  // Count the total number of PP projectors of this atom
  Int type   = atom.type;
  PTEntry& ptentry = ptemap_[type];

  Int numpp = 0;
  for( Int g = ptsample_.NONLOCAL; g < ptentry.samples.n(); g = g + 2 ){
    Int typ = ptentry.types(g);

    if(typ==0)
      numpp=numpp+1;
    if(typ==1)
      numpp=numpp+3;
    if(typ==2)
      numpp=numpp+5;
    if(typ==3)
      numpp=numpp+7;
  }

  return numpp;
}         // -----  end of method PeriodTable::CountNonlocalPP

void PeriodTable::CalculateCoefSOC    ( const Atom& atom, CpxNumTns& coefMat )    
{
  Int type   = atom.type;
  PTEntry& ptentry = ptemap_[type];

  Int numpp = 0;
  for( Int g = ptsample_.NONLOCAL; g < ptentry.samples.n(); g = g + 2 ){
    Int typ = ptentry.types(g);

    if(typ==0)
      numpp=numpp+1;
    if(typ==1)
      numpp=numpp+3;
    if(typ==2)
      numpp=numpp+5;
    if(typ==3)
      numpp=numpp+7;
  }

  // The coefMat needs to be computed if SOC is include
  coefMat.Resize( numpp, numpp, 4 ); SetValue( coefMat, Z_ZERO );

  // In the spin-orbit coupling case we need the unitary matrix u
  // which rotates the real spherical harmonics and yields the complex one.
  Int lmaxx = 4;
  Int lqmax = 2 * lmaxx + 1;
  CpxNumMat rot_ylm( lqmax, lqmax ); SetValue( rot_ylm, Z_ZERO );

  Int l = lmaxx;
  rot_ylm( l, 0 ) = Z_ONE;
  Int m, n;
  for( Int n1 = 2; n1 <= 2*l+1; n1 += 2 ){
    m = n1 / 2;
    n = l + 1 - m;
    rot_ylm( n-1, n1-1 ) = Complex( pow(-1.0, m) / sqrt(2.0), 0.0 );
    rot_ylm( n-1, n1 ) = Complex( 0.0, -pow(-1.0, m) / sqrt(2.0) );
    n = l + 1 + m;
    rot_ylm( n-1, n1-1 ) = Complex( 1.0 / sqrt(2.0), 0.0 );
    rot_ylm( n-1, n1 ) = Complex( 0.0, 1.0 / sqrt(2.0) );
  }

  // First calculate the fcoef coefficients
  CpxNumTns fcoef( numpp, numpp, 4 ); SetValue( fcoef, Z_ZERO );
  Int lenl = ( ptentry.samples.n() - 5 ) / 2;
  IntNumVec pplll( lenl );
  DblNumVec ppjjj( lenl );
  DblNumVec ppdii( lenl );

  Int iidx = 0;
  Int jidx = 0;

  // Read the orbital angular momentum and total angular momentum
  // of projectors
  for(Int g=ptsample_.NONLOCAL; g<ptentry.samples.n(); g=g+2) {
    pplll( iidx ) = ptentry.types(g);
    ppjjj( iidx ) = ptentry.soctypes(g);
    ppdii( iidx ) = ptentry.weights(g);
    iidx ++;
  }
  
  iidx = 0;
  Int ill, jll, ijs;
  Real ijj, jjj;
  Complex coeff;
  for( Int iit = 0; iit < lenl; iit++ ){
    ill = pplll[iit];
    ijj = ppjjj[iit];

    for( Int iitm = -ill; iitm <= ill; iitm++ ){
      iidx++;
      jidx = 0;

      for( Int jit = 0; jit < lenl; jit++ ){
        jll = pplll[jit];
        jjj = ppjjj[jit];
        for( Int jitm = -jll; jitm <= jll; jitm++ ){
          jidx++;
          if( ill == jll && ijj == jjj ){
            ijs = 0;
            for( Int is1 = 0; is1 < 2; is1 ++ ){
              for( Int is2 = 0; is2 < 2; is2++ ){
                coeff = Z_ZERO;
                for( m = -ill-1; m <= ill; m++ ){
                  Int m0 = sph_ind(ill,ijj,m,is1)+lmaxx;
                  Int m1 = sph_ind(jll,jjj,m,is2)+lmaxx;

                  coeff += ( rot_ylm(m0,iitm+ill)*spinor(ill,ijj,m,is1)
                      *conj(rot_ylm(m1,jitm+jll))*spinor(jll,jjj,m,is2) );
                }

                fcoef(iidx-1,jidx-1,ijs++) = coeff;
              } //for(is2)
            } // for(is1)
          } // if( ill == jll && ijj == jjj )
        } // for(jitm)
      } // for(jit)
    } // for(iitm)
  } // for (iit)
  
  // Then calculate the bare coefficients
  iidx = 0;
  for( Int iit = 0; iit < lenl; iit++ ){
    ill = pplll[iit];
    for( Int iitm = -ill; iitm <= ill; iitm++ ){
      iidx++;
      jidx = 0;
      for( Int jit = 0; jit < lenl; jit++ ){
        jll = pplll[jit];
        for( Int jitm = -jll; jitm <= jll; jitm++ ){
          jidx++;
          ijs = 0;
          for( Int is1 = 0; is1 < 2; is1++ ){
            for( Int is2 = 0; is2 < 2; is2++ ){
              if( iit == jit ){
                coefMat(iidx-1,jidx-1,ijs) = ppdii[iit] * fcoef(iidx-1,jidx-1,ijs);
                fcoef(iidx-1,jidx-1,ijs) = Z_ZERO;
              }
              ijs ++;
            } //for(is2)
          } // for(is1)
        } // for(jitm)
      } // for(jit)
    } // for(iitm)
  } // for (iit)  

  return;
}         // -----  end of method PeriodTable::CalculateCoefSOC

void
PeriodTable::CalculateNonlocalPP    (
    const Atom& atom,
    const Domain& dm,
    const std::vector<DblNumVec>& gridpos,
    std::vector<NonlocalPP>&      vnlList,
    std::vector<CpxNumVec>&       vnlPhase)
{
  Point3 Ls       = dm.length;
  Point3 posStart = dm.posStart;
  Index3 Ns       = dm.numGrid;
 
  vnlList.clear();

#ifndef _NO_NONLOCAL_ // Nonlocal potential is used. Debug option
  Int type   = atom.type;
  Point3 pos = atom.pos_frac;

  //get entry data and spline data
  PTEntry& ptentry = ptemap_[type];
  std::map< Int, std::vector<DblNumVec> >& spldata = splmap_[type];

  Real Rzero = this->RcutNonlocal( type );

  // Initialize
  // First count all the pseudopotentials
  Int numpp = 0;
  for(Int g=ptsample_.NONLOCAL; g<ptentry.samples.n(); g=g+2) {
    Int typ = ptentry.types(g);

    if(typ==0)
      numpp=numpp+1;
    if(typ==1)
      numpp=numpp+3;
    if(typ==2)
      numpp=numpp+5;
    if(typ==3)
      numpp=numpp+7;
  }

  {
    vnlList.resize( numpp );

    SparseVec empty;
    for( Int p = 0; p < numpp; p++ ){
      vnlList[p] = NonlocalPP( empty, 0.0 );
    }
  }
  
  Real dist;
  Point3 gridpos_, gridpos_car;

  Int irad = 0;
  std::vector<Int>  idx;
  std::vector<Real> rad;
  std::vector<Real> xx, yy, zz;
  std::vector<Point3> vnlPos;

  const Index3 &nG = dm.numGridFine;
  const DblNumMat &rec = dm.recipcell;
  Index3 nmin = Index3( 0, 0, 0 );
  Index3 nmax = Index3( 0, 0, 0 );
  Int ii, jj, kk;

  for( Int dim = 0; dim < DIM; dim++ ){
    nmin[dim] = round( ( pos[dim] - Rzero / (2 * PI) * sqrt( rec(dim,0)*rec(dim,0) + 
        rec(dim,1)*rec(dim,1) + rec(dim,2)*rec(dim,2) ) ) * nG[dim] );
    nmax[dim] = round( ( pos[dim] + Rzero / (2 * PI) * sqrt( rec(dim,0)*rec(dim,0) +
        rec(dim,1)*rec(dim,1) + rec(dim,2)*rec(dim,2) ) ) * nG[dim] );
  }
  
  for( Int k = nmin[2]; k <= nmax[2]; k++ ){
    kk = k % nG[2];
    if( kk < 0 ) kk += nG[2];
    if( kk < 0 || kk >= nG[2] ) continue;

    for( Int j = nmin[1]; j <= nmax[1]; j++ ){
      jj = j % nG[1];
      if( jj < 0 ) jj += nG[1];
      if( jj < 0 || jj >= nG[1] ) continue;

      for( Int i = nmin[0]; i <= nmax[0]; i++ ){
        ii = i % nG[0];
        if( ii < 0) ii += nG[0];

        gridpos_ = Point3( i / double( nG[0] ), j / double( nG[1] ), k / double( nG[2] ) );
        gridpos_ = ( gridpos_ - pos );
        // Transform to Cartesian coordinates
        gridpos_car = Point3( 0.0, 0.0, 0.0 );
        for( Int ip = 0; ip < DIM; ip++ ){
          for( Int jp = 0; jp < DIM; jp++ ){
            gridpos_car[ip] += dm.supercell(jp,ip) * gridpos_[jp];
          }
        }

        dist = std::sqrt( dot( gridpos_car, gridpos_car ) );
        if( dist < Rzero ){
          irad = ii + jj * nG[0] + kk * nG[0] * nG[1];
          idx.push_back(irad);
          rad.push_back(dist);
          xx.push_back(gridpos_car[0]);
          yy.push_back(gridpos_car[1]);
          zz.push_back(gridpos_car[2]);
          // Store the position difference in vnlPos to calculate k-dependent phase
          vnlPos.push_back(gridpos_car);  
        }
      }  // for(i)
    }  // for (j)
  }  // for(k)

  Int idxsize = idx.size();
#ifdef _COMPLEX_
  // Calculate k-dependent and atom-dependent phase exp( i k \dot r_a )
  const std::vector<DblNumVec> &klist = dm.klist;
  Int nk = dm.NumKGridTotal();
  Point3 rpos( 0.0, 0.0, 0.0 );
  Point3 kpos( 0.0, 0.0, 0.0 );

  vnlPhase.resize( nk );
  for( Int ik = 0; ik < nk; ik++ ){
    kpos = Point3( klist[0][ik], klist[1][ik], klist[2][ik] );

    vnlPhase[ik].Resize( idxsize );
    for( Int i = 0 ; i < idxsize; i++ ){
      rpos = vnlPos[i];
      vnlPhase[ik][i] = std::exp( Complex( 0.0, ( kpos[0]*rpos[0] +
          kpos[1]*rpos[1] + kpos[2]*rpos[2] ) ) );
    }
  }
#endif
  //process non-local pseudopotential one by one
  Int cntpp = 0;
  for(Int g=ptsample_.NONLOCAL; g<ptentry.samples.n(); g=g+2) {
    Real wgt = ptentry.weights(g);
    Int typ = ptentry.types(g);

    std::vector<DblNumVec>& valspl = spldata[g];
    std::vector<Real> val(idxsize,0.0);
    seval(&(val[0]), idxsize, &(rad[0]), valspl[0].m(), valspl[0].Data(), valspl[1].Data(), valspl[2].Data(), valspl[3].Data(), valspl[4].Data());
      
    std::vector<DblNumVec>& derspl = spldata[g+1]; 
    std::vector<Real> der(idxsize,0.0);
    seval(&(der[0]), idxsize, &(rad[0]), derspl[0].m(), derspl[0].Data(), derspl[1].Data(), derspl[2].Data(), derspl[3].Data(), derspl[4].Data());
   
    if(typ==pttype_.L0) {
      Real coef = sqrt(1.0/(4.0*PI)); //spherical harmonics
      IntNumVec iv(idx.size(), true, &(idx[0]));
      DblNumMat dv(idx.size(), DIM + 1); // Value and its three derivatives
      
      for(Int g=0; g<idx.size(); g++) {
        if( rad[g]>MIN_RADIAL ) {
          dv(g,VAL) = coef * val[g];
          dv(g,DX) = coef * der[g] * xx[g]/rad[g];
          dv(g,DY) = coef * der[g] * yy[g]/rad[g];
          dv(g,DZ) = coef * der[g] * zz[g]/rad[g];
        } else {
          dv(g,VAL) = coef * val[g];
          dv(g,DX) = 0;
          dv(g,DY) = 0;
          dv(g,DZ) = 0;
        }
      }
      vnlList[cntpp++] = NonlocalPP( SparseVec(iv,dv), wgt );
    } // if(typ==pttype_.L0);

    if(typ==pttype_.L1) {
      Real coef = sqrt(3.0/(4.0*PI)); //spherical harmonics
      {
        IntNumVec iv(idx.size(), true, &(idx[0]));
        DblNumMat dv(idx.size(), DIM + 1); // Value and its three derivatives
        for(Int g=0; g<idx.size(); g++) {
          if(rad[g]>MIN_RADIAL) {
            dv(g,VAL) = coef*( (zz[g]/rad[g]) * val[g] );
            dv(g,DX) = coef*( (der[g]-val[g]/rad[g])*(zz[g]/rad[g])*(xx[g]/rad[g])                 );
            dv(g,DY) = coef*( (der[g]-val[g]/rad[g])*(zz[g]/rad[g])*(yy[g]/rad[g])                 );
            dv(g,DZ) = coef*( (der[g]-val[g]/rad[g])*(zz[g]/rad[g])*(zz[g]/rad[g]) + val[g]/rad[g] );
          } else {
            dv(g,VAL) = 0;
            dv(g,DX) = 0;
            dv(g,DY) = 0;
            dv(g,DZ) = coef*der[g];
          }
        }
        vnlList[cntpp++] = NonlocalPP( SparseVec(iv,dv), wgt );
      }
      {
        IntNumVec iv(idx.size(), true, &(idx[0]));
        DblNumMat dv(idx.size(), DIM + 1); // Value and its three derivatives
        for(Int g=0; g<idx.size(); g++) {
          if( rad[g]> MIN_RADIAL ) {
            dv(g,VAL) = coef*( (xx[g]/rad[g]) * val[g] );
            dv(g,DX) = coef*( (der[g]-val[g]/rad[g])*(xx[g]/rad[g])*(xx[g]/rad[g]) + val[g]/rad[g] );
            dv(g,DY) = coef*( (der[g]-val[g]/rad[g])*(xx[g]/rad[g])*(yy[g]/rad[g])                 );
            dv(g,DZ) = coef*( (der[g]-val[g]/rad[g])*(xx[g]/rad[g])*(zz[g]/rad[g])                 );
          } else {
            dv(g,VAL) = 0;
            dv(g,DX) = coef*der[g];
            dv(g,DY) = 0;
            dv(g,DZ) = 0;
          }
        }
        vnlList[cntpp++] = NonlocalPP( SparseVec(iv,dv), wgt );
      }
      {
        IntNumVec iv(idx.size(), true, &(idx[0]));
        DblNumMat dv(idx.size(), DIM + 1); // Value and its three derivatives
        for(Int g=0; g<idx.size(); g++) {
          if(rad[g]> MIN_RADIAL) {
            dv(g,VAL) = coef*( (yy[g]/rad[g]) * val[g] );
            dv(g,DX) = coef*( (der[g]-val[g]/rad[g])*(yy[g]/rad[g])*(xx[g]/rad[g])                 );
            dv(g,DY) = coef*( (der[g]-val[g]/rad[g])*(yy[g]/rad[g])*(yy[g]/rad[g]) + val[g]/rad[g] );
            dv(g,DZ) = coef*( (der[g]-val[g]/rad[g])*(yy[g]/rad[g])*(zz[g]/rad[g])                 );
          } else {
            dv(g,VAL) = 0;
            dv(g,DX) = 0;
            dv(g,DY) = coef*der[g];
            dv(g,DZ) = 0;
          }
        }
        vnlList[cntpp++] = NonlocalPP( SparseVec(iv,dv), wgt );
      }
    } // if(typ==pttype_.L1)

    if(typ==pttype_.L2) {
      // d_z2
      {
        Real coef = 1.0/4.0*sqrt(5.0/PI); // Coefficients for spherical harmonics
        IntNumVec iv(idx.size(), true, &(idx[0]));
        DblNumMat dv(idx.size(), DIM + 1); // Value and its three derivatives
        DblNumVec Ylm( DIM + 1 ); //LLIN: Spherical harmonics (0) and its derivatives (1-3)
        for(Int g=0; g<idx.size(); g++) {
          if(rad[g]>MIN_RADIAL) {
            Ylm(0) = coef*(-xx[g]*xx[g]-yy[g]*yy[g]+2.0*zz[g]*zz[g]) / (rad[g]*rad[g]);
            Ylm(1) = coef*(-6.0 * xx[g]*pow(zz[g],2.0) / pow(rad[g],4.0));
            Ylm(2) = coef*(-6.0 * yy[g]*pow(zz[g],2.0) / pow(rad[g],4.0));
            Ylm(3) = coef*( 6.0 * zz[g]*(pow(xx[g],2.0)+pow(yy[g],2.0)) / pow(rad[g], 4.0));

            dv(g,VAL) = Ylm(0) * val[g] ;
            dv(g,DX) = Ylm(0) * der[g] * (xx[g] / rad[g]) + Ylm(1) * val[g];
            dv(g,DY) = Ylm(0) * der[g] * (yy[g] / rad[g]) + Ylm(2) * val[g];
            dv(g,DZ) = Ylm(0) * der[g] * (zz[g] / rad[g]) + Ylm(3) * val[g];
          } else {
            dv(g,VAL) = 0;
            dv(g,DX) = 0;
            dv(g,DY) = 0;
            dv(g,DZ) = 0;
          }
        }
        vnlList[cntpp++] = NonlocalPP( SparseVec(iv,dv), wgt );
      }
      // d_xz
      {
        Real coef = 1.0/2.0*sqrt(15.0/PI);
        IntNumVec iv(idx.size(), true, &(idx[0]));
        DblNumMat dv(idx.size(), DIM + 1); // Value and its three derivatives

        DblNumVec Ylm( DIM + 1 ); //LLIN: Spherical harmonics (0) and its derivatives (1-3)
        for(Int g=0; g<idx.size(); g++) {
          if(rad[g]>MIN_RADIAL) {
            Ylm(0) = coef*(zz[g]*xx[g]) / (rad[g]*rad[g]);
            Ylm(1) = coef*(     zz[g]*(pow(zz[g],2.0)-pow(xx[g],2.0)+pow(yy[g],2.0)) / 
                pow(rad[g],4.0));
            Ylm(2) = coef*(-2.0*xx[g]*yy[g]*zz[g] / pow(rad[g],4.0));
            Ylm(3) = coef*(     xx[g]*(pow(xx[g],2.0)+pow(yy[g],2.0)-pow(zz[g],2.0)) /
                pow(rad[g],4.0));

            dv(g,VAL) = Ylm(0) * val[g] ;
            dv(g,DX) = Ylm(0) * der[g] * (xx[g] / rad[g]) + Ylm(1) * val[g];
            dv(g,DY) = Ylm(0) * der[g] * (yy[g] / rad[g]) + Ylm(2) * val[g];
            dv(g,DZ) = Ylm(0) * der[g] * (zz[g] / rad[g]) + Ylm(3) * val[g];

          } else {
            dv(g,VAL) = 0;
            dv(g,DX) = 0;
            dv(g,DY) = 0;
            dv(g,DZ) = 0;
          }
        }
        vnlList[cntpp++] = NonlocalPP( SparseVec(iv,dv), wgt );
      }
      // d_yz
      {
        Real coef = 1.0/2.0*sqrt(15.0/PI);
        IntNumVec iv(idx.size(), true, &(idx[0]));
        DblNumMat dv(idx.size(), DIM + 1); // Value and its three derivatives

        DblNumVec Ylm( DIM + 1 ); //LLIN: Spherical harmonics (0) and its derivatives (1-3)
        for(Int g=0; g<idx.size(); g++) {
          if(rad[g]>MIN_RADIAL) {
            Ylm(0) = coef*(yy[g]*zz[g]) / (rad[g]*rad[g]);
            Ylm(1) = coef*(-2.0*xx[g]*yy[g]*zz[g] / pow(rad[g],4.0));
            Ylm(2) = coef*(     zz[g]*(pow(zz[g],2.0)+pow(xx[g],2.0)-pow(yy[g],2.0)) / 
                pow(rad[g],4.0));
            Ylm(3) = coef*(     yy[g]*(pow(yy[g],2.0)+pow(xx[g],2.0)-pow(zz[g],2.0)) /
                pow(rad[g],4.0));

            dv(g,VAL) = Ylm(0) * val[g] ;
            dv(g,DX) = Ylm(0) * der[g] * (xx[g] / rad[g]) + Ylm(1) * val[g];
            dv(g,DY) = Ylm(0) * der[g] * (yy[g] / rad[g]) + Ylm(2) * val[g];
            dv(g,DZ) = Ylm(0) * der[g] * (zz[g] / rad[g]) + Ylm(3) * val[g];
          } else {
            dv(g,VAL) = 0;
            dv(g,DX) = 0;
            dv(g,DY) = 0;
            dv(g,DZ) = 0;
          }
        }
        vnlList[cntpp++] = NonlocalPP( SparseVec(iv,dv), wgt );
      }
      // d_x^2-y^2
      {
        Real coef = 1.0/4.0*sqrt(15.0/PI);
        IntNumVec iv(idx.size(), true, &(idx[0]));
        DblNumMat dv(idx.size(), DIM + 1); // Value and its three derivatives
        DblNumVec Ylm( DIM + 1 ); //LLIN: Spherical harmonics (0) and its derivatives (1-3)
        for(Int g=0; g<idx.size(); g++) {
          if(rad[g]>MIN_RADIAL) {
            Ylm(0) = coef*(xx[g]*xx[g]-yy[g]*yy[g]) / (rad[g]*rad[g]);
            Ylm(1) = coef*( 2.0*xx[g]*(2.0*pow(yy[g],2.0)+pow(zz[g],2.0)) / 
                pow(rad[g],4.0));
            Ylm(2) = coef*(-2.0*yy[g]*(2.0*pow(xx[g],2.0)+pow(zz[g],2.0)) /
                pow(rad[g],4.0));
            Ylm(3) = coef*(-2.0*zz[g]*(pow(xx[g],2.0) - pow(yy[g],2.0)) / pow(rad[g],4.0));

            dv(g,VAL) = Ylm(0) * val[g] ;
            dv(g,DX) = Ylm(0) * der[g] * (xx[g] / rad[g]) + Ylm(1) * val[g];
            dv(g,DY) = Ylm(0) * der[g] * (yy[g] / rad[g]) + Ylm(2) * val[g];
            dv(g,DZ) = Ylm(0) * der[g] * (zz[g] / rad[g]) + Ylm(3) * val[g];
          } else {
            dv(g,VAL) = 0;
            dv(g,DX) = 0;
            dv(g,DY) = 0;
            dv(g,DZ) = 0;
          }
        }
        vnlList[cntpp++] = NonlocalPP( SparseVec(iv,dv), wgt );
      }
      // d_xy
      {
        Real coef = 1.0/2.0*sqrt(15.0/PI);
        IntNumVec iv(idx.size(), true, &(idx[0]));
        DblNumMat dv(idx.size(), DIM + 1); // Value and its three derivatives
        DblNumVec Ylm( DIM + 1 ); //LLIN: Spherical harmonics (0) and its derivatives (1-3)
        for(Int g=0; g<idx.size(); g++) {
          if(rad[g]>MIN_RADIAL) {
            Ylm(0) = coef*(xx[g]*yy[g]) / (rad[g]*rad[g]);
            Ylm(1) = coef*(     yy[g]*(pow(yy[g],2.0)-pow(xx[g],2.0)+pow(zz[g],2.0)) / 
                pow(rad[g],4.0));
            Ylm(2) = coef*(     xx[g]*(pow(xx[g],2.0)-pow(yy[g],2.0)+pow(zz[g],2.0)) /
                pow(rad[g],4.0));
            Ylm(3) = coef*(-2.0*xx[g]*yy[g]*zz[g] / pow(rad[g],4.0));

            dv(g,VAL) = Ylm(0) * val[g] ;
            dv(g,DX) = Ylm(0) * der[g] * (xx[g] / rad[g]) + Ylm(1) * val[g];
            dv(g,DY) = Ylm(0) * der[g] * (yy[g] / rad[g]) + Ylm(2) * val[g];
            dv(g,DZ) = Ylm(0) * der[g] * (zz[g] / rad[g]) + Ylm(3) * val[g];
          } else {
            dv(g,VAL) = 0;
            dv(g,DX) = 0;
            dv(g,DY) = 0;
            dv(g,DZ) = 0;
          }
        }
        vnlList[cntpp++] = NonlocalPP( SparseVec(iv,dv), wgt );
      }
    } // if(typ==pttype_.L2)

    // FIXME: The derivative at r=0 for the f orbital MAY NOT BE CORRECT.
    // LLIN: 10/21/2013
    if(typ==pttype_.L3) {
      // f_z3
      {
        Real coef = 1.0/4.0*sqrt(7.0/PI); // Coefficients for spherical harmonics
        IntNumVec iv(idx.size(), true, &(idx[0]));
        DblNumMat dv(idx.size(), DIM + 1); // Value and its three derivatives
        DblNumVec Ylm( DIM + 1 ); //LLIN: Spherical harmonics (0) and its derivatives (1-3)
        for(Int g=0; g<idx.size(); g++) {
          if(rad[g]>MIN_RADIAL) {
            Real x2 = xx[g]*xx[g];
            Real y2 = yy[g]*yy[g];
            Real z2 = zz[g]*zz[g];
            Real r3 = pow(rad[g], 3.0);
            Real r5 = pow(rad[g], 5.0);

            Ylm(0) = coef*zz[g]*(-3.*x2 - 3.*y2 + 2.*z2) / r3;
            Ylm(1) = coef*3.*xx[g]*zz[g]*(x2 + y2 - 4.*z2) / r5;
            Ylm(2) = coef*3.*yy[g]*zz[g]*(x2 + y2 - 4.*z2) / r5;
            Ylm(3) = -coef*3.*(x2 + y2)*(x2 + y2 - 4.*z2) / r5;

            dv(g,VAL) = Ylm(0) * val[g] ;
            dv(g,DX) = Ylm(0) * der[g] * (xx[g] / rad[g]) + Ylm(1) * val[g];
            dv(g,DY) = Ylm(0) * der[g] * (yy[g] / rad[g]) + Ylm(2) * val[g];
            dv(g,DZ) = Ylm(0) * der[g] * (zz[g] / rad[g]) + Ylm(3) * val[g];
          } else {
            dv(g,VAL) = 0;
            dv(g,DX) = 0;
            dv(g,DY) = 0;
            dv(g,DZ) = 0;
          }
        }
        vnlList[cntpp++] = NonlocalPP( SparseVec(iv,dv), wgt );
      }
      // f_xzz
      {
        Real coef = 1.0/4.0*sqrt(21./(2.*PI)); // Coefficients for spherical harmonics
        IntNumVec iv(idx.size(), true, &(idx[0]));
        DblNumMat dv(idx.size(), DIM + 1); // Value and its three derivatives
        DblNumVec Ylm( DIM + 1 ); //LLIN: Spherical harmonics (0) and its derivatives (1-3)
        for(Int g=0; g<idx.size(); g++) {
          if(rad[g]>MIN_RADIAL) {
            Real x2 = xx[g]*xx[g];
            Real y2 = yy[g]*yy[g];
            Real z2 = zz[g]*zz[g];
            Real r3 = pow(rad[g], 3.0);
            Real r5 = pow(rad[g], 5.0);

            Ylm(0) = coef*xx[g]*(-x2 - y2 + 4.*z2) / r3;
            Ylm(1) = -coef*(y2*y2 - 3.*y2*z2 - 4.*z2*z2 + x2*(y2+11.*z2)) / r5;
            Ylm(2) = coef*xx[g]*yy[g]*(x2 + y2 - 14.*z2) / r5;
            Ylm(3) = coef*xx[g]*zz[g]*(11.*x2 + 11.*y2 - 4.*z2) / r5;

            dv(g,VAL) = Ylm(0) * val[g] ;
            dv(g,DX) = Ylm(0) * der[g] * (xx[g] / rad[g]) + Ylm(1) * val[g];
            dv(g,DY) = Ylm(0) * der[g] * (yy[g] / rad[g]) + Ylm(2) * val[g];
            dv(g,DZ) = Ylm(0) * der[g] * (zz[g] / rad[g]) + Ylm(3) * val[g];
          } else {
            dv(g,VAL) = 0;
            dv(g,DX) = 0;
            dv(g,DY) = 0;
            dv(g,DZ) = 0;
          }
        }
        vnlList[cntpp++] = NonlocalPP( SparseVec(iv,dv), wgt );
      }
      // f_yzz
      {
        Real coef = 1.0/4.0*sqrt(21./(2.*PI)); // Coefficients for spherical harmonics
        IntNumVec iv(idx.size(), true, &(idx[0]));
        DblNumMat dv(idx.size(), DIM + 1); // Value and its three derivatives
        DblNumVec Ylm( DIM + 1 ); //LLIN: Spherical harmonics (0) and its derivatives (1-3)
        for(Int g=0; g<idx.size(); g++) {
          if(rad[g]>MIN_RADIAL) {
            Real x2 = xx[g]*xx[g];
            Real y2 = yy[g]*yy[g];
            Real z2 = zz[g]*zz[g];
            Real r3 = pow(rad[g], 3.0);
            Real r5 = pow(rad[g], 5.0);

            Ylm(0) = coef*yy[g]*(-x2 - y2 + 4.*z2) / r3;
            Ylm(1) = coef*xx[g]*yy[g]*(x2 + y2 - 14.*z2) / r5;
            Ylm(2) = -coef*(x2*x2 + 11.*y2*z2- 4.*z2*z2 + x2*(y2-3.*z2)) / r5;
            Ylm(3) = coef*yy[g]*zz[g]*(11.*x2 + 11.*y2 - 4.*z2) / r5;

            dv(g,VAL) = Ylm(0) * val[g] ;
            dv(g,DX) = Ylm(0) * der[g] * (xx[g] / rad[g]) + Ylm(1) * val[g];
            dv(g,DY) = Ylm(0) * der[g] * (yy[g] / rad[g]) + Ylm(2) * val[g];
            dv(g,DZ) = Ylm(0) * der[g] * (zz[g] / rad[g]) + Ylm(3) * val[g];
          } else {
            dv(g,VAL) = 0;
            dv(g,DX) = 0;
            dv(g,DY) = 0;
            dv(g,DZ) = 0;
          }
        }
        vnlList[cntpp++] = NonlocalPP( SparseVec(iv,dv), wgt );
      }
      // f_z(xx-yy)
      {
        Real coef = 1.0/4.0*sqrt(105./(PI)); // Coefficients for spherical harmonics
        IntNumVec iv(idx.size(), true, &(idx[0]));
        DblNumMat dv(idx.size(), DIM + 1); // Value and its three derivatives
        DblNumVec Ylm( DIM + 1 ); //LLIN: Spherical harmonics (0) and its derivatives (1-3)
        for(Int g=0; g<idx.size(); g++) {
          if(rad[g]>MIN_RADIAL) {
            Real x2 = xx[g]*xx[g];
            Real y2 = yy[g]*yy[g];
            Real z2 = zz[g]*zz[g];
            Real r3 = pow(rad[g], 3.0);
            Real r5 = pow(rad[g], 5.0);

            Ylm(0) = coef*zz[g]*(x2 - y2) / r3;
            Ylm(1) = coef*xx[g]*zz[g]*(-x2 + 5.*y2 + 2.*z2) / r5;
            Ylm(2) = coef*yy[g]*zz[g]*(-5.*x2 + y2 - 2.*z2) / r5;
            Ylm(3) = coef*(x2 - y2)*(x2 + y2 - 2.*z2) / r5;

            dv(g,VAL) = Ylm(0) * val[g] ;
            dv(g,DX) = Ylm(0) * der[g] * (xx[g] / rad[g]) + Ylm(1) * val[g];
            dv(g,DY) = Ylm(0) * der[g] * (yy[g] / rad[g]) + Ylm(2) * val[g];
            dv(g,DZ) = Ylm(0) * der[g] * (zz[g] / rad[g]) + Ylm(3) * val[g];
          } else {
            dv(g,VAL) = 0;
            dv(g,DX) = 0;
            dv(g,DY) = 0;
            dv(g,DZ) = 0;
          }
        }
        vnlList[cntpp++] = NonlocalPP( SparseVec(iv,dv), wgt );
      }
      // f_xyz
      {
        Real coef = 1.0/2.0*sqrt(105./(PI)); // Coefficients for spherical harmonics
        IntNumVec iv(idx.size(), true, &(idx[0]));
        DblNumMat dv(idx.size(), DIM + 1); // Value and its three derivatives
        DblNumVec Ylm( DIM + 1 ); //LLIN: Spherical harmonics (0) and its derivatives (1-3)
        for(Int g=0; g<idx.size(); g++) {
          if(rad[g]>MIN_RADIAL) {
            Real x2 = xx[g]*xx[g];
            Real y2 = yy[g]*yy[g];
            Real z2 = zz[g]*zz[g];
            Real r3 = pow(rad[g], 3.0);
            Real r5 = pow(rad[g], 5.0);

            Ylm(0) = coef*xx[g]*yy[g]*zz[g] / r3;
            Ylm(1) = coef*yy[g]*zz[g]*(-2.*x2 + y2 + z2) / r5;
            Ylm(2) = coef*xx[g]*zz[g]*(x2 - 2.*y2 + z2) / r5;
            Ylm(3) = coef*xx[g]*yy[g]*(x2 + y2 - 2.*z2) / r5;

            dv(g,VAL) = Ylm(0) * val[g] ;
            dv(g,DX) = Ylm(0) * der[g] * (xx[g] / rad[g]) + Ylm(1) * val[g];
            dv(g,DY) = Ylm(0) * der[g] * (yy[g] / rad[g]) + Ylm(2) * val[g];
            dv(g,DZ) = Ylm(0) * der[g] * (zz[g] / rad[g]) + Ylm(3) * val[g];
          } else {
            dv(g,VAL) = 0;
            dv(g,DX) = 0;
            dv(g,DY) = 0;
            dv(g,DZ) = 0;
          }
        }
        vnlList[cntpp++] = NonlocalPP( SparseVec(iv,dv), wgt );
      }
      // f_x(xx-3yy)
      {
        Real coef = 1.0/4.0*sqrt(35./(2.*PI)); // Coefficients for spherical harmonics
        IntNumVec iv(idx.size(), true, &(idx[0]));
        DblNumMat dv(idx.size(), DIM + 1); // Value and its three derivatives
        DblNumVec Ylm( DIM + 1 ); //LLIN: Spherical harmonics (0) and its derivatives (1-3)
        for(Int g=0; g<idx.size(); g++) {
          if(rad[g]>MIN_RADIAL) {
            Real x2 = xx[g]*xx[g];
            Real y2 = yy[g]*yy[g];
            Real z2 = zz[g]*zz[g];
            Real r3 = pow(rad[g], 3.0);
            Real r5 = pow(rad[g], 5.0);

            Ylm(0) = coef*xx[g]*(x2 - 3.*y2) / r3;
            Ylm(1) = coef*3.*(-y2*(y2+z2) + x2*(3.*y2+z2)) / r5;
            Ylm(2) = coef*3.*xx[g]*yy[g]*(-3.*x2 + y2 - 2.*z2) / r5;
            Ylm(3) = -coef*3.*xx[g]*zz[g]*(x2 - 3.*y2) / r5;

            dv(g,VAL) = Ylm(0) * val[g] ;
            dv(g,DX) = Ylm(0) * der[g] * (xx[g] / rad[g]) + Ylm(1) * val[g];
            dv(g,DY) = Ylm(0) * der[g] * (yy[g] / rad[g]) + Ylm(2) * val[g];
            dv(g,DZ) = Ylm(0) * der[g] * (zz[g] / rad[g]) + Ylm(3) * val[g];
          } else {
            dv(g,VAL) = 0;
            dv(g,DX) = 0;
            dv(g,DY) = 0;
            dv(g,DZ) = 0;
          }
        }
        vnlList[cntpp++] = NonlocalPP( SparseVec(iv,dv), wgt );
      }
      // f_y(3xx-yy)
      {
        Real coef = 1.0/4.0*sqrt(35./(2.*PI)); // Coefficients for spherical harmonics
        IntNumVec iv(idx.size(), true, &(idx[0]));
        DblNumMat dv(idx.size(), DIM + 1); // Value and its three derivatives
        DblNumVec Ylm( DIM + 1 ); //LLIN: Spherical harmonics (0) and its derivatives (1-3)
        for(Int g=0; g<idx.size(); g++) {
          if(rad[g]>MIN_RADIAL) {
            Real x2 = xx[g]*xx[g];
            Real y2 = yy[g]*yy[g];
            Real z2 = zz[g]*zz[g];
            Real r3 = pow(rad[g], 3.0);
            Real r5 = pow(rad[g], 5.0);

            Ylm(0) = coef*yy[g]*(3.*x2 - y2) / r3;
            Ylm(1) = -coef*3.*xx[g]*yy[g]*(x2 - 3.*y2 - 2.*z2) / r5;
            Ylm(2) = coef*3.*(x2*x2 - y2*z2 + x2*(-3.*y2+z2)) / r5;
            Ylm(3) = coef*3.*yy[g]*zz[g]*(-3.*x2 + y2) / r5;

            dv(g,VAL) = Ylm(0) * val[g] ;
            dv(g,DX) = Ylm(0) * der[g] * (xx[g] / rad[g]) + Ylm(1) * val[g];
            dv(g,DY) = Ylm(0) * der[g] * (yy[g] / rad[g]) + Ylm(2) * val[g];
            dv(g,DZ) = Ylm(0) * der[g] * (zz[g] / rad[g]) + Ylm(3) * val[g];
          } else {
            dv(g,VAL) = 0;
            dv(g,DX) = 0;
            dv(g,DY) = 0;
            dv(g,DZ) = 0;
          }
        }
        vnlList[cntpp++] = NonlocalPP( SparseVec(iv,dv), wgt );
      }    
    } // if(typ==pttype_.L3)
  } // for (g)
     
  // Check the number of pseudopotentials
  if( cntpp != numpp ){
    ErrorHandling("cntpp != numpp.  Seriously wrong with nonlocal pseudopotentials.");
  }
#endif // #ifndef _NO_NONLOCAL_

  return ;
}         // -----  end of method PeriodTable::CalculateNonlocalPP ( Real space method )  ----- 

void
PeriodTable::CalculateNonlocalPP    (
    const Atom& atom,
    const Domain& dm,
    Fourier& fft,
    IntNumVec& grididx,
    Int        k,
    Point3&    kpoint,
    DblNumVec& weight,
    CpxNumMat& vkb )
{
  Int type   = atom.type;
  PTEntry& ptentry = ptemap_[type];

  Real dq = 0.01;
  Int nqxq = round( std::sqrt( 2 * esdfParam.ecutWavefunction ) / dq + 4.0 );

  // Radius grids for integral
  Int upf_mesh_size = ptentry.samples.m();
  Real Rzero = this->RcutNonlocal( type ); 
  Real upf_rab = 0.01;

  DblNumVec upf_r( upf_mesh_size, false, ptentry.samples.VecData(ptsample_.RADIAL_GRID) );
  Int nr = 0;
  for( Int i = 0; i < upf_r.m(); i++ ){
    if( upf_r[i] <= Rzero ) nr++;
  }

  // Initialize
  // First count all the pseudopotentials
  Int numpp = 0;
  for( Int g = ptsample_.NONLOCAL; g < ptentry.samples.n(); g = g + 2 ){
    Int typ = ptentry.types(g);

    if(typ==0)
      numpp=numpp+1;
    if(typ==1)
      numpp=numpp+3;
    if(typ==2)
      numpp=numpp+5;
    if(typ==3)
      numpp=numpp+7;
  }
#ifdef _COMPLEX_  
  IntNumVec& idxc = fft.idxCoarseCut[k];
#else
  IntNumVec& idxc = fft.idxCoarseCut;
#endif
  Int ntotG = grididx.Size();
  DblNumVec sqrtgkk( ntotG );
  DblNumVec gkx( ntotG );
  DblNumVec gky( ntotG );
  DblNumVec gkz( ntotG );
  DblNumVec vq( ntotG );

  Real px, ux, vx, wx, gk;
  Int it0, it1, it2, it3;

  vkb.Resize(ntotG, numpp);
  SetValue( vkb, Z_ZERO );
  weight.Resize(numpp);

  Int idxW = 0;
  for( Int g = ptsample_.NONLOCAL; g < ptentry.samples.n(); g = g + 2 ){
    Real wgt = ptentry.weights(g);
    Int typ = ptentry.types(g);
    for( Int i = 0; i < 2*typ + 1; i++ ){
      weight(idxW++) = wgt * std::pow(4*PI, 2) / fft.domain.Volume();
    }
  }

  Int cntpp = 0;
  for(Int g=ptsample_.NONLOCAL; g<ptentry.samples.n(); g=g+2) {
    Int typ = ptentry.types(g);

    // Step 1. Converts non-local pseudo potential at irregular radial grid in
    // real-space to uniform radial grid in G-space     
    DblNumVec tab( nqxq );
    SetValue( tab, 0.0 );
    // Bessel function with order typ
    DblNumMat besr( nr, nqxq ), rmat( nr, nqxq ); 
    for( Int j = 0; j < nqxq; j++ ){
      for( Int i = 0; i < nr; i++ ){
        rmat(i, j) = upf_r(i) * dq * j;
      }
    }
    bessel( typ, nr*nqxq, rmat.Data(), besr.Data() );

    // Beta function in pseudopotential
    DblNumVec beta( nr, false, ptentry.samples.VecData(g) );

    DblNumVec aux( nr );
    for( Int j = 0; j < nqxq; j++ ){
      for( Int i = 0; i < nr; i++ ){
        aux(i) = beta(i) * besr(i, j) * upf_r(i);
      }        
      tab(j) = Simpson( nr, aux , upf_rab );
    }

    for( Int ig = 0; ig < ntotG; ig++ ){
      Int idx = idxc(grididx(ig));
#ifdef _COMPLEX_
      gkx(ig) = fft.ik[0][idx].imag() + kpoint[0];
      gky(ig) = fft.ik[1][idx].imag() + kpoint[1];
      gkz(ig) = fft.ik[2][idx].imag() + kpoint[2];
#else
      gkx(ig) = fft.ikR2C[0][idx].imag() + kpoint[0];
      gky(ig) = fft.ikR2C[1][idx].imag() + kpoint[1];
      gkz(ig) = fft.ikR2C[2][idx].imag() + kpoint[2];
#endif
      gk = std::sqrt(gkx(ig)*gkx(ig) + gky(ig)*gky(ig) + gkz(ig)* gkz(ig));
      sqrtgkk(ig) = gk;

      px = gk / dq - int(gk / dq);
      ux = 1.0 - px;
      vx = 2.0 - px;
      wx = 3.0 - px;
      it0 = int(gk / dq);
      it1 = it0 + 1;
      it2 = it0 + 2;
      it3 = it0 + 3;
      vq(ig) = tab(it0) * ux * vx * wx / 6.0
          + tab(it1) * px * vx * wx / 2.0
          - tab(it2) * px * ux * wx / 2.0
          + tab(it3) * px * ux * vx / 6.0;
    } // for (ig)

    // Step 2. Multiply the spherical harmonics
    Complex coef;
    if(typ==pttype_.L0) {
      // s
      coef = sqrt(1.0/(4.0*PI)); 
      for( Int ig = 0; ig < ntotG; ig++ ){
        vkb(ig, cntpp) = coef * vq(ig);
      }      
      cntpp = cntpp + 1;
    } // if(typ==pttype_.L0);
 
    if(typ==pttype_.L1) {
      // px, py, pz
      coef = Complex(0.0, -1.0) * sqrt(3.0/(4.0*PI)); 
      for( Int ig = 0; ig < ntotG; ig++ ){
        if( sqrtgkk(ig) > MIN_RADIAL ){
          vkb(ig, cntpp) = coef * gkz(ig) / sqrtgkk(ig) * vq(ig);
          vkb(ig, cntpp+1) = coef * gkx(ig)  / sqrtgkk(ig) * vq(ig);
          vkb(ig, cntpp+2) = coef * gky(ig)  / sqrtgkk(ig) * vq(ig);
        }
      }
      cntpp = cntpp + 3;
    } // if(typ==pttype_.L1)

    if(typ==pttype_.L2) {
      // d_z2
      coef = -1.0/4.0*sqrt(5.0/PI);
      for( Int ig = 0; ig < ntotG; ig++ ){
        if( sqrtgkk(ig) > MIN_RADIAL ){
          vkb(ig, cntpp) = coef * ( 2 * gkz(ig) * gkz(ig) - gkx(ig) * gkx(ig)
              - gky(ig) * gky(ig) ) / ( sqrtgkk(ig) * sqrtgkk(ig) ) * vq(ig);
        }
      }
      // d_yz, d_xz, d_xz
      coef = -1.0/2.0*sqrt(15.0/PI);
      for( Int ig = 0; ig < ntotG; ig++ ){
        if( sqrtgkk(ig) > MIN_RADIAL ){
          vkb(ig, cntpp+1) = coef * gkx(ig) * gkz(ig) / ( sqrtgkk(ig) * sqrtgkk(ig) ) * vq(ig);
          vkb(ig, cntpp+2) = coef * gky(ig) * gkz(ig) / ( sqrtgkk(ig) * sqrtgkk(ig) ) * vq(ig);
          vkb(ig, cntpp+4) = coef * gkx(ig) * gky(ig) / ( sqrtgkk(ig) * sqrtgkk(ig) ) * vq(ig);
        }
      }
      // d_x^2-y^2
      coef = -1.0/4.0*sqrt(15.0/PI);
      for( Int ig = 0; ig < ntotG; ig++ ){
        if( sqrtgkk(ig) > MIN_RADIAL ){
          vkb(ig, cntpp+3) = coef * ( gkx(ig) * gkx(ig) - gky(ig) * gky(ig) ) 
              / ( sqrtgkk(ig) * sqrtgkk(ig) ) * vq(ig);
        }
      }
      cntpp = cntpp + 5;
    } // if(typ==pttype_.L2)

    if(typ==pttype_.L3) {
      // f_z3
      coef = Complex(0.0, 1.0)*1.0/4.0*sqrt(7.0/PI);
      for( Int ig = 0; ig < ntotG; ig++ ){
        if( sqrtgkk(ig) > MIN_RADIAL ){
          vkb(ig, cntpp) = coef * gkz(ig) * ( - 3.0 * gkx(ig) * gkx(ig) - 
              3.0 * gky(ig) * gky(ig) + 2.0 * gkz(ig) * gkz(ig) ) / 
              std::pow( sqrtgkk(ig), 3 ) * vq(ig);
        }
      }
      // f_y(3xx-yy)
      coef = Complex(0.0, 1.0)*1.0/4.0*sqrt(35.0/(2.0*PI));
      for( Int ig = 0; ig < ntotG; ig++ ){
        if( sqrtgkk(ig) > MIN_RADIAL ){
          vkb(ig, cntpp+6) = coef * gky(ig) * ( 3.0 * gkx(ig) * gkx(ig) -
              gky(ig) * gky(ig) ) / std::pow( sqrtgkk(ig), 3 ) * vq(ig);
        }
      }
      // f_x(xx-3yy)
      coef = Complex(0.0, 1.0)*1.0/4.0*sqrt(35.0/(2.0*PI));
      for( Int ig = 0; ig < ntotG; ig++ ){
        if( sqrtgkk(ig) > MIN_RADIAL ){
          vkb(ig, cntpp+5) = coef * gkx(ig) * ( gkx(ig) * gkx(ig) - 3.0 *
              gky(ig) * gky(ig) ) / std::pow( sqrtgkk(ig), 3 ) * vq(ig);
        }
      }
      // f_z(xx-yy)
      coef = Complex(0.0, 1.0)*1.0/4.0*sqrt(105.0/(PI));
      for( Int ig = 0; ig < ntotG; ig++ ){
        if( sqrtgkk(ig) > MIN_RADIAL ){
          vkb(ig, cntpp+3) = coef * gkz(ig) * ( gkx(ig) * gkx(ig) -
              gky(ig) * gky(ig) ) / std::pow( sqrtgkk(ig), 3 ) * vq(ig);
        }
      }
      // f_xyz
      coef = Complex(0.0, 1.0)*1.0/2.0*sqrt(105.0/(PI));
      for( Int ig = 0; ig < ntotG; ig++ ){
        if( sqrtgkk(ig) > MIN_RADIAL ){
          vkb(ig, cntpp+4) = coef * gkx(ig) * gky(ig) * gkz(ig)
              / std::pow( sqrtgkk(ig), 3 ) * vq(ig);
        }
      }
      // f_yzz
      coef = Complex(0.0, 1.0)*1.0/4.0*sqrt(21.0/(2.0*PI));
      for( Int ig = 0; ig < ntotG; ig++ ){
        if( sqrtgkk(ig) > MIN_RADIAL ){
          vkb(ig, cntpp+2) = coef * gky(ig) * ( - gkx(ig) * gkx(ig) - 
              gky(ig) * gky(ig) + 4.0 * gkz(ig) * gkz(ig) ) 
              / std::pow( sqrtgkk(ig), 3 ) * vq(ig);
        }
      }
      // f_xzz
      coef = Complex(0.0, 1.0)*1.0/4.0*sqrt(21.0/(2.0*PI));
      for( Int ig = 0; ig < ntotG; ig++ ){
        if( sqrtgkk(ig) > MIN_RADIAL ){
          vkb(ig, cntpp+1) = coef * gkx(ig) * ( - gkx(ig) * gkx(ig) -
              gky(ig) * gky(ig) + 4.0 * gkz(ig) * gkz(ig) ) 
              / std::pow( sqrtgkk(ig), 3 ) * vq(ig);
        }
      }
      cntpp = cntpp + 7;
    } // if(typ==pttype_.L3)
  } // for (g)

  // Check the number of pseudopotentials
  if( cntpp != numpp ){
    ErrorHandling("cntpp != numpp.  Seriously wrong with nonlocal pseudopotentials.");
  }
  
  return;
}         // -----  end of method PeriodTable::CalculateNonlocalPP ( Fourier space method ) ----- 

Real PeriodTable::SelfIonInteraction(Int type) 
{
  Real eself;
  if( esdfParam.isUseVLocal == false ){
    eself = ptemap_[type].params(ptparam_.ESELF);
  }
  else{
    Real Rzero = this->RcutPseudoCharge( type );
    Real RGaussian = this->RGaussian( type );
    Real Zion = this->Zion( type );
    eself = Zion * Zion / ( std::sqrt(2.0 * PI) * RGaussian );
  }
  
  return eself;
}         // -----  end of method PeriodTable::SelfIonInteraction ---- 

// Serialization / Deserialization
Int serialize(const Atom& val, std::ostream& os, const std::vector<Int>& mask)
{
  serialize(val.type, os, mask);
  serialize(val.pos,  os, mask);
  serialize(val.vel,  os, mask);
  serialize(val.force,  os, mask);
  return 0;
}

Int deserialize(Atom& val, std::istream& is, const std::vector<Int>& mask)
{
  deserialize(val.type, is, mask);
  deserialize(val.pos,  is, mask);
  deserialize(val.vel,  is, mask);
  deserialize(val.force,  is, mask);
  return 0;
}

Real MaxForce( const std::vector<Atom>& atomList ){
  Int numAtom = atomList.size();
  Real maxForce = 0.0;
  for( Int i = 0; i < numAtom; i++ ){
    Real forceMag = atomList[i].force.l2();
    maxForce = ( maxForce < forceMag ) ? forceMag : maxForce;
  }
  return maxForce;
}

// *********************************************************************
// The following comes from the UPF2QSO subroutine.
// Do not 
// *********************************************************************
struct Element
{
  int z;
  std::string symbol;
  std::string config;
  double mass;
  Element(int zz, std::string s, std::string c, double m) : z(zz), symbol(s), config(c),
    mass(m) {}
};

class PeriodicTable
{
  private:

  std::vector<Element> ptable;
  std::map<std::string,int> zmap;

  public:

  PeriodicTable(void);
  int z(std::string symbol) const;
  std::string symbol(int zval) const;
  std::string configuration(int zval) const;
  std::string configuration(std::string symbol) const;
  double mass(int zval) const;
  double mass(std::string symbol) const;
  int size(void) const;
};

/// the following code are merged from the UPF2QSO package
int PeriodicTable::z(std::string symbol) const
{
  std::map<std::string,int>::const_iterator i = zmap.find(symbol);
  assert( i != zmap.end() );
  return (*i).second;
}

////////////////////////////////////////////////////////////////////////////////
std::string PeriodicTable::symbol(int z) const
{
  assert(z>0 && z<=ptable.size());
  return ptable[z-1].symbol;
}

////////////////////////////////////////////////////////////////////////////////
std::string PeriodicTable::configuration(int z) const
{
  assert(z>0 && z<=ptable.size());
  return ptable[z-1].config;
}

////////////////////////////////////////////////////////////////////////////////
std::string PeriodicTable::configuration(std::string symbol) const
{
  return ptable[z(symbol)-1].config;
}

////////////////////////////////////////////////////////////////////////////////
double PeriodicTable::mass(int z) const
{
  assert(z>0 && z<=ptable.size());
  return ptable[z-1].mass;
}

////////////////////////////////////////////////////////////////////////////////
double PeriodicTable::mass(std::string symbol) const
{
  return ptable[z(symbol)-1].mass;
}

////////////////////////////////////////////////////////////////////////////////
int PeriodicTable::size(void) const
{
  return ptable.size();
}

////////////////////////////////////////////////////////////////////////////////
PeriodicTable::PeriodicTable(void)
{
  ptable.push_back(Element(1,"H","1s1",1.00794));
  ptable.push_back(Element(2,"He","1s2",4.00260));
  ptable.push_back(Element(3, "Li","1s2 2s1",     6.941));
  ptable.push_back(Element(4, "Be","1s2 2s2",     9.01218));
  ptable.push_back(Element(5, "B", "1s2 2s2 2p1",10.811));
  ptable.push_back(Element(6, "C", "1s2 2s2 2p2",12.0107));
  ptable.push_back(Element(7, "N", "1s2 2s2 2p3",14.00674));
  ptable.push_back(Element(8, "O", "1s2 2s2 2p4",15.9994));
  ptable.push_back(Element(9, "F", "1s2 2s2 2p5",18.9884));
  ptable.push_back(Element(10,"Ne","1s2 2s2 2p6",20.1797));

  ptable.push_back(Element(11,"Na","[Ne] 3s1",    22.98977));
  ptable.push_back(Element(12,"Mg","[Ne] 3s2",    24.3050));
  ptable.push_back(Element(13,"Al","[Ne] 3s2 3p1",26.98154));
  ptable.push_back(Element(14,"Si","[Ne] 3s2 3p2",28.0855));
  ptable.push_back(Element(15,"P", "[Ne] 3s2 3p3",30.97376));
  ptable.push_back(Element(16,"S", "[Ne] 3s2 3p4",32.066));
  ptable.push_back(Element(17,"Cl","[Ne] 3s2 3p5",35.4527));
  ptable.push_back(Element(18,"Ar","[Ne] 3s2 3p6",39.948));

  ptable.push_back(Element(19,"K", "[Ar] 4s1",39.0983));
  ptable.push_back(Element(20,"Ca","[Ar] 4s2",40.078));
  ptable.push_back(Element(21,"Sc","[Ar] 3d1 4s2",44.95591));
  ptable.push_back(Element(22,"Ti","[Ar] 3d2 4s2",47.867));
  ptable.push_back(Element(23,"V", "[Ar] 3d3 4s2",50.9415));
  ptable.push_back(Element(24,"Cr","[Ar] 3d5 4s1",51.9961));
  ptable.push_back(Element(25,"Mn","[Ar] 3d5 4s2",54.93805));
  ptable.push_back(Element(26,"Fe","[Ar] 3d6 4s2",55.845));
  ptable.push_back(Element(27,"Co","[Ar] 3d7 4s2",58.9332));
  ptable.push_back(Element(28,"Ni","[Ar] 3d8 4s2",58.6934));
  ptable.push_back(Element(29,"Cu","[Ar] 3d10 4s1",63.546));
  ptable.push_back(Element(30,"Zn","[Ar] 3d10 4s2",65.39));
  ptable.push_back(Element(31,"Ga","[Ar] 3d10 4s2 4p1",69.723));
  ptable.push_back(Element(32,"Ge","[Ar] 3d10 4s2 4p2",72.61));
  ptable.push_back(Element(33,"As","[Ar] 3d10 4s2 4p3",74.9216));
  ptable.push_back(Element(34,"Se","[Ar] 3d10 4s2 4p4",78.96));
  ptable.push_back(Element(35,"Br","[Ar] 3d10 4s2 4p5",79.904));
  ptable.push_back(Element(36,"Kr","[Ar] 3d10 4s2 4p6",83.80));

  ptable.push_back(Element(37,"Rb","[Kr] 5s1",85.4678));
  ptable.push_back(Element(38,"Sr","[Kr] 5s2",87.62));
  ptable.push_back(Element(39,"Y" ,"[Kr] 4d1 5s2",88.90585));
  ptable.push_back(Element(40,"Zr","[Kr] 4d2 5s2",91.224));
  ptable.push_back(Element(41,"Nb","[Kr] 4d4 5s1",92.90638));
  ptable.push_back(Element(42,"Mo","[Kr] 4d5 5s1",95.94));
  ptable.push_back(Element(43,"Tc","[Kr] 4d5 5s2",98.0));
  ptable.push_back(Element(44,"Ru","[Kr] 4d7 5s1",101.07));
  ptable.push_back(Element(45,"Rh","[Kr] 4d8 5s1",102.9055));
  ptable.push_back(Element(46,"Pd","[Kr] 4d10",106.42));
  ptable.push_back(Element(47,"Ag","[Kr] 4d10 5s1",107.8682));
  ptable.push_back(Element(48,"Cd","[Kr] 4d10 5s2",112.411));
  ptable.push_back(Element(49,"In","[Kr] 4d10 5s2 5p1",114.818));
  ptable.push_back(Element(50,"Sn","[Kr] 4d10 5s2 5p2",118.710));
  ptable.push_back(Element(51,"Sb","[Kr] 4d10 5s2 5p3",121.760));
  ptable.push_back(Element(52,"Te","[Kr] 4d10 5s2 5p4",127.60));
  ptable.push_back(Element(53,"I" ,"[Kr] 4d10 5s2 5p5",126.90447));
  ptable.push_back(Element(54,"Xe","[Kr] 4d10 5s2 5p6",131.29));

  ptable.push_back(Element(55,"Cs","[Xe] 6s1",132.90545));
  ptable.push_back(Element(56,"Ba","[Xe] 6s2",137.327));
  ptable.push_back(Element(57,"La","[Xe] 5d1 6s2",138.9055));
  ptable.push_back(Element(58,"Ce","[Xe] 4f1 5d1 6s2",140.116));
  ptable.push_back(Element(59,"Pr","[Xe] 4f3 6s2",140.90765));
  ptable.push_back(Element(60,"Nd","[Xe] 4f4 6s2",144.24));
  ptable.push_back(Element(61,"Pm","[Xe] 4f5 6s2",145.0));
  ptable.push_back(Element(62,"Sm","[Xe] 4f6 6s2",150.36));
  ptable.push_back(Element(63,"Eu","[Xe] 4f7 6s2",151.964));
  ptable.push_back(Element(64,"Gd","[Xe] 4f7 5d1 6s2",157.25));
  ptable.push_back(Element(65,"Tb","[Xe] 4f9 6s2",158.92534));
  ptable.push_back(Element(66,"Dy","[Xe] 4f10 6s2",162.50));
  ptable.push_back(Element(67,"Ho","[Xe] 4f11 6s2",164.93032));
  ptable.push_back(Element(68,"Er","[Xe] 4f12 6s2",167.26));
  ptable.push_back(Element(69,"Tm","[Xe] 4f13 6s2",168.93421));
  ptable.push_back(Element(70,"Yb","[Xe] 4f14 6s2",173.04));
  ptable.push_back(Element(71,"Lu","[Xe] 4f14 5d1 6s2",174.967));
  ptable.push_back(Element(72,"Hf","[Xe] 4f14 5d2 6s2",178.49));
  ptable.push_back(Element(73,"Ta","[Xe] 4f14 5d3 6s2",180.9479));
  ptable.push_back(Element(74,"W" ,"[Xe] 4f14 5d4 6s2",183.84));
  ptable.push_back(Element(75,"Re","[Xe] 4f14 5d5 6s2",186.207));
  ptable.push_back(Element(76,"Os","[Xe] 4f14 5d6 6s2",190.23));
  ptable.push_back(Element(77,"Ir","[Xe] 4f14 5d7 6s2",192.217));
  ptable.push_back(Element(78,"Pt","[Xe] 4f14 5d9 6s1",195.078));
  ptable.push_back(Element(79,"Au","[Xe] 4f14 5d10 6s1",196.96655));
  ptable.push_back(Element(80,"Hg","[Xe] 4f14 5d10 6s2",200.59));
  ptable.push_back(Element(81,"Tl","[Xe] 4f14 5d10 6s2 6p1",204.3833));
  ptable.push_back(Element(82,"Pb","[Xe] 4f14 5d10 6s2 6p2",207.2));
  ptable.push_back(Element(83,"Bi","[Xe] 4f14 5d10 6s2 6p3",208.98038));
  ptable.push_back(Element(84,"Po","[Xe] 4f14 5d10 6s2 6p4",209.0));
  ptable.push_back(Element(85,"At","[Xe] 4f14 5d10 6s2 6p5",210.0));
  ptable.push_back(Element(86,"Rn","[Xe] 4f14 5d10 6s2 6p6",222.0));

  ptable.push_back(Element(87,"Fr","[Rn] 7s1",223.0));
  ptable.push_back(Element(88,"Ra","[Rn] 7s2",226.0));
  ptable.push_back(Element(89,"Ac","[Rn] 6d1 7s2",227.0));
  ptable.push_back(Element(90,"Th","[Rn] 6d2 7s2",232.0381));
  ptable.push_back(Element(91,"Pa","[Rn] 5f2 6d1 7s2",231.03588));
  ptable.push_back(Element(92,"U" ,"[Rn] 5f3 6d1 7s2",238.0289));
  ptable.push_back(Element(93,"Np","[Rn] 5f4 6d1 7s2",237.0));
  ptable.push_back(Element(94,"Pu","[Rn] 5f5 6d1 7s2",244.0));

  for ( int i = 0; i < ptable.size(); i++ )
    zmap[ptable[i].symbol] = i+1;
}

// change the main subroutine to a readin function.
Int ReadUPF( std::string file_name, PTEntry * tempEntry, Int * atom)
{
  bool realspace = esdfParam.isUseRealSpace;
  bool lspinorb = esdfParam.SpinOrbitCoupling;

  DblNumVec & params  = (*tempEntry).params;
  DblNumMat & samples = (*tempEntry).samples;
  DblNumVec & weights = (*tempEntry).weights;
  IntNumVec & types   = (*tempEntry).types;
  DblNumVec & soctypes = (*tempEntry).soctypes;
  DblNumVec & cutoffs = (*tempEntry).cutoffs;

  params.Resize(5); // in the order of the ParamPT
  
  PeriodicTable pt;

  std::string buf,s;
  std::istringstream is;

  // determine UPF version
  int upf_version = 0;

  // The first line of the UPF potential file contains either of the following:
  // <PP_INFO>  (for UPF version 1)
  // <UPF version="2.0.1"> (for UPF version 2)

  std::string::size_type p;
  std::ifstream upfin( file_name );
  getline(upfin,buf);
  p = buf.find("<PP_INFO>");
  if( p != std::string::npos ){
    upf_version = 1;
  }
  else
  {
    p = buf.find("<UPF version=\"2.0.1\">");
    if ( p != std::string::npos )
      upf_version = 2;
  }

  if ( upf_version == 0 )
  {
    statusOFS << " Format of UPF file not recognized " << std::endl;
    statusOFS << " First line of file: " << buf << std::endl;
    ErrorHandling( " Format of UPF file not recognized " );
    return 1;
  }

  if ( upf_version == 1 )
  {
    ErrorHandling( " Format of UPF file 1.0 not supported" );
  }
  else if ( upf_version == 2 )
  { 
    // process UPF version 2 potential
    seek_str("<PP_INFO>", upfin);
    std::string upf_pp_info;
    bool done = false;
    while (!done)
    {
      getline(upfin,buf);
      is.clear();
      is.str(buf);
      is >> s;
      done = ( s == "</PP_INFO>" );
      if ( !done )
      {
        upf_pp_info += buf + '\n';
      }
    }

    // remove all '<' and '>' characters from the PP_INFO field
    // for XML compatibility
    p = upf_pp_info.find_first_of("<>");
    while ( p != std::string::npos )
    {
      upf_pp_info[p] = ' ';
      p = upf_pp_info.find_first_of("<>");
    }

    std::string tag = find_start_element("PP_HEADER", upfin);

    // get attribute "element"
    std::string upf_symbol = get_attr(tag,"element");

    upf_symbol.erase(remove_if(upf_symbol.begin(), upf_symbol.end(), isspace), upf_symbol.end());

    // get atomic number and mass
    const int atomic_number = pt.z(upf_symbol);
    const double mass = pt.mass(upf_symbol);

    *atom = atomic_number;
    params[0] = atomic_number;
    params[1] = mass;

    // check if potential is norm-conserving or semi-local
    std::string pseudo_type = get_attr(tag,"pseudo_type");

    // SOC flag
    std::string upf_soc_flag = get_attr(tag,"has_so");
    if ( upf_soc_flag == "T" )
    {
      if( !lspinorb ){
        ErrorHandling("Pseudopotential includes SOC effect.");
      }
    }
    else{
      if( lspinorb ){
        ErrorHandling("Pseudopotential does not include SOC effect.");
      }
    }

    // NLCC flag
    std::string upf_nlcc_flag = get_attr(tag,"core_correction");
    if ( upf_nlcc_flag == "T" )
    {
      statusOFS << " Potential includes a non-linear core correction" << std::endl;
    }

    // XC functional (add in description)
    std::string upf_functional = get_attr(tag,"functional");
    // add XC functional information to description
    upf_pp_info += "functional = " + upf_functional + '\n';

    // valence charge
    double upf_zval = 0.0;
    std::string buf = get_attr(tag,"z_valence");
    is.clear();
    is.str(buf);
    is >> upf_zval;

    // FIXME rhocut readin from the PSP file. 
    // for SG15, the default value is 6.01.
    // but may change for other user defined PSP files. 
    double rhocut = 6.01; 
    double rhoatomcut = 6.01;
    buf = get_attr(tag,"rho_cutoff");
    is.clear();
    is.str(buf);
    // The cutoff is same for rho and rhoatom
    is >> rhocut;
    is >> rhoatomcut;

    const Int ZION = 2;
    const Int RGAUSSIAN = 3;
    params[ZION] = upf_zval;
    // The calculation of a proper value of Rgauss 
    // is done in PeriodTable::Setup()
    params[RGAUSSIAN] = 1.0;

    // max angular momentum
    int upf_lmax;
    buf = get_attr(tag,"l_max");
    is.clear();
    is.str(buf);
    is >> upf_lmax;

    // local angular momentum
    int upf_llocal;
    buf = get_attr(tag,"l_local");
    is.clear();
    is.str(buf);
    is >> upf_llocal;

    // number of points in mesh
    int upf_mesh_size;
    buf = get_attr(tag,"mesh_size");
    is.clear();
    is.str(buf);
    is >> upf_mesh_size;

    // number of wavefunctions
    int upf_nwf;
    buf = get_attr(tag,"number_of_wfc");
    is.clear();
    is.str(buf);
    is >> upf_nwf;

    // number of projectors
    int upf_nproj;
    buf = get_attr(tag,"number_of_proj");
    is.clear();
    is.str(buf);
    is >> upf_nproj;

    samples.Resize( upf_mesh_size, 5+2*upf_nproj );
    weights.Resize( 5+2*upf_nproj );
    cutoffs.Resize( 5+2*upf_nproj );
    types.Resize( 5+2*upf_nproj );

    SetValue( samples, 0.0 );
    SetValue( weights, 0.0 );
    SetValue( types, 0 );
    SetValue( cutoffs, 0.0 );

    std::vector<int> upf_l(upf_nwf);

    // read mesh
    find_start_element("PP_MESH", upfin);
    find_start_element("PP_R", upfin);
    std::vector<double> upf_r(upf_mesh_size);
    for ( int i = 0; i < upf_mesh_size; i++ )
      upfin >> upf_r[i];
    find_end_element("PP_R", upfin);
    find_start_element("PP_RAB", upfin);
    std::vector<double> upf_rab(upf_mesh_size);
    for ( int i = 0; i < upf_mesh_size; i++ )
      upfin >> upf_rab[i];
    find_end_element("PP_RAB", upfin);
    find_end_element("PP_MESH", upfin);

    // add the mesh into samples.
    for( int i = 0; i < upf_mesh_size; i++)
      samples(i, 0) = upf_r[i];
    weights[0] = -1;
    types[0] = 9;
    // labels
    const Int RADIAL_GRID = 0;
    const Int VLOCAL = 1;
    const Int DRV_VLOCAL = 2;
    const Int RHOATOM = 3;
    const Int DRV_RHOATOM = 4;
    // rhoatomcut should be given by a table according to the element type.
    // nonlocal potential cutoff read from the pseduopotential file below. 4.0 is just initial value.
    double nlcut = 4.0;

    cutoffs[RADIAL_GRID] = rhocut;
    cutoffs[VLOCAL] = rhocut;
    cutoffs[DRV_VLOCAL] = rhocut;

    cutoffs[RHOATOM] = rhoatomcut;
    cutoffs[DRV_RHOATOM] = rhoatomcut;

    // NLCC not used
    std::vector<double> upf_nlcc;
    if ( upf_nlcc_flag == "T" )
    {
      find_start_element("PP_NLCC", upfin);
      upf_nlcc.resize(upf_mesh_size);
      for ( int i = 0; i < upf_mesh_size; i++ )
        upfin >> upf_nlcc[i];
      find_end_element("PP_NLCC", upfin);
    }

    find_start_element("PP_LOCAL", upfin);
    std::vector<double> upf_vloc(upf_mesh_size);
    for ( int i = 0; i < upf_mesh_size; i++ )
      upfin >> upf_vloc[i];
    find_end_element("PP_LOCAL",upfin);

    // add the vlocal into samples.
    // vlocal derivative is 0.0
    for( int i = 0; i < upf_mesh_size; i++)
        upf_vloc[i] = 0.5*upf_vloc[i];

    // The interpolation is needed only when the real space method is taken
    if( realspace )
    {
      std::vector < double > r; 
      std::vector < double > vr; 
      splinerad( upf_r, upf_vloc, r, vr, 1);

      Int n = r.size();
      DblNumVec spla(n,true,&vr[0]); 
      DblNumVec splb(n), splc(n), spld(n);
      spline(n, &r[0], spla.Data(), splb.Data(), splc.Data(), spld.Data());

      seval(&upf_vloc[0], upf_r.size(), &upf_r[0], n, &r[0], spla.Data(), splb.Data(),
          splc.Data(), spld.Data());
    }

    for( int i = 0; i < upf_mesh_size; i++)
       samples(i, 1) = upf_vloc[i];

    // set weights 0-4 to -1
    weights[1] = -1;
    weights[2] = -1;
    weights[3] = -1;
    weights[4] = -1;
    types[1] = 99;
    types[2] = 99;
    types[3] = 999;
    types[4] = 999;

    find_start_element("PP_NONLOCAL", upfin);
    std::vector<std::vector<double> > upf_vnl;
    upf_vnl.resize(upf_nproj);
    std::vector<int> upf_proj_l(upf_nproj);

    std::ostringstream os;
    for ( int j = 0; j < upf_nproj; j++ )
    {
      int index, angular_momentum;
      double cutoff_radius = 4.0; // 4.0 is big enough for Si as default

      os.str("");
      os << j+1;
      std::string element_name = "PP_BETA." + os.str();
      tag = find_start_element(element_name, upfin);

      buf = get_attr(tag,"index");
      is.clear();
      is.str(buf);
      is >> index;

      // reset nlcut
      buf = get_attr(tag,"cutoff_radius");
      is.clear();
      is.str(buf);
      is >> cutoff_radius;
      nlcut = cutoff_radius;

      buf = get_attr(tag,"angular_momentum");
      is.clear();
      is.str(buf);
      is >> angular_momentum;
 
      assert(angular_momentum <= upf_lmax);
      upf_proj_l[j] = angular_momentum;

      upf_vnl[j].resize(upf_mesh_size);
      for ( int i = 0; i < upf_mesh_size; i++ )
        upfin >> upf_vnl[j][i];

      // take the element nonlocal part.
      find_end_element(element_name, upfin);

      // The interpolation is needed only when the real space method is taken
      if( realspace )
      {
        std::vector < double > r; 
        std::vector < double > vr; 
        if(angular_momentum % 2 == 0) 
          splinerad( upf_r, upf_vnl[j], r, vr, 0);
        else
          splinerad( upf_r, upf_vnl[j], r, vr, 1);

        for( int i = 0; i < r.size(); i++)
          vr[i] = vr[i] / r[i];

        Int n = r.size();
        DblNumVec spla(n,true,&vr[0]); 
        DblNumVec splb(n), splc(n), spld(n);
        spline(n, &r[0], spla.Data(), splb.Data(), splc.Data(), spld.Data());

        seval(&upf_vnl[j][0], upf_r.size(), &upf_r[0], n, &r[0], spla.Data(), splb.Data(),
            splc.Data(), spld.Data());
      }
 
      // nonlocal is written.
      // nonlocal derivative is 0.0
      for( int i = 0; i < upf_mesh_size; i++)
        samples(i, 5+2*j) = upf_vnl[j][i];
    }

    tag = find_start_element("PP_DIJ", upfin);
    int size;
    buf = get_attr(tag,"size");
    is.clear();
    is.str(buf);
    is >> size;

    if ( size != upf_nproj*upf_nproj )
    {
      statusOFS << " Number of non-zero Dij differs from number of projectors"
           << std::endl;
      return 1;
    }
    int upf_ndij = size;

    std::vector<double> upf_d(upf_ndij);
    for ( int i = 0; i < upf_ndij; i++ )
    {
      upfin >> upf_d[i];
    }
    int imax = sqrt(size+1.e-5);
    assert(imax*imax==size);

    // Check if Dij has non-diagonal elements
    // non-diagonal elements are not supported
    for ( int m = 0; m < imax; m++ )
      for ( int n = 0; n < imax; n++ )
        if ( (m != n) && (upf_d[n*imax+m] != 0.0) )
        {
          statusOFS << " Non-local Dij has off-diagonal elements" << std::endl;
          statusOFS << " m=" << m << " n=" << n << std::endl;
          return 1;
        }

    find_end_element("PP_DIJ", upfin);

    find_end_element("PP_NONLOCAL", upfin);

    // add the weights to the Dij
    for ( int j = 0; j < upf_nproj; j++ )
    {
      weights[5+2*j] = 0.5 * upf_d[j*imax+j];
      weights[6+2*j] = 0.5 * upf_d[j*imax+j];

      types[5+2*j] = upf_proj_l[j];
      types[6+2*j] = upf_proj_l[j];
      // FIXME nonlocal cutoff should be given by a table according to the element type.
      cutoffs[5+2*j] = nlcut;
      cutoffs[6+2*j] = nlcut;
    }

    find_start_element("PP_RHOATOM", upfin);
    std::vector<double> upf_rho_atom(upf_mesh_size);
    for ( int i = 0; i < upf_mesh_size; i++ )
      upfin >> upf_rho_atom[i];

    // add the spline part
    // The interpolation is needed only when the real space method is taken
    if( esdfParam.isUseRealSpace == true )
    {
      std::vector < double > r; 
      std::vector < double > vr; 
      splinerad( upf_r, upf_rho_atom, r, vr, 1);
      for( int i = 0; i < r.size(); i++)
        vr[i] = vr[i] / ( 4.0 * PI * r[i] * r[i] );

       Int n = r.size();
       DblNumVec spla(n,true,&vr[0]); 
       DblNumVec splb(n), splc(n), spld(n);
       spline(n, &r[0], spla.Data(), splb.Data(), splc.Data(), spld.Data());

       seval(&upf_rho_atom[0], upf_r.size(), &upf_r[0], n, &r[0], spla.Data(), splb.Data(),
           splc.Data(), spld.Data());
    }
 
    // add the rho_atom to the samples
    // rho_atom derivative is 0.0
    for( int i = 0; i < upf_mesh_size; i++)
      samples(i, 3) = upf_rho_atom[i] ;

    find_end_element("PP_RHOATOM", upfin);

    if( upf_soc_flag == "T" )
    { 
      soctypes.Resize(5+2*upf_nproj);
      SetValue( soctypes, 0.0);
      soctypes[1] = 99;
      soctypes[2] = 99;
      soctypes[3] = 999;
      soctypes[4] = 999;

      Real upf_jmax = Real( upf_lmax ) + 0.5;
      find_start_element("PP_SPIN_ORB", upfin);
      std::vector<Real> upf_proj_j(upf_nproj);
      
      for ( int j = 0; j < upf_nproj; j++ )
      { 
        int index;
        Real total_angular_momentum;
        
        os.str("");
        os << j+1;
        std::string element_name = "PP_RELBETA." + os.str();
        tag = find_start_element(element_name, upfin);
        
        buf = get_attr(tag,"index");
        is.clear();
        is.str(buf);
        is >> index;
        
        buf = get_attr(tag,"jjj");
        is.clear();
        is.str(buf);
        is >> total_angular_momentum;
        
        assert(total_angular_momentum <= upf_jmax);
	      upf_proj_j[j] = total_angular_momentum;
        
        soctypes[5+2*j] = upf_proj_j[j];
        soctypes[6+2*j] = upf_proj_j[j];
      }
      find_end_element("PP_SPIN_ORB", upfin);
    }
  } // version 1 or 2
  
  return 0;
}

} // namespace pwdft

