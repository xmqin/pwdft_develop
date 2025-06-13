/// @file utility.cpp
/// @brief Utility subroutines
/// @date 2023-07-01
#include "ppcg/utility.hpp"

// *********************************************************************
// IO functions
// *********************************************************************
//---------------------------------------------------------

namespace PPCG {

Int SeparateRead(std::string name, std::istringstream& is)
{
  int mpirank;  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  int mpisize;  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  //
  char filename[100];
  sprintf(filename, "%s_%d_%d", name.c_str(), mpirank, mpisize);  
  std::ifstream fin(filename);
  if( !fin.good() ){
    ErrorHandling( "File cannot be open!" );
  }

  is.str( std::string(std::istreambuf_iterator<char>(fin), std::istreambuf_iterator<char>()) );
  fin.close();
  return 0;
}

//---------------------------------------------------------
Int SeparateRead(std::string name, std::istringstream& is, Int outputIndex)
{
  char filename[100];
  sprintf(filename, "%s_%d", name.c_str(), outputIndex);
  std::ifstream fin(filename);
  if( !fin.good() ){
    ErrorHandling( "File cannot be open!" );
  }

  is.str( std::string(std::istreambuf_iterator<char>(fin), std::istreambuf_iterator<char>()) );
  fin.close();
  return 0;
}


//---------------------------------------------------------
Int SeparateWrite(std::string name, std::ostringstream& os)
{
  int mpirank;  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  int mpisize;  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  //
  char filename[100];
  sprintf(filename, "%s_%d_%d", name.c_str(), mpirank, mpisize);
  std::ofstream fout(filename);
  if( !fout.good() ){
    ErrorHandling( "File cannot be open!" );
  }
  fout<<os.str();
  fout.close();
  return 0;
}


//---------------------------------------------------------
Int SeparateWrite(std::string name, std::ostringstream& os, Int outputIndex)
{
  char filename[100];
  sprintf(filename, "%s_%d", name.c_str(), outputIndex);
  std::ofstream fout(filename);
  if( !fout.good() ){
    ErrorHandling( "File cannot be open!" );
  }
  fout<<os.str();
  fout.close();
  return 0;
}

//---------------------------------------------------------
Int SharedRead(std::string name, std::istringstream& is)
{
  MPI_Barrier(MPI_COMM_WORLD);
  int mpirank;  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  int mpisize;  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  //
  std::vector<char> tmpstr;
  if(mpirank==0) {
    std::ifstream fin(name.c_str());
    if( !fin.good() ){
      ErrorHandling( "File cannot be open!" );
    }
    //std::string str(std::istreambuf_iterator<char>(fin), std::istreambuf_iterator<char>());
    //tmpstr.insert(tmpstr.end(), str.begin(), str.end());
    tmpstr.insert(tmpstr.end(), std::istreambuf_iterator<char>(fin), std::istreambuf_iterator<char>());
    fin.close();
    int size = tmpstr.size();    
    MPI_Bcast((void*)&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast((void*)&(tmpstr[0]), size, MPI_BYTE, 0, MPI_COMM_WORLD);
  } else {
    int size;
    MPI_Bcast((void*)&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    tmpstr.resize(size);
    MPI_Bcast((void*)&(tmpstr[0]), size, MPI_BYTE, 0, MPI_COMM_WORLD);
  }
  is.str( std::string(tmpstr.begin(), tmpstr.end()) );
  //
  MPI_Barrier(MPI_COMM_WORLD);
  return 0;
}

//---------------------------------------------------------
Int SharedWrite(std::string name, std::ostringstream& os)
{
  MPI_Barrier(MPI_COMM_WORLD);
  int mpirank;  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  int mpisize;  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  //
  if(mpirank==0) {
    std::ofstream fout(name.c_str());
    if( !fout.good() ){
      ErrorHandling( "File cannot be open!" );
    }
    fout<<os.str();
    fout.close();
  }
  MPI_Barrier(MPI_COMM_WORLD);
  return 0;
}


//---------------------------------------------------------
Int SeparateWriteAscii(std::string name, std::ostringstream& os)
{
  MPI_Barrier(MPI_COMM_WORLD);
  int mpirank;  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  int mpisize;  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  //
  char filename[100];
  sprintf(filename, "%s_%d_%d", name.c_str(), mpirank, mpisize);
  std::ofstream fout(filename, std::ios::trunc);
  if( !fout.good() ){
    ErrorHandling( "File cannot be open!" );
  }
  fout<<os.str();
  fout.close();
  //
  MPI_Barrier(MPI_COMM_WORLD);
  return 0;
}


void AlltoallForward( DblNumMat& A, DblNumMat& B, MPI_Comm comm )
{

  int mpirank, mpisize;
  MPI_Comm_rank( comm, &mpirank );
  MPI_Comm_size( comm, &mpisize );

  Int height = A.m();
  Int widthTemp = A.n();

  Int width = 0;
  MPI_Allreduce( &widthTemp, &width, 1, MPI_INT, MPI_SUM, comm );

  Int widthBlocksize = width / mpisize;
  Int heightBlocksize = height / mpisize;
  Int widthLocal = widthBlocksize;
  Int heightLocal = heightBlocksize;

  if(mpirank < (width % mpisize)){
    widthLocal = widthBlocksize + 1;
  }
  
  if(mpirank < (height % mpisize)){
    heightLocal = heightBlocksize + 1;
  }
  
  DblNumVec sendbuf(height*widthLocal); 
  DblNumVec recvbuf(heightLocal*width);
  IntNumVec sendcounts(mpisize);
  IntNumVec recvcounts(mpisize);
  IntNumVec senddispls(mpisize);
  IntNumVec recvdispls(mpisize);
  IntNumMat  sendk( height, widthLocal );
  IntNumMat  recvk( heightLocal, width );

  for( Int k = 0; k < mpisize; k++ ){ 
    sendcounts[k] = heightBlocksize * widthLocal;
    if( k < (height % mpisize)){
      sendcounts[k] = sendcounts[k] + widthLocal;  
    }
  }

  for( Int k = 0; k < mpisize; k++ ){ 
    recvcounts[k] = heightLocal * widthBlocksize;
    if( k < (width % mpisize)){
      recvcounts[k] = recvcounts[k] + heightLocal;  
    }
  }

  senddispls[0] = 0;
  recvdispls[0] = 0;
  for( Int k = 1; k < mpisize; k++ ){ 
    senddispls[k] = senddispls[k-1] + sendcounts[k-1];
    recvdispls[k] = recvdispls[k-1] + recvcounts[k-1];
  }

  if((height % mpisize) == 0){
    for( Int j = 0; j < widthLocal; j++ ){ 
      for( Int i = 0; i < height; i++ ){
        sendk(i, j) = senddispls[i / heightBlocksize] + j * heightBlocksize + i % heightBlocksize;
      } 
    }
  }
  else{
    for( Int j = 0; j < widthLocal; j++ ){ 
      for( Int i = 0; i < height; i++ ){
        if( i < ((height % mpisize) * (heightBlocksize+1)) ){
          sendk(i, j) = senddispls[i / (heightBlocksize+1)] + j * (heightBlocksize+1) + i % (heightBlocksize+1);
        }
        else {
          sendk(i, j) = senddispls[(height % mpisize) + (i-(height % mpisize)*(heightBlocksize+1))/heightBlocksize]
            + j * heightBlocksize + (i-(height % mpisize)*(heightBlocksize+1)) % heightBlocksize;
        }
      }
    }
  }

  for( Int j = 0; j < width; j++ ){ 
    for( Int i = 0; i < heightLocal; i++ ){
      recvk(i, j) = recvdispls[j % mpisize] + (j / mpisize) * heightLocal + i;
    }
  }

  for( Int j = 0; j < widthLocal; j++ ){ 
    for( Int i = 0; i < height; i++ ){
      sendbuf[sendk(i, j)] = A(i, j); 
    }
  }
  MPI_Alltoallv( &sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE, 
      &recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE, comm );
  for( Int j = 0; j < width; j++ ){ 
    for( Int i = 0; i < heightLocal; i++ ){
      B(i, j) = recvbuf[recvk(i, j)];
    }
  }


  return ;
}        // -----  end of function AlltoallForward ----- 

void AlltoallForward( CpxNumMat& A, CpxNumMat& B, MPI_Comm comm )
{

  int mpirank, mpisize;
  MPI_Comm_rank( comm, &mpirank );
  MPI_Comm_size( comm, &mpisize );

  Int height = A.m();
  Int widthTemp = A.n();

  Int width = 0;
  MPI_Allreduce( &widthTemp, &width, 1, MPI_INT, MPI_SUM, comm );

  Int widthBlocksize = width / mpisize;
  Int heightBlocksize = height / mpisize;
  Int widthLocal = widthBlocksize;
  Int heightLocal = heightBlocksize;

  if(mpirank < (width % mpisize)){
    widthLocal = widthBlocksize + 1;
  }
  
  if(mpirank < (height % mpisize)){
    heightLocal = heightBlocksize + 1;
  }
  
  CpxNumVec sendbuf(height*widthLocal); 
  CpxNumVec recvbuf(heightLocal*width);
  IntNumVec sendcounts(mpisize);
  IntNumVec recvcounts(mpisize);
  IntNumVec senddispls(mpisize);
  IntNumVec recvdispls(mpisize);
  IntNumMat  sendk( height, widthLocal );
  IntNumMat  recvk( heightLocal, width );

  for( Int k = 0; k < mpisize; k++ ){ 
    sendcounts[k] = heightBlocksize * widthLocal;
    if( k < (height % mpisize)){
      sendcounts[k] = sendcounts[k] + widthLocal;  
    }
  }

  for( Int k = 0; k < mpisize; k++ ){ 
    recvcounts[k] = heightLocal * widthBlocksize;
    if( k < (width % mpisize)){
      recvcounts[k] = recvcounts[k] + heightLocal;  
    }
  }

  senddispls[0] = 0;
  recvdispls[0] = 0;
  for( Int k = 1; k < mpisize; k++ ){ 
    senddispls[k] = senddispls[k-1] + sendcounts[k-1];
    recvdispls[k] = recvdispls[k-1] + recvcounts[k-1];
  }

  if((height % mpisize) == 0){
    for( Int j = 0; j < widthLocal; j++ ){ 
      for( Int i = 0; i < height; i++ ){
        sendk(i, j) = senddispls[i / heightBlocksize] + j * heightBlocksize + i % heightBlocksize;
      } 
    }
  }
  else{
    for( Int j = 0; j < widthLocal; j++ ){ 
      for( Int i = 0; i < height; i++ ){
        if( i < ((height % mpisize) * (heightBlocksize+1)) ){
          sendk(i, j) = senddispls[i / (heightBlocksize+1)] + j * (heightBlocksize+1) + i % (heightBlocksize+1);
        }
        else {
          sendk(i, j) = senddispls[(height % mpisize) + (i-(height % mpisize)*(heightBlocksize+1))/heightBlocksize]
            + j * heightBlocksize + (i-(height % mpisize)*(heightBlocksize+1)) % heightBlocksize;
        }
      }
    }
  }

  for( Int j = 0; j < width; j++ ){ 
    for( Int i = 0; i < heightLocal; i++ ){
      recvk(i, j) = recvdispls[j % mpisize] + (j / mpisize) * heightLocal + i;
    }
  }

  for( Int j = 0; j < widthLocal; j++ ){ 
    for( Int i = 0; i < height; i++ ){
      sendbuf[sendk(i, j)] = A(i, j); 
    }
  }
  MPI_Alltoallv( &sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE_COMPLEX, 
      &recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE_COMPLEX, comm );
  for( Int j = 0; j < width; j++ ){ 
    for( Int i = 0; i < heightLocal; i++ ){
      B(i, j) = recvbuf[recvk(i, j)];
    }
  }


  return ;
}        // -----  end of function AlltoallForward ----- 


void AlltoallBackward( DblNumMat& A, DblNumMat& B, MPI_Comm comm )
{

  int mpirank, mpisize;
  MPI_Comm_rank( comm, &mpirank );
  MPI_Comm_size( comm, &mpisize );

  Int height = B.m();
  Int widthTemp = B.n();

  Int width = 0;
  MPI_Allreduce( &widthTemp, &width, 1, MPI_INT, MPI_SUM, comm );

  Int widthBlocksize = width / mpisize;
  Int heightBlocksize = height / mpisize;
  Int widthLocal = widthBlocksize;
  Int heightLocal = heightBlocksize;

  if(mpirank < (width % mpisize)){
    widthLocal = widthBlocksize + 1;
  }

  if(mpirank < (height % mpisize)){
    heightLocal = heightBlocksize + 1;
  }

  DblNumVec sendbuf(height*widthLocal); 
  DblNumVec recvbuf(heightLocal*width);
  IntNumVec sendcounts(mpisize);
  IntNumVec recvcounts(mpisize);
  IntNumVec senddispls(mpisize);
  IntNumVec recvdispls(mpisize);
  IntNumMat  sendk( height, widthLocal );
  IntNumMat  recvk( heightLocal, width );

  for( Int k = 0; k < mpisize; k++ ){ 
    sendcounts[k] = heightBlocksize * widthLocal;
    if( k < (height % mpisize)){
      sendcounts[k] = sendcounts[k] + widthLocal;  
    }
  }

  for( Int k = 0; k < mpisize; k++ ){ 
    recvcounts[k] = heightLocal * widthBlocksize;
    if( k < (width % mpisize)){
      recvcounts[k] = recvcounts[k] + heightLocal;  
    }
  }

  senddispls[0] = 0;
  recvdispls[0] = 0;
  for( Int k = 1; k < mpisize; k++ ){ 
    senddispls[k] = senddispls[k-1] + sendcounts[k-1];
    recvdispls[k] = recvdispls[k-1] + recvcounts[k-1];
  }

  if((height % mpisize) == 0){
    for( Int j = 0; j < widthLocal; j++ ){ 
      for( Int i = 0; i < height; i++ ){
        sendk(i, j) = senddispls[i / heightBlocksize] + j * heightBlocksize + i % heightBlocksize;
      } 
    }
  }
  else{
    for( Int j = 0; j < widthLocal; j++ ){ 
      for( Int i = 0; i < height; i++ ){
        if( i < ((height % mpisize) * (heightBlocksize+1)) ){
          sendk(i, j) = senddispls[i / (heightBlocksize+1)] + j * (heightBlocksize+1) + i % (heightBlocksize+1);
        }
        else {
          sendk(i, j) = senddispls[(height % mpisize) + (i-(height % mpisize)*(heightBlocksize+1))/heightBlocksize]
            + j * heightBlocksize + (i-(height % mpisize)*(heightBlocksize+1)) % heightBlocksize;
        }
      }
    }
  }

  for( Int j = 0; j < width; j++ ){ 
    for( Int i = 0; i < heightLocal; i++ ){
      recvk(i, j) = recvdispls[j % mpisize] + (j / mpisize) * heightLocal + i;
    }
  }

  for( Int j = 0; j < width; j++ ){ 
    for( Int i = 0; i < heightLocal; i++ ){
      recvbuf[recvk(i, j)] = A(i, j);
    }
  }
  MPI_Alltoallv( &recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE, 
      &sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE, comm );
  for( Int j = 0; j < widthLocal; j++ ){ 
    for( Int i = 0; i < height; i++ ){
      B(i, j) = sendbuf[sendk(i, j)]; 
    }
  }


  return ;
}        // -----  end of function AlltoallBackward ----- 

void AlltoallBackward( CpxNumMat& A, CpxNumMat& B, MPI_Comm comm )
{

  int mpirank, mpisize;
  MPI_Comm_rank( comm, &mpirank );
  MPI_Comm_size( comm, &mpisize );

  Int height = B.m();
  Int widthTemp = B.n();

  Int width = 0;
  MPI_Allreduce( &widthTemp, &width, 1, MPI_INT, MPI_SUM, comm );

  Int widthBlocksize = width / mpisize;
  Int heightBlocksize = height / mpisize;
  Int widthLocal = widthBlocksize;
  Int heightLocal = heightBlocksize;

  if(mpirank < (width % mpisize)){
    widthLocal = widthBlocksize + 1;
  }

  if(mpirank < (height % mpisize)){
    heightLocal = heightBlocksize + 1;
  }

  CpxNumVec sendbuf(height*widthLocal); 
  CpxNumVec recvbuf(heightLocal*width);
  IntNumVec sendcounts(mpisize);
  IntNumVec recvcounts(mpisize);
  IntNumVec senddispls(mpisize);
  IntNumVec recvdispls(mpisize);
  IntNumMat  sendk( height, widthLocal );
  IntNumMat  recvk( heightLocal, width );

  for( Int k = 0; k < mpisize; k++ ){ 
    sendcounts[k] = heightBlocksize * widthLocal;
    if( k < (height % mpisize)){
      sendcounts[k] = sendcounts[k] + widthLocal;  
    }
  }

  for( Int k = 0; k < mpisize; k++ ){ 
    recvcounts[k] = heightLocal * widthBlocksize;
    if( k < (width % mpisize)){
      recvcounts[k] = recvcounts[k] + heightLocal;  
    }
  }

  senddispls[0] = 0;
  recvdispls[0] = 0;
  for( Int k = 1; k < mpisize; k++ ){ 
    senddispls[k] = senddispls[k-1] + sendcounts[k-1];
    recvdispls[k] = recvdispls[k-1] + recvcounts[k-1];
  }

  if((height % mpisize) == 0){
    for( Int j = 0; j < widthLocal; j++ ){ 
      for( Int i = 0; i < height; i++ ){
        sendk(i, j) = senddispls[i / heightBlocksize] + j * heightBlocksize + i % heightBlocksize;
      } 
    }
  }
  else{
    for( Int j = 0; j < widthLocal; j++ ){ 
      for( Int i = 0; i < height; i++ ){
        if( i < ((height % mpisize) * (heightBlocksize+1)) ){
          sendk(i, j) = senddispls[i / (heightBlocksize+1)] + j * (heightBlocksize+1) + i % (heightBlocksize+1);
        }
        else {
          sendk(i, j) = senddispls[(height % mpisize) + (i-(height % mpisize)*(heightBlocksize+1))/heightBlocksize]
            + j * heightBlocksize + (i-(height % mpisize)*(heightBlocksize+1)) % heightBlocksize;
        }
      }
    }
  }

  for( Int j = 0; j < width; j++ ){ 
    for( Int i = 0; i < heightLocal; i++ ){
      recvk(i, j) = recvdispls[j % mpisize] + (j / mpisize) * heightLocal + i;
    }
  }

  for( Int j = 0; j < width; j++ ){ 
    for( Int i = 0; i < heightLocal; i++ ){
      recvbuf[recvk(i, j)] = A(i, j);
    }
  }
  MPI_Alltoallv( &recvbuf[0], &recvcounts[0], &recvdispls[0], MPI_DOUBLE_COMPLEX, 
      &sendbuf[0], &sendcounts[0], &senddispls[0], MPI_DOUBLE_COMPLEX, comm );
  for( Int j = 0; j < widthLocal; j++ ){ 
    for( Int i = 0; i < height; i++ ){
      B(i, j) = sendbuf[sendk(i, j)]; 
    }
  }


  return ;
}        // -----  end of function AlltoallBackward ----- 

void IP_c2f(int* Nc,int* Nf,Complex* tilde,Complex* tildef){
        int iF,jF,kF;
        int ptrc,ptrF;
        int cjF=Nf[1]-Nc[1]/2;
        int ckF=Nf[2]-Nc[2]/2;
        double fac=2.0;
        for(int k=0;k<Nc[2];k++){
                for(int j=0;j<Nc[1];j++){
                        for(int i=0;i<Nc[0]/2+1;i++){
                                ptrc=i+j*(Nc[0]/2+1)+k*(Nc[0]/2+1)*Nc[1];
                                iF=i;
                                if(j<=Nc[1]/2){
                                        jF=j;
                                }
                                else{
                                        jF=Nf[1]-Nc[1]+j;
                                }
                                if(k<=Nc[2]/2){
                                        kF=k;
                                }
                                else{
                                        kF=Nf[2]-Nc[2]+k;
                                }
                                ptrF=iF+jF*(Nf[0]/2+1)+kF*(Nf[0]/2+1)*Nf[1];
                                tildef[ptrF]=tilde[ptrc];
                                if((iF==Nc[0]/2)&&(jF==Nc[1]/2)&&(kF==Nc[2]/2)){
                                        tildef[ptrF]/=fac;
                                }
                                else{
                                        if((iF==Nc[0]/2)&&(jF==Nc[1]/2)){
                                                tildef[ptrF]/=fac;
                                        }
                                        else if((iF==Nc[0]/2)&&(kF==Nc[2]/2)){
                                                tildef[ptrF]/=fac;
                                        }
                                        else if((jF==Nc[1]/2)&&(kF==Nc[2]/2)){
                                                tildef[ptrF]/=fac;
                                            jF=cjF;
                                            kF=ckF;
                                            ptrF=iF+jF*(Nf[0]/2+1)+kF*(Nf[0]/2+1)*Nf[1];
                                            tildef[ptrF]=tilde[ptrc]/fac;
                                        }
                                        else if(iF==Nc[0]/2){
                                                tildef[ptrF]/=fac;
                                        }
                                        else if(jF==Nc[1]/2){
                                                tildef[ptrF]/=fac;
                                                jF=cjF;
                                            ptrF=iF+jF*(Nf[0]/2+1)+kF*(Nf[0]/2+1)*Nf[1];
                                            tildef[ptrF]=tilde[ptrc]/fac;
                                        }
                                        else if(kF==Nc[2]/2){
                                                tildef[ptrF]/=fac;
                                                kF=ckF;
                                            ptrF=iF+jF*(Nf[0]/2+1)+kF*(Nf[0]/2+1)*Nf[1];
                                            tildef[ptrF]=tilde[ptrc]/fac;
                                        }
                                }
                        }
                }
        }
}

//fine to coarse, equals to take the real part of Complex(fine to coarse)

void IP_f2c(int* Nc,int* Nf,Complex* tilde,Complex* tildef){
        int i,j,k;
        int iF,jF,kF;
        int cjF=Nf[1]-Nc[1]/2;
        int ckF=Nf[2]-Nc[2]/2;
        int ptrc,f1,f2,f3,f4;
        for(k=0;k<Nc[2];k++){
                for(j=0;j<Nc[1];j++){
                        for(i=0;i<Nc[0]/2+1;i++){
                                ptrc=i+j*(Nc[0]/2+1)+k*(Nc[0]/2+1)*Nc[1];
                                iF=i;
                                if(j<=Nc[1]/2){
                                        jF=j;
                                }
                                else{
                                        jF=Nf[1]-Nc[1]+j;
                                }
                                if(k<=Nc[2]/2){
                                        kF=k;
                                }
                                else{
                                        kF=Nf[2]-Nc[2]+k;
                                }
                                f1=iF+jF*(Nf[0]/2+1)+kF*(Nf[0]/2+1)*Nf[1];
                                if((i==Nc[0]/2)&&(j==Nc[1]/2)&&(k==Nc[2]/2)){
                                        tilde[ptrc]+=Complex(tildef[f1].real(),0.0);
                                }
                                else if((i==Nc[0]/2)&&(j==Nc[1]/2)){
                                        kF=(Nf[2]-kF)%Nf[2];
                                        f2=iF+jF*(Nf[0]/2+1)+kF*(Nf[0]/2+1)*Nf[1];
                                        tilde[ptrc]+=0.5*(tildef[f1]+std::conj(tildef[f2]));
                                }
                                else if((i==Nc[0]/2)&&(k==Nc[2]/2)){
                                        jF=(Nf[1]-jF)%Nf[1];
                                        f2=iF+jF*(Nf[0]/2+1)+kF*(Nf[0]/2+1)*Nf[1];
                                        tilde[ptrc]+=0.5*(tildef[f1]+std::conj(tildef[f2]));
                                }
                                else if((j==Nc[1]/2)&&(k==Nc[2]/2)){
                                        f2=iF+cjF*(Nf[0]/2+1)+ckF*(Nf[0]/2+1)*Nf[1];
                                        tilde[ptrc]+=0.5*(tildef[f1]+tildef[f2]);
                                }
                                else if(i==Nc[0]/2){
                                        jF=(Nf[1]-jF)%Nf[1];
                                        kF=(Nf[2]-kF)%Nf[2];
                                        f2=iF+jF*(Nf[0]/2+1)+kF*(Nf[0]/2+1)*Nf[1];
                                        tilde[ptrc]+=0.5*(tildef[f1]+std::conj(tildef[f2]));
                                }
                                else if(j==Nc[1]/2){
                                        f2=iF+cjF*(Nf[0]/2+1)+kF*(Nf[0]/2+1)*Nf[1];
                                        tilde[ptrc]+=0.5*(tildef[f1]+tildef[f2]);
                                }
                                else if(k==Nc[2]/2){
                                        f2=iF+jF*(Nf[0]/2+1)+ckF*(Nf[0]/2+1)*Nf[1];
                                        tilde[ptrc]+=0.5*(tildef[f1]+tildef[f2]);
                                }
                                else{
                                        tilde[ptrc]+=tildef[f1];
                                }
                        }
                }
        }
}

} // namespace PPCG

