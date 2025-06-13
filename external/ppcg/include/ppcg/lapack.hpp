/// @file lapack.hpp
/// @brief Thin interface to LAPACK
/// @date 2023-07-01
#ifndef _PPCG_LAPACK_HPP_
#define _PPCG_LAPACK_HPP_

#include  "ppcg/environment.hpp"

namespace PPCG {

/// @namespace lapack
///
/// @brief Thin interface to LAPACK.
namespace lapack {

typedef  int                    Int; 
typedef  std::complex<float>    scomplex;
typedef  std::complex<double>   dcomplex;


// *********************************************************************
// Cholesky factorization
// *********************************************************************

void Potrf( char uplo, Int n, const float* A, Int lda );
void Potrf( char uplo, Int n, const double* A, Int lda );
void Potrf( char uplo, Int n, const scomplex* A, Int lda );
void Potrf( char uplo, Int n, const dcomplex* A, Int lda );


// *********************************************************************
// LU factorization (with partial pivoting)
// *********************************************************************

void Getrf( Int m, Int n, float* A, Int lda, Int* p );
void Getrf( Int m, Int n, double* A, Int lda, Int* p );
void Getrf( Int m, Int n, scomplex* A, Int lda, Int* p );
void Getrf( Int m, Int n, dcomplex* A, Int lda, Int* p );

// *********************************************************************
// For reducing well-conditioned Hermitian generalized-definite EVP's
// to standard form.
// *********************************************************************

void Hegst
  ( Int itype, char uplo, 
    Int n, float* A, Int lda, const float* B, Int ldb );
void Hegst
  ( Int itype, char uplo,
    Int n, double* A, Int lda, const double* B, Int ldb );
void Hegst
  ( Int itype, char uplo,
    Int n, scomplex* A, Int lda, const scomplex* B, Int ldb );
void Hegst
  ( Int itype, char uplo,
    Int n, dcomplex* A, Int lda, const dcomplex* B, Int ldb );

// *********************************************************************
// For solving the standard eigenvalue problem using the divide and
// conquer algorithm
// *********************************************************************

void Syevd
  ( char jobz, char uplo, Int n, double* A, Int lda, double* eigs );

void Syevd
  ( char jobz, char uplo, Int n, dcomplex* A, Int lda, double* eigs );

// *********************************************************************
// For solving the generalized eigenvalue problem using the divide and
// conquer algorithm  --- added by Eugene
// *********************************************************************

void Sygvd
  ( Int itype, char jobz, char uplo, Int n, double* A, Int lda, double* B, Int ldb, double* eigs );

void Sygvd
  ( Int itype, char jobz, char uplo, Int n, dcomplex* A, Int lda, dcomplex* B, Int ldb, double* eigs );

// *********************************************************************
// For computing the inverse of a triangular matrix
// *********************************************************************

void Trtri
  ( char uplo, char diag, Int n, const float* A, Int lda );
void Trtri
  ( char uplo, char diag, Int n, const double* A, Int lda );
void Trtri
  ( char uplo, char diag, Int n, const scomplex* A, Int lda );
void Trtri
  ( char uplo, char diag, Int n, const dcomplex* A, Int lda );


// *********************************************************************
// Compute the SVD of a general matrix using a divide and conquer algorithm
// *********************************************************************

void DivideAndConquerSVD
  ( Int m, Int n, float* A, Int lda, 
    float* s, float* U, Int ldu, float* VTrans, Int ldvt );
void DivideAndConquerSVD
  ( Int m, Int n, double* A, Int lda, 
    double* s, double* U, Int ldu, double* VTrans, Int ldvt );
void DivideAndConquerSVD
  ( Int m, Int n, scomplex* A, Int lda, 
    float* s, scomplex* U, Int ldu, scomplex* VAdj, Int ldva );
void DivideAndConquerSVD
  ( Int m, Int n, dcomplex* A, Int lda, 
    double* s, dcomplex* U, Int ldu, dcomplex* VAdj, Int ldva );

//
// Compute the SVD of a general matrix using the QR algorithm
//

void QRSVD
  ( Int m, Int n, float* A, Int lda, 
    float* s, float* U, Int ldu, float* VTrans, Int ldvt );
void QRSVD
  ( Int m, Int n, double* A, Int lda, 
    double* s, double* U, Int ldu, double* VTrans, Int ldvt );
void QRSVD
  ( Int m, Int n, scomplex* A, Int lda, 
    float* s, scomplex* U, Int ldu, scomplex* VAdj, Int ldva );
void QRSVD
  ( Int m, Int n, dcomplex* A, Int lda, 
    double* s, dcomplex* U, Int ldu, dcomplex* VAdj, Int ldva );


// *********************************************************************
// Compute the singular values of a general matrix using the QR algorithm
// *********************************************************************

void SingularValues( Int m, Int n, float* A, Int lda, float* s );
void SingularValues( Int m, Int n, double* A, Int lda, double* s );
void SingularValues( Int m, Int n, scomplex* A, Int lda, float* s );
void SingularValues( Int m, Int n, dcomplex* A, Int lda, double* s );

// *********************************************************************
// Compute the SVD of a bidiagonal matrix using the QR algorithm
// *********************************************************************

void BidiagQRAlg
  ( char uplo, Int n, Int numColsVTrans, Int numRowsU,
    float* d, float* e, float* VTrans, Int ldVTrans, float* U, Int ldU );
void BidiagQRAlg
  ( char uplo, Int n, Int numColsVTrans, Int numRowsU, 
    double* d, double* e, double* VTrans, Int ldVTrans, double* U, Int ldU );
void BidiagQRAlg
  ( char uplo, Int n, Int numColsVAdj, Int numRowsU,
    float* d, float* e, scomplex* VAdj, Int ldVAdj, scomplex* U, Int ldU );
void BidiagQRAlg
  ( char uplo, Int n, Int numColsVAdj, Int numRowsU, 
    double* d, double* e, dcomplex* VAdj, Int ldVAdj, dcomplex* U, Int ldU );

// *********************************************************************
// Compute the linear least square problem using SVD
// *********************************************************************
void SVDLeastSquare( Int m, Int n, Int nrhs, float * A, Int lda,
    float * B, Int ldb, float * S, float rcond,
    Int* rank );
void SVDLeastSquare( Int m, Int n, Int nrhs, double * A, Int lda,
    double * B, Int ldb, double * S, double rcond,
    Int* rank );
void SVDLeastSquare( Int m, Int n, Int nrhs, scomplex * A, Int lda,
    scomplex * B, Int ldb, float * S, float rcond,
    Int* rank );
void SVDLeastSquare( Int m, Int n, Int nrhs, dcomplex * A, Int lda,
    dcomplex * B, Int ldb, double * S, double rcond,
    Int* rank );

// *********************************************************************
// Copy
// *********************************************************************

void Lacpy( char uplo, Int m, Int n, const double* A, Int lda,
    double* B, Int ldb    );

void Lacpy( char uplo, Int m, Int n, const dcomplex* A, Int lda,
    dcomplex* B, Int ldb    );

// *********************************************************************
// Inverting a factorized matrix: Getri
// *********************************************************************


void Getri ( Int n, double* A, Int lda, const Int* ipiv );

void Getri ( Int n, dcomplex* A, Int lda, const Int* ipiv );


// *********************************************************************
// // Compute the factorization of a real symmetric matrix: Sytrf
// // *********************************************************************
void Sytrf ( char uplo, Int n, double* A, Int lda, Int* ipiv );



// *********************************************************************
// // // Inverting a real symmetric matrix factorized by Sytrf
// // // *********************************************************************
void Sytri ( char uplo, Int n, double* A, Int lda, Int* ipiv );



/// @brief Orthogonalization.
///
/// The first min(m,n) columns of the matrix A is overwritten by the
/// orthonormal columns. This is similar to MATLAB's orth function, and
/// is performed using SVD.
void Orth( Int m, Int n, double* A, Int lda );

//-----lijl open Orth for dcomplex
void Orth( Int m, Int n, dcomplex* A, Int lda );
//--------lijl

/// @brief QRCP Wrapper for xgeqp3 routine.
///
/// @todo Another interface for QRCP that directly returns the Q factor.
/// This should use DORGQR routine to generate the Q matrix.
///
/// The index piv is adjusted to be 0-based.
void QRCP( Int m, Int n, double* A, Int lda, Int * piv, 
    double* tau );

/// @brief QR wrapper for xgeqrf routine.
///
/// @todo Another interface for QRCP that directly returns the Q factor.
/// This should use DORGQR routine to generate the Q matrix.
void QR( Int m, Int n, double* A, Int lda, double* tau );


/// @brief QRCP Wrapper for xgeqp3 routine that returns the Q and R
/// factors.
///
/// The index piv is adjusted to be 0-based.
///
/// The A matrix is still destroyed in the output.
void QRCP( Int m, Int n, double* A, double* Q, double* R, 
    Int lda, Int * piv );

} // namespace lapack
} // namespace PPCG
#endif // _PPCG_LAPACK_HPP_
