// File: cuda_errors.c
// CUBLAS and CUFFT error checking.

// includes standard headers

#ifdef GPU  // only used for the GPU version of the PWDFT code. 
#ifndef _CUDA_ERRORS_HPP__
#define _CUDA_ERRORS_HPP__
#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <cufft.h>

/******************************************************/
// CUBLAS and CUFFT error checking, in library

// returns string for CUBLAS API error
char *cublasGetErrorString(cublasStatus_t error);
char *cufftGetErrorString(cufftResult error);
#endif
#endif
/******************************************************/
