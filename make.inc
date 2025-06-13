#/usr/bin/bash

COMPILE_MODE     = release
#COMPILE_MODE     = debug
USE_COMPLEX      = 0
USE_OPENMP       = 0
USE_PROFILE      = 0
USE_PEXSI        = 0
USE_GPU          = 0

ifeq (${COMPILE_MODE}, release)
  COMPILE_DEF    = -DRELEASE
#  COMPILE_FLAG   = -fast -w -D GPUDIRECT -no-ipo
  COMPILE_FLAG   = -O3 -w -g
endif
ifeq (${COMPILE_MODE}, debug)
  COMPILE_DEF    = -DDEBUG=1
  COMPILE_FLAG   = -O2 -w
endif

ifeq (${USE_COMPLEX}, 1)
  SCALAR_DEF    = -DCOMPLEXSCALAR -DCPX
endif


ifeq (${USE_OPENMP}, 1)
  OPENMP_DEF   = -DOPENMP
	OPENMP_FLAG  = -fopenmp
endif

ifeq (${USE_PROFILE}, 1)
	PROFILE_FLAG  = -g -pg
endif

ifeq (${USE_PEXSI}, 1)
  PEXSI_DEF        = -DPEXSI
  PEXSI_INCLUDE    = -I${PEXSI_DIR}/include
  PARMETIS_INCLUDE = -I${PARMETIS_DIR}/include
  DSUPERLU_INCLUDE = -I${DSUPERLU_DIR}/SRC

  PEXSI_SRC_LIB    =  ${PEXSI_DIR}/lib/libpexsi_v2.0.a 
  DSUPERLU_LIB     =  ${DSUPERLU_DIR}/lib64/libsuperlu_dist.a
  PARMETIS_LIB     =  ${PARMETIS_DIR}/lib/libparmetis.a
  METIS_LIB        =  ${PARMETIS_DIR}/lib/libmetis.a
  PEXSI_LIB        =  ${PEXSI_SRC_LIB} ${DSUPERLU_LIB} ${PARMETIS_LIB} ${METIS_LIB} ${PTSCOTCH_LIB}
endif

# inclues
DGDFT_INCLUDE    = -I${DGDFT_DIR}/include 
LIBXC_INCLUDE    = -I$(LIBXC_DIR)/include
LIBXC_INCLUDE    = -I$(LIBXC_DIR)/include
FFTW_INCLUDE     = -I${FFTW_DIR}/include
YAML_INCLUDE     = -I$(YAML_DIR)/include
INCLUDES         = ${DGDFT_INCLUDE} ${PEXSI_INCLUDE} ${DSUPERLU_INCLUDE} ${PARMETIS_INCLUDE} ${LIBXC_INCLUDE} ${BLOPEX_INCLUDE} ${FFTW_INCLUDE} ${HIPCC_INC} ${YAML_INCLUDE}

# Libraries
LIBXC_LIB        = $(LIBXC_DIR)/lib/libxc.a
FFTW_LIB         = -L${FFTW_DIR}/lib -lfftw3_mpi -lfftw3 -lm
MKL_LIB          = -L${MKL_ROOT}/lib/intel64 -lmkl_core -lmkl_intel_lp64  -lmkl_sequential -lmkl_blacs_intelmpi_lp64  -lmkl_scalapack_lp64 -lpthread -lgfortran
GFORTRAN_LIB     = -lgfortran
YAML_LIB         = $(YAML_DIR)/lib64/libyaml-cpp.a
DGDFT_LIB        = ${DGDFT_DIR}/src/libpwdft.a
 #${DGDFT_DIR}/src/EigenSolver/libeigs.a ${DGDFT_DIR}/src/MathLibs/libmath.a ${DGDFT_DIR}/src/SCF/libscf.a 
RQRCP_LIB        = ${DGDFT_DIR}/external/rqrcp/librqrcp.a
LBFGS_LIB        = ${DGDFT_DIR}/external/lbfgs/liblbfgs.a
LIBS             = ${DGDFT_LIB} ${PEXSI_LIB} ${LIBXC_LIB} ${GFORTRAN_LIB} ${BLOPEX_LIB} ${LBFGS_LIB} ${RQRCP_LIB} ${FFTW_LIB} ${MKL_LIB} ${YAML_LIB} ${IPM} ${HIP_LIB} ${BOOST_LIB} ${PPCG_LIB} -lifcore -lstdc++

CC	= mpiicc
CXX	= mpiicpc
FC	= ifort
LOADER	= mpiicpc

AR           = ar
ARFLAGS      = rvcu
# For System V based machine without ranlib, like Cray and SGI,
# use touch instead.
#RANLIB      = touch
RANLIB       = ranlib

RM           = rm
RMFLAGS      = -f

# Different compiling and linking options.
#

CFLAGS       = ${COMPILE_FLAG} ${OPENMP_FLAG} ${PROFILE_FLAG} ${INCLUDES} ${HIP_FLAG}
FFLAGS       = ${COMPILE_FLAG} ${OPENMP_FLAG} ${PROFILE_FLAG} ${INCLUDES} #${HIP_FLAG}
CXXFLAGS     = ${COMPILE_FLAG} ${OPENMP_FLAG} ${PROFILE_FLAG} ${INCLUDES} ${HIP_FLAG} -std=c++11 -fpermissive
CCDEFS       = ${COMPILE_DEF} ${PEXSI_DEF} ${SCALAR_DEF} ${OPENMP_DEF}
CPPDEFS      = ${COMPILE_DEF} ${PEXSI_DEF} ${SCALAR_DEF} ${OPENMP_DEF}
LOADOPTS     = ${OPENMP_FLAG} ${PROFILE_FLAG} ${LIBS} 


# Compilation replacement rules

%.o: %.c
	${CC} -c ${CFLAGS} ${CCDEFS} $< 
%.o: %.cpp
	${CXX} -c ${CXXFLAGS} ${CPPDEFS} $< 
%.o: %.f
	${FC} -c ${FFLAGS} $<
%.o: %.F
	${FC} -c ${FFLAGS} $<
%.o: %.f90
	${FC} -c ${FFLAGS} $<
%.o: %.F90
	${FC} -c ${FFLAGS} $<
%.o: %.cu
	$(HIPCC) -c -fPIC -fno-gpu-rdc -amdgpu-target=gfx906 -fpermissive ${HIPCC_FLAG} $<

# Generate auto-dependencies (for cpp files now)
%.d: %.cpp
	@set -e; rm -f $@; \
	$(CXX) -M $(CXXFLAGS) $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@;\
	rm -f $@.$$$$
