# NOTE: This Makefile does NOT support auto-dependency for the .h files.
# If the header files are changed, do "make clean" first.

# FIXME How to write Makefile when the compiled files are in different directory
# I have tried many methods but all fails
# a bad method is taken as following 

include ../make.inc

SRCS = esdf.cpp global.cpp utility.cpp
SRCS += EigenSolver/davidson.cpp EigenSolver/eigensolver.cpp EigenSolver/lobpcg.cpp EigenSolver/ppcg.cpp
SRCS += MathLibs/blas.cpp MathLibs/lapack.cpp MathLibs/mpi_interf.cpp MathLibs/scalapack.cpp
SRCS += MD/iondynamics.cpp 
SRCS += SCF/scf.cpp SCF/hamiltonian.cpp SCF/periodtable.cpp SCF/spinor.cpp SCF/fourier.cpp
ifeq (${USE_COMPLEX}, 1)
SRCS += SCF/hamiltonian_complex.cpp SCF/spinor_complex.cpp EigenSolver/eigensolver_complex.cpp EigenSolver/davidson_complex.cpp \
        EigenSolver/lobpcg_complex.cpp EigenSolver/ppcg_complex.cpp  TDDFT/tddft.cpp TDDFT/sgmres.cpp 
else
SRCS += TDDFT/lrtddft.cpp
endif

SRCS1 = esdf.cpp global.cpp utility.cpp
SRCS1 += davidson.cpp eigensolver.cpp lobpcg.cpp ppcg.cpp
SRCS1 += blas.cpp lapack.cpp mpi_interf.cpp scalapack.cpp
SRCS1 += iondynamics.cpp
SRCS1 += scf.cpp hamiltonian.cpp periodtable.cpp spinor.cpp fourier.cpp
ifeq (${USE_COMPLEX}, 1)
SRCS1 += hamiltonian_complex.cpp spinor_complex.cpp eigensolver_complex.cpp davidson_complex.cpp \
         lobpcg_complex.cpp ppcg_complex.cpp tddft.cpp sgmres.cpp
else
SRCS1 += lrtddft.cpp
endif

OBJS = ${SRCS:.cpp=.o} ${SRCF90:.f90=.o} ${SRCF:.f=.o} 

OBJS1 = ${SRCS1:.cpp=.o}

DEPS = ${SRCS:.cpp=.d} ${SRCF90:.f90=.d} ${SRCF:.f=.d}

all: libpwdft.a

-include ${DEPS}

libpwdft.a: ${OBJS}
	$(AR) $(ARFLAGS) $@ ${OBJS1}
	$(RANLIB) $@

libeigs: 
	cd EigenSolver ; make cleanall; make all -j ; cd ..

libmath: 
	cd MathLibs ; make cleanall; make all -j ; cd ..

libscf: 
	cd SCF ; make cleanall; make all -j ; cd ..

libmd:
	cd MD ; make cleanall; make all -j ; cd ..

cleanall:
	${RM} -f ${OBJS} ${DEPS} libpwdft.a */*.d.* */*.a *.d.* *.o *.d */*.d
