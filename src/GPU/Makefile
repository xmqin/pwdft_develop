# NOTE: This Makefile does NOT support auto-dependency for the .h files.
# If the header files are changed, do "make clean" first.

include ../../make.inc

SRCS = eigensolver_gpu.cpp  fourier_gpu.cpp  hamiltonian_gpu.cpp spinor_gpu.cpp 

#SRCS += cublas.cpp cuda_utils.cu cusolver.cpp cuda_errors.cpp cufft.cu magma.cpp

OBJS = ${SRCS:.cpp=.o} ${SRCF90:.f90=.o} ${SRCF:.f=.o} 

DEPS = ${SRCS:.cpp=.d} ${SRCF90:.f90=.d} ${SRCF:.f=.d}

all: libgpu.a

-include ${DEPS}

libgpu.a: ${OBJS}
	$(AR) $(ARFLAGS) $@ ${OBJS}
	$(RANLIB) $@

cleanall:
	${RM} -f ${OBJS} ${DEPS} libgpu.a *.d.* *.o *.d
