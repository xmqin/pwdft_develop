# NOTE: This Makefile does NOT support auto-dependency for the .h files.
# If the header files are changed, do "make clean" first.

include ../make.inc

SRCS   = global.cpp blas.cpp eigensolver.cpp fourier.cpp hamiltonian.cpp lapack.cpp spinor.cpp utility.cpp interf_pwdft.cpp

OBJS = ${SRCS:.cpp=.o} 
DEPS = ${SRCS:.cpp=.d} 

all: libppcg.a

-include ${DEPS}

libppcg.a: ${OBJS}
	$(AR) $(ARFLAGS) $@ ${OBJS}
	$(RANLIB) $@

cleanall:
	${RM} -f ${OBJS} ${DEPS} libppcg.a *.d.*
