# NOTE: This Makefile does NOT support auto-dependency for the .h files.
# If the header files are changed, do "make clean" first.

include ../../make.inc

SRCS = blas.cpp lapack.cpp mpi_interf.cpp scalapack.cpp

OBJS = ${SRCS:.cpp=.o} ${SRCF90:.f90=.o} ${SRCF:.f=.o} 

DEPS = ${SRCS:.cpp=.d} ${SRCF90:.f90=.d} ${SRCF:.f=.d}

all: libmath.a

-include ${DEPS}

libmath.a: ${OBJS}
	$(AR) $(ARFLAGS) $@ ${OBJS}
	$(RANLIB) $@

cleanall:
	${RM} -f ${OBJS} ${DEPS} libmath.a *.d.* *.o *.d
