# NOTE: This Makefile does NOT support auto-dependency for the .h files.
# If the header files are changed, do "make clean" first.

include ../../make.inc

SRCS = scf.cpp hamiltonian.cpp periodtable.cpp spinor.cpp fourier.cpp 

OBJS = ${SRCS:.cpp=.o} ${SRCF90:.f90=.o} ${SRCF:.f=.o} 

DEPS = ${SRCS:.cpp=.d} ${SRCF90:.f90=.d} ${SRCF:.f=.d}

all: libscf.a

-include ${DEPS}

libscf.a: ${OBJS}
	$(AR) $(ARFLAGS) $@ ${OBJS}
	$(RANLIB) $@

cleanall:
	${RM} -f ${OBJS} ${DEPS} libscf.a *.d.* *.o *.d
