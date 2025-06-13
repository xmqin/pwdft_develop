#! /bin/bash

source ./env.sh
echo ${DGDFT_DIR}

cd  ${DGDFT_DIR}/external/lbfgs
make cleanall && make
cd  ${DGDFT_DIR}/external/rqrcp
make cleanall && make
cd  ${DGDFT_DIR}/external/blopex/blopex_abstract
make clean && make
cd  ${DGDFT_DIR}/src
make cleanall && make -j

#cd EigenSolver
#make cleanall && make -j

#cd ../MathLibs
#make cleanall && make -j

#cd ../RPA
#make cleanall && make -j

#cd ../SCF
#make cleanall && make -j
 
cd  ${DGDFT_DIR}/examples
make cleanall && make pwdft -j #&& make pwdft
