cd external/lbfgs
make
cd ../rqrcp
make
cd ../blopex/blopex_abstract
make
cd ../../../src
make -j

#cd EigenSolver
#make cleanall && make -j

#cd ../MathLibs
#make cleanall && make -j

#cd ../RPA
#make cleanall && make -j

#cd ../SCF
#make cleanall && make -j
 
cd ../examples
make cleanall && make pwdft -j #&& make pwdft
