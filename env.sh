#! /bin/bash

DIST_DIR=$( cd -P -- "$(dirname -- "$0")" && pwd -P )
export DGDFT_DIR=${DIST_DIR}
export LIBXC_DIR=/public/home/xmqin/MathLibs/libxc/6.2.2/intel2020
export FFTW_DIR=/public/home/xmqin/MathLibs/fftw/3.3.10/intel2020
export MKL_ROOT=/public/software/compiler/intel/2020.update4/compilers_and_libraries_2020.4.304/linux/mkl
export YAML_DIR=/public/home/xmqin/MathLibs/yaml-cpp/0.7.0/intel
export PEXSI_DIR=/public/home/xmqin/MathLibs/pexsi/2.0.0/intel
export DSUPERLU_DIR=/public/home/xmqin/MathLibs/superlu_dist/7.2.0/intel
export PARMETIS_DIR=/public/home/xmqin/MathLibs/parmetis/4.0.3/intel
export METIS_DIR=/public/home/xmqin/MathLibs/parmetis/4.0.3/intel

