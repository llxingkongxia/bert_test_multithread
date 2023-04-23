#!/bin/bash

workdir=`pwd`
hygon_build_dir="${workdir}/build"
processor_num=`cat /proc/cpuinfo | grep processor | grep -v grep | wc -l`

#options="-DProtobuf_LIBRARIES=/home/liulei/miniconda2/envs/test/lib -DProtobuf_INCLUDE_DIR=/home/liulei/miniconda2/envs/test/include -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${hygon_build_dir}/install $*"
options="-DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${hygon_build_dir}/install $*"

#export CC=/tools/speccpu/compiler/gcc-8.2.0-static/bin/gcc
#export CXX=/tools/speccpu/compiler/gcc-8.2.0-static/bin/g++

#export CC=/tools/speccpu/compiler/gcc-10.2.0_install/bin/gcc
#export CXX=/tools/speccpu/compiler/gcc-10.2.0_install/bin/g++

mkdir ${hygon_build_dir}
cd ${hygon_build_dir}
cmd="cmake $options .. && make -j${processor_num} "
echo "cmd -> $cmd"
eval "$cmd"
