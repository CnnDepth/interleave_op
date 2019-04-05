#!/bin/bash

TF_CFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

nvcc -std=c++11 -c -o interleave_kernels.cu.o interleave_kernels.cu.cc ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -DNDEBUG -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -I/usr/local/
g++ -std=c++11 -shared -o interleave_kernels.so interleave_kernels.cc interleave_ops.cc interleave_kernels.cu.o ${TF_CFLAGS[@]} -fPIC -L/usr/local/cuda/lib64 -lcudart ${TF_LFLAGS[@]}
