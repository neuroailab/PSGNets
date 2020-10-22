#/bin/bash
rm tf_nndistance_4_so.so

TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

/usr/local/cuda-10.0/bin/nvcc tf_nndistance_4_g.cu -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -c -o tf_nndistance_4_g.cu.o -I /usr/local/lib/python2.7/dist-packages/tensorflow/include -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2

g++ -std=c++11 tf_nndistance_4.cpp tf_nndistance_4_g.cu.o -o tf_nndistance_4_so.so -shared -fPIC -I /usr/local/cuda-10.0/include -I /usr/local/lib/python2.7/dist-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-10.0/lib64 ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
