#/bin/bash

/usr/local/cuda-9.0/bin/nvcc tf_nndistance_2_g.cu -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -c -o tf_nndistance_2_g.cu.o -I /usr/local/lib/python2.7/dist-packages/tensorflow/include -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2

g++ -std=c++11 tf_nndistance_2.cpp tf_nndistance_2_g.cu.o -o tf_nndistance_2_so.so -shared -fPIC -I /usr/local/lib/python2.7/dist-packages/tensorflow/include -I /usr/local/cuda-9.0/include -I /usr/local/lib/python2.7/dist-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-9.0/lib64/ -L/usr/local/lib/python2.7/dist-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
