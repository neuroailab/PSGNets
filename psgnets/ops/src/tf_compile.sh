#/bin/bash

rm ./tf_connected_components.so
rm ./tf_labelprop.so
rm ./tf_labelprop_fc.so

TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
echo $TF_CFLAGS
echo $TF_LFLAGS
g++ -std=c++11 -shared graphs.cc tf_connected_components.cc -o tf_connected_components.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
g++ -std=c++11 -shared graphs.cc tf_labelprop.cc -o tf_labelprop.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
g++ -std=c++11 -shared graphs.cc tf_labelprop_fc.cc -o tf_labelprop_fc.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2

