from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
import numpy as np

from vvn.ops.dimensions import DimensionDict
from vvn.data.utils import * # lots of stuff for reading data
from vvn.ops.rendering import hw_to_xy
from vvn.ops.utils import *

def convnet_stem(
        images, train, ksize,
        strides=2, hidden_ksizes=[3], hidden_separables=False, hidden_channels=[32], out_channels=16, max_pool=True, pool_ksize=2, conv_kwargs={"activation": "swish"}, **kwargs):
    '''
    Performs the first few convolutions/nonlinearities typical of a convnet.
    These operations also usually spatially downsample the inputs by a factor of 2-4 with strides and/or pooling
    '''

    endpoints = {}
    if len(images.shape.as_list()) == 4:
        B,H,W,C = images.shape.as_list()
        T = 0
    elif len(images.shape.as_list()) == 5:
        # apply same conv to all inputs in the time dimension
        B,T,H,W,C = images.shape.as_list()
        images = tf.reshape(images, [B*T,H,W,C])

    if hidden_channels is None:
        hidden_channels = []
    channels = hidden_channels + [out_channels]

    if not isinstance(hidden_separables, (list, tuple)):
        hidden_separables = [hidden_separables] * len(hidden_channels)

    with tf.compat.v1.variable_scope("convnet_stem"):
        # initial conv-bn-relu for example
        with tf.compat.v1.variable_scope("conv0_0"):
            images = conv(images,
                          out_depth=channels[0],
                          ksize=ksize, strides=strides, padding="SAME", train=train,
                          **conv_kwargs)
            if max_pool:
                images = tf.compat.v1.nn.max_pool(images, ksize=[1,pool_ksize,pool_ksize,1], strides=[1,pool_ksize,pool_ksize,1],
                                                  padding="SAME", data_format='NHWC')
            endpoints['conv0'] = tf.identity(images, name="conv0_output")
        # any extra convs
        for L, ocs in enumerate(hidden_channels):
            with tf.compat.v1.variable_scope("conv0_"+str(L+1)):
                conv_op = depth_conv if hidden_separables[L] else conv
                if hidden_separables[L]:
                    assert channels[L+1] == images.shape.as_list()[-1], "depth_conv can't change # of channels"
                images = conv_op(images,
                                 out_depth=channels[L+1],
                                 ksize=hidden_ksizes[L], strides=1, padding="SAME", train=train,
                                 **conv_kwargs)

            if max_pool:
                images = tf.compat.v1.nn.max_pool(images, ksize=[1,pool_ksize,pool_ksize,1], strides=[1,pool_ksize,pool_ksize,1],
                                                  padding="SAME", data_format='NHWC')
            endpoints['conv'+str(L+1)] = tf.identity(images, name="conv"+str(L)+"_output")

    # reshape if there was a time dimension
    if T:
        images = tf.reshape(images, [B,T] + images.shape.as_list()[1:])
        for ep in endpoints.keys():
            endpoints[ep] = tf.reshape(endpoints[ep], [B,T] + endpoints[ep].shape.as_list()[1:])

    return images, endpoints

def build_convrnn_inputs(
        inputs_dict,
        inp_sequence_len,
        ntimes,
        time_dilation,
        num_temporal_splits,
        train=True,
        stem_model_func=None,
        stem_model_func_kwargs={},
        images_key='images',
        hsv_input=False,
        xy_input=False,
        pmat_key='projection_matrix',
        depths_input=False,
        depths_key='depths',
        depth_normalization=100.1,
        background_depth=30.0,
        near_plane=None,
        negative_depths=True,
        normals_from_depths_input=False,
        diff_x_input=False,
        diff_t_input=False,
        normals_input=False,
        unit_normals=True,
        normals_key='normals',
        flows_input=False,
        flows_key='flows',
        hw_input=False,
        ones_input=False,
        objects_mask_input=False,
        objects_mask_key='objects',
        color_normalize=False,
        color_scale=1.0,
        static_mode=False,
        **kwargs
):
    Dims = DimensionDict() # keep track of which channels are in which position
    if not color_normalize:
        inp_ims = tf.cast(inputs_dict[images_key], dtype=tf.float32)
        inp_ims /= color_scale
    elif color_normalize and (inputs_dict[images_key].dtype != tf.float32):
        inp_ims = color_normalize_imnet(inputs_dict[images_key])
    else:
        inp_ims = tf.cast(inputs_dict[images_key], dtype=tf.float32)

    inp_shape = inp_ims.shape.as_list()
    bs = inp_shape[0]
    if len(inp_shape) == 4: # add time dimension
        H,W,C = inp_shape[1:]
        static_mode = True
        T = 1
    elif len(inp_shape) == 5 and static_mode:
        T,H,W,C = inp_shape[1:]
        inp_ims = inp_ims[:,0]
    else:
        T,H,W,C = inp_shape[1:]
    B = bs
    BT = B*T
    inp_ims = Dims.append_attr_to_vector('rgb', inp_ims)

    # read the depths, which are needed for computing xy
    if xy_input or depths_input:
        depths = inputs_dict[depths_key]
        depths = read_depths_image(depths, mask=None, new=True, normalization=depth_normalization, background_depth=background_depth)
        if near_plane is not None:
            depths = tf.maximum(depths, tf.cast(near_plane, tf.float32))
        if negative_depths:
            depths = -1.0 * depths

    # compute the image coordinates
    if xy_input or hw_input:
        hw_ims = coordinate_ims(bs,T,[H,W])
        if static_mode:
            hw_ims = hw_ims[:,0]

    if xy_input:
        pmat = inputs_dict[pmat_key]
        assert pmat.shape.as_list() == [B,T,4,4], "Pmat must have standard format from Unity but is %s" % (pmat.shape.as_list())
        focal_lengths = tf.stack([pmat[:,:,0,0], pmat[:,:,1,1]], axis=-1) # [B,T,2] [fx,fy]
        xy_ims = hw_to_xy(hw_ims, depths, focal_lengths, negative_z=negative_depths, near_plane=near_plane)
        inp_ims = Dims.append_attr_to_vector('xy', xy_ims, inp_ims)

    if depths_input: # depths are stored as 3 channels of uint8; convert to single float32
        inp_ims = Dims.append_attr_to_vector('z', depths, inp_ims)
        if xy_input:
            Dims.delete('xy')
            Dims.delete('z')
            Dims['positions'] = [-3,0]

    if hsv_input: # 3 channels
        rgb = tf.cast(inputs_dict[images_key], tf.float32) / 255.
        hsv = tf.image.rgb_to_hsv(rgb)
        inp_ims = Dims.append_attr_to_vector('hsv', hsv, inp_ims)

    if diff_x_input: # sobel filter input
        diff_x = compute_sobel_features(inputs_dict[images_key], norm=255., normalize_range=True, to_rgb=True, eps=1e-8)
        inp_ims = Dims.append_attr_to_vector('sobel', diff_x, inp_ims)

    if normals_from_depths_input:
        assert depths_input, "Can't make normals from depths if not depths_input"
        if normals_from_depths_scale is None:
            normals_from_depths_scale = 900.0
        raise NotImplementedError("TODO: normals from depths via spatial derivatives")

    if normals_input:
        normals = inputs_dict[normals_key]
        normals = tf.cast(normals, dtype=tf.float32) / 255.0
        if unit_normals:
            normals = normals*2.0 - 1.0
        inp_ims = Dims.append_attr_to_vector('normals', normals, inp_ims)

    if flows_input:
        flows = inputs_dict[flows_key]
        flows = tf.image.rgb_to_hsv(tf.cast(flows, tf.float32) / 255.)
        inp_ims = Dims.append_attr_to_vector('flows', flows, inp_ims)

    if diff_t_input: # forward Euler Ims_{t+1} - Ims_{t}
        diff_t = image_time_derivative(inp_ims)
        inp_ims = tf.concat([inp_ims, diff_t], axis=-1)
        Dims.insert_from(Dims.copy(suffix='_backward_euler'))

    if hw_input:
        inp_ims = Dims.append_attr_to_vector('hw', hw_ims, inp_ims)

    if ones_input:
        ones = tf.ones_like(inp_ims[...,0:1])
        inp_ims = Dims.append_attr_to_vector('visible', ones, inp_ims)

    if objects_mask_input:
        seg_mask = tf.cast(read_background_segmentation_mask_new(inputs_dict[objects_mask_key]), dtype=tf.float32)
        inp_ims = Dims.append_attr_to_vector('foreground', seg_mask, inp_ims)

    # apply a stem function, which may reduce the imput size by a lot
    if stem_model_func is not None:
        inp_ims = stem_model_func(inp_ims, train=train, **stem_model_func_kwargs)

    # trim and dilate temporal input sequence
    if (inp_sequence_len is not None) and not static_mode:
        assert inp_sequence_len <= T
        inp_ims = inp_ims[:, :inp_sequence_len]
        T = inp_sequence_len

    inp_ims_list = input_temporal_preproc(inp_ims=inp_ims, static_mode=static_mode, ntimes=ntimes, seq_length=T, time_dilation=time_dilation, num_temporal_splits=num_temporal_splits)

    print("input channels")
    Dims.sort()
    for k,v in Dims.items():
        print(k,v[:2])
    print(Dims.ndims, "inp_ims", inp_ims.shape.as_list())

    return inp_ims_list, Dims # list in temporal order for input to TNN model

def mlp(inp,
        hidden_dims=[20],
        activations = None,
        kernel_initializer = tf.variance_scaling_initializer(scale=0.001, seed = 0),
        bias_initializer = tf.constant_initializer(0.),
        scope = 'mlp',
        dropout = None,
        seed = 0,
        share_weights = False,
        trainable = True,
        return_vars=False,
        **kwargs):

    # flatten input
    inp_shape = inp.shape.as_list()
    if len(inp_shape) != 2:
        inp = tf.reshape(inp, [-1, inp_shape[-1]])

    # Convert hidden_dims to list
    if not isinstance(hidden_dims, (list, tuple)):
        hidden_dims = [hidden_dims]

    # Make sure there are the same number of activations and feature layers
    if activations is None:
        activations = tf.nn.elu
    if isinstance(activations, type(tf.identity)):
        activations = [activations] * len(hidden_dims)
        activations[-1] = tf.identity

    assert len(hidden_dims) == len(activations), ('One activation per feature layer!')

    weights = []
    biases = []

    for i, (num_feature, activation) in enumerate(zip(hidden_dims, activations)):
        iscope = scope if share_weights else scope + str(i)
        with tf.variable_scope(iscope, reuse=tf.AUTO_REUSE, use_resource=return_vars):
            # Infer kernel shape
            kernel_shape = inp.get_shape().as_list()
            kernel_shape = kernel_shape[:-2] + kernel_shape[-1:] + [num_feature]
            # Initialize kernel variable
            kernel = tf.get_variable(
                    initializer = kernel_initializer,
                    shape = kernel_shape,
                    dtype = inp.dtype,
                    name = 'weights',
                    trainable = trainable,
                use_resource=return_vars
                    )

            # Initialize bias variable
            bias = tf.get_variable(
                    initializer = bias_initializer,
                    shape = [num_feature],
                    dtype = inp.dtype,
                    name = 'bias',
                    trainable = trainable,
                use_resource=return_vars
                    )
            if return_vars:
                weights.append(kernel)
                biases.append(bias)

            # Compute fully connected
            inp = activation(tf.matmul(inp, kernel) + bias)
            if dropout is not None:
                inp = tf.nn.dropout(inp, dropout, seed)

    # restore inp shape
    if len(inp_shape) != 2:
        inp = tf.reshape(inp, inp_shape[:-1] + [-1])

    return inp if not return_vars else (inp, weights, biases)

def conv(inp,
         out_depth,
         ksize=[3,3],
         strides=[1,1,1,1],
         data_format='channels_last',
         padding='SAME',
         kernel_init='xavier',
         kernel_init_kwargs=None,
         use_bias=True,
         bias=0,
         weight_decay=None,
         activation='relu',
         batch_norm=False,
         group_norm=False,
         num_groups=32,
         train=False,
         batch_norm_decay=0.9,
         batch_norm_epsilon=1e-5,
         batch_norm_gamma_init=None,
         init_zero=None,
         dropout=None,
         dropout_seed=0,
         time_sep=False,
         time_suffix=None,
         name='conv',
         **kwargs
):

    # assert out_shape is not None
    if time_sep:
        assert time_suffix is not None

    if batch_norm or group_norm:
        use_bias = False

    if weight_decay is None:
        weight_decay = 0.
    if isinstance(ksize, int):
        ksize = [ksize, ksize]
    if isinstance(strides, int):
        strides = [1, strides, strides, 1]
    if kernel_init_kwargs is None:
        kernel_init_kwargs = {}
    in_depth = inp.get_shape().as_list()[-1]
    if out_depth is None:
        out_depth = in_depth

    # weights
    init = initializer(kernel_init, **kernel_init_kwargs)
    kernel = tf.compat.v1.get_variable(initializer=init,
                            shape=[ksize[0], ksize[1], in_depth, out_depth],
                            dtype=tf.float32,
                            regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                            name='weights')
    print("conv kernel", kernel)

    if use_bias:
        init = initializer(kind='constant', value=bias)
        biases = tf.compat.v1.get_variable(initializer=init,
                            shape=[out_depth],
                            dtype=tf.float32,
                            regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                            name='bias')
    # ops
    if dropout is not None: # dropout is not the dropout_rate, not the keep_porb
        inp = tf.nn.dropout(inp, keep_prob=(1.0-dropout), seed=dropout_seed, name='dropout')

    conv = tf.nn.conv2d(inp, kernel,
                        strides=strides,
                        padding=padding)

    if use_bias:
        output = tf.nn.bias_add(conv, biases, name=name)
    else:
        output = tf.identity(conv, name=name)

    if batch_norm:
        output = batchnorm_corr(inputs=output,
                                train=train,
                                data_format=data_format,
                                decay = batch_norm_decay,
                                epsilon = batch_norm_epsilon,
                                constant_init=batch_norm_gamma_init,
                                init_zero=init_zero,
                                activation=activation,
                                time_suffix=time_suffix)
    elif group_norm:
        output = groupnorm(inputs=output,
                           G=num_groups,
                           data_format=data_format,
                           weight_decay=weight_decay,
                           gamma_init=(0.0 if init_zero else 1.0),
                           epsilon=batch_norm_epsilon)


    # if activation is not None:
    #     output = getattr(tf.nn, activation)(output, name=activation)

    if activation is not None:
        if activation == 'relu':
            output = tf.nn.relu(output)
        elif activation == 'elu':
            output = tf.nn.elu(output)
        elif activation == 'swish':
            output = tf.nn.swish(output)
        elif activation == 'sigmoid':
            output = tf.nn.sigmoid(output)
        else:
            raise NotImplementedError("activation is %s" % activation)

    return output

def depth_conv(inp,
               multiplier=1,
               out_depth=None,
               ksize=3,
               strides=1,
             padding='SAME',
             kernel_init='xavier',
             kernel_init_kwargs=None,
             activation='relu6',
             weight_decay=None,
             batch_norm = False,
               group_norm=False,
               num_groups=32,
               use_bias=False,
             train=True,
             batch_norm_decay=0.9,
             batch_norm_epsilon=1e-5,
               batch_norm_gamma_init=None,
             init_zero=None,
             data_format='channels_last',
             time_sep=False,
             time_suffix=None,
             name='depth_conv'
             ):

    # assert out_shape is not None

    if time_sep:
        assert time_suffix is not None

    if weight_decay is None:
        weight_decay = 0.
    if isinstance(ksize, int):
        ksize = [ksize, ksize]
    if isinstance(strides, int):
        strides = [1, strides, strides, 1]

    if kernel_init_kwargs is None:
        kernel_init_kwargs = {}

    in_depth = inp.get_shape().as_list()[-1]

    out_depth = multiplier * in_depth

    # weights
    init = initializer(kernel_init, **kernel_init_kwargs)
    kernel = tf.compat.v1.get_variable(initializer=init,
                            shape=[ksize[0], ksize[1], in_depth, multiplier],
                            dtype=tf.float32,
                            regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                            name='weights')

    output = tf.nn.depthwise_conv2d(inp, kernel,
                            strides=strides,
                            padding=padding)

    if batch_norm:
        output = batchnorm_corr(inputs=output,
                                train=train,
                                data_format=data_format,
                                decay = batch_norm_decay,
                                epsilon = batch_norm_epsilon,
                                constant_init=batch_norm_gamma_init,
                                init_zero=init_zero,
                                activation=activation,
                                time_suffix=time_suffix)
    elif group_norm:
        output = groupnorm(inputs=output,
                           G=num_groups,
                           data_format=data_format,
                           weight_decay=weight_decay,
                           gamma_init=(0.0 if init_zero else 1.0),
                           epsilon=batch_norm_epsilon)

    elif use_bias:
        init = initializer(kind='constant', value=1.0)
        biases = tf.compat.v1.get_variable(initializer=init,
                                shape=[out_depth],
                                dtype=tf.float32,
                                regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                name='bias')
        output = tf.nn.bias_add(output, biases, name=name)

    # if activation is not None:
    #     output = getattr(tf.nn, activation)(output, name=activation)

    if activation is not None:
        if activation == 'relu':
            output = tf.nn.relu(output)
        elif activation == 'elu':
            output = tf.nn.elu(output)
        elif activation == 'swish':
            output = tf.nn.swish(output)
        elif activation == 'sigmoid':
            output = tf.nn.sigmoid(output)
        else:
            raise NotImplementedError("activation is %s" % activation)

    return output

def fc(inputs,
       out_depth,
       train=True,
       kernel_init='xavier',
       kernel_init_kwargs={'seed':0},
       use_bias=True,
       bias=1,
       weight_decay=None,
       activation='relu',
       batch_norm=False,
       batch_norm_decay=0.9,
       batch_norm_epsilon=1e-5,
       init_zero=None,
       dropout=None,
       dropout_seed=0,
       time_sep=False,
       time_suffix=None,
       scope='fc', **kwargs):

    if batch_norm:
        use_bias = False

    if weight_decay is None:
        weight_decay = 0.
    # assert out_shape is not None
    if kernel_init_kwargs is None:
        kernel_init_kwargs = {}

    resh = tf.reshape(inputs, [inputs.get_shape().as_list()[0], -1], name='reshape')
    in_depth = resh.get_shape().as_list()[-1]

    # weights
    init = initializer(kernel_init, **kernel_init_kwargs)
    with tf.compat.v1.variable_scope(scope):
        kernel = tf.compat.v1.get_variable(initializer=init,
                                shape=[in_depth, out_depth],
                                dtype=tf.float32,
                                regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                name='weights')

        if use_bias:
            init = initializer(kind='constant', value=bias)
            biases = tf.compat.v1.get_variable(initializer=init,
                                shape=[out_depth],
                                dtype=tf.float32,
                                regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                name='bias')

    # ops
    if dropout is not None: # dropout is not the dropout rate, not the keep_prob
        resh = tf.nn.dropout(resh, keep_prob=(1.0-dropout), seed=dropout_seed, name='dropout')
    fcm = tf.matmul(resh, kernel)

    if use_bias:
        output = tf.nn.bias_add(fcm, biases, name=scope)
    else:
        output = tf.identity(fcm, name=scope)

    # if activation is not None:
    #     output = getattr(tf.nn, activation)(output, name=activation)
    if activation is not None:
        if activation == 'relu':
            output = tf.nn.relu(output)
        elif activation == 'elu':
            output = tf.nn.elu(output)
        elif activation == 'swish':
            output = tf.nn.swish(output)
        elif activation == 'sigmoid':
            output = tf.nn.sigmoid(output)
        else:
            raise NotImplementedError("activation is %s" % activation)

    if batch_norm:
        # if activation is none, should use zeros; else ones
        if init_zero is None:
            init_zero = True if activation is None else False
        if init_zero:
            gamma_init = tf.zeros_initializer()
        else:
            gamma_init = tf.ones_initializer()

        if time_suffix is not None:
            bn_op_name = "post_conv_BN_" + time_suffix
            reuse_flag = tf.AUTO_REUSE # create bn variables per timestep if they do not exist
        else:
            bn_op_name = "post_conv_BN"
            reuse_flag = None

        with tf.compat.v1.variable_scope(scope):
            output = tf.layers.batch_normalization(inputs=output,
                                                   axis=-1,
                                                   momentum=batch_norm_decay,
                                                   epsilon=batch_norm_epsilon,
                                                   center=True,
                                                   scale=True,
                                                   training=train,
                                                   trainable=True,
                                                   fused=True,
                                                   gamma_initializer=gamma_init,
                                                   name=bn_op_name,
                                                   reuse=reuse_flag)
    return output

def temporal_fc(
        inputs, out_depth, train=True,
        bottleneck_depth=None,
        **kwargs
):

    shape = inputs.shape.as_list()
    if len(shape) == 3:
        B,T,N = inputs.shape.as_list()
    else:
        assert len(shape) == 5, shape
        B,T,H,W,C = shape
        N = H*W*C

    inp = tf.reshape(inputs, [B,T*N])
    with tf.variable_scope('temporal_bottleneck_fc'):
        inp = fc(inp, out_depth=(bottleneck_depth or N), train=train, **kwargs)
    with tf.variable_scope('final_fc'):
        fc_kwargs = copy.deepcopy(kwargs)
        fc_kwargs['activation'] = None
        inp = fc(inp, out_depth=out_depth, train=train, **fc_kwargs)

    return inp



def global_pool(inputs, kind='avg', keep_dims=False, name=None, **kwargs):
    if kind not in ['max', 'avg']:
        raise ValueError('Only global avg or max pool is allowed, but'
                            'you requested {}.'.format(kind))
    if name is None:
        name = 'global_{}_pool'.format(kind)
    h, w = inputs.get_shape().as_list()[1:3]
    out = getattr(tf.nn, kind + '_pool2d')(inputs,
                                    ksize=[1,h,w,1],
                                    strides=[1,1,1,1],
                                    padding='VALID')
    if keep_dims:
        output = tf.identity(out, name=name)
    else:
        output = tf.reshape(out, [out.get_shape().as_list()[0], -1], name=name)

    return output

def shared_spatial_mlp(inp,
                        out_depth,
                        scope="shared_spatial_mlp",
                        hidden_dims=[],
                        bias=0.0,
                        activation=tf.nn.elu,
                        kernel_initializer='xavier',
                        kernel_initializer_kwargs=None):
    '''
    Applies same mlp to every every spatial feature
    '''

    if kernel_initializer_kwargs is None:
        kernel_initializer_kwargs = {}
    try:
        kernel_init = tfutils.model.initializer(kind=kernel_initializer, **kernel_initializer_kwargs)
    except:
        kernel_init = tfutils.model_tool_old.initializer(kind=kernel_initializer, **kernel_initializer_kwargs)
    bias_init = tf.constant_initializer(value=bias)

    if activation is None:
        activation = tf.identity

    assert len(inp.shape.as_list()) == 3
    B,N,F = inp.shape.as_list()
    output = tf.reshape(inp, [B*N, F], name="reshape")
    input_dim = output.shape.as_list()[1]
    mlp_dims = hidden_dims + [out_depth]

    with tf.compat.v1.variable_scope(scope):
        dim_now = input_dim
        num_layers = len(mlp_dims)
        for i, hidden_dim in enumerate(mlp_dims):
            kernel = tf.compat.v1.get_variable(initializer=kernel_init,
                                     shape=[dim_now, hidden_dim],
                                     dtype=tf.float32,
                                     name=("layer_"+str(i+1)+"_weights"))

            biases = tf.compat.v1.get_variable(initializer=bias_init,
                                     shape=[hidden_dim],
                                     dtype=tf.float32,
                                     name=("layer_"+str(i+1)+"_bias"))

            # ops
            output = tf.matmul(output, kernel)
            output = tf.nn.bias_add(output, biases)
            if i+1 != num_layers:
                output = activation(output, name=("layer_"+str(i+1)+"_output"))
                dim_now = output.shape.as_list()[1]
            else:
                output = tf.identity(output, name="mlp_output_batch")

    output = tf.reshape(output, [B,N,out_depth], name="mlp_output")

    return output


def groupnorm(inputs,
              G=32,
              data_format='channels_last',
              weight_decay=0.0,
              epsilon=1e-5,
              training=True,
              gamma_init=1,
              beta_init=0):
    '''
    Like LayerNorm, z-scores features along the channel dimension only.
    However, it only normalizes within G groups of C/G channels each.
    Optionally applies learnable scale/shift parameters.
    '''
    assert len(inputs.shape.as_list()) == 4, "Applies only to conv2D layers"
    if data_format == 'channels_first':
        inputs = tf.transpose(inputs, [0,2,3,1])
    elif data_format == 'channels_last':
        pass
    else:
        raise ValueError("data_format must be 'channels_first' or 'channels_last'")

    B,H,W,C = inputs.shape.as_list()
    assert C % G == 0, "num groups G must divide C"
    CpG = C // G

    inputs = tf.reshape(inputs, [B,H,W,CpG,G])
    mean, var = tf.nn.moments(inputs, axes=[1,2,3], keep_dims=True)
    inputs = tf.div(inputs - mean, tf.sqrt(var + epsilon))
    inputs = tf.reshape(inputs, [B,H,W,C])

    if training:
        gamma = tf.get_variable("groupnorm_scale", shape=[1,1,1,C], dtype=tf.float32,
                                initializer=initializer("constant", float(gamma_init)))
                                # regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
        beta = tf.get_variable("groupnorm_shift", shape=[1,1,1,C], dtype=tf.float32,
                               initializer=initializer("constant", float(beta_init)))
                               # regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
    else:
        gamma = tf.constant(gamma_init, dtype=tf.float32)
        beta = tf.constant(beta_init, dtype=tf.float32)

    inputs = gamma*inputs + beta
    if data_format == 'channels_first':
        inputs = tf.transpose(inputs, [0,3,1,2])

    return inputs
