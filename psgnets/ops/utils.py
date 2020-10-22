from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from psgnets.ops.dimensions import DimensionDict

hung = tf.load_op_library('../ops/src/hungarian.so')

def initializer(kind='xavier', *args, **kwargs):
    if kind == 'xavier':
        init = tf.contrib.layers.xavier_initializer(*args, **kwargs)
    elif kind == 'normal':
        init = normal_initializer
    else:
        init = getattr(tf, kind + '_initializer')(*args, **kwargs)

    return init

def normal_initializer(shape, dtype=None, partition_info=None):
    '''
    Used for EfficientNets
    '''
    H, W, _, C_out = shape
    fan_out = int(H * W * C_out)
    return tf.random_normal(
        shape, mean=0.0, stddev=np.sqrt(2.0 / fan_out), dtype=dtype)

def mask_tensor(t, mask, mask_value=0.0, dtype=tf.float32):
    t_dtype = t.dtype
    mask_dtype = mask.dtype
    assert (t_dtype == dtype) and (mask_dtype == dtype)
    if not isinstance(mask_value, tf.Tensor):
        mask_value = tf.constant(mask_value, dtype)
    return t*mask + mask_value*tf.cast(1-mask, dtype=dtype)

def relative_attr_dists2(particles, mask=None, mask_value=100.):
    deltas = tf.expand_dims(particles, axis=-2) - tf.expand_dims(particles, axis=-3)
    dists2 = tf.reduce_sum(tf.square(deltas), axis=-1, keepdims=False) # [B,T,N,N]

    if mask is not None:
        dists2 = mask_tensor(dists2, tf.expand_dims(mask[...,0], axis=-1)*tf.expand_dims(mask[...,0], axis=-2), mask_value)

    return dists2

def inversion_map(k, p, eps=0.0):
    return lambda t: tf.div(1., 1. + tf.pow(k*(t+eps), p))

def get_monotonic_segment_ids(segment_ids, valid_nodes):
    '''
    segment_ids: [B,H,W] <tf.int32> output of 2D labelprop
    valid_nodes: [B,N] <tf.bool> indicating which nodes are valid out of max N=num_nodes

    returns:
    valid_segment_ids: [B,H,W] <tf.int32> segment_ids relabeled so that id [b,i] < N*(b+1)
    '''
    shape = segment_ids.shape.as_list()
    valid_nodes = tf.reshape(tf.cast(valid_nodes, tf.int32), [shape[0], -1])
    N = valid_nodes.shape.as_list()[-1]
    num_valid_nodes = tf.reduce_sum(tf.cast(valid_nodes, tf.int32), axis=1) # [B]
    Ntotal = tf.reduce_sum(num_valid_nodes, keepdims=True)
    segment_ids -= tf.reduce_min(segment_ids, axis=[1,2], keepdims=True)
    segment_ids = tf.where(segment_ids < num_valid_nodes[:,tf.newaxis,tf.newaxis], segment_ids, Ntotal*tf.ones_like(segment_ids))
    segment_ids += tf.cumsum(num_valid_nodes, exclusive=True)[:,tf.newaxis,tf.newaxis]

    return segment_ids

def num_segments_from_segment_ids(segment_ids, num_segments=None):
    '''
    computes the number of segments in each example and
    returns segment_ids that increase monotically across examples with no overlap
    '''
    BT,HW,_,rank = dims_and_rank(segment_ids)
    if num_segments is not None:
        assert num_segments.shape.as_list() == [BT], "num segments must be shape [num_examples] but is %s" % num_segments.shape
    else:
        # assert rank == 4, "Must pass in a tensor of shape [B,T,H,W] but is %s" % segment_ids.shape.as_list()
        num_segments = tf.reduce_max(segment_ids, axis=[ax for ax in range(1,rank)]) -\
                       tf.reduce_min(segment_ids, axis=[ax for ax in range(1,rank)]) +\
                       tf.cast(1, tf.int32)
        B = num_segments.shape.as_list()[0]
        T = BT // B
        num_segments = tf.reshape(tf.tile(num_segments[:,tf.newaxis], [1,T]), [-1]) if BT > B else num_segments

    segment_ids = segment_ids - tf.reduce_min(segment_ids, axis=[ax for ax in range(1,rank)], keepdims=True)
    segment_ids = tf.reshape(segment_ids, [BT,HW])
    segments_monotonic = segment_ids + tf.cumsum(num_segments[:,tf.newaxis], axis=0, exclusive=True)

    return num_segments, segments_monotonic

def labels_list_to_parent_inds(labels_list, num_parent_segments, valid_child_nodes,
                               labels_monotonic=False, max_parent_nodes=64):
    '''
    Convert a list of labels shape [R = sum(num_valid_child_nodes)] to a
    rectangular indices tensor of shape valid_child_nodes = [B,N]

    inputs
    labels_list: [R] <tf.int32> assigning each of the valid child nodes to a parent
    num_parent_segments: [B] <tf.int32> how many parents there are per example
    valid_child_nodes: [B,N] <tf.int32> which of the child nodes are valid
    max_parent_nodes: <int> which will determine the maximum value of the returned indices tensor
    '''
    B,N = valid_child_nodes.shape.as_list()
    M = max_parent_nodes
    R = tf.reduce_sum(tf.cast(valid_child_nodes > 0.5, tf.int32))
    num_valid_child_nodes = tf.reduce_sum(tf.cast(valid_child_nodes > 0.5, tf.int32), axis=1) # [B]
    if labels_monotonic:
        offsets = tf.gather_nd(
            params=tf.cumsum(num_parent_segments, exclusive=True),
            indices=build_batch_inds(num_valid_child_nodes)[:,tf.newaxis]) # [R]
        labels_list = labels_list - offsets # [R]
    labels_list = tf.where(labels_list < M, labels_list, tf.ones([R], dtype=tf.int32)*(M-1))
    valid_inds = tf.where(valid_child_nodes > 0.5) # [R,2]
    parent_inds = tf.scatter_nd(valid_inds, labels_list, shape=[B,N])
    b_inds = tf.tile(tf.range(B, dtype=tf.int32)[:,tf.newaxis], [1,N])
    parent_inds = tf.stack([b_inds, parent_inds], axis=-1) # [B,N,2]
    return parent_inds

def parent_inds_to_labels_list(parent_inds, valid_children, num_parent_segments, monotonic=True):
    '''
    Converts a [B,N_children] tensor of parent indices to a [sum(num_valid_children)] tensor of monotonically increasing ids

    inputs
    parent_inds: [B,N_children] <tf.int32>
    valid_chilren: [B,N_children] which children are valid
    num_parent_segments: [B] number of parents per example

    returns
    labels_list: [sum(num_valid_children)] <tf.int32> tensor with the parent inds increasing monotonically by batch
    '''
    valid_children = valid_children > 0.5
    valid_inds = tf.where(valid_children) # [num_valid, 2]
    if monotonic:
        labels_list = tf.gather_nd(parent_inds + num_parent_segments[:,tf.newaxis], valid_inds)
    else:
        labels_list = tf.gather_nd(parent_inds, valid_inds)

    return labels_list

def within_example_segment_ids(num_segments, Nmax):
    '''
    inputs
    num_segments: [B] <tf.int32> number of segments in each example
    Nmax: int, max number of segments per example.

    outputs
    segment_ids: [Ntotal] <tf.int32> in range [0,Nmax)
    '''
    Ntotal = tf.reduce_sum(num_segments)
    cumu_num_segments = tf.cumsum(num_segments, exclusive=False)
    offsets = tf.cumsum(
        tf.scatter_nd(
            cumu_num_segments[:-1,tf.newaxis],
            num_segments[:-1], shape=[Ntotal]),
        exclusive=False)

    seg_ids = tf.range(Ntotal, dtype=tf.int32) - offsets
    seg_ids = tf.where(seg_ids < Nmax,
                       seg_ids,
                       tf.ones(Ntotal, tf.int32)*(Nmax-1)
    )

    return seg_ids

def build_batch_inds(num_segments):
    B = num_segments.shape.as_list()[0]
    Ntotal = tf.reduce_sum(num_segments)
    cumu_ns = tf.cumsum(num_segments, exclusive=False) # [B]
    boundaries = tf.scatter_nd(
        cumu_ns[:-1,tf.newaxis], tf.ones([B-1], tf.int32), shape=[Ntotal])
    b_inds = tf.cumsum(boundaries, exclusive=False) # [Ntotal]
    return b_inds

def inds_from_num_segments(num_segments, max_labels):
    b_inds = build_batch_inds(num_segments)
    n_inds = within_example_segment_ids(num_segments, max_labels)
    return b_inds, n_inds

def dims_and_rank(tensor):
    shape = tensor.shape.as_list()
    C = None
    if len(shape) == 2:
        BT,HW = shape
    elif len(shape) == 3:
        BT,H,W = shape
        HW = H*W
    elif len(shape) == 4:
        B,T,H,W = shape
        HW = H*W
        BT = B*T
    elif len(shape) == 5:
        B,T,H,W,C = shape
        HW = H*W
        BT = B*T
    else:
        raise ArgumentError()

    return BT, HW, C, len(shape)

def inds_image(batch_size, seq_len, imsize, to_float=False):
    ones = tf.ones([batch_size, seq_len] + imsize, dtype=tf.int32)
    H,W = imsize
    h = tf.reshape(tf.range(H, dtype=tf.int32), [1,1,H,1]) * ones
    w = tf.reshape(tf.range(W, dtype=tf.int32), [1,1,1,W]) * ones
    inds = tf.stack([h,w], axis=-1)
    return inds if not to_float else tf.cast(inds, tf.float32)

def coordinate_ims(batch_size, seq_len, imsize):
    bs = batch_size
    T = seq_len
    H,W = imsize
    ones = tf.ones([bs,H,W,1], dtype=tf.float32)
    h = tf.reshape(tf.math.divide(tf.range(H, dtype=tf.float32), tf.cast(H-1, dtype=tf.float32) / 2.0),
                   [1,H,1,1]) * ones
    h -= 1.0
    w = tf.reshape(tf.math.divide(tf.range(W, dtype=tf.float32), tf.cast(W-1, dtype=tf.float32) / 2.0),
                   [1,1,W,1]) * ones
    w -= 1.0
    h = tf.stack([h]*T, axis=1)
    w = tf.stack([w]*T, axis=1)
    hw_ims = tf.concat([h,w], axis=-1)
    return hw_ims

def deriv_kernels():
    h = np.array([[1.,2.,1.,]])
    d = np.array([[1.,0.,-1.]])
    hu = np.dot(d.T, h)
    hv = np.dot(h.T, d)
    L = np.array([[0.5, 1.0, 0.5],
                  [1.0, -6.0, 1.0],
                  [0.5, 1.0, 0.5]])
    return np.stack([hu, hv, L], axis=2)

def rotate_vectors_with_quaternions(quaternions, vectors):
    '''
    vectors: [B,...,N,3]
    unit_quaternions: [B,...,4] with dimensions matching those of vectors
    '''
    unit_quaternions = tf.nn.l2_normalize(quaternions, axis=-1)
    vectors_shape = vectors.shape.as_list()
    base_shape = vectors_shape[:-2]
    N = vectors_shape[-2]
    if len(unit_quaternions.shape.as_list()) != len(vectors_shape):
        unit_quaternions = tf.expand_dims(unit_quaternions, axis=len(base_shape)) # now [B,...,1,4]
    Q = unit_quaternions.shape.as_list()[-2]
    assert Q in [1,N], (unit_quaternions.shape.as_list(), vectors_shape, Q, N)

    # apply quaternion rotation formula:
    #
    # v_new = v_old + 2*q x (q x v_old + w*v_old)
    #
    # where w is the scalar and q the vector part of the unit quaternion

    # separate w and q into [B,...,1,1/3] components
    w,q = tf.split(unit_quaternions, [1,3], axis=-1)

    # tile for cross products
    if Q != N:
        q = tf.tile(q, ([1]*len(base_shape)) + [N,1]) # [B,...,N,3]
    assert q.shape.as_list() == vectors_shape, (q.shape.as_list(), vectors.shape.as_list(), vectors.name)

    delta_v = tf.cross(2.0*q, tf.cross(q, vectors) + w*vectors)
    return vectors + delta_v

def preproc_hsv(hsv, circular=False):
    h,sv = tf.split(hsv, [1,2], axis=-1)
    if circular:
        h = tf.floormod(h, tf.constant(1.0, tf.float32)) # circular in [0,1]
    sv = tf.nn.sigmoid(sv) # squashed monotonic in [0,1]
    return tf.concat([h,sv], axis=-1)

def input_temporal_preproc(inp_ims, static_mode, ntimes, seq_length, time_dilation, num_temporal_splits):
    if static_mode:
        if len(inp_ims.shape.as_list()) == 5:
            assert inp_ims.shape.as_list()[1] == 1
            inp_ims = [inp_ims[:,0]]*ntimes
        elif len(inp_ims.shape.as_list()) == 4:
            inp_ims = [inp_ims]*ntimes
        else:
            raise ValueError("If static_mode, inputs must either have shape [B,1,H,W,C] or [B,H,W,C]")

    else:
        inp_shape = inp_ims.shape.as_list()
        B,T = inp_shape[:2]
        assert seq_length == T
        assert ntimes == (seq_length * time_dilation) / num_temporal_splits
        assert seq_length % num_temporal_splits == 0, "Must split sequence into equal pieces"

        if num_temporal_splits > 1:
            inp_ims = tf.reshape(inp_ims, [B*num_temporal_splits, -1] + inp_shape[2:])
        if time_dilation > 1:
            inp_ims = dilate_tensor(inp_ims, dilation_factor=time_dilation, axis=1) # [B,seq_len*time_dilation,...]

        # make list of length num_temporal_splits, each [B,split_length,...]
        inp_ims = tf.split(inp_ims, num_or_size_splits=inp_ims.shape.as_list()[1], axis=1)
        inp_ims = inp_ims[:ntimes]
        inp_ims = [tf.squeeze(im, axis=[1]) for im in inp_ims]

    return inp_ims

def color_normalize_imnet(image):
    print("color normalizing")
    image = tf.cast(image, tf.float32) / 255.0
    imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - imagenet_mean) / imagenet_std
    return image


def image_time_derivative(ims, norm=255.):
    if ims.dtype != tf.float32:
        ims = tf.cast(ims, tf.float32) / 255.

    shape = ims.shape.as_list()
    if shape[1] == 1:
        return tf.zeros(shape, dtype=tf.float32)
    else:
        shape[1] = 1
        diff_t = ims[:,1:] - ims[:,:-1]
        diff_t = tf.concat([tf.zeros(shape=shape, dtype=tf.float32), diff_t], axis=1)
        return diff_t

def preproc_segment_ids(segment_ids, Nmax, return_valid_segments=False):
    assert len(segment_ids.shape.as_list()) in [3,4], "Segment ids must be shape [B,[T],H,W]"
    assert segment_ids.dtype == tf.int32, "segment ids must be dtype tf.int32"
    segment_ids = segment_ids - tf.reduce_min(segment_ids, axis=[-2,-1], keepdims=True) # now they start at 0 per example
    valid_segments = tf.cast(segment_ids < Nmax, tf.float32)
    segment_ids = tf.where(segment_ids < Nmax,
                           segment_ids,
                           tf.ones_like(segment_ids)*(Nmax-1))
    if return_valid_segments:
        num_valid_px = tf.maximum(tf.reduce_sum(valid_segments, axis=[-2,-1]), 1.0)
        return segment_ids, valid_segments, num_valid_px
    else:
        return segment_ids

def dilate_tensor(tensor, dilation_factor, axis=1):

    if tensor is None or dilation_factor <= 1:
        return tensor

    axis_length = tensor.shape.as_list()[axis]

    # convert to list
    tensor = tf.split(tensor, num_or_size_splits=axis_length, axis=axis)
    new_tensor = []
    for i in range(axis_length):
        new_t = [tensor[i]] * dilation_factor
        new_tensor.extend(new_t)
    new_tensor = tf.concat(new_tensor, axis=axis)
    return new_tensor

def find_segment_overlap(segs1, segs2, segs_valid=None, max_segs=420):
    '''
    Find which segment ids overlap in segs1, segs2
    '''
    B,H,W = segs1.shape.as_list()
    N = max_segs
    assert segs2.shape == segs1.shape, (segs1.shape, segs2.shape)

    # set each in the range [0,max_segs)
    def _preproc(segs):
        segs -= tf.reduce_min(segs, axis=[1,2], keepdims=True)
        segs = tf.where(segs < N, segs, tf.ones_like(segs)*(N-1))
        return segs

    # hash pairs to unique value
    segs1 = _preproc(segs1)
    segs2 = _preproc(segs2)

    segs_hash = segs1 * N + segs2 # now in range [0,N**2)
    segs_hash += N * N * tf.reshape(tf.range(B, dtype=tf.int32), [-1,1,1]) # now in range [0, B*N*N)
    if segs_valid is not None:
        segs_hash = tf.where(segs_valid, segs_hash, tf.ones_like(segs_hash)*(B*N*N))

    # compute overlap
    overlap = tf.math.unsorted_segment_sum(
        data=tf.ones([B*H*W], dtype=tf.int32),
        segment_ids=tf.reshape(segs_hash, [-1]),
        num_segments=(B*N*N + 1))
    overlap = tf.minimum(overlap[:-1], tf.constant(1, tf.int32)) > 0
    overlap = tf.reshape(overlap, [B,N,N])

    return overlap

def l2_cost(nodes1, nodes2, **kwargs):
    assert len(nodes1.shape.as_list()) == 3, "shape must be [B,N,D] but is %s" % nodes1.shape.as_list()
    cost = tf.square(tf.expand_dims(nodes1, 2) - tf.expand_dims(nodes2, 1)) # [B,N,N,D]
    cost = tf.reduce_sum(cost, axis=-1, keepdims=False)
    return cost

def dice_cost(masks_pred, masks_gt, **kwargs):
    B,N,K = masks_pred.shape.as_list()
    _,_,O = masks_gt.shape.as_list()

    numer = tf.reduce_sum(masks_pred[...,tf.newaxis] * masks_gt[...,tf.newaxis,:], axis=1) # [B,K,O]
    denom = tf.reduce_sum(masks_pred, axis=1)[...,tf.newaxis] + tf.reduce_sum(masks_gt, axis=1)[:,tf.newaxis] # [B,K,O]

    dice = (2. * numer + 1.) / (denom + 1.)
    cost = 1. - dice

    return cost

def permute_nodes(nodes, assignment):

    B,N,D = nodes.shape.as_list()
    ones = tf.ones([B,N,1], tf.int32)
    b_inds = tf.reshape(tf.range(B, dtype=tf.int32), [B,1,1]) * ones
    n_inds = assignment[...,tf.newaxis]
    inds = tf.concat([b_inds, n_inds], axis=-1)
    pnodes = tf.gather_nd(nodes, inds)
    return pnodes

def hungarian_node_matching(nodes1, nodes2, dims_list=[[0,9]], cost_func=l2_cost, preproc_list=None, dim_weights=None, max_cost=10000.0, **kwargs):

    nodes1_valid = nodes1[...,-1]
    nodes2_valid = nodes2[...,-1]

    # get preprocs and dim weights
    if preproc_list is None:
        preproc_list = [tf.identity] * len(dims_list)
    assert len(preproc_list) == len(dims_list), "must pass one preproc per dim set"

    ndims = sum([d[1] - d[0] for d in dims_list])
    if dim_weights is None:
        dim_weights = tf.ones(shape=[1,1,ndims], dtype=tf.float32)
    else:
        assert len(dim_weights) == ndims, (len(dim_weights), ndims, dims_list)
        dim_weights = tf.reshape(tf.constant(dim_weights, dtype=tf.float32), [1,1,ndims])

    # get node dims that should be matched, preproc, and weight the output
    nodes1 = dim_weights * tf.concat([preproc_list[i](nodes1[...,d[0]:d[1]]) for i,d in enumerate(dims_list)], axis=-1)
    nodes2 = dim_weights * tf.concat([preproc_list[i](nodes2[...,d[0]:d[1]]) for i,d in enumerate(dims_list)], axis=-1)

    # set values of [B,N,N] cost matrix as weighted L2 distance
    cost_matrix = cost_func(nodes1, nodes2, **kwargs.get('cost_func_kwargs', {}))
    # cost_matrix = tf.reduce_sum(tf.square(tf.expand_dims(nodes1, 2) - tf.expand_dims(nodes2, 1)), -1, keepdims=False)

    # edges between two invalid nodes have 0.0 cost
    val_val = nodes1_valid[...,tf.newaxis] * nodes2_valid[:,tf.newaxis,:]
    cost_matrix = mask_tensor(cost_matrix, val_val, mask_value=0.0)

    # edges between a valid and invalid node have "infinite" cost
    val_inv = nodes1_valid[...,tf.newaxis] * (1.0 - nodes2_valid[:,tf.newaxis,:]) +\
              (1.0 - nodes1_valid[...,tf.newaxis]) * nodes2_valid[:,tf.newaxis,:]
    val_inv = tf.minimum(val_inv, 1.0)
    cost_matrix = mask_tensor(cost_matrix, 1.0 - val_inv, mask_value=max_cost)
    assignment = hung.hungarian(cost_matrix)

    return assignment

def matching_cost(nodes1, nodes2, dims_list=[[0,9]], cost_func=l2_cost, preproc_list=None, dim_weights=None, **kwargs):
    if preproc_list is None:
        preproc_list = [tf.identity] * len(dims_list)
    assert len(preproc_list) == len(dims_list)

    ndims = sum([d[1] - d[0] for d in dims_list])
    if dim_weights is None:
        dim_weights = tf.ones(shape=[1,1,ndims], dtype=tf.float32)
    else:
        assert len(dim_weights) == ndims, (len(dim_weights), ndims, dims_list)
        dim_weights = tf.reshape(tf.constant(dim_weights, dtype=tf.float32), [1,1,ndims])

    # get node dims that should be matched, preproc, and weight the output
    nodes1 = dim_weights * tf.concat([preproc_list[i](nodes1[...,d[0]:d[1]]) for i,d in enumerate(dims_list)], axis=-1)
    nodes2 = dim_weights * tf.concat([preproc_list[i](nodes2[...,d[0]:d[1]]) for i,d in enumerate(dims_list)], axis=-1)

    l2_cost = tf.reduce_sum(tf.square(nodes1 - nodes2), axis=-1, keepdims=True) # [B,N,1]
    return l2_cost

if __name__ == '__main__':

    B,N,D = [4,12,8]
    M = 10
    # nodes_pred = tf.random.normal([B,N,D], dtype=tf.float32)
    # nodes_gt = tf.random.normal([B,M,D], dtype=tf.float32)

    # nodes_pred = tf.concat([nodes_pred, tf.ones_like(nodes_pred[...,-1:])], -1)
    # nodes_gt = tf.concat([nodes_gt, tf.ones_like(nodes_gt[...,-1:])], -1)
    # ass = hungarian_node_matching(nodes_pred, nodes_gt, dims_list=[[0,4]])
    pred = tf.nn.sigmoid(tf.random.normal([B,512,16], dtype=tf.float32) * 10.)
    gt = tf.random.uniform(shape=[B,512], minval=0, maxval=8, dtype=tf.int32)
    gt = tf.cast(tf.one_hot(gt, depth=8, axis=-1), tf.float32)


    dice = dice_cost(pred, gt)
    ass = hung.hungarian(tf.transpose(dice, [0,2,1]))

    print(ass)

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sess = tf.Session()
    print(sess.run(ass))
