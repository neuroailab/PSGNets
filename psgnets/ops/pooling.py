from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from collections import OrderedDict
import os
import sys

from .convolutional import mlp, shared_spatial_mlp
from .dimensions import DimensionDict
from .utils import inversion_map, mask_tensor
from .graphical import *

lp = tf.load_op_library('../ops/src/tf_labelprop.so')
lpfc = tf.load_op_library('../ops/src/tf_labelprop_fc.so')
hung = tf.load_op_library('../ops/src/hungarian.so')
from .tf_nndistance import * # chamfer/nn distances

PRINT = False

def compute_segments_by_label_prop(edges, size, valid_nodes=None, labels_init=None, num_steps=10, defensive=False, synchronous=True, mode='index', seed=0, **kwargs):
    '''
    This is the wrapper for the LabelProp C++ op.
    Computes unique labels for each feature vector in a [B,H,W,C] tensor.

    Inputs
    edges: [B,HW,(2k+1)**2] <tf.bool> tensor of edges in a [(2k+1),(2k+1)]
                                      neighborhood of each feature in h,w space
    size: [2] <tf.int32> of [H,W]
    num_steps: int, number of steps to run the label prop algorithm for per example

    Outputs
    labels: [B,HW] <tf.int32> segment ids numbered 0 through (num_segments).sum(). Really should be thought of as a
                   single [BHW]-length vector, as segment ids increase over examples.
    num_segments: [B] <tf.int32> the number of segments in each example.

    '''
    shape = edges.shape.as_list()
    if synchronous:
        if size is None:
            B,N,_N = edges.shape.as_list()
            assert N == _N, "Must pass a square adjacency matrix"
            segment_ids, num_segments = labelprop_fc_sync(
                valid_nodes, edges, num_steps=num_steps, labels_init=labels_init, **kwargs)
        elif len(shape) == 3:
            B,N,K = edges.shape.as_list()
            newshape = tf.concat([tf.cast([B], tf.int32), size, tf.cast([K], tf.int32)], axis=0)
            segment_ids, num_segments = labelprop_image_sync(tf.reshape(edges, newshape), num_steps, mode=mode, seed=seed)
        elif len(shape) == 4:
            segment_ids, num_segments = labelprop_image_sync(edges, num_steps, mode=mode, seed=seed)
    else:
        if len(shape) == 4:
            B,H,W,K = edges.shape.as_list()
            segment_ids, num_segments = lp.label_prop(tf.reshape(edges, [B,H*W,K]), size, num_steps=num_steps, defensive=defensive)
        elif len(shape) == 3:
            segment_ids, num_segments = lp.label_prop(edges, size, num_steps=num_steps, defensive=defensive)

    return segment_ids, num_segments

def labelprop_fc(num_nodes_per_ex, edges_list, num_steps=5, sort_edges=True):

    assert len(num_nodes_per_ex.shape.as_list()) == 1
    assert edges_list.shape.as_list()[1] == 3

    labels, num_segments = lpfc.label_prop_fc(num_nodes_per_ex, edges_list, num_steps=num_steps, sort_edges=sort_edges)
    return labels, num_segments

def labelprop_fc_sync(valid_nodes, edges, num_steps=10, noise=0.001, seed=0, tau=0.0, labels_init=None, **kwargs):
    '''
    synchronous labelprop on valid nodes with weighted edge matrix

    inputs:
    valid_nodes: [B,N] <tf.float32>
    edges: [B,N,N] <tf.float32>

    outputs:
    labels: [B,N] <tf.int32> in increasing order across examples
    num_segments: [B] <tf.int32> number of unique labels per example
    '''

    B,N = valid_nodes.shape.as_list()
    if edges.dtype != tf.float32:
        edges = tf.cast(edges, tf.float32)

    # make sure there are self edges and that adj is symmetric
    adj = mask_tensor(edges, mask=(1.0 - tf.eye(N, batch_shape=[B], dtype=tf.float32)), mask_value=0.0)
    adj += tf.eye(N, batch_shape=[B], dtype=tf.float32)
    adj = tf.minimum(adj + tf.transpose(adj, [0,2,1]), 1.0)

    # structural equivalence group propagation
    kmat = tf.matmul(adj, adj)
    kmat *= (1.0 - tf.eye(N, batch_shape=[B], dtype=tf.float32))

    # init labels as one-hot identity matrix [B,N,N]
    if labels_init is None:
        labels = tf.eye(N, batch_shape=[B], dtype=tf.float32)
    else:
        assert labels_init.shape.as_list() == [B,N]
        assert labels_init.dtype == tf.int32
        labels = tf.one_hot(labels_init, axis=-1, depth=N, dtype=tf.float32)
        labels = tf.Print(labels, [tf.reduce_sum(tf.reduce_max(labels, axis=1), axis=-1)], message='unique_init_labels')

    rn = noise * tf.random.normal(shape=[B,N], seed=seed)[:,tf.newaxis] # [B,1,N] noise same in each row

    for _ in range(num_steps):
        labels = tf.matmul(adj * (1.0 + tau*kmat), labels)
        labels = tf.argmax(labels + rn, axis=2, output_type=tf.int32) # [B,N] tf.int32
        labels = tf.one_hot(labels, depth=N, axis=-1, dtype=tf.float32) # [B,N,N] tf.float32

    # now return to index format and mask out invalid labels
    labels = tf.argmax(labels, axis=2, output_type=tf.int32) # [B,N] in range [0,N)
    valid_inds = tf.cast(tf.where(valid_nodes > 0.5), tf.int32) # [?,2]
    b_ids = tf.tile(tf.reshape(tf.range(B, dtype=tf.int32), [B,1]), [1,N])
    labels = tf.gather_nd(params=tf.stack([b_ids, labels], axis=-1), indices=valid_inds) # [?,2] tuples of (b_id, label_id)

    # find unique labels
    unique_labels = tf.scatter_nd(indices=labels, updates=tf.ones_like(labels[:,0]), shape=[B,N])
    unique_labels = tf.minimum(unique_labels, 1) # [B,N] with 1s where there's a valid label
    num_segments = tf.reduce_sum(unique_labels, axis=-1) # [B]
    if PRINT:
        num_segments = tf.Print(num_segments, [num_segments, tf.shape(valid_nodes)], message='num_fc_segments')
    relabels = tf.cumsum(unique_labels, axis=-1, exclusive=True) # [B,N]

    # hash to lower values and add offsets
    offsets = tf.cumsum(num_segments, exclusive=True) # [B]
    offsets = tf.gather_nd(params=offsets, indices=labels[:,0:1]) # [?]
    labels = tf.gather_nd(params=relabels, indices=labels) + offsets # [?]

    return labels, num_segments

def labelprop_image_sync(edges, num_steps=10, mode='index', noise=0.001, seed=0, labels_init=None):
    '''
    synchronous labelprop on valid nodes with edges only to neighboring pixels

    inputs:
    edges: [B,H,W,K] <tf.bool> where K is the set of (2k+1)**2 neighbors in Manhattan Distance k from the starting pixel

    outputs:
    labels: [B,N] <tf.int32> label ids increasing order across examples
    num_segments: [B] <tf.int32> number of unique labels per example
    '''
    B,H,W,K = edges.shape.as_list()
    ksize = tf.constant(np.sqrt(K), tf.int32)
    r = tf.floordiv(tf.cast(ksize - tf.constant(1, tf.int32), tf.int32), tf.constant(2, tf.int32))

    # initialize labels
    N = H*W
    if labels_init is None:
        labels = tf.range(N + 1, dtype=tf.float32) # node 0 will be "invalid"
        labels = tf.tile(labels[tf.newaxis,:], [B,1]) # [B,N+1] in [0,N] inclusive
    else:
        assert labels_init.shape.as_list() in [[B,H,W], [B,N]]
        assert labels_init.dtype == tf.int32
        labels_init = tf.where(labels_init < tf.cast(N, tf.int32), labels_init,
                               tf.cast(N-1, tf.int32)*tf.ones_like(labels_init))
        labels = tf.reshape(labels_init, [B,N]) + tf.cast(1, tf.int32)
        labels = tf.concat([tf.zeros([B,1], dtype=tf.int32), labels], axis=1)

    # make sure self edges are present
    self_edges = tf.ones([B,H,W], dtype=tf.bool)
    zeros = tf.zeros_like(self_edges)
    self_edges = tf.stack([zeros]*(K // 2) + [self_edges] + [zeros]*(K // 2), axis=-1)
    edges = tf.logical_or(edges, self_edges)

    # convert values in each of the K positions to H,W positions
    def _hwk_to_neighbors(h,w,k):
        dh = tf.math.floordiv(k, ksize) - r
        dw = tf.math.floormod(k, ksize) - r
        in_view = tf.logical_and(
            tf.logical_and(h + dh >= 0, h + dh < H),
            tf.logical_and(w + dw >= 0, w + dw < W)) # [?,1] <tf.bool>
        neighb_inds = (h + dh)*W + (w + dw) + tf.constant(1, tf.int32) # use 1-indexing so that label 0 can be invalid
        neighb_inds = tf.where(in_view, neighb_inds, tf.zeros_like(neighb_inds)) # [?,1] value of 0 in all invalid positions
        return neighb_inds

    edge_inds = tf.cast(tf.where(edges), tf.int32) # [?,4] list of b,h,w,neighbor inds
    b_inds, h_inds, w_inds, k_inds = tf.split(edge_inds, [1,1,1,1], axis=-1) # [?,1] each
    node_inds = h_inds*W + w_inds + tf.constant(1, tf.int32) # use 1-indexing [?,1]
    neighbor_inds = _hwk_to_neighbors(h_inds, w_inds, k_inds) # [?,1] of inds into valid node positions in [0,HW] inclusive

    # At each step, get the set of labels connected to node n and find the argmax
    if mode == 'matmul':
        adj = tf.eye(N+1, batch_shape=[B], dtype=tf.float32) # [B,N+1,N+1] identity (self edges)
        scatter_inds = tf.concat([b_inds, node_inds, neighbor_inds], axis=-1) # [?,3]
        adj = adj + tf.scatter_nd(indices=scatter_inds, updates=tf.cast(tf.ones_like(neighbor_inds[:,0]), tf.float32), shape=[B,N+1,N+1])
        adj = tf.minimum(adj, tf.cast(1, tf.float32)) # [B,N+1,N+1] one-hot
        mask = tf.tile(tf.constant([0] + [1]*N, tf.float32)[tf.newaxis,tf.newaxis,:], [B,1,1]) # [B,1,N+1]
        labels = tf.one_hot(labels, depth=(N+1), axis=-1, dtype=tf.float32) # [B,N+1,N+1]

        rn = tf.constant(noise, tf.float32) * tf.random_normal(shape=[B,N+1], seed=seed)[:,tf.newaxis] # [B,1,N+1]
        for _ in range(num_steps):
            print("large matmul!")
            labels = tf.matmul(adj, labels) # [B,N+1,N+1]
            labels = labels * mask # mask out first column -- those are invalid labels
            labels = tf.argmax(tf.cast(labels, tf.float32) + rn, axis=2, output_type=tf.int32) # [B,N+1]
            labels = tf.one_hot(labels, depth=(N+1), axis=-1, dtype=tf.float32)

        # convert to index format
        labels = tf.argmax(labels, axis=2, output_type=tf.int32) # [B,N+1] with true labels starting at 1
    else:
        # scatter back to [B,N+1,K] as inds in range [0,N] inclusive
        scatter_inds = tf.concat([b_inds, node_inds, k_inds], axis=-1) # [?,3]
        neighbor_inds = tf.scatter_nd(indices=scatter_inds, updates=neighbor_inds[:,0], shape=[B,N+1,K])
        neighbor_inds = tf.stack([
            tf.tile(tf.range(B, dtype=tf.int32)[:,tf.newaxis,tf.newaxis], [1,N+1,K]),
            neighbor_inds], axis=-1) # [B,N+1,K,2]
        b_inds = tf.tile(tf.range(B, dtype=tf.int32)[:,tf.newaxis], [1,N+1])
        n_inds = tf.tile(tf.range(N+1, dtype=tf.int32)[tf.newaxis,:], [B,1])

        rn = tf.constant(noise, tf.float32) * tf.random.normal(shape=[B,N+1,K], seed=seed, dtype=tf.float32)
        for j in range(1,num_steps+1):
            # get current labels
            neighbor_labels = tf.gather_nd(params=labels, indices=neighbor_inds) # [B,N+1,K]
            num_neighbors = tf.reduce_sum(
                tf.cast(tf.equal(neighbor_labels[...,tf.newaxis], neighbor_labels[:,:,tf.newaxis,:]), tf.float32), axis=-1) # [B,N+1,K]
            num_neighbors = num_neighbors * tf.minimum(neighbor_labels, tf.constant(1.0, tf.float32))

            # update
            new_label_inds = tf.argmax(num_neighbors + rn, axis=-1, output_type=tf.int32) # [B,N+1] in [0,K)
            new_label_inds = tf.stack([
                b_inds, n_inds, new_label_inds], axis=-1) # [B,N+1,3]
            labels = tf.gather_nd(params=neighbor_labels, indices=new_label_inds)

        labels = tf.cast(labels, tf.int32)

    # remove invalid and return to 0 indexing
    labels = labels[:,1:] - tf.constant(1, tf.int32) # [B,N]

    # relabel so that there are no skipped label values and they range from [0, NB) at most
    b_inds = tf.tile(tf.range(B, dtype=tf.int32)[:,tf.newaxis], [1,N])
    unique_labels = tf.scatter_nd(
        tf.stack([b_inds, labels], axis=-1), updates=tf.ones_like(labels), shape=[B,N])
    unique_labels = tf.minimum(unique_labels, tf.constant(1, tf.int32)) # [B,N] where 1 is where there's a valid label
    num_segments = tf.reduce_sum(unique_labels, axis=-1) # [B]
    if PRINT:
        num_segments = tf.Print(num_segments, [num_segments], message='num_img_segments')
    relabels = tf.cumsum(unique_labels, axis=-1, exclusive=True)

    # hash to reordered values and add offsets
    offsets = tf.cumsum(num_segments, exclusive=True)[:,tf.newaxis] # [B,1]
    labels = tf.gather_nd(params=relabels, indices=tf.stack([b_inds, labels], axis=-1)) # [B,N]
    labels += offsets # now unique label for every segment

    return labels, num_segments

def num_segments_from_labels(labels):
    B,N = labels.shape.as_list()
    b_inds = tf.tile(tf.range(B, dtype=tf.int32)[:,tf.newaxis], [1,N])
    unique_labels = tf.scatter_nd(
        tf.stack([b_inds, labels], axis=-1), updates=tf.ones_like(labels), shape=[B,N])
    unique_labels = tf.minimum(unique_labels, tf.constant(1, tf.int32)) # [B,N] where 1 is where there's a valid label
    num_segments = tf.reduce_sum(unique_labels, axis=-1) # [B]
    relabels = tf.cumsum(unique_labels, axis=-1, exclusive=True)

    # hash to reordered values and add offsets
    offsets = tf.cumsum(num_segments, exclusive=True)[:,tf.newaxis] # [B,1]
    labels = tf.gather_nd(params=relabels, indices=tf.stack([b_inds, labels], axis=-1)) # [B,N]
    labels += offsets # now unique label for every segment

    return labels, num_segments

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
    b_inds = tf.cumsum(boundaries, exclusive=False)
    return b_inds

def inds_from_num_segments(num_segments, max_labels):
    b_inds = build_batch_inds(num_segments)
    n_inds = within_example_segment_ids(num_segments, max_labels)
    return b_inds, n_inds

def labels_to_segment_edges(labels, im_size, filter_type='sobel', padding='SAME'):
    '''
    Computes a float-valued, binary image where pixels are 1.0 if they represent an edge between segments and 0.0 otherwise
    '''
    H,W = im_size
    segment_ids = tf.cast(tf.reshape(labels, [-1, H, W, 1]), tf.float32)
    kernel = get_edge_filters(filter_type)[:,:,tf.newaxis,:]
    seg_edges = tf.nn.depthwise_conv2d(segment_ids, kernel, strides=[1,1,1,1], padding=padding)
    seg_edges = tf.reduce_sum(tf.square(seg_edges), axis=-1, keepdims=True)
    seg_edges = tf.cast(seg_edges > 0.5, tf.float32)

    if padding == 'VALID':
        pad = kernel.shape.as_list()[0] // 2
        pad = tf.constant([[0,0],[pad,pad],[pad,pad],[0,0]], tf.int32)
        seg_edges = tf.pad(seg_edges, paddings=pad, constant_values=tf.cast(0.0, tf.float32))
    return seg_edges

def labels_from_nodes(nodes, **kwargs):
    B,N,D = nodes.shape.as_list()
    valid_nodes = nodes[...,-1]
    num_nodes_per_ex = tf.cast(tf.reduce_sum(valid_nodes, axis=1, keepdims=False), tf.int32)
    edges_list = compute_fc_edges_list(nodes, valid_nodes, **kwargs)

    num_steps = kwargs.get('num_steps', 5)
    labels, num_segments = labelprop_fc(num_nodes_per_ex, edges_list, num_steps=num_steps, sort_edges=False) # [?], [B]
    Ntotal = tf.reduce_sum(num_segments)

    return labels, num_segments

def labels_from_nodes_and_edges(nodes, edges, synchronous=False, **kwargs):
    B,N,D = nodes.shape.as_list()
    assert edges.shape.as_list() == [B,N,N], ([B,N,N], edges.shape.as_list())
    num_steps = kwargs.get('num_steps', 5)

    valid_nodes = nodes[...,-1]

    valid_edges = (valid_nodes[...,tf.newaxis] * valid_nodes[:,tf.newaxis,:]) > 0.5
    if edges.dtype == tf.bool:
        edges = tf.logical_and(edges, valid_edges)
    elif edges.dtype == tf.float32:
        edges = edges * tf.cast(valid_edges, tf.float32)
    else:
        raise TypeError("edges must be bool or float")

    if PRINT:
        tf.Print(edges, [tf.reduce_mean(tf.reduce_sum(tf.cast(edges, tf.int32), axis=[1,2]))], message='avg_fc_edges')

    if synchronous:
        print("synchronous LP")
        labels, num_segments = labelprop_fc_sync(valid_nodes, edges, num_steps)
    else: # asynchronous
        num_nodes_per_ex = tf.cast(tf.reduce_sum(valid_nodes, axis=1, keepdims=False), tf.int32)
        edges_list = tf.cast(tf.where(edges), tf.int32)
        labels, num_segments = labelprop_fc(num_nodes_per_ex, edges_list, num_steps=num_steps, sort_edges=False) # [?], [B]
        Ntotal = tf.reduce_sum(num_segments)

    return labels, num_segments
