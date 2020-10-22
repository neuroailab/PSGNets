from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from collections import OrderedDict
import os
import sys

from psgnets.ops.convolutional import mlp, shared_spatial_mlp
from psgnets.ops.dimensions import DimensionDict
from psgnets.ops.utils import inversion_map, mask_tensor

#lp = tf.load_op_library('../ops/src/tf_labelprop.so')
#lpfc = tf.load_op_library('../ops/src/tf_labelprop_fc.so')
#hung = tf.load_op_library('../ops/src/hungarian.so')
import psgnets.ops.tf_nndistance # chamfer/nn distances

PRINT = False

def get_edge_filters(filter_type='sobel'):
    if filter_type == 'sobel':
        h = np.array([[1.,2.,1.,]])
        d = np.array([[1.,0.,-1.]])
    elif filter_type == 'simoncelli':
        h  = np.array([[0.030320,  0.249724,  0.439911,  0.249724,  0.030320]])
        d  = np.array([[0.104550,  0.292315,  0.000000, -0.292315, -0.104550]])
    hu = np.dot(d.T, h)
    hv = np.dot(h.T, d)

    return tf.constant(np.stack([hu,hv], axis=2), dtype=tf.float32)

def euclidean_dist2_fc(nodes, valid_nodes, thresh='mean', thresh_scale=0.25):
    B,N,D = nodes.shape.as_list()
    dists2 = tf.reduce_sum(tf.square(nodes[:,:,tf.newaxis,:] - nodes[:,tf.newaxis,:,:]), axis=-1) # [B,N,N]

    #thresh
    if thresh == 'mean':
        thresh = tf.reduce_mean(dists2, axis=[1,2], keepdims=True)
    thresh = thresh * thresh_scale
    edges_mat = dists2 < thresh

    # no edges between inval-inval or val-inval
    valid_mat = (valid_nodes[:,:,tf.newaxis] * valid_nodes[:,tf.newaxis,:]) > 0.5
    edges_mat = tf.logical_and(edges_mat, valid_mat)

    return edges_mat

def compute_fc_edges_list(nodes, valid_nodes=None, dims_list=[[0,9]], preproc_list=None, dim_weights=None,
                          metric=euclidean_dist2_fc, metric_kwargs={}, **kwargs):

    B,N,D = nodes.shape.as_list()
    total_dims = sum([(d[1] % D) - (d[0] % D) for d in dims_list])

    if valid_nodes is None:
        valid_nodes = tf.ones([B,N], tf.float32)

    # functions for processing each set of dims
    if preproc_list is None:
        preproc_list = [tf.identity] * len(dims_list)
    assert len(preproc_list) == len(dims_list), "Must pass one preproc per set of dims"

    # weights for scaling dimensions
    if dim_weights is None:
        dim_weights = tf.ones([1,1,total_dims], tf.float32)
    else:
        dim_weights = tf.reshape(tf.constant(dim_weights, tf.float32), [1,1,total_dims])

    nodes = tf.concat([preproc_list[i](nodes[...,d[0]:d[1]]) for i,d in enumerate(dims_list)], axis=-1) * dim_weights
    edges_mat_bool = metric(nodes, valid_nodes, **metric_kwargs)

    # format into edges list
    edges_list = tf.cast(tf.where(edges_mat_bool), tf.int32)

    return edges_list

def augment_features(features, kernel_list, channel_inds=None):
    B,H,W,C = features.shape.as_list()
    if channel_inds is None:
        channel_inds = [None] * len(kernel_list)
    aug_features = []
    new_channels = 0
    for i,kernel in enumerate(kernel_list):
        # supplied kernels may be numpy arrays
        if not isinstance(kernel, tf.Tensor):
            kernel = tf.constant(kernel, tf.float32)
        if len(kernel.shape.as_list()) == 2: # [H,W] kernel
            kernel = tf.tile(kernel[:,:,tf.newaxis,tf.newaxis], [1,1,C,1]) # [H,W,C,1]
            op = tf.nn.depthwise_conv2d
        elif len(kernel.shape.as_list()) == 3: # [H,W,Cout] kernels
            kernel = tf.tile(kernel[:,:,tf.newaxis,:], [1,1,C,1]) # [H,W,C,Cout]
            op = tf.nn.depthwise_conv2d
        else:
            assert kernel.shape.as_list() == 4, "Must be full conv kernel"
            assert channel_inds[i] is not None, "Must supply channel inds if doing an ordinary conv"
            assert (channel_inds[i][1] - channel_inds[i][0]) == kernel.shape.as_list()[2], "kernel input channels must match selected feature channels"
            op = tf.nn.conv2d

        # select channels
        inp_feats = features if channel_inds[i] is None else features[...,channel_inds[i][0]:channel_inds[i][1]]
        # op
        new_feats = op(inp_feats, kernel, strides=[1,1,1,1], padding='SAME')
        new_channels += new_feats.shape.as_list()[-1]
        aug_features.append(new_feats)

    return aug_features, new_channels

def euclidean_dist2(v1, v2, thresh='local', return_affinities=False, eps=1e-12, **kwargs):
    B,N,C,F = v2.shape.as_list()
    assert v1.shape.as_list() == [B,N,C,1]

    dists2 = tf.reduce_sum(tf.square(v1-v2), axis=2, keepdims=False) # [B,N,F]

    if thresh is None or thresh == 'local':
        channel_means = tf.reduce_mean(v1, axis=[1,3], keepdims=True) # [B,1,C,1]
        thresh = tf.reduce_sum(tf.square(v1 - channel_means), axis=2, keepdims=False) # [B,N,1]
    elif thresh == 'mean':
        thresh = tf.reduce_mean(dists2, axis=[1,2], keepdims=True)

    if return_affinities:
        affinities = tf.divide(1., tf.maximum(dists2, eps))
        return affinities, thresh
    else:
        adjacency = tf.cast(dists2 < thresh, tf.bool)
        return adjacency

def euclidean_dist2_valid(v1, v2, thresh='local', return_affinities=False, **kwargs):

    v1, v1_valid = tf.split(v1, [-1,1], axis=-2)
    v2, v2_valid = tf.split(v2, [-1,1], axis=-2)
    v1_valid = v1_valid[:,:,0] > 0.5
    v2_valid = v2_valid[:,:,0] > 0.5

    adj = euclidean_dist2(v1, v2, thresh=thresh, return_affinities=False)
    valid_adj = tf.logical_and(v1_valid, v2_valid)

    # no edges between valid and invalid
    adj = tf.logical_and(adj, valid_adj)

    # all edges between invalid/invalid are true
    adj = tf.logical_or(
        adj,
        tf.logical_not(tf.logical_or(v1_valid, v2_valid))
    )

    if return_affinities:
        return tf.cast(adj, tf.float32), adj
    else:
        return adj

def compute_adjacency_from_features(features, k=1, metric=euclidean_dist2, metric_kwargs={}, return_neighbors=False, return_affinities=False, symmetric=False, extract_patches=True, **kwargs):
    '''
    Inputs

    features: [B,H,W,C] <tf.float32>
    k: int, features are compared if they are manhattan distance <= k from each feature. ksize = (2k+1)
    metric: a nonnegative function that computes a distance between two feature vectors and returns a bool
            indicating whether they are to be connected in the PixGraph;

            signature:
            adjacency <tf.bool> = metric(v1, v2, **metric_kwargs),
            adjacency.shape == v1.shape == v2.shape (v1 may be broadcast)
    metric_kwargs: optional parameters to the metric function, such as a distance threshold

    Outputs

    adjacency: [B,H*W,(2k+1)**2] <tf.bool> whether feature at (h,w) is connected
               to a feature in its (2k+1)x(2k+1) local neighborhood

    '''
    B,H,W,C = features.shape.as_list()

    # optionally add coordinates
    if metric_kwargs.get('add_coordinates', False):
        if metric_kwargs.get('coordinate_scale', None) is None:
            coord_scale = tf.reduce_mean(features, axis=[1,2,3], keepdims=True)
        else:
            coord_scale = tf.constant(metric_kwargs['coordinate_scale'], dtype=tf.float32)
        ones = tf.ones([B,H,W,1], dtype=tf.float32)
        hc = tf.reshape(tf.range(H, dtype=tf.float32), [1,H,1,1]) * ones
        wc = tf.reshape(tf.range(W, dtype=tf.float32), [1,1,W,1]) * ones
        hc = coord_scale * ((hc / ((H-1.0)/2.0)) - 1.0)
        wc = coord_scale * ((wc / ((W-1.0)/2.0)) - 1.0)
        features = tf.concat([features[...,:-1], hc, wc, features[...,-1:]], axis=3)
        C += 2

    # construct kernels
    ksize = 2*k + 1
    if extract_patches:
        neighbors = tf.image.extract_patches(
            features,
            sizes=([1] + [ksize, ksize] + [1]),
            strides=[1,1,1,1],
            rates=[1,1,1,1],
            padding='SAME'
        )
        neighbors = tf.reshape(neighbors, [B,H*W,ksize**2,C])
        neighbors = tf.transpose(neighbors, [0,1,3,2])
    else:
        kernel = tf.range(ksize**2, dtype=tf.int32) # [ksize**2]
        kernel = tf.one_hot(kernel, depth=(ksize**2), axis=-1) # [ksize**2, ksize**2]
        kernel = tf.reshape(kernel, [ksize, ksize, 1, ksize**2])
        kernel = tf.cast(tf.tile(kernel, [1,1,C,1]), tf.float32)  # [ksize, ksize, C, ksize**2]

        # get neighboring feature vectors
        neighbors = tf.nn.depthwise_conv2d(features, kernel, strides=[1,1,1,1], padding='SAME') # [B,H,W,C*(ksize**2)]
        neighbors = tf.reshape(neighbors, [B,H*W,C,ksize**2])

    if return_neighbors:
        return neighbors

    # compare all neighbors to base points
    if return_affinities:
        affinities, thresh = metric(
            tf.reshape(features, [B,H*W,C,1]),
            neighbors, # [B,H*W,C,ksize**2]
            return_affinities=True,
            **metric_kwargs
        ) # [B,HW,ksize**2]
        adjacency = (1. / affinities) < thresh
    else:
        adjacency = metric(
            tf.reshape(features, [B,H*W,C,1]),
            neighbors,
            return_affinities=False,
            **metric_kwargs
        )

    if symmetric:
        edge_inds = tf.cast(tf.where(adjacency), tf.int32) # [?,3]
        b_inds, n_inds, k_inds = tf.split(edge_inds, [1,1,1], axis=-1)
        h_inds = tf.math.floordiv(n_inds, W)
        w_inds = tf.math.floormod(n_inds, W)
        dh = tf.math.floordiv(k_inds, ksize) - k # in [-k,k]
        dw = tf.math.floormod(k_inds, ksize) - k # in [-k,k]
        in_view = tf.logical_and(
            tf.logical_and(h_inds + dh >= 0, h_inds + dh < H),
            tf.logical_and(w_inds + dw >= 0, w_inds + dw < W)) # [?,1]
        neighb_inds = (h_inds + dh)*W + (w_inds + dw)
        neighb_inds = tf.where(in_view, neighb_inds, H*W*tf.ones_like(neighb_inds))
        new_k_inds = (k - dh)*ksize + (k - dw) # in [0,ksize**2)
        swapped_inds = tf.concat([b_inds, neighb_inds, new_k_inds], axis=-1) # [?,3]
        swapped_adj = tf.scatter_nd(swapped_inds, updates=tf.ones_like(neighb_inds[:,0]), shape=[B,H*W+1,ksize**2])
        swapped_adj = swapped_adj[:,:-1] > 0 # [B,N,ksize**2] <tf.bool>
        adjacency = tf.logical_or(adjacency, swapped_adj)

    # TODO: Deal with literal edge/corner cases -- though this may be handled by LabelProp for some metrics
    if return_affinities:
        return affinities, adjacency
    else:
        return adjacency

def local_matmul(affinities, nodes, size=None):
    '''
    Compute equivalent of tf.matmul(affinities, nodes) but for an
    affinity matrix [B,N,K] where the K dimension indexes a local ksize x ksize patch around each node n (originally from a tensor of shape [B]+size+[D])
    '''

    B,N,K = affinities.shape.as_list()
    assert K % int(np.sqrt(K)) == 0, K
    ksize = int(np.sqrt(K))
    if len(nodes.shape) == 3 and size is None:
        _,_N,D = nodes.shape.as_list()
        assert N == _N
        assert N % int(np.sqrt(N)) == 0, N
        size = [int(np.sqrt(N))] * 2
        nodes = tf.reshape(nodes, [B, size[0], size[1], -1])
    elif len(nodes.shape) == 4:
        size = nodes.shape.as_list()[1:3]
        D = nodes.shape.as_list()[-1]
    else:
        assert N == np.prod(size), (N, size)
        D = nodes.shape.as_list()[-1]
        nodes = tf.reshape(nodes, [B, size[0], size[1], D])

    patches = tf.image.extract_patches(
        nodes, sizes=[1, ksize, ksize, 1],
        strides=[1,1,1,1], rates=[1,1,1,1],
        padding='SAME')
    patches = tf.reshape(patches, [B,N,K,D])
    out = patches * affinities[...,tf.newaxis]
    out = tf.reduce_sum(out, axis=-2)
    return out

def sum_effects(effects,
                receiver_idxs,
                batch_size,
                time,
                num_nodes,
                agg_type='sum',
                **kwargs):
    """
    Constructs a node list in which effects with the same receiver index are summed into
    the same node list position.

    :param effects: Tensor(?, effect_dim), pairwise effects on receiver
    :param receiver_idxs: Tensor(..., 3), list of (batch_index, time, receiver_index)
    :param batch_size: int, batch size
    :param time: int, number of time steps
    :param num_nodes: int, number of nodes

    :return node_effects: Tensor(batch_size, time, num_nodes, effect_dim),
        summed per node effects
    """
    effect_dim = effects.get_shape().as_list()[1]
    receiver_nidxs = receiver_idxs.get_shape().as_list()[-1]
    assert receiver_nidxs == 3, ('receiver_idxs must be a list of ' + \
            '(batch_index, time_index, receiver_node_index)')
    # Assign same identifier to effects that have the same receiver
    receiver_idxs_sum = \
            tf.reshape(receiver_idxs, [-1, receiver_nidxs])
    cum = tf.math.cumprod([batch_size, time, num_nodes], axis=-1,
            reverse=True, exclusive=True)
    cum = tf.cast(cum, receiver_idxs_sum.dtype)
    receiver_idxs_sum = tf.matmul(receiver_idxs_sum, tf.expand_dims(cum, axis=-1))
    receiver_idxs_sum = tf.reshape(receiver_idxs_sum, [-1])
    if agg_type == 'sum':
        # Sum across all effects with the same identifier
        node_effects = \
                tf.math.unsorted_segment_sum(effects, receiver_idxs_sum,
                        batch_size * time * num_nodes)
    elif agg_type == 'mean':
        # Sum across all effects with the same identifier
        node_effects = \
                tf.math.unsorted_segment_mean(effects, receiver_idxs_sum,
                        batch_size * time * num_nodes)
    else:
        raise ValueError("agg_type must be in ['sum', 'mean'] but is %s" % agg_type)
    # Reshape back into original node shape
    node_effects = tf.reshape(node_effects,
            [batch_size, time, num_nodes, effect_dim])
    return node_effects

def graphconv_pairwise(nodes,
                       adjacency,
                       hidden_dims,
                       activations=tf.nn.elu,
                       scope='graphconv',
                       share_weights=False,
                       scale_diffs_by_effects=False,
                       right_receivers=True,
                       **kwargs):
    '''
    A hard graph convolution assuming hard connections between nodes as specified
    by adjacency.

    :param nodes: Tensor(batch_size, time, num_nodes, node_dim), input nodes
    :param adjacency: Tensor(batch_size, time, num_nodes, num_nodes, edge_dim),
        adjacency
    :param hidden_dims: list([int, int, ...], node feature dimensions for hidden
        and output layer convolutions
    :param activation_func: func, activation function used after
        graph convolutional layer
    :param scope: str, variable scope
    :param share_weights: bool, shares weights across graph convolution filters

    :return nodes: Tensor(batch_size, time, num_nodes, hidden_dims[-1]), convolved nodes
    '''
    # add time dim
    temporal = True
    if len(nodes.shape) == 3:
        nodes = nodes[:,tf.newaxis]
        temporal = False
    if len(adjacency.shape) == 3:
        adjacency = adjacency[:,tf.newaxis,...,tf.newaxis]
    elif len(adjacency.shape) == 4:
        adjacency = adjacency[:,tf.newaxis]
    assert len(nodes.shape) == 4, \
            ('Must be a tensor of dimension [batch, time, nodes, node_dim]')
    assert len(adjacency.shape) == 5, \
            ('Must be a tensor of dimension' + \
            '[batch, time, nodes, nodes, valid + edge_dim]')
    assert adjacency.shape[2] == adjacency.shape[3], (adjacency.shape)
    assert nodes.shape[2] == adjacency.shape[2], (nodes.shape, adjacency.shape)
    if scale_diffs_by_effects:
        assert hidden_dims[-1] == nodes.shape.as_list()[-1], \
            "Node dim and final effects dim must match"

    batch_size, time, num_nodes = nodes.get_shape().as_list()[0:3]

    # Gather node, node pairs
    n2n_idxs = tf.cast(tf.where(adjacency[:, :, :, :, 0]), tf.int32)
    # n2n_idxs = tf.Print(n2n_idxs, [tf.shape(n2n_idxs)[0]], message='num_edges')
    batch_time_idxs = n2n_idxs[:, 0:2]
    lidxs = n2n_idxs[:, 2:3]
    ridxs = n2n_idxs[:, 3:4]
    lnodes = tf.gather_nd(nodes, tf.concat([batch_time_idxs, lidxs], axis=-1))
    rnodes = tf.gather_nd(nodes, tf.concat([batch_time_idxs, ridxs], axis=-1))
    diffs = rnodes - lnodes
    pairs = tf.concat([lnodes, rnodes, diffs, tf.square(diffs)], axis = -1)

    # Add edge features if any
    if adjacency.get_shape().as_list()[-1] > 1:
        edge_features = tf.gather_nd(adjacency[:, :, :, :, 1:], n2n_idxs)
        pairs = tf.concat([pairs, edge_features], axis = -1)

    # Graph convolve node, node, edge pairs
    effects = mlp(pairs, hidden_dims, activations,
            scope=scope, share_weights=share_weights, **kwargs)
    # output effects may be rescaled diffs between connected nodes
    if scale_diffs_by_effects:
        effects = effects * diffs # [?, node_dim]
    rec_inds = ridxs if right_receivers else lidxs
    effects = sum_effects(effects, tf.concat([batch_time_idxs, rec_inds], axis=-1),
                        batch_size, time, num_nodes, **kwargs)

    return effects if temporal else effects[:,0]

def graphconv_pairwise_from_inds(
        lnodes,
        rnodes,
        edges,
        hidden_dims=[50],
        activations=None,
        scope='',
        share_weights=False,
        right_receivers=False,
        edge_effects=False,
        **kwargs):

    assert len(lnodes.shape) == 4
    assert len(rnodes.shape) == 4
    assert edges.shape.as_list()[1] == 4

    B,T,lN,lD = lnodes.shape.as_list()
    _,_,rN,rD = rnodes.shape.as_list()

    # Construct hidden layer activations, output layer is assumed to be identity
    if activations is None:
        activations = tf.nn.elu
    if isinstance(activations, type(tf.identity)):
        activations = [activations] * len(hidden_dims)
        activations[-1] = tf.identity

    # get the pairs of lnodes and rnodes connected by edges
    batch_time_inds = edges[:,0:2]
    linds = edges[:,2:3]
    rinds = edges[:,3:4]

    # get the pairs of nodes connected by edges
    lnodes = tf.gather_nd(lnodes, tf.concat([batch_time_inds, linds], axis=-1)) # [num_edges, lD]
    rnodes = tf.gather_nd(rnodes, tf.concat([batch_time_inds, rinds], axis=-1)) # [num_edges, rD]
    pairs = tf.concat([lnodes, rnodes], axis=-1) # [num_edges, lD+rD]

    # get effects per edge from an mlp, then sum across effects that have the same lnode
    effects = mlp(pairs, hidden_dims, activations, scope=scope, share_weights=share_weights, **kwargs) # [num_edges, hidden_dims[-1]]
    rec_inds = rinds if right_receivers else linds
    recN = rN if right_receivers else lN
    rec_effects = sum_effects(effects, receiver_idxs=tf.concat([batch_time_inds, rec_inds], axis=-1),
                              batch_size=B, time=T, num_nodes=recN, **kwargs)
    if edge_effects:
        return rec_effects, effects
    else:
        return rec_effects

def find_nearest_k_parent_inds(lnodes, rnodes, kNN=5, nn_dims=[-4,-2], dmax=10000., **kwargs):
    B,lN,D = lnodes.shape.as_list()
    _,rN,D = rnodes.shape.as_list()
    dtype = lnodes.dtype
    valid_lnodes = lnodes[...,-1:] # [B,lN,1]
    valid_rnodes = rnodes[...,-1:] # [B,rN,1]
    coord_lnodes = lnodes[...,nn_dims[0]:nn_dims[1]]
    coord_rnodes = rnodes[...,nn_dims[0]:nn_dims[1]]

    # mask for invalid nodes
    valid_dists2 = valid_lnodes * tf.transpose(valid_rnodes, [0,2,1]) # [B,lN,rN]

    # compute dists2 and mask
    dists2 = tf.square(tf.expand_dims(coord_lnodes, 2) - tf.expand_dims(coord_rnodes, 1))
    dists2 = tf.reduce_sum(dists2, axis=-1, keepdims=False) # [B,lN,rN]
    dists2 = (dists2 * valid_dists2) + (tf.cast(dmax, dtype) * (tf.cast(1, dtype) - valid_dists2))

    # get inds
    assert kNN < rN
    if kNN > 1:
        _, nearest_k_inds = tf.nn.top_k(-dists2, k=kNN, sorted=True) # [B,N,kNN << rN]
    else:
        nearest_k_inds = tf.argmax(-dists2, axis=-1, output_type=tf.int32)[...,tf.newaxis]
    return nearest_k_inds

def find_nearest_k_node_inds(nodes, kNN=1, nn_dims=[-4,-2], dmax=10000., **kwargs):
    '''
    Find the indices of the nearest k0:k nodes to each node in nodes.
    '''

    B,N,D = nodes.shape.as_list()
    DD = nn_dims[1] - nn_dims[0]
    dtype = nodes.dtype
    valid_nodes = nodes[...,-1:] # [B,N,1]
    coord_nodes = nodes[...,nn_dims[0]:nn_dims[1]] # [B,N,DD]

    # get inds
    # mask for self-edges and invalid nodes
    valid_dists2 = valid_nodes * tf.transpose(valid_nodes, [0,2,1]) # [B,N,N]
    valid_dists2 = valid_dists2 * (tf.cast(1, dtype) - tf.eye(N, batch_shape=[B], dtype=dtype))

    # compute dists and mask
    dists2 = tf.square(tf.expand_dims(coord_nodes, 2) - tf.expand_dims(coord_nodes, 1))
    dists2 = tf.reduce_sum(dists2, axis=-1, keepdims=False) # [B,N,N]
    dists2 = (dists2 * valid_dists2) + (tf.cast(dmax, dtype) * (tf.cast(1, dtype) - valid_dists2))
    _, nearest_k_inds = tf.nn.top_k(-dists2, k=kNN, sorted=True) # [B,N,kNN]

    assert nearest_k_inds.dtype == tf.int32

    return nearest_k_inds

def attr_diffs_from_neighbor_inds(nodes, neighbor_inds,
                                  rnodes=None,
                                  valid_nodes=None,
                                  attr_dims_list=[[2,3]],
                                  attr_metrics_list=[lambda x,y: y],
                                  mask_self=False,
                                  **kwargs):
    '''
    Get the diffs between pairs of nodes given by neighbor_inds

    nodes: [B,N,D] <tf.float32>
    neighbor_inds: [B,N,K] <tf.int32>
    valid_inds: [B,N,K] <tf.float32> in {0.,1.}

    '''
    B,N,D = nodes.shape.as_list()
    if valid_nodes is None:
        valid_nodes = nodes[...,-1:] # [B,N,1]

    # get the valid_diffs
    if rnodes is None:
        valid_diffs = valid_nodes * tf.transpose(valid_nodes, [0,2,1]) # [B,N,N]
        # self mask
        if mask_self:
            valid_diffs *= (1.0 - tf.eye(N, batch_shape=[B], dtype=tf.float32))
    else:
        valid_rnodes = rnodes[...,-1:] # [B,N,1]
        valid_diffs = valid_nodes * tf.transpose(valid_rnodes, [0,2,1])

    b_inds = tf.reshape(tf.range(B, dtype=tf.int32), [B,1,1]) * tf.ones_like(neighbor_inds) # [B,N,K]
    n_inds = tf.reshape(tf.range(N, dtype=tf.int32), [1,N,1]) * tf.ones_like(neighbor_inds) # [B,N,K]
    gather_inds = tf.stack([b_inds, n_inds, neighbor_inds], axis=-1, name="valid_gather_inds") # [B,N,K,3]
    valid_diffs = tf.gather_nd(valid_diffs, gather_inds)[...,tf.newaxis] # [B,N,K,1] <tf.float32>

    # get the node_diffs
    gather_inds = tf.stack([b_inds, neighbor_inds], axis=-1, name="node_gather_inds") # [B,N,K,2]
    neighbor_nodes = tf.gather_nd(rnodes if rnodes is not None else nodes, gather_inds) # [B,N,K,D]
    base_nodes = nodes[:,:,tf.newaxis] # [B,N,1,D]
    attr_diffs_list = [ # list< [B,N,K,D] >
        attr_metrics_list[i](
            base_nodes[...,ad[0]:ad[1]],
            neighbor_nodes[...,ad[0]:ad[1]])
        for i,ad in enumerate(attr_dims_list)]

    return attr_diffs_list, valid_diffs

def compute_same_parent_within_layer_edges(n2n_idxs,
        remove_self_connections=True):
    """
    Given node-to-node leaf to parent indices outputs edges between leaf nodes
    that share the same parent.

    :param n2n_idxs: Tensor(batch_size, time, lN, num_top_k_edges, 4),
        leaf to parent edges formatted as batch_index, time_index, lindex, rindex
    :param remove_self_connections: bool, removes self connections between leaf nodes

    :return lledges: Tensor(?, 4), same parent leaf to leaf edges
        formatted as batch_index, time_index, lindex, lindex
    """
    batch_size, time, lN, rN, nidxs = n2n_idxs.shape
    assert nidxs == 4, nidxs

    # Construct all possible combinations of edge connections
    # lledge format: batch_index, time_index, lindex1, lindex2, rindex2, rindex1
    lledges = tf.reshape(tf.tile(n2n_idxs[:, :, :, tf.newaxis, :, :],
        [1, 1, 1, lN, 1, 1]), [batch_size, time, -1, rN, nidxs])
    lledges = tf.concat([
        lledges[:, :, :, :, 0:3],
        tf.reshape(tf.tile(n2n_idxs[:, :, :, tf.newaxis, :, 2:4],
            [1, 1, lN, 1, 1, 1]), [batch_size, time, -1, rN, nidxs - 2]),
        lledges[:, :, :, :, 3:4]
        ], axis = -1)
    # lnodes belong to the same group if they share the same rindex
    valid_edges = tf.equal(lledges[:, :, :, :, 5] - lledges[:, :, :, :, 4], 0)
    if remove_self_connections:
        # self connections occur between lnodes with lindex1 == lindex2
        valid_edges = tf.logical_and(valid_edges,
                tf.not_equal(lledges[:, :, :, :, 3] - lledges[:, :, :, :, 2], 0))
    # Gather valid edges
    valid_inds = tf.cast(tf.where(valid_edges), tf.int32)
    lledges = tf.gather_nd(lledges, valid_inds)[:, 0:4]

    return lledges

def compute_any_same_parent_within_layer_edges(n2p_idxs, remove_self_connections=True):
    '''
    Like above but within-layer nodes are connected if they share any parent
    '''
    bs, ts, lN, rN, nidxs = n2p_idxs.shape.as_list()
    assert nidxs == 4, nidxs
    parent_inds = n2p_idxs[...,3] # [bs,ts,lN,rN]
    parent_inds = parent_inds[:,:,:,tf.newaxis,:,tf.newaxis] # [bs,ts,lN,1,rN,1]

    # node pairs that have parent in common
    same_parent = tf.equal(
        parent_inds - tf.transpose(parent_inds, [0,1,3,2,5,4]), tf.cast(0, tf.int32)) # [bs,ts,lN,lN,rN,rN] bool

    # node pairs that have any parents in common (in case there's multiple parents per node)
    same_parent = tf.reduce_any(same_parent, axis=[4,5]) # [bs,ts,lN,lN]

    # nodes will not be connected to themselves even though by definition they have the same parents
    if remove_self_connections:
        same_parent = tf.logical_and(
            same_parent,
            tf.logical_not(tf.eye(lN, batch_shape=[bs,ts], dtype=tf.bool))
        )

    # lledges are indices where the nodes had any same parent
    lledges = tf.cast(tf.where(same_parent), tf.int32) # [?,4]

    return lledges

def nearest_neighbor_edges(lnodes, rnodes, rnode_idx_offset=0):
    '''

    '''
    B,lN,lD = lnodes.shape.as_list()
    _,rN,rD = rnodes.shape.as_list()
    assert lD == rD, (lD,rD)
    if lD not in [3,6,12]:
        pD = min([D - lD for D in [3,6,12] if D - lD > 0])
        lnodes = tf.concat([lnodes, tf.zeros([B,lN,pD], tf.float32)], axis=-1)
        rnodes = tf.concat([rnodes, tf.zeros([B,rN,pD], tf.float32)], axis=-1)
    # assert lD in [3, 6, 12], "Nearest Neighbor only works on dimensions 3, 6, 12 for now"

    ldists2, lnn_idxs, rdists2, rnn_idxs = tf_nndistance.nn_distance(lnodes, rnodes)
    ldists2.set_shape([B,lN])
    lnn_idxs.set_shape([B,lN])
    ldists2 = tf.cast(ldists2, tf.float32)
    lnn_idxs = tf.cast(lnn_idxs, tf.int32)

    ones = tf.ones([B,1,lN,1,1], tf.int32)
    batch_idxs = tf.reshape(tf.range(B, dtype=tf.int32), [B,1,1,1,1]) * ones
    time_idxs = tf.reshape(tf.range(1, dtype=tf.int32), [1,1,1,1,1]) * ones
    lnodes_idxs = tf.reshape(tf.range(lN, dtype=tf.int32), [1,1,lN,1,1]) * ones
    edge_idxs = lnn_idxs[:,tf.newaxis,:,tf.newaxis,tf.newaxis] + tf.cast(rnode_idx_offset, tf.int32)
    edge_idxs = tf.concat([batch_idxs, time_idxs, lnodes_idxs, edge_idxs], axis=-1) # [B,1,lN,1,4]

    return edge_idxs

def compute_edges_spacetime_nn(lnodes, rnodes, scope, dists_k=1.0, dists_p=1.0, out_times=2):
    '''
    Finds the nearest neighbor in rnodes for each lnodes
    '''
    B,T,lN,lD = lnodes.shape.as_list()
    _,_,rN,rD = rnodes.shape.as_list()

    assert lD == rD, "Spatial dimensions must match"
    assert lD == 3, "Only for computing distances in Euclidian space"

    # find nearest neighbors in spacetime
    if T in [1,2,4]:
        lnodes_traj = tf.concat([lnodes[:,t] for t in range(T)], axis=-1)
        rnodes_traj = tf.concat([rnodes[:,t] for t in range(T)], axis=-1)
        ldists2, lnn_idxs, rdists2, rnn_idxs = tf_nndistance.nn_distance(lnodes_traj, rnodes_traj)
        ldists2.set_shape([B,lN])
        lnn_idxs.set_shape([B,lN])
        ldists2 = tf.cast(ldists2, tf.float32)
        lnn_idxs = tf.cast(lnn_idxs, tf.int32)
    else:
        raise NotImplementedError("Can't compute trajectory NNs if T not in [1,2,4]")

    # postprocessing for Damian's indexing format
    ones = tf.ones([B,out_times,lN,1,1], tf.int32)
    batch_idxs = tf.reshape(tf.range(B, dtype=tf.int32), [B,1,1,1,1]) * ones
    time_idxs = tf.reshape(tf.range(out_times, dtype=tf.int32), [1,out_times,1,1,1]) * ones
    lnodes_idxs = tf.reshape(tf.range(lN, dtype=tf.int32), [1,1,lN,1,1]) * ones
    edge_idxs = lnn_idxs[:,tf.newaxis,:,tf.newaxis,tf.newaxis] * ones # [B,T-2,lN,1,1]
    edge_idxs = tf.concat([batch_idxs, time_idxs, lnodes_idxs, edge_idxs], axis=-1) # [B,T-2,lN,1,4]
    edges = inversion_map(dists_k, dists_p, eps=1e-6)(ldists2) # [B,lN]
    edges = edges[:,tf.newaxis,:,tf.newaxis,tf.newaxis] * tf.cast(ones, tf.float32) # [B,T-2,lN,1,1] single attribute is inversion_map(dists2)

    return edges, edge_idxs

def get_grandparents(l0r1edges, l1r2edges):
    """
    Takes edges between l0nodes <-> r1nodes and l1nodes <-> r2nodes and
    outputs edges between l0nodes <-> r2nodes.

    :param l0r1edges: Tensor(batch_size, time, lN, rN,
        [batch_index, time_index, lindex, rindex, edge_features])
    :param l1r2edges: Tensor(batch_size, time, lN, rN,
        [batch_index, time_index, lindex, rindex, edge_features])

    :return l0r2edges: Tensor(batch_size, time, lN, rN,
        [batch_index, time_index, lindex, rindex, edge_features])
    """
    # print("l0r1edges shape", l0r1edges.shape.as_list())
    # print("l1r2edges shape", l1r2edges.shape.as_list())

    l0r2shape = l0r1edges.get_shape().as_list()
    l0r2shape[2] = -1
    batch_time_idxs = l0r1edges[:, :, :, :, 0:2]
    ridxs = l0r1edges[:, :, :, :, 3:4]
    l0r2edges = tf.reshape(tf.gather_nd(l1r2edges, \
            tf.cast(tf.concat([batch_time_idxs, ridxs], axis = -1), tf.int32)),
            l0r2shape)
    return l0r2edges


def get_ancestors(lredges_hierarchy):
    """
    Takes list of edges [l0nodes <-> r1nodes, l1nodes <-> r2nodes, ...] and
    outputs list of edges [l0nodes <-> r1nodes, l0nodes <-> r2nodes, ...].

    :param lredges_hierarchy: list([Tensor(batch_size, time, lN, rN,
        [batch_index, time_index, lindex, rindex, edge_features]), ...]),
        ordered edges describing hierarchy with child to parent edges

    :return l0rXedges: list([Tensor(batch_size, time, lN, rN,
        [batch_index, time_index, lindex, rindex, edge_features]), ...]),
        ordered edges describing hierarchy with leaf to ancestor edges
    """
    l0rXedges = [lredges_hierarchy[0]]
    for lXrYedges in lredges_hierarchy[1:]:
        l0rXedges.append(get_grandparents(l0rXedges[-1], lXrYedges))
    # print("l0rXedges", [edges.shape.as_list() for edges in l0rXedges])

    return l0rXedges

def l2_cost(nodes1, nodes2, **kwargs):
    assert len(nodes1.shape.as_list()) == 3, "shape must be [B,N,D] but is %s" % nodes1.shape.as_list()
    assert nodes1.shape == nodes2.shape
    # cost_matrix = tf.reduce_sum(tf.square(tf.expand_dims(nodes1, 2) - tf.expand_dims(nodes2, 1)), -1, keepdims=False)
    print("l2 cost")
    cost = tf.square(tf.expand_dims(nodes1, 2) - tf.expand_dims(nodes2, 1)) # [B,N,N,D]
    cost = tf.reduce_sum(cost, axis=-1, keepdims=False)
    return cost

def l1_cost(nodes1, nodes2, **kwargs):
    assert len(nodes1.shape.as_list()) == 3, "shape must be [B,N,D] but is %s" % nodes1.shape.as_list()
    assert nodes1.shape == nodes2.shape
    cost = tf.abs(nodes1[:,:,tf.newaxis] - nodes1[:,tf.newaxis]) # [B,N,N,D]
    cost = tf.reduce_sum(cost, axis=-1, keepdims=False)
    return cost

def lp_cost(nodes1, nodes2, p=4, **kwargs):
    assert len(nodes1.shape.as_list()) == 3, "shape must be [B,N,D] but is %s" % nodes1.shape.as_list()
    assert nodes1.shape == nodes2.shape
    cost = tf.pow(nodes1[:,:,tf.newaxis] - nodes1[:,tf.newaxis], p) # [B,N,N,D]
    cost = tf.reduce_sum(cost, axis=-1, keepdims=False)
    return cost

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

def permute_nodes(nodes, assignment):

    B,N,D = nodes.shape.as_list()
    ones = tf.ones([B,N,1], tf.int32)
    b_inds = tf.reshape(tf.range(B, dtype=tf.int32), [B,1,1]) * ones
    n_inds = assignment[...,tf.newaxis]
    inds = tf.concat([b_inds, n_inds], axis=-1)
    pnodes = tf.gather_nd(nodes, inds)
    return pnodes

def compute_segment_border_mats(segment_ids, edge_pixels=None, num_segments=None, num_pixels_thresh=1, max_segments=128, **kwargs):
    '''
    segment_ids: [B,[T],H,W] <tf.int32> map of segment ids in range [0,inf)
    num_segments: if None, gets computed on the fly and implicitly assumes tf.reshape(segment_ids, [B*T,etc.]) gives segments in original order.
                  Otherwise, this should be the output of labelprop.
    max_segments: <int> number of out channels to assume

    returns
    seg_border_mats: [B,[T],Nmax,Nmax] <tf.bool> a matrix that is True wherever segment n_i shares a pixel border with segment n_j.
    '''
    edge_kernels = {
        'left': tf.reshape(tf.constant([1,-1], tf.float32), [1,2,1,1]),
        'right': tf.reshape(tf.constant([-1,1], tf.float32), [1,2,1,1]),
        'up': tf.reshape(tf.constant([1,-1], tf.float32), [2,1,1,1]),
        'down': tf.reshape(tf.constant([-1,1], tf.float32), [2,1,1,1])
    }
    padding = {
        'left': tf.constant([[0,0],[0,0],[0,1],[0,0]], tf.int32),
        'right': tf.constant([[0,0],[0,0],[1,0],[0,0]], tf.int32),
        'up': tf.constant([[0,0],[0,1],[0,0],[0,0]], tf.int32),
        'down': tf.constant([[0,0],[1,0],[0,0],[0,0]], tf.int32)
    }

    assert segment_ids.dtype == tf.int32
    seg_shape = segment_ids.shape.as_list()
    if len(seg_shape) == 4:
        B,T,H,W = segment_ids.shape.as_list()
        segment_ids = tf.reshape(segment_ids, [B*T,H,W])
    elif len(seg_shape) == 3:
        B,H,W = segment_ids.shape.as_list()
        T = 1
    else:
        raise ValueError("segment_ids must be [B,T,H,W] or [B,H,W]")

    # the edge pixels that will mask the borders
    if edge_pixels is not None:
        edge_pixels = tf.minimum(tf.reshape(tf.cast(edge_pixels, tf.int32), [B*T,H,W,1]), tf.constant(1, tf.int32))
    else:
        edge_pixels = tf.ones([B*T,H,W,1], dtype=tf.int32)

    # preproc
    offsets = tf.reduce_min(segment_ids, axis=[1,2], keepdims=True)
    segment_ids -= offsets

    segment_ids_map = tf.where(segment_ids < tf.ones_like(segment_ids)*max_segments,
                               segment_ids,
                               tf.ones_like(segment_ids) * (max_segments-1))

    if num_segments is None:
        num_segments = tf.reduce_max(segment_ids_map, axis=[1,2], keepdims=False) + tf.constant(1, tf.int32) # [B*T]
    else:
        assert num_segments.shape.as_list() == [B*T], "If you passed num_segments, it must have shape [B*T] and correspond to segment_ids"


    # now find borders of each kind
    directional_diffs = []
    for edir in ['left', 'right', 'up', 'down']:
        kern = edge_kernels[edir]
        pad = padding[edir]
        seg_diffs = tf.cast(tf.nn.conv2d(tf.cast(segment_ids_map[...,tf.newaxis], tf.float32), kern, strides=[1,1,1,1], padding='VALID'), tf.int32)
        seg_diffs = tf.pad(seg_diffs, pad, mode='CONSTANT', constant_values=tf.constant(0, tf.int32)) # [B*T,H,W,1]
        diff_mask = tf.minimum(tf.abs(seg_diffs), tf.constant(1, tf.int32))
        diff_mask  = diff_mask * edge_pixels # restricts borders to edge pixels
        seg_diffs = (segment_ids_map[...,tf.newaxis] - seg_diffs) # now edges have value of segment to the [edir]
        seg_diffs = seg_diffs*diff_mask - (tf.constant(1, tf.int32) - diff_mask) # null values have -1
        directional_diffs.append(seg_diffs)

    seg_diffs = tf.concat(directional_diffs, axis=-1) # [BT,H,W,4]
    seg_diffs = tf.one_hot(seg_diffs, depth=max_segments, axis=3, dtype=tf.int32) # [BT,H,W,N,4]
    seg_diffs = tf.reduce_max(seg_diffs, axis=-1) # [BT,H,W,N] hot for any of N bordering segments

    # now aggregate across segments
    N = max_segments
    Ntotal = tf.reduce_sum(num_segments)
    offsets = tf.cumsum(num_segments, axis=0, exclusive=True)
    seg_border_mats = tf.math.unsorted_segment_sum(
        tf.reshape(seg_diffs, [-1,N]),
        # segment_ids=tf.reshape(segment_ids + offsets[:,tf.newaxis,tf.newaxis], [-1]),
        segment_ids=tf.reshape(segment_ids_map + offsets[:,tf.newaxis,tf.newaxis], [-1]),
        num_segments=Ntotal) # [Ntotal,N]
    seg_border_mats = seg_border_mats >= num_pixels_thresh

    scatter_inds = tf.stack(inds_from_num_segments(num_segments, N), axis=-1)
    seg_border_mats = tf.scatter_nd(
        scatter_inds,
        seg_border_mats,
        shape=[B*T,N,N]
    )

    # ensure border mats are symmetric
    seg_border_mats = tf.logical_or(seg_border_mats, tf.transpose(seg_border_mats, [0,2,1]))

    # reshape if necessary
    if len(seg_shape) == 4:
        seg_diffs = tf.reshape(seg_diffs, [B,T,H,W,N])
        seg_border_mats = tf.reshape(seg_border_mats, [B,T,N,N])

    return seg_diffs, num_segments, seg_border_mats


def compute_segment_edges(segment_ids, max_segments=32, thresh=2.0):
    '''
    segment_ids: [B,[T],H,W] <tf.int32>
    max_segments: <int> max number of segments per map

    returns
    segment_edges: [B,[T],H,W]
    '''
    assert segment_ids.dtype == tf.int32
    seg_shape = segment_ids.shape.as_list()
    if len(seg_shape) == 4:
        B,T,H,W = seg_shape
        segment_ids = tf.reshape(segment_ids, [B*T,H,W])
    elif len(seg_shape) == 3:
        B,H,W = seg_shape
        T = 1
    else:
        raise ValueError("segment_ids must be shape [B,[T],H,W]")

    # preproc
    segment_edges = segment_ids - tf.reduce_min(segment_ids, axis=[1,2], keepdims=True)
    segment_edges = tf.one_hot(segment_edges, depth=max_segments, dtype=tf.float32) # [B*T,H,W,Nmax]

    # edge filter and thresh
    segment_edges = tf.image.sobel_edges(segment_edges)
    segment_edges = tf.sqrt(tf.reduce_sum(tf.square(segment_edges), axis=-1))
    segment_edges = tf.cast(tf.reduce_any(segment_edges > thresh, axis=-1), tf.float32)

    if len(seg_shape) == 4:
        segment_edges = tf.reshape(segment_edges, [B,T,H,W])
    return segment_edges

def compute_image_edges(images, resize=None, image_preproc=None, to_hsv=False, thresh=0.2):
    '''
    images: [B,[T],H,W,C] <tf.?>
    resize: [H0,W0] optional resize
    thresh: if max edge filter value is above thresh, pixel will be called an edge
    '''

    assert images.dtype in [tf.uint8, tf.float32]
    if images.dtype == tf.uint8:
        images = tf.cast(images, tf.float32) / 255.

    im_shape = images.shape.as_list()
    if len(im_shape) == 5:
        B,T,H,W,C = im_shape
        images = tf.reshape(images, [B*T,H,W,C])
    elif len(im_shape) == 4:
        B,H,W,C = im_shape
        T = 1
    else:
        raise ValueError("images must have shape [B,[T],H,W,C]")

    if resize is not None:
        Hout, Wout = resize
        if (Hout != H) or (Wout != W):
            images = tf.image.resize_images(images, [Hout,Wout])

    if image_preproc is not None:
        images = image_preproc(images)
    elif to_hsv:
        images = tf.image.rgb_to_hsv(images)

    image_edges = tf.image.sobel_edges(images)
    image_edges = tf.sqrt(tf.reduce_sum(tf.square(image_edges), axis=-1))
    image_edges = tf.cast(
        tf.reduce_max(image_edges, axis=-1) > thresh,
        tf.float32)
    if len(im_shape) == 5:
        image_edges = tf.reshape(image_edges, [B,T,Hout,Wout])
    return image_edges

def compute_sobel_features(ims, norm=255., normalize_range=False, to_rgb=False, eps=1e-6):
    '''
    Takes in a [B,T,H,W,C] image and computes the following features:
    sobel_x: C features
    sobel_y: C features
    sobel_mag: 1 feature
    sobel_angle: 1 feature

    for a total of 2C + 2 features. Therefore outputs a tensor of
    [B,T,H,W,2C+2]
    '''
    assert ims.dtype in [tf.uint8, tf.int32, tf.float32]
    if ims.dtype != tf.float32:
        ims = tf.cast(ims, tf.float32) / tf.constant(norm, tf.float32)

    shape = ims.shape.as_list()
    if len(shape) == 5:
        B,T,H,W,C = shape
        ims = tf.reshape(ims, [B*T] + shape[2:])
    elif len(shape) == 4:
        B,H,W,C = shape
        T = 1

    edges = tf.image.sobel_edges(ims)
    edges_y = edges[...,0] # [BT,H,W,C]
    edges_x = -edges[...,1]
    edges_mag = tf.sqrt(tf.square(edges_x) + tf.square(edges_y) + eps)
    edges_mag_sum = tf.reduce_sum(edges_mag, axis=-1, keepdims=True) # [BT,H,W,1]

    edges_y_sum = tf.reduce_sum(edges_y, axis=-1, keepdims=True) # [BT,H,W,1]
    edges_x_sum = tf.reduce_sum(edges_x, axis=-1, keepdims=True)
    edges_ang = tf.atan2(edges_y_sum, edges_x_sum + eps) / tf.constant(np.pi, tf.float32)


    if normalize_range:
        def _nrange(x):
            xmin = tf.reduce_min(x, axis=[1,2,3], keepdims=True)
            xmax = tf.reduce_max(x, axis=[1,2,3], keepdims=True)
            xnorm = (x - xmin) / (xmax - xmin + eps)
            return xnorm
        edges_x = _nrange(edges_x)
        edges_y = _nrange(edges_y)
        edges_mag = _nrange(edges_mag)

    if to_rgb:
        edges_hue = (edges_ang + 1.0) / 2.0 # in [0.,1.]
        edges_val = 1. / (1. + edges_mag_sum)
        edges_val = 1.0 - edges_val # in [0., 1.]
        edges_hsv = tf.concat([edges_hue, edges_val, edges_val], axis=-1) # [BT,H,W,3]
        edges_rgb = tf.image.hsv_to_rgb(edges_hsv)
        edges_feats = tf.concat([edges_x, edges_y, edges_mag, edges_rgb], axis=-1) # [BT,H,W,3C+3]
    else:
        edges_feats = tf.concat([edges_x, edges_y, edges_mag, edges_ang, edges_mag_sum, edges_mag_sum], axis=-1) # [BT,H,W,3C+3]

    if len(shape) == 5:
        edges_feats = tf.reshape(edges_feats, [B,T,H,W,-1])

    return edges_feats

def find_false_edges(segments, image_pred, image_gt, valid_nodes, thresh_gate=0.25, thresh=0.25, dist_func=tf.abs, num_pixels_thresh=2, **kwargs):
    '''
    find the edges that should be 0: between bordering nodes where residual errors differ and are high for at least one node
    '''
    # shapes
    N = valid_nodes.shape.as_list()[-2]

    # first find nodes where errors are high
    image_error = image_pred - image_gt
    err_gate = tf.abs(image_error) # [B,[T],H,W,C]
    err_gate = nodes_from_segments_and_feature_map(segments, err_gate, max_segments=N) # [B,[T],N,C]
    err_gate = tf.reduce_sum(err_gate, axis=-1, keepdims=True) > thresh_gate
    err_gate = tf.logical_or(err_gate, tf.transpose(err_gate, [0,1,3,2])) # [B,[T],N,N]
    err_gate = tf.logical_and(
        err_gate,
        tf.logical_and(valid_nodes, tf.transpose(valid_nodes, [0,1,3,2]))) # [B,[T],N,N] node pairs with err > thresh_gate

    # now compute distance in errors between adjacent nodes
    nodes_error = nodes_from_segments_and_feature_map(segments, image_error, max_segments=N) # [B,[T],N,C]
    err_dists = tf.reduce_sum(dist_func(
        tf.expand_dims(nodes_error, -2) - tf.expand_dims(nodes_error, -3)), axis=-1) # [B,[T],N,N]

    false_edges = tf.logical_and(err_dists > thresh, err_gate)
    _,_,bordering_nodes = compute_segment_border_mats(segments, edge_pixels=None, max_segments=N, num_pixels_thresh=num_pixels_thresh, **kwargs)
    false_edges = tf.logical_and(false_edges, bordering_nodes) # [B,[T],N,N] <tf.bool>
    return false_edges

def find_true_edges(
        segments_lvl0,
        segments_lvl1,
        image_gt,
        max_segments=32,
        image_edges_thresh=0.2,
        image_edges_weight=1.0,
        segment_edges_thresh=2.0,
        num_pixels_thresh=2,
        to_hsv=False,
        invert=False,
        image_preproc=None,
        **kwargs):
    '''
    Find the edges that should be 1: between bordering nodes where there's a "hallucinated" edge (i.e. an edge between segments that's not in the image features)
    '''
    H,W = segments_lvl1.shape.as_list()[-2:]
    sign = tf.constant(1.0, tf.float32)
    sign = -sign if invert else sign
    # compute segment edges and image edges
    seg_edges = compute_segment_edges(segments_lvl1, max_segments=max_segments, thresh=segment_edges_thresh)
    image_edges = compute_image_edges(image_gt, resize=[H,W], image_preproc=image_preproc, to_hsv=to_hsv, thresh=image_edges_thresh)
    hallucinated_edges = tf.cast(sign * (seg_edges - image_edges * image_edges_weight) > 0.0, tf.int32)

    # find where hallucinated edges overlap with segments_lvl0 borders
    _, _, true_edges = compute_segment_border_mats(segments_lvl0, hallucinated_edges, max_segments=max_segments, num_pixels_thresh=num_pixels_thresh, **kwargs)
    return true_edges

def build_edge_corrections(
        segments_lvl0,
        segments_lvl1,
        img_lvl1,
        img_gt,
        valid_nodes,
        num_segments_lvl1=32,
        thresh_gate0=0.25,
        thresh0=0.25,
        seg_edges_thresh=2.0,
        img_edges_thresh=0.3,
        img_edges_thresh_false=0.3,
        img_edges_weight=1.0,
        num_pixels_thresh=2,
        recon_false_edges=True,
        border_false_edges=False,
        to_hsv=False,
        **kwargs
):
    '''
    segments_lvl0: [B,[T],H,W] <tf.int32> labels for lvl0 segments
    segments_lvl1: [B,[T],H,W] <tf.int32> labels for lvl1 segments
    img_lvl0: [B,[T],H,W,C] <tf.float32> an image decoded from the lvl0 segments and nodes
    img_lvl1: [B,[T],H,W,C] <tf.float32> an image decoded from the lvl1 segments and nodes
    img_gt: [B,[T],Him,Wim,C] <?> the gt image; will be preprocessed if not correct size and tf.dtype,
    valid_nodes: [B,[T],N,1] <tf.bool> indicating which nodes are valid (and how many)
    '''

    # check and preprocess
    assert segments_lvl0.dtype == tf.int32
    assert segments_lvl1.dtype == tf.int32
    assert segments_lvl0.shape == segments_lvl1.shape
    assert valid_nodes.dtype == tf.bool
    assert valid_nodes.shape.as_list()[-1] == 1
    if img_gt.dtype != tf.float32:
        assert img_gt.dtype == tf.uint8
        img_gt = tf.cast(img_gt, tf.float32) / 255.

    B = valid_nodes.shape.as_list()[0]
    N = valid_nodes.shape.as_list()[-2]
    valid_edges = tf.logical_and(valid_nodes, tf.transpose(valid_nodes, [0,3,2,1]))
    H,W = segments_lvl0.shape.as_list()[-2:]
    Him,Wim,C = img_gt.shape.as_list()[-3:]
    if [H,W] != [Him,Wim]:
        img_gt = tf.reshape(tf.image.resize_images(
            tf.reshape(img_gt, [-1,Him,Wim,C]), [H,W]), img_gt.shape.as_list()[:-3] + [H,W,C]) # [B,[T],H,W,C]

    # assert img_gt.shape == img_lvl1.shape, "resized gt images and decoded images must all be same shape"
    # assert img_gt.dtype == img_lvl1.dtype, "gt images and decoded images must be the same dtype"

    # find edges that should be 1
    true_edges = find_true_edges(
        segments_lvl0, segments_lvl1, img_gt,
        max_segments=N, segment_edges_thresh=seg_edges_thresh, to_hsv=to_hsv, invert=False,
        image_edges_thresh=img_edges_thresh, num_pixels_thresh=num_pixels_thresh, **kwargs)
    true_edges = tf.logical_and(true_edges, valid_edges)

    # Find edges that should be 0
    false_edges = tf.zeros_like(true_edges)
    if recon_false_edges:
        print("using recon false edges")
        false_edges = tf.logical_or(
            false_edges,
            find_false_edges(
                segments_lvl0, img_lvl1, img_gt, valid_nodes,
                thresh_gate=thresh_gate0, thresh=thresh0,
                num_pixels_thresh=num_pixels_thresh, **kwargs)
        )
    if border_false_edges:
        print("using border false edges")
        false_edges = tf.logical_or(
            false_edges,
            find_true_edges(
                segments_lvl0, segments_lvl1, img_gt, image_edges_weight=img_edges_weight,
                max_segments=N, segment_edges_thresh=seg_edges_thresh, to_hsv=to_hsv, invert=True,
                image_edges_thresh=img_edges_thresh_false, num_pixels_thresh=num_pixels_thresh, **kwargs)
        )

    # now combine into a single error signal
    return false_edges, true_edges

def update_edges(edges_prev, false_edges, true_edges, conflict_resolution='unchanged', **kwargs):
    '''
    edges_prev: [...,N,N] tf.bool
    false_edges: [...,N,N] tf.bool (True where edges should become 0)
    true_edges: [...,N,N] tf.bool (True where edges should become 1)
    '''
    conflicts = tf.logical_and(false_edges, true_edges)
    edges_next = tf.logical_or(edges_prev, true_edges)
    edges_next = tf.logical_and(edges_next, tf.logical_not(false_edges))

    if conflict_resolution == 'false':
        edges_next = tf.logical_and(edges_next, tf.logical_not(conflicts))
    elif conflict_resolution == 'true':
        edges_next = tf.logical_or(edges_next, conflicts)

    elif conflict_resolution == 'unchanged':
        edges_next = tf.where(conflicts,
                              edges_prev, # true
                              edges_next) # false
    else:
        raise ValueError("conflict_resolution must be in ['true', 'false', 'unchanged'])")

    return edges_next

def update_edges_soft(errors_prev, false_edges, true_edges, conflict_resolution='unchanged', edge_memory=0.5, **kwargs):
    '''
    errors_prev: in range (0,1)
    '''
    assert errors_prev.dtype == tf.float32
    update_gate = tf.cast(tf.logical_or(false_edges, true_edges), tf.float32)
    conflicts = tf.cast(tf.logical_and(false_edges, true_edges), tf.float32)
    false_edges = tf.cast(false_edges, tf.float32)
    true_edges = tf.cast(true_edges, tf.float32)

    update = 0.5 * (1.0 - update_gate) # 0.5 everywhere without updates
    update += true_edges # 1.0 in places with true_edges
    update = update * (1.0 - false_edges) # 0.0 in places with false_edges

    if conflict_resolution == 'false':
        update = update * (1.0 - conflicts)
    elif conflict_resolution == 'true':
        update = update*(1.0-conflicts) + conflicts
    elif conflict_resolution == 'unchanged':
        update = update*(1.0-conflicts) + 0.5*conflicts

    errors_next = errors_prev*edge_memory + update*(1.0 - edge_memory)
    return errors_next


def add_batch_time_node_index(x):
    """
    Expands index vector x by batch index, time index and node dimension index

    :param x: Tensor(shape) index vector

    :return x: Tensor(new_shape), index vector with batch index, time index and
        1st dimension index
    """
    shape = x.get_shape().as_list()
    batch_index = tf.tile(tf.reshape(tf.range(shape[0]), \
            [shape[0]] + [1] * (len(shape) - 1)),
            [1] + shape[1:])
    time_index = tf.tile(tf.reshape(tf.range(shape[1]), \
            [1, shape[1]] + [1] * (len(shape) - 2)),
            [shape[0], 1] + shape[2:])
    node_index = tf.tile(tf.reshape(tf.range(shape[2]), \
            [1, 1, shape[2]] + [1] * (len(shape) - 3)),
            [shape[0], shape[1], 1] + shape[3:])
    x = tf.stack([
        batch_index,
        time_index,
        node_index,
        tf.cast(x, tf.int32)], axis = len(shape))
    return x


def add_batch_time_index(x):
    """
    Expands index vector x by batch index and time index

    :param x: Tensor(shape) index vector

    :return x: Tensor(new_shape), index vector with batch index and time index
    """
    shape = x.get_shape().as_list()
    batch_index = tf.tile(tf.reshape(tf.range(shape[0]), \
            [shape[0]] + [1] * (len(shape) - 1)),
            [1] + shape[1:])
    time_index = tf.tile(tf.reshape(tf.range(shape[1]), \
            [1, shape[1]] + [1] * (len(shape) - 2)),
            [shape[0], 1] + shape[2:])
    x = tf.stack([
        batch_index,
        time_index,
        tf.cast(x, tf.int32)], axis = len(shape))
    return x

def add_batch_index(x):
    """
    Expands index vector x by batch index

    :param x: Tensor(shape) index vector

    :return x: Tensor(new_shape), index vector with batch index
    """
    shape = x.get_shape().as_list()
    batch_index = tf.tile(tf.reshape(tf.range(shape[0]), \
            [shape[0]] + [1] * (len(shape) - 1)),
            [1] + shape[1:])
    x = tf.stack([
        batch_index,
        tf.cast(x, tf.int32)], axis = len(shape))
    return x

def local_to_global_adj(local_adj, size, affinities=None):
    B, N, K = local_adj.shape.as_list()
    H, W = size
    ksize = int(K ** 0.5)
    k = int((ksize - 1) / 2)
    edge_inds = tf.cast(tf.where(local_adj), tf.int32)  # [?,3]
    b_inds, n_inds, k_inds = tf.split(edge_inds, [1, 1, 1], axis=-1)
    h_inds = tf.math.floordiv(n_inds, W)
    w_inds = tf.math.floormod(n_inds, W)
    dh = tf.math.floordiv(k_inds, ksize) - k  # in [-k,k]
    dw = tf.math.floormod(k_inds, ksize) - k  # in [-k,k]
    in_view = tf.logical_and(
        tf.logical_and(h_inds + dh >= 0, h_inds + dh < H),
        tf.logical_and(w_inds + dw >= 0, w_inds + dw < W))  # [?,1]
    neighb_inds = (h_inds + dh) * W + (w_inds + dw)
    neighb_inds = tf.where(in_view, neighb_inds, H * W * tf.ones_like(neighb_inds))
    scatter_inds = tf.concat([b_inds, n_inds, neighb_inds], axis=-1)

    # if affinities is a float tensor
    if affinities is not None:
        assert affinities.shape == local_adj.shape, (affinities, local_adj)
        updates = tf.gather_nd(affinities, edge_inds)
    else:
        updates = tf.ones_like(scatter_inds[:,0])

    global_adj = tf.scatter_nd(indices=scatter_inds,
                               updates=updates,
                               shape=tf.constant([B, N, N+1]))
    global_adj = global_adj[...,:N]
    global_adj = tf.cast(global_adj, tf.float32)
    return global_adj

def normalize_adj(adj, eps=1e-6):
    deg_inv_sqrt = tf.reduce_sum(adj, axis=-1)
    deg_inv_sqrt = tf.clip_by_value(deg_inv_sqrt, 1.0, tf.float32.max)  # clamp min = 1.0
    deg_inv_sqrt = 1. / tf.sqrt(deg_inv_sqrt + eps)
    adj = tf.expand_dims(deg_inv_sqrt, -1) * adj * tf.expand_dims(deg_inv_sqrt, -2)
    return adj
