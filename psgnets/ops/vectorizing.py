from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from psgnets.ops.utils import *
from psgnets.ops.dimensions import DimensionDict, OrderedDict

PRINT = False

def aggregate_mean_and_var_features(
        features, segment_ids,
        num_segments=None,
        dimension_dict=None,
        max_segments=256,
        augmentation_kernels=[],
        attrs_to_augment=None,
        agg_vars=False,
        **kwargs
):
    '''
    Aggregate features from segment ids, e.g. as computed by label prop
    for the C original feature channels in features, aggregate by averaging.

    Also compute the centroid (average (h,w) location) for each segment and optional
    aggregate statistics.

    Then reformat summary features into a [B,N_max,C'] tensor where examples with
    fewer than N_max segments are padded with zeros and indicated as fake.

    features: [B,H,W,C] <tf.float32>
    segment_ids: [B,HW] <tf.int32>
    num_segments: [B] <tf.int32>
    max_segments_per_example: int Nmax, determines shape of output tensor

    outputs:
    summary_features: [B,Nmax,C'] where C' = C + 2 (h,w coordinate means) + 1 (segment area) + kC (optional derived features)

    '''
    BT,HW,_,rank = dims_and_rank(segment_ids)
    # get num_segments and monotnic segment_ids
    num_segments, segment_ids = num_segments_from_segment_ids(
        segment_ids, num_segments)

    # augment features and get their named Dims
    features, Dims = augment_features_and_dims(features, augmentation_kernels, dimension_dict, attrs_to_augment)

    # append "extra" channels
    B,H,W,C = features.shape.as_list()
    ones = tf.ones([B,H,W,1], dtype=tf.float32)
    b_inds = tf.reshape(tf.range(B, dtype=tf.float32), [B,1,1,1]) * ones
    hw_inds = coordinate_ims(B, 1, [H,W])[:,0] # [B,H,W,2]
    features = tf.concat([ones, b_inds, hw_inds, features], axis=-1)
    features = tf.reshape(features, [B*HW,C+4])

    if PRINT and (max_segments > 1):
        num_segments = tf.Print(num_segments, [num_segments, tf.constant(max_segments)], message="num_segments")

    # now aggregate over segments
    Ntotal = tf.reduce_sum(num_segments)
    mean_feats = tf.math.unsorted_segment_sum(
        features, tf.reshape(segment_ids, [-1]), num_segments=Ntotal)
    areas, mean_feats = tf.split(mean_feats, [1,-1], axis=-1)
    mean_feats = mean_feats / tf.maximum(1.0, areas)
    b_inds, hw_centroids, mean_feats = tf.split(mean_feats, [1,2,-1], axis=-1) # first two channels now centroid

    # if computing variances, do so here
    if agg_vars:
        ss_feats = tf.math.unsorted_segment_sum(
            tf.square(features[...,4:]), tf.reshape(segment_ids, [-1]), num_segments=Ntotal)
        var_feats = (ss_feats / tf.maximum(1.0, areas)) - tf.square(mean_feats)
        mean_feats = tf.concat([mean_feats, var_feats], axis=-1)
        Dims.insert_from(Dims.copy(suffix='_var'))

    # scatter back to rectangular tensor
    Nmax = max_segments
    b_inds = tf.cast(b_inds, tf.int32)
    s_inds = within_example_segment_ids(num_segments, Nmax)[:,tf.newaxis] # [Ntotal,1]
    areas = areas / HW
    valid = tf.ones([Ntotal,1], dtype=tf.float32)
    nodes = Dims.extend_vector(
        tensor_list=[('hw_centroids', hw_centroids), ('areas', areas), ('valid', valid)],
        base_tensor=mean_feats)

    assert nodes.shape.as_list()[-1] == Dims.ndims, (nodes.shape, Dims.ndims)
    nodes = tf.scatter_nd(
        tf.concat([b_inds,s_inds], axis=-1),
        nodes, shape=[B,Nmax,Dims.ndims])
    nodes_valid = nodes[...,-1:]
    nodes_valid = tf.cast(tf.logical_and(nodes_valid > 0.5, nodes_valid < 1.5), tf.float32)
    nodes = tf.concat([nodes[...,:-1]*nodes_valid, nodes_valid], axis=-1) # valid is 0.0 or 1.0, mask out all invalid attrs

    return nodes, num_segments, Dims

def augment_features_and_dims(features, kernel_list, dimension_dims=None, attrs_to_augment=None):
    '''
    Applies each of a list of conv kernels to all (or selected) channels in features
    '''
    B,H,W,C = features.shape.as_list()
    if not isinstance(kernel_list, (tuple, list)):
        kernel_list = [kernel_list]

    # figure out which inputs to augment
    if not isinstance(dimension_dims, DimensionDict):
        dimension_dims = DimensionDict({'inputs':C})
        attrs_to_augment = ['inputs']
        inp_features = features
    elif attrs_to_augment is None:
        latent_dims = {'latent'+str(i):dims for i,dims in enumerate(dimension_dims.get_unassigned_dims())}
        dimension_dims.update(latent_dims)
        attrs_to_augment = dimension_dims.sort().keys()
    inp_features = dimension_dims.get_tensor_from_attrs(features, attrs_to_augment)
    C = inp_features.shape.as_list()[-1]

    aug_features = []
    for i,kernel in enumerate(kernel_list):
        # supplied kernels may be numpy arrays
        if not isinstance(kernel, tf.Tensor):
            kernel = tf.constant(kernel, tf.float32)
        if len(kernel.shape.as_list()) == 2: # [H,W] kernel
            kernel = tf.tile(kernel[:,:,tf.newaxis,tf.newaxis], [1,1,C,1]) # [H,W,C,1]
            op = tf.nn.depthwise_conv2d
            cmult = 1
        elif len(kernel.shape.as_list()) == 3: # [H,W,Cout] kernels
            kernel = tf.tile(kernel[:,:,tf.newaxis,:], [1,1,C,1]) # [H,W,C,Cout]
            op = tf.nn.depthwise_conv2d
            cmult = kernel.shape.as_list()[-1]
        else:
            op = tf.nn.conv2d
            raise NotImplementedError()

        # op
        new_feats = op(inp_features, kernel, strides=[1,1,1,1], padding='SAME') # [B,H,W,C*cmult]
        new_feats = tf.transpose(tf.reshape(new_feats, [B,H,W,C,cmult]), [0,1,2,4,3]) # now [B,H,W,cmult,C]
        new_feats = tf.split(new_feats, [d[1]-d[0] for d in [dimension_dims[attr] for attr in attrs_to_augment]], axis=-1)
        new_feats = tf.concat([tf.reshape(tens, [B,H,W,-1]) for tens in new_feats], axis=-1)

        aug_dims = [(attr+'_aug'+str(i)+str(j), dimension_dims[attr][1] - dimension_dims[attr][0])
                    for attr in attrs_to_augment for j in range(cmult)]
        dimension_dims.insert_from(aug_dims)
        aug_features.append(new_feats)

    return tf.concat([features] + aug_features, axis=-1), dimension_dims

def agg_child_attrs_within_parents(nodes, labels, num_segments,
                                   valid_nodes=None,
                                   max_labels=32,
                                   dimension_dict=None,
                                   dim_suffix='',
                                   agg_ranges=False, agg_vars=False,
                                   rectangular_output=True, remove_valid=True,
                                   **kwargs
):
    '''
    aggregation of attrs in nodes (children) within each parent (as indicated by labels)

    nodes: [B,N,D] children nodes with last dimension indicating whether it's valid
    labels = [Ntotal = sum(valid_nodes)] <tf.int32> increasing monotonically across batch
    num_segments: [B] how many parent segments there are per example
    '''
    B,N,D = nodes.shape.as_list()
    Ntotal = tf.reduce_sum(num_segments)

    if dimension_dict is None:
        Dims = DimensionDict(D, {'c2p_agg_inputs':[0,D]})
    else:
        Dims = dimension_dict.copy(suffix=dim_suffix)
        assert Dims.ndims in [D,D-1], (Dims.ndims, D)

    if PRINT:
        labels = tf.Print(labels, [tf.reduce_min(labels), tf.reduce_max(labels), tf.shape(labels), num_segments], message='labels')

    # aggregate mean attrs for valid nodes
    if valid_nodes is None:
        valid_nodes = nodes[...,-1]
    valid_inds = tf.where(valid_nodes > 0.5)
    real_nodes = tf.gather_nd(nodes, valid_inds) # [?,D]
    sum_attrs = tf.math.unsorted_segment_sum(real_nodes, labels, num_segments=Ntotal)
    if remove_valid:
        sum_attrs, num_nodes_attr = tf.split(sum_attrs, [-1,1], axis=-1) # D
        Dims.ndims -= 1
    else:
        num_nodes_attr = sum_attrs[...,-1:]
    mean_attrs = sum_attrs / tf.maximum(num_nodes_attr, 1.0)

    # add new attributes via aggregation statistics
    valid_attr = tf.ones_like(num_nodes_attr)
    attrs_list = [mean_attrs, num_nodes_attr, valid_attr]

    origDims = Dims.copy(suffix='')
    if remove_valid:
        real_nodes = real_nodes[...,:-1]
    if agg_ranges:
        min_attrs = tf.math.unsorted_segment_min(real_nodes, labels, num_segments=Ntotal)
        max_attrs = tf.math.unsorted_segment_max(real_nodes, labels, num_segments=Ntotal)
        range_attrs = max_attrs - min_attrs
        attrs_list.insert(-2, range_attrs)
        Dims.insert_from(origDims.copy(suffix='_ranges'))
    if agg_vars:
        ss_attrs = tf.math.unsorted_segment_mean(tf.square(real_nodes), labels, num_segments=Ntotal)
        var_attrs = ss_attrs - tf.square(mean_attrs)
        attrs_list.insert(-2, var_attrs)
        Dims.insert_from(origDims.copy(suffix='_vars'))

    # now concat all aggregated attrs
    agg_attrs = tf.concat(attrs_list, axis=-1) # [Ntotal, D+D*agg_ranges+D*agg_vars+2]
    b_inds, n_inds = inds_from_num_segments(num_segments, max_labels)
    inds = tf.stack([b_inds, n_inds], axis=-1) # [Ntotal,2]
    if rectangular_output:
        # reshape into a rectangular tensor
        # assert Dims.ndims + 2 == agg_attrs.shape.as_list()[-1], (Dims.ndims, agg_attrs)
        Dims['num_nodes'] = 1
        Dims['valid'] = 1
        agg_nodes = tf.scatter_nd(inds, agg_attrs, shape=[B,max_labels,agg_attrs.shape.as_list()[-1]])
        valid_attr = tf.cast(tf.logical_and(agg_nodes[...,-1:] < 1.1, agg_nodes[...,-1:] > 0.9), tf.float32)
        agg_nodes = tf.concat([agg_nodes[...,:-1], valid_attr], axis=-1)
        return agg_nodes, Dims
    else:
        return agg_attrs, inds, Dims

def compute_attr_spatial_moments(
        child_nodes,
        parent_segment_map,
        child_to_parent_labels,
        parent_num_segments=None,
        valid_child_nodes=None,
        features=None,
        features_dimension_dict=None,
        nodes_dimension_dict=None,
        labels_monotonic=False,
        max_parent_nodes=64,
        agg_features=False,
        agg_spatial_ranges=False,
        agg_spatial_vars=False,
        remove_valid=True,
        hw_attr='hw_centroids',
        hw_dims=[-4,-2],
        **kwargs
):
    '''
    child_nodes: [B,N,D] the "spatial" or base nodes to be summarized
    parent_segment_map: [B,H,W] the map indicating which pixel belongs to which summary node
    child_to_parent_labels: [R = sum(num_valid_child_nodes)] index label of each child's parent node
    parent_num_segments: [B] the number of summary nodes in each example
    features: [B,H,W,C] optional features to aggregate over the summary_segment_ids

    returns
    summary_spatial_attrs: [B,M,nc*D + C + 2] where M := num_summary_segments, nc := 2 if computing linear spatial attributes
    AggDims: the new dimensions of the spatial moments
    '''
    B,N,D = child_nodes.shape.as_list()
    if len(parent_segment_map.shape) == 4:
        _B,T,H,W = parent_segment_map.shape.as_list()
    elif len(parent_segment_map.shape) == 3:
        _B,H,W = parent_segment_map.shape.as_list()
        T = 1
    assert B == _B*T, (B,_B,T)

    if (features is None) or not agg_features:
        features = tf.zeros([B,H,W,0], tf.float32)
    elif features_dimension_dict is None:
        features_dimension_dict = DimensionDict({'features_mean': features.shape.as_list()[-1]})

    M = max_parent_nodes
    if valid_child_nodes is None:
        valid_child_nodes = tf.cast(child_nodes[...,-1] > 0.5, tf.float32) # [B,N]
    num_valid_child_nodes = tf.cast(tf.reduce_sum(valid_child_nodes, axis=1), tf.float32) # [B] in range [0,N]
    R = tf.reduce_sum(tf.cast(valid_child_nodes > 0.5, tf.float32))

    # setup Dims
    if nodes_dimension_dict is None:
        Dims = DimensionDict(child_nodes.shape.as_list()[-1])
        Dims.update({'inputs': [0,Dims.ndims], hw_attr:hw_dims})
    else:
        Dims = nodes_dimension_dict
        Dims.update({'spatial_latent'+str(i):dims for i,dims in enumerate(Dims.get_unassigned_dims())})
        if hw_attr not in Dims.keys():
            Dims[hw_attr] = hw_dims

    # get summary centroids from map
    parent_features, parent_num_segments, DimsFeats = aggregate_mean_and_var_features(
        features=features,
        segment_ids=parent_segment_map,
        num_segments=parent_num_segments,
        dimension_dict=features_dimension_dict,
        max_segments=M
    )
    parent_features, parent_hws, parent_areas, _ = tf.split(parent_features, [-1,2,1,1], axis=-1) # [B,M,?]

    if not labels_monotonic:
        _, child_to_parent_labels = num_segments_from_segment_ids(parent_segment_map, parent_num_segments)
        child_to_parent_labels = tf.reshape(child_to_parent_labels, [-1])

    # compute the difference between each node's hw coordinates and its summary centroid
    parent_inds = labels_list_to_parent_inds( # [B,N,2] of (b_ind, parent_n_ind) tuples
        child_to_parent_labels, parent_num_segments, valid_child_nodes, labels_monotonic=labels_monotonic, max_parent_nodes=M)

    # monotonic
    # child_to_parent_labels = parent_inds_to_labels_list(
    #     parent_inds[...,-1], valid_child_nodes, parent_num_segments, monotonic=True)

    parent_hws_per_child = tf.gather_nd(parent_hws, parent_inds) # [B,N,2]
    child_hws = Dims.get_tensor_from_attrs(child_nodes, hw_attr)
    child_node_delta_hws = (child_hws - parent_hws_per_child) * valid_child_nodes[...,tf.newaxis]  # [B,N,2]

    # now compute spatial moments of all
    child_node_spatial_attrs = child_nodes[:,:,tf.newaxis,:] * child_node_delta_hws[...,tf.newaxis] # [B,N,2,D]
    child_node_spatial_attrs = tf.reshape(child_node_spatial_attrs, [B,N,2*D]) # 2D channels go [attrs*dH, attrs*dW]

    AggDims = Dims.copy(suffix='_hmoment')
    AggDims.insert_from(Dims.copy(suffix='_wmoment'))
    assert AggDims.ndims == 2*D
    child_node_spatial_attrs = tf.concat([child_node_spatial_attrs, valid_child_nodes[...,tf.newaxis]], axis=-1) # 2D+1 dims for now

    parent_spatial_moments, AggDims = agg_child_attrs_within_parents(
        child_node_spatial_attrs, child_to_parent_labels, parent_num_segments,
        max_labels=M, dimension_dict=AggDims, remove_valid=remove_valid,
        agg_ranges=agg_spatial_ranges, agg_vars=agg_spatial_vars) # [B,M,2D+2D*(agg_ranges + agg_vars) + 2]
    if remove_valid:
        parent_spatial_moments = parent_spatial_moments[...,:-2] # don't need the last 2 channels num_nodes, valid
    else:
        parent_spatial_moments = parent_spatial_moments[...,:-3]
    AggDims.delete('num_nodes')
    AggDims.delete('valid')
    AggDims.ndims -= 2

    assert AggDims.ndims == 2*D + 2*D*(int(agg_spatial_ranges) + int(agg_spatial_vars))

    # update all dims before concating and returning
    if parent_features.shape.as_list()[-1] > 0:
        DimsFeats.delete_from(['hw_centroids', 'areas', 'valid'])
        DimsFeats.ndims -= 4
        AggDims.insert_from(DimsFeats, position=0)
    parent_spatial_attrs = tf.concat([parent_features, parent_spatial_moments], axis=-1) # [B,M,2D+2D*(agg_ranges+agg_vars)+C]

    return parent_spatial_attrs, parent_hws, parent_areas, parent_inds[...,-1], AggDims

def compute_border_attributes(
        nodes,
        segment_map,
        features=None,
        shape_feats=True,
        resize=None,
        hw_dims=[-4,-2],
        border_thresh=0.05,
        eps=1e-6,
        divide_by_quadrant=True,
        divide_features_by_quadrant=False,
        **kwargs
):
    '''
    Compute statistics along the 1D "boundary" of each segment in the 2D segment_map
    '''
    B,M,D = nodes.shape.as_list()
    _B,T,H,W = segment_map.shape.as_list()
    assert B == _B*T, (B,_B,T)
    BT,HW,_,rank = dims_and_rank(segment_map)
    valid_nodes = nodes[...,-1:]

    if resize is not None:
        assert (H % resize[0] == 0) and (W % resize[1]) == 0
        strides = [H // resize[0], W // resize[1]]
        H = H // strides[0]
        W = W // strides[1]
        segment_map = segment_map[:,:,::strides[0],::strides[1]]
    else:
        strides = [1,1]

    # features to aggregate
    if features is not None:
        F = features.shape.as_list()[-1]
        features = features[:,::strides[0],::strides[1]]
    else:
        F = 0

    # convert to one-hot segment maps
    segment_map = segment_map - tf.reduce_min(segment_map, axis=[ax for ax in range(1,rank)], keepdims=True) # start at 0 per example
    segment_map = tf.reshape(segment_map, [BT,H,W])
    segment_map = tf.where(segment_map < M,
                           segment_map, tf.ones_like(segment_map)*(M-1)) # remove segments over the limit
    segment_ims = tf.one_hot(segment_map, depth=M, axis=-1, dtype=tf.float32) # [B,H,W,M]
    border_ims = tf.image.sobel_edges(segment_ims) # [B,H,W,M,2]
    border_angles = tf.atan2(border_ims[...,1:2], border_ims[...,0:1] + eps) # angle between [-pi, pi] on border pix
    border_valid = tf.cast(tf.reduce_sum(tf.square(border_ims), axis=-1, keepdims=True) > border_thresh, tf.float32) # [B,H,W,M,1] pix with borders

    # add xy channels and other features along the borders
    ones = tf.ones([B,H,W,M,1], tf.float32)
    hs = tf.reshape(tf.range(H, dtype=tf.float32), [1,H,1,1,1]) * ones
    hs = (hs / ((H - 1.0)/2.0)) - 1.0
    ws = tf.reshape(tf.range(W, dtype=tf.float32), [1,1,W,1,1]) * ones
    ws = (ws / ((W - 1.0)/2.0)) - 1.0
    if F:
        features = features[:,:,:,tf.newaxis] * border_valid # [B,H,W,M,F] only valid where there are borders
        border_features = tf.concat([ws, -hs, border_angles, border_valid, features], axis=-1, name="concat_border_features") # [B,H,W,M, F + 4] each is (xs, ys, angles, valid)
    else:
        border_features = tf.concat([ws, -hs, border_angles, border_valid], axis=-1, name="concat_border_features") # [B,H,W,M, F + 4] each is (xs, ys, angles, valid)

    # features should be relative to node centroids
    node_centroids = nodes[...,hw_dims[0]:hw_dims[1]] # [B,M,2]
    node_centroids_xy = tf.concat([node_centroids[...,1:2], -node_centroids[...,0:1], tf.zeros([B,M,2+F], tf.float32)], axis=-1) # [B,M,4+F]
    border_features = border_features - node_centroids_xy[:,tf.newaxis,tf.newaxis] # [B,H,W,M,4+F] where F is 0 if not integrating along boundary
    if divide_by_quadrant:
        q1 = tf.logical_and(border_features[...,0:1] >= 0., border_features[...,1:2] >= 0.)
        q2 = tf.logical_and(border_features[...,0:1] < 0., border_features[...,1:2] >= 0.)
        q3 = tf.logical_and(border_features[...,0:1] < 0., border_features[...,1:2] < 0.)
        q4 = tf.logical_and(border_features[...,0:1] >= 0., border_features[...,1:2] < 0.)
        quad_masks = tf.stack([tf.cast(q, tf.float32) for q in [q1,q2,q3,q4]], axis=-2) * border_valid[...,tf.newaxis] # [B,H,W,M,4,1]
    else:
        quad_masks = border_valid[...,tf.newaxis] # [B,H,W,M,1,1]

    # denominator
    num_border_px = tf.maximum(1.0, tf.reduce_sum(quad_masks, axis=[1,2], keepdims=False)) # [B,M,4/1,1]

    attr_list = []
    border_features = tf.expand_dims(border_features, -2) # [B,H,W,M,1,F+4]
    if shape_feats:
    # now aggregate features
        mean_xy_offset = tf.reduce_sum(border_features[...,0:2]*quad_masks, axis=[1,2]) / num_border_px # [B,M,4/1,2]
        mean_d_offset = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(border_features[...,0:2]*quad_masks), axis=-1, keepdims=True) + eps), axis=[1,2]) / num_border_px #[B,M,4/1,1]
        mean_angle = tf.reduce_sum(border_features[...,2:3]*quad_masks, axis=[1,2]) / num_border_px # [B,M,4/1,1]
        var_xy_offset = tf.reduce_sum(quad_masks * tf.square(border_features[...,0:2] - mean_xy_offset[:,tf.newaxis,tf.newaxis]), axis=[1,2]) / num_border_px # [B,M,4/1,2]
        var_angle = tf.reduce_sum(quad_masks * tf.square(border_features[...,2:3] - mean_angle[:,tf.newaxis,tf.newaxis]), axis=[1,2]) / num_border_px # [B,M,4/1,1]
        attr_list += [mean_xy_offset, mean_d_offset, mean_angle, var_xy_offset, var_angle, num_border_px]

    # generic features
    if F:
        mean_feats = tf.reduce_sum(border_features[...,4:]*quad_masks, axis=[1,2]) / num_border_px
        var_feats = tf.reduce_sum(quad_masks * tf.square(border_features[...,4:] - mean_feats[:,tf.newaxis,tf.newaxis]), axis=[1,2]) / num_border_px
        x_feats = tf.reduce_sum(border_features[...,0:1]*border_features[...,4:]*quad_masks, axis=[1,2]) / num_border_px
        y_feats = tf.reduce_sum(border_features[...,1:2]*border_features[...,4:]*quad_masks, axis=[1,2]) / num_border_px
        attr_list += [mean_feats, var_feats, x_feats, y_feats]

    if not len(attr_list):
        attr_list = [num_border_px]

    border_attrs = tf.concat(attr_list, axis=-1) # [B,M,4/1,8 + 4F]
    border_attrs = tf.reshape(border_attrs, [B,M,-1]) * valid_nodes

    return border_attrs

def add_history_attributes(nodes, dimension_dict=None, attrs_list=None, prev_times=None):
    '''
    Concats attributes from previous time steps to each node and adds them as new dims
    '''
    B,T,N,D = nodes.shape.as_list()
    prev_times = prev_times or T-1
    assert prev_times < T
    if dimension_dict is None:
        Dims = DimensionDict({'inputs':D})
    else:
        Dims = dimension_dict

    # get the attrs from all time steps
    attrs_list = [k for k in Dims.sort().keys() if k in attrs_list] or Dims.sort().keys()
    attrs_all_times = Dims.get_tensor_from_attrs(
        nodes, attrs_list, concat=True) # [B,T,N,D']
    DD = attrs_all_times.shape.as_list()[-1]
    NewDims = DimensionDict(
        OrderedDict(
            [(attr, Dims[attr][1] - Dims[attr][0]) for attr in attrs_list]
        ))

    for t in range(1, prev_times+1):
        zeros = tf.zeros([B,t,N,DD], dtype=tf.float32)
        prev_attrs_t = tf.concat([zeros, attrs_all_times[:,:T-t]], axis=1) # [B,T,N,DD]
        valid_prev_t = tf.concat([
            tf.zeros([B,t,N,1], tf.float32),
            tf.ones([B,T-t,N,1], tf.float32)
        ], axis=1) # [B,T,N,1]
        nodes = tf.concat([nodes, prev_attrs_t, valid_prev_t], axis=-1) # add DD+1 new dims
        Dims.insert_from(NewDims.copy(suffix='_prev'+str(t)))
        Dims['valid_prev'+str(t)] = 1

    return nodes, Dims

if __name__ == '__main__':
    D = DimensionDict(24, {'normals':[6,9,lambda n: tf.nn.l2_normalize(n, axis=-1)],
                           'colors':[3,6, tf.image.hsv_to_rgb], 'pos':[0,3]})
    D.insert('cats', 12, 8)
    print(D.sort().items(), D.get_unassigned_dims())
    tensor = tf.random_normal(shape=[8,64,64,D.ndims])
    deriv = deriv_kernels()

    # segment_ids = tf.random_uniform(shape=[8,64,64], minval=0, maxval=8, dtype=tf.int32)
    segment_ids = tf.image.resize_images(tf.reshape(tf.range(4, dtype=tf.int32), [1,2,2,1]), [64,64], method=1)[...,0]
    segment_ids = tf.tile(segment_ids, [8,1,1])
    segment_ids = tf.reshape(segment_ids, [8,-1])
    nodes, num_segments, D = aggregate_mean_and_var_features(
        tensor, segment_ids, max_segments=16, dimension_dict=D, augmentation_kernels=deriv[...,0],
        agg_vars=True, attrs_to_augment=['normals', 'cats']
    )
    print("nodes", nodes)
    print("num_segments", num_segments)
    print("dims", D)

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sess = tf.Session()
    nodes, ns = sess.run([nodes, num_segments])
    print(nodes[-1,:,-5:])
    print(ns)

    # D.insert_from(D.copy(suffix='_copy'), 0)

    # tensor = tf.ones([8,3,16,D.ndims], dtype=tf.float32)
    # zeros = tf.zeros([8,3,16,5], dtype=tf.float32)

    # tensor = D.append_attr_to_vector('zeros', zeros)
    # print("tensor", tensor.shape.as_list())
    # print(D.sort(), D.get_unassigned_dims(), D.get_latent_vector(tensor))
