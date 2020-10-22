from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf

# from graph.common import propdict
from .base import propdict

import vvn.ops.tf_nndistance as tf_nndistance
from vvn.ops.dimensions import DimensionDict
import vvn.ops.rendering as rendering
import vvn.ops.graphical as graphical
import vvn.ops.utils as utils
from vvn.data.utils import object_id_hash

from vvn.models.preprocessing import preproc_rgb, preproc_depths, preproc_normals

# for debugging training
PRINT = False

def identity_loss(logits, labels=None, **kwargs):
    return logits

def l2_loss(logits, labels, **kwargs):
    return tf.square(logits - labels)

def sigmoid_cross_entropy_with_logits(logits, labels, **kwargs):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)

def sparse_ce(logits, labels, **kwargs):
    labels = tf.Print(labels, [tf.reduce_min(labels), tf.reduce_max(labels), tf.reduce_mean(tf.cast(labels, tf.float32))], message='labels_minmaxmean_ce')
    return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)

def is_moving_metric(vx, vy, k=2.0, p=1., binarize_motion=True, dmax=100.):

    sx = tf.reduce_sum(tf.square(vx), axis=-1, keepdims=True)
    sy = tf.reduce_sum(tf.square(vy), axis=-1, keepdims=True)

    if binarize_motion:
        invmap = lambda t: tf.cast(inversion_map(k,p)(t) > 0.5, tf.float32)
    else:
        invmap = inversion_map(k,p)

    sx = invmap(sx) # in (0.,1.)
    sy = invmap(sy)

    is_moving_diff = (1.0 - tf.maximum(sx*sy, (1.0 - sx)*(1.0 - sy))) * dmax
    return is_moving_diff

def l1_diff(x,y):
    return tf.abs(x-y)

def gated_velocity_metric(vx, vy, is_moving_dims=None, metric=l1_diff, vk=2.0, vp=1.0, binarize_motion=True, dmax=100.):

    if is_moving_dims is None:
        d0,d1 = [0,vx.shape.as_list()[-1]]
    else:
        d0,d1 = is_moving_dims

    is_moving_diff = is_moving_metric(vx[...,d0:d1], vy[...,d0:d1], k=vk, p=vp, binarize_motion=True, dmax=dmax) # [0., dmax]
    velocity_diff = metric(vx, vy)
    gated_velocity_diff = mask_tensor(velocity_diff, tf.cast(is_moving_diff < (0.5*dmax), tf.float32), mask_value=dmax)
    return gated_velocity_diff

def normals_metric(nx, ny, normalize=True):
    nx = tf.nn.l2_normalize(nx, axis=-1)
    ny = tf.nn.l2_normalize(ny, axis=-1)
    diff = -tf.reduce_sum(nx*ny, axis=-1, keepdims=True) + 1.0
    return diff

def hue_metric(hx, hy):
    diff = hx - hy
    diff = tf.minimum(tf.floormod(diff, 1.0), tf.floormod(-diff, 1.0))
    return diff

def rgb_metric(rx, ry):
    return tf.reduce_mean(tf.abs(rx - ry), axis=-1, keepdims=True)

def depth_metric(dx, dy):
    return tf.subtract(dx,dy)

def flow_metric(hsv_x, hsv_y):
    '''
    Flows are coded as
    hue in [0,1] codes angle,
    sat in {0,1} is binary, codes absence or presence of flow,
    val in [0,1] codes speed in some unknown scale

    returns a diff in range [0,2.5] (since hue_metric has max 0.5) but really [0,1.5] for two moving objects
    '''
    diff_h = hue_metric(hsv_x[...,0:1], hsv_y[...,0:1])
    diff_sv = tf.reduce_sum(tf.square(hsv_x[...,1:3] - hsv_y[...,1:3]), axis=-1, keepdims=True)
    return diff_h + diff_sv

def velocity_2d_to_flow_hsv(vel, max_speed=5.0, sat_k=None, sat_p=None, eps=1e-8):

    # get speed and angle
    speed = tf.sqrt(tf.reduce_sum(tf.square(vel), axis=-1, keepdims=True) + eps)
    angle = tf.atan2(vel[...,1:2], vel[...,0:1] + eps)

    # convert to the hsv format of optic flow images
    hue = tf.floormod(tf.divide(angle, 2.0*np.pi), 1.0)
    val = tf.minimum(speed, max_speed)
    if sat_k is None or sat_p is None:
        sat = tf.ones_like(val)
    else:
        sat_map = inversion_map(sat_k, sat_p)
        sat = 1.0 - sat_map(speed)

    flows_hsv = tf.concat([hue, sat, val], axis=-1)
    return flows_hsv

def velocity_metric(vec_1, vec_2, vdims=[16,19], zdims=[2,3], eps=1e-8, sat_k=10.0, sat_p=4, max_speed=5.0):
    '''
    '''
    vel_1 = vec_1[...,vdims[0]:vdims[1]]
    vel_2 = vec_2[...,vdims[0]:vdims[1]]
    depth_1 = -vec_1[...,zdims[0]:zdims[1]]
    depth_2 = -vec_2[...,zdims[0]:zdims[1]]

    vel_1 = vel_1[...,0:2] / (depth_1 + eps)
    vel_2 = vel_2[...,0:2] / (depth_2 + eps)

    hsv_1 = velocity_2d_to_flow_hsv(vel_1, sat_k=sat_k, sat_p=sat_p, max_speed=max_speed, eps=eps)
    hsv_2 = velocity_2d_to_flow_hsv(vel_2, sat_k=sat_k, sat_p=sat_p, max_speed=max_speed, eps=eps)

    return flow_metric(hsv_1, hsv_2)

def hungarian_loss(nodes1, nodes2, nodes_init=None, loss_scale=1.0, match_dist_thresh=None, velocity_thresh=None, velocity_dims=[16,19], thresh_dims_list=[[0,3],[-4,-2]], stop_gradient=True, loss_weights=None, loss_dims_list=[[0,-1]], loss_preprocs_list=None, **kwargs):
    '''
    nodes1: prediction of nodes at some time
    nodes2: "ground truth" of nodes at some time (i.e. output directly from encoder)
    nodes_init: what the true nodes were before being rolled out by a dynamics model to become nodes1 (predicted future nodes)
    '''

    if not isinstance(nodes1, tf.Tensor):
        nodes1 = tf.constant(nodes1, tf.float32)
        nodes2 = tf.constant(nodes2, tf.float32)

    # may want to stop the gradient on the "ground truth"
    if stop_gradient:
        nodes2 = tf.stop_gradient(nodes2)
        if nodes_init is not None:
            nodes_init = tf.stop_gradient(nodes_init)

    print("nodes init", nodes_init)
    B,N,D = nodes1.shape.as_list()

    if nodes_init is not None:
        match = hungarian_node_matching(nodes_init, nodes2, **kwargs) # [B,N] indices into nodes2 that minimize assignment cost between nodes_init and nodes2
    else:
        match = hungarian_node_matching(nodes1, nodes2, **kwargs) # [B,N] indices into nodes2 that minimize assignment cost between nodes_init and nodes2
    nodes2p = permute_nodes(nodes2, match)

    if match_dist_thresh is not None:
        if nodes_init is not None:
            thresh_nodes1 = tf.concat([nodes_init[...,td[0]:td[1]] for td in thresh_dims_list], axis=-1)
        else:
            thresh_nodes1 = tf.concat([nodes1[...,td[0]:td[1]] for td in thresh_dims_list], axis=-1)
        thresh_nodes2p = tf.concat([nodes2p[...,td[0]:td[1]] for td in thresh_dims_list], axis=-1)
        valid_matches = tf.sqrt(tf.reduce_sum(tf.square(thresh_nodes1-thresh_nodes2p), -1, keepdims=True)) < match_dist_thresh
        valid_matches = tf.cast(valid_matches, tf.float32)
        print("thresholding matches by dist", match_dist_thresh)
    else:
        valid_matches = tf.ones_like(nodes1[...,-1:])


    valid_matches = valid_matches * nodes1[...,-1:] * nodes2p[...,-1:] # [B,N,1]
    num_valid_matches = tf.reduce_sum(valid_matches, axis=[1,2], keepdims=False) # [B]
    if PRINT:
        num_valid_matches = tf.Print(num_valid_matches, [tf.reduce_mean(num_valid_matches)], message='num_valid_matches')

    if loss_preprocs_list is None:
        loss_preprocs_list = [tf.identity] * len(loss_dims_list)
    else:
        assert len(loss_preprocs_list) == len(loss_dims_list), "Must pass one preproc func per set of loss dims"
    nodes1_attrs = tf.concat([loss_preprocs_list[i](nodes1[...,d[0]:d[1]]) for i,d in enumerate(loss_dims_list)], axis=-1)
    nodes2p_attrs = tf.concat([loss_preprocs_list[i](nodes2p[...,d[0]:d[1]]) for i,d in enumerate(loss_dims_list)], axis=-1)

    total_dims = sum([(d[1] % D) - (d[0] % D) for d in loss_dims_list])
    loss_weights = tf.ones([total_dims], tf.float32) if loss_weights is None else tf.constant(loss_weights, dtype=tf.float32)
    loss_weights = tf.reshape(loss_weights, [1,1,-1])
    assert loss_weights.shape.as_list()[-1] == nodes1_attrs.shape.as_list()[-1], "must pass one loss weight per attribute being compared"

    speed_nodes = nodes_init if nodes_init is not None else nodes1
    speeds = tf.reduce_sum(tf.square(speed_nodes[...,velocity_dims[0]:velocity_dims[1]]), axis=-1, keepdims=True) # [B,N,1]
    if velocity_thresh is not None:
        valid_speeds = tf.cast(speeds > velocity_thresh, tf.float32)
        if PRINT:
            valid_speeds = tf.Print(valid_speeds, [tf.reduce_mean(tf.reduce_sum(valid_speeds, axis=[1,2]))], message='num_valid_speeds')
        # valid_matches = valid_matches * valid_speeds
    else:
        valid_speeds = tf.ones_like(valid_matches)

    if PRINT:
        valid_matches = tf.Print(valid_matches, [tf.reduce_max(speeds), tf.reduce_mean(tf.cast(speeds > 0.1, tf.float32))], message='max_speeds')

    loss_per_dim = loss_scale * valid_speeds * valid_matches * tf.square(nodes1_attrs - nodes2p_attrs) # [B,N,total_dims]

    if PRINT:
        loss_per_dim = tf.Print(loss_per_dim, [tf.reduce_max(loss_per_dim[...,0:3] / loss_scale, axis=[0,1])], message="hung_loss_per_dim_03")
        loss_per_dim = tf.Print(loss_per_dim, [tf.reduce_max(loss_per_dim[...,3:6] / loss_scale, axis=[0,1])], message="hung_loss_per_dim_36")
        loss_per_dim = tf.Print(loss_per_dim, [tf.reduce_max(loss_per_dim[...,6:7] / loss_scale, axis=[0,1])], message="hung_loss_per_dim_67")
    loss = tf.reduce_sum(loss_per_dim*loss_weights, axis=-1) # [B,N]
    loss += (1.0 - valid_speeds[...,0]) * valid_matches[...,0] * tf.reduce_sum(tf.square(nodes1[...,velocity_dims[0]:velocity_dims[1]]), axis=-1, keepdims=False) # penalize nonzero v
    # append loss to nodes2p
    nodes2p = tf.concat([nodes2p, loss_per_dim / loss_scale], axis=-1)
    loss = tf.reduce_sum(loss, axis=1) /  tf.maximum(num_valid_matches, 1.0)

    return tf.reduce_mean(loss), nodes2p, match

def dice_loss(logits, labels, **kwargs):

    assert logits.shape == labels.shape
    B,N,K = logits.shape.as_list()

    numer = 2.*tf.reduce_sum(logits * labels, axis=1) + 1.
    denom = tf.reduce_sum(logits, axis=1) + tf.reduce_sum(labels, axis=1) + 1.
    loss = 1. - (numer / denom) # [B,K]
    loss = tf.reduce_sum(loss, axis=-1)
    return loss # [B]

def hungarian_dice_loss(logits, labels, **kwargs):

    B,N,K = logits.shape.as_list()
    _,_,O = labels.shape.as_list()
    assert K >= O, "predict more masks than the max you might get as labels"

    cost_mat = utils.dice_cost(logits, labels) # [B,K,O]
    cost_mat = tf.transpose(cost_mat, [0,2,1]) # [B,O,K]
    assignment = tf.cast(utils.hung.hungarian(cost_mat), tf.int32) # [B,O] inds into K dim
    b_inds = tf.range(B, dtype=tf.int32)[:,tf.newaxis] * tf.ones_like(assignment)
    inds = tf.stack([b_inds, assignment], axis=-1) # [B,O,2]
    logits = tf.transpose(logits, [0,2,1]) # [B,K,N]
    logits = tf.gather_nd(logits, inds) # [B,O,N]
    logits = tf.transpose(logits, [0,2,1])
    loss = dice_loss(logits, labels)
    return loss

def chamfer_loss(logits, labels, num_pred_particles=None, num_gt_particles=None, mask_logits=None, mask_labels=None, loss_multiplier=5000.0, two_way=True, forward_loss=True, dist_thresh=None, mask_match=False):
    '''
    logits: particles of shape [B, N_pred, 3]
    labels: gt particles of shape [B, N_gt, 3]
    num_pred_particles: [B] or int
    num_gt_particles: [B] or int
    mask_logits: [B, N_pred] or None to zero out losses from fake particles
    mask_labels: [B, N_gt] or None to zero out losses from fake particles
    '''

    batch_size, n_pred = logits.shape.as_list()[0:2]
    _, n_gt = labels.shape.as_list()[0:2]
    pred_part_positions = logits
    gt_positions = labels
    if num_pred_particles is None and mask_logits is not None:
        num_pred_particles = tf.reduce_sum(mask_logits, axis=1)
    if num_gt_particles is None and mask_labels is not None:
        num_gt_particles = tf.reduce_sum(mask_labels, axis=1)

    dists_forward, inds_forward, dists_backward, inds_backward = tf_nndistance.nn_distance(pred_part_positions, gt_positions) # [B,N_pred] and [B, N_gt]

    if mask_logits is not None:
        dists_forward *= tf.cast(mask_logits, dtype=tf.float32)
    if mask_labels is not None:
        dists_backward *= tf.cast(mask_labels, dtype=tf.float32)
    if mask_match:
        assert mask_logits is not None and mask_labels is not None
        b_inds = tf.range(batch_size, dtype=tf.int32)[:,tf.newaxis]
        valid_matches_fwd = tf.gather_nd(mask_labels, tf.stack([tf.tile(b_inds, [1,n_pred]), inds_forward], axis=-1))
        dists_forward *= valid_matches_fwd

        valid_matches_back = tf.gather_nd(mask_logits, tf.stack([tf.tile(b_inds, [1,n_gt]), inds_backward], axis=-1))
        dists_backward *= valid_matches_back

    if dist_thresh is not None:
        dists_forward = dists_forward * tf.cast(dists_forward - dist_thresh > 0.0, tf.float32)
        dists_backward = dists_backward * tf.cast(dists_backward - dist_thresh > 0.0, tf.float32)

    # mean by number of true particles in the 1st dimension
    # print("pred and gt particle shapes", num_pred_particles.shape, num_gt_particles.shape)
    loss_f = tf.div(tf.reduce_sum(dists_forward, axis=1), tf.maximum(1.0, num_pred_particles))
    loss_b = tf.div(tf.reduce_sum(dists_backward, axis=1), tf.maximum(1.0, num_gt_particles))

    # now mean over batch dimension
    if two_way:
        loss = tf.reduce_mean(loss_f + loss_b) * loss_multiplier
        return loss
    else: # just use forward or backward distance, i.e. nn distance from each of num_pred_particles/num_gt_particles
        loss = tf.reduce_mean(loss_f) * loss_multiplier if forward_loss else tf.reduce_mean(loss_b) * loss_multiplier
        return loss

def camera_correspondence_loss(nodes, pmats, cmats,
                               position_dims=[0,3],
                               velocity_dims=None,
                               vectors_dims_list=[[6,9],[13,16],[16,19]],
                               loss_dims_list=[[0,12]],
                               stop_gradient=False,
                               right_handed=True, positive_z=True, mask_match=False,
                               moving_particles_in_view=False):
    '''
    nodes: [B,T,N,D]
    pmats: [B,T,4,4]
    cmats: [B,T,4,4]
    position_dims_list: list of vectors to transform with the full 4x4 camera matrix
    vectors_dim1s_list: list of vectors to transform with the 3x3 camera rotation matrix
    right_handed & positive_z: if both True, positive y is DOWN and positive Z is FORWARD from camera

    stop_gradient: stops gradient on xyz (position) vectors
    '''
    if right_handed and positive_z:
        rotate_180 = True
    else:
        rotate_180 = False

    if position_dims in loss_dims_list:
        loss_dims_list = [ld for ld in loss_dims_list if ld != position_dims]

    B,T,N,D = nodes.shape.as_list()
    valid_nodes = nodes[...,-1:]
    def _build_nodes(prev_nodes, new_vec, dims):
        B,N,D = prev_nodes.shape.as_list()
        dims = [d % D for d in dims]
        assert dims[1] - dims[0] == new_vec.shape.as_list()[-1]
        next_nodes = tf.concat([
            prev_nodes[...,:dims[0]],
            new_vec,
            prev_nodes[...,dims[1]:]], axis=-1)
        return next_nodes

    def _nodes_in_frame(pos_vec, pmat):
        h_new, w_new = tf.split(camera_projection(pos_vec[:,tf.newaxis], pmat)[:,0,:,0:2], [1,1], axis=-1) # [B,N,1] * 2
        h_inview = tf.logical_and(h_new >= -1., h_new <= 1.)
        w_inview = tf.logical_and(w_new >= -1., w_new <= 1.)
        hw_inview = tf.logical_and(h_inview, w_inview)
        inview_mask = tf.cast(hw_inview, tf.float32)
        return inview_mask

    loss = 0.0
    for t in range(T-1):
        nodes_t = nodes[:,t] # [B,N,D]
        new_nodes_t = nodes_t + 0.0
        cam1, cam2 = tf.split(cmats[:,t:t+2], [1,1], axis=1) # [B,1,4,4]
        pmat1, pmat2 = tf.split(pmats[:,t:t+2], [1,1], axis=1) # [B,1,4,4]

        # transform position vector and find nodes in both frames
        pos_vec = nodes_t[...,position_dims[0]:position_dims[1]]
        pos_vec = tf.stop_gradient(pos_vec)
        if velocity_dims is not None:
            vel_vec = nodes_t[...,velocity_dims[0]:velocity_dims[1]]
            pos_vec += vel_vec
        pos_vec_new = transform_vector_between_cameras(pos_vec, cam1[:,0], cam2[:,0], rotate_180)
        new_nodes_t = _build_nodes(new_nodes_t, pos_vec_new, position_dims)

        # project the new nodes to find which ones are still in frame
        if moving_particles_in_view:
            inview_mask_t0 = _nodes_in_frame(
                transform_vector_between_cameras(nodes_t[...,position_dims[0]:position_dims[1]], cam1[:,0], cam2[:,0], rotate_180), pmat2)
        else:
            inview_mask_t0 = _nodes_in_frame(pos_vec_new, pmat2) # [B,N,1]

        # also need to mask out nodes in frame t+1 that would be out of view in frame t
        inview_mask_t1 = _nodes_in_frame(
            transform_vector_between_cameras(nodes[:,t+1,:,position_dims[0]:position_dims[1]], cam2[:,0], cam1[:,0], rotate_180),
            pmat1)

        # transform the rest of the vecotrs in nodes_t
        for dims in vectors_dims_list:
            vec = nodes_t[...,dims[0]:dims[1]]
            vec_new = rotate_vector_between_cameras(vec, cam1[:,0], cam2[:,0], rotate_180)
            new_nodes_t = _build_nodes(new_nodes_t, vec_new, dims)

        # build full mask and then compute chamfer loss
        mask_t0 = valid_nodes[:,t] * inview_mask_t0
        mask_t1 = valid_nodes[:,t+1] * inview_mask_t1
        # mask_t1 = tf.Print(mask_t1, [tf.reduce_mean(tf.reduce_sum(mask_t0, 1)), tf.reduce_mean(tf.reduce_sum(valid_nodes[:,t], 1)),
        #                              tf.reduce_mean(tf.reduce_sum(mask_t1, 1)), tf.reduce_mean(tf.reduce_sum(valid_nodes[:,t+1], 1)),
        #                              tf.shape(valid_nodes[0,t])
        # ],
        #                    message="inview_mask_t"+str(t))
        nodes_t0 = new_nodes_t[...,position_dims[0]:position_dims[1]]
        sg_func = tf.stop_gradient if stop_gradient else tf.identity
        nodes_t0 = mask_tensor(tf.concat([nodes_t0] + [sg_func(new_nodes_t[...,d[0]:d[1]]) for d in loss_dims_list], axis=-1), mask_t0, mask_value=1000.)
        nodes_t1 = tf.stop_gradient(nodes[:,t+1]) if stop_gradient else nodes[:,t+1]
        nodes_t1 = mask_tensor(tf.concat([nodes_t1[...,d[0]:d[1]] for d in [position_dims] + loss_dims_list], axis=-1), mask_t1, mask_value=1000.)
        loss += chamfer_loss(
            logits=nodes_t0, labels=nodes_t1,
            mask_logits=mask_t0[...,0], mask_labels=mask_t1[...,0],
            loss_multiplier=1.0, two_way=True, mask_match=mask_match)

    return loss

def cross_entropy(logits, labels, eps=1e-3, keepdims=True):
    '''
    x = logits in [0,1.0]
    z = labels in [0,1.0]
    loss = z * -log(x) + (1-z) * -log(1-x)

    clip x to avoid exceeding max float
    '''
    x = tf.clip_by_value(logits, eps, 1.0-eps)
    loss = -(labels * tf.math.log(x) + (1.0-labels) * tf.math.log(1.0 - x))
    loss = tf.where(tf.logical_or(tf.math.is_inf(loss), tf.math.is_nan(loss)),
                    tf.zeros_like(loss), # if true
                    loss) # if false
    if keepdims:
        return loss
    else:
        loss = tf.reduce_mean(loss)
        return loss

def build_pairwise_segment_labels(objects, inds1, inds2):
    '''
    For a ground truth segmentation image "objects" and pairs of indices inds1, inds2 (optionally broadcasted),
    determine whether each pair ((h1,w1),(h2,w2)) are in the same segment or not. Return indicator labels.

    objects: [B,T,H,W,C] <tf.uint8> an RGB segmentation map of the objects
    inds1: [B,T,P,K1,2] <tf.int32> (h,w) indices for P*K1 points into the image
    inds2: [B,T,P,K2,2] <tf.int32> (h,w) indices for P*K2 points into the image. Must have K1 == 1 or K1 == K2.
    '''
    B,T,H,W,C = objects.shape.as_list()
    _,_,P,K1,_ = inds1.shape.as_list()
    _,_,P2,K,_ = inds2.shape.as_list()
    assert P == P2, "Must use same number of points"
    assert K1 == 1 or K1 == K, "inds1 must have the same shape as inds2 or be broadcastable to it."
    if objects.dtype == tf.uint8:
        objects = object_id_hash(objects, tf.int32, val=256)
    else:
        assert objects.dtype == tf.int32
    assert objects.shape.as_list()[-1] == 1, objects

    inds1 = tf.reshape(inds1, [B,T,P*K1,2])
    inds2 = tf.reshape(inds2, [B,T,P*K,2])
    inds = tf.concat([inds1, inds2], axis=2)
    segvals = rendering.get_image_values_from_indices(objects, inds) # [B,T,P*K1 + P*K,1]
    segvals1, segvals2 = tf.split(segvals, [P*K1, P*K], axis=2)
    segvals1 = tf.reshape(segvals1, [B,T,P,K1]) # [B,T,P,K1,1]
    segvals2 = tf.reshape(segvals2, [B,T,P,K])
    labels = tf.cast(tf.equal(segvals1, segvals2), tf.float32) # [B,T,P,K]

    return labels

def affinity_cross_entropy_from_nodes_and_segments(
        affinities, nodes, segments, dimension_dict=None,
        affinities_key='affinities', hw_attr='hw_centroids', hw_dims=[-4,-2], start_time=0,
        valid_attr='valid', size=None, downsample_labels=True, **kwargs):
    '''
    for ground truth spatially registered segments and a set of nodes, determine whether each node pair belongs
    in the same segment or not.

    nodes: [B,T,N,D] <tf.float32> with some of the components indication hw position in the image
    segments: [B,T,H,W,C] <tf.uint8> an RGB segmentation map of the objects
    dimension_dict: a legend for which attributes contain which information
    hw_attr: which attribute indicates the position of the nodes
    hw_dims: if hw_attr isn't a key into dimension_dict, index into these node dims
    '''
    # check inputs
    if isinstance(nodes, propdict):
        nodes = nodes['vector']
    if len(nodes.shape) == 3:
        nodes = nodes[:,tf.newaxis]
    assert len(nodes.shape) == 4, "Nodes must have shape [B,T,N,D] but nodes are %s" % nodes
    B,T,N,_ = nodes.shape.as_list()

    # assume square image
    if size is None:
        size = [int(np.sqrt(N))]*2

    # check segments
    if segments.dtype == tf.uint8 and len(segments.shape) == 4:
        segments = segments[:,tf.newaxis]
    elif segments.dtype == tf.int32 and len(segments.shape) == 3:
        segments = segments[:,tf.newaxis,...,tf.newaxis]
    assert len(segments.shape) == 5 and segments.dtype in [tf.uint8, tf.int32], segments
    im_size = ind_size = segments.shape.as_list()[2:4]

    if downsample_labels:
        assert (not im_size[0] % size[0]) and (not im_size[1] % size[1]), (im_size, size)
        strides = [im_size[0] // size[0], im_size[1] // size[1]]
        segments = segments[:,:,::strides[0],::strides[1],:]
        ind_size = segments.shape.as_list()[2:4]

    # if affinities are coming from a Graph.edges value
    if isinstance(affinities, propdict):
        affinities = affinities[affinities_key]

    # if it's a local affinity matrix, convert it to a global one
    valid_edges = None
    if affinities.shape.as_list() not in [[B*T,N,N], [B,T,N,N]]:
        affinities = tf.reshape(affinities, [B*T,N,-1])
        affinities = graphical.local_to_global_adj(
            local_adj=tf.ones_like(affinities, tf.bool),
            size=size, affinities=affinities)
        valid_edges = graphical.local_to_global_adj(
            local_adj=tf.ones_like(affinities, tf.bool),
            size=size, affinities=None)
        affinities = tf.reshape(affinities, [B,T,N,N])
        valid_edges = tf.reshape(valid_edges, [B,T,N,N])
        valid_edges = valid_edges[:,start_time:]

    if len(affinities.shape) == 3:
        affinities = affinities[:,tf.newaxis]
    assert affinities.shape.as_list() == nodes.shape.as_list()[0:2] + [N,N],\
                                        "Affinities must have shape [B,T,N,N] but are %s" % affinities


    if dimension_dict is None:
        dimension_dict = DimensionDict(nodes.shape.as_list()[-1], {hw_attr:hw_dims, 'valid':[-1,0]})

    # get hw_attrs
    nodes = nodes[:,start_time:]
    segments = segments[:,start_time:]
    try:
        hw_key = [k for k in dimension_dict.sort().keys() if hw_attr in k][-1]
        nodes_hw = dimension_dict.get_tensor_from_attrs(nodes, hw_key)
    except IndexError:
        nodes_hw = utils.coordinate_ims(B,T,size)
        nodes_hw = tf.reshape(nodes_hw, [B,T,N,2])
        nodes_hw = nodes_hw[:,start_time:]
    except KeyError:
        nodes_hw = nodes[...,hw_dims[0]:hw_dims[1]]

    base_inds = tf.expand_dims(rendering.hw_attrs_to_image_inds(nodes_hw, ind_size), axis=-2) # [B,T',N,1,2]
    pair_inds = tf.tile(tf.transpose(base_inds, [0,1,3,2,4]), [1,1,N,1,1]) # [B,T',N,N,2]
    edge_labels = build_pairwise_segment_labels(segments, base_inds, pair_inds) # [B,T',N,N]
    edge_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=edge_labels, logits=affinities[:,start_time:])

    # compute valid losses
    if valid_edges is None: # already set if it were a local mat
        valid_key = [k for k in dimension_dict.sort().keys() if valid_attr in k][-1]
        valid_nodes = dimension_dict.get_tensor_from_attrs(nodes, valid_key, postproc=True) # [B,T',N,1]
        valid_edges = valid_nodes * tf.transpose(valid_nodes, [0,1,3,2])
    edge_loss = edge_loss * valid_edges
    num_valid_edges = tf.reduce_sum(valid_edges, axis=[2,3])
    edge_loss = tf.reduce_sum(edge_loss, axis=[2,3]) / tf.maximum(1., num_valid_edges) # [B,T']

    return edge_loss

def node_attr_mse(logits, labels, dimension_dict=None, loss_attrs=['positions'], valid_attr='valid',
                  postproc=True, mean_across_bt=False, **kwargs):

    '''
    Get the [postproc] attrs from logits and labels using the dimension_dict and compute MSE.
    '''
    B,T,N,D = logits.shape.as_list()
    if dimension_dict is None:
        dimension_dict = DimensionDict(D, {'vector':[0,0], 'valid':[-1,0]})
        loss_attrs = ['vector']
        valid_attr = 'valid'
    else:
        assert isinstance(dimension_dict, DimensionDict), "dimension_dict must be a DimensionDict but was %s" % type(dimension_dict)

    # get attrs
    logits_attrs = dimension_dict.get_tensor_from_attrs(
        logits, loss_attrs, postproc=postproc, stop_gradient=False)
    labels_attrs = dimension_dict.get_tensor_from_attrs(
        labels, loss_attrs, postproc=postproc, stop_gradient=True)

    # find out which nodes are valid
    logits_valid = dimension_dict.get_tensor_from_attrs(
        logits, valid_attr, postproc=postproc, stop_gradient=True)
    labels_valid = dimension_dict.get_tensor_from_attrs(
        labels, valid_attr, postproc=postproc, stop_gradient=True)
    valid_nodes = logits_valid * labels_valid
    assert valid_nodes.shape.as_list() == [B,T,N,1], "Valid must be a single dimensional vector"
    num_valid_nodes = tf.maximum(
        tf.reduce_sum(valid_nodes, axis=[2,3]), 1.0) # [B,T]
    # num_valid_nodes = tf.Print(num_valid_nodes, [tf.reduce_mean(num_valid_nodes)], message='num_inview')

    if PRINT:
        print("node attr loss logits")
        print(logits_attrs.shape.as_list())
        print("node attr loss labels")
        print(labels_attrs.shape.as_list())
        logits_attrs = tf.Print(logits_attrs, [logits_attrs[0,0,1,:]], message='xyz')

    loss = tf.square(logits_attrs - labels_attrs) * valid_nodes
    if PRINT:
        loss = tf.Print(loss, [tf.reduce_max(loss, axis=[0,1,2])], message='loss_dims')
    loss = tf.reduce_sum(loss, axis=[2,3]) / num_valid_nodes # [B,T]
    if mean_across_bt:
        loss = tf.reduce_mean(loss) # mean across batch and time

    return loss

def depth_per_point_loss(
        pred_depths, gt_depths,
        log_space=False, z_offset=0., depth_hinge=0.0, depth_max=30.,
        **kwargs
):

    if log_space:
        pred_depths = tf.log(1.0 + tf.nn.relu(pred_depths))
        gt_depths = tf.log(1.0 + gt_depths)

    depth_loss = tf.square(pred_depths - gt_depths - z_offset)

    if depth_hinge is not None:
        hinge_loss = tf.square(tf.nn.relu(pred_depths - depth_hinge))

    return depth_loss + hinge_loss

def normals_per_point_loss(
        pred_normals, gt_normals,
        normalize_gt=False, loss_offset=1.0, dot_loss_scale=1.0, l2_loss_scale=1.0,
        **kwargs
):

    if normalize_gt:
        gt_normals = tf.nn.l2_normalize(gt_normals, axis=-1, epsilon=1e-3)

    normals_loss = tf.zeros_like(pred_normals[...,0:1])
    normals_loss += loss_offset
    normals_loss += -1.0 * dot_loss_scale * tf.reduce_sum(
        pred_normals * gt_normals, axis=-1, keepdims=True)
    normals_loss += l2_loss_scale * tf.reduce_sum(
        tf.square(pred_normals - gt_normals), axis=-1, keepdims=True)

    return normals_loss

def spatial_attr_loss(
        pred_attr,
        image,
        spatial_inds,
        valid_attr=None,
        valid_image=None,
        image_preproc=lambda im: tf.cast(im, tf.float32),
        loss_per_point_func=l2_loss,
        loss_name='loss',
        **kwargs
):
    B,T,P,D = pred_attr.shape.as_list()
    image = image_preproc(image)
    _,_,H,W,Dim = image.shape.as_list()
    assert D == Dim, (D,Dim)
    assert spatial_inds.shape.as_list() == [B,T,P,2]
    if valid_image is not None:
        image *= tf.cast(valid_image, tf.float32)

    image_vals = rendering.get_image_values_from_indices(
        image, spatial_inds) # [B,T,P,Dim]
    image_vals = tf.stop_gradient(image_vals)
    if PRINT:
        image_vals = tf.Print(image_vals, [tf.reduce_max(tf.abs(pred_attr)), tf.reduce_max(tf.abs(image_vals))], message='max_vals_'+loss_name)
    loss_per_point = loss_per_point_func(pred_attr, image_vals, **kwargs)

    return loss_per_point

def rendered_attrs_images_loss(
        pred_attrs,
        spatial_inds,
        labels,
        size,
        valid_attrs=None,
        valid_images=None,
        attr_to_image_dict=None,
        image_preprocs={'images': lambda im: tf.image.rgb_to_hsv(preproc_rgb(im))},
        loss_per_point_funcs={},
        loss_per_point_kwargs={},
        loss_scales={},
        **kwargs
):
    '''
    preprocs the gt images, samples them at the needed points, then computes losses
    '''
    attrs = pred_attrs.keys()
    if attr_to_image_dict is None:
        attr_to_image_dict = {attr: ''.join(attr.split('_')[1:]) for attr in attrs}
    images = {k:labels[k] for k in attr_to_image_dict.values()}
    if valid_images is None:
        valid_images = tf.ones(images[images.keys()[0]].shape, dtype=tf.bool)[...,0:1] # [B,T,H,W,1]
    assert all((len(im.shape) == 5 for im in images.values())), images
    assert len(valid_images.shape) == 5, valid_images
    imsize = images[images.keys()[0]].shape.as_list()[2:4]
    if size != imsize:
        strides = [imsize[0] // size[0], imsize[1] // size[1]]
        images = {k:im[:,:,::strides[0],::strides[1],:] for k,im in images.items()}
        valid_images = valid_images[:,:,::strides[0],::strides[1],:]

    if valid_attrs is None:
        valid_attrs = tf.ones_like(pred_attrs[attrs[0]], dtype=tf.float32)[...,0:1] # [B,T,P,1]

    valid_points = rendering.get_image_values_from_indices(
        valid_images, spatial_inds) # [B,T,P,1]

    valid_points = tf.cast(valid_points, tf.float32) * valid_attrs
    num_valid_points = tf.reduce_sum(valid_points, axis=[-2,-1]) # [B,T]
    num_valid_points = tf.maximum(1., num_valid_points)
    B,T = valid_points.shape.as_list()[0:2]

    loss = tf.zeros([B,T], dtype=tf.float32)
    for attr, imkey in attr_to_image_dict.items():
        loss_attr = spatial_attr_loss(
            pred_attr=pred_attrs[attr],
            image=images[imkey],
            spatial_inds=spatial_inds,
            valid_image=valid_images,
            image_preproc=image_preprocs.get(imkey, lambda im: tf.cast(im, tf.float32)),
            loss_per_point_func=loss_per_point_funcs.get(imkey, l2_loss),
            loss_name=imkey,
            **loss_per_point_kwargs.get(imkey, {})
        )
        assert len(loss_attr.shape) == 4, loss_attr
        loss_attr *= valid_points
        loss_attr = tf.reduce_sum(loss_attr, axis=[-2,-1]) / num_valid_points # [B,T]
        if PRINT:
            loss_attr = tf.Print(loss_attr, [tf.reduce_sum(loss_attr)], message='loss_'+imkey)
        loss += loss_attr * loss_scales.get(imkey, 1.0)

    return loss

def masked_l2_image_loss(
        pred_image,
        gt_image,
        valid_image,
        valid_pred=True,
        gt_preproc=lambda im: tf.image.rgb_to_hsv(preproc_rgb(im)),
        pred_times=[1,None],
        gt_times=[2,None],
        loss_per_point_func=l2_loss,
        loss_per_point_func_kwargs={},
        **kwargs):

    B,Tp,H,W,C = pred_image.shape.as_list()
    _,Tg,Him,Wim,Cim = gt_image.shape.as_list()
    if pred_times[1] is None:
        pred_times[1] = Tp
    if gt_times[1] is None:
        gt_times[1] = Tg
    ## process time
    pred_image = pred_image[:,pred_times[0]:pred_times[1]]
    gt_image = gt_image[:,gt_times[0]:gt_times[1]]

    if valid_pred:
        valid_image = valid_image[:,pred_times[0]:pred_times[1]]
    else:
        valid_image = valid_iamge[:,gt_times[0]:gt_times[1]]

    ## process space and dtype
    strides = [Him // H, Wim // W]
    gt_image = gt_image[:,:,::strides[0],::strides[1],:]
    gt_image = gt_preproc(gt_image)

    ## check
    assert gt_image.shape == pred_image.shape, (gt_image, pred_image)
    assert valid_image.shape[:-1] == pred_image.shape[:-1]

    ## compute loss
    num_valid_px = tf.maximum(tf.reduce_sum(valid_image, axis=[2,3,4]), 1.0)
    num_valid_px = tf.Print(num_valid_px, [tf.reduce_mean(num_valid_px)], message='num_valid_future')
    loss = loss_per_point_func(logits=pred_image, labels=gt_image, **loss_per_point_func_kwargs)
    loss = tf.reduce_sum(loss * valid_image, axis=[2,3,4]) / num_valid_px

    return loss

def projected_particle_color_loss(particles,
                                  images,
                                  particles_im_indices,
                                  not_occluded_mask,
                                  particles_mask=None,
                                  image_foreground_masks=None,
                                  color_dims=[0,1],
                                  hsv_space=True,
                                  cos_dist=True,
                                  add_sv_dist=False,
):
    '''
    Projects particles_xyz onto an image and gets the colors of the corresponding pixels.
    Then computes distance between particles_colors and pixels_colors

    particles: [B,T,N,D] where the last axis contains both xyz_dims and color_dims (slices)
    image: [B,T,H,W,3] the rgb image to project onto (optionally converting to hsv)
    particles_im_indices: [B,T,N,2] of int32 indices into H and W dimensions of an image of size H,W
    not_occluded_mask: [B,T,N,1] float32 where 1.0 indicates the particle wasn't occluded at radius p_radius, 0.0 otherwise

    particles_mask: [B,T,N,1] indicates which particles are valid at each time point
    image_foreground_masks: [B,T,H,W,1] binary mask of which pixels are foreground. Loss will be computed only from these
    color_dims: [Red, Blue+1] // [Hue, Value+1] or [Hue, Hue+1] if not using SV for distance
    hsv_space: if true, the particle_colors are considered to be hsv values and the loss will be computed in hsv space
    add_sv_dist: if true, will compute distane on Saturation and Value as well as Hue. These values tend to vary across lighting conditions

    returns
    color_loss: scalar of mean across unmasked particles of color differences between particles and ground truth

    '''
    B,T,N,D = particles.shape.as_list()
    _,_,H,W,Dim = images.shape.as_list()
    if Dim != 3:
        raise NotImplementedError("only project onto rgb or hsv images")

    image_colors = get_image_values_from_indices(images, particles_im_indices) # [B,T,N,Dim]
    if image_foreground_masks is not None:
        foreground_masks = get_image_values_from_indices(image_foreground_masks, particles_im_indices) # [B,T,N,1]
    else:
        foreground_masks = tf.ones([B,T,N,1], dtype=tf.float32)

    # get particles_colors and the appropriate mask
    particles_colors = particles[:,:,:,color_dims[0]:color_dims[1]]
    if particles_mask is None:
        particles_mask = tf.ones([B,T,N,1], dtype=tf.float32)
    total_mask = foreground_masks * not_occluded_mask * particles_mask
    num_particles = tf.reduce_sum(total_mask, axis=2, keepdims=True) # [B,T,1,1]

    # num_particles = tf.Print(num_particles, [tf.reduce_mean(num_particles)], message="hue particles")

    # compute color loss and apply mask so occluded/background/fake particles don't contribute
    if hsv_space: # need to treat the circular h coordinate differently from s and v
        _image_colors = tf.image.rgb_to_hsv(tf.cast(image_colors, dtype=tf.float32) / 255.)
        particles_hues = particles_colors[:,:,:,0:1]
        image_hues = _image_colors[:,:,:,0:1]

        if cos_dist:
            hues_dist = tf.square(tf.cos(2.0*np.pi*image_hues) - tf.cos(2.0*np.pi*particles_hues))
            hues_dist += tf.square(tf.sin(2.0*np.pi*image_hues) - tf.sin(2.0*np.pi*particles_hues))
        else:
            hues_dist = tf.square(image_hues - particles_hues)
        # hues_dist = 1.0 - tf.cos(2.0*np.pi*(image_hues - particles_hues)) if cos_dist\
        #             else tf.square(image_hues - particles_hues)

        color_loss = hues_dist
        if add_sv_dist:
            sv_dist = tf.reduce_sum(tf.square(particles_colors[:,:,:,1:] - _image_colors[:,:,:,1:]), axis=3, keepdims=True)
            color_loss += sv_dist
    else: # rgb space or another where euclidian distance applies
        _image_colors = tf.cast(image_colors, dtype=tf.float32) / 255.
        color_loss = tf.reduce_sum(tf.square(particles_colors - _image_colors), axis=3, keepdims=True) # [B,T,N,1]

    color_loss *= total_mask
    # color_loss = tf.Print(color_loss, [tf.reduce_max(color_loss[0,-1,:,0])], message="color loss")
    color_loss = tf.div(tf.reduce_sum(color_loss, axis=2, keepdims=True), tf.maximum(1.0, num_particles))
    color_loss = tf.reduce_mean(color_loss)

    # color_loss = tf.Print(color_loss, [tf.reduce_mean(num_particles), tf.reduce_mean(hues_dist), tf.reduce_mean(sv_dist), color_loss], message="color_loss")

    return color_loss

def pinhole_projection_loss(
        nodes, dimension_dict, projection_matrix,
        positions_attr='position', hw_attr='hw_centroids', valid_attr='valid', p_radius=0.0, xyz_dims=[0,3], stop_gradient=True,
        **kwargs
):

    nodes_xyz, pos_key = dimension_dict.get_attr(nodes, positions_attr, position=0, sort=True)
    # don't allow loss to minimize by changing z
    if stop_gradient:
        nodes_xyz = tf.concat([
            nodes_xyz[...,0:2], tf.stop_gradient(nodes_xyz[...,2:3])], axis=-1)
    nodes_hw, hw_key = dimension_dict.get_attr(nodes, hw_attr, position=-1, sort=True)
    nodes_valid, val_key = dimension_dict.get_attr(nodes, valid_attr, position=-1, sort=True)

    # get image positions
    _, not_occluded, nodes_hw_pred = rendering.project_and_occlude_particles(
        nodes_xyz, projection_matrix, im_size=[1,1], particles_mask=nodes_valid,
        p_radius=p_radius, xyz_dims=xyz_dims, **kwargs)

    # transform nodes_hw_pred from range [0,1] to range [-1,1]
    nodes_hw_pred = 2.0*nodes_hw_pred - 1.0

    if PRINT:
        nodes_hw_pred = tf.Print(nodes_hw_pred, [
            tf.reduce_max(nodes_xyz[...,0:2] * nodes_valid, axis=[0,1,2]),
            tf.reduce_min(nodes_xyz[...,2:3] * nodes_valid, axis=[0,1,2]),
            tf.reduce_max(nodes_xyz[...,2:3] * nodes_valid, axis=[0,1,2]),
            tf.reduce_max(nodes_hw_pred * nodes_valid, axis=[0,1,2]),
            tf.reduce_max(nodes_hw * nodes_valid, axis=[0,1,2])],
                                 message='xy_zmin_zmax_hwpred_hw')

    # L2 loss
    nodes_valid *= tf.minimum(not_occluded, 1.0)
    num_valid = tf.reduce_sum(nodes_valid, axis=[2,3])
    loss = tf.square(nodes_hw_pred - nodes_hw) * nodes_valid
    loss = tf.reduce_sum(loss, axis=[2,3])
    loss = loss / tf.maximum(1., num_valid)
    return loss

def projected_particle_xy_loss(particles_im_indices,
                               particles_hw_norm,
                               size,
                               particles_mask=None,
                               dists2_thresh=0.0
):
    B,T,N,D = particles_im_indices.shape.as_list()
    _,_,_,D2 = particles_hw_norm.shape.as_list()
    assert D == D2

    if particles_mask is None:
        particles_mask = tf.ones([B,T,N,1], tf.float32)

    # transform normalized hw values into those of image indices
    size = tf.reshape(tf.constant(size, tf.float32), [1,1,1,2])
    particles_hw = (particles_hw_norm + 1.0) * (size - 1.0) * (0.5)

    # L2 loss
    num_particles = tf.reduce_sum(particles_mask, axis=2) # [B,T,1]
    dists2 = tf.reduce_sum(particles_mask * tf.square(particles_im_indices - particles_hw), axis=3) # [B,T,N]
    dists2 = tf.maximum(dists2 - dists2_thresh, 0.0)
    loss = tf.div(tf.reduce_sum(dists2, axis=2, keepdims=True),
                  tf.maximum(num_particles, 1.0))
    loss = tf.reduce_mean(loss)
    # loss = tf.Print(loss, [tf.reduce_sum(num_particles), tf.reduce_min(particles_hw), tf.reduce_max(particles_hw),\
    #                        tf.reduce_min(particles_im_indices), tf.reduce_max(particles_im_indices),
    #                        loss], message='xy_loss')
    return loss

def projected_particle_depth_loss(particles_depths,
                                  depth_images,
                                  particles_im_indices,
                                  not_occluded_mask,
                                  particles_mask=None,
                                  image_foreground_masks=None,
                                  log_space=False,
                                  depth_hinge=None,
                                  depth_max=None,
                                  z_offset=0.0
):
    '''
    Projects particles_xyz onto an image and gets the depths of the corresponding pixels.
    Then computes distance between particles_depths and pixels_depths

    particles: [B,T,N,1] of depths
    image: [B,T,H,W,1] the depths image to project onto
    particles_im_indices: [B,T,N,2] of int32 indices into H and W dimensions of an image of size H,W
    not_occluded_mask: [B,T,N,1] float32 where 1.0 indicates the particle wasn't occluded at radius p_radius, 0.0 otherwise

    particles_mask: [B,T,N,1] indicates which particles are valid at each time point
    image_foreground_masks: [B,T,H,W,1] binary mask of which pixels are foreground. Loss will be computed only from these

    returns
    depth_loss: scalar of mean across unmasked particles of depth differences between particles and ground truth

    '''
    B,T,N,D = particles_depths.shape.as_list()
    _,_,H,W,Dim = depth_images.shape.as_list()
    if Dim != 1:
        raise NotImplementedError("only project onto depth or images")

    image_depths = get_image_values_from_indices(depth_images, particles_im_indices) # [B,T,N,Dim]
    # max depth
    if depth_max is not None:
        image_depths = tf.minimum(image_depths, depth_max)

    # masks
    if image_foreground_masks is not None:
        foreground_masks = get_image_values_from_indices(image_foreground_masks, particles_im_indices) # [B,T,N,1]
    else:
        foreground_masks = tf.ones([B,T,N,1], dtype=tf.float32)

    if particles_mask is None:
        particles_mask = tf.ones([B,T,N,1], dtype=tf.float32)
    total_mask = foreground_masks * not_occluded_mask * particles_mask
    # total_mask = tf.Print(total_mask, [tf.reduce_min(total_mask), tf.reduce_max(total_mask)], message='total_mask')

    num_particles = tf.reduce_sum(total_mask, axis=2, keepdims=True) # [B,T,1,1]
    # num_particles = tf.Print(num_particles, [tf.reduce_mean(num_particles)], message='num_depth_particles')
    # image_depths = tf.Print(image_depths, [tf.reduce_min(image_depths), tf.reduce_max(image_depths)], message='gt_depths_minmax')

    # transform depths to log scale; remember particles_depths are negative for ps in real space
    if log_space:
        ps_depths = tf.log(1.0 + tf.nn.relu(-particles_depths))
        image_depths = tf.log(1.0 + image_depths)
    else:
        ps_depths = tf.nn.relu(-particles_depths)

    # L2 loss masekd by which particles project onto real things in image
    depth_loss = tf.square(ps_depths - image_depths - z_offset) # [B,T,N,1]
    depth_loss *= total_mask

    depth_loss = tf.div(tf.reduce_sum(depth_loss, axis=2, keepdims=True), tf.maximum(1.0, num_particles))
    depth_loss = tf.reduce_mean(depth_loss)

    # hinge loss so that depth must be below a certain value
    if depth_hinge is not None:
        hinge_loss = tf.square(tf.maximum(particles_depths - depth_hinge, 0.0) * particles_mask)
        hinge_loss = tf.reduce_mean(tf.reduce_sum(hinge_loss, axis=[2,3]))
        # hinge_loss = tf.Print(hinge_loss, [tf.reduce_sum(num_particles), tf.reduce_max(particles_depths), tf.reduce_max(particles_depths*particles_mask), hinge_loss], message='depth_hinge_loss')
        depth_loss += hinge_loss

    return depth_loss

def projected_particle_normals_loss(particles_normals,
                                    normals_images,
                                    particles_im_indices,
                                    not_occluded_mask,
                                    real_normals_mask=None,
                                    particles_mask=None,
                                    image_foreground_masks=None,
                                    dot_loss=1,
                                    l2_loss=1,
                                    normals_loss_offset=0.0,
                                    normalize_gt=False,
):

    '''
    particles_normals: [B,T,N,3] coded such that [nx,ny,nz] are in range [-1,1] and are unit vectors. positive z is toward camera
    real_normals_mask: [B,T,N,1] some unoccluded particles could have no normal vectors, because they didn't have enough kNN. This masks them out
    '''
    B,T,N,D = particles_normals.shape.as_list()
    _,_,H,W,Dim = normals_images.shape.as_list()
    assert normals_images.dtype == tf.uint8, "You must feed in unprocessed normals in [0,255]"

    # masks
    if image_foreground_masks is not None:
        foreground_masks = get_image_values_from_indices(image_foreground_masks, particles_im_indices) # [B,T,N,1]
    else:
        foreground_masks = tf.ones([B,T,N,1], dtype=tf.float32)
    if real_normals_mask is None:
        real_normals_mask = tf.ones([B,T,N,1], dtype=tf.float32)
    if particles_mask is None:
        particles_mask = tf.ones([B,T,N,1], dtype=tf.float32)

    total_mask = foreground_masks * not_occluded_mask * particles_mask * real_normals_mask
    num_particles = tf.reduce_sum(total_mask, axis=2, keepdims=True)

    # get labels
    image_normals = get_image_values_from_indices(normals_images, particles_im_indices) # [B,T,N,Dim]
    image_normals = tf.cast(image_normals, dtype=tf.float32) / 255.0 # now floats in range [0., 1.]
    image_normals = (2.0*image_normals) - 1.0 # now in range [-1., 1.] like true unit vectors
    if normalize_gt:
        image_normals = tf.nn.l2_normalize(image_normals, axis=-1)

    # add loss terms
    normals_loss = tf.zeros([B,T,N,1], dtype=tf.float32)
    if dot_loss > 0:
        normals_loss += tf.cast(dot_loss, dtype=tf.float32) * (-1.0) * tf.reduce_sum(particles_normals * image_normals, axis=3, keepdims=True)
        normals_loss += normals_loss_offset
    if l2_loss > 0:
        normals_loss += tf.cast(l2_loss, dtype=tf.float32) * tf.reduce_sum(tf.square(particles_normals - image_normals), axis=3, keepdims=True)

    # mask out fake particles
    normals_loss *= total_mask
    normals_loss = tf.div(tf.reduce_sum(normals_loss, axis=2, keepdims=True), tf.maximum(1.0, num_particles))
    normals_loss = tf.reduce_mean(normals_loss)

    return normals_loss

def projected_particle_is_moving_loss(
        is_moving_logits,
        images,
        particles_im_indices,
        depth_images=None,
        particles_mask=None,
        image_foreground_masks=None,
        image_thresh=0.1,
        depth_thresh=0.0,
        cross_entropy=True,
        **kwargs):
    '''
    Cross entropy on whether a given node/projected segment of the image is moving or not;
    "moving" is defined as a change from one frame to the next in the image that's in the foreground.

    is_moving_logits: [B,T,N,1] must be in (-inf, inf)
    images: [B,T,H,W,3] <tf.uint8>
    particles_im_indices: [B,T,N,2] <tf.int32> where the particles project to
    depth_images: [B,T,H,W,1] <tf.float32> depth image, with higher values indicated farther pixels
    particles_mask: [B,T,N,1] <tf.float32> in {0,1} which particles are valid
    image_thresh: <np.float>, if absolute image intensity changes by more than this it will be considered changing
    depth_thresh: if depth_t1 + depth_thresh < depth_t0, then the pixel at depth_t1 is considered foreground
    '''

    B,T,H,W,C = images.shape.as_list()
    _B,_T,N,D = is_moving_logits.shape.as_list()
    if depth_images is not None:
        assert depth_images.shape.as_list() == [B,T,H,W,1]

    # find the pixels that are changing
    ims = tf.cast(images, tf.float32) / 255.
    intensities = tf.reduce_mean(ims, axis=-1, keepdims=True) # [B,T,H,W,1]
    delta_ims = tf.abs(intensities[:,1:] - intensities[:,:-1])
    delta_ims = tf.concat([tf.zeros_like(delta_ims[:,0:1]), delta_ims], axis=1) # [B,T,H,W,1]
    if cross_entropy:
        delta_ims = delta_ims > image_thresh

    # find pixels that are nearer than background
    if depth_images is not None and depth_thresh is not None:
        max_depth = tf.reduce_max(depth_images, axis=1, keepdims=True) # [B,1,H,W,1]
        nearer = (depth_images + depth_thresh) < max_depth
        if cross_entropy:
            delta_ims = tf.logical_and(delta_ims, nearer)
        else:
            delta_ims *= tf.cast(nearer, tf.float32)
        print("using depth images to get motion")

    # get these values as image labels
    is_moving_labels = get_image_values_from_indices(delta_ims, particles_im_indices) # [B,T,N,1] <tf.bool>
    is_moving_labels = tf.cast(is_moving_labels, tf.float32)

    # masking
    if particles_mask is None:
        particles_mask = tf.ones_like(is_moving_logits[...,0:1])
    if image_foreground_masks is not None:
        particles_mask *= tf.cast(get_image_values_from_indices(image_foreground_masks, particles_im_indices), tf.float32)

    num_particles = tf.reduce_sum(particles_mask, axis=2, keepdims=True) # [B,T,1,1]


    # loss
    if cross_entropy:
        is_moving_logits = tf.Print(is_moving_logits, [
            tf.reduce_mean(tf.reduce_sum(particles_mask * tf.cast(is_moving_logits > 0.0, tf.float32), axis=[2,3], keepdims=True) /
                           tf.maximum(1., num_particles)),
            tf.reduce_mean(tf.reduce_sum(particles_mask * tf.cast(is_moving_labels > 0.5, tf.float32), axis=[2,3], keepdims=True) /
                           tf.maximum(1., num_particles))
        ], message='is_movng_logits_labels')
        is_moving_loss = particles_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=is_moving_labels, logits=is_moving_logits)

    else: # l2 regression
        is_moving_logits = tf.Print(is_moving_logits, [
            tf.reduce_mean(tf.reduce_sum(particles_mask * tf.cast(is_moving_logits > image_thresh, tf.float32), axis=[2,3], keepdims=True) /
                           tf.maximum(1., num_particles)),
            tf.reduce_mean(tf.reduce_sum(particles_mask * tf.cast(is_moving_labels > image_thresh, tf.float32), axis=[2,3], keepdims=True) /
                           tf.maximum(1., num_particles))
        ], message='is_movng_reg_logits_labels')
        is_moving_loss = particles_mask * tf.square(is_moving_labels - is_moving_logits)

    is_moving_loss = tf.reduce_sum(is_moving_loss, axis=[2,3], keepdims=True) / tf.maximum(1., num_particles)
    is_moving_loss = tf.reduce_mean(is_moving_loss)

    return is_moving_loss

def projected_optic_flow_loss(
        velocities,
        flow_images,
        particles_im_indices,
        positions,
        projection_matrix,
        not_occluded_mask,
        particles_mask=None,
        image_foreground_masks=None,
        sat_k=None, sat_p=None, max_speed=5.0, eps=1.0e-8,
        **kwargs):
    '''
    velocities: [B,T,N,3] velocities in 3space
    flow_images: [B,T,H,W,3] <tf.uint8> coded as RGB. Hue is angle, Sat is 1.0 for pixels with nonzero flow and 0.0 elsewhere, Val is proportional to flow speed.
    particles_im_indices: [B,T,N,2] <tf.int32> where the particles project to
    depths: [B,T,N,1] needed for velocity projection
    projection_matrix: [B,T,4,4] projection matrix from 3d to 2d
    not_occluded_mask: [B,T,N,1] nodes that aren't occluded
    particles_mask: [B,T,N,1] nodes that are real
    image_foreground_masks: [B,T,H,W,1] or None
    '''
    B,T,N,D = velocities.shape.as_list()
    _,_,H,W,Dim = flow_images.shape.as_list()
    assert D == 3, "Pass 3-velocities"
    assert flow_images.dtype == tf.uint8, "pass uint8 RGB-coded optic flows"

    # compute mask on the nodes
    if image_foreground_masks is not None:
        foreground_masks = get_image_values_from_indices(image_foreground_masks, particles_im_indices)
    else:
        foreground_masks = tf.ones([B,T,N,1], dtype=tf.float32)
    if particles_mask is None:
        particles_mask = tf.ones([B,T,N,1], dtype=tf.float32)

    total_mask = foreground_masks * not_occluded_mask * particles_mask
    num_particles = tf.reduce_sum(total_mask, axis=2, keepdims=True)

    # get flow labels and convert to hsv
    flow_labels = get_image_values_from_indices(flow_images, particles_im_indices) # [B,T,N,3]
    flow_labels = tf.image.rgb_to_hsv(tf.cast(flow_labels, tf.float32) / 255.0)

    # convert 3-velocities to 2D polar coordinates to match labels
    flow_preds = velocities_to_optic_flows_hsv(velocities, positions, projection_matrix, sat_k=sat_k, sat_p=sat_p, vmax=max_speed, eps=eps)

    # compute loss
    h_labels, sv_labels = tf.split(flow_labels, [1,2], axis=-1)
    h_preds, sv_preds = tf.split(flow_preds, [1,2], axis=-1)

    h_loss = tf.square(tf.cos(2.0*np.pi*h_labels) - tf.cos(2.0*np.pi*h_preds))
    h_loss += tf.square(tf.sin(2.0*np.pi*h_labels) - tf.sin(2.0*np.pi*h_preds))
    sv_loss = tf.reduce_sum(tf.square(sv_labels - sv_preds), axis=-1, keepdims=True)
    flow_loss = (h_loss + sv_loss) * total_mask # [B,T,N,1]
    flow_loss = tf.div(tf.reduce_sum(flow_loss, axis=2, keepdims=True),
                        tf.maximum(1.0, num_particles))
    flow_loss = tf.reduce_mean(flow_loss)
    flow_loss = tf.Print(flow_loss, [tf.reduce_mean(h_loss * total_mask), tf.reduce_mean(sv_loss * total_mask)], message='flow_losses')

    return flow_loss

def photometric_optical_flow_loss(
        pred_attrs,
        spatial_inds,
        images,
        size,
        valid_attrs=None,
        valid_images=None,
        images_preproc=preproc_rgb,
        backward=False,
        xy_flows=False,
        gate_on_motion=False,
        flow_attr='pred_flows',
        motion_attr='pred_delta_images',
        motion_preproc=tf.nn.sigmoid,
        motion_thresh=0.5,
        alpha=None,
        eps=1e-3,
        **kwargs
):

    # check and preproc inputs
    flows = pred_attrs[flow_attr]
    if xy_flows:
        flows = tf.concat([-flows[...,1:2], flows[...,0:1]], axis=-1)
    B,T,P,D = flows.shape.as_list()
    assert D == 2, "Must pass estimated dh, dw per point, %s" % flows
    _B,_T,H,W,C = images.shape.as_list()
    assert [_B,_T] == [B,T]
    imsize = [H,W]

    strides = [imsize[0] // size[0], imsize[1] // size[1]]
    if size != imsize:
        images = images[:,:,::strides[0],::strides[1],:]

    images = images_preproc(tf.stop_gradient(images))

    # find valid points
    if valid_attrs is None:
        valid_attrs = tf.ones_like(flows[...,0:1])
    if valid_images is None:
        valid_images = tf.ones_like(images[...,0:1])
    else:
        valid_images = valid_images[:,:,::strides[0],::strides[1],:]
        assert valid_images.shape[:-1] == images.shape[:-1], (valid_images, images)

    if backward:
        spatial_inds = spatial_inds[:,::-1]
        flows = flows[:,::-1]
        images = images[:,::-1]
        valid_images = valid_images[:,::-1]

    valid_points = rendering.get_image_values_from_indices(
        valid_images, spatial_inds)
    valid_points = tf.cast(valid_points, tf.float32) * valid_attrs
    if gate_on_motion:
        is_moving = motion_preproc(pred_attrs[motion_attr][:,::(-1 if backward else 1)]) > motion_thresh
        valid_points *= tf.cast(is_moving, tf.float32)

    num_valid_points = tf.reduce_sum(valid_points, axis=[-2,-1]) # [B,T]
    num_valid_points = tf.maximum(1., num_valid_points)

    # use the flows to infer inds at previous time point
    # flows should be in a normalized range where the image has size [2.,2.]
    inds_t1 = tf.cast(spatial_inds[:,1:], tf.float32)
    size_float = tf.reshape(tf.cast(size, tf.float32), [1,1,1,2])
    flows_scaled = flows[:,1:] * (size_float / 2.0)
    inds_t0 = inds_t1 - flows_scaled
    inds_t0 = tf.minimum(tf.maximum(inds_t0, 0.0), size_float - 1.0)
    inds_t0h, inds_t0w = tf.split(inds_t0, [1,1], axis=-1) # [B,T-1,N,1] each <tf.float32>

    # differentiable bilinear sampling, i.e. spatial transformer
    floor_t0h = tf.floor(inds_t0h)
    floor_t0w = tf.floor(inds_t0w)
    ceil_t0h = tf.ceil(inds_t0h)
    ceil_t0w = tf.ceil(inds_t0w)

    bot_right_weight = (inds_t0h - floor_t0h) * (inds_t0w - floor_t0w)
    bot_left_weight = (inds_t0h - floor_t0h) * (ceil_t0w - inds_t0w)
    top_right_weight = (ceil_t0h - inds_t0h) * (inds_t0w - floor_t0w)
    top_left_weight = (ceil_t0h - inds_t0h) * (ceil_t0w - inds_t0w)

    if PRINT:
        top_left_weight = tf.Print(top_left_weight, [
            tf.reduce_mean(bot_right_weight),
            tf.reduce_mean(bot_left_weight),
            tf.reduce_mean(top_right_weight),
            tf.reduce_mean(top_left_weight),
            tf.reduce_min(bot_right_weight + bot_left_weight + top_right_weight + top_left_weight),
            tf.reduce_max(bot_right_weight + bot_left_weight + top_right_weight + top_left_weight),
            tf.reduce_mean(bot_right_weight + bot_left_weight + top_right_weight + top_left_weight)], message='sample_inds/weights')

    _to_int = lambda x: tf.cast(x, tf.int32)
    ims_t0 = images[:,:-1]
    top_left_vals = rendering.get_image_values_from_indices( # [B,T-1,P,C]
        ims_t0, _to_int(tf.concat([floor_t0h, floor_t0w], axis=-1)))
    top_right_vals = rendering.get_image_values_from_indices(
        ims_t0, _to_int(tf.concat([floor_t0h, ceil_t0w], axis=-1)))
    bot_left_vals = rendering.get_image_values_from_indices(
        ims_t0, _to_int(tf.concat([ceil_t0h, floor_t0w], axis=-1)))
    bot_right_vals = rendering.get_image_values_from_indices(
        ims_t0, _to_int(tf.concat([ceil_t0h, ceil_t0w], axis=-1)))

    ims_t0_vals = top_left_vals * top_left_weight +\
                  top_right_vals * top_right_weight +\
                  bot_left_vals * bot_left_weight +\
                  bot_right_vals * bot_right_weight

    ims_t1_vals = rendering.get_image_values_from_indices(
        images[:,1:], spatial_inds[:,1:])

    # photo_loss = tf.reduce_sum(tf.square(ims_t0_vals - ims_t1_vals), axis=-1, keepdims=True) # [B,T-1,P,1]
    # if alpha is not None:
    #     photo_loss = tf.pow(photo_loss + (eps**2), alpha)

    if alpha is not None:
        charb_func = lambda x: tf.pow(x**2 + eps**2, alpha)
    else:
        charb_func = lambda x: x**2
    photo_loss = tf.reduce_sum(charb_func(ims_t0_vals - ims_t1_vals), axis=-1, keepdims=True)

    # mask out invalid
    in_bounds = (bot_right_weight + bot_left_weight + top_right_weight + top_left_weight) > 0.49
    in_bounds = tf.cast(in_bounds, tf.float32)
    valid_points = valid_points[:,1:] * in_bounds

    photo_loss *= valid_points
    photo_loss = tf.reduce_sum(photo_loss, axis=[-2,-1]) / num_valid_points[:,1:]

    # concat on zeros for consistency
    photo_loss = tf.concat([tf.zeros_like(photo_loss[:,0:1]), photo_loss], axis=1)

    return photo_loss


def projected_photometric_optic_flow_loss(
        velocities,
        images,
        positions,
        particles_im_indices,
        projection_matrix,
        particles_mask=None,
        is_moving_particles=None,
        gate_on_motion=False,
        projected_flows=True,
        image_foreground_masks=None,
        backward_euler=True,
        images_norm=255.,
        alpha=None,
        eps=1e-3,
        **kwargs
):
    '''
    velocities: [B,T,N,3]
    images: [B,T,H,W,C] could be RGB images or feature channels
    particles_im_indices: [B,T,N,2] <tf.int32>
    positions: [B,T,N,3] <tf.float32> xyz
    projection_matrix: [B,T,4,4] for projecting from 3d onto normalized image grid


    based on https://arxiv.org/pdf/1608.05842.pdf
    '''
    B,T,N,Dv = velocities.shape.as_list()
    _B,_T,H,W,C = images.shape.as_list()
    if images.dtype != tf.float32:
        images = tf.cast(images, tf.float32) / images_norm

    # don't want to backprop into features!
    images = tf.stop_gradient(images)

    if projected_flows:
        vels = velocities[:,1:] # [B,T-1,N,3]
        hw_t1 = agent_particles_to_image_coordinates(
            positions[:,1:], projection_matrix[:,1:], H_out=H, W_out=W, to_integers=False)
        hw_t0 = agent_particles_to_image_coordinates(
            positions[:,1:] - vels, projection_matrix[:,1:], H_out=H, W_out=W, to_integers=False)
        flows_back = hw_t0 - hw_t1 # how to get from t1 to t0 in units of image indices
        inds_t1 = tf.cast(particles_im_indices[:,1:], tf.float32)
        inds_t0 = tf.minimum(tf.maximum( # don't go off the grid
            inds_t1 + flows_back, 0.0), tf.reshape(tf.constant([H-1.,W-1.], tf.float32), [1,1,1,2]))

    else:
        print("just using 2D flows with no camera projection")
        vels = velocities[:,1:,...,0:2] # just take these to be the 2d flows
        inds_t1 = tf.cast(particles_im_indices[:,1:], tf.float32)
        inds_t0 = inds_t1 - vels
        inds_t0 = tf.minimum(tf.maximum(inds_t0, 0.0), tf.reshape(tf.constant([H-1.,W-1.], tf.float32), [1,1,1,2]))

    inds_t0h, inds_t0w = tf.split(inds_t0, [1,1], axis=-1) # [B,T-1,N,1] each <tf.float32>

    # differentiable bilinear sampling, i.e. spatial transformer
    floor_t0h = tf.floor(inds_t0h)
    floor_t0w = tf.floor(inds_t0w)
    ceil_t0h = tf.ceil(inds_t0h)
    ceil_t0w = tf.ceil(inds_t0w)

    bot_right_weight = (inds_t0h - floor_t0h) * (inds_t0w - floor_t0w)
    bot_left_weight = (inds_t0h - floor_t0h) * (ceil_t0w - inds_t0w)
    top_right_weight = (ceil_t0h - inds_t0h) * (inds_t0w - floor_t0w)
    top_left_weight = (ceil_t0h - inds_t0h) * (ceil_t0w - inds_t0w)

    top_left_weight = tf.Print(top_left_weight, [
        tf.reduce_mean(bot_right_weight),
        tf.reduce_mean(bot_left_weight),
        tf.reduce_mean(top_right_weight),
        tf.reduce_mean(top_left_weight),
        tf.reduce_min(bot_right_weight + bot_left_weight + top_right_weight + top_left_weight),
        tf.reduce_max(bot_right_weight + bot_left_weight + top_right_weight + top_left_weight),
        tf.reduce_mean(bot_right_weight + bot_left_weight + top_right_weight + top_left_weight)], message='sample_inds/weights')

    _to_int = lambda x: tf.cast(x, tf.int32)
    ims_t0 = images[:,:-1]
    top_left_vals = get_image_values_from_indices( # [B,T-1,N,C]
        ims_t0, _to_int(tf.concat([floor_t0h, floor_t0w], axis=-1)))
    top_right_vals = get_image_values_from_indices(
        ims_t0, _to_int(tf.concat([floor_t0h, ceil_t0w], axis=-1)))
    bot_left_vals = get_image_values_from_indices(
        ims_t0, _to_int(tf.concat([ceil_t0h, floor_t0w], axis=-1)))
    bot_right_vals = get_image_values_from_indices(
        ims_t0, _to_int(tf.concat([ceil_t0h, ceil_t0w], axis=-1)))

    ims_t0_vals = top_left_vals * top_left_weight +\
                  top_right_vals * top_right_weight +\
                  bot_left_vals * bot_left_weight +\
                  bot_right_vals * bot_right_weight

    ims_t1_vals = get_image_values_from_indices(
        images[:,1:], particles_im_indices[:,1:])

    photo_loss = tf.reduce_sum(tf.square(ims_t0_vals - ims_t1_vals), axis=-1, keepdims=True) # [B,T-1,N,1]
    if alpha is not None:
        photo_loss = tf.pow(photo_loss + eps, alpha)

    # mask
    if particles_mask is None:
        particles_mask = tf.ones([B,T-1,N,1])
    else:
        particles_mask = particles_mask[:,1:]
    if image_foreground_masks is not None:
        foreground_masks = get_image_values_from_indices(image_foreground_masks, particles_im_indices[:,1:])
        particles_mask *= foreground_masks
    if gate_on_motion:
        assert is_moving_particles is not None
        particles_mask *= tf.cast(tf.nn.sigmoid(is_moving_particles[:,1:]) > 0.5, tf.float32)

    in_bounds = (bot_right_weight + bot_left_weight + top_right_weight + top_left_weight) > 0.49
    in_bounds = tf.cast(in_bounds, tf.float32)
    particles_mask *= in_bounds

    num_particles = tf.reduce_sum(particles_mask, axis=2, keepdims=True) # [B,T-1,1,1]
    photo_loss = tf.reduce_sum(particles_mask * photo_loss, axis=[2,3], keepdims=True) / tf.maximum(1., num_particles)
    photo_loss = tf.reduce_mean(photo_loss)
    photo_loss = tf.Print(photo_loss, [
        tf.reduce_mean(tf.reduce_sum(velocities[:,1:] * particles_mask, axis=2, keepdims=True) / tf.maximum(1., num_particles), axis=[0,1,2]),
        photo_loss], message='vels/flow_photo_loss')
    return photo_loss

def attr_variance_loss(images_list, segments, attrs, valid_attrs,
                       valid_images=None,
                       mask_images=[False, True, True],
                       attrs_dims_list=[[9,12]],
                       images_preproc_list=[tf.image.rgb_to_hsv],
                       attrs_preproc_list=[tf.identity],
                       loss_scales_list=None, **kwargs):

    B,T,H,W = segments.shape.as_list() # int32 in [0,N)
    image_shape = images_list[0].shape.as_list()
    if len(image_shape) == 4: # add time
        images_list = [im[:,tf.newaxis] for im in images_list]
    _, Tim, Him, Wim, _ = images_list[0].shape.as_list()
    s = [Him // H, Wim // W]
    # _, Tim, Him, Wim, Cim = images.shape.as_list()

    N,D = attrs.shape.as_list()[-2:]
    assert len(images_list) == len(attrs_dims_list) == len(images_preproc_list) == \
        len(attrs_preproc_list) == len(loss_scales_list) == len(mask_images), "Must pass one set of dims, preproc, and loss scale for each attr"

    # preproc segments
    segments, valid_segments, num_valid_pix = preproc_segment_ids(segments, Nmax=N, return_valid_segments=True) # [B,T,H,W] now in range [0,N)
    segments = tf.reshape(segments, [B*Tim,H,W])

    offsets = N * tf.reshape(tf.range(B*Tim, dtype=tf.int32), [B*Tim,1,1])
    segments += offsets # now inds for each batch/time start at Nmax*(b*Tim + t)
    segments = tf.reshape(segments, [B,Tim,H,W])
    Nmax = N*B*Tim

    # preproc valid
    valid_segments = tf.expand_dims(valid_segments, -1)

    # preproc images
    images_list = [preproc_func(images_list[i]) for i, preproc_func in enumerate(images_preproc_list)]

    # segments and images must be same size
    if (H != Him) or (W != Wim):
        images_list = [image[:,:,::s[0],::s[1]] for image in images_list]

    # valid images
    if valid_images is not None:
        valid_images = tf.cast(valid_images[:,:,::s[0],::s[1]], tf.float32)
    else:
        valid_images = tf.ones([B,Tim,H,W,1], tf.float32)
    valid_images = [valid_images if mask else tf.ones_like(valid_images) for mask in mask_images] # list

    # get gt variances
    gt_areas = [tf.maximum(1.0, tf.math.unsorted_segment_sum(valid_images[i], segments, num_segments=Nmax)) for i in range(len(images_list))]
    gt_vars = [
        (tf.math.unsorted_segment_sum(tf.square(image * valid_images[i]), segments, num_segments=Nmax) / gt_areas[i]) - \
        tf.square(tf.math.unsorted_segment_sum(image * valid_images[i], segments, num_segments=Nmax) / gt_areas[i])
        for i,image in enumerate(images_list)] # t3

    gt_vars = [tf.reshape(gtv, [B,Tim,N,-1]) for gtv in gt_vars]

    # compute valid regions of the input
    gt_valid = [tf.ones([B,Tim,N,1], dtype=tf.float32) for i in range(len(images_list))]

    # preproc attrs
    attr_vars = [attrs_preproc_list[i](attrs[...,d[0]:d[1]]) for i,d in enumerate(attrs_dims_list)]
    num_valid_attrs = [tf.reduce_sum(valid_attrs * gt_valid[i], axis=[2,3]) for i in range(len(attrs_dims_list))] # [B,T]


    # loss
    loss_scales_list = loss_scales_list or [1.0] * len(attr_vars)

    losses = [tf.square(gt_vars[i] - attr_vars[i]) * valid_attrs * gt_valid[i] for i in range(len(attr_vars))]
    losses = [tf.reduce_mean(tf.reduce_sum(attr_loss, axis=[2,3]) / tf.maximum(1.0, num_valid_attrs[i])) * loss_scales_list[i] for i,attr_loss in enumerate(losses)]

    total_loss = tf.cast(0.0, tf.float32)
    for i,loss in enumerate(losses):
        total_loss += loss * tf.cast(tf.logical_not(tf.logical_or(tf.math.is_nan(loss), tf.is_inf(loss))), tf.float32)

    return total_loss

def metric_diffs_to_order_labels(diffs, eps=0.1, **kwargs):
    '''
    Converts a tensor [B,...,1] <tf.float32> of real values in (-inf, inf) to
    values in {-1., 0., 1.} as follows:

    if |d_in| < eps:
        d_out = 0.
    else:
        d_out = sign(d_in)
    '''
    return tf.where(tf.abs(diffs) < tf.cast(eps, tf.float32),
                    tf.zeros_like(diffs, dtype=tf.float32), # True
                    tf.cast(tf.sign(diffs), tf.float32)) # False

def relative_depth_loss(relative_depth_preds, relative_depth_labels, invert=False, clip_val=30.0, **kwargs):
    '''
    from http://papers.nips.cc/paper/6489-single-image-depth-perception-in-the-wild.pdf

    preds: [shape[:-1],1] in (-inf, inf)
    labels: [shape[:-1],1] in {-1., 0., 1.}
    invert: If True, labels |--> -labels
    '''
    signs = -1.0 if invert else 1.0
    # clip to avoid overflow in tf.exp
    relative_depth_preds = tf.where(tf.abs(relative_depth_preds) > clip_val,
                                    clip_val * tf.cast(tf.sign(relative_depth_preds), tf.float32), # true
                                    relative_depth_preds) # false

    # case of equal gt depth:
    loss = (1.0 - tf.abs(relative_depth_labels)) * tf.square(relative_depth_preds)
    # case of unequal gt depth:
    # e.g. if label = 1.0, invert = False, loss --> 0 as pred --> inf
    loss += tf.abs(relative_depth_labels) * tf.log(1.0 + tf.exp(-1.0 * signs * relative_depth_labels * relative_depth_preds))

    return loss # [...,1] same shape as original

def relative_normals_loss(logits, labels, beta=2.0, **kwargs):
    '''
    labels: {0. if normals are "same", 1.0 if different}
    logits: normals dot distance, assumed to be in range [0.,2.]
    '''

    loss = (1.0 - labels) * logits # in [0., 2.]
    loss += labels * tf.log(1.0 + tf.exp(-1.0 * beta * logits))
    # loss = tf.Print(loss, [tf.reduce_mean(loss),
    #                        tf.reduce_sum(1.0 - tf.abs(labels)),
    #                        tf.reduce_sum(tf.cast(tf.equal(labels, 1.0), tf.float32)),
    #                        tf.reduce_sum(tf.cast(tf.equal(labels, -1.0), tf.float32))],
    #                 message='rN_loss')

    return loss

def attr_diffs_loss(pred_attr_diffs,
                    gt_attr_diffs,
                    valid_diffs,
                    diffs_to_labels_func=metric_diffs_to_order_labels,
                    loss_func=relative_depth_loss, # signature func(preds, labels)
                    loss_scale=1.0,
                    **kwargs):
    '''
    Convert gt attr diffs to labels if necessary, then pass through loss func
    Take mean over valid diffs
    '''

    assert pred_attr_diffs.shape.as_list() == gt_attr_diffs.shape.as_list(), (pred_attr_diffs.shape, gt_attr_diffs.shape)
    B,N,K,D = pred_attr_diffs.shape.as_list()
    num_valid_diffs = tf.reduce_sum(valid_diffs, axis=[1,2,3]) # [B]
    gt_attr_labels = diffs_to_labels_func(gt_attr_diffs, **kwargs)
    loss = loss_func(pred_attr_diffs, gt_attr_labels, **kwargs)
    assert loss.shape.as_list() == [B,N,K,D]
    loss = tf.reduce_sum(loss * valid_diffs, axis=[1,2,3]) / tf.maximum(1.0, num_valid_diffs)
    # loss = tf.reduce_mean(loss)
    return loss * loss_scale

def relative_spatial_attributes_loss(
        nodes, dimension_dict, labels, valid_images=None,
        attr_to_image_dict={'pred_depths': 'depths'},
        attr_metrics={'pred_depths': depth_metric},
        image_preprocs={'depths': preproc_depths},
        image_loss_funcs={'depths': relative_depth_loss},
        image_loss_func_kwargs={'depths': {'eps':0.1}},
        loss_scales={'depths': 1.0},
        num_sample_points=256, kNN=128,
        hw_attr='hw_centroids',
        valid_attr='valid',
        **kwargs
):
    imkeys = attr_to_image_dict.values()
    B,T,H,W,_ = labels[imkeys[0]].shape.as_list()
    P = np.minimum(num_sample_points, nodes['vector'].shape.as_list()[2])
    imsize = [H,W]
    if valid_images is None:
        valid_images = tf.ones([B,T,H,W,1], dtype=tf.float32)
    valid_images = tf.cast(valid_images, tf.float32)

    nodes_hw = dimension_dict.get_attr(nodes, hw_attr, position=-1, sort=True, with_key=False)[:,:,:P]
    nodes_valid = dimension_dict.get_attr(nodes, valid_attr, position=-1, sort=True, with_key=False)[:,:,:P]
    nodes_hw = tf.reshape(nodes_hw, [B*T,P,2])
    nodes_valid = tf.reshape(nodes_valid, [B*T,P,1])
    sample_inds = rendering.hw_attrs_to_image_inds(nodes_hw, imsize)
    sample_inds = tf.reshape(sample_inds, [B,T,P,2])

    knn_inds = graphical.find_nearest_k_node_inds(
        nodes=tf.concat([nodes_hw, nodes_valid], axis=-1),
        kNN=kNN, nn_dims=[0,2]) # [BT,P,K]

    b_inds = tf.tile(tf.range(B*T, dtype=tf.int32)[:,tf.newaxis,tf.newaxis],
                     [1,P,kNN])
    pred_inds = tf.stack([b_inds, knn_inds], axis=-1) # [BT,P,K,2]

    # get valid image positions
    valid_points = rendering.get_image_values_from_indices(
        valid_images, sample_inds)
    valid_points = tf.reshape(valid_points, [B*T,P,1])
    valid_image_pairs = tf.gather_nd(valid_points, pred_inds) # [BT,P,K,1]

    # node attrs
    nodes_attrs = dimension_dict.get_tensor_from_attrs(
        nodes['vector'], attr_to_image_dict.keys(), postproc=True, stop_gradient=False, concat=False)
    nodes_attrs = {k:tf.reshape(v[:,:,:P], [B*T,P,-1]) for k,v in nodes_attrs.items()}
    nodes_attrs_diffs = {
        attr: graphical.attr_diffs_from_neighbor_inds(
            nodes=nodes_attr,
            neighbor_inds=knn_inds,
            valid_nodes=nodes_valid,
            attr_dims_list=[[0,nodes_attr.shape.as_list()[-1]]],
            attr_metrics_list=[attr_metrics.get(attr, tf.subtract)],
            mask_self=True)
        for attr, nodes_attr in nodes_attrs.items()}
    valid_diffs = nodes_attrs_diffs[nodes_attrs.keys()[0]][1]
    valid_diffs = valid_diffs * valid_image_pairs
    nodes_attrs_diffs = {k:tf.concat(diff[0], axis=-1) for k,diff in nodes_attrs_diffs.items()}

    loss = tf.zeros([B,T], dtype=tf.float32)
    for attr, imkey in attr_to_image_dict.items():
        im_preproc = image_preprocs.get(imkey, tf.identity)
        image = im_preproc(labels[imkey])
        Cim = image.shape.as_list()[-1]

        image_points = rendering.get_image_values_from_indices(
            image, sample_inds) # [B,T,P,Cim]
        image_points = tf.reshape(image_points, [B*T,P,Cim])
        image_pairs = tf.gather_nd(image_points, pred_inds) # [BT,P,K,Cim]
        image_diffs = attr_metrics.get(attr, tf.subtract)(
            image_points[:,:,tf.newaxis], image_pairs) # [BT,P,K,Cim]
        pred_diffs = nodes_attrs_diffs[attr][...,:Cim] # [BT,P,K,Cim]

        loss_func = image_loss_funcs.get(imkey, relative_depth_loss)
        loss_kwargs = image_loss_func_kwargs.get(imkey, {})
        loss_scale = loss_scales.get(imkey, 1.0)

        attr_loss = attr_diffs_loss(
            pred_diffs, image_diffs, valid_diffs,
            loss_func=loss_func, loss_scale=loss_scale, **loss_kwargs)

        attr_loss = tf.reshape(attr_loss, [B,T])
        loss += attr_loss

    return loss

def depth_normals_consistency_loss(nodes, kNN=4, nn_dims=None, xyz_dims=[0,3], normals_dims=[6,9], eps=0.05, loss_scale=1.0):
    '''
    Constrains neighboring nodes such that dot(d_xyz, avg_normals) ~= 0.
    '''
    nn_dims = nn_dims or xyz_dims
    nearest_k_inds = find_nearest_k_node_inds(nodes, kNN=kNN, nn_dims=nn_dims)
    def _unit_delta(x,y):
        return tf.nn.l2_normalize(x-y, axis=-1)
    def _avg_normal(nx,ny):
        nx = tf.nn.l2_normalize(nx, axis=-1)
        ny = tf.nn.l2_normalize(ny, axis=-1)
        avg = 0.5 * (nx + ny)
        return tf.nn.l2_normalize(avg, axis=-1)

    neighbors, valid_diffs = attr_diffs_from_neighbor_inds(
        nodes, nearest_k_inds, attr_dims_list=[xyz_dims, normals_dims],
        attr_metrics_list=[_unit_delta, _avg_normal])
    deltas, avg_normals = neighbors

    num_valid_diffs = tf.reduce_sum(valid_diffs, axis=[1,2,3]) # [B]
    dots = tf.reduce_sum(deltas * avg_normals, axis=-1, keepdims=True)
    loss = tf.nn.relu(tf.abs(dots) - eps) # [B,N,K,1]
    loss = tf.reduce_sum(loss * valid_diffs, axis=[1,2,3]) / tf.maximum(1.0, num_valid_diffs)
    loss = tf.reduce_mean(loss)
    return loss * loss_scale

def projected_particle_foreground_loss(particles_im_indices,
                                       not_occluded_mask,
                                       image_foreground_masks,
                                       im_size=[256,256],
                                       stride=None,
                                       mask_value=-100.0,
                                       batch_scale=None):
    '''
    Works differently from other projection losses.

    Uses tf.where to find all (h,w) indices where foreground mask is nonzero
    Then computes a chamfer distance on these and the predicted indices.
    A batch dimension value is added to the particles_im_indices and scaled by a large amount
        so that nearest neighbor particles will always be members of the same example;
        this is a hack but better than running chamfer_distance once per batch.
    '''
    B,T,N,D = particles_im_indices.shape.as_list()
    H,W = im_size
    if stride is None:
        stride = H // 32
    assert D == 2, "Pass h,w indices"
    assert particles_im_indices.dtype == tf.float32, "particle indices must be floats, otherwise no gradient"
    if image_foreground_masks is not None:
        assert image_foreground_masks.shape.as_list()[2:4] == im_size, "must pass correct image size if using a mask"
    # set up a uniform mask if one isn't provided
    if image_foreground_masks is not None:
        image_foreground_masks = tf.cast(image_foreground_masks, tf.bool)
    if batch_scale is None:
        batch_scale = 25.0 * H

    loss = 0.0
    for t in range(T):
        if image_foreground_masks is not None:
            dist_thresh = None
            fg_mask = image_foreground_masks[:,t,:,:,0]
            num_gt_particles = tf.maximum(tf.cast(tf.reduce_sum(tf.cast(fg_mask, tf.int32)), tf.float32), 1.0)
            gt_inds = tf.where(fg_mask) # [?, 3]
        else:
            m = H // stride
            dist_thresh = 2.0*(((stride-1.0) / 2.0)**2)
            # dist_thresh=0.0
            num_gt_particles = tf.constant(B*m*m, tf.float32)
            gt_inds = tf.range(0,H,stride, dtype=tf.int32)
            gt_h = tf.reshape(tf.stack([gt_inds]*m, axis=1), [m**2,1])
            gt_w = tf.concat([gt_inds]*m, axis=0)[:,tf.newaxis]
            gt_b = tf.reshape(
                tf.stack([tf.range(B, dtype=tf.int32)]*(m**2), axis=1),
                [B*(m**2),1])
            gt_inds = tf.concat([
                gt_b,
                tf.concat([gt_h]*B, axis=0),
                tf.concat([gt_w]*B, axis=0)
            ], axis=1) # [B*(m**2), 3]

        # scale up batch dimension to be very large
        gt_batch, gt_hw = tf.split(gt_inds, [1,2], axis=-1)
        gt_batch = batch_scale * (tf.cast(gt_batch, tf.float32) + 1.0) # need to 1-index to avoid nearest neighbors
        gt_inds = tf.concat([gt_batch, tf.cast(gt_hw, tf.float32)], axis=-1)

        # add fake particles to be nearest neighbors with occluded particles
        fake_gt_inds = tf.concat([
            batch_scale * tf.reshape(tf.range(1, B+1, dtype=tf.float32), [B,1]),
            tf.constant(value=mask_value, shape=[B,2], dtype=tf.float32)
        ], axis=-1) # [B,3]
        gt_inds = tf.concat([gt_inds, fake_gt_inds], axis=0) # [?+B, 3]
        gt_inds = gt_inds[tf.newaxis,...] # [1, ?+B, 3]

        # add batch index to pred_inds and reshape
        pred_inds = particles_im_indices[:,t] # [B,N,2]
        pred_inds = mask_tensor(pred_inds, not_occluded_mask[:,t], mask_value)
        batch_inds = batch_scale * tf.reshape(tf.range(1, B+1, dtype=tf.float32), [B,1,1]) * tf.ones_like(pred_inds[...,0:1])
        pred_inds = tf.reshape(tf.concat([batch_inds, pred_inds], axis=-1), [1,-1,3]) # [1, B*N, 3]
        num_pred_particles = tf.maximum(tf.reduce_sum(not_occluded_mask[:,t]), 1.0)

        loss += training_utils.particle_nn_loss(
            logits=pred_inds,
            labels=gt_inds,
            num_pred_particles=num_pred_particles,
            num_gt_particles=num_gt_particles,
            loss_multiplier=1.0,
            dist_thresh=dist_thresh,
            two_way=True
        )

        if t == T-1:
            loss = tf.Print(loss, [num_pred_particles, num_gt_particles, loss], message="num pred/gt particles/loss")

    return loss

if __name__ == '__main__':

    # B = 4
    # pred = tf.nn.sigmoid(tf.random.normal([B,512,16], dtype=tf.float32) * 10.)
    # gt = tf.random.uniform(shape=[B,512], minval=0, maxval=8, dtype=tf.int32)
    # gt = tf.cast(tf.one_hot(gt, depth=8, axis=-1), tf.float32) # [B,512,8]
    # pred = tf.concat([gt[...,4:], gt[...,0:4]], axis=-1)
    # loss = hungarian_dice_loss(pred, gt)

    ## test of hungarian dice loss to optimize some masks
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sess = tf.Session()

    STEPS = 10

    ## some labels and random initial logits
    labels = tf.constant([0,0,5,3,1,2,7,7,5,4,3,3,6,0,0], dtype=tf.int32)
    labels = tf.cast(tf.one_hot(labels, depth=8, axis=-1), tf.float32)
    logits = 0.01 * tf.random.normal(shape=labels.shape, dtype=tf.float32)

    ## give batch dim
    logits = tf.stack([logits]*2, axis=0)
    labels = tf.stack([labels]*2, axis=0)

    ## gradient descent on logits
    for step in range(STEPS):
        logits = tf.minimum(tf.nn.relu(logits), 1.)
        logits /= tf.reduce_max(logits, axis=2, keepdims=True) + 1e-8
        logits /= tf.reduce_sum(logits, axis=1, keepdims=True)
        loss = hungarian_dice_loss(logits, labels)
        print("loss at step {}: {}".format(step, sess.run(loss)))
        if step + 1 != STEPS:
            grad = tf.gradients(loss, logits)[0]
            logits -= grad

    ## normalize
    logits /= tf.reduce_max(logits, axis=1, keepdims=True)
    print("final logits")
    print(sess.run(logits[-1]))
    print("labels")
    print(sess.run(labels[-1]))
