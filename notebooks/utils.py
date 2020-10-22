import numpy as np
import pickle as cPickle
import gridfs
import scipy.signal as signal
import pymongo as pm
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.cm as cm
import sys
import tensorflow as tf
import copy
import os

def return_metrics(conn, dbname='test_phys', collname='po_net', exp_id='run2', valid_key='valid0', loss_keys='particle_only_loss'):
    if '.files' not in collname:
        coll = conn[dbname][collname+'.files']
    else:
        coll = conn[dbname][collname]

    if not isinstance(loss_keys, list):
        loss_keys = [loss_keys]

    if loss_keys == ['val_loss']:
        val_loss = [r['validation_results'][valid_key]['val_loss'] for r in coll.find({'exp_id': exp_id,
                                                         'validation_results': {'$exists': True}}).sort('step')]
        return {'val_loss': val_loss}
    elif len(loss_keys) == 1:
        val_loss = [r['validation_results'][valid_key]['loss'] for r in coll.find({'exp_id': exp_id,
                                                         'validation_results': {'$exists': True}}).sort('step')]

        particle_only_loss = [r['validation_results'][valid_key][loss_keys[0]] for r in coll.find({'exp_id': exp_id,
                                                         'validation_results': {'$exists': True}}).sort('step')]


        return val_loss, particle_only_loss
    else:
        loss_dict = {}
        # if 'loss' not in loss_keys:
        #     loss_dict['loss'] = [r['validation_results'][valid_key]['loss'] for r in coll.find({'exp_id': exp_id,
        #                                                  'validation_results': {'$exists': True}}).sort('step')]
        for k in loss_keys:
            loss_dict[k] = [r['validation_results'][valid_key][k] for r in coll.find({'exp_id': exp_id,
                                                         'validation_results': {'$exists': True}}).sort('step')]

        return loss_dict

def load_from_gridfs(conn, dbname='test_phys', collname='po_net', exp_id='run2', expid_prefix='eval_part', load_step=190000):
#     print('Loading results for file ' + str(file))
    if load_step is not None:
        suffix = '_'+str(load_step)
    else:
        suffix = ''
    expidstr = expid_prefix+exp_id+suffix
    print(expidstr)
    r = conn[dbname][collname+'.files'].find_one({'exp_id': expidstr})
    _id = r['_id']
    fn = str(_id) + '_fileitems'
    fsys = gridfs.GridFS(conn[dbname], collname)
    fh = fsys.get_last_version(fn)
    fstr = fh.read()
    fh.close()
    obj = cPickle.loads(fstr, encoding='latin1') # python3
#     obj = pickle.loads(fstr)
    targets = obj['validation_results'][expid_prefix]
    # targets = obj['validation_results']
    return targets

def plot_smooth_trainloss(conn, dbname, collname, exp_id, N=1, ylim=None, xlim=None, nanfilter=True, loss_keys=['loss', 'learning_rate'], loss_scales={}):
    coll = conn[dbname][collname+'.files']
    train_loss = np.concatenate([[[_r[lkey] for lkey in loss_keys] for _r in r['train_results']]
                            for r in coll.find(
                                       {'exp_id': exp_id, 'train_results': {'$exists': True}},
                                        projection=['train_results'])])

    for l, lk in enumerate(loss_keys):
        if lk == 'learning_rate':
            continue
        this_loss = train_loss[:,l:l+1]
        if nanfilter:
            this_loss = this_loss[~np.isnan(this_loss[:,0]),:]
        smooth_this_loss = np.convolve(this_loss[:,0], (1./N)*np.ones(N), 'valid')
        mult = loss_scales.get(lk, 1.0)
        plt.plot(mult * smooth_this_loss, label=lk)
    if ylim is not None:
        plt.ylim(ylim)
    if xlim is not None:
        plt.xlim(xlim)
    plt.legend(loc='upper right')
    plt.xlabel('training steps', fontsize=16)
    plt.ylabel('training loss', fontsize=16)
    plt.title('training losses for ' + exp_id)
    plt.show()

    return train_loss

def plot_val_losses(conn, dbname, collname, exp_id, valid_key='object_metrics', save_valid_freq=5000, plot_ticks_freq=5000, get_losses=False, start=0, end=None, validate_first=False,
                    loss_keys=['mIoU_matched_t0'], maxes=False,
                    transform_y=None, ylabel_prefix="", colors=['tab:blue', 'tab:red', 'tab:green', 'tab:purple', 'tab:orange', 'tab:cyan', 'magenta', 'gold', 'gray']):

    loss_dict = return_metrics(conn=conn, dbname=dbname, collname=collname, exp_id=exp_id,
                               valid_key=valid_key, loss_keys=loss_keys)

    # return loss_dict
    num_iters = save_valid_freq * len(loss_dict[loss_keys[0]])
    x = np.arange(0, num_iters, save_valid_freq)
    if not validate_first:
        x = [_x + x[1] for _x in x]
    if end is None:
        end = len(x)
    val_loss = np.array(loss_dict[loss_keys[0]])
    val_loss[np.isnan(val_loss)] = np.max(val_loss[~np.isnan(val_loss)])

    host = host_subplot(111, axes_class=AA.Axes)
    plt.subplots_adjust(right=0.75)

    transform_y = transform_y or (lambda x: x)
    host.plot(x[start:end], transform_y(val_loss)[start:end], color=colors[0])
    host.set_xlabel("val step (10^3)")
    ptf = plot_ticks_freq // save_valid_freq
    host.set_xticks(x[start:end:ptf])
    host.set_xticklabels(["%.0f" % (xi/1000) for xi in x[start:end:ptf]])
    host.set_ylabel((ylabel_prefix+"  %s") % (loss_keys[0]))
    host.axis["left"].label.set_color(colors[0])

    pars = []
    if maxes:
        loss_mins = {loss_keys[0]: ((start+np.argmax(val_loss[start:end])+(1-int(validate_first)))*save_valid_freq, np.max(val_loss[start:end]))}
    else:
        loss_mins = {loss_keys[0]: ((start+np.argmin(val_loss[start:end])+(1-int(validate_first)))*save_valid_freq, np.min(val_loss[start:end]))}
    val_losses = {loss_keys[0]: val_loss}
    n_axes = len(loss_keys) - 1
    offset=60
    if n_axes:
        for i in range(n_axes):
            par = host.twinx()
            if i>0:
                new_axis = par.get_grid_helper().new_fixed_axis
                par.axis["right"] = new_axis(loc="right", axes=par, offset=((i)*offset, 0))
            par.axis["right"].toggle(all=True)
            val_loss = loss_dict[loss_keys[i+1]]
            val_losses[loss_keys[i+1]] = val_loss
            par.plot(x[start:end], transform_y(val_loss)[start:end], color=colors[i+1])
            par.set_ylabel("%s %s" % (ylabel_prefix, loss_keys[i+1]))
            par.axis["right"].label.set_color(colors[i+1])

            if maxes:
                loss_mins[loss_keys[i+1]] = ((start+np.argmax(val_loss[start:end])+(1-int(validate_first)))*save_valid_freq, np.max(val_loss[start:end]))
            else:
                loss_mins[loss_keys[i+1]] = ((start+np.argmin(val_loss[start:end])+(1-int(validate_first)))*save_valid_freq, np.min(val_loss[start:end]))

    plt.draw()
    plt.title('val losses for ' + exp_id)
    plt.show()

    if get_losses:
        return val_losses
    else:
        return loss_mins

def get_val_results_from_db(conn, dbname, collname, load_exp_id, suffix, group='val', step=None, metrics=['object_metrics'], python=3):
    exp_id = load_exp_id + '_' + group + '_' + suffix + '_' + ('last' if step is None else str(step))
    raw_data = multi_load_from_gridfs(conn, dbname, collname, exp_id, expid_prefix=None, load_step=None, datasets=metrics, python=python)
    return raw_data

def get_results_from_db(conn, dbname, collname, exp_id, expid_prefix, load_step, datasets, group='val', return_raw=False, python=3):
    exp_id += group
    raw_data = multi_load_from_gridfs(conn, dbname, collname, exp_id, expid_prefix, load_step, datasets=datasets, python=python)
    if return_raw:
        return raw_data
    else:
        combine_data = combine_val_files(raw_data, expid_prefix, datasets)
        return combine_data

def multi_load_from_gridfs(conn, dbname='test_phys', collname='po_net', exp_id='run2', expid_prefix='eval_part',
                           load_step=190000, datasets=[], python=3):
#     print('Loading results for file ' + str(file))
    # expidstr = expid_prefix+exp_id+'_'+str(load_step)
    expidstr = exp_id
    if expid_prefix is not None:
        expidstr = expid_prefix + expidstr
    if load_step is not None:
        expidstr += '_' + str(load_step)
    print(expidstr)
    r = conn[dbname][collname+'.files'].find_one({'exp_id': expidstr})
    _id = r['_id']
    fn = str(_id) + '_fileitems'
    fsys = gridfs.GridFS(conn[dbname], collname)
    fh = fsys.get_last_version(fn)
    fstr = fh.read()
    fh.close()

    if python==3:
        obj = cPickle.loads(fstr, encoding='latin1')
    elif python==2:
        obj = cPickle.loads(fstr)
    targets = {}
    for d in datasets:
        try:
            targets[expid_prefix+'_'+d] = obj['validation_results'][expid_prefix+'_'+d]
        except KeyError:
            print(obj['validation_results'].keys())
            targets[expid_prefix+'_'+d] = obj['validation_results'][d]
        except TypeError:
            targets[d] = obj['validation_results'][d]
    return targets

def velocities_to_optical_flow(vels, max_speed=1.0, to_rgb=True):
    speed = np.sqrt(np.square(vels[...,:2]).sum(axis=-1, keepdims=True))
    angle = np.arctan2(vels[...,1:2], vels[...,0:1])

    # h = 0.5 * (angle / np.pi) + 0.5
    h = (0.5*(angle / np.pi)) % 1.0
    s = np.minimum(speed / max_speed, 1.0)
    v = np.ones_like(s).astype(float)

    hsv = np.concatenate([h,v,s], axis=-1)
    if to_rgb:
        rgb = matplotlib.colors.hsv_to_rgb(hsv)
        return rgb
    else:
        return hsv

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def pred_hsv_to_rgb(hsv):
    h = hsv[...,0:1]
    sv = hsv[...,1:3]
    h = np.minimum(np.maximum(0.,h),1.)
    sv = sigmoid(sv)
    _hsv = np.concatenate([h,sv], axis=-1)
    rgb = matplotlib.colors.hsv_to_rgb(_hsv)
    return rgb

def hw_to_xy(hw):
    return np.stack([hw[...,1], -hw[...,0]], axis=-1)

def feature_map_from_segments(features, segment_ids):
    '''
    features: [N,D]
    segment_ids: [H,W] of indices into N

    returns:
    feature_map: [H,W,D] where each position is "colored" by the value features[segment_ids[h,w],:]
    '''
    if len(features.shape) == 1:
        features = features[...,np.newaxis]
    H,W = segment_ids.shape
    N,D = features.shape
    valid_ids = np.unique(segment_ids)
    offset = valid_ids.min()
    valid_ids = list([v - offset for v in valid_ids if ((v - offset) < N)])
    feature_map = np.zeros([H,W,D]).astype(float)

    for idx in valid_ids:
        inds = np.where(segment_ids == (idx + offset))
        feature_map[inds[0],inds[1],:] = features[idx,:]

    return feature_map if D > 1 else feature_map[...,0]

def agg_features_from_segments(feature_map, segment_ids, max_nodes=128, out_map=True, agg_func=np.mean):
    '''
    feature_map : [Him,Wim,C]
    segment_ids: [H,W]
    '''
    H,W = segment_ids.shape
    Him,Wim,C = feature_map.shape
    if Him != H or Wim != W:
        feats = skimage.transform.resize(feature_map.astype(float), [H,W], order=1).astype(float)
    else:
        feats = feature_map.astype(float)

    valid_ids = np.unique(segment_ids)
    offset = valid_ids.min()
    valid_ids = list([v - offset for v in valid_ids])

    if out_map:
        out_map = np.zeros([H,W,C]).astype(float)
        for idx in valid_ids:
            inds = np.where(segment_ids == (idx + offset))
            out_map[inds[0],inds[1],:] = agg_func(feats[inds[0],inds[1],:], axis=0)
        # out_map = skimage.transform.resize(out_map.astype(float), [Him,Wim], order=1).astype(float) / 255.
        return out_map
    else:
        out_nodes = np.zeros([max_nodes,C])
        for idx in valid_ids:
            inds = np.where(segment_ids == (idx + offset))
            out_nodes[idx,:] = agg_func(feats[inds[0],inds[1],:], axis=0)
        return out_nodes

def preproc_segment_ids(segment_ids):

    seg_mins = segment_ids.min(axis=(-2,-1), keepdims=True)
    return segment_ids - seg_mins

def segs_to_one_hot(segment_ids, Nmax=16):
    segs = preproc_segment_ids(segment_ids) # [...,H,W] starting at 0
    dtype = segs.dtype
    shape = segs.shape
    segs = (segs[...,np.newaxis] == np.arange(Nmax).reshape([1]*len(shape) + [-1])).astype(np.float32)
    return segs

def preproc_flows(flows, backward=False, scale=True, xy_flows=True, stop_gradient=True, scale_factor=0.5):
    if stop_gradient:
        flows = tf.stop_gradient(flows)
    if backward:
        flows = -flows
    if xy_flows:
        flows = tf.stack([-flows[...,1], flows[...,0]], axis=-1)
    if scale:
        H,W = flows.shape.as_list()[-3:-1]
        flows *= tf.reshape(tf.constant([H,W], tf.float32), [1]*(len(flows.shape)-1) + [2]) * scale_factor

    return flows

def get_image_values_from_indices(images, particles_im_indices):
    B,N,_ = particles_im_indices.shape.as_list()
    _,H,W,Dim = images.shape.as_list()

    # get the images values at particles_im_indices -- output is [B,T,N,Dim]
    ones = tf.ones([B,N], dtype=tf.int32)
    inds_b = tf.reshape(tf.range(B, dtype=tf.int32), [B,1]) * ones
    inds_h = particles_im_indices[...,0] # [B,T,N]
    inds_w = particles_im_indices[...,1] # [B,T,N]
    gather_inds = tf.stack([inds_b, inds_h, inds_w], axis=-1)

    image_values = tf.gather_nd(images, gather_inds) # [B,N,Dim]

    return image_values

def warp_image(im, flows):
    B,H,W,C = im.shape.as_list()
    assert flows.shape.as_list()[-1] == 2

    ones = tf.ones([B,H,W,1], dtype=tf.float32)
    h_ims = tf.reshape(tf.range(H, dtype=tf.float32), [1,-1,1,1]) * ones
    w_ims = tf.reshape(tf.range(W, dtype=tf.float32), [1,1,-1,1]) * ones
    base_inds = tf.concat([h_ims, w_ims], axis=-1) # [B,H,W,2]

    pred_inds = base_inds + flows
    pred_inds = tf.reshape(pred_inds, [B,H*W,2])
    pred_h, pred_w = tf.split(pred_inds, [1,1], axis=-1)
    pred_h = tf.maximum(tf.minimum(pred_h, tf.cast(H-1, tf.float32)), 0.)
    pred_w = tf.maximum(tf.minimum(pred_w, tf.cast(W-1, tf.float32)), 0.)

    floor_h = tf.cast(tf.floor(pred_h), tf.float32)
    floor_w = tf.cast(tf.floor(pred_w), tf.float32)
    ceil_h = tf.cast(tf.ceil(pred_h), tf.float32)
    ceil_w = tf.cast(tf.ceil(pred_w), tf.float32)

    bot_right_weight = (pred_h - floor_h) * (pred_w - floor_w)
    bot_left_weight = (pred_h - floor_h) * (ceil_w - pred_w)
    top_right_weight = (ceil_h - pred_h) * (pred_w - floor_w)
    top_left_weight = (ceil_h - pred_h) * (ceil_w - pred_w)

    _to_int = lambda x: tf.cast(x, tf.int32)
    top_left_vals = get_image_values_from_indices(
        im, _to_int(tf.concat([floor_h, floor_w], -1)))
    top_right_vals = get_image_values_from_indices(
        im, _to_int(tf.concat([floor_h, ceil_w], -1)))
    bot_left_vals = get_image_values_from_indices(
        im, _to_int(tf.concat([ceil_h, floor_w], -1)))
    bot_right_vals = get_image_values_from_indices(
        im, _to_int(tf.concat([ceil_h, ceil_w], -1)))

    im_vals = top_left_vals * top_left_weight +\
              top_right_vals * top_right_weight +\
              bot_left_vals * bot_left_weight +\
              bot_right_vals * bot_right_weight

    im_vals = tf.reshape(im_vals, [B,H,W,-1])

    return im_vals

def forward_warp_image(im, flows, stop_gradient=True):
    '''
    Scatter points in an image forward by flows according to bilinear interp
    '''

    reshape = False
    if len(im.shape) == 5: # merge time into batch dim
        reshape = True
        _B,T,H,W,C = im.shape.as_list()
        B = _B*T
        im = tf.reshape(im, [B,H,W,C])
        flows = tf.reshape(flows, [B,H,W,2])
    else:
        B,H,W,C = im.shape.as_list()
    assert flows.shape.as_list()[-1] == 2

    if stop_gradient:
        im = tf.stop_gradient(im)

    ones = tf.ones([B,H,W,1], dtype=tf.float32)
    h_ims = tf.reshape(tf.range(H, dtype=tf.float32), [1,-1,1,1]) * ones
    w_ims = tf.reshape(tf.range(W, dtype=tf.float32), [1,1,-1,1]) * ones
    base_inds = tf.concat([h_ims, w_ims], axis=-1) # [B,H,W,2]

    pred_inds = base_inds + flows
    pred_inds = tf.reshape(pred_inds, [B,H*W,2])
    pred_h, pred_w = tf.split(pred_inds, [1,1], axis=-1)
    pred_h = tf.maximum(tf.minimum(pred_h, tf.cast(H-1, tf.float32)), 0.)
    pred_w = tf.maximum(tf.minimum(pred_w, tf.cast(W-1, tf.float32)), 0.)

    floor_h = tf.cast(tf.floor(pred_h), tf.float32)
    floor_w = tf.cast(tf.floor(pred_w), tf.float32)
    ceil_h = tf.cast(tf.ceil(pred_h), tf.float32)
    ceil_w = tf.cast(tf.ceil(pred_w), tf.float32)

    bot_right_weight = (pred_h - floor_h) * (pred_w - floor_w)
    bot_left_weight = (pred_h - floor_h) * (ceil_w - pred_w)
    top_right_weight = (ceil_h - pred_h) * (pred_w - floor_w)
    top_left_weight = (ceil_h - pred_h) * (ceil_w - pred_w)

    N = H*W
    ones = tf.ones([B,N,1], dtype=tf.int32)
    inds_b = tf.reshape(tf.range(B, dtype=tf.int32), [B,1,1]) * ones
    _to_int = lambda x: tf.cast(x, tf.int32)
    def _scatter(inds_h, inds_w):
        sc_inds = tf.concat([
            inds_b, _to_int(inds_h), _to_int(inds_w)], axis=-1)
        im_vals = tf.scatter_nd(
            updates=tf.reshape(im, [B,N,C]),
            indices=sc_inds,
            shape=[B,H,W,C])
        im_vals = tf.reshape(im_vals, [B,N,C])
        return im_vals

    image_values = top_left_weight * _scatter(floor_h, floor_w) +\
                   top_right_weight * _scatter(floor_h, ceil_w) +\
                   bot_left_weight * _scatter(ceil_h, floor_w) +\
                   bot_right_weight * _scatter(ceil_h, ceil_w)

    image_values = tf.reshape(image_values, [B,H,W,C])

    if reshape:
        image_values = tf.reshape(image_values, [_B,T,H,W,C])

    return image_values

def compute_occlusion_map(forward_flows, backward_flows, scale=True, xy_flows=True, stop_gradient=True, scale_factor=0.5):
    '''
    '''
    B,T,H,W,_ = forward_flows.shape.as_list()

    forward_flows = tf.reshape(forward_flows, [B*T,H,W,2])
    backward_flows = tf.reshape(backward_flows, [B*T,H,W,2])

    forward_flows = preproc_flows(forward_flows, backward=True, scale=scale, xy_flows=xy_flows, stop_gradient=stop_gradient, scale_factor=scale_factor)
    backward_flows = preproc_flows(backward_flows, backward=True, scale=scale, xy_flows=xy_flows, stop_gradient=stop_gradient, scale_factor=scale_factor)

    ones = tf.ones([B*T,H,W,1], dtype=tf.float32)
    forward_occlusions = tf.minimum(forward_warp_image(im=ones, flows=forward_flows), 1.)
    backward_occlusions = tf.minimum(forward_warp_image(im=ones, flows=backward_flows), 1.)

    forward_occlusions = tf.reshape(forward_occlusions, [B,T,H,W,1])
    backward_occlusions = tf.reshape(backward_occlusions, [B,T,H,W,1])

    return forward_occlusions, backward_occlusions

def coordinate_ims(batch_size, seq_len, imsize):
    bs = batch_size
    T = seq_len
    H,W = imsize
    ones = tf.ones([bs,H,W,1], dtype=tf.float32)
    h = tf.reshape(tf.divide(tf.range(H, dtype=tf.float32), tf.cast(H-1, dtype=tf.float32) / 2.0),
                   [1,H,1,1]) * ones
    h -= 1.0
    w = tf.reshape(tf.divide(tf.range(W, dtype=tf.float32), tf.cast(W-1, dtype=tf.float32) / 2.0),
                   [1,1,W,1]) * ones
    w -= 1.0
    h = tf.stack([h]*T, axis=1)
    w = tf.stack([w]*T, axis=1)
    hw_ims = tf.concat([h,w], axis=-1)
    return hw_ims

def compute_occlusion_directions(occlusions, ksize=[5,5]):

    B,T,H,W,_ = occlusions.shape.as_list()
    kernel = coordinate_ims(1, 1, ksize)[0,0,:,:,tf.newaxis,:]
    kernel *= tf.reshape(tf.constant([
        (ksize[0] - 1.) / 2.,
        (ksize[1] - 1.) / 2.], tf.float32), [1,1,1,2])
    occlusions = tf.reshape(occlusions, [B*T,H,W,1])
    dirs = tf.nn.conv2d(occlusions, kernel, strides=[1,1,1,1], padding='SAME')
    xy_dirs = tf.stack([dirs[...,1], -dirs[...,0]], axis=-1)
    angles = tf.math.atan2(xy_dirs[...,1], xy_dirs[...,0])

    xy_dirs = tf.reshape(xy_dirs, [B,T,H,W,2])

    return xy_dirs

def propagate_index_map(segment_ids, flows, valid_nodes, **kwargs):
    '''
    Predict where next indices in segment_ids end up according to flow predictions
    '''
    B,T,H,W = segment_ids.shape.as_list()
    assert flows.shape.as_list() == [B,T,H,W,2]
    _,_,N,_ = valid_nodes.shape.as_list()
    flows = preproc_flows(flows, **kwargs)
    index_map = tf.one_hot(segment_ids, depth=N, axis=-1, dtype=tf.float32) # [B,T,H,W,N]
    pred_index_map = forward_warp_image(im=index_map[:,0:-1], flows=flows[:,0:-1]) # [B,T-1,H,W,N]
    pred_index_map *= valid_nodes[:,0:-1,tf.newaxis,tf.newaxis,:,0]
    contested_pixels = tf.cast(tf.reduce_sum(pred_index_map, axis=-1, keepdims=True) > 1.0, tf.float32)
    mask = tf.cast(pred_index_map > 0., tf.float32) # [B,T-1,H,W,N]

    return pred_index_map, contested_pixels, mask # [B,T-1,H,W,N], [B,T-1,H,W,1], [B,T-1,H,W,N]

def resolve_depth_order(index_masks, node_depths, valid_nodes, min_depth=-30., beta=1.0, **kwargs):
    '''
    '''
    B,T,H,W,N = index_masks.shape.as_list()
    _,_T,_N,_ = node_depths.shape.as_list()
    assert _N == N, (_N, N)
    if _T == T + 1:
        node_depths = node_depths[:,0:-1]
        valid_nodes = valid_nodes[:,0:-1]
    else:
        assert _T == T, (_T,T)

    ## more positive (less negative) is closer
    valids = valid_nodes[:,:,tf.newaxis,tf.newaxis,:,0]
    depths = index_masks * node_depths[:,:,tf.newaxis,tf.newaxis,:,0] # [B,T,H,W,N]
    depths += (1. - index_masks) * min_depth
    depths = valids * depths + (1.- valids) * min_depth

    depth_weights = tf.nn.softmax(depths * beta, axis=-1)

    return depth_weights

def render_attrs_from_segment_weights(weights, nodes, dims_list=[[43,46]]):
    '''
    '''
    B,T,H,W,N = weights.shape.as_list()
    _,_T,_N,D = nodes.shape.as_list()
    assert _N == N, (_N,N)
    if _T == T + 1:
        nodes = nodes[:,:-1]

    attrs = tf.concat([nodes[...,d[0]:d[1]] for d in dims_list], -1)
    attr_image = attrs[:,:,tf.newaxis,tf.newaxis] * weights[...,tf.newaxis]
    attr_image = tf.reduce_sum(attr_image, axis=-2) # [B,T,H,W,Dattr]
    return attr_image
