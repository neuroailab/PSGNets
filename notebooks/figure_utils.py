import numpy as np
import scipy as sp
import pickle
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.cm as cm
import matplotlib.animation as animation
from matplotlib import rcParams
from matplotlib.backends.backend_pdf import PdfPages
import itertools
from IPython.display import HTML

import utils

def permute_labels(labels, seed=0):
    rng = np.random.RandomState(seed=seed)
    ids = np.unique(labels)
    new_ids = rng.permutation(ids)
    new_labels = np.zeros_like(labels)
    for i,_id in enumerate(ids):
        new_labels[labels == _id] = new_ids[i]
    return new_labels

def align_segmentation_movie(segments, trajectories, max_segs=12., cmap=matplotlib.cm.jet):
    '''
    segments: [T,H,W]
    trajectory_centroids: [T,N,2] in range [-1, 1.]
    '''
    T,H,W = segments.shape
    T2,N,D = trajectories.shape
    assert T==T2

    traj_hws = (trajectories[...,-4:-2] + 1.) / 2. # now in [0,1.]
    traj_hws *= np.array([H,W]).reshape([1,1,2]).astype(float)
    traj_hws = np.minimum(np.maximum(0, np.round(traj_hws).astype(int)), np.array([H-1,W-1]).reshape([1,1,2]))

    seg_movie = []
    for t in range(T):
        seg_img = np.zeros([H,W]).astype(float)
        valid_ids = np.where(trajectories[t,:,-1] > 0.5)[0]
        for i,n in enumerate(valid_ids):
            centroid_inds = traj_hws[t,n] # [h,w]
            centroid_seg = segments[t,centroid_inds[0],centroid_inds[1]]
            seg_img[np.where(segments[t] == centroid_seg)] = ((i+1) / max_segs)
        seg_movie.append(seg_img)
    seg_movie = np.stack(seg_movie, axis=0) # [T,H,W]
    seg_movie = cmap(seg_movie) # [T,H,W,3]
    return seg_movie

def map_edges_from_node(bool_edges, node, segment_ids):
    connected_edges = bool_edges[node,:,np.newaxis].astype(float) # [N,1]
    connected_edges = np.concatenate([connected_edges]*3, axis=-1)
    connected_edges *= np.array([1.,1.,0.]).reshape(1,3) # [N,3] yellow
    connected_edges[node,:] = np.array([1.,0.,0.]) # red
    edge_map = utils.feature_map_from_segments(connected_edges, segment_ids)
    return edge_map

def map_errors_from_node(errors, node, segment_ids):
    connected_errors = errors[node,:,np.newaxis]
    connected_errors[node] = 0.0
    error_map = utils.feature_map_from_segments(connected_errors, segment_ids)
    return error_map

def plot_edges_from_node(data, ex, t=-1, node=0, pixel=None, level='level2', seg_level=None, node_key='child_nodes', err_key='affinities', edge_key='within_edges', edge_thresh=None, max_err=1., stride=2, fname=None, attr_dims=[43,46], use_sigmoid=True):

    fig, axes = plt.subplots(1,4, sharey=False, figsize=(16,4))
    seg_key = 'parent_segment_ids'
    if seg_level is not None:
        seg_key = seg_level + '/' + seg_key
    else:
        seg_key = level[:-1] + str(int(level[-1]) - 1) + '/' + seg_key
    seg_ids = data[seg_key][ex,t]

    err_key = level + '/' + err_key
    func = utils.sigmoid if use_sigmoid else lambda x:x
    if edge_thresh is not None:
        bool_edges = func(data[err_key][ex,t]) > edge_thresh
    else:
        edge_key = level + '/' + edge_key
        bool_edges = data[edge_key][ex,t]

    if node is None:
        assert pixel is not None
        node = seg_ids[pixel[0], pixel[1]] - seg_ids.min()

    edge_map = map_edges_from_node(bool_edges, node, seg_ids)
    error_map = map_errors_from_node(func(data[err_key][ex,t]), node, seg_ids)
    error_map[np.where(seg_ids - seg_ids.min() == node)] = max_err
    error_mask = (error_map < max_err).astype(float)
    error_map *= error_mask
    error_map += (1. - error_mask) * max_err

    nodes = data[level + '/child_nodes']
    pred_hsv = nodes[ex,t,:,attr_dims[0]:attr_dims[1]]
    pred_rgb_map = utils.feature_map_from_segments(
        STANDARD_ATTR_FUNCS['pred_images'](pred_hsv),
        seg_ids)

    pred_rgb_map[np.where(seg_ids - seg_ids.min() == node)] = np.array([1.,0.,0.])

    axes[1].imshow(pred_rgb_map)
    axes[1].set_title('pred rgb of nodes')
    axes[2].imshow(edge_map)
    axes[2].set_title('nodes connected to red node %d' % node)
    axes[3].imshow(error_map, vmin=0., vmax=max_err, cmap='magma')
    axes[3].set_title('affinities with node %d' % node)

    axes[0].imshow(data['inputs/images'][ex,t][::stride,::stride] / 255.)
    axes[0].set_title('input image ex=%d, t=%d' % (ex,t))

    for i in range(1,4):
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        if fname is not None:
            axes[i].set_title('')

    if fname is not None:
        plt.savefig(fname, bbox_inches='tight')

    plt.show()

def make_video_from_images(
        images,
        sta_idx=0, end_idx=-1,
        resize=None,
        *args,
        **kwargs
        ):
    image_array = images[sta_idx : end_idx]

    dpi = 300.0
    if resize is not None:
        new_array = []
        for t in range(image_array.shape[0]):
            new_array.append(skimage.transform.resize(image_array[t], [resize]*2, order=1))
        image_array = np.stack(new_array, axis=0)
    xpixels, ypixels = image_array[0].shape[0], image_array[0].shape[1]
    fig = plt.figure(figsize=(ypixels/dpi, xpixels/dpi), dpi=dpi)
    im = plt.figimage(image_array[0])

    def animate(i):
        im.set_array(image_array[i])
        return (im,)
    anim = animation.FuncAnimation(
            fig, animate,
            frames=len(image_array), interval=200)
    return anim.to_html5_video()

def scale_depth(x, min_depth=100.0, max_depth=100., **kwargs):
    return np.maximum(np.minimum(x, min_depth), -max_depth)

STANDARD_ATTR_FUNCS = {
    'valid': lambda im: im,
    'pred_images': utils.pred_hsv_to_rgb,
    'pred_delta_images': utils.sigmoid,
    'pred_flows': utils.velocities_to_optical_flow,
    'pred_back_flows': utils.velocities_to_optical_flow,
    'pred_depths': scale_depth,
    'pred_occlusion_depths': lambda x: x,
    'pred_normals': lambda n: n / np.sqrt(np.sum(n**2, axis=-1, keepdims=True))
}

def plot_pred_attrs(data, ex, t, level='level1', child_nodes=True, attr_name='valid', attr_dims=None, attr_func=None, attr_func_kwargs={}, level_suffix='',
                    cmap='magma', plot_gt_segments=False, segments_key='objects', seed=0, get_attrs=False, key_pos=0):

    # get data
    segs = data[level + '/parent_segment_ids'][ex,t]
    size = segs.shape
    images = data['inputs/images'][ex] / 255.
    if plot_gt_segments:
        dimages = data['inputs/'+segments_key][ex,t]
    else:
        try:
            dimages = data['inputs/delta_images'][ex,t,...,0]
        except KeyError:
            dimages = np.abs(images[1:] - images[:-1]).mean(axis=-1) # [t-1,him,wim]
            dimages = dimages[t-1]
    images = images[t]
    imsize = images.shape[:2]
    strides = [imsize[0] // size[0], imsize[1] // size[1]]
    images = images[::strides[0],::strides[1]]
    dimages = dimages[::strides[0],::strides[1]]

    nkey = (level + '/parent_nodes') if not child_nodes else (level[:-1] + str(int(level[-1])+1) + level_suffix + '/child_nodes')
    nodes = data[nkey][ex,t]

    # use dims to find attributes
    dims = data['dims/'+level+'_dims']
    if attr_name is not None:
        try:
            attr_dims = dims[attr_name]
        except KeyError:
            try:
                attr_dims = dims[attr_name + '_' + level]
            except KeyError:
                attr_dims = dims[[k for k in dims.keys() if attr_name in k][key_pos]]
    else:
        assert attr_dims is not None
    attrs = nodes[...,attr_dims[0]:attr_dims[1]]
    if attr_func is None:
        attr_func = STANDARD_ATTR_FUNCS.get(attr_name, lambda x: x)
    try:
        attrs = attr_func(attrs, **attr_func_kwargs)
    except TypeError:
        attrs = attr_func(attrs)
    attrs_im = utils.feature_map_from_segments(attrs, segs)
    colors_im = utils.agg_features_from_segments(images, segs, max_nodes=512, out_map=True)

    if get_attrs:
        return attrs_im

    # plot
    titles = ['image', ('gt segments' if plot_gt_segments else 'delta images'), level+' segs', 'image colored segs',\
              '%s colored segs' % (attr_name or 'dims%s-%s' % (attr_dims[0], attr_dims[1]))]
    n_plots = len(titles)

    fig, axes = plt.subplots(1, n_plots, sharey=True, figsize=(4*n_plots, 4))
    axes[0].imshow(images)
    axes[1].imshow(dimages)
    axes[2].imshow(permute_labels(segs, seed=seed), cmap='Spectral')
    axes[3].imshow(colors_im)
    try:
        axes[4].imshow(attrs_im, cmap=cmap)
    except TypeError:
        axes[4].imshow(attrs_im[...,0], cmap=cmap)

    for i,ax in enumerate(axes):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(titles[i], fontsize=16)
    axes[0].set_ylabel('ex=%s, t=%s' % (ex,t), fontsize=16)
    plt.show()

    return attrs_im
