import tensorflow.compat.v1 as tf
import numpy as np

from psgnets.ops.dimensions import DimensionDict, OrderedDict
from psgnets.data.utils import read_depths_image
from .base import Model

def preproc_rgb(img, norm=255., to_hsv=False):
    if img.dtype == tf.uint8:
        img = tf.cast(img, tf.float32) / norm
    if to_hsv:
        img = tf.image.rgb_to_hsv(tf.clip_by_value(img, 0., 1.))
    return img

def preproc_hsv(img, circular=False):
    assert img.dtype == tf.float32, img.dtype
    h,sv = tf.split(img, [1,2], axis=-1)
    if circular:
        h = tf.floormod(h, tf.constant(1., tf.float32))
    sv = tf.nn.sigmoid(sv)
    return tf.concat([h,sv], axis=-1)

def preproc_depths(depth, normalization=100.1, background_depth=120., invert=True):
    sgn = -1. if invert else 1.
    return sgn * read_depths_image(depth, new=True, normalization=normalization, background_depth=background_depth)

def preproc_normals(normals):

    if normals.dtype == tf.uint8:
        normals = tf.cast(normals, tf.float32) / 255.
    else:
        assert normals.dtype == tf.float32

    return normals*2.0 - 1.0

def delta_rgb(images, preproc=preproc_rgb, **kwargs):

    assert len(images.shape) == 5, images
    images = preproc(images)
    deltas = images[:,1:] - images[:,:-1]
    deltas = tf.concat([
        tf.zeros_like(images[:,0:1]), deltas], axis=1)
    return deltas

def delta_images(images, preproc=preproc_rgb, thresh=0.03, **kwargs):
    '''
    Get a map of parts of the images that are changing
    '''
    assert len(images.shape) == 5, images
    if images.shape.as_list()[1] == 1:
        return tf.cast(tf.zeros_like(images)[...,0:1], tf.float32)

    images = preproc(images)
    intensities = tf.reduce_mean(images, axis=-1, keepdims=True)
    delta_images = tf.abs(intensities[:,1:] - intensities[:,:-1])
    delta_images = tf.concat([
        tf.zeros_like(delta_images[:,0:1]), delta_images], axis=1)
    if thresh is not None:
        delta_images = tf.cast(delta_images > thresh, tf.float32)

    return delta_images

def compute_sobel_features(ims, norm=255., size=None, normalize_range=False, to_mag=False, to_rgb=False, eps=1e-6):
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
    elif to_mag:
        edges_feats = tf.concat([edges_ang, edges_mag_sum], axis=-1)
    else:
        edges_feats = tf.concat([edges_x, edges_y, edges_mag, edges_ang, edges_mag_sum, edges_mag_sum], axis=-1) # [BT,H,W,3C+3]

    if size is not None:
        edges_feats = tf.image.resize_images(edges_feats, size)

    if len(shape) == 5:
        edges_feats = tf.reshape(edges_feats, [B,T,H,W,-1])

    return edges_feats

def concat_and_name_tensors(tensor_dict, tensor_order=[], dimensions=None, dimension_names={}, dimension_preprocs={}, **kwargs):

    assert isinstance(tensor_dict, dict)
    assert all([isinstance(v, tf.Tensor) for v in tensor_dict.values()]), "Must pass a dict of all tensors"
    if dimensions is None:
        dimensions = DimensionDict()

    out = []
    for nm in tensor_order:
        func = dimension_preprocs.get(nm, tf.identity)
        tensor = func(tensor_dict[nm])
        out.append(tensor)
        dimensions[dimension_names.get(nm, nm)] = tensor.shape.as_list()[-1]

    return tf.concat(out, axis=-1)

def preproc_tensors_by_name(
        tensor_dict,
        dimension_order=['images'],
        dimension_preprocs={'images': lambda rgb: tf.cast(rgb, tf.float32) / 255.},
        dimensions=None, dimension_names={}, **kwargs
):
    '''
    For each tensor in tensor_dict named by a key in tensor_order, apply associated preproc func.
    Then concat the results and update the DDict passed by dimensions

    '''
    tensor_names = [dimension_names.get(nm, nm) for nm in dimension_order]
    funcs = {nm: dimension_preprocs.get(nm, lambda data: tf.identity(tf.cast(data, tf.float32), name=nm+'_preproc'))
             for nm in tensor_names}

    if dimensions is None:
        dimensions = DimensionDict()

    ## create the tensor and dimensions
    preproc_tensor = concat_and_name_tensors(tensor_dict, dimension_order, dimensions, dimension_names, dimension_preprocs=funcs, **kwargs)

    ## update dim preprocs
    dimensions.set_postprocs(funcs)

    ## apply the postprocs
    # preproc_tensor = dimensions.get_tensor_from_attrs(
    #     preproc_tensor, tensor_names, postproc=False, stop_gradient=kwargs.get('stop_gradient', False))

    return preproc_tensor

class Preprocessor(Model):

    def __init__(
            self,
            name=None,
            model_func=preproc_tensors_by_name,
            **kwargs
    ):

        self.Dims = None
        super(Preprocessor, self).__init__(model_func, name=name, **kwargs)

    def __call__(self, inputs, train=True, **kwargs):
        assert isinstance(inputs, dict), "Must pass a dict of tensors to the preprocessor"
        assert all((isinstance(val, tf.Tensor) for val in inputs.values())), "Must pass a dict of tensors to preprocssor"

        self.Dims = DimensionDict()
        outputs = super(Preprocessor, self).__call__(
            inputs, dimensions=self.Dims, **kwargs)

        assert self.Dims.ndims == outputs.shape.as_list()[-1],\
            "All dims must be logged in Preprocessor.dims but outputs shape = %s and dims = %s"\
            % (outputs.shape.as_list(), self.Dims)

        return outputs
