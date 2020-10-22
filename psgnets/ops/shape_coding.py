import numpy as np
import tensorflow as tf

PRINT = False

def sharpness(x, clip_val=20.0):
    '''
    x in (-inf, inf): the larger x, the higher y in (0, inf)
    '''
    x = tf.nn.sigmoid(x)
    y = x / (1.0 - tf.maximum(x, 1./clip_val))
    return y

def vec_to_rtheta(vec, eps=1e-6):
    assert vec.shape.as_list()[-1] == 2
    vx, vy = tf.split(vec, 2, axis=-1)
    voff = tf.sqrt(vx**2 + vy**2 + eps)
    theta = tf.atan2(vy, vx + eps)
    return voff, theta

def build_sdfs_from_codes(codes, xy_input=False, **kwargs):
    '''
    codes: [B,T,N,C,4] 4 parameters each for encoding one of C linear/quadratic signed distance functions per node
                       The values in each code vector are (voff, angle, curvature, hoff); Setting the last 2 to zero gives a linear sdf

    sdf: a quadratic map from [H,W,2] xy coordinates to [B,T,N,C,H,W] images
    '''
    assert codes.shape.as_list()[-1] == 4
    shape = codes.shape.as_list()
    vec, curv, hoff = tf.split(codes, [2,1,1], axis=-1)
    if PRINT:
        vec = tf.Print(vec, [
            tf.reduce_min(vec[:,:,0]), tf.reduce_max(vec[:,:,0]),
            tf.reduce_min(curv[:,:,0]), tf.reduce_max(curv[:,:,0]),
            tf.reduce_min(hoff[:,:,0]), tf.reduce_max(hoff[:,:,0])
        ], message='code_minmax')
    if xy_input:
        voff, theta = vec_to_rtheta(vec)
    else:
        voff, theta = tf.split(vec, [1,1], axis=-1)

    # takes in points of shape [H,W,2]
    def sdf(points, trans=None, rots=None, scales=None):
        H,W = points.shape.as_list()[0:2]
        points = tf.reshape(points, [-1, 2]) # [HW,2]
        points = transform_points(points, trans, rots, scales) # shape[:-1] + [2,HW]
        x, y = tf.split(points, [1,1], axis=-2) # shape[:-1] + [HW]
        if x.shape.as_list()[:-2] != shape[:-2]:
            print("reshape", x.shape.as_list(), shape)
            x = tf.reshape(x, shape[:-1] + [-1])
            y = tf.reshape(y, shape[:-1] + [-1])
        # if x.shape.as_list()[:-2] != shape[:-2]:
        #     print("reshaping points!")
        #     x = tf.reshape(x, [1]*len(shape[:-1]) + [-1])
        #     y = tf.reshape(y, [1]*len(shape[:-1]) + [-1])
        val = -(x*tf.cos(theta) + y*tf.sin(theta) - voff)
        val += curv * tf.square((-x*tf.sin(theta) + y*tf.cos(theta) - hoff))
        val = tf.reshape(val, shape[:-1] + [H,W])
        return val

    return sdf

def get_test_points(imsize):
    H,W = imsize
    xs = tf.reshape(tf.range(W, dtype=tf.float32), [1,W]) - (float(W-1)/2.)
    ys = tf.reshape(tf.range(H, dtype=tf.float32), [H,1]) - (float(H-1)/2.)
    points = tf.stack([
        tf.tile(xs, [H,1]),
        -tf.tile(ys, [1,W])
    ], axis=-1)
    return points

def get_hw_grid(imsize):
    H,W = imsize
    hs = tf.tile(tf.reshape(tf.range(H, dtype=tf.float32), [H,1]), [1,W])
    hs = (hs / (float(H-1.0)/2.0)) - 1.0 # in [-1.,1.]
    ws = tf.tile(tf.reshape(tf.range(W, dtype=tf.float32), [1,W]), [H,1])
    ws = (ws / (float(W-1.0)/2.0)) - 1.0
    hw_grid = tf.stack([hs, ws], axis=-1) # [H,W,2]
    return hw_grid

def transform_points(points, trans, rots, scales):
    '''
    points: [N,2]
    trans: [...,2]
    rots: [...,2,2]
    scales: [...,1] or [...,2]
    '''
    if trans is not None:
        shape = trans.shape.as_list()[:-1]
    else:
        shape = []

    points = tf.transpose(points, [1,0]) # [2,N]
    points = tf.reshape(points, [1]*len(shape) + [2,-1]) # [1,...,1,2,N]
    if trans is not None:
        points = points - trans[...,tf.newaxis] # [...,2,N]
    if rots is not None:
        points = tf.matmul(rots, points) # [...,2,N]
    if scales is not None:
        points = points * scales[...,tf.newaxis] # [...,2,N]
    return points

def rotations_from_angles(angles):
    '''
    Converts angles [...,1] in [-pi, pi] to rotation matrices
    '''
    shape = angles.shape.as_list()[:-1]
    a = tf.cos(angles)
    b = tf.sin(angles)
    rotmat = tf.concat([a, b, -b, a], axis=-1)
    rotmat = tf.reshape(rotmat, shape + [2,2])
    return rotmat

def build_shape_from_codes(codes, translations=None, rotations=None, scales=None, imsize=[64,64], xy_input=False, beta=1.0, sigma=1.0, sigmoid_first=True, **kwargs):
    '''
    codes: shape + [C,4]
    translations: shape + [2]
    rotations: shape + [1] or shape + [2,2]
    scales: shape + [1]
    '''

    assert codes.shape.as_list()[-1] == 4
    sdfs = build_sdfs_from_codes(codes, xy_input, **kwargs)
    points = get_test_points(imsize)

    if rotations is not None:
        if rotations.shape.as_list()[-1] == 1:
            rotations = rotations_from_angles(rotations)
    vals = sdfs(points, translations, rotations, scales)

    # reduce across constraints
    if sigmoid_first:
        constraints = tf.nn.sigmoid(sigma * vals)
        shapes = tf.reduce_min(constraints, axis=-3)
    else:
        constraints = vals
        shapes = tf.reduce_min(constraints, axis=-3)
        shapes = tf.nn.sigmoid(sigma * shapes)

    if PRINT:
        shapes = tf.Print(shapes, [tf.reduce_max(constraints), tf.reduce_max(shapes)], message='cons_shapes_max')

    return constraints, shapes

def build_edges_from_codes(codes, length_scales, translations, xy_input=True, imsize=[64,64], scale_xy=True, edge_sharpness=2.5, edge_strength=1.0, **kwargs):

    assert length_scales.shape.as_list()[:-1] == translations.shape.as_list()[:-1], (length_scales.shape.as_list(), translations.shape.as_list())
    assert length_scales.shape.as_list()[-1] in [1,2], "Must provide a length scale for the envelope or length_x, length_y"
    if scale_xy:
        xy_scale = (tf.constant(imsize, tf.float32) - 1.0)/2.0
        length_scales = length_scales * xy_scale
        translations = translations * xy_scale
        r = tf.sqrt(xy_scale[0]**2 + xy_scale[1]**2)
        if xy_input:
            codes = tf.concat([codes[...,0:2]*xy_scale, codes[...,2:3], codes[...,3:4]*(r/np.sqrt(2.))], axis=-1)
        else:
            codes = tf.concat([codes[...,0:1]*r, codes[...,1:3], codes[...,3:4]*(r/np.sqrt(2.))], axis=-1)

    sharp = sharpness(tf.constant(edge_sharpness, tf.float32))
    constraints, _ = build_shape_from_codes(codes, translations=translations, xy_input=xy_input, imsize=imsize, **kwargs)
    lines = tf.constant(edge_strength, tf.float32) * tf.exp(-sharp * tf.square(constraints - 0.5))

    points = tf.reshape(get_test_points(imsize), [1]*len(translations.shape.as_list()[:-1]) + imsize + [2])
    # print("points, trans, lengths", points.shape.as_list(), translations.shape.as_list(), length_scales.shape.as_list())
    envelope = tf.exp(-tf.reduce_sum(tf.square((translations[...,tf.newaxis,tf.newaxis,:] - points) / length_scales[...,tf.newaxis,tf.newaxis,:]), axis=-1))
    # print("lines, env", lines.shape.as_list(), envelope.shape.as_list())
    lines = tf.reduce_max(lines * tf.expand_dims(envelope, axis=-3), axis=-3)
    return lines

def segments_to_segment_edges(segments, num_segments=512, thresh=None):
    if not isinstance(segments, tf.Tensor):
        segments = tf.convert_to_tensor(segments, tf.int32)
    shape = segments.shape.as_list()
    if len(shape) == 3:
        B,H,W = shape
        T = 1
    else:
        assert len(shape) == 4 and shape[-1] != 1, "segments must have shape [B,T,H,W] but is %s" % shape
        B,T,H,W = shape
        segments = tf.reshape(segments, [B*T,H,W])

    # convert to one-hot and edge filter
    segments = tf.one_hot(segments, axis=-1, depth=num_segments, dtype=tf.float32)
    edges = tf.image.sobel_edges(segments)
    edges = tf.reduce_sum(tf.square(edges), axis=-1)

    if thresh is not None:
        edges = tf.cast(edges > thresh, tf.float32)

    if len(shape) == 4:
        edges = tf.reshape(edges, [B,T,H,W,-1])

    return edges

if __name__ == '__main__':

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sess = tf.Session()

    # code = tf.zeros([1,4], dtype=tf.float32)
    code = tf.constant([[0., 0., 0.1, 0.]], tf.float32)
    translations = tf.zeros([2], dtype=tf.float32)
    cons, shape = build_shape_from_codes(code, translations, imsize=[8,8], xy_input=True, sigma=0.1)
    print(shape)
    print(sess.run(shape))
