import numpy as np
import tensorflow as tf

from psgnets.ops.utils import mask_tensor

def read_rendering_matrix(mat, out_shape=[4,4]):
    mat_shape = mat.shape.as_list()
    assert len(mat_shape) == 3
    out_mat = tf.reshape(mat, [mat_shape[0], mat_shape[1], out_shape[0], out_shape[1]])
    return out_mat

def raw_to_agent_particles(particles, Vmat):
    '''
    Map raw particle coordinates to agent-relative coordinates

    #### INPUTS ####
    raw_particles: [B,T,N,3] tensor of N particle (x,y,z) world coordinates
    Vmat: [4,4] matrix that transforms raw to agent-relative homogeneous coordinates

    #### RETURNS ####
    agent_relative_particles: [B,T,N,3] agent-relative coordinates of particles
    '''
    shape = particles.shape.as_list()
    Vmat = tf.zeros([shape[0],shape[1],4,4], tf.float32) + tf.reshape(Vmat, [1,1,4,4])
    raw_particles = tf.concat([particles, tf.ones(shape[0:3]+[1], dtype=tf.float32)],
                              axis=3) # [B,T,N,4]
    raw_particles = tf.transpose(raw_particles, (0,1,3,2)) # [B,T,4,N]
    agent_relative_particles = tf.matmul(Vmat, raw_particles)
    agent_relative_particles = tf.transpose(agent_relative_particles, (0,1,3,2))
    agent_relative_particles = agent_relative_particles[:,:,:,0:3] # remove homogeneous coordinate

    return agent_relative_particles

def invert_camera_matrix(w2c_mat):
    '''
    Take a camera matrix that converts world to camera coordinates and invert it so that it converts camera to world
    '''
    assert w2c_mat.shape.as_list()[-2:] == [4,4]
    B,T,_,_ = w2c_mat.shape.as_list()
    rot = w2c_mat[:,:,:3,:3] # [B,T,3,3]
    trans = w2c_mat[:,:,:3,3:4] # [B,T,3,1]
    rot_inv = tf.transpose(rot, [0,1,3,2]) # [B,T,3,3]
    trans_inv = -tf.matmul(rot_inv, trans) # [B,T,3,1]
    c2w_mat = tf.concat([rot_inv, trans_inv], axis=-1) # [B,T,3,4]
    onevec = tf.tile(tf.reshape(tf.constant([0.,0.,0.,1.], tf.float32), [1,1,1,4]), [B,T,1,1])
    c2w_mat = tf.concat([c2w_mat, onevec], axis=2) # [B,T,4,4]
    return c2w_mat

def transform_vector_between_cameras(vec, cam1, cam2, rotate_180=False):
    '''
    Take 3-vectors of shape [B,...,3] and transform them from cam1's coordinates to cam2's

    vec: [B,N,3] a set of 3-vectors in camera1's coordinate system
    cam1: [B,4,4] world2camera coordinate transformation matrix
    cam2: [B,4,4] world2camera coordinate transformation matrix

    rotate_180: if True, flips y and z of input vector before and after transformation

    returns:
    vec_cam2: [B,N,3]

    '''
    shape = vec.shape.as_list()
    assert vec.shape[-1] == 3, "Only transforms 3-vectors"
    if rotate_180:
        vec = tf.stack([vec[...,0], -vec[...,1], -vec[...,2]], axis=-1)

    vec = tf.concat([vec, tf.ones(shape[:-1] + [1], dtype=tf.float32)], axis=-1) # [B,N,4]
    vec = tf.transpose(vec, [0,2,1]) # [B,4,N]
    cam1inv = invert_camera_matrix(cam1[:,tf.newaxis])[:,0]
    vec = tf.matmul(cam1inv, vec) # now in world coordinates
    vec = tf.matmul(cam2, vec) # now in camera 2's coordinates
    vec = tf.transpose(vec, [0,2,1]) # [B,N,4]
    vec = vec[...,0:3] # [B,N,3]
    if rotate_180:
        vec = tf.stack([vec[...,0], -vec[...,1], -vec[...,2]], axis=-1)

    return vec

def rotate_vector_between_cameras(vec, cam1, cam2, rotate_180=False):
    '''
    Rotate a 3-vector using cam1's and cam2's rotation matrices; ignore translation

    vec: [B,N,3]
    cam1: [B,4,4]
    cam2: [B,4,4]
    '''
    shape = vec.shape.as_list()
    assert vec.shape[-1] == 3
    if rotate_180:
        vec = tf.stack([vec[...,0], -vec[...,1], -vec[...,2]], axis=-1)
    vec = tf.transpose(vec, [0,2,1]) # [B,3,N]
    rot1 = cam1[:,:3,:3]
    rot1_inv = tf.transpose(rot1, [0,2,1]) # [B,3,3]
    rot2 = cam2[:,:3,:3]
    vec = tf.matmul(rot2, tf.matmul(rot1_inv, vec))
    vec = tf.transpose(vec, [0,2,1])
    if rotate_180:
        vec = tf.stack([vec[...,0], -vec[...,1], -vec[...,2]], axis=-1)
    return vec

def camera_projection(particles, Pmat, eps=1.0e-8):
    shape = particles.shape.as_list()
    if len(Pmat.shape.as_list()) != 4:
        Pmat = tf.zeros([shape[0], shape[1], 4, 4], dtype=tf.float32) + tf.reshape(Pmat, [1,1,4,4])
    ps = tf.concat([particles, tf.ones(shape[0:3]+[1], dtype=tf.float32)], axis=3)
    ps = tf.transpose(ps, (0,1,3,2))
    ps = tf.matmul(Pmat, ps)
    ps = tf.transpose(ps, (0,1,3,2))
    # normalize by homogeneous coordinate, then remove it
    ps = tf.divide(ps, ps[:,:,:,3:4]+eps)[:,:,:,0:3]

    return ps

def agent_particles_to_image_coordinates(particles, Pmat=None, H_out=128, W_out=170, to_integers=False, int_type=tf.int32, eps=1.0e-8):
    '''
    Map agent-relative particles in 3-space to image coordinates in 2-space
    via a perspective projection, encoded in Pmat

    #### INPUTS ####
    ps: [B,T,N,3] agent-relative particle (x,y,z) coordinates
    Pmat: [4,4] matrix that performs perspective projection on homogeneous coordinates
    H_out, W_out: the dimensions of the image to project onto
    to_integers: if True, returned coordinates are typed as integers and can serve as indices

    #### RETURNS ####
    particle_image_coordinates: [B,T,N,2] coordinates in 2-space between [0,H_out] x [0,W_out]
                                By convention the first coordinate indexes the vertical dimension.
    '''
    shape = particles.shape.as_list()
    if len(Pmat.shape.as_list()) != 4:
        Pmat = tf.zeros([shape[0], shape[1], 4, 4], dtype=tf.float32) + tf.reshape(Pmat, [1,1,4,4])
    ps = tf.concat([particles, tf.ones(shape[0:3]+[1], dtype=tf.float32)], axis=3)
    ps = tf.transpose(ps, (0,1,3,2))
    ps = tf.matmul(Pmat, ps)
    ps = tf.transpose(ps, (0,1,3,2))
    # normalize by homogeneous coordinate, then remove it
    ps = tf.divide(ps, ps[:,:,:,3:4]+eps)[:,:,:,0:3]

    # map to image coordinates; y axis is inverted by convention
    xims = float(W_out) * 0.5 * (ps[:,:,:,0:1] + 1.0)
    yims = float(H_out) * (1.0 - 0.5*(ps[:,:,:,1:2] + 1.0))

    if to_integers:
        xims = tf.cast(tf.floor(xims), dtype=int_type)
        yims = tf.cast(tf.floor(yims), dtype=int_type)

        xims = tf.minimum(tf.maximum(xims, 0), W_out-1)
        yims = tf.minimum(tf.maximum(yims, 0), H_out-1)

    particle_image_coordinates = tf.concat([yims, xims], axis=3)
    return particle_image_coordinates

def occlude_particles_in_camera_volume(particles_image_coordinates, particles_depths, p_radius, particles_mask=None):
    '''
    particle_depths are in real space (not camera volume), so they are negative. p_z > q_z implies p_z is in front.

    particles_image_coordinates: [B,T,N,2]
    particles_depths: [B,T,N,1]

    returns:
    not_occluded_mask [B,T,N,1] mask of particles which are NOT occluded
    '''
    # particles behind camera shouldn't occlude anything
    # if particles_mask is not None:
    #     pim_coordinates = mask_tensor(particles_image_coordinates, particles_mask, mask_value=-100.0*p_radius)
    # else:
    #     pim_coordinates = particles_image_coordinates
    # ps_in_front_of_camera = tf.cast(particles_depths < 0.0, tf.float32)
    # if particles_mask is not None:
        # ps_in_front_of_camera = ps_in_front_of_camera * particles_mask
    # ps_in_front_of_camera = tf.cast(particles_depths < 0.0, dtype=tf.float32)
    # coord_mask_val = -100.0 * p_radius * tf.range(1, 1+particles_depths.shape.as_list()[2], dtype=tf.float32)
    # coord_mask_val = tf.reshape(coord_mask_val, [1,1,-1,1]) # now masked particles won't occlude each other
    coord_mask_val=-100.0*p_radius
    mask = particles_mask * tf.cast(particles_depths < 0.0, tf.float32)
    pim_coordinates = mask_tensor(particles_image_coordinates, mask=mask, mask_value=coord_mask_val)
    # pim_coordinates = mask_tensor(particles_image_coordinates, mask=ps_in_front_of_camera, mask_value=coord_mask_val)
    # if particles_mask is not None:
    #     pim_coordinates = mask_tensor(pim_coordinates, mask=particles_mask, mask_value=-100.0*p_radius)

    p_q_hw_dist2 = tf.reduce_sum(tf.square(tf.expand_dims(pim_coordinates, 3) - tf.expand_dims(pim_coordinates, 2)), axis=4) # [B,T,N,N]
    q_disc_includes_p = p_q_hw_dist2 < tf.square(tf.cast(p_radius, dtype=tf.float32))
    q_in_front_of_p = tf.transpose(particles_depths, [0,1,3,2]) > particles_depths # [B,T,N,N] last axis is potential occluders

    q_occluding_p = tf.cast(tf.logical_and(q_disc_includes_p, q_in_front_of_p), dtype=tf.float32) # [B,T,N,N]
    p_occluded = tf.reduce_max(q_occluding_p, axis=3, keepdims=True) # [B,T,N,1]

    not_occluded_mask = 1.0 - p_occluded

    # all fake particles are occluded
    if particles_mask is not None:
        not_occluded_mask *= particles_mask

    return not_occluded_mask

def project_and_occlude_particles(particles,
                                  projection_matrix,
                                  im_size=[1,1],
                                  particles_mask=None,
                                  p_radius=0.0,
                                  xyz_dims=[0,3],
                                  particles_agent_centered=True,
                                  camera_matrix=None,
                                  coord_max=100.0,
                                  **kwargs
):
    '''
    particles: [B,T,N,D] where the last axis contains both xyz_dims and color_dims (slices)
    im_size: [H,W] ints
    projection_matrix: [B,T,4,4] the matrix for projecting from euclidean space into canonical image coordinates
    xyz_dims: [xdim, zdim+1] must have a difference of 3

    p_radius: float that determines how particles will occlude one another.
    particles_agent_centered: if false, use camera_matrix ("V") to put particles in camera-relative coordinates
    camera_matrix: [B,T,4,4] the matrix "V" for converting global to camera-relative coordinates

    returns
    particles_im_indices: [B,T,N,2] of int32 indices into H and W dimensions of an image of size H,W
    not_occluded_mask: [B,T,N,1] float32 where 1.0 indicates the particle wasn't occluded at radius p_radius, 0.0 otherwise
    '''
    B,T,N,D = particles.shape.as_list()
    H,W = im_size
    if p_radius is None:
        p_radius = 3.0 * (np.minimum(H,W).astype(float) / 256.0)
        print("p radius", p_radius)
    # get indices into image
    particles_xyz = particles[...,xyz_dims[0]:xyz_dims[1]]
    if not particles_agent_centered:
        assert camera_matrix is not None, "need a camera matrix to put into agent coordinates"
        particles_agent = raw_to_agent_particles(particles_xyz, Vmat=camera_matrix)
    else:
        particles_agent = particles_xyz

    # [B,T,N,2] <tf.float32> values in range [0,H]x[0,W]
    particles_depths = particles_agent[...,-1:]
    particles_im_coordinates = agent_particles_to_image_coordinates(particles_agent,
                                                                    Pmat=projection_matrix,
                                                                    H_out=H, W_out=W,
                                                                    to_integers=False)


    # clip coordinates
    particles_im_coordinates = tf.where(tf.logical_or(tf.is_nan(particles_im_coordinates), tf.is_inf(particles_im_coordinates)),
                                        coord_max * tf.ones_like(particles_im_coordinates), # true
                                        particles_im_coordinates # false
    )
    particles_im_coordinates = tf.maximum(tf.minimum(particles_im_coordinates, coord_max), -coord_max)

    # particles_depths = tf.Print(particles_depths, [particles_depths.shape[2], tf.reduce_max(particles_depths), tf.reduce_min(particles_im_coordinates), tf.reduce_max(particles_im_coordinates)], message='p_depth/coords')

    # resolve occlusions
    not_occluded_mask = occlude_particles_in_camera_volume(particles_im_coordinates,
                                                           particles_depths,
                                                           p_radius=p_radius,
                                                           particles_mask=particles_mask
                                                           # particles_mask=None
    )
    particles_im_indices = tf.cast(tf.round(particles_im_coordinates), dtype=tf.int32)
    particles_im_indices = tf.minimum(tf.maximum(particles_im_indices, 0),
                                      tf.reshape(tf.constant(np.array([H-1,W-1]), dtype=tf.int32), [1,1,1,2]))

    return particles_im_indices, not_occluded_mask, particles_im_coordinates # last is float

def nodes_from_spatial_inds(nodes, spatial_inds, segment_ids, im_size):
    B,T,N,D = nodes.shape.as_list()
    _,_,H,W = segment_ids.shape.as_list()
    P = spatial_inds.shape.as_list()[2]
    H_im, W_im = im_size
    strides = tf.reshape(tf.constant([H_im // H, W_im // W], dtype=tf.int32), [1,1,1,2])

    # preproc segment ids
    segment_ids -= tf.reduce_min(segment_ids, axis=[2,3], keepdims=True) # now they start at 0 per example
    segment_ids = tf.where(segment_ids < N, # can't go over the number of summary nodes
                           segment_ids,
                           tf.ones_like(segment_ids)*(N-1))
    spatial_inds_ds = tf.floordiv(spatial_inds, strides)

    segments_to_decode = get_image_values_from_indices(
        segment_ids[...,tf.newaxis],
        spatial_inds_ds) # [B,T,P,1] <int32> where P is num points to decode
    ones = tf.ones([B,T,P,1], dtype=tf.int32)
    # create inds into centroids and vectors to call with tf.gather_nd
    segments_to_decode = tf.concat([
        tf.reshape(tf.range(B, dtype=tf.int32), [B,1,1,1])*ones,
        tf.reshape(tf.range(T, dtype=tf.int32), [1,T,1,1])*ones,
        segments_to_decode
    ], axis=-1) # [B,T,P,3]

    sampled_nodes = tf.gather_nd(nodes, segments_to_decode) # [B,T,P,D]
    return sampled_nodes

def hw_attrs_to_image_inds(hw_attrs, im_size):
    '''
    inputs
    hw_attrs: [...,2] in range [-1., 1.]

    returns
    hw_inds: [...,2] in range [0,H-1] x [0,W-1]
    '''
    shape = hw_attrs.shape.as_list()
    assert shape[-1] == 2
    im_size = tf.reshape(tf.constant(im_size, tf.float32), [1]*(len(shape)-1) +[2]) # [shape[:-1],2]
    hw_inds = (hw_attrs + 1.0) / 2.0
    hw_inds = hw_inds * (im_size - 1.0)

    hw_inds = tf.cast(tf.floor(hw_inds), tf.int32)
    hw_inds = tf.minimum(tf.maximum(hw_inds, 0), tf.cast(im_size, tf.int32) - 1)
    return hw_inds

def hw_to_xy(hw_vals, z_vals, focal_lengths, negative_z=True, near_plane=0.1):
    '''
    hw_vals: [bdims,...,2] in (-1., 1.)
    z_vals: [bdims,...,1] in [-zmax, 0) or [0, zmax) if not negative_depth
    focal_lengths: [bdims,2] divisors fx,fy
    '''
    if negative_z:
        z_vals = -z_vals # now positive
    if near_plane is not None:
        z_vals = tf.maximum(z_vals, tf.cast(near_plane, tf.float32))

    batch_shape = focal_lengths.shape.as_list()[:-1]
    vals_shape = hw_vals.shape.as_list()[len(batch_shape):-1]
    if len(vals_shape):
        focal_lengths = tf.reshape(focal_lengths, batch_shape + [1]*len(vals_shape) + [2])

    # inverse pinhole projection
    focal_lengths = tf.stack([focal_lengths[...,1], focal_lengths[...,0]], -1) # [fy, fx]
    yx_vals = (hw_vals / focal_lengths) * z_vals
    xy_vals = tf.stack([yx_vals[...,1], -yx_vals[...,0]], axis=-1)

    return xy_vals

def get_image_values_from_indices(images, particles_im_indices):
    reshape = False
    if len(images.shape) == 4:
        assert len(particles_im_indices.shape) == 3, (images, particles_im_indices)
        images = images[:,tf.newaxis]
        particles_im_indices = particles_im_indices[:,tf.newaxis]
        reshape = True

    B,T,N,_ = particles_im_indices.shape.as_list()
    _,Tim,H,W,Dim = images.shape.as_list()

    if Tim == 1 and T > 1:
        images = tf.tile(images, [1,T,1,1,1])

    # get the images values at particles_im_indices -- output is [B,T,N,Dim]
    ones = tf.ones([B,T,N], dtype=tf.int32)
    inds_b = tf.reshape(tf.range(B, dtype=tf.int32), [B,1,1]) * ones
    inds_t = tf.reshape(tf.range(T, dtype=tf.int32), [1,T,1]) * ones
    inds_h = particles_im_indices[...,0] # [B,T,N]
    inds_w = particles_im_indices[...,1] # [B,T,N]
    gather_inds = tf.stack([inds_b, inds_t, inds_h, inds_w], axis=3)

    image_values = tf.gather_nd(images, gather_inds) # [B,T,N,Dim]

    if reshape:
        assert T == 1, T
        image_values = image_values[:,0]

    return image_values

def render_nodes_with_segment_ids(nodes, segment_ids):
    assert segment_ids.dtype == tf.int32
    reshape = False
    if len(segment_ids.shape) == 3:
        reshape = True
        segment_ids = segment_ids[:,tf.newaxis]
        nodes = nodes[:,tf.newaxis]
    else:
        assert len(segment_ids.shape) == 4 and len(nodes.shape) == 4

    B,T,H,W = segment_ids.shape.as_list()
    _B,_T,N,D = nodes.shape.as_list()
    assert [_B,_T] == [B,T]

    ones = tf.ones_like(segment_ids)
    b_inds = tf.reshape(tf.range(B, dtype=tf.int32), [B,1,1,1]) * ones
    t_inds = tf.reshape(tf.range(T, dtype=tf.int32), [1,T,1,1]) * ones
    n_inds = segment_ids
    gather_inds = tf.stack([b_inds, t_inds, n_inds], -1) # [B,T,H,W,3]

    rendered = tf.gather_nd(nodes, gather_inds)
    if reshape:
        rendered = rendered[:,0]

    return rendered

def sample_image_inds(out_shape, im_size, train, **kwargs):

    B,T,P = out_shape
    random_points = train and kwargs.get('random_points', True)
    if random_points:
        spatial_inds = tf.stack([
            tf.random_uniform([B, T, P], minval=0, maxval=im_size[0], dtype=tf.int32),
            tf.random_uniform([B, T, P], minval=0, maxval=im_size[1], dtype=tf.int32)
        ], axis=-1)
    else: # grid
        grid_spacing = kwargs.get('grid_spacing', [im_size[0] // int(np.sqrt(P)), im_size[1] // int(np.sqrt(P))])
        h_inds = tf.tile(tf.range(0, im_size[0], grid_spacing[0], dtype=tf.int32)[:,tf.newaxis], [1, int(im_size[1] // grid_spacing[1])])
        h_inds = tf.reshape(h_inds, [1,1,-1,1])
        w_inds = tf.tile(tf.range(0, im_size[1], grid_spacing[1], dtype=tf.int32), [int(im_size[0] // grid_spacing[0])])
        w_inds = tf.reshape(w_inds, [1,1,-1,1])
        ones = tf.ones([B, T, 1, 1], tf.int32)
        spatial_inds = tf.concat([h_inds, w_inds], axis=-1) * ones # [B,T,P,2]
    return spatial_inds

def sample_delta_image_inds(images, num_points, static=False, rgb_max=255.0, eps=1e-6, use_cpu=True, **kwargs):
    '''
    preferentially sample indices from parts of the image where im[:,t+1] - im[:,t] is high
    '''
    B,T,H,W,C = images.shape.as_list()
    images = tf.cast(images, tf.float32) / rgb_max
    delta_ims = tf.reduce_sum(tf.square(images[:,1:] - images[:,:-1]), axis=-1, keepdims=False)
    delta_ims = tf.reshape(delta_ims, [B,T-1,-1])
    if static:
        delta_max = tf.reduce_max(delta_ims, axis=-1, keepdims=True)
        delta_ims = delta_max - delta_ims # invert so that sample is from the non-moving parts
    probs = tf.div(tf.maximum(0.0, delta_ims), tf.reduce_sum(delta_ims, axis=-1, keepdims=True) + eps) # [B,T-1,HW]

    if use_cpu:
        with tf.device("/cpu:0"):
            dist = tf.distributions.Categorical(probs=probs, dtype=tf.int32)
            points = dist.sample(num_points) # [num_points, B, T-1]
            points = tf.transpose(points, [1,2,0]) # [B,T-1,num_points] <tf.int32>
    else:
        dist = tf.distributions.Categorical(probs=probs, dtype=tf.int32)
        points = dist.sample(num_points) # [num_points, B, T-1]
        points = tf.transpose(points, [1,2,0]) # [B,T-1,num_points] <tf.int32>

    # turn back into indices
    points_h = tf.minimum(tf.maximum(tf.floordiv(points, W), 0), H-1)
    points_w = tf.minimum(tf.maximum(tf.floormod(points, W), 0), W-1)
    points = tf.stack([points_h, points_w], axis=-1) # [B,T-1,P,2]
    return points

## optical flows stuff
def preproc_flows(flows, backward=False, scale=True, xy_flows=True, stop_gradient=True, scale_factor=0.5, **kwargs):
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

def backward_warp_image(im, flows):
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

def forward_warp_image(im, flows, stop_gradient=True, **kwargs):
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

def compute_occlusion_map(forward_flows, backward_flows, scale=True, xy_flows=True, stop_gradient=True, scale_factor=0.5, **kwargs):
    '''
    '''
    B,T,H,W,_ = forward_flows.shape.as_list()

    forward_flows = tf.reshape(forward_flows, [B*T,H,W,2])
    backward_flows = tf.reshape(backward_flows, [B*T,H,W,2])

    forward_flows = preproc_flows(forward_flows, backward=True, scale=scale, xy_flows=xy_flows, stop_gradient=stop_gradient, scale_factor=scale_factor)
    backward_flows = preproc_flows(backward_flows, backward=True, scale=scale, xy_flows=xy_flows, stop_gradient=stop_gradient, scale_factor=scale_factor)

    ones = tf.ones([B*T,H,W,1], dtype=tf.float32)
    forward_occlusions = 1. - tf.minimum(forward_warp_image(im=ones, flows=forward_flows), 1.) # occlusions are 1.
    backward_occlusions = 1. - tf.minimum(forward_warp_image(im=ones, flows=backward_flows), 1.) # occlusions are 1.

    forward_occlusions = tf.reshape(forward_occlusions, [B,T,H,W,1])
    backward_occlusions = tf.reshape(backward_occlusions, [B,T,H,W,1])

    return forward_occlusions, backward_occlusions

def propagate_index_map(segment_ids, flows, valid_nodes, **kwargs):
    '''
    Predict where next indices in segment_ids end up according to flow predictions
    '''
    B,T,H,W = segment_ids.shape.as_list()
    assert flows.shape.as_list() == [B,T,H,W,2]
    _,_,N,_ = valid_nodes.shape.as_list()
    flows = preproc_flows(flows, **kwargs)
    index_map = tf.one_hot(segment_ids, depth=N, axis=-1, dtype=tf.float32) # [B,T,H,W,N]
    pred_index_map = forward_warp_image(im=index_map[:,0:-1], flows=flows[:,0:-1], **kwargs) # [B,T-1,H,W,N]
    pred_index_map *= valid_nodes[:,0:-1,tf.newaxis,tf.newaxis,:,0]
    contested_pixels = tf.cast(tf.reduce_sum(pred_index_map, axis=-1, keepdims=True) > 1.0, tf.float32)
    mask = tf.cast(pred_index_map > 0., tf.float32) # [B,T-1,H,W,N]

    return pred_index_map, contested_pixels, mask # [B,T-1,H,W,N], [B,T-1,H,W,1], [B,T-1,H,W,N]

def resolve_depth_order(index_masks, node_depths, valid_nodes, min_depth=-100., beta=1.0, valid_mask=True, softmax=True, **kwargs):
    '''
    '''
    B,T,H,W,N = index_masks.shape.as_list()
    _,_T = node_depths.shape.as_list()[:2]
    # assert _N == N, (_N, N)
    if _T == T + 1:
        node_depths = node_depths[:,0:-1]
        valid_nodes = valid_nodes[:,0:-1]
    else:
        assert _T == T, (_T,T)

    ## more positive (less negative) is closer
    valids = valid_nodes[:,:,tf.newaxis,tf.newaxis,:,0]
    if len(node_depths.shape) == 4:
        depths = index_masks * node_depths[:,:,tf.newaxis,tf.newaxis,:,0] # [B,T,H,W,N]
    elif len(node_depths.shape) == 5:
        assert node_depths.shape == index_masks.shape
        depths = index_masks * node_depths
    # depths += (1. - index_masks) * min_depth
    depths += (1. - index_masks) * tf.minimum(
        min_depth, tf.reduce_min(depths, axis=-1, keepdims=True))

    if valid_mask:
        depths = valids * depths + (1.- valids) * min_depth

    depth_weights = tf.nn.softmax(depths * beta, axis=-1) if softmax else (depths * beta)

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
