import tensorflow as tf
import numpy as np

PRINT = True

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
        dh = tf.floordiv(k, ksize) - r
        dw = tf.floormod(k, ksize) - r
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

        rn = tf.constant(noise, tf.float32) * tf.random_normal(shape=[B,N+1,K], seed=seed, dtype=tf.float32)
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

def euclidean_dist2(v1, v2, thresh='local', return_affinities=False, eps=1e-12, **kwargs):
    B,N,C,F = v2.shape.as_list()
    assert v1.shape.as_list() == [B,N,C,1], (v1.shape.as_list(), v2.shape.as_list())

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
    # valid_adj = tf.logical_or(
    #     tf.logical_and(v1_valid, v2_valid),
    #     tf.logical_and(tf.logical_not(v1_valid), tf.logical_not(v2_valid))
    # )
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
        h_inds = tf.floordiv(n_inds, W)
        w_inds = tf.floormod(n_inds, W)
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
    overlap = tf.unsorted_segment_sum(
        data=tf.ones([B*H*W], dtype=tf.int32),
        segment_ids=tf.reshape(segs_hash, [-1]),
        num_segments=(B*N*N + 1))
    overlap = tf.minimum(overlap[:-1], tf.constant(1, tf.int32))
    overlap = tf.reshape(overlap, [B,N,N])

    return overlap
