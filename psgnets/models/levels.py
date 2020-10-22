import copy
import os
import pdb

import numpy as np
import tensorflow.compat.v1 as tf
import vvn.ops.convolutional as convolutional
import vvn.ops.graphical as graphical
import vvn.ops.pooling as pooling
import vvn.ops.vectorizing as vectorizing
import vvn.ops.rendering as rendering
import vvn.ops.utils as utils
from preprocessing import compute_sobel_features, preproc_rgb
from vvn.models.gcn.model import GCN
from vvn.ops.dimensions import DimensionDict
from vvn.models.losses import cross_entropy
from base import Model
from vae import VAE

PRINT = False

class BaseLevel(Model):

    def __init__(self, name='level0', input_name='base_tensor', **model_params):

        self.input_name = input_name
        self.num_nodes = self.size = self.num_attrs = None
        self.Dims = None
        super(BaseLevel, self).__init__(
            name=name,
            model_func=self.reshape_base_tensor,
            time_shared=True,
            **model_params)

    def reshape_base_tensor(self, x, **kwargs):
        shape = x.shape.as_list()
        assert len(shape) in [4,5], shape
        BT = shape[0]
        C = shape[-1]
        self.size = shape[-3:-1]
        self.num_nodes = np.prod(self.size)
        self.num_attrs = C
        self.Dims = DimensionDict({'vector':C})
        if len(shape) == 4:
            self.parent_nodes = tf.reshape(x, [BT,self.num_nodes,C])
        elif len(shape) == 5:
            self.parent_nodes = tf.reshape(x, [BT,-1,self.num_nodes,C])
        self.parent_segment_ids = tf.tile(tf.reshape(
            tf.range(self.num_nodes, dtype=tf.int32)[tf.newaxis],
            [1]+self.size), [BT,1,1])
        return {
            'parent_nodes': self.parent_nodes,
            'parent_segment_ids': self.parent_segment_ids
        }

class PSGLevel(Model):

    def __init__(self, name, input_name='level0', num_nodes=128, grouping_time=0, vectorization_time=-1, **model_params):

        super(PSGLevel, self).__init__(model_func=None, name=name, **model_params)
        self.input_name = input_name
        self.num_nodes = num_nodes
        self.t_group = grouping_time
        self.t_vec = vectorization_time
        self.nodes = self.segment_ids = None
        self.affinities = self.edges = self.parent_edges = None
        self.parent_segment_ids = self.parent_nodes = self.parent_attributes = None
        self.Dims = None
        self.loss = None
        assert "level" in self.name, "Must name each PSG level with 'level' but name is %s" % self.name

    def reshape_inputs(self, **kwargs):
        '''
        Give the input nodes, features, and segment_ids their proper shape
        '''
        assert any((x is not None for x in [self.nodes, self.features]))
        return

    def compute_affinities(self, **kwargs):
        '''
        Compute affinities between nodes as function of the node attribute vectors and/or segment_ids
        '''
        assert self.nodes is not None
        raise NotImplementedError

    def threshold_affinities(self, **kwargs):
        '''
        Convert real-valued affinities to boolean edges
        '''
        assert self.affinities is not None
        raise NotImplementedError

    def compute_parents(self, **kwargs):
        '''
        Assign a parent node index to each input (child) node using the derived affinity and/or edge matrix; may be implemented as label propagation alg.
        '''
        assert self.edges is not None
        raise NotImplementedError

    def register_parents(self, **kwargs):
        assert self.parent_edges is not None
        assert self.segment_ids is not None
        raise NotImplementedError

    def aggregate_nodes_and_features(self, **kwargs):
        assert self.nodes is not None
        assert self.parent_edges is not None
        assert self.parent_segment_ids is not None
        raise NotImplementedError

    def predict_parent_attributes(self, **kwargs):
        assert self.parent_nodes is not None
        raise NotImplementedError

    def format_output_nodes(self, **kwargs):
        assert self.parent_nodes is not None
        assert self.parent_attributes is not None
        raise NotImplementedError

    def build_model(self, **model_params):

        def model(input_nodes, input_segment_ids=None, features=None, actions={}, train=True, inputDims=None, **kwargs):

            self.is_training = train
            self.nodes = input_nodes
            self.segment_ids = input_segment_ids
            self.features = features
            self.actions = actions
            self.inputDims = inputDims
            self.new_features = tf.zeros([self.BT], dtype=tf.float32)
            self.loss = tf.zeros([self.BT], dtype=tf.float32) # a loss e.g. from a VAE
            kwargs.update(self.params)

            # If there are more inputs than the level L-1 nodes, reshape those and get shapes
            self.reshape_inputs(**kwargs)

            # Compute <tf.float32> affinities from nodes/features/actions, [BT,N,N] (fc) or [BT,N,K**2] (local)
            self.affinities = self.compute_affinities(**kwargs)

            # Compute <tf.bool> edges from affinities (same shape)
            self.edges = self.threshold_affinities(**kwargs)

            # Compute the parent node ids by pooling graph, [BT,N] <tf.int32>
            self.parent_edges = self.compute_parents(**kwargs)

            # Create a [BT,H,W] <tf.int32> label map of the input features from scratch or the previous label map
            self.parent_segment_ids = self.register_parents(**kwargs)

            # Create the new aggregate nodes, [BT,M,C]
            self.parent_nodes = self.aggregate_nodes_and_features(**kwargs)

            # Predict new attributes (e.g. by MLP and GC) from nodes, [BT,M,D]
            self.parent_attrs = self.predict_parent_attributes(**kwargs)

            # Arrange node attributes [BT,M,F] (where F is at most C + D)
            self.parent_nodes = self.format_output_nodes(**kwargs)

            return {
                'child_valid': self.valid_nodes,
                'child_nodes': self.nodes,
                'parent_nodes': self.parent_nodes,
                'affinities': self.affinities,
                'within_edges': self.edges,
                'parent_edges': self.parent_edges,
                'parent_segment_ids': self.parent_segment_ids,
                'new_features': self.new_features,
                'loss': self.loss
            }

        self.model_func = model
        self.model_func_name = type(self).__name__ + '_model'

class P1Level(PSGLevel):

    def __init__(self, name, input_name,
                 num_nodes=128,
                 num_attrs=24,
                 affinity_kwargs={},
                 pooling_kwargs={},
                 aggregation_kwargs={},
                 mlp_kwargs={},
                 graphconv_kwargs=None,
                 format_kwargs={},
                 **kwargs
    ):

        super(P1Level, self).__init__(name, input_name, num_nodes, time_shared=True, **kwargs)
        self.num_attrs = num_attrs
        self.affinity_kwargs = copy.deepcopy(affinity_kwargs)
        self.pooling_kwargs = copy.deepcopy(pooling_kwargs)
        self.aggregation_kwargs = copy.deepcopy(aggregation_kwargs)
        self.mlp_kwargs = copy.deepcopy(mlp_kwargs)
        self.graphconv_kwargs = None if not graphconv_kwargs else copy.deepcopy(graphconv_kwargs)
        self.format_kwargs = copy.deepcopy(format_kwargs)

    def reshape_inputs(self, **kwargs):
        # if output was from a convrnn
        if len(self.nodes.shape) > 3:
            self.grouping_nodes = self.nodes[:,self.t_group]
            self.nodes = self.nodes[:,self.t_vec]
        else:
            self.grouping_nodes = self.nodes

        self.N = self.nodes.shape.as_list()[1]
        self.M = self.num_nodes # shorthand
        self.D = self.num_attrs # shorthand
        self.valid_nodes = tf.ones([self.BT, self.N], dtype=tf.float32)

        # deal w features
        if self.features is not None:
            assert self.features.shape.as_list()[0:2] == [self.B, self.T]
            if len(self.features.shape) > 5:
                self.features = self.features[:,:,self.t_vec]
            self.features = self.reshape_batch_time(self.features, merge=True)
            self.im_shape = self.features.shape.as_list()[1:]
            self.size = self.im_shape[0:2]
        else:
            H = np.sqrt(self.nodes.shape.as_list()[1]).astype(int)
            self.size = kwargs.get('size', None) or [H,H]
            assert np.prod(self.size) == self.nodes.shape.as_list()[1]
            self.im_shape = self.size + [self.nodes.shape.as_list()[-1]]
        self.C = self.im_shape[-1]

        # new feature
        self.new_features = tf.zeros([self.BT] + self.size + [0], dtype=tf.float32)

        # deal w actions
        if self.actions is not None:
            self.actions = {ak: self.reshape_batch_time(act, merge=True)
                            for ak,act in self.actions.items()}

        # make nodes spatially registered for P1 affinities
        self.grouping_nodes = tf.reshape(self.grouping_nodes, [self.BT] + self.size + [-1])

    def compute_affinities(self, **kwargs):
        '''
        Inverse Euclidean distance affinities in a 2k+1 neighborhood of each convolutional feature (stored in nodes)
        '''
        kwargs.update(self.affinity_kwargs)
        affinities, edges = graphical.compute_adjacency_from_features(
            features=self.grouping_nodes, return_affinities=True, **kwargs)

        self.edges = edges
        return affinities

    def threshold_affinities(self, **kwargs):
        '''
        Do nothing but check edges were computed directly from affinities
        '''
        assert self.edges.shape == self.affinities.shape
        assert self.edges.dtype == tf.bool, "edges must be thresholded affinities but are type %s" % self.edges.dtype
        return self.edges

    def compute_parents(self, **kwargs):
        '''
        Compute parent node assignments (i.e. parent edges) by labelprop on local edge matrix derived from image-like tensor
        '''
        kwargs.update(self.pooling_kwargs)
        labels, num_segments = pooling.compute_segments_by_label_prop(
            edges=self.edges, size=self.size, valid_nodes=self.valid_nodes, **kwargs)
        self.num_parents = num_segments # [BT]

        return labels

    def register_parents(self, **kwargs):
        '''
        Create an image of parent_segment_ids [BT,H,W]
        '''

        image_inds = utils.inds_image(
            batch_size=self.B, seq_len=self.T, imsize=self.size)
        image_inds = tf.reshape(image_inds, [self.BT,1,self.N,2])
        parent_segment_ids = rendering.get_image_values_from_indices(
            images=tf.reshape(self.parent_edges, [self.BT,1] + self.size + [1]),
            particles_im_indices=image_inds)
        parent_segment_ids = tf.reshape(parent_segment_ids, [self.BT] + self.size) # [BT,H,W]
        parent_segment_ids = utils.preproc_segment_ids(
            parent_segment_ids, Nmax=self.M, return_valid_segments=False)
        return parent_segment_ids

    def aggregate_nodes_and_features(self, **kwargs):

        kwargs.update(self.aggregation_kwargs)
        agg_shape = [self.BT] + self.size
        assert self.features.shape.as_list()[:3] == agg_shape
        assert self.parent_edges.shape.as_list() == [self.BT, self.N]
        assert self.num_parents.shape.as_list() == [self.BT]

        # concat new features if they've been added along the way
        if kwargs.get('concat_new_features', False):
            self.features = tf.concat([self.features, self.new_features], axis=-1)
            self.nodes = tf.concat([
                self.nodes, tf.reshape(self.new_features, [self.BT, self.N, -1])], axis=-1)
            self.C += self.new_features.shape.as_list()[-1]

        Dims = DimensionDict({'features': self.features.shape.as_list()[-1]})
        if kwargs.get('concat_diff_features', False):
            self.features = self.reshape_batch_time(self.features, merge=False)
            func = tf.stop_gradient if kwargs.get('stop_gradient_diff_features', False) else tf.identity
            diff_features = func(self.features[:,1:] - self.features[:,:-1])
            diff_features = tf.concat([tf.zeros_like(self.features[:,:1]), diff_features], axis=1)
            self.features = tf.concat([self.features, diff_features], axis=-1)
            self.features = self.reshape_batch_time(self.features, merge=True)
            Dims['diff_features'] = self.C

        parent_nodes, _, Dims = vectorizing.aggregate_mean_and_var_features(
            features=self.features, segment_ids=self.parent_edges, num_segments=self.num_parents, max_segments=self.M, dimension_dict=Dims, **kwargs)

        if kwargs.get('concat_edge_attrs', False):
            edges_key = kwargs.get('edge_inputs', None)
            if edges_key:
                sobel_inputs = self.actions[edges_key]
            else:
                sobel_inputs = self.features[...,0:self.C]

            sobel_features = compute_sobel_features(sobel_inputs, size=self.size, normalize_range=False, to_mag=True) # just get angle and mag_sum
            edge_attrs = vectorizing.compute_border_attributes(
                nodes=parent_nodes,
                segment_map=self.reshape_batch_time(self.parent_segment_ids, merge=False),
                features=sobel_features,
                shape_feats=False,
                **kwargs)
            parent_nodes = tf.concat([
                parent_nodes[...,:-4],
                edge_attrs,
                parent_nodes[...,-4:]
            ], axis=-1)
            Dims.insert_from(DimensionDict({'edge_attrs': edge_attrs.shape.as_list()[-1]}), position=-4)

        # within example ids
        self.parent_edges -= tf.reduce_min(self.parent_edges, axis=1, keepdims=True)
        self.Dims = Dims

        return parent_nodes

    def predict_parent_attributes(self, **kwargs):

        mlp_name = self.name + '_attr_mlp'
        mlp_kwargs = copy.deepcopy(kwargs)
        mlp_kwargs.update(self.mlp_kwargs)
        near_plane = mlp_kwargs.get('near_plane', 0.1)

        # predict unary attributes from aggregate nodes
        mlp_kwargs['hidden_dims'] = mlp_kwargs.get('hidden_dims', []) + [self.D]
        pred_attrs_unary = convolutional.mlp(
            inp=self.parent_nodes, scope=mlp_name, **mlp_kwargs)
        uDims = DimensionDict({'unary_attrs':pred_attrs_unary.shape.as_list()[-1]})

        # xyz computation using provided or learned focal length
        pmat = self.actions.get('projection_matrix', None)
        if pmat is not None:
            f = pmat[:,0,0]
        else:
            f = tf.get_variable(name=(self.name+'_focal_length'), shape=[], dtype=tf.float32, initializer=tf.constant_initializer(value=1.0))
            f = tf.maximum(f, near_plane)

        if PRINT:
           f = tf.Print(f, [f], message='focal_length')

        if self.format_kwargs.get('xyz_attr', True):
            f = tf.reshape(f,[-1,1,1])
            uDims['z'] = [0,1]
            z_attr = uDims.get_tensor_from_attrs(
                pred_attrs_unary, 'z', postproc=True)
            hw_attr = self.Dims.get_tensor_from_attrs(
                self.parent_nodes, 'hw_centroids')
            yx_attr = hw_attr * (-z_attr / f)
            x_attr = yx_attr[...,1:2]
            y_attr = -yx_attr[...,0:1]
            uDims.insert_from([('x', [0,1]), ('y', [1,2])])
            uDims['position'] = [0,3]
            uDims['unary_attrs'] = [2,uDims.ndims]
            pred_attrs_unary = tf.concat([x_attr, y_attr, pred_attrs_unary], axis=-1)
            self.D += 2

        # self.parent_attrs = pred_attrs_unary
        self.predDims = uDims

        # if no graphconv
        if self.graphconv_kwargs is None:
            return pred_attrs_unary

        # do a graphconv
        gc_kwargs = copy.deepcopy(kwargs)
        gc_kwargs.update(self.graphconv_kwargs)

        concat_effs = gc_kwargs.get('concat_effects', False)
        if not concat_effs:
            gc_kwargs['hidden_dims'] = gc_kwargs.get('hidden_dims', []) + pred_attrs_unary.shape.as_list()[-1:]
        else:
            self.D += gc_kwargs['hidden_dims'][-1]

        valid_attr = self.Dims.get_tensor_from_attrs(
            self.parent_nodes, 'valid')
        valid_edges = tf.cast(valid_attr * tf.transpose(valid_attr, [0,2,1]), tf.bool)

        if gc_kwargs.get('hw_thresh', None):
            hw_attr = self.Dims.get_tensor_from_attrs(
                self.parent_nodes, 'hw_centroids')
            hw_dists2 = utils.relative_attr_dists2(
                hw_attr, mask=valid_attr)
            valid_edges = tf.logical_and(
                valid_edges, hw_dists2 < gc_kwargs['hw_thresh'])

        gc_name = self.name + '_attr_gc'
        gc_inputs = tf.concat([pred_attrs_unary, self.parent_nodes], axis=-1)
        pred_attrs_binary = graphical.graphconv_pairwise(
            gc_inputs, valid_edges, scope=gc_name, **gc_kwargs)

        if not concat_effs:
            parent_attrs = pred_attrs_unary + pred_attrs_binary
        else:
            parent_attrs = self.predDims.append_attr_to_vector('binary_attrs', pred_attrs_binary, pred_attrs_unary)

        return parent_attrs

    def format_output_nodes(self, **kwargs):

        kwargs.update(self.format_kwargs)
        if kwargs.get('keep_features', True):
            self.Dims.insert_from(self.predDims, position=-4)
            self.Dims = self.Dims.copy(suffix='_'+self.name)
            self.parent_nodes = tf.concat([
                self.parent_nodes[...,:-4],
                self.parent_attrs,
                self.parent_nodes[...,-4:]], axis=-1)
        else:
            keep_attrs = [k for k in self.Dims.sort().keys() if 'features' not in k]
            keep_nodes = self.Dims.get_tensor_from_attrs(self.parent_nodes, keep_attrs, stop_gradient=False, concat=True)
            self.Dims = self.Dims.copy(
                keys=keep_attrs, suffix='_'+self.name)
            self.Dims.insert_from(self.predDims.copy(suffix='_'+self.name), position=-4)
            self.parent_nodes = tf.concat([
                keep_nodes[...,:-4], self.parent_attrs, keep_nodes[...,-4:]], axis=-1)

        return self.parent_nodes

class DiffP1Level(P1Level):

    def __init__(self, name, input_name,
                 num_nodes, num_attrs,
                 affinity_kwargs={},
                 pooling_kwargs={},
                 aggregation_kwargs={},
                 mlp_kwargs={},
                 graphconv_kwargs=None,
                 format_kwargs={},
                 estimator_kwargs={},
                 **kwargs
                 ):

        super(DiffP1Level, self).__init__(
            name, input_name, num_nodes, num_attrs,
            affinity_kwargs=affinity_kwargs,
            pooling_kwargs=pooling_kwargs,
            aggregation_kwargs=aggregation_kwargs,
            mlp_kwargs=mlp_kwargs,
            graphconv_kwargs=graphconv_kwargs,
            format_kwargs=format_kwargs,
            **kwargs)

        # initialization for training a differentiable estimator
        self.estimator_kwargs = copy.deepcopy(estimator_kwargs)
        self.num_label_prop_runs = estimator_kwargs.get('num_lp_runs', 10)

    def reshape_inputs(self, **kwargs):

        super(DiffP1Level, self).reshape_inputs(**kwargs)
        CC = 2 if self.estimator_kwargs.get('add_coordinates', False) else 0
        gcn = GCN(input_dim=(self.C+CC), train=self.is_training, **self.estimator_kwargs)
        def gcn_model(nodes, local_adj):
            nodes = tf.stop_gradient(nodes)
            local_adj = tf.stop_gradient(local_adj)
            out = gcn(nodes, local_adj)
            return tf.nn.softmax(out, axis=-1)
        self.backbone = gcn_model

    def compute_sobel_edge_target(self, sobel_images_key='images',
                                  sobel_images_preproc=preproc_rgb,
                                  return_feats=False,
                                  sobel_edge_threshold=0.5, **kwargs):
        '''
        Compute edge labels by sobel filtering where 0 indicate an edge, 1 not an edge
        '''
        if sobel_images_key == 'features':
            feats = self.reshape_batch_time(self.features, merge=False)
            feats = tf.stop_gradient(feats)
            strides = [1,1]
        else:
            feats = self.reshape_batch_time(self.actions[sobel_images_key], merge=False)
            feats = sobel_images_preproc(feats)
            imsize = feats.shape.as_list()[-3:-1]
            strides = [imsize[0] // self.size[0], imsize[1] // self.size[1]]

        assert feats.dtype == tf.float32
        sobel_feats = compute_sobel_features(feats, size=None, to_mag=True) # magnitude_sum
        sobel_feats = sobel_feats[:,:,::strides[0],::strides[1]] # downsample
        if return_feats:
            return sobel_feats
        sobel_feats = sobel_feats[...,-1] / tf.cast(feats.shape.as_list()[-1], tf.float32) # average edge intensity
        edge_target = tf.cast(sobel_feats > sobel_edge_threshold, tf.float32)
        edge_target = self.reshape_batch_time(edge_target, merge=True) # [BT,H,W]
        edge_target = tf.Print(edge_target, [tf.reduce_mean(tf.reduce_sum(edge_target, axis=[1,2]))], message='sobel_edge_px')
        edge_target = 1.0 - edge_target # 0 is an edge

        return edge_target

    def compute_pseudo_edge_target(self, edge_k=1, edge_threshold=0.9, **kwargs):

        def compute_binary_boundary_map(labels):
            labels = tf.cast(labels, tf.float32)
            B, N = labels.shape.as_list()
            ksize = 2 * edge_k + 1
            H, W = self.size
            labels_grid = tf.reshape(labels, [B, H, W, 1])
            neighbors = graphical.compute_adjacency_from_features(labels_grid, k=edge_k, return_neighbors=True)  # [B, N, 1, 9]
            neighbors_grid = tf.reshape(neighbors, [B, H, W, ksize ** 2])
            not_equal = tf.logical_not(tf.equal(labels_grid, neighbors_grid))
            # not_equal = tf.cast(not_equal, tf.float32)
            # is_boundary = tf.reduce_sum(not_equal, axis=-1) > 0
            is_boundary = tf.reduce_any(not_equal, axis=-1)
            return tf.cast(is_boundary, tf.float32)

        sum_binary_boundary_map = 0.
        for i in range(self.num_label_prop_runs):
            labels, _ = pooling.compute_segments_by_label_prop(
                edges=self.edges, size=self.size, valid_nodes=self.valid_nodes, seed=i, **kwargs)
            sum_binary_boundary_map += compute_binary_boundary_map(labels)

        avg_binary_boundary_map = sum_binary_boundary_map / self.num_label_prop_runs
        edge_target = avg_binary_boundary_map < edge_threshold
        return tf.cast(edge_target, tf.float32)

    def encode(self):
        nodes = self.grouping_nodes
        C = self.C
        if self.estimator_kwargs.get('add_coordinates', False):
            B,H,W,C = nodes.shape.as_list()
            ones = tf.ones([B, H, W, 1], dtype=tf.float32)
            hc = tf.reshape(tf.range(H, dtype=tf.float32), [1, H, 1, 1]) * ones
            wc = tf.reshape(tf.range(W, dtype=tf.float32), [1, 1, W, 1]) * ones
            hc = (hc / ((H - 1.0) / 2.0)) - 1.0
            wc = (wc / ((W - 1.0) / 2.0)) - 1.0
            nodes = tf.concat([nodes, hc, wc], axis=3)
            C += 2
        nodes = tf.reshape(nodes, [self.BT, self.N, C])
        local_adj = self.edges
        # global_adj = graphical.local_to_global_adj(local_adj, self.size)
        # output = self.backbone(nodes, global_adj)
        output = self.backbone(nodes, local_adj)

        return output

    def compute_segment_label(self, x):
        B, N, D = x.shape.as_list()
        labels = tf.argmax(x, axis=-1)
        labels = tf.cast(labels, tf.int32)

        # relabel so that there are no skipped label values and they range from [0, NB) at most
        b_inds = tf.tile(tf.range(B, dtype=tf.int32)[:, tf.newaxis], [1, N])
        unique_labels = tf.scatter_nd(
            tf.stack([b_inds, labels], axis=-1), updates=tf.ones_like(labels), shape=[B, N])
        unique_labels = tf.minimum(unique_labels,
                                   tf.constant(1, tf.int32))  # [B,N] where 1 is where there's a valid label
        num_segments = tf.reduce_sum(unique_labels, axis=-1)  # [B]
        relabels = tf.cumsum(unique_labels, axis=-1, exclusive=True)

        # hash to reordered values and add offsets
        offsets = tf.cumsum(num_segments, exclusive=True)[:, tf.newaxis]  # [B,1]
        labels = tf.gather_nd(params=relabels, indices=tf.stack([b_inds, labels], axis=-1))  # [B,N]
        labels += offsets  # now unique label for every segment
        return labels, num_segments

    def compute_edge_probability(self, x, edge_k=1):
        # x should has shape [B, H, W, C]

        B, N, C = x.shape.as_list()
        H, W = self.size
        ksize = edge_k * 2 + 1
        x = tf.reshape(x, [B, H, W, C])
        neighbors = graphical.compute_adjacency_from_features(x, k=edge_k, return_neighbors=True)  # [B, N, C, ksize**2]
        neighbors = tf.reshape(neighbors, [B, H, W, C, ksize ** 2])
        x = tf.math.l2_normalize(x, axis=-1) # normalize before computing cosine similarity
        x = tf.reshape(x, [B, H, W, C, 1])
        neighbors = tf.math.l2_normalize(neighbors, axis=3)

        cosine_similarity = tf.reduce_sum((x * neighbors), axis=3) # [B, H, W, ksize**2]
        edge_prob = tf.reduce_mean(cosine_similarity, axis=-1) # now in range [-1., 1.]

        edge_prob = tf.Print(edge_prob, [tf.reduce_min(edge_prob), tf.reduce_max(edge_prob)], message='edge_prob_minmax')

        # put into range [0., 1.]
        edge_prob = 0.5 * (edge_prob + 1.0)
        edge_prob = tf.Print(edge_prob, [tf.reduce_min(edge_prob), tf.reduce_max(edge_prob)], message='edge_prob_minmax')

        return edge_prob

    def compute_edge_loss(self, edge_prob, edge_target):
        loss = cross_entropy(logits=edge_prob, labels=edge_target,
                             eps=self.estimator_kwargs.get('cross_entropy_epsilon', 1e-3),
                             keepdims=True)
        interior_mask = tf.pad(
            tf.ones(shape=self.size, dtype=tf.float32)[1:-1,1:-1],
            paddings=tf.constant([[1,1],[1,1]], dtype=tf.int32),
            constant_values=0.0
        )[tf.newaxis]

        if self.estimator_kwargs.get('focal_edge_loss', False):
            alpha = self.estimator_kwargs.get('alpha', 1.0)
            gamma = self.estimator_kwargs.get('gamma', 0.0)
            alpha_factor = edge_target * alpha + (1. - edge_target) * (1. - alpha)
            gamma_factor = tf.pow(1.0 - tf.exp(-loss), gamma)
            loss = alpha_factor * gamma_factor * loss

        loss = tf.reduce_mean(loss * interior_mask, axis=[1,2])
        return loss

    def compute_parents(self, **kwargs):
        '''
        Compute parent node assignments (i.e. parent edges) by labelprop on local edge matrix derived from image-like tensor
        '''
        kwargs.update(self.pooling_kwargs)

        # Create pseudo edge label from multiple LP runs
        pseudo_edge_target = self.compute_pseudo_edge_target(**self.estimator_kwargs)

        # obtain soft cluster assignment via GCN encoder
        output = self.encode()  # [B, H, W, D]

        # compute edge probability from labels
        edge_prob = self.compute_edge_probability(output)

        # compute binary CE loss between edge prob. and pseudo edge target
        self.loss = self.compute_edge_loss(edge_prob, pseudo_edge_target)

        # obtain discrete cluster assignment via argmax
        labels, num_segments = self.compute_segment_label(output)

        self.num_parents = num_segments  # [BT]

        return labels

class P0LocalLevel(P1Level):
    '''
    Produce affinities and edges and parent nodes in the same format as P1 affinities, but the affinities are just predicted by an MLP and so can be supervised with any ground truth or proxy signal
    '''

    def compute_affinities(self, **kwargs):
        '''
        Learn an MLP on node pairs that outputs affinity logits
        '''
        kwargs.update(self.affinity_kwargs)
        # get neighbors in k-radius window [BT,HW,C,ksize**2]
        neighbors = graphical.compute_adjacency_from_features(
            features=self.grouping_nodes, return_neighbors=True, **kwargs)
        F = neighbors.shape.as_list()[-1]
        base_nodes = tf.reshape(self.grouping_nodes, [self.BT,self.N,-1,1])
        if kwargs.get('symmetric', False):
            node_pairs = tf.abs(base_nodes - neighbors)
        else:
            node_pairs = tf.concat([
                tf.tile(base_nodes, [1,1,1,F]), neighbors], axis=-2)
        node_pairs = tf.transpose(node_pairs, [0,1,3,2]) # [BT,N,F,2C]

        mlp_name = self.name + '_affinity_mlp'
        kwargs['hidden_dims'] = kwargs.get('hidden_dims', []) + [1]
        affinities = convolutional.mlp(
            inp=node_pairs, scope=mlp_name, **kwargs)[...,0]

        return affinities

    def threshold_affinities(self, **kwargs):

        thresh = self.affinity_kwargs.get('edge_thresh', 0.5)
        edges = tf.nn.sigmoid(self.affinities) > thresh
        edges = tf.logical_and(
            edges, tf.logical_and(
                self.valid_nodes[...,tf.newaxis] > 0.5,
                self.valid_nodes[:,tf.newaxis,:] > 0.5))

        return edges

class DiffP0LocalLevel(DiffP1Level, P0LocalLevel):

    def compute_affinities(self, **kwargs):
        return P0LocalLevel.compute_affinities(self, **kwargs)

class P0GlobalLevel(P0LocalLevel):

    def reshape_inputs(self, **kwargs):
        self.N, self.C = self.nodes.shape.as_list()[1:3]
        self.M = self.num_nodes
        self.D = self.num_attrs
        self.size = None
        if self.inputDims is None:
            self.inputDims = DimensionDict(self.C)
            self.inputDims['valid'] = [-1,0,lambda v: tf.cast(v > 0.5, tf.float32)]

        # get valid nodes
        valid_key = [k for k in self.inputDims.keys() if 'valid' in k][0]
        self.valid_nodes = self.inputDims.get_tensor_from_attrs(
            self.nodes, valid_key, stop_gradient=True)[...,0] # [BT,N]

        # reshape segment ids
        shape = self.segment_ids.shape.as_list()
        assert len(shape) == 4, "Must pass in segment_ids of shape [B,T,H,W]"
        assert np.prod(shape[0:2]) == self.BT
        self.segment_ids = self.reshape_batch_time(self.segment_ids, merge=True)

        # reshape features
        if self.features is not None:
            if len(self.features.shape) > 5:
                self.features = self.features[:,:,self.t_vec]
            assert self.features.shape.as_list()[0:2] == [self.B, self.T]
            self.features = self.reshape_batch_time(self.features, merge=True)

        # deal w actions
        if self.actions is not None:
            self.actions = {ak: self.reshape_batch_time(act, merge=True)
                            for ak,act in self.actions.items()}

    def compute_affinities(self, nodes=None, nodes_other=None, **kwargs):
        '''
        Predict affinities between all node pairs
        '''
        kwargs.update(self.affinity_kwargs)
        nodes = tf.expand_dims(nodes if nodes is not None else self.nodes, axis=-2) # [BT,N,1,D]
        if nodes_other is None:
            nodes_other = tf.transpose(nodes, [0,2,1,3])
        else:
            nodes_other = tf.expand_dims(nodes_other, axis=-3)

        if kwargs.get('symmetric', False):
            node_pairs = tf.abs(nodes - nodes_other)
        elif kwargs.get('diff_inputs', False):
            node_pairs = nodes - nodes_other
        else:
            N = nodes.shape.as_list()[1]
            node_pairs = tf.concat([
                tf.tile(nodes, [1,1,N,1]),
                tf.tile(nodes_other, [1,N,1,1])], axis=-1) #[BT,N,N,2D]

        mlp_name = self.name + '_affinity_mlp'
        mlp_kwargs = copy.deepcopy(kwargs)
        mlp_kwargs['hidden_dims'] = mlp_kwargs.get('hidden_dims', []) + [1]
        affinities = convolutional.mlp(
            inp=node_pairs, scope=mlp_name, **mlp_kwargs)[...,0]

        return affinities

    def register_parents(self, **kwargs):
        '''
        Index into the new parent edges with the input segment ids
        '''
        assert self.segment_ids is not None, "Must provide a spatial registration of the input nodes"
        assert len(self.parent_edges.shape) == 1, "Parent edges a list"
        BT,H,W = self.segment_ids.shape.as_list()
        assert BT == self.BT

        monotonic_segment_ids = utils.get_monotonic_segment_ids(
            self.segment_ids, self.valid_nodes)
        parent_segment_ids = tf.gather_nd(
            params=tf.concat([self.parent_edges, tf.constant([-1], tf.int32)], axis=0),
            indices=monotonic_segment_ids[...,tf.newaxis])
        parent_segment_ids = utils.preproc_segment_ids(
            parent_segment_ids, Nmax=self.M, return_valid_segments=False)

        return parent_segment_ids

    def aggregate_nodes_and_features(self, **kwargs):
        '''
        Aggregate over nonspatially registered nodes and optional spatial registration
        '''
        kwargs.update(self.aggregation_kwargs)
        assert len(self.parent_edges.shape) == 1
        assert self.num_parents.shape.as_list() == [self.BT]


        # aggregate over child nodes
        parent_nodes, self.Dims = vectorizing.agg_child_attrs_within_parents(
            nodes=self.nodes, labels=self.parent_edges,
            num_segments=self.num_parents, valid_nodes=self.valid_nodes,
            max_labels=self.M, dimension_dict=self.inputDims, dim_suffix='_mean',
            rectangular_output=True, remove_valid=False, **kwargs)

        # remove old num_nodes attributes that shouldn't be used
        parent_nodes = tf.concat([
            parent_nodes[...,:-2], parent_nodes[...,-1:]], axis=-1)
        self.Dims.ndims -= 2
        self.Dims.delete('num_nodes')
        self.Dims.delete('valid')

        # add spatial attributes
        hw_key = [k for k in self.inputDims.keys() if 'hw_centroid' in k][0]
        if self.aggregation_kwargs.get('concat_new_features', False):
            features = tf.concat([self.features, self.new_features], axis=-1)
        else:
            features = self.features
        spatial = vectorizing.compute_attr_spatial_moments(
            self.nodes, self.parent_segment_ids, self.parent_edges,
            self.num_parents, valid_child_nodes=self.valid_nodes, features=features,
            nodes_dimension_dict=self.inputDims, labels_monotonic=True, remove_valid=True,
            max_parent_nodes=self.M, hw_attr=hw_key, **kwargs)
        spatial_attrs, hw_centroids, areas, self.parent_edges, SpaDims = spatial

        parent_nodes, valid_attr = tf.split(parent_nodes, [-1,1], axis=-1)
        if self.aggregation_kwargs.get('concat_spatial_attrs', False):
            parent_nodes = tf.concat([parent_nodes, spatial_attrs], axis=-1)
            self.Dims.insert_from(SpaDims)

        if self.aggregation_kwargs.get('concat_border_attrs', False):
            agg_feat_borders = kwargs.get('agg_feature_borders', False)
            border_attrs = vectorizing.compute_border_attributes(
                nodes=tf.concat([parent_nodes, hw_centroids, areas, valid_attr], axis=-1),
                segment_map=self.reshape_batch_time(self.parent_segment_ids, merge=False),
                features=(features if agg_feat_borders else None),
                hw_dims=[-4,-2], **kwargs)
            parent_nodes = tf.concat([parent_nodes, border_attrs], axis=-1)
            self.Dims['border_attrs'] = border_attrs.shape.as_list()[-1]

        parent_nodes = self.Dims.extend_vector([
            ('hw_centroids', hw_centroids),
            ('areas', areas),
            ('valid', valid_attr)], base_tensor=parent_nodes)

        # cleanup unwanted dims names
        for k in self.Dims.keys():
            if 'valid' in k:
                self.Dims.delete(k)
        self.Dims['valid'] = [-1,0]
        self.Dims.name_unassigned_dims(prefix='aggval')

        return parent_nodes

class P2Level(P0GlobalLevel):

    def __init__(self, name, input_name,
                 num_nodes, num_attrs,
                 vae_attrs=None,
                 vae_kwargs={
                     'encoder_dims': [50],
                     'latent_dims': 5,
                     'decoder_dims': [50],
                     'activations': tf.nn.elu,
                     'beta': 10.0
                 },
                 affinity_kwargs={'vae_kNN':1, 'vae_thresh': 2.5},
                 **kwargs
    ):
        super(P2Level, self).__init__(
            name, input_name, num_nodes, num_attrs,
            affinity_kwargs=affinity_kwargs,
            **kwargs)
        self.vae_attrs = vae_attrs
        self.affinity_kwargs['vae_kwargs'] = copy.deepcopy(vae_kwargs)

    @staticmethod
    def get_knn_node_pairs(nodes, dimension_dict, kNN=1, hw_attr='hw_centroids', **kwargs):
        '''
        Get the indices and vector for each node associated with its kNN in (h,w) space
        '''
        hw_key = [k for k in dimension_dict.keys() if hw_attr in k][-1]
        nodes_hw = dimension_dict.get_tensor_from_attrs(nodes, hw_key)
        valid_key = [k for k in dimension_dict.keys() if 'valid' in k][-1]
        nodes_valid = dimension_dict.get_tensor_from_attrs(nodes, valid_key)

        nodes_input = tf.concat([nodes_hw, nodes_valid], axis=-1)
        knn_inds = graphical.find_nearest_k_node_inds(
            nodes_input, kNN=kNN, nn_dims=[0,2])
        nodes_knn, valid_knn = graphical.attr_diffs_from_neighbor_inds(
            nodes=nodes, neighbor_inds=knn_inds, valid_nodes=nodes_valid,
            attr_dims_list=[[0,nodes.shape.as_list()[-1]]],
            attr_metrics_list=[lambda x,y: y], mask_self=True)
        nodes_knn = tf.concat(nodes_knn, axis=-1)

        return nodes_knn, valid_knn, knn_inds

    def kNN_to_rectangular_affinities(self, affinities, knn_inds, valid_nodes):
        BT,N,K = affinities.shape.as_list()
        assert knn_inds.shape.as_list()[-1] == K, (affinities, knn_inds)
        ones = tf.ones([BT,N,K], dtype=tf.int32)
        sc_inds = tf.stack([
            tf.reshape(tf.range(BT, dtype=tf.int32), [BT,1,1]) * ones,\
            tf.reshape(tf.range(N, dtype=tf.int32), [1,N,1]) * ones,\
            knn_inds], axis=-1) # [B,N,kNN,3]
        affinities = tf.scatter_nd(sc_inds, affinities, shape=[BT,N,N])
        affinities *= valid_nodes[...,tf.newaxis]*valid_nodes[:,tf.newaxis,:]
        return affinities

    def compute_affinities(self, nodes=None, valid_nodes=None, **kwargs):

        kwargs.update(self.affinity_kwargs)
        if self.vae_attrs is None:
            self.vae_attrs = [k for k in self.inputDims.keys() if 'vector' in k][-1:]
        vae_kwargs = kwargs['vae_kwargs']
        vae_kNN = kwargs.get('kNN_train', 1)
        kNN = kwargs.get('kNN', None) or self.N
        assert vae_kNN <= kNN, (vae_kNN, kNN)

        # get [BT,N,kNN,D] nearest neighbor nodes, valid, inds
        nodes = self.nodes if nodes is None else nodes
        nodes_knn, valid_knn, knn_inds = self.get_knn_node_pairs(
            nodes, dimension_dict=self.inputDims, kNN=kNN,
            hw_attr='hw_centroids')

        nodes = tf.expand_dims(nodes, axis=-2)
        nodes = self.inputDims.get_tensor_from_attr_dims(nodes, self.vae_attrs, stop_gradient=True)
        nodes_knn = self.inputDims.get_tensor_from_attr_dims(nodes_knn, self.vae_attrs, stop_gradient=True)

        if kwargs.get('symmetric', True):
            node_pairs = tf.stop_gradient(tf.abs(nodes - nodes_knn)) # [BT,N,kNN,D]
        elif kwargs.get('diff_inputs', False):
            node_pairs = tf.stop_gradient(nodes - nodes_knn)
        else:
            node_pairs = tf.concat([
                tf.tile(nodes, [1,1,kNN,1]),
                nodes_knn], axis=-1) #[BT,N,kNN,2D]
            node_pairs = tf.stop_gradient(node_pairs)

        print("VAE attrs", self.vae_attrs)
        print("P2 node pairs", node_pairs.shape.as_list())

        # Create VAE model
        vae_model = VAE(**vae_kwargs)

        # Create VAE inputs
        valid_nodes = self.valid_nodes if valid_nodes is None else valid_nodes
        valid_inds = (valid_nodes[...,tf.newaxis] * valid_knn[:,:,:vae_kNN,0]) > 0.5
        valid_inds = tf.cast(tf.where(valid_inds), tf.int32)
        valid_inds = tf.cast(valid_inds, tf.int32) # [?,3]
        vae_input_train = tf.gather_nd(node_pairs, valid_inds) # [?,D]
        if PRINT:
            vae_input_train = tf.Print(vae_input_train, [tf.shape(vae_input_train)], message='vae_input_shape')
        _, vae_loss = vae_model.predict(vae_input_train)
        self.loss += tf.reshape(vae_loss, [1]) # broadcast; will take mean anyway

        node_pairs_recon, _ = vae_model.predict(node_pairs)
        affinities = tf.sqrt(tf.reduce_sum(tf.square(
            node_pairs - node_pairs_recon), axis=-1) + kwargs.get('epsilon', 1.0e-3))
        vae_thresh = kwargs.get('vae_thresh', 2.5)
        vae_power = kwargs.get('vae_power', 1.0)
        affinities = 1. / (1. + tf.pow(affinities / vae_thresh, vae_power))
        affinities = tf.stop_gradient(affinities)

        if PRINT:
            affinities = tf.Print(affinities, [tf.reduce_mean(affinities), tf.reduce_mean(tf.cast(affinities > 0.5, tf.float32))], message='p2_affinities_inv')

        # scatter back to rectangular
        affinities = self.kNN_to_rectangular_affinities(affinities, knn_inds, valid_nodes)

        return affinities

    def threshold_affinities(self, **kwargs):

        thresh = self.affinity_kwargs.get('edge_thresh', 0.5)
        self.affinities *= self.valid_nodes[...,tf.newaxis] * self.valid_nodes[:,tf.newaxis,:]
        edges = self.affinities > thresh
        if self.affinity_kwargs.get('symmetric_output', True):
            edges = tf.logical_or(edges, tf.transpose(edges, [0,2,1]))

        if PRINT:
            num_valid = tf.reduce_sum(self.valid_nodes[0])
            valid_edges = self.valid_nodes[0][:,tf.newaxis] * self.valid_nodes[0][tf.newaxis,:]
            edges = tf.Print(edges, [tf.reduce_sum(
                tf.cast(edges, tf.float32)[0] * valid_edges), num_valid*num_valid],
                             message='vae_edges_max')
        return edges

class P2GeoLevel(P2Level):

    def __init__(self, name, input_name,
                 num_nodes, num_attrs,
                 geo_attrs=None, geo_metrics=None,
                 **kwargs
    ):
        super(P2GeoLevel, self).__init__(
            name, input_name, num_nodes, num_attrs,
            **kwargs)
        self.geo_attrs = geo_attrs
        if geo_metrics is None:
            self.geo_metrics = [lambda x,y: tf.abs(x-y)] * len(self.geo_attrs)
        else:
            assert len(geo_metrics) == len(self.geo_attrs), (geo_metrics, len(self.geo_attrs))
            self.geo_metrics = [
                met or (lambda x,y: tf.abs(x-y)) for met in geo_metrics
            ]

    def compute_affinities(self, **kwargs):
        kwargs.update(self.affinity_kwargs)

        if self.geo_attrs is None:
            self.geo_attrs = [k for k in self.inputDims.keys() if 'vector' in k][-1:]
        kNN = kwargs.get('kNN', None) or self.N
        nodes_knn, valid_knn, knn_inds = self.get_knn_node_pairs(
            self.nodes, dimension_dict=self.inputDims, kNN=kNN,
            hw_attr='hw_centroids')
        nodes = tf.expand_dims(self.nodes, axis=-2)
        nodes = self.inputDims.get_tensor_from_attr_dims(nodes, self.geo_attrs, stop_gradient=True, concat=False)
        nodes_knn = self.inputDims.get_tensor_from_attr_dims(nodes_knn, self.geo_attrs, stop_gradient=True, concat=False)
        node_pairs = tf.concat([
            met(nodes[i], nodes_knn[i]) for i,met in enumerate(self.geo_metrics)
        ], axis=-1)

        geo_thresh = kwargs.get('geo_thresh', 3.5)
        geo_power = kwargs.get('geo_power', 1.0)
        geo_affinities = tf.reduce_sum(node_pairs, axis=-1) # [B,N,K]
        geo_affinities = 1. / (1. + tf.pow(geo_affinities / geo_thresh, geo_power))

        geo_affinities = self.kNN_to_rectangular_affinities(
            geo_affinities, knn_inds, self.valid_nodes)
        geo_weight = np.minimum(1.0, kwargs.get('geo_weight', 0.5))
        if (1.0 - geo_weight) > 0.0:
            vae_affinities = super(P2GeoLevel, self).compute_affinities(**kwargs)
        else:
            vae_affinities = tf.zeros_like(geo_affinities)

        affinities = (geo_weight * geo_affinities) + (1.0 - geo_weight) * vae_affinities

        return affinities

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sess = tf.Session()

    SHAPE = [2,4,32,32,40]
    B,T,H,W,C = SHAPE
    N0 = H*W
    # make an image with four quadrants
    features = tf.reshape(tf.range(4, dtype=tf.float32), [1,2,2,1])
    features = tf.image.resize_images(features, size=SHAPE[2:4])
    features = tf.tile(features[:,tf.newaxis], SHAPE[0:2] + [1,1] + SHAPE[-1:])
    # features = tf.random.normal(SHAPE, dtype=tf.float32)
    input_nodes = tf.reshape(features, [B,T,N0,-1])

    # dummy projection matrix
    actions = {'projection_matrix': 1.95 * tf.eye(4, batch_shape=SHAPE[0:2], dtype=tf.float32)}

    DiffP1 = DiffP0LocalLevel(name="level1", input_name="level0", num_nodes=256, num_attrs=24,
                 # affinity_kwargs={'k': 2, 'symmetric': True, 'metric_kwargs': {'thresh': 'local'}},
                 affinity_kwargs={'k': 2, 'symmetric': True, 'hidden_dims':[100,100]},
                 pooling_kwargs={'num_steps': 20},
                 aggregation_kwargs={'agg_vars': True, 'concat_new_features': True},
                 mlp_kwargs={'activations': tf.nn.elu, 'hidden_dims': [100]},
                 graphconv_kwargs={'agg_type': 'mean', 'hw_thresh': 0.5, 'hidden_dims': [100], 'concat_effects': False},
                 format_kwargs={'keep_features': True, 'xyz_attr': True},
                 estimator_kwargs={'num_lp_runs': 10}
                 )
    outputs = DiffP1(input_nodes, features=features, actions=actions)
    print("Diff P1 outputs", outputs)
    import pdb
    pdb.set_trace()

    # P1 = P1Level(name="level1", input_name="level0", num_nodes=256, num_attrs=24,
    #              affinity_kwargs={'k': 3, 'symmetric': True, 'metric_kwargs': {'thresh': 'local'}},
    #              pooling_kwargs={'num_steps': 20},
    #              aggregation_kwargs={'agg_vars': True},
    #              mlp_kwargs={'activations': tf.nn.elu, 'hidden_dims': [100]},
    #              graphconv_kwargs={'agg_type':'mean', 'hw_thresh': 0.5, 'hidden_dims':[100],'concat_effects':False},
    #              format_kwargs={'keep_features':False, 'xyz_attr': True}
    # )
    # outputs = P1(input_nodes, features=features, actions=actions)
    # print("P1 outputs", outputs)

    # P0 = P0LocalLevel(name="level1", num_nodes=256, num_attrs=24,
    #              affinity_kwargs={'k': 3, 'symmetric': False, 'hidden_dims':[100,100]},
    #              pooling_kwargs={'num_steps': 20},
    #              aggregation_kwargs={'agg_vars': True},
    #              mlp_kwargs={'activations': tf.nn.elu, 'hidden_dims': [100]},
    #              graphconv_kwargs={'agg_type':'mean', 'hw_thresh': 0.5, 'hidden_dims':[100],'concat_effects':False},
    #              format_kwargs={'keep_features':False}
    # )

    # P0fc = P0GlobalLevel(name="level2", input_name="level1", num_nodes=64, num_attrs=36,
    #                  affinity_kwargs={'symmetric': False, 'hidden_dims':[100,100]},
    #                  pooling_kwargs={'num_steps': 20, 'tau':0.5},
    #                  aggregation_kwargs={'agg_vars': False, 'agg_ranges':False, 'concat_spatial_attrs':True, 'agg_spatial_vars':False, 'agg_features':True},
    #                  mlp_kwargs={'activations': tf.nn.elu, 'hidden_dims': [100]},
    #                  graphconv_kwargs={'agg_type':'mean', 'hw_thresh': None, 'hidden_dims':[100],'concat_effects':False},
    #                  format_kwargs={'keep_features':True, 'xyz_attr':False}
    # )
    # outputs = P0fc(outputs['parent_nodes'], input_segment_ids=outputs['parent_segment_ids'], inputDims=DiffP1.Dims, features=features, actions=actions)
    # print("P0fc outputs", outputs, P0fc.Dims, P0fc.Dims.ndims)
    # import pdb
    # pdb.set_trace()

    # sess.run(tf.global_variables_initializer())
    # outputs = sess.run(outputs)
    # print(outputs['parent_segment_ids'][-1,-1])
    # print(outputs['parent_edges'].min(axis=-1), outputs['parent_edges'].max(axis=-1))

    # print("hw", outputs['parent_nodes'][-1,-1,0:20,-4:-2])
    # print("av", outputs['parent_nodes'][-1,-1,0:20,-2:])
