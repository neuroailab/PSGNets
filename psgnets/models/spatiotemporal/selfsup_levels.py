import copy
1;95;0cimport pdb

import numpy as np
import tensorflow.compat.v1 as tf
import vvn.ops.convolutional as convolutional
import vvn.ops.graphical as graphical
import vvn.ops.pooling as pooling
import vvn.ops.vectorizing as vectorizing
import vvn.ops.rendering as rendering
import vvn.ops.utils as utils
import vvn.models.losses as losses
from vvn.ops.dimensions import DimensionDict, OrderedDict

from vvn.models.levels import *
from vvn.models.spatiotemporal.motion_levels import P3Level, EdgeFloodP3Level

class P4Level(P0GlobalLevel):

    def __init__(self, use_target_segments=False, vae_loss_scale=1., selfsup_loss_scale=1.,
                 static_attrs=None, estimator_kwargs={}, **kwargs):

        self.static_attrs = static_attrs
        self.use_target_segments = use_target_segments
        self.vae_loss_scale = vae_loss_scale
        self.selfsup_loss_scale = selfsup_loss_scale
        self.estimator_kwargs = copy.deepcopy(estimator_kwargs)
        P0GlobalLevel.__init__(self, **kwargs)
        with tf.variable_scope('motion_segment_estimator'):
            self.motion_segment_estimator = P3Level(
                name=self.name+'_estimator', input_name=self.input_name,
                num_nodes=self.num_nodes, num_attrs=0, **self.estimator_kwargs)

    def compute_segment_targets(self, **kwargs):
        '''
        Use the motion segment estimator to compute segments
        '''
        Est = self.motion_segment_estimator
        Est.nodes = self.nodes
        Est.valid_nodes = self.valid_nodes
        Est.segment_ids = self.segment_ids
        Est.features = self.features
        Est.actions = self.actions
        Est.inputDims = self.inputDims.copy(suffix='')
        Est.size = self.size
        Est.loss = tf.zeros_like(self.loss)
        Est.B, Est.T, Est.BT, Est.N, Est.M = [self.B, self.T, self.BT, self.N, self.M]
        with tf.variable_scope('motion_segment_estimator'):
            Est.affinities = Est.compute_affinities(nodes=Est.nodes)
            Est.edges = Est.threshold_affinities()
            Est.parent_edges = Est.compute_parents()
            motion_segment_ids = Est.register_parents()
            Est.loss = tf.Print(Est.loss, [tf.reduce_mean(Est.loss), tf.reduce_max(motion_segment_ids, axis=[-2,-1])], message='est_vae_loss')
            self.loss += Est.loss * self.vae_loss_scale

        return motion_segment_ids

    def compute_affinities(self, **kwargs):

        ## predict affinities w mlp
        kwargs.update(self.affinity_kwargs)
        if self.static_attrs is None:
            self.static_attrs = [k for k in self.inputDims.keys() if 'vector' in k][-1:]
        stop_grad = kwargs.get('stop_gradient', False)
        static_nodes = self.inputDims.get_tensor_from_attr_dims(self.nodes, self.static_attrs, stop_gradient=stop_grad)
        # affinities = super(P4Level, self).compute_affinities(nodes=static_nodes, **kwargs)
        affinities = P0GlobalLevel.compute_affinities(self, nodes=static_nodes, **kwargs)

        ## compute a loss
        self.target_segment_ids = self.compute_segment_targets()
        start_time = self.estimator_kwargs.get('loss_start_time', 1)
        selfsup_loss = losses.affinity_cross_entropy_from_nodes_and_segments(
            affinities=self.reshape_batch_time(affinities, merge=False),
            nodes=self.reshape_batch_time(self.nodes, merge=False),
            segments=self.reshape_batch_time(self.target_segment_ids[...,tf.newaxis], merge=False),
            dimension_dict=self.inputDims, size=self.target_segment_ids.shape.as_list()[1:3],
            start_time=start_time, downsample_labels=False)
        selfsup_loss = tf.concat([tf.zeros([self.B, start_time], dtype=tf.float32), selfsup_loss], axis=1)
        selfsup_loss = tf.Print(selfsup_loss, [tf.reduce_mean(selfsup_loss)], message='selfsulp_loss')
        self.loss += self.reshape_batch_time(selfsup_loss, merge=True) * self.selfsup_loss_scale

        return affinities

    def register_parents(self, **kwargs):
        if self.use_target_segments:
            self.parent_edges = self.motion_segment_estimator.parent_edges
        return P0GlobalLevel.register_parents(self, **kwargs)
