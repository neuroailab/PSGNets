import copy
import pdb

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

PRINT = False

class P3Level(P2Level):

    def compute_affinities(self, nodes=None, **kwargs):

        kwargs.update(self.affinity_kwargs)
        motion_nodes = self.motion_nodes if nodes is None else nodes

        # spatial
        with tf.variable_scope('space_affinities'):
            affinities = P2Level.compute_affinities(
                self, nodes=motion_nodes, valid_nodes=self.valid_nodes, **kwargs)

        # temporal
        time_kwargs = copy.deepcopy(kwargs)
        self.NT = self.N * self.T
        time_kwargs['kNN'] = self.NT
        motion_time_nodes = tf.reshape(
            self.reshape_batch_time(motion_nodes, merge=False),
            [self.B, self.T * self.N, -1])
        valid_time_nodes = tf.reshape(
            self.reshape_batch_time(self.valid_nodes, merge=False),
            [self.B, self.T * self.N])
        with tf.variable_scope('time_affinities'):
            time_affinities = P2Level.compute_affinities(
                self, nodes=motion_time_nodes, valid_nodes=valid_time_nodes, **time_kwargs)
            time_affinities *= valid_time_nodes[...,tf.newaxis] * valid_time_nodes[:,tf.newaxis,:]
            inds = tf.reshape(tf.range(self.NT, dtype=tf.int32), [1,-1,1])
            inds = tf.math.floordiv(inds, tf.constant(self.N, tf.int32))
            off_diag = tf.logical_not(tf.equal(inds, tf.transpose(inds, [0,2,1])))
            time_affinities *= tf.cast(off_diag, tf.float32)
            self.affinities_temporal = time_affinities
            self.edges_temporal = self.affinities_temporal > kwargs.get('thresh', 0.5)
        self.edges_temporal = time_affinities > kwargs.get('edge_thresh', 0.5)

        return affinities

    def compute_parents(self, **kwargs):

        kwargs.update(self.pooling_kwargs)

        spacetime_edges = self.combine_spacetime_affinities(
            self.reshape_batch_time(self.edges, merge=False), self.edges_temporal)
        spacetime_valid_nodes = tf.reshape(
            self.reshape_batch_time(self.valid_nodes, merge=False),
            [self.B, self.T*self.N])
        labels, num_segments = pooling.compute_segments_by_label_prop(
            edges=spacetime_edges, size=None,
            valid_nodes=spacetime_valid_nodes, **kwargs)

        # convert back to [BT] as leading dimension rather than [B]
        valid_inds = tf.where(spacetime_valid_nodes > 0.5) # [?,2]
        labels = tf.scatter_nd(
            indices=valid_inds,
            updates=labels,
            shape=[self.B, self.T*self.N]
        ) # 0 at invalid node positions
        offsets = tf.cumsum(num_segments, axis=0, exclusive=True) # [B]
        labels -= offsets[:,tf.newaxis] # now start at 0 per example

        # make each time point have distinct numbers, but node ids indicate time tracking

        num_segments = tf.reshape(tf.tile(num_segments[:,tf.newaxis], [1,self.T]), [self.BT])
        offsets = tf.cumsum(num_segments, axis=0, exclusive=True)
        labels = self.reshape_batch_time(
            tf.reshape(labels, [self.B, self.T, self.N]), merge=True) # [BT,N]
        labels += offsets[:,tf.newaxis]
        valid_inds = tf.where(tf.reshape(self.valid_nodes, [self.BT, self.N]) > 0.5) # [?,2]
        labels = tf.gather_nd(labels, valid_inds)

        self.num_parents = num_segments
        return labels

    @staticmethod
    def combine_spacetime_affinities(space_adj, time_adj):
        B,T,N,N = space_adj.shape.as_list()
        NT = N*T
        assert time_adj.shape.as_list() == [B,NT,NT], time_adj
        dtype = space_adj.dtype
        assert time_adj.dtype == dtype, (time_adj, dtype)
        space_adj = tf.tile(tf.reshape(space_adj, [B,NT,N]), [1,1,T])
        inds = tf.reshape(tf.range(NT, dtype=tf.int32), [1,-1,1])
        inds = tf.math.floordiv(inds, tf.constant(N, tf.int32))
        block_diag = tf.cast(tf.equal(inds, tf.transpose(inds, [0,2,1])), dtype)
        if dtype != tf.bool:
            spacetime_adj = (space_adj * block_diag) + (time_adj * (1. - block_diag))
        else:
            spacetime_adj = tf.logical_or(
                tf.logical_and(space_adj, block_diag),
                tf.logical_and(time_adj, tf.logical_not(block_diag))
            )

        return spacetime_adj
