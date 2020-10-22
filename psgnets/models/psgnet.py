from __future__ import division, print_function, absolute_import
from collections import OrderedDict
import os
import sys
import pdb

import numpy as np
import tensorflow.compat.v1 as tf
import copy

# Generic graphs
# from graph.common import Graph, propdict, graph_op, node_op, edge_op

# Visual Extraction and Graph Building
from vvn.ops.dimensions import DimensionDict
from vvn.ops import convolutional
from vvn.ops import pooling
from vvn.ops import vectorizing
from vvn.ops import graphical
from vvn.ops import utils
from .base import Model, Loss, Graph, propdict
from .preprocessing import *
from .extracting import *
from .levels import *
from .decoding import *
from .losses import *

# from vvn.models.convrnn.convrnn_model import ConvRNN

## for debugging
from vvn.data.tdw_data import TdwSequenceDataProvider
from vvn.data.utils import *

class PSGNet(Model):

    def __init__(
            self,
            preprocessor=(Preprocessor, {}),
            extractor=(Extractor, {}),
            graph_levels=[(P1Level, {})],
            decoders=[(Decoder, {})],
            losses=[(Loss, {})],
            vectorize_nodes=False,
            **model_params
    ):
        self.num_models = 0
        self.Preproc = self.init_model(Preprocessor, preprocessor)
        self.Extract = self.init_model(Extractor, extractor)

        # set graph levels
        self.num_levels = 0
        self.Levels = [BaseLevel(name='level0', input_name='base_tensor')]
        for level in graph_levels:
            self.num_levels += 1
            level_cls, level_params = level
            level_params['name'] = level_params.get('name', 'level'+str(self.num_levels))
            level_params['input_name'] = level_params.get('input_name', 'level'+str(self.num_levels-1))
            assert level_params['name'] not in [lev.name for lev in self.Levels], "graph level name %s is not unique" % level_params['name']
            assert level_params['input_name'] in [lev.name for lev in self.Levels], "graph level %s must have input_name set as a previously initialized graph level in %s, but is %s" % (level_params['name'], [lev.name for lev in self.Levels], level_params['input_name'])
            graph_level = level_cls(**level_params)
            self.Levels.append(graph_level)

        # set decoders
        self.Decode = OrderedDict()
        self.num_decoders = self.num_models
        for decoder in decoders:
            DecoderNew = self.init_model(Decoder, decoder)
            self.Decode[DecoderNew.name] = DecoderNew
        self.decoders = self.Decode.keys()
        self.num_decoders = self.num_models - self.num_decoders
        assert self.num_decoders == len(set(self.decoders)), "Decoders must have unique names"

        # set losses
        self.Losses = OrderedDict()
        self.num_losses = self.num_models
        for loss in losses:
            LossNew = self.init_model(Loss, loss)
            self.Losses[LossNew.name] = LossNew
            print(self.num_models, LossNew.name)
        self.loss_names = self.Losses.keys()
        self.num_losses = self.num_models - self.num_losses
        assert self.num_losses == len(set(self.loss_names)), (self.num_losses, self.loss_names)

        # init values
        self.features = {}
        self.psg = {'nodes': {}, 'edges': {}, 'spatial': {}, 'dims': {}}

        self.vectorize_nodes = vectorize_nodes
        super(PSGNet, self).__init__(**model_params)

    def init_model(self, model_class, params):
        if isinstance(params, dict):
            name = params.get('name', type(model_class).__name__ + '_' + str(self.num_models))
            params['name'] = name
            self.num_models += 1
            return model_class(**params)
        elif isinstance(params, (tuple, list)):
            assert len(params) == 2
            name = params[1].get('name', type(params[0]).__name__ + '_' + str(self.num_models))
            params[1]['name'] = name
            self.num_models += 1
            return params[0](**params[1])

    def update_shapes(self, tensor):
        B,T,H,W,C = tensor.shape.as_list()
        BT,HW,_,R = utils.dims_and_rank(tensor)
        self.B,self.T,self.H,self.W,self.C = B,T,H,W,C
        self.BT,self.HW,self.R = BT,HW,R

    def set_batch_time(self, model):
        assert isinstance(model, Model), type(model)
        model.B = self.B
        model.T = self.T
        model.BT = self.BT

    def preprocess_inputs(self, inputs, train_targets, inp_sequence_len, scope='Input', **kwargs):

        with tf.variable_scope(scope):
            self.inputs = inputs
            self.labels = {k: inputs.get(k, None) for k in train_targets}
            input_tensor = self.Preproc(inputs, self.is_training, **kwargs)
            self.inputDims = self.Preproc.Dims
            self.input_sequence_len = inp_sequence_len or input_tensor.shape.as_list()[1]
            input_tensor = input_tensor[:,:self.input_sequence_len]
            self.update_shapes(input_tensor)
        return input_tensor

    def extract_features(self, input_tensor, scope='Extract', **kwargs):

        with tf.variable_scope(scope):
            base_tensor = self.Extract(input_tensor, self.is_training, **kwargs)
            self.features = self.Extract.outputs
            self.Tb, self.Hb, self.Wb, self.Cb = base_tensor.shape.as_list()[-4:]
            self.baseDims = DimensionDict({'features':self.Cb})

        return base_tensor

    def build_psg(self, base_tensor, scope='GraphBuild', **kwargs):

        # initialize psg
        psg = Graph()
        BaseLevel = self.Levels[0]
        outputs = {BaseLevel.name: BaseLevel(self.base_tensor)}
        psg.spatial = OrderedDict([(BaseLevel.name, outputs[BaseLevel.name]['parent_segment_ids'])])
        psg.dims = OrderedDict([(BaseLevel.name, BaseLevel.Dims)])
        psg.losses = OrderedDict()
        _base_nodes = outputs[BaseLevel.name]['parent_nodes']
        if len(_base_nodes.shape) > 4:
            base_nodes = _base_nodes[:,:,-1] # take last convrnn pass
        else:
            base_nodes = _base_nodes
        psg.nodes[BaseLevel.name] = propdict(
            BaseLevel.Dims.get_tensor_from_attrs(
                base_nodes, 'vector', concat=False))

        # pass nodes through each level
        for Level in self.Levels[1:]:
            # feed outputs of a previous level into the current one
            inp_nodes = outputs[Level.input_name]['parent_nodes']
            inp_segments = outputs[Level.input_name]['parent_segment_ids']
            inp_dims = psg.dims[Level.input_name]

            # if nodes were overwritten by a different level
            if inp_nodes.shape.as_list()[-1] != inp_dims.ndims:
                inp_nodes = psg.nodes[Level.input_name]['vector']

            assert inp_dims.ndims == inp_nodes.shape.as_list()[-1], (inp_dims, inp_dims.ndims, inp_nodes)

            # update actions with new features
            new_fts_key = Level.input_name + '_new_features'
            self.actions[new_fts_key] = outputs[Level.input_name].get('new_features', None)
            # run the level
            outputs[Level.name] = Level(
                inp_nodes, input_segment_ids=inp_segments,
                features=self.base_tensor,
                inputDims=inp_dims,
                actions=self.actions,
                **kwargs
            )
            Level.Dims['vector'] = [0,Level.Dims.ndims]

            # update psg dims and nodes and registrations and losses
            psg.dims[Level.name] = Level.Dims
            psg.spatial[Level.name] = outputs[Level.name]['parent_segment_ids']
            psg.losses[Level.name] = outputs[Level.name]['loss']
            psg.nodes[Level.name] = propdict(
                Level.Dims.get_tensor_from_attrs(
                    outputs[Level.name]['parent_nodes'],
                    Level.Dims.keys() if self.vectorize_nodes else 'vector',
                    concat=False))

            # if children were changed, update
            changed = (outputs[Level.name]['child_nodes'].shape.as_list() != outputs[Level.input_name]['parent_nodes'].shape.as_list())
            if changed:
                psg.nodes.pop(Level.input_name)
                psg.nodes[Level.input_name] = propdict(
                    Level.inputDims.get_tensor_from_attrs(
                        outputs[Level.name]['child_nodes'],
                        Level.inputDims.keys() if self.vectorize_nodes else 'vector',
                        concat=False
                    )
                )

            # update psg edges
            c2p_key = 'child_to_parent_edges_%s_to_%s' % (Level.input_name, Level.name)
            valid_children = tf.cast(outputs[Level.name]['child_valid'], tf.bool)
            c2p_edges = graphical.add_batch_time_node_index(
                outputs[Level.name]['parent_edges'])
            # c2p_edges_list = tf.gather_nd(c2p_edges, tf.cast(tf.where(valid_children), tf.int32))
            psg.edges[c2p_key] = propdict({
                'layer': np.array([Level.input_name, Level.name]).reshape([1,1,1,2]),
                'idx': c2p_edges
            })

            # update within-level affinities
            aff_key = 'within_' + Level.input_name + '_for_' + Level.name
            psg.edges[aff_key] = propdict({
                'layer': np.array([Level.input_name]*2).reshape([1,1,1,2]),
                'affinities': outputs[Level.name]['affinities'],
                'adjacency': outputs[Level.name]['within_edges']
            })

        if not self.is_training:
            self.outputs.update(outputs)

        # remove ConvRNN pass dimension
        if len(self.base_tensor.shape) > 5:
            self.base_tensor = self.base_tensor[:,:,-1]
            try:
                self.features['outputs'] = self.features['outputs'][:,:,-1]
            except KeyError:
                print("no outputs features")

        return psg

    def flatten_features_nodes_edges(self, rename=True):
        all_outputs = {}
        def _rename(prefix, key):
            return prefix + '/' + str(key) if rename else key
        all_outputs.update({_rename('features', layer): self.features[layer]
                            for layer in self.features.keys()})
        all_outputs.update({_rename('nodes', level): self.psg.nodes[level]
                            for level in self.psg.nodes.keys()})
        all_outputs.update({_rename('edges', edge_set): self.psg.edges[edge_set]
                            for edge_set in self.psg.edges.keys()})
        all_outputs.update({_rename('spatial',level+'_segments'): self.psg.spatial[level]
                            for level in self.psg.spatial.keys()})
        all_outputs.update({_rename('dims',level+'_dims'): self.psg.dims[level]
                            for level in self.psg.dims.keys()})
        all_outputs.update({_rename('losses',level+'_loss'): self.psg.losses[level]
                            for level in self.psg.losses.keys()})
        all_outputs.update({_rename('inputs', key): self.inputs[key]
                            for key in self.inputs.keys()})

        all_outputs.update({
            'sizes/input': [self.H, self.W],
            'sizes/base_tensor': [self.Hb,self.Wb]
        })

        return all_outputs

    def decode_from(self, name, scope='Decode', **kwargs):

        with tf.variable_scope(scope):
            decode_params = copy.deepcopy(kwargs)
            decode_params.update(self.Decode[name].params)
            self.set_batch_time(self.Decode[name])
            decoded_outputs = self.Decode[name](
                self.decoder_inputs, train=self.is_training, **decode_params)
        return decoded_outputs

    def compute_loss(self, name, logits_mapping=None, labels_mapping=None, **kwargs):
        Loss = self.Losses[name]
        loss_params = copy.deepcopy(kwargs)
        loss_params.update(Loss.params)
        if logits_mapping is not None:
            loss_params['logits_mapping'] = logits_mapping
        if labels_mapping is not None:
            loss_params['labels_mapping'] = labels_mapping

        to_decode = set(Loss.required_decoders + [logits_nm.split('/')[0] for logits_nm in loss_params['logits_mapping'].values()])

        # compute required outputs
        for decoder in to_decode:
            if decoder not in [out_nm.split('/')[0] for out_nm in self.outputs.keys()]:
                decode_kwargs = copy.deepcopy(kwargs)
                decode_kwargs['name'] = decoder
                try:
                    self.outputs.update(self.decode_from(**decode_kwargs))
                except KeyError:
                    self.outputs.update({k:self.decoder_inputs[k] for k in self.decoder_inputs.keys()
                                         if decoder in k})

        # get required labels
        self.labels['inputs'] = self.inputs
        for label in loss_params['labels_mapping'].values():
            if label not in self.labels.keys():
                try:
                    self.labels[label] = self.decoder_inputs[label]
                except KeyError:
                    try:
                        self.labels[label] = self.outputs[label]
                    except KeyError:
                        self.labels[label] = self.inputs[label]

        # get valid
        if loss_params.get('valid_logits_key', None) is not None:
            valid_logits = self.outputs[loss_params['valid_logits_key']]
        else:
            valid_logits = None

        if loss_params.get('valid_labels_key', None) is not None:
            raise NotImplementedError("Fetch valid labels from outputs or labels")

        # compute the loss
        loss_here = Loss(self.outputs, self.labels, valid_logits=valid_logits, **loss_params)

        return loss_here

    def build_model(self, **model_params):

        def model(inputs, train=True, train_targets=[], action_keys=[], inp_sequence_len=None, to_decode=None, losses_now=None, rename=True, **kwargs):

            self.is_training = train
            self.outputs = {}
            self.actions = {k:inputs.get(k, None) for k in action_keys}
            self.input_tensor = self.preprocess_inputs(inputs, train_targets, inp_sequence_len)
            self.base_tensor = self.extract_features(self.input_tensor)
            with tf.variable_scope("GraphBuild"):
                self.psg = self.build_psg(self.base_tensor)

            # compute outputs necessay for losses
            self.decoder_inputs = self.flatten_features_nodes_edges(rename=rename)
            if train:
                losses = {}
                if losses_now is None:
                    losses_now = [{'name': loss_name} for loss_name in self.loss_names]
                for loss_params in losses_now:
                    losses.update(self.compute_loss(**loss_params))
                self.losses = losses
                return losses, self.params
            else:
                outputs = {}

                # might need decoder inputs
                outputs.update(self.decoder_inputs)

                if to_decode is None:
                    to_decode = [{'name': decoder_name} for decoder_name in self.decoders]

                for decoder_params in to_decode:
                    outputs.update(self.decode_from(**decoder_params))

                # things of type tf.String interfere w multigpu agg
                for e in [k for k in outputs.keys() if 'edge' in k]:
                    _ = outputs[e].pop('layer', None)

                self.outputs.update(outputs)
                self.losses = {}
                return self.outputs, self.params

        self.model_func = model

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    SHAPE = [1,4,64,64]
    B,T,H,W = SHAPE
    TRAIN = True

    from resnets.resnet_model import resnet_v2
    resnet_18 = resnet_v2(18, get_features=True)

    # inputs = {
    #     'images': tf.random.normal(SHAPE + [3], dtype=tf.float32),
    #     'depths': tf.random.normal(SHAPE + [1], dtype=tf.float32),
    #     'normals': tf.random.normal(SHAPE + [3], dtype=tf.float32),
    #     'objects': tf.random.uniform(SHAPE + [1], minval=0, maxval=16, dtype=tf.int32),
    #     'valid': tf.ones(SHAPE + [1], dtype=tf.bool),
    #     'labels': tf.random.uniform([B,T], minval=0, maxval=1001, dtype=tf.int32)
    # }

    # # add diff images
    # inputs['diff_images'] = utils.image_time_derivative(inputs['images'])
    # inputs['hw'] = utils.coordinate_ims(B,T,[H,W])

    # M = PSGNet(
    #     preprocessor=(Preprocessor, {
    #         'model_func': preproc_tensors_by_name, 'name': 'Preproc',
    #         'dimension_order': ['images', 'depths', 'normals', 'diff_images', 'hw'],
    #         'dimension_preprocs': {'images': preproc_rgb}
    #     }),
    #     extractor=(ConvRNN, {
    #         'name': '05LConvRNNegu',
    #         'base_config': './convrnn/base_configs/enetB0like64_nobn_05Lscene_nosh',
    #         'cell_config': './convrnn/cell_configs/egu_05L_config',
    #         'feedback_edges': [('conv2', 'conv1')], 'base_tensor_name': 'conv1',
    #         'time_dilation': 3
    #     }),
    #     # extractor=(Extractor, {
    #     #     'model_func': resnet_18, 'name': 'ResNet18', 'layer_names': ['block'+str(i) for i in range(5)]+['pool'], 'base_tensor_name': 'block0',
    #         # 'model_func': convolutional.convnet_stem, 'name': 'ConvNet', 'layer_names': ['conv'+str(i) for i in range(5)],
    #         # 'ksize': 7, 'conv_kwargs': {'activation': 'relu'}, 'max_pool': True,
    #         # 'hidden_ksizes': [3,3,3,3], 'hidden_channels': [40,80,120,240], 'out_channels': 1024, 'base_tensor_name': 'conv0'
    #     # }),
    #     graph_levels=[
    #         (DiffP0LocalLevel, {'num_nodes':128, 'num_attrs':24,
    #                    'affinity_kwargs': {'k':3, 'symmetric':False, 'metric_kwargs': {'thresh':'local'}, 'hidden_dims':[100,100]},
    #                    'pooling_kwargs': {'num_steps':20},
    #                    'aggregation_kwargs': {},
    #                    'mlp_kwargs': {'hidden_dims': [100]},
    #                    'graphconv_kwargs': {
    #                        'agg_type':'mean',
    #                        'hw_thresh': 0.5,
    #                        'hidden_dims': [100],
    #                        'concat_effects': False
    #                    },
    #                    'format_kwargs': {'keep_features': False, 'xyz_attr': True}
    #         }),
    #         # (P0GlobalLevel, {'num_nodes':64, 'num_attrs':36,
    #         #                  'affinity_kwargs': {'symmetric': True, 'hidden_dims': [100,100]},
    #         #                  'pooling_kwargs': {'num_steps': 20, 'tau':0.5},
    #         #                  'aggregation_kwargs': {
    #         #                      'agg_vars':False, 'agg_spatial_vars':False,
    #         #                      'agg_features': True, 'concat_spatial_attrs':True
    #         #                  },
    #         #                  'mlp_kwargs': {'hidden_dims': [100]},
    #         #                  'graphconv_kwargs': {
    #         #                      'agg_type':'mean', 'hidden_dims': [100]
    #         #                  },
    #         #                  'format_kwargs': {'keep_features': True, 'xyz_attr':False}
    #         # })
    #         (P2Level, {'num_nodes':64, 'num_attrs':36,
    #                    'affinity_kwargs': {'symmetric': True},
    #                    'vae_kwargs': {'encoder_dims': [50,50],
    #                                   'latent_dims': 5,
    #                                   'decoder_dims': [50,50],
    #                                   'activations': tf.nn.elu,
    #                                   'beta': 10.0},
    #                          'pooling_kwargs': {'num_steps': 20, 'tau':0.0},
    #                          'aggregation_kwargs': {
    #                              'agg_vars':False, 'agg_spatial_vars':False,
    #                              'agg_features': True, 'concat_spatial_attrs':True
    #                          },
    #                          'mlp_kwargs': {'hidden_dims': [100]},
    #                          'graphconv_kwargs': {
    #                              'agg_type':'mean', 'hidden_dims': [100]
    #                          },
    #                          'format_kwargs': {'keep_features': True, 'xyz_attr':False}
    #         })
    #     ],
    #     decoders=[
    #         # (Decoder, {'name': 'avg_pool', 'model_func': convolutional.global_pool, 'kind': 'avg', 'keep_dims': True, 'input_mapping':{'inputs':'features/pool'}}),
    #         # (Decoder, {'name': 'classifier', 'model_func': convolutional.fc, 'out_depth': 1000, 'input_mapping': {'inputs':'features/pool'}}),
    #         (QtrDecoder, {
    #             'name': 'qtr_level2', 'input_mapping': {
    #                 'nodes': 'nodes/level2',
    #                 'segment_ids': 'spatial/level2_segments',
    #                 'dimension_dict': 'dims/level2_dims'
    #             },
    #             'latent_vector_key': 'unary_attrs',
    #             'hw_attr': 'hw_centroids', 'num_sample_points': 1024,
    #             'method': 'linear'
    #         }),
    #         (DeltaImages, {
    #             'name': 'deltas', 'input_mapping': {
    #                 'images': 'inputs/images'},
    #             'thresh': 0.01
    #         })
    #     ],
    #     losses=[
    #         (Loss, {'name': 'qtr_loss', 'required_decoders': ['qtr_level2'],
    #                 'loss_func': rendered_attrs_images_loss,
    #                 'logits_mapping': {
    #                     'pred_attrs': 'qtr_level2/sampled_pred_attrs',
    #                     'valid_attrs': 'qtr_level2/sampled_valid_attrs',
    #                     'spatial_inds': 'qtr_level2/sampled_hw_inds',
    #                     'size': 'sizes/base_tensor'
    #                 },
    #                 'labels_mapping': {
    #                     'labels': 'inputs',
    #                     'valid_images': 'valid'
    #                 },
    #                 'image_preprocs': {
    #                     'images': lambda im: tf.image.rgb_to_hsv(preproc_rgb(im))
    #                 },
    #                 'loss_per_point_funcs': {
    #                     'images': l2_loss
    #                 },
    #                 'loss_scales': {'images': 10.0}
    #         }),
    #         # (Loss, {'name': 'classification', 'required_decoders': ['classifier'], 'scale':1., 'loss_func':tf.nn.sparse_softmax_cross_entropy_with_logits, 'logits_keys': ['logits'], 'labels_keys': ['labels']}),
    #         # (Loss, {'name': 'L2', 'loss_func': l2_loss, 'scale':1.0}),
    #         (Loss, {'name': 'edges_level0', 'loss_func': affinity_cross_entropy_from_nodes_and_segments,
    #                 'logits_mapping': {
    #                     'affinities': 'edges/within_level0_for_level1',
    #                     'nodes': 'nodes/level0',
    #                     'dimension_dict': 'dims/level0_dims',
    #                     'size': 'sizes/base_tensor'
    #                 },
    #                 'labels_mapping': {'segments': 'objects'},
    #                 'start_time':0, 'size': None, 'scale': 1.0
    #         }),
    #         (Loss, {'name': 'edges_level1', 'loss_func': affinity_cross_entropy_from_nodes_and_segments,
    #                 'logits_mapping': {
    #                     'affinities': 'edges/within_level1_for_level2',
    #                     'nodes': 'nodes/level1',
    #                     'dimension_dict': 'dims/level1_dims',
    #                     'size': 'sizes/base_tensor'
    #                 },
    #                 'labels_mapping': {'segments': 'objects'},
    #                 'start_time':0, 'scale': 1.0
    #         }),
    #         (Loss, {'name': 'level2', 'loss_func': None,
    #                 'logits_mapping': {'logits': 'losses/level2_loss'},
    #                 'labels_mapping': {},
    #                 'scale': 1.0
    #         }),
    #         (Loss, {'name': 'level1', 'loss_func': None,
    #                 'logits_mapping': {'logits': 'losses/level1_loss'},
    #                 'labels_mapping': {},
    #                 'scale': 1.0
    #         }),
    #     ],
    #     vectorize_nodes=False
    # )

    # losses = [
    #     # {'name': 'classification', 'logits_mapping': {'logits': 'classifier/outputs'}, 'labels_mapping': {'labels': 'labels'}},
    #     # {'name': 'L2', 'logits_mapping': {'logits': 'avg_pool/outputs'}, 'labels_mapping': {'labels': 'avg_pool/outputs'}},
    #     {'name': 'edges_level0'}, {'name': 'edges_level1'}, {'name': 'level2'}, {'name': 'level1'}, {'name': 'qtr_loss'}
    # ]

    # print("outputs", M(inputs, train=True, train_targets=['objects', 'labels', 'images', 'depths', 'normals', 'valid'], losses_now=losses, rename=True))
    # # print("psg ndims", [D.ndims for D in M.psg.dims.values()])
    # # print("psg nodes", M.psg.nodes)
    # # print("psg edges", M.psg.edges)
    # # print("features", M.features)
    # import pdb
    # pdb.set_trace()

    # # sess = tf.Session()
    # # sess.run(tf.global_variables_initializer())
    # # losses = sess.run(M.losses)
    # # print(losses)
    # # edge_loss = affinity_cross_entropy_from_nodes_and_segments(
    # #     M.decoder_inputs['edges/within_level1_for_level2'], M.decoder_inputs['nodes/level1'], M.inputs['objects'],
    # #     dimension_dict=M.psg.dims['level1'])
    # # pdb.set_trace()
