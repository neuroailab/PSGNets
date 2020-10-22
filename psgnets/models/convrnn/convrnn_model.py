from __future__ import division, print_function, absolute_import
from collections import OrderedDict
import os
import sys
import pdb

import numpy as np
import tensorflow.compat.v1 as tf
import copy

import vvn.ops.utils as utils
from vvn.models.extracting import Extractor

import tnn.main
import tnn.cell as cell
from tnn.efficientgaternn import tnn_EfficientGateCell, ConvRNNCell
import json

class ConvRNN(Extractor):

    def __init__(self,
                 name,
                 base_config,
                 cell_config,
                 cell_type=tnn_EfficientGateCell,
                 feedback_edges=[],
                 time_dilation=1,
                 num_temporal_splits=None,
                 ntimes=None,
                 output_times=None,
                 input_layer_name='conv0',
                 base_tensor_name='conv1',
                 layer_names=['conv2'],
                 **model_params
    ):
        self.name = name or type(self).__name__
        self.base_json = base_config
        if '.json' not in self.base_json:
            self.base_json += '.json'
        self.cell_type = cell_type
        if isinstance(cell_config, str):
            self.cell_config = self.load_cell_config(cell_config)
        else:
            self.cell_config = cell_config
        self.convrnn_cells = copy.deepcopy(self.cell_config)
        self.feedback_edges = feedback_edges
        self.time_dilation = time_dilation
        self.num_temporal_splits = num_temporal_splits
        self.ntimes = ntimes
        self.output_times = output_times

        self.input_layer_name = input_layer_name

        super(ConvRNN, self).__init__(
            name=self.name, model_func=None, base_tensor_name=base_tensor_name,
            layer_names=layer_names, time_shared=False, **model_params)

    def load_cell_config(self, path):

        if '.py' != path [-3:]:
            path += '.py'

        import imp
        config = imp.load_source('config', path).config
        return config

    def build_model(self, func=None, trainable=True):

        # build a NetworkX graph from a json config
        G_tnn = tnn.main.graph_from_json(self.base_json)

        # get out depths
        self.out_depths = {}
        for layer, attr in G_tnn.nodes(data=True):
            if len(self.convrnn_cells.get(layer, {}).keys()):
                self.out_depths[layer] = attr['kwargs']['memory'][1]['out_depth']

        # input layers
        if not isinstance(self.input_layer_name, list):
            input_layers = [self.input_layer_name]

        def convrnn(inputs, train, **kwargs):
            call_params = kwargs
            call_params.update(self.params)

            # Loop over the nodes and set their attributes
            for layer, attr in G_tnn.nodes(data=True):
                cell_params = copy.deepcopy(self.convrnn_cells.get(layer, {}))
                ## change input shape
                if layer in input_layers:
                    attr['shape'] = inputs.shape.as_list()[-3:]

                if len(cell_params.keys()):
                    attr['cell'] = self.cell_type
                    cell_params['out_depth'] = self.out_depths[layer]
                    layer_params = {
                        'time_sep': True,
                        'convrnn_cell': self.cell_type.mro()[1].__name__, # hack
                        'convrnn_cell_kwargs': copy.deepcopy(cell_params)
                    }
                    attr['kwargs']['memory'] = (cell.memory, layer_params)
                else:
                    attr['cell'] = self.cell_type
                    layer_params = {'convrnn_cell': None}
                    attr['kwargs']['memory'] = (cell.memory, layer_params)

            if len(self.feedback_edges):
                G_tnn.add_edges_from(self.feedback_edges)

            for layer, attr in G_tnn.nodes(data=True):
                if isinstance(attr['cell'], self.cell_type):
                    attr['kwargs']['memory'][1]['is_training'] = train

            # inputs = self.reshape_batch_time(inputs, merge=False)
            self.num_temporal_splits = self.num_temporal_splits or self.T
            self.ntimes = self.ntimes or (self.time_dilation)
            self.output_times = self.output_times or range(self.time_dilation)
            input_temporal_list = self.movie_to_temporal_list(inputs)

            tnn_inputs = {input_layers[0]: input_temporal_list}
            tnn.main.init_nodes(G_tnn, input_nodes=input_layers, batch_size=self.B*self.num_temporal_splits, channel_op='concat')

            # unroll the tnn model
            if call_params.get('unroll_tf', True):
                tnn.main.unroll_tf(G_tnn, input_seq=tnn_inputs, ntimes=self.ntimes, ff_order=call_params.get('ff_order', None))
            else:
                tnn.main.unroll(G_tnn, input_seq=tnn_inputs, ntimes=self.ntimes)
            # get the outputs
            output_key = self.params.get('base_tensor_outputs', 'states')
            base_tensor = G_tnn.node[self.base_nm][output_key]
            if isinstance(base_tensor[0], dict):
                base_tensor = [bt['convrnn_cell_output'] for bt in base_tensor]
            base_tensor = tf.stack([base_tensor[t] for t in self.output_times], axis=1)
            endpoints = {
                nm: tf.stack([G_tnn.node[nm]['outputs'][t] for t in self.output_times], axis=1) for nm in self.layer_names}

            base_tensor = tf.reshape(base_tensor, [self.B, self.T, len(self.output_times)] + base_tensor.shape.as_list()[2:])
            endpoints = {
                nm: tf.reshape(out[:,-1], [self.B, self.T] + out.shape.as_list()[2:])
                for nm,out in endpoints.items()
            }

            return base_tensor, endpoints

        self.model_func = convrnn
        self.model_func_name = convrnn.__name__

    def movie_to_temporal_list(self, image_tensor):

        B,T,H,W,C = image_tensor.shape.as_list()
        if self.num_temporal_splits > 1:
            image_tensor = tf.reshape(image_tensor, [B*self.num_temporal_splits,-1] + [H,W,C])

        if self.time_dilation > 1:
            image_tensor = utils.dilate_tensor(image_tensor, dilation_factor=self.time_dilation, axis=1)

        inputs_temporal_list = tf.split(image_tensor, image_tensor.shape.as_list()[1], axis=1)
        inputs_temporal_list = [tf.squeeze(im, axis=[1]) for im in inputs_temporal_list[:self.ntimes]]
        inputs_temporal_list = [tf.identity(inp, name=(self.name+'_input_t'+str(t))) for t,inp in enumerate(inputs_temporal_list)]
        return inputs_temporal_list

if __name__ == '__main__':

    import imp
    config_path = './cell_configs/egu_05L_config.py'
    cell_config = imp.load_source('config', config_path).config

    Ex = ConvRNN(
        name='05LConvRNNegu',
        base_config='./base_configs/enetB0like128_nobn_05Lscene',
        cell_config=cell_config,
        feedback_edges=[('conv3', 'conv1')],
        time_dilation=3,
        output_times=[0,1,2]
    )

    B,T,H,W,C = [2,4,256,380,3]
    inputs = tf.random.normal([B,T,H,W,C], dtype=tf.float32)
    outputs = Ex(inputs, train=True)
    import pdb
    pdb.set_trace()
