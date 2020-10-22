from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import copy
import functools

import tensorflow.compat.v1 as tf

class Model(object):
    '''
    Abstract model class from which Extractor, PSGBuilder, Decoder can inherit
    '''
    def __init__(self, model_func=None, name=None, trainable=True, time_shared=False, **model_params):
        self.name = name or type(self).__name__
        self.params = copy.deepcopy(model_params)
        self.trainable = trainable
        self.time_shared = time_shared
        self.build_model(func=model_func, trainable=self.trainable)

    def build_model(self, func=None, trainable=True):
        if func:
            self.model_func_name = func.__name__
            def model(inputs, train, **kwargs):
                call_params = kwargs
                call_params.update(self.params)
                if trainable:
                    return func(inputs, train=train, **call_params)
                else:
                    return func(inputs, **call_params)

        else:
            self.model_func_name = 'passthrough'
            def model(inputs, train, **kwargs):
                return inputs

        self.model_func = model

    def reshape_batch_time(self, inputs, merge=True, **kwargs):
        if not self.time_shared:
            return inputs

        # merge batch and time dims
        if merge:
            def _merge(x):
                return tf.reshape(x, [self.B*self.T] + x.shape.as_list()[2:])
            if isinstance(inputs, tf.Tensor):
                inputs = _merge(inputs)
            elif isinstance(inputs, list):
                inputs = [_merge(inp) for inp in inputs]
            elif isinstance(inputs, dict):
                inputs = {k:_merge(tensor) for k,tensor in inputs.items()}

        else: #split
            def _split(x):
                return tf.reshape(x, [self.B, self.T] + x.shape.as_list()[1:])
            if isinstance(inputs, tf.Tensor):
                inputs = _split(inputs)
            elif isinstance(inputs, list):
                inputs = [_split(inp) for inp in inputs]
            elif isinstance(inputs, dict):
                inputs = {k:_split(tensor) for k,tensor in inputs.items()}

        return inputs

    def __call__(self, inputs, train=True, **kwargs):

        self.outputs = {}
        if isinstance(inputs, tf.Tensor):
            self.B,self.T = inputs.shape.as_list()[0:2]
        elif isinstance(inputs, dict):
            self.B,self.T = inputs[inputs.keys()[0]].shape.as_list()[0:2]
        else:
            raise TypeError("inputs to a Model must be a tf.Tensor a dict of them, but inputs=%s" % inputs)
        self.BT = self.B*self.T

        inputs = self.reshape_batch_time(inputs, merge=True, **kwargs)
        with tf.variable_scope(self.name):
            outputs = self.model_func(inputs, train=train, **kwargs)

        if isinstance(outputs, (list, tuple)):
            assert isinstance(outputs[0], (dict, tf.Tensor)) and isinstance(outputs[1], dict)
            outputs = [outputs[0], outputs[1]]
            outputs[0] = self.reshape_batch_time(outputs[0], merge=False, **outputs[1])
            for k,v in outputs[1].items():
                if isinstance(v, tf.Tensor):
                    outputs[1][k] = self.reshape_batch_time(v, merge=False)
        else:
            assert isinstance(outputs, dict) or isinstance(outputs, tf.Tensor), outputs
            outputs = self.reshape_batch_time(outputs, merge=False)

        return outputs


class ModelGeneric(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, name=None, tfutils_model=False):
        self.name = name or type(self).__name__
        self.tfutils_model = tfutils_model

    def preprocess_inputs(self, inputs, **kwargs):
        return inputs

    def postprocess_inputs(self, outputs, **kwargs):
        return outputs

    @abc.abstractmethod
    def build_outputs(self, *args, **kwargs):
        return

    @abc.abstractmethod
    def build_losses(self, *args, **kwargs):
        return

    def __call__(self, inputs, train, outputs=None, **kwargs):
        with tf.variable_scope(self.name):
            model_outputs = self.build_outputs(
                inputs, train, outputs=outputs, **kwargs)

            with tf.variable_scope('loss'):
                losses = self.build_losses(
                    model_outputs, inputs, **kwargs)
                model_outputs.update(losses)

        if not self.tfutils_model:
            return model_outputs
        else:
            return model_outputs, {}  # 2nd return value required by tfutils


def _l2_loss(logits, labels, **kwargs):
    return tf.square(logits - labels)

class Loss(object):
    '''
    Abstract loss class
    '''
    def __init__(
            self, name, required_decoders=[],
            loss_func=_l2_loss, scale=1.0,
            logits_keys=None,
            labels_keys=None,
            **kwargs
    ):
        self.loss_type = type(self).__name__
        self.required_decoders = required_decoders
        self.name = name
        self.scale = scale
        self.logits_keys = logits_keys
        self.labels_keys = labels_keys
        self.params = copy.deepcopy(kwargs)

        if loss_func is None:
            def _loss(logits, labels=None, **kwargs):
                return logits
            loss_func = _loss

        self.build_loss_func(func=loss_func)

    def build_loss_func(self, func):

        self.loss_func_name = func.__name__

        def loss_func(logits, labels, valid_logits=None, valid_labels=None, labels_first=True, **kwargs):

            _logits = self.get_logits(logits, valid_logits)
            _labels = self.get_labels(labels, valid_labels)
            loss_params = kwargs
            loss_params.update(self.params)
            if isinstance(_logits, dict) and isinstance(_labels, dict):
                loss_params.update(_logits)
                loss_params.update(_labels)
                loss_tensor = func(**loss_params)
            elif isinstance(_logits, list) and isinstance(_labels, list):
                loss_inputs = _labels + _logits if labels_first else _logits + _labels
                loss_tensor = func(*loss_inputs, **loss_params)
            elif isinstance(_logits, tf.Tensor) and isinstance(_labels, tf.Tensor):
                loss_inputs = [_labels, _logits] if labels_first else [_logits, _labels]
                loss_tensor = func(*loss_inputs, **loss_params)
            else:
                raise TypeError("logits and labels must both be either lists or dicts or tensors")

            loss_mask = self.get_loss_mask(
                loss_tensor, valid_logits, valid_labels, **loss_params)
            loss_scalar = self.reduce_loss_tensor(loss_tensor, loss_mask, **loss_params)
            return loss_scalar

        self.loss_func = loss_func

    def get_logits(self, logits, valid_logits):
        if self.logits_keys:
            return {k:logits[k] for k in self.logits_keys}
        else:
            return logits

    def get_labels(self, labels, valid_labels):
        if self.labels_keys:
            return {k:labels[k] for k in self.labels_keys}
        else:
            return labels

    def get_loss_mask(self, loss_tensor, valid_logits=None, valid_labels=None, **kwargs):

        if valid_logits is None:
            valid_logits = tf.ones(loss_tensor.shape, tf.float32)
        if valid_labels is None:
            valid_labels = tf.ones(loss_tensor.shape, tf.float32)
        loss_mask = tf.logical_and(valid_logits > 0.5, valid_labels > 0.5)
        loss_mask = tf.cast(loss_mask, loss_tensor.dtype)
        return loss_mask

    def reduce_loss_tensor(self, loss_tensor, loss_mask, mean_across_dims=[0,1], **kwargs):

        shape = loss_tensor.shape.as_list()
        rank = len(shape)
        sum_dims = [d for d in range(rank) if d not in mean_across_dims]
        num_valid = tf.reduce_sum(loss_mask, axis=sum_dims, keepdims=True) # e.g. [B,T,1,1]
        loss = tf.reduce_sum(loss_tensor, axis=sum_dims, keepdims=True) / tf.maximum(1., num_valid)
        loss_scalar = tf.reduce_mean(loss)
        return loss_scalar

    def __call__(self, outputs, labels,
                 valid_logits=None, valid_labels=None,
                 logits_mapping={'logits': 'classifier/outputs'},
                 labels_mapping={'labels': 'labels'},
                 suffix='', **kwargs
    ):

        logits = {k: outputs[logits_mapping[k]] for k in logits_mapping.keys()}
        labels_here = {k: labels[labels_mapping[k]] for k in labels_mapping.keys()}
        loss = self.loss_func(logits, labels_here, valid_logits, valid_labels, **kwargs)
        loss *= self.scale
        loss_nm = self.name + ('_' + suffix if len(suffix) else '')
        return {loss_nm: loss}

class propdict(dict):
    """
    Property dictionary in which each value is guaranteed to have the same broadcastable
    shape aside from the inner most dimension
    """
    def __init__(self, *args, **kwargs):
        self.shape = None
        self.ndim = None
        self.update(*args, **kwargs)


    def __setitem__(self, key, value):
        if self.ndim and len(value.shape) != self.ndim:
            raise ValueError(
                    'Dimension mismatch! Shape must equal %s, but is %s for key "%s"' % \
                            (self.shape, value.shape, key))
        else:
            self.ndim = len(value.shape)

        if self.shape:
            value_shape = [value.shape[dim] if value.shape[dim] != 1 else self.shape[dim] \
                    for dim in range(self.ndim - 1)] + [1]
            self.shape = [self.shape[dim] if self.shape[dim] != 1 else value.shape[dim] \
                    for dim in range(self.ndim - 1)] + [1]

            if not self.is_equal_shape(value_shape, self.shape):
                raise ValueError(
                        'Shape mismatch! Shape must equal %s, but is %s for key "%s"' % \
                        (self.shape[:self.ndim - 1], value_shape[:self.ndim - 1], key))
        else:
            self.shape = list(value.shape[:self.ndim - 1]) + [1]

        super(propdict, self).__setitem__(key, value)


    def update(self, *args, **kwargs):
        if args:
            if len(args) > 1:
                raise TypeError('update expected at most 1 arguments, '
                                'got %d' % len(args))
            other = dict(args[0])
            for key in other:
                self[key] = other[key]
        for key in kwargs:
            self[key] = kwargs[key]


    def setdefault(self, key, value=None):
        if key not in self:
            self[key] = value
        return self[key]


    def is_equal_shape(self, shape1, shape2):
        if len(shape1) != len(shape2):
            return False

        for s1, s2 in zip(shape1, shape2):
            if s1 == None or None == (s1 == None) or \
                    s2 == None or None == (s2 == None):
                continue
            if s1 != s2:
                return False
        return True


class uniquedict(dict):
    def __setitem__(self, key, value):
        if key not in self:
            dict.__setitem__(self, key, value)
        else:
            raise KeyError("Key '%s' already exists" % key)


class Graph(dict):
    """
    Graph describes a graph through colored nodes and edges.
    """
    # Enable dot notation
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self,
            nodes = {},
            edges = {},
            *args,
            **kwargs):
        """
        :param nodes: dict(node_groups: propdict(node_attributes)),
            each value in nodes describes a layer or group of nodes with attributes
        :param edges: dict(edge_groups: propdict(layer, idx, edge_attrobites)),
            each value in edges describes a group of edges with node layer, node index
            and edge attributes
        """
        # Update nodes and edges
        self.update({"nodes": uniquedict(nodes), "edges": uniquedict(edges)})


    def __setitem__(self, key, value):
        # assert key in ["nodes", "edges"], ("Graph only contains nodes or edges")
        if key in ["nodes", "edges"]:
            # Check that nodes and edges are of the right format
            assert isinstance(value, dict), ("%s must be of type dict!" % key)
            assert all([isinstance(value[k], propdict) for k in value]), \
                    ("%s values must be of type propdict: %s" % \
                    (key, [(k, type(value[k])) for k in value]))

        # Edges must contain layer and idx
        if key == "edges":
            assert all([prop in value[k] for prop in ["layer", "idx"] for k in value]), \
                    ("Each edge must contain 'idx' and 'layer' properties!", value)

        super(Graph, self).__setitem__(key, value)


    def update(self, *args, **kwargs):
        if args:
            if len(args) > 1:
                raise TypeError('update expected at most 1 arguments, '
                        'got %d' % len(args))
            other = dict(args[0])
            for key in other:
                self[key] = other[key]
        for key in kwargs:
            self[key] = kwargs[key]


    def setdefault(self, key, value=None):
        if key not in self:
            self[key] = value
        return self[key]


# OPS
def node_op(func):
    def wrapper(graph,
            node_key,
            node_feat_keys,
            output_node_feat_key,
            *args,
            **kwargs):
        # Construct node features
        node_feat_keys = node_feat_keys if node_feat_keys is not None else \
                list(graph.nodes.keys())
        node_features = dict([(k, graph.nodes[node_key][k]) for k in node_feat_keys])

        # Compute update
        output_feature = func(node_features,
                *args, **kwargs)

        # Update receiver with computed feature
        graph.nodes[node_key][output_node_feat_key] = output_feature

        return graph
    return wrapper


def edge_op(func):
    def wrapper(graph,
            edge_key,
            edge_feat_keys,
            output_edge_feat_key,
            *args,
            **kwargs):
        # Construct edges with layers, idxs and features
        edge = graph.edges[edge_key]
        edge_features = dict([(k, edge[k]) for k in edge_feat_keys])

        # Compute edge update
        output_edge_feature = func(edge_features,
                *args, **kwargs)

        # Update edge with computed feature
        graph.edges[edge_key][output_edge_feat_key] = output_edge_feature

        return graph
    return wrapper

def graph_op(func):
    def wrapper(graph,
            edge_key,
            sender_feat_keys = None,
            receiver_feat_keys = None,
            edge_feat_keys = [],
            output_receiver_feat_key = None,
            output_edge_feat_key = None,
            *args,
            **kwargs):
        # Construct edges with layers, idxs and features
        edge = graph.edges[edge_key]
        edge_layers = np.squeeze(edge["layer"])
        edge_idxs = edge["idx"]
        edge_features = [edge[k] for k in edge_feat_keys]

        # Construct sender and receiver node features
        sender_feat_keys = sender_feat_keys if sender_feat_keys is not None else \
                list(graph.nodes.keys())
        receiver_feat_keys = receiver_feat_keys if receiver_feat_keys is not None else \
                sender_feat_keys

        # reverse from old convention
        sender = dict([(k, graph.nodes[edge_layers[0]][k]) for k in sender_feat_keys])
        receiver = dict([(k, graph.nodes[edge_layers[1]][k]) for k in receiver_feat_keys])

        # Compute receiver and edge update
        output_receiver_feature, output_edge_feature = func(edge_idxs,
                sender,
                receiver,
                edge_features,
                *args, **kwargs)

        # Update receiver with computed feature
        if output_receiver_feat_key:
            graph.nodes[edge_layers[1]][output_receiver_feat_key] = output_receiver_feature
        # Update edge with computed feature
        if output_edge_feat_key:
            graph.edges[edge_key][output_edge_feat_key] = output_edge_feature
        return graph
    return wrapper

if __name__ == '__main__':
    import numpy as np
    # Example Graph
    nodes = propdict({
            "layer1": propdict({"pos": np.zeros((1,3)), "mass": np.zeros((1,3))}),
            "layer2": propdict({"pos": np.zeros((1,3)), "mass": np.zeros((1,3))}),
            })
    edges = propdict({
            "within1": propdict({"layer": np.array([["layer1", "layer2"]]),
                "idx": np.ones((1,2)),
                "feat1": np.zeros((1,3)), "feat2": np.zeros((1,3))}),
            "1up2": propdict({"layer": np.array([["layer2", "layer2"]]),
                "idx": np.ones((1,2)),
                "feat1": np.zeros((1,3)), "feat2": np.zeros((1,3))}),
            })
    graph = Graph({"nodes": nodes, "edges": edges})
