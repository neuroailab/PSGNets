from __future__ import absolute_import, division, print_function

import copy
import imp
import os

import dill
import gridfs
import pymongo as pm
import tensorflow.compat.v1 as tf
from absl import logging

from tfutils.db_interface import DBInterface
from vvn.trainval.training_configs import DEFAULT_TFUTILS_PARAMS


def collect_and_flatten(inputs, outputs, targets, **kwargs):
    print("targets in collect and flatten", targets)
    target_outputs = {}
    for target in targets:
        if target in outputs.keys():
            if isinstance(outputs[target], dict):
                for t in outputs[target]:
                    target_outputs[t] = outputs[target][t]
            else:
                target_outputs[target] = outputs[target]
    return target_outputs

def total_loss(logits, labels, **kwargs):

    loss = tf.constant(0.0, tf.float32)
    assert all((isinstance(loss, tf.Tensor) for loss in logits.values()))
    for loss_name, loss_val in logits.items():
        print("Using loss %s" % loss_name)
        loss += loss_val

    return loss

def load_config(config_path):
    config_path = os.path.abspath(config_path)
    if config_path[-3:] != ".py":
        config_path += ".py"
    config = imp.load_source('config', config_path)
    logging.info("Config loaded from %s" % config_path)
    default_params = copy.deepcopy(DEFAULT_TFUTILS_PARAMS)
    config = config.config
    config['config_path'] = config_path
    config['default_params'] = default_params
    return config

def update_tfutils_params(which_params='save', base_params={}, new_params={}, config={}):

    if new_params is None:
        return
    key = which_params + '_params'
    params = copy.deepcopy(base_params.get(key, {}))
    config_params = copy.deepcopy(config.get(key, {}))
    config_params.update(new_params)
    params.update(config_params)
    base_params[key] = params
    return

def save_config(tfutils_params, save_dir=None):
    save_params = tfutils_params['save_params']
    fname = save_params['dbname'] + '.' + save_params['collname'] + '.' + save_params['exp_id'] + '.pkl'
    if save_dir is not None:
        with open(os.path.join(save_dir, fname), 'wb') as f:
            dill.dump(tfutils_params, f)
            f.close()

def has_ckpt(port, db, coll, exp_id):
    with pm.MongoClient(port=port) as conn:
        collfs = gridfs.GridFS(conn[db], coll)
        count = collfs.find({
            'exp_id': exp_id,
            'saved_filters': True
        }).count()

        return count > 0

def load_ckpt_vars(port, db, coll, exp_id, cache_dir):
    dbinterface = DBInterface(
        params={},
        load_params={
            'host': 'localhost',
            'port': port,
            'dbname': db,
            'collname': coll,
            'exp_id': exp_id,
            'do_restore': True,
            'cache_dir': cache_dir
        }
    )

    dbinterface.load_rec()
    _, ckpt_filename = dbinterface.load_data
    reader = tf.train.NewCheckpointReader(ckpt_filename)
    var_shapes = reader.get_variable_to_shape_map()
    return [v for v in var_shapes.keys() if v != 'global_step']