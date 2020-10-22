from __future__ import division, print_function, absolute_import

import dill
import imp
import os
import pprint
import copy

from absl import app
from absl import flags
from absl import logging
import tensorflow.compat.v1 as tf

from tfutils import base, optimizer

# from vvn.data.data_utils import get_data_params
from vvn.data.tdw_data import TdwSequenceDataProvider
import vvn.models.psgnet as psgnet
from utils import collect_and_flatten
from training_configs import DEFAULT_TFUTILS_PARAMS

import pdb

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "dbname", "vvn", help="Name of database to store in mongodb")
flags.DEFINE_string(
    "collname", "psgnet", help="Name of collection to store in mongodb")
flags.DEFINE_string(
    "exp_id", None, help="Name of experiment to store in mongodb")
flags.DEFINE_integer(
    "port", 27024, help="localhost port of the mongodb to save to")
flags.DEFINE_integer(
    "load_port", None, help="localhost port of the mongodb to save to")
flags.DEFINE_string(
    "config_path", None, help="Path to the config for a VVN model")
flags.DEFINE_string(
    "load_config_path", None, help="Path to the config for a VVN model")
flags.DEFINE_integer(
    "batch_size", 1, help="batch size for model")
flags.DEFINE_integer(
    "sequence_length", 1, help="movie sequence length for model")
flags.DEFINE_string(
    "data_dir", None, help="the directory with various datasets"),
flags.DEFINE_string(
    "dataset_names", "playroom_v1", help="The comma-separated names of the TDW datasets to train from")
flags.DEFINE_integer(
    "minibatch_size", None, help="minibatch size for model")
flags.DEFINE_integer(
    "num_gpus", 1, help="number of gpus to run model on")
flags.DEFINE_bool(
    "use_default_params", True, help="Use the default tfutils train params other than model, saving, etc.")
flags.DEFINE_string(
    "gpus", None, help="GPUs to train on")
flags.DEFINE_bool(
    "train", True, help="whether to train or test")
flags.DEFINE_bool(
    "trainval", False, help="whether to train or test")
flags.DEFINE_bool(
    "load", False, help="whether to load a previous model saved under this name")
flags.DEFINE_string(
    "load_exp_id", None, help="name of exp to load from")
flags.DEFINE_integer(
    "step", None, help="Which step to load from"),
flags.DEFINE_bool(
    "validate_first", False, help="Whether to validate first")
flags.DEFINE_string(
    "trainable", None, help="Comma-separated list of trainable scope names. Default trains all variables")
flags.DEFINE_integer(
    "seed", 0, help="random seed")
flags.DEFINE_string(
    "save_dir", None, help="where to save a pickle of the tfutils_params")
flags.DEFINE_string(
    "save_tensors", None, help="comma-separated list of which tensors to save in gfs")


flags.mark_flag_as_required("exp_id")
flags.mark_flag_as_required("config_path")
flags.mark_flag_as_required("gpus")

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

def build_trainval_params(config, loss_names=[]):
    data_params = config.get('data_params', {'func': TdwSequenceDataProvider})
    data_provider_cls = data_params['func']
    def _data_input_fn_wrapper(batch_size, train, **kwargs):
        data_provider = data_provider_cls(**kwargs)
        return data_provider.input_fn(batch_size, train)

    train_data_params, val_data_params = data_provider_cls.get_data_params(
        batch_size=FLAGS.batch_size,
        sequence_len=FLAGS.sequence_length,
        dataprefix=FLAGS.data_dir or data_provider_cls.DATA_PATH,
        dataset_names=FLAGS.dataset_names.split(','),
        **data_params)

    train_data_params.update({'func': _data_input_fn_wrapper, 'batch_size': FLAGS.batch_size, 'train': True})
    val_data_params.update({'func': _data_input_fn_wrapper, 'batch_size': FLAGS.batch_size, 'train': False if not FLAGS.trainval else True})

    train_params_targets = {
        'func': collect_and_flatten,
        'targets': loss_names
    }
    train_params = {
        'minibatch_size': FLAGS.minibatch_size or FLAGS.batch_size,
        'data_params': train_data_params,
        'targets': train_params_targets,
        'validate_first': FLAGS.validate_first
    }
    val_params = config.get('validation_params', {})
    for val_key, val_dict in val_params.items():
        val_dict.update({
            'data_params': val_data_params if not FLAGS.trainval else train_data_params,
            'num_steps': val_params[val_key].get('val_length', 50000) // (FLAGS.batch_size * FLAGS.sequence_length)
        })

    return train_params, val_params

def initialize_psgnet_model(config):
    model_params = config['model_params']
    Model = psgnet.PSGNet(**model_params)
    model_call_params = copy.deepcopy(Model.params)
    model_call_params['func'] = Model
    logging.info(pprint.pformat(model_params))
    logging.info(pprint.pformat(model_call_params))
    return model_call_params

def save_config(tfutils_params, save_dir=None):
    save_params = tfutils_params['save_params']
    fname = save_params['dbname'] + '.' + save_params['collname'] + '.' + save_params['exp_id'] + '.pkl'
    if save_dir is not None:
        with open(os.path.join(save_dir, fname), 'wb') as f:
            dill.dump(tfutils_params, f)
            f.close()

def load_tfutils_params(save_dir, dbname, collname, exp_id):
    fname = dbname + '.' + collname + '.' + exp_id + '.pkl'
    fpath = os.path.join(save_dir, fname)
    with open(fpath, 'rb') as f:
        tfutils_params = dill.load(f)
        f.close()
    return tfutils_params

def train(config, dbname, collname, exp_id, port, gpus=[0], use_default=True, load=True):

    tfutils_params = config['default_params'] if use_default else {}

    ### MODEL ###
    model_params = initialize_psgnet_model(config)
    loss_names = model_params['func'].Losses.keys()
    model_params.update({
        'devices': ['/gpu:' + str(i) for i in range(len(gpus))],
        'num_gpus': len(gpus),
        'seed': FLAGS.seed,
        'prefix': 'model_0'
    })
    tfutils_params['model_params'] = model_params

    ### INPUT DATA ###
    train_params, val_params = build_trainval_params(config, loss_names=loss_names)
    update_tfutils_params('train', tfutils_params, train_params, config={})
    update_tfutils_params('validation', tfutils_params, val_params, config={})

    ### OPTIMIZATION ###
    trainable = FLAGS.trainable
    if trainable is not None:
        trainable = trainable.split(',')
        if len(trainable) == 0:
            trainable = None
        opt_params = {'trainable_scope': trainable}
    else:
        opt_params = {}
    update_tfutils_params('optimizer', tfutils_params, opt_params, config)
    update_tfutils_params('loss', tfutils_params, {}, config)
    update_tfutils_params('learning_rate', tfutils_params, {}, config)

    ### SAVE AND LOAD ###
    save_params = {
        'dbname': dbname,
        'collname': collname,
        'exp_id': exp_id,
        'port': port
    }
    update_tfutils_params('save', tfutils_params, save_params, config)

    if 'load_params' not in config.keys() or (FLAGS.load_exp_id is not None):
        load_params = copy.deepcopy(save_params)
        load_params.pop('exp_id')
        load_exp_id = FLAGS.load_exp_id if load else None
        load_params.update({
            'do_restore': True,
            'query': {'step': FLAGS.step} if FLAGS.step else None,
            'restore_global_step': True if (exp_id == load_exp_id) else False
        })
        update_tfutils_params('load', tfutils_params, load_params if load else None, config)
    else:
        load_params = copy.deepcopy(config['load_params'])
        load_exp_id = load_params['exp_id']
        load_params.update({'do_restore': True, 'restore_global_step': True if (exp_id == load_exp_id) else False})
        tfutils_params['load_params'] = load_params

    if load and (load_exp_id is not None): # overwrite if passed as arg
        tfutils_params['load_params']['exp_id'] = load_exp_id
    elif load and not tfutils_params['load_params'].get('exp_id', None):
        tfutils_params['load_params']['exp_id'] = exp_id
    elif config.get('load_params', None) is not None and not load:
        raise ValueError("It looks like you're trying to load from an experiment specified in the config, but you need to set '--load 1'")

    ### SAVE OUT CONFIG ###
    save_config(tfutils_params, save_dir=FLAGS.save_dir)

    logging.info(pprint.pformat(tfutils_params))
    base.train_from_params(**tfutils_params)

def test(config, dbname, collname, load_exp_id, port, gpus=[0], suffix='_val0', use_default=True):

    if FLAGS.save_dir is not None:
        val_tfutils_params = load_tfutils_params(FLAGS.save_dir, dbname, collname, load_exp_id)
        model_params = initialize_psgnet_model(val_tfutils_params['model_params'])
    else:
        assert FLAGS.load_config_path is not None, "Must pass a model config to load from if not restoring pickled tfutils params"
        val_tfutils_params = {}
        model_config = load_config(FLAGS.load_config_path)
        model_params = initialize_psgnet_model(model_config)

    ### MODEL ###
    model_params.update({
        'devices': ['/gpu:' + str(i) for i in range(len(gpus))],
        'num_gpus': len(gpus),
        'seed': FLAGS.seed,
        'prefix': 'model_0'
    })
    val_tfutils_params['model_params'] = model_params

    ### INPUT DATA ###
    _, val_params = build_trainval_params(config, loss_names=[])
    for k in val_params.keys():
        val_params[k]['data_params']['train'] = False
    update_tfutils_params('validation', val_tfutils_params, val_params, config={})

    ### LOAD AND SAVE ###
    load_params = {
        'host': 'localhost',
        'port': FLAGS.load_port or port,
        'dbname': dbname,
        'collname': collname,
        'exp_id': load_exp_id,
        'do_restore': True,
        'query': {'step': FLAGS.step} if FLAGS.step else None,
    }
    update_tfutils_params('load', val_tfutils_params, load_params, config={})

    try:
        save_to_gfs = config['save_to_gfs']
    except KeyError:
        save_to_gfs = FLAGS.save_tensors.split(',')

    step_name = 'last' if not FLAGS.step else str(FLAGS.step)
    save_params = {
        'dbname': dbname,
        'collname': collname,
        'exp_id': load_exp_id + suffix + '_' + step_name,
        'port': port,
        'save_to_gfs': save_to_gfs
    }
    update_tfutils_params('save', val_tfutils_params, save_params, config={})

    ### SAVE OUT CONFIG ###
    save_config(val_tfutils_params, save_dir=FLAGS.save_dir)

    logging.info(pprint.pformat(val_tfutils_params))
    base.test_from_params(**val_tfutils_params)

def main(argv):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpus
    tf.logging.set_verbosity(tf.logging.ERROR)
    config = load_config(FLAGS.config_path)
    gpus = FLAGS.gpus.split(',')
    if FLAGS.train:
        train(config, FLAGS.dbname, FLAGS.collname, FLAGS.exp_id, FLAGS.port,
              gpus=gpus, use_default=FLAGS.use_default_params, load=FLAGS.load)
    else:
        assert FLAGS.load_exp_id is not None, "Must load from a named experiment passed to --load_exp_id"
        suffix = ('_val_' if not FLAGS.trainval else '_trainval_') + FLAGS.exp_id
        logging.info("Saving test results under the exp_id: %s" % FLAGS.load_exp_id + suffix)
        test(config, FLAGS.dbname, FLAGS.collname, FLAGS.load_exp_id, FLAGS.port,
             gpus=gpus, suffix=suffix, use_default=FLAGS.use_default_params)

if __name__ == '__main__':
    app.run(main)
