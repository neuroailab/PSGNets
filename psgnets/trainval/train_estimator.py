from __future__ import division, print_function, absolute_import

#import importlib.util
import dill
import functools
import imp
import os
import pprint

from absl import app
from absl import flags
from absl import logging
import tensorflow.compat.v1 as tf

from vvn.data.data_utils import get_data_params
from vvn.models.dynamics import VisualPhysicsModel, PhysicsModel
from vvn.models.visual_encoder import vectorize_inputs_model
from vvn.data.tdw_data import TdwSequenceDataProvider

import pdb

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "exp_dir",
    None,
    "Path to directore where to store results")
flags.DEFINE_string(
    "config_path",
    None,
    "Path to config file")
flags.DEFINE_string(
    "ckpt_path",
    None,
    "Path to ckpt file")
flags.DEFINE_string(
    "data_dir",
    None,
    "Path to data")
flags.DEFINE_string(
    "gpus",
    None,
    help="GPUs to train on")

flags.mark_flag_as_required("gpus")
flags.mark_flag_as_required("exp_dir")
flags.mark_flag_as_required("data_dir")
flags.mark_flag_as_required("config_path")


def load_config(config_path):
    #spec = importlib.util.spec_from_file_location("config", config_path)
    #config = importlib.util.module_from_spec(spec)
    #spec.loader.exec_module(config)
    config = imp.load_source('config', config_path)
    logging.info("Config loaded from %s" % config_path)
    config = config.config
    config['config_path'] = config_path
    return config


def train(config):
    gpus = FLAGS.gpus.split(',')
    num_gpus = len(gpus)

    distribution_strategy = None
    if num_gpus == 1:
        distribution_strategy = tf.contrib.distribute.OneDeviceStrategy("device:GPU:0")
    elif num_gpus > 1:
        distribution_strategy = tf.distribute.MirroredStrategy(
            devices=['device:GPU:%d' % i for i in range(num_gpus)]
        )
    else:
        raise RuntimeError('Only support GPU')

    warm_start_settings = None
    if FLAGS.ckpt_path:
        warm_start_settings = tf.estimator.WarmStartSettings(
            FLAGS.ckpt_path
        )

    batch_size = config['train_params']['batch_size']
    train_data_params = config['train_data_params']
    train_data_provider = train_data_params['data_class'](
                            FLAGS.data_dir,
                            **train_data_params['data_init_kwargs'])
    train_input_fn = functools.partial(train_data_provider.build_dataset,
                                       int(batch_size / num_gpus),
                                       train=True)
    val_data_params = config['val_data_params']
    val_data_provider = val_data_params['data_class'](
                            FLAGS.data_dir,
                            **val_data_params['data_init_kwargs'])
    val_input_fn = functools.partial(val_data_provider.build_dataset,
                                     int(batch_size / num_gpus),
                                     train=False)

    ### Model ###
    model_params = config['model_params']
    loss_params = config['loss_params']
    lr_params = config['learning_rate_params']
    opt_params = config["optimizer_params"]
    metrics_params = config["metrics_params"]
    def model_fn(features, mode, params=None):
        is_training = mode == tf.estimator.ModeKeys.TRAIN

        # Model
        model = model_params['model_class'](**model_params['model_init_kwargs'])
        _, outputs = model(features['images'], train=is_training, **params)

        # Loss
        loss = loss_params['loss_func'](outputs, features, **loss_params['loss_func_kwargs'])
        tf.identity(loss, name='loss')
        tf.summary.scalar('loss', loss)

        l2_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        tf.summary.scalar('l2_loss', l2_loss)

        loss = loss + l2_loss

        if mode == tf.estimator.ModeKeys.TRAIN:
            # Learning rate
            global_step = tf.train.get_or_create_global_step()
            if 'batch_denom' in lr_params:
                base_lr = lr_params['lr_func_kwargs']['learning_rate']
                batch_denom = lr_params['batch_denom']
                lr_params['lr_func_kwargs']['learning_rate'] = \
                    base_lr * batch_size / batch_denom
            learning_rate = lr_params['lr_func'](global_step=global_step,
                                                 **lr_params['lr_func_kwargs'])

            tf.identity(learning_rate, name='learning_rate')
            tf.summary.scalar('learning_rate', learning_rate)

            # Optimizer
            optimizer = opt_params["optimizer_class"](learning_rate,
                                                      **opt_params["optimizer_init_kwargs"])
            grad_vars = optimizer.compute_gradients(loss)
            minimize_op = optimizer.apply_gradients(grad_vars, global_step)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            train_op = tf.group(minimize_op, update_ops)
        else:
            train_op = None

        metrics = metrics_params['metrics_func'](outputs, features,
                                                 **metrics_params['metrics_func_kwargs'])
        for k, v in metrics.items():
            tf.identity(v[1], name=k)
            tf.summary.scalar(k, v[1])

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=metrics,
        )

    run_config = tf.estimator.RunConfig(
        train_distribute=distribution_strategy,
        save_checkpoints_steps=config['save_params']['save_steps']
    )

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=FLAGS.exp_dir,
        config=run_config,
        warm_start_from=warm_start_settings
    )

    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn,
        max_steps=config['train_params']['max_train_steps'],
    )

    eval_spec = tf.estimator.EvalSpec(
        input_fn=val_input_fn,
        steps=config['train_params']['eval_steps']
    )

    tf.estimator.train_and_evaluate(
        estimator, train_spec, eval_spec
    )


def main(argv):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpus
    config = load_config(FLAGS.config_path)
    config["exp_dir"] = FLAGS.exp_dir
    logging.info(pprint.pformat(config))

    train(config)

if __name__ == "__main__":
    app.run(main)
