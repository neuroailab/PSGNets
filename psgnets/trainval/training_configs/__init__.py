from tfutils import optimizer, defaults
import tensorflow.compat.v1 as tf

DEFAULT_TFUTILS_PARAMS = {
    'save_params': {
        'do_save': True,
        'save_initial_filters': True,
        'save_metrics_freq': 500,
        'host': 'localhost',
        'save_valid_freq': 10000,
        'save_filters_freq': 10000,
        'cache_filters_freq': 10000
    },
    'train_params': {
        'queue_params': None,
        'thres_loss': float('inf'),
        'num_steps': float('inf'),  # number of steps to train
        'validate_first': False
    },
    'loss_params': {
        'labels_to_dict': True,
        'agg_func': defaults.mean_and_reg_loss
    },
    'learning_rate_params': {
        'func': tf.train.exponential_decay,
        'learning_rate': 1e-4,
        'decay_rate': 1.0, # constant learning rate
        'decay_steps': 1e6
    },
    'optimizer_params': {
        'optimizer': optimizer.ClipOptimizer,
        'optimizer_class': tf.train.AdamOptimizer,
        'clip': True,
        'clipping_method': 'norm',
        'clipping_value': 10000.0
    },
    'log_device_placement': False,  # if variable placement has to be logged
}
