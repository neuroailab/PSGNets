import numpy as np
import tensorflow.compat.v1 as tf
import copy

NUM_LAYERS = 6
LAYER_NAMES = ['conv'+str(L) for L in range(1,NUM_LAYERS)]
EGU_PARAMS_BASE = {
    'tau_filter_size': [5,5],
    'cell_depth': 0,
    'weight_decay': 1e-5,
    'batch_norm': False,
    'group_norm': False,
    'num_groups': 8,
    'se_ratio': 0.25,
    'residual_add': True,
    'activation': 'swish',
    'bypass_state': False
}

config = {layer: copy.deepcopy(EGU_PARAMS_BASE) for layer in LAYER_NAMES}
config['conv1']['tau_filter_size'] = [7,7]
