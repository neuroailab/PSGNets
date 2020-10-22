import tensorflow.compat.v1 as tf
from tfutils import optimizer
import psgnets.models as models
import psgnets.ops as ops
import psgnets.trainval.eval_metrics as eval_metrics
from psgnets.trainval.utils import collect_and_flatten, total_loss
from psgnets.data.imagenet_data import ImageNet

MODEL_PREFIX = 'model_0'
CROP_SIZE = 256
RESIZE = 256
INPUT_SEQUENCE_LEN = 1

config = {
    'save_params': {
        'save_valid_freq': 5000,
        'save_filters_freq': 5000,
        'cache_filters_freq': 5000
    },
    'model_params': {
        'preprocessor': {
            'model_func': models.preprocessing.preproc_tensors_by_name,
            'dimension_order': ['images'],
            'dimension_preprocs': {'images': models.preprocessing.preproc_rgb}
        },
        'extractor': {
            'model_func': ops.convolutional.convnet_stem,
            'name': 'ConvNet', 'layer_names': ['conv'+str(i) for i in range(5)],
            'ksize': 7, 'max_pool': True, 'conv_kwargs': {'activation': 'relu'},
            'hidden_ksizes': [3,3,3,3], 'hidden_channels': [64,128,256,512], 'out_channels': 1024
        },
        'decoders': [
            {'name': 'classifier', 'model_func': ops.convolutional.fc, 'out_depth': 1000, 'kernel_init': 'truncated_normal', 'kernel_init_kwargs':{'stddev': 0.1, 'seed':0}, 'input_mapping': {'inputs': 'features/outputs'}}
        ],
        'losses': [
            {
                'name': 'CE', 'required_decoders': ['classifier'], 'loss_func': models.losses.sparse_ce, 'scale':1.0,
                'logits_mapping': {'logits': 'classifier/outputs'}, 'labels_mapping': {'labels': 'labels'}
            }
        ],
        'inp_sequence_len': INPUT_SEQUENCE_LEN,
        'train_targets': ['labels']
    },
    'data_params': {
        'func': ImageNet,
        'prep_type': 'resnet',
        'crop_size': CROP_SIZE,
        'resize': RESIZE,
        'images_key': 'images',
        'labels_key': 'labels',
        'do_color_normalize': False
    },
    'loss_params': {
        'pred_targets': ['labels'],
        'loss_func': total_loss,
        'loss_func_kwargs': {}
    },
    'validation_params': {
        'accuracy': {
            'targets': {
                'func': eval_metrics.loss_and_in_top_k,
                'target': 'labels',
                'logits_key': 'classifier/outputs'
            },
            'val_length': ImageNet.VAL_LEN,
            'online_agg_func': eval_metrics.online_agg,
            'agg_func': eval_metrics.mean_res

        }
    }
}
