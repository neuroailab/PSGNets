from collections import OrderedDict
import tensorflow.compat.v1 as tf
from tfutils import optimizer
from tfutils.defaults import mean_and_reg_loss
import psgnets.models as models
from psgnets.models.resnets.resnet_model import resnet_v1
from psgnets.models.decoding import Decoder, QtrDecoder, DEFAULT_PRED_DIMS
import psgnets.ops as ops
from psgnets.trainval.utils import collect_and_flatten, total_loss
import psgnets.trainval.eval_metrics as eval_metrics
from psgnets.data.imagenet_data import ImageNet
import psgnets.models.losses as losses
from psgnets.models.preprocessing import preproc_hsv, preproc_rgb

MODEL_PREFIX = 'model_0'
POSTFIX = ''
IMAGES = 'images'
INPUT_SIZE = 256
RESIZE = 256
CROP_SIZE = 256

INPUT_SEQUENCE_LEN = 1

WEIGHT_DECAY = 1e-4
## no batchnorm
RESNET18 = resnet_v1(18, norm_act_layer='relu', num_classes=None, weight_decay=WEIGHT_DECAY)

gt_preproc_hsv = lambda im: preproc_rgb(im, to_hsv=True)

PRED_DIMS = OrderedDict([
    ('pred_images', [3, preproc_hsv]),
])

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
            'dimension_preprocs': {
                'images': models.preprocessing.preproc_rgb
            }
        },
        'extractor': { # block0 will be 128x128; block0_pool is 64x64
            'model_func': RESNET18, 'base_tensor_name': 'block0',
            'name': 'ResNet18', 'layer_names': ['pool']
        },
        'graph_levels': [
            (models.levels.P1Level, {
                'num_nodes':512, 'num_attrs':24,
                'grouping_time': 0, 'vectorization_time': -1,
                'affinity_kwargs': {'k':3, 'symmetric':False},
                'pooling_kwargs': {'num_steps':10},
                'aggregation_kwargs': {
                    'agg_vars':False,
                    'concat_diff_features': False,
                    'concat_edge_attrs': False, 'edge_inputs': 'images',
                    'divide_by_quadrant': False
                },
                'mlp_kwargs': {'hidden_dims': [100,100]},
                'format_kwargs': {'keep_features': True, 'xyz_attr': False}
            }),
            (models.levels.P2GeoLevel, {
                'num_nodes':64, 'num_attrs':36,
                'geo_attrs': OrderedDict((
                    ('unary_attrs', [[0,3]]),
                    ('hw_centroids', [[0,2]])
                )),
                'vae_attrs': OrderedDict((
                    ('vector', [ [0,40], [40,43], [-4,-1] ]),
                )),
                'affinity_kwargs': {'symmetric_output': False, 'kNN':512, 'kNN_train':1, 'vae_thresh': 3.5, 'geo_thresh': 1.0, 'geo_weight': 0.5},
                'vae_kwargs': {'encoder_dims': [50], 'decoder_dims': [50], 'activations': tf.nn.relu},
                'pooling_kwargs': {'num_steps': 10, 'tau':0.0},
                'aggregation_kwargs': {
                    'agg_vars':False, 'agg_spatial_vars':False,
                    'agg_features': True, 'concat_spatial_attrs':False,
                    'concat_border_attrs': False, 'agg_feature_borders': False
                },
                'mlp_kwargs': {'hidden_dims': [100]},
                'format_kwargs': {'keep_features': True, 'xyz_attr':False}
            })
        ],
        'decoders': [
            (QtrDecoder, {
                'name': 'qtr_level1', 'input_mapping': {
                    'nodes': 'nodes/level1',
                    'segment_ids': 'spatial/level1_segments',
                    'dimension_dict': 'dims/level1_dims'
                },
                'latent_vector_key': 'unary_attrs_level1',
                'hw_attr': 'hw_centroids', 'num_sample_points': 4096,
                'attribute_dims_to_decode': PRED_DIMS,
                'method': 'constant'
            }),
            (QtrDecoder, {
                'name': 'qtr_level2', 'input_mapping': {
                    'nodes': 'nodes/level2',
                    'segment_ids': 'spatial/level2_segments',
                    'dimension_dict': 'dims/level2_dims'
                },
                'latent_vector_key': 'unary_attrs_level2',
                'hw_attr': 'hw_centroids', 'num_sample_points': 4096,
                'attribute_dims_to_decode': PRED_DIMS,
                'method': 'quadratic'
            }),
            (Decoder, {
                'name': 'classifier', 'model_func': ops.convolutional.fc, 'out_depth': 1000, 'kernel_init': 'random_normal', 'kernel_init_kwargs':{'stddev': 0.01, 'seed':0}, 'weight_decay': WEIGHT_DECAY, 'input_mapping': {'inputs': 'features/pool'}
            })
        ],
        'losses': [
            {'name': 'level1_qtr_loss',
             'required_decoders': ['qtr_level1'],
             'loss_func': losses.rendered_attrs_images_loss,
             'logits_mapping': {
                 'pred_attrs': 'qtr_level1/sampled_pred_attrs',
                 'valid_attrs': 'qtr_level1/sampled_valid_attrs',
                 'spatial_inds': 'qtr_level1/sampled_hw_inds',
                 'size': 'sizes/base_tensor'
             },
             'labels_mapping': {
                 'labels': 'inputs',
                 'valid_images': 'valid'
             },
             'attr_to_image_dict': {
                 'pred_images': 'images',
             },
             'image_preprocs': {
                 'images': gt_preproc_hsv,
             },
             'loss_per_point_funcs': {
                 'images': losses.l2_loss,
             },
             'loss_scales': {'images': 10.0}
            },
            {'name': 'level2_qtr_loss',
             'required_decoders': ['qtr_level2'],
             'loss_func': losses.rendered_attrs_images_loss,
             'logits_mapping': {
                 'pred_attrs': 'qtr_level2/sampled_pred_attrs',
                 'valid_attrs': 'qtr_level2/sampled_valid_attrs',
                 'spatial_inds': 'qtr_level2/sampled_hw_inds',
                 'size': 'sizes/base_tensor'
             },
             'labels_mapping': {
                 'labels': 'inputs',
                 'valid_images': 'valid'
             },
             'attr_to_image_dict': {
                 'pred_images': 'images',
             },
             'image_preprocs': {
                 'images': gt_preproc_hsv,
             },
             'loss_per_point_funcs': {
                 'images': losses.l2_loss,
             },
             'loss_scales': {'images': 10.0}
            },
            {'name': 'level1_rel_attr_loss',
             'loss_func': losses.relative_spatial_attributes_loss,
             'num_sample_points': 512, 'kNN': 128,
             'logits_mapping': {
                 'nodes': 'nodes/level1',
                 'dimension_dict': 'dims/level1_dims',
             },
             'labels_mapping': {
                 'labels': 'inputs',
                 'valid_images': 'valid'
             },
             'attr_to_image_dict': {
                 'pred_images': 'images',
             },
             'attr_metrics': {
                 'pred_images': losses.hue_metric,
             },
             'image_preprocs': {
                 'images': lambda im: gt_preproc_hsv(im)[...,0:1],
             },
             'image_loss_funcs': {
                 'images': losses.relative_normals_loss,
             },
             'image_loss_func_kwargs': {
                 'images': {'eps': 0.05, 'beta': 1.0},
             },
             'loss_scales': {
                 'images': 5.0,
             },
             'hw_attr': 'hw_centroids', 'valid_attr': 'valid'
            },
            {'name': 'level2_vae',
             'loss_func': None,
             'logits_mapping': {'logits': 'losses/level2_loss'},
             'labels_mapping': {}, 'scale': 1.0
            },
            {'name': 'CE', 'required_decoders': ['classifier'], 'loss_func': models.losses.sparse_ce, 'scale':1.0,
                'logits_mapping': {'logits': 'classifier/outputs'}, 'labels_mapping': {'labels': 'labels'}
            }
        ],
        'inp_sequence_len': INPUT_SEQUENCE_LEN,
        'to_decode': None,
        'train_targets': [IMAGES],
        'vectorize_nodes': False
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
        'pred_targets': [IMAGES, 'labels'],
        'loss_func': total_loss,
        'loss_func_kwargs': {},
        'agg_func': mean_and_reg_loss
    },
    # 'optimizer_params': {
    #     'optimizer': optimizer.ClipOptimizer,
    #     'optimizer_class': tf.train.MomentumOptimizer,
    #     'clip': False,
    #     'momentum': 0.9,
    #     'trainable_scope': None
    # },
    # 'learning_rate_params': {
    #     'func': tf.train.exponential_decay,
    #     'learning_rate': 0.01,
    #     'decay_rate': 0.1,
    #     'decay_steps': (ImageNet.TRAIN_LEN // 16) * 30,
    #     'staircase': True
    # },
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
