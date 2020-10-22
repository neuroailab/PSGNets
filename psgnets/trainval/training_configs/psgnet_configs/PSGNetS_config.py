from collections import OrderedDict
import tensorflow.compat.v1 as tf
from tfutils import optimizer
import vvn.models as models
from vvn.models.convrnn.convrnn_model import ConvRNN
from vvn.models.decoding import QtrDecoder, DeltaImages, DEFAULT_PRED_DIMS
import vvn.ops as ops
from vvn.trainval.utils import collect_and_flatten, total_loss
import vvn.trainval.eval_metrics as eval_metrics
from vvn.data.tdw_data import TdwSequenceDataProvider
import vvn.models.losses as losses
from vvn.models.preprocessing import *

gt_preproc_hsv = lambda im: preproc_rgb(im, to_hsv=True)

import os
CONVRNN_PATH = os.path.abspath('../models/convrnn')

MODEL_PREFIX = 'model_0'
POSTFIX = ''
IMAGES = 'images' + POSTFIX
DEPTHS = 'depths' + POSTFIX
NORMALS = 'normals' + POSTFIX
OBJECTS = 'objects' + POSTFIX
DELTAS = 'delta_images'
PMAT = 'projection_matrix'
VALID = 'valid'
INPUT_SIZE = 256
RESIZES = {k:[INPUT_SIZE]*2 for k in [IMAGES, DEPTHS, NORMALS, OBJECTS]}

INPUT_SEQUENCE_LEN = 4

MAX_DEPTH = 30.
MIN_DEPTH = 0.1
PRED_DIMS = OrderedDict([
    ('pred_depths', [1, lambda z: tf.clip_by_value(z, -MAX_DEPTH, MAX_DEPTH)]),
    ('pred_images', [3, preproc_hsv]),
    ('pred_normals', [3, lambda n: tf.nn.l2_normalize(n, axis=-1, epsilon=1e-3)]),
])

config = {
    'save_params': {
        'save_valid_freq': 10000,
        'save_filters_freq': 10000,
        'cache_filters_freq': 10000
    },
    'model_params': {
        'preprocessor': {
            'model_func': models.preprocessing.preproc_tensors_by_name,
            'dimension_order': ['images'],
            'dimension_preprocs': {
                'images': preproc_rgb
            }
        },
        'extractor': (ConvRNN, { # 128x128 features in conv1
            'name': 'ConvRNN_05Legu', 'base_tensor_name': 'conv1', 'layer_names': [],
            'base_config': os.path.join(CONVRNN_PATH, 'base_configs', 'enetB0like128_nobn_05Lscene'),
            'cell_config': os.path.join(CONVRNN_PATH, 'cell_configs', 'egu_05L_config'),
            'feedback_edges': [('conv2', 'conv1'), ('conv3', 'conv1'), ('conv4', 'conv1'), ('conv5', 'conv2'), ('scene', 'conv3')],
            'time_dilation': 3,
            'ff_order': ['conv'+str(L) for L in range(6)] + ['scene'],
        }),
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
                # 'graphconv_kwargs': {
                #     'hidden_dims': [100], 'concat_effects': False,
                #     'agg_type': 'mean', 'hw_thresh':None,
                # },
                'format_kwargs': {'keep_features': True, 'xyz_attr': True}
            }),
            (models.levels.P2GeoLevel, {
                'num_nodes':64, 'num_attrs':36,
                'geo_attrs': OrderedDict((
                    # ('position', [[0,2]]),
                    # ('unary_attrs', [[0,1],[3,6],[12,15],[21,22]])
                    ('unary_attrs', [[0,3]]),
                    ('hw_centroids', [[0,2]])
                )),
                # 'vae_attrs': ['vector'], # fts, xyz, hsv, nor, hwa = 52
                'vae_attrs': OrderedDict((
                    # ('vector', [ [0,40], [40,43], [-4,-1] ]),
                    ('vector', [ [0,40], [40,43], [43,46], [46,49], [-4,-1] ]),
                )),
                'affinity_kwargs': {'symmetric_output': False, 'kNN':256, 'kNN_train':1, 'vae_thresh': 3.5, 'geo_thresh': 1.0, 'geo_weight': 0.5},
                # 'vae_kwargs': {'encoder_dims': [50], 'decoder_dims': [50], 'activations': tf.nn.elu},
                'vae_kwargs': {'encoder_dims': [50], 'decoder_dims': [50], 'activations': tf.nn.relu},
                'pooling_kwargs': {'num_steps': 10, 'tau':0.0},
                'aggregation_kwargs': {
                    'agg_vars':False, 'agg_spatial_vars':False,
                    'agg_features': True, 'concat_spatial_attrs':True,
                    'concat_border_attrs': False, 'agg_feature_borders': False
                },
                'mlp_kwargs': {'hidden_dims': [100]},
                # 'graphconv_kwargs': {
                #     'agg_type':'mean', 'hidden_dims': [100]
                # },
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
                'latent_vector_key': 'unary_attrs_level1_mean_level2',
                'hw_attr': 'hw_centroids', 'num_sample_points': 4096,
                'attribute_dims_to_decode': PRED_DIMS,
                'method': 'constant'
            })
        ],
        'losses': [
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
                 'pred_depths': 'depths',
                 'pred_images': 'images',
                 'pred_normals': 'normals'
             },
             'attr_metrics': {
                 'pred_depths': losses.depth_metric,
                 'pred_images': losses.hue_metric,
                 'pred_normals': losses.normals_metric
             },
             'image_preprocs': {
                 'depths': models.preprocessing.preproc_depths,
                 'images': lambda im: gt_preproc_hsv(im)[...,0:1],
                 'normals': models.preprocessing.preproc_normals
             },
             'image_loss_funcs': {
                 'depths': losses.relative_depth_loss,
                 'images': losses.relative_normals_loss,
                 'normals': losses.relative_normals_loss,
             },
             'image_loss_func_kwargs': {
                 'depths': {'eps': 0.1},
                 'images': {'eps': 0.05, 'beta': 1.0},
                 'normals': {'eps': 0.025, 'beta': 2.0}
             },
             'loss_scales': {
                 'depths': 1.0,
                 'images': 5.0,
                 'normals': 2.5
             },
             'hw_attr': 'hw_centroids', 'valid_attr': 'valid'
            },
            {'name': 'level1_ppxy_loss',
             'loss_func': losses.pinhole_projection_loss,
             'logits_mapping': {
                 'nodes': 'nodes/level1',
                 'dimension_dict': 'dims/level1_dims'
             },
             'labels_mapping': {
                 'projection_matrix': 'projection_matrix'
             },
             'positions_attr': 'position', 'p_radius': 0.5,
             'stop_gradient': False,
             'scale': 10.0
            },
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
                 'pred_depths': 'depths',
                 'pred_normals': 'normals'
             },
             'image_preprocs': {
                 'images': gt_preproc_hsv,
                 'depths': models.preprocessing.preproc_depths,
                 'normals': models.preprocessing.preproc_normals
             },
             'loss_per_point_funcs': {
                 'images': losses.l2_loss,
                 'depths': losses.depth_per_point_loss,
                 'normals': losses.normals_per_point_loss
             },
             'loss_scales': {'images': 10.0, 'normals': 1.0, 'depths': 1.0}
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
                 'pred_depths': 'depths',
                 'pred_normals': 'normals'
             },
             'image_preprocs': {
                 'images': gt_preproc_hsv,
                 'depths': models.preprocessing.preproc_depths,
                 'normals': models.preprocessing.preproc_normals
             },
             'loss_per_point_funcs': {
                 'images': losses.l2_loss,
                 'depths': losses.depth_per_point_loss,
                 'normals': losses.normals_per_point_loss
             },
             'loss_scales': {'images': 10.0, 'normals': 1.0, 'depths': 1.0}
            },
            {'name': 'level2_vae',
             'loss_func': None,
             'logits_mapping': {'logits': 'losses/level2_loss'},
             'labels_mapping': {}, 'scale': 1.0
            }
        ],
        'inp_sequence_len': INPUT_SEQUENCE_LEN,
        'to_decode': None,
        'train_targets': [IMAGES, DEPTHS, NORMALS, OBJECTS, PMAT, VALID],
        'action_keys': [PMAT, IMAGES],
        'vectorize_nodes': False
    },
    'data_params': {
        'func': TdwSequenceDataProvider,
        'delta_time': 4,
        'enqueue_batch_size': 20,
        'buffer_mult': 20,
        'max_depth': 20.,
        'get_delta_images': True,
        'motion_filter': True,
        'motion_thresh': 0.03,
        'motion_area_thresh': 0.1,
        'train_filter_rule': None,
        'val_filter_rule': None,
        'resizes': RESIZES,
        'sources': [IMAGES, DEPTHS, NORMALS, OBJECTS, PMAT],
        'n_tr_per_dataset': 102400,
        'n_val_per_dataset': 10240
    },
    'loss_params': {
        'pred_targets': [IMAGES, DEPTHS, NORMALS, OBJECTS, PMAT],
        'loss_func': total_loss,
        'loss_func_kwargs': {}
    },
    'validation_params': {
        'object_metrics': {
            'targets': {
                'func': eval_metrics.get_pred_and_gt_segments,
                'segments_key': 'spatial/level2_segments',
                'gt_key': 'objects',
                'imsize': [64,64],
                'compute_matched': 1,
                'agg_mean': 1
            },
            'online_agg_func': eval_metrics.object_mask_and_boundary_metrics,
            'agg_func': eval_metrics.agg_mean_per_time,
            'val_length': (102400 // 9)
        }
    }
}
