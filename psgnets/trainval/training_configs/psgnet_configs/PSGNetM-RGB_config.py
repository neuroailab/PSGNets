from collections import OrderedDict
import tensorflow.compat.v1 as tf
from tfutils import optimizer
import psgnets.models as models
from psgnets.models.spatiotemporal import motion_levels, selfsup_levels
from psgnets.models.convrnn.convrnn_model import ConvRNN
from psgnets.models.decoding import QtrDecoder, FutureQtrDecoder, QsrDecoder, DEFAULT_PRED_DIMS
import psgnets.ops as ops
from psgnets.trainval.utils import collect_and_flatten, total_loss
import psgnets.trainval.eval_metrics as eval_metrics
from psgnets.data.tdw_data import TdwSequenceDataProvider
import psgnets.models.losses as losses
from psgnets.models.preprocessing import preproc_hsv, preproc_rgb, delta_images

gt_preproc_hsv = lambda im: preproc_rgb(im, to_hsv=True)
def sigmoid_motion_metric(ni, nj):
    dist = tf.abs(ni[...,0:-1] - nj[...,0:-1])
    dist = tf.concat([dist, tf.abs(
        tf.nn.sigmoid(ni[...,-1:]) - tf.nn.sigmoid(nj[...,-1:])
    )], axis=-1)
    return dist

def clip_by(v):
    return lambda x: tf.clip_by_value(x, -v, v)

import os
CONVRNN_PATH = os.path.abspath('../models/convrnn')

MODEL_PREFIX = 'model_0'
POSTFIX = ''
IMAGES = 'images' + POSTFIX
DEPTHS = 'depths' + POSTFIX
NORMALS = 'normals' + POSTFIX
OBJECTS = 'objects' + POSTFIX
DELTAS = 'delta_images'
DELTAS_RGB = 'delta_rgb'
PMAT = 'projection_matrix'
VALID = 'valid'
INPUT_SIZE = 256
RESIZES = {k:[INPUT_SIZE]*2 for k in [IMAGES, DEPTHS, NORMALS, OBJECTS]}

INPUT_SEQUENCE_LEN = 4

MAX_DEPTH = 30.
MIN_DEPTH = 0.1
PRED_DIMS_1 = OrderedDict([
    ('pred_images', [3, preproc_hsv]),
    # ('pred_depths', [1, clip_by(MAX_DEPTH)]),
    # ('pred_flows', [2, clip_by(2.)]),
    # ('pred_back_flows', [2, clip_by(2.)])
])

PRED_DIMS_2 = OrderedDict([
    ('pred_images', [3, preproc_hsv]),
    ('pred_delta_images', [1, tf.identity]),
    ('pred_flows', [2, clip_by(2.)]),
    # ('pred_back_flows', [2, clip_by(2.)])
])

PRED_MO_DIMS = OrderedDict([
    ('pred_delta_images', [1, tf.identity]),
    ('pred_flows', [2, lambda f: tf.clip_by_value(f, -2., 2.)]),
    ('pred_back_flows', [2, lambda f: tf.clip_by_value(f, -2., 2.)]),
])

config = {
    'load_params': {
        'do_restore': True,
        'dbname': 'psgnets',
        'collname': 'psgnet',
        'exp_id': 'EWP1frz_EFP3_0nadj1grnn0rbp5iterLin_seq4dt1bs1_1',
        # 'query': {'step': 200000},
        'query': {'step': 75000},
        'restore_global_step': False
    },
    'optimizer_params': {'trainable_scope': ['level2d', 'level3', 'Decode']},
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
                'images': preproc_rgb
            }
        },
        'extractor': (ConvRNN, { # 64x64 features in conv2, 128x128 in conv1
            'name': 'ConvRNN_05Legu', 'base_tensor_name': 'conv1', 'layer_names': [],
            'base_config': os.path.join(CONVRNN_PATH, 'base_configs', 'enetB0like128_nobn_05Lscene'),
            'cell_config': os.path.join(CONVRNN_PATH, 'cell_configs', 'egu_05L_config'),
            'feedback_edges': [],
            'time_dilation': 1,
            'ff_order': ['conv'+str(L) for L in range(6)] + ['scene'],
        }),
        'graph_levels': [
            (motion_levels.EdgeWarpDiffP1Level, {
                'num_nodes':512, 'num_attrs':24,
                'grouping_time': 0, 'vectorization_time': -1,
                'affinity_kwargs': {'k':3, 'symmetric':False},
                'use_gcn': False,
                'estimator_kwargs': {
                    'num_lp_runs': 10, 'num_steps': 10, 'edge_threshold': 0.8, 'edge_k': 1, 'add_coordinates': False, 'warp_edge_thresh': 0.8,
                    'ksize': [7,7], 'warp_ksize': [9,9], 'cross_entropy_warp_loss': False, 'mask_with_deltas': False, 'deltas_thresh': 0.1,
                    'warp_softmax': False, 'warp_affinities_loss_scale': 0, 'edge_feats_key': None, 'stop_gradient_features': True, 'sobel_feats': True,
                    'focal_edge_loss': False, 'alpha': 0.25, 'gamma': 2.0, 'sobel_edge_target': True, 'sobel_images_key': 'images',
                    'warp_loss_scale': 1.
                },
                'pooling_kwargs': {'num_steps':10},
                'aggregation_kwargs': {
                    'agg_vars':False,
                    'concat_new_features': True,
                    'concat_diff_features': False,
                    'concat_edge_attrs': False, 'edge_inputs': 'images',
                    'divide_by_quadrant': False
                },
                'mlp_kwargs': {'hidden_dims': [100,100]},
                'format_kwargs': {'keep_features': True, 'xyz_attr': False}
            }),
            (motion_levels.EdgeFloodP3Level, {
                'name': 'level2', 'input_name': 'level1',
                'num_nodes':128, 'num_attrs': 48, 'use_vae': False, 'vae_attrs': None, 'compute_time_edges': False, 'spacetime_cluster': False,
                'estimator_kwargs': {
                    'edge_threshold': 0.8, 'k':3, 'edge_agg_vars': True, 'concat_contour_attrs': True,
                    'motion_attrs': [DELTAS_RGB, DELTAS], 'motion_agg_vars': True,
                    'embedding_mlp_kwargs': {'hidden_dims': [40,40], 'activations': tf.nn.elu},
                    'flood_activation': None,
                    'output_mlp_kwargs': {'hidden_dims': [25], 'activations': None},
                    'normalize_adj': False, 'num_flood_iters': 5, 'load_edge_effects': True, 'zero_init_flood': False,
                    'stop_gradient_edge_features': True, 'edge_to_edge_affinities': False, 'binary_flood_edges': False,
                    'use_graph_rnn': True, 'graph_rnn_kwargs': {'use_rbp': False, 'lcp_kwargs': {'tau': 0.95}, 'linear': True, 'scale_gradient': False, 'nonlinearity': None, 'input_drive': True},
                    'stop_gradient_motion': True
                },
                'affinity_kwargs': {
                    'symmetric': False, 'diff_inputs': True, 'hidden_dims': [100,100], 'activations': tf.nn.elu,
                    'kNN_train': 24, 'vae_thresh': 5.0
                },
                'pooling_kwargs': {'num_steps': 10, 'tau':0.0},
                'aggregation_kwargs': {
                    'agg_vars':False, 'agg_spatial_vars':False,
                    'concat_diff_features': False,
                    'agg_features': True, 'concat_spatial_attrs':False,
                    'concat_border_attrs': False, 'agg_feature_borders': False
                },
                'mlp_kwargs': {'hidden_dims': [100]},
                'format_kwargs': {'keep_features': False, 'xyz_attr':False}
            }),
            (motion_levels.DepthFromMotionP3Level, {
                'name': 'level2d', 'input_name': 'level1',
                'num_nodes':128, 'num_attrs': 48, 'use_vae': False, 'vae_attrs': None, 'compute_time_edges': False, 'spacetime_cluster': False,
                'estimator_kwargs': {
                    'edge_threshold': 0.8, 'k':3, 'edge_agg_vars': True, 'concat_contour_attrs': True, 'edge_feature_dims': [0,6], 'edge_agg_vars': False,
                    'motion_attrs': [DELTAS_RGB, DELTAS], 'motion_agg_vars': True,
                    'embedding_mlp_kwargs': {'hidden_dims': [40,40], 'activations': tf.nn.elu},
                    'flood_activation': None,
                    'output_mlp_kwargs': {'hidden_dims': [25], 'activations': None},
                    'normalize_adj': False, 'num_flood_iters': 5, 'load_edge_effects': True, 'zero_init_flood': False,
                    'stop_gradient_edge_features': True, 'edge_to_edge_affinities': False, 'binary_flood_edges': False,
                    'use_graph_rnn': True, 'graph_rnn_kwargs': {'use_rbp': False, 'lcp_kwargs': {'tau': 0.95}, 'linear': True, 'scale_gradient': False, 'nonlinearity': None, 'input_drive': True},
                    'stop_gradient_motion': True
                },
                'occlusion_kwargs': {
                    'flows_dims': ('pred_flood_attrs_level2', [[1,3]]),
                    'back_flows_dims': ('pred_flood_attrs_level2', [[3,5]]),
                    'scale_factor': 2.5, 'xy_flows': True, 'stop_gradient': True,
                    'depths_beta': 0.5, 'occlusion_thresh': 1., 'loss_times': [1,4]
                },
                'affinity_kwargs': {
                    'symmetric': False, 'diff_inputs': True, 'hidden_dims': [100,100], 'activations': tf.nn.elu,
                    'kNN_train': 24, 'vae_thresh': 5.0
                },
                'pooling_kwargs': {'num_steps': 10, 'tau':0.0},
                'aggregation_kwargs': {
                    'agg_vars':False, 'agg_spatial_vars':False,
                    'concat_diff_features': False,
                    'concat_new_features': False,
                    'agg_features': True, 'concat_spatial_attrs':False,
                    'concat_border_attrs': False, 'agg_feature_borders': False
                },
                'mlp_kwargs': {'hidden_dims': [100]},
                'format_kwargs': {'keep_features': False, 'xyz_attr':False}
            }),
            (selfsup_levels.P4Level, {
                'name': 'level3', 'input_name': 'level1', 'num_nodes': 64, 'num_attrs': 60,
                'vae_loss_scale': 1., 'selfsup_loss_scale': 100.0, 'use_target_segments': False,
                'static_attrs': OrderedDict((
                    ('features_level1', [[0,40]]),
                    ('unary_attrs_level1', [[0,3]]),
                    ('hw_centroids', [[0,2]]),
                    ('pred_flood_attrs_level2d', [[0,25]])
                )),
                'affinity_kwargs': {'symmetric_output': True, 'symmetric': False, 'diff_inputs': True, 'hidden_dims': [100,100,100]},
                'pooling_kwargs': {'num_steps': 10, 'tau': 0.},
                'aggregation_kwargs': {'agg_vars': False, 'concat_spatial_attrs': False, 'agg_features': True, 'concat_border_attrs': True},
                'mlp_kwargs': {'hidden_dims': [100]},
                'format_kwargs': {'keep_features': False, 'xyz_attr': False},
                'estimator_kwargs': {
                    'vae_attrs': OrderedDict((
                        ('features_level1', [[0,40]]),
                        ('unary_attrs_level1', [[0,3]]),
                        ('pred_flood_attrs_level2', [[0,5]]),
                        ('pred_flood_attrs_level2d', [[0,1]]),
                        ('hw_centroids', [[0,2]])
                    )),
                    'vae_kwargs': {'encoder_dims': [50], 'decoder_dims': [50], 'activations': tf.nn.relu},
                    'affinity_kwargs': {
                        'symmetric_output': True, 'symmetric': False, 'diff_inputs': True, 'kNN': 512, 'kNN_train': 8, 'vae_thresh': 2.5
                    },
                    'pooling_kwargs': {'num_steps': 10, 'tau': 0.}
                },
                'format_kwargs': {'keep_features': True, 'xyz_attr': False}
            })
        ],
        'decoders': [
            (QtrDecoder, {
                'name': 'qtr_level3', 'input_mapping': {
                    'nodes': 'nodes/level3',
                    'segment_ids': 'spatial/level3_segments',
                    'dimension_dict': 'dims/level3_dims'
                },
                'latent_vector_key': 'unary_attrs_level3',
                'hw_attr': 'hw_centroids', 'num_sample_points': 4096,
                'attribute_dims_to_decode': PRED_DIMS_1,
                'method': 'quadratic'
            }),
            # (FutureQtrDecoder, {
            #     'name': 'futqtr_level3', 'input_mapping': {
            #         'nodes': 'nodes/level3',
            #         'segment_ids': 'spatial/level3_segments',
            #         'dimension_dict': 'dims/level3_dims'
            #     },
            #     'flows_dims': ('pred_flood_attrs_level2', [[1,3]]), 'key_pos':0,
            #     # 'back_flows_dims': ('pred_back_flows', [[0,2]]),
            #     'depths_dims': ('pred_depths', [[0,1]]),
            #     'attribute_dims_to_decode': {'unary_attrs_level1': [[[0,3]], preproc_hsv]},
            #     'stop_gradient_attrs': True, # so grad must go through depths
            #     'scale_factor': 1., 'xy_flows': True, 'scale': True, 'beta': 0.5

            # }),
            (QsrDecoder, {
                'name': 'qsr_level3', 'input_mapping': {
                    'nodes': 'nodes/level3',
                    'dimension_dict': 'dims/level3_dims',
                    'size': 'sizes/base_tensor'
                },
                'num_constraints': 8, 'scale_codes_by_imsize': True,
                'shape_dims': ('unary_attrs_level3_remainder', [[0,32]]),
                'depths_dims': ('pred_flood_attrs_level2d', [[0,1]]),
                # 'depths_dims': ('pred_flood_attrs_level2d', [[1,25]]),
                # 'depths_dims': ('pred_depths', [[0,6]]),
                # 'depths_conv_kwargs': {'ksize': [1,1]},
                'hw_attr': 'hw_centroids',
                'zero_max': False, 'valid_mask': True,
                # 'beta': 3.0, 'min_depth': -500.
            }),
            (QtrDecoder, {
                'name': 'qtr_flomo', 'input_mapping': {
                    'nodes': 'nodes/level1',
                    'segment_ids': 'spatial/level1_segments',
                    'dimension_dict': 'dims/level1_dims'
                },
                'latent_vector_key': 'pred_flood_attrs_level2', 'key_pos':0,
                'hw_attr': 'hw_centroids', 'num_sample_points': 4096,
                'attribute_dims_to_decode': PRED_MO_DIMS,
                'method': 'constant'
            })
        ],
        'losses': [
            {'name': 'level2d_occlusion',
             'loss_func': None,
             'logits_mapping': {'logits': 'losses/level2d_loss'},
             'labels_mapping': {}, 'scale': 10.0
            },
            {'name': 'level3_selfsup',
             'loss_func': None,
             'logits_mapping': {'logits': 'losses/level3_loss'},
             'labels_mapping': {}, 'scale': 1.0
            },
            {'name': 'level3_qtr_loss',
             'required_decoders': ['qtr_level3'],
             'loss_func': losses.rendered_attrs_images_loss,
             'logits_mapping': {
                 'pred_attrs': 'qtr_level3/sampled_pred_attrs',
                 'valid_attrs': 'qtr_level3/sampled_valid_attrs',
                 'spatial_inds': 'qtr_level3/sampled_hw_inds',
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
            # {'name': 'level3_future_loss',
            #  'required_decoders': ['futqtr_level3'],
            #  'loss_func': losses.masked_l2_image_loss,
            #  'logits_mapping': {
            #      'pred_image': 'futqtr_level3/unary_attrs_level1',
            #      'valid_image': 'futqtr_level3/contested_pixels'
            #  },
            #  'labels_mapping': {
            #      'gt_image': 'images'
            #  },
            #  'gt_preproc': gt_preproc_hsv,
            #  'pred_times': [1,None], 'gt_times': [2,None],
            #  # 'pred_times': [0,None], 'gt_times': [1,None],
            #  'scale': 100.
            # },
            {'name': 'qsr_loss',
             'required_decoders': ['qsr_level3'],
             'loss_func': losses.sparse_ce,
             'logits_mapping': {
                 'logits': 'qsr_level3/shape_logits',
             },
             'labels_mapping': {
                 'labels': 'spatial/level3_segments'
             },
             'valid_logits_key': 'qsr_level3/shapes_valid',
             'scale': 10.
            }
        ],
        'inp_sequence_len': INPUT_SEQUENCE_LEN,
        'to_decode': None,
        'train_targets': [IMAGES, DELTAS, OBJECTS, PMAT, VALID],
        'action_keys': [PMAT, IMAGES, DELTAS, DELTAS_RGB],
        'vectorize_nodes': False
    },
    'data_params': {
        'func': TdwSequenceDataProvider,
        'delta_time': 1,
        'enqueue_batch_size': 20,
        'buffer_mult': 20,
        'max_depth': 20.,
        'get_segments': True,
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
        'pred_targets': [],
        'loss_func': total_loss,
        'loss_func_kwargs': {}
    },
    'validation_params': {
        'object_metrics': {
            'targets': {
                'func': eval_metrics.get_pred_and_gt_segments,
                'segments_key': 'spatial/level3_segments',
                'gt_key': 'segments',
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
