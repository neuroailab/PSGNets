from collections import OrderedDict
import tensorflow.compat.v1 as tf
from tfutils import optimizer
import vvn.models as models
from vvn.models.spatiotemporal import motion_levels, selfsup_levels
from vvn.models.convrnn.convrnn_model import ConvRNN
from vvn.models.decoding import QtrDecoder, DeltaImages, DEFAULT_PRED_DIMS
import vvn.ops as ops
from vvn.trainval.utils import collect_and_flatten, total_loss
import vvn.trainval.eval_metrics as eval_metrics
from vvn.data.tdw_data import TdwSequenceDataProvider
import vvn.models.losses as losses
from vvn.models.preprocessing import preproc_hsv, preproc_rgb, delta_images

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
    ('pred_images', [3, preproc_hsv])
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
    ('pred_back_flows', [2, lambda f: tf.clip_by_value(f, -2., 2.)])
])

config = {
    'load_params': {
        'do_restore': True,
        'dbname': 'vvn',
        'collname': 'psgnet',
        'exp_id': 'EWP1_0dele0sob1sobfts_seq4dt1bs1_1',
        'query': {'step': 320000},
        'restore_global_step': False
    },
    'optimizer_params': {'trainable_scope': ['level2', 'level3']},
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
                    'agg_features': True, 'concat_spatial_attrs':True,
                    'concat_border_attrs': False, 'agg_feature_borders': False
                },
                'mlp_kwargs': {'hidden_dims': [100]},
                'format_kwargs': {'keep_features': False, 'xyz_attr':False}
            }),
            (selfsup_levels.P4Level, {
                'name': 'level3', 'input_name': 'level1', 'num_nodes': 64, 'num_attrs': 48,
                'vae_loss_scale': 1., 'selfsup_loss_scale': 100.0, 'use_target_segments': False,
                'static_attrs': OrderedDict((
                    ('features_level1', [[0,40]]),
                    ('unary_attrs_level1', [[0,3]]),
                    ('hw_centroids', [[0,2]])
                )),
                'affinity_kwargs': {'symmetric_output': True, 'symmetric': False, 'diff_inputs': True, 'hidden_dims': [100,100,100]},
                'pooling_kwargs': {'num_steps': 10, 'tau': 0.},
                'aggregation_kwargs': {'agg_vars': False, 'concat_spatial_attrs': True, 'agg_features': True},
                'mlp_kwargs': {'hidden_dims': [100]},
                'format_kwargs': {'keep_features': False, 'xyz_attr': False},
                'estimator_kwargs': {
                    'vae_attrs': OrderedDict((
                        ('features_level1', [[0,40]]),
                        # ('features_level1', [[0,43]]),
                        ('unary_attrs_level1', [[0,3]]),
                        ('pred_flood_attrs', [[0,5]]),
                        ('hw_centroids', [[0,2]])
                    )),
                    'vae_kwargs': {'encoder_dims': [50], 'decoder_dims': [50], 'activations': tf.nn.relu},
                    'affinity_kwargs': {
                        'symmetric_output': True, 'symmetric': False, 'diff_inputs': True, 'kNN': 512, 'kNN_train': 8, 'vae_thresh': 2.5
                    },
                    'pooling_kwargs': {'num_steps': 10, 'tau': 0.}
                }
            })
        ],
        'decoders': [
            (QtrDecoder, {
                'name': 'qtr_level2', 'input_mapping': {
                    'nodes': 'nodes/level2',
                    'segment_ids': 'spatial/level2_segments',
                    'dimension_dict': 'dims/level2_dims'
                },
                'latent_vector_key': 'unary_attrs_level2',
                'hw_attr': 'hw_centroids', 'num_sample_points': 4096,
                'attribute_dims_to_decode': PRED_DIMS_2,
                'method': 'quadratic'
            }),
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
            (QtrDecoder, {
                'name': 'qtr_flomo', 'input_mapping': {
                    'nodes': 'nodes/level1',
                    'segment_ids': 'spatial/level1_segments',
                    'dimension_dict': 'dims/level1_dims'
                },
                'latent_vector_key': 'pred_flood_attrs',
                'hw_attr': 'hw_centroids', 'num_sample_points': 4096,
                'attribute_dims_to_decode': PRED_MO_DIMS,
                'method': 'constant'
            })
        ],
        'losses': [
            {'name': 'level3_selfsup',
             'loss_func': None,
             'logits_mapping': {'logits': 'losses/level3_loss'},
             'labels_mapping': {}, 'scale': 1.0
            },
            {'name': 'level2_vae',
             'loss_func': None,
             'logits_mapping': {'logits': 'losses/level2_loss'},
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
            {'name': 'level1_flows_loss',
             'required_decoders': ['qtr_flomo'],
             'loss_func': losses.photometric_optical_flow_loss,
             'flow_attr': 'pred_flows', 'motion_attr': 'pred_delta_images',
             'logits_mapping': {
                 'pred_attrs': 'qtr_flomo/sampled_pred_attrs',
                 'valid_attrs': 'qtr_flomo/sampled_valid_attrs',
                 'spatial_inds': 'qtr_flomo/sampled_hw_inds',
                 'size': 'sizes/base_tensor'
             },
             'labels_mapping': {
                 # 'images': 'features/outputs'
                 'images': 'images',
                 'valid_images': 'valid'
             },
             'images_preproc': preproc_rgb,
             'alpha': 0.25,
             # 'images_preproc': tf.identity,
             'gate_on_motion': False,
             'motion_preproc': tf.nn.sigmoid,
             'motion_thresh': 0.5,
             'scale': 100.0, 'backward': False, 'xy_flows': True
             # 'scale': 0, 'backward': False, 'xy_flows': True
            },
            {'name': 'level1_back_flows_loss',
             'required_decoders': ['qtr_flomo'],
             'loss_func': losses.photometric_optical_flow_loss,
             'flow_attr': 'pred_back_flows', 'motion_attr': 'pred_delta_images',
             'logits_mapping': {
                 'pred_attrs': 'qtr_flomo/sampled_pred_attrs',
                 'valid_attrs': 'qtr_flomo/sampled_valid_attrs',
                 'spatial_inds': 'qtr_flomo/sampled_hw_inds',
                 'size': 'sizes/base_tensor'
             },
             'labels_mapping': {
                 # 'images': 'features/outputs'
                 'images': 'images',
                 'valid_images': 'valid'
             },
             'images_preproc': preproc_rgb,
             'alpha': 0.25,
             # 'images_preproc': tf.identity,
             'gate_on_motion': False,
             'motion_preproc': tf.nn.sigmoid,
             'motion_thresh': 0.5,
             'scale': 100.0, 'backward': True, 'xy_flows': True
             # 'scale': 0, 'backward': True, 'xy_flows': True
            },
            {'name': 'level1_deltas_loss',
             'required_decoders': ['qtr_flomo'],
             'loss_func': losses.rendered_attrs_images_loss,
             'logits_mapping': {
                 'pred_attrs': 'qtr_flomo/sampled_pred_attrs',
                 'valid_attrs': 'qtr_flomo/sampled_valid_attrs',
                 'spatial_inds': 'qtr_flomo/sampled_hw_inds',
                 'size': 'sizes/base_tensor'
             },
             'labels_mapping': {
                 'labels': 'inputs',
                 'valid_images': 'valid'
             },
             'attr_to_image_dict': {
                 'pred_delta_images': 'delta_images'
             },
             'image_preprocs': {
                 'delta_images': lambda im: tf.cast(im > 0.03, tf.float32),
             },
             'loss_per_point_funcs': {
                 'delta_images': losses.sigmoid_cross_entropy_with_logits
             },
             'loss_scales': {'delta_images': 10.0}
             # 'loss_scales': {'delta_images': 0}
            },
            {'name': 'level2_deltas_loss',
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
                 'pred_delta_images': 'delta_images'
             },
             'image_preprocs': {
                 'delta_images': lambda im: tf.cast(im > 0.03, tf.float32),
             },
             'loss_per_point_funcs': {
                 'delta_images': losses.sigmoid_cross_entropy_with_logits
             },
             'loss_scales': {'delta_images': 10.0}
            },
            {'name': 'level2_flows_loss',
             'required_decoders': ['qtr_level2'],
             'loss_func': losses.photometric_optical_flow_loss,
             'flow_attr': 'pred_flows', 'motion_attr': 'pred_delta_images',
             'logits_mapping': {
                 'pred_attrs': 'qtr_level2/sampled_pred_attrs',
                 'valid_attrs': 'qtr_level2/sampled_valid_attrs',
                 'spatial_inds': 'qtr_level2/sampled_hw_inds',
                 'size': 'sizes/base_tensor'
             },
             'labels_mapping': {
                 'images': 'images',
                 'valid_images': 'valid'
             },
             'images_preproc': preproc_rgb,
             'gate_on_motion': False,
             'motion_preproc': tf.nn.sigmoid,
             'motion_thresh': 0.5,
             'scale': 100.0, 'backward': False, 'xy_flows': True
            },
            # {'name': 'level2_back_flows_loss',
            #  'required_decoders': ['qtr_level2'],
            #  'loss_func': losses.photometric_optical_flow_loss,
            #  'flow_attr': 'pred_back_flows', 'motion_attr': 'pred_delta_images',
            #  'logits_mapping': {
            #      'pred_attrs': 'qtr_level2/sampled_pred_attrs',
            #      'valid_attrs': 'qtr_level2/sampled_valid_attrs',
            #      'spatial_inds': 'qtr_level2/sampled_hw_inds',
            #      'size': 'sizes/base_tensor'
            #  },
            #  'labels_mapping': {
            #      'images': 'images',
            #      'valid_images': 'valid'
            #  },
            #  'images_preproc': preproc_rgb,
            #  'gate_on_motion': False,
            #  'motion_preproc': tf.nn.sigmoid,
            #  'motion_thresh': 0.5,
            #  'scale': 100.0, 'backward': True, 'xy_flows': True
            # }
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
