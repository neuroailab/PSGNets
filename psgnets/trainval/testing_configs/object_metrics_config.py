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
from vvn.models.preprocessing import preproc_hsv, preproc_rgb, delta_images

gt_preproc_hsv = lambda im: preproc_rgb(im, to_hsv=True)

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

PRED_DIMS = OrderedDict([
    ('pred_images', [3, preproc_hsv]),
    ('pred_delta_images', [1, tf.identity]),
    ('pred_flows', [2, lambda flow: tf.clip_by_value(flow, -2., 2.)])
])

INP_SEQUENCE_LEN = 4
OBJECT_METRICS = ['recall', 'mIoU_matched', 'boundary_f_measure']
FOREGROUND_ARI = ['foreground_ari']
METRICS = [
    [metric + '_t' + str(t) for t in range(INP_SEQUENCE_LEN)]
    for metric in OBJECT_METRICS + FOREGROUND_ARI
]
SAVE_TENSORS = []
for metric in METRICS:
    SAVE_TENSORS.extend(metric)

config = {
    'save_to_gfs': SAVE_TENSORS,
    'data_params': {
        'func': TdwSequenceDataProvider,
        'delta_time': 1,
        'enqueue_batch_size': 20,
        'buffer_mult': 256,
        'max_depth': 20.,
        'get_segments': True,
        'get_delta_images': True,
        'motion_filter': True,
        'motion_thresh': 0.03,
        'motion_area_thresh': 0.1,
        'train_filter_rule': None,
        'val_filter_rule': None,
        'shuffle_val': True,
        'shuffle_seed': 0,
        'resizes': RESIZES,
        'sources': [IMAGES, DEPTHS, NORMALS, OBJECTS, PMAT],
        'n_tr_per_dataset': 102400,
        'n_val_per_dataset': 10240
    },
    'validation_params': {
        'object_metrics': {
            'targets': {
                'func': eval_metrics.get_pred_and_gt_segments,
                'segments_key': 'spatial/level2_segments',
                'gt_key': 'segments',
                'imsize': [64,64],
                'compute_matched': 1,
                'agg_mean': 1,
                'filter_less_than': 7
            },
            'online_agg_func': eval_metrics.object_mask_and_boundary_metrics,
            'agg_func': eval_metrics.agg_mean_per_time,
            'val_length': 2000
        },
    #     'foreground_ari': {
    #         'targets': {
    #             'func': eval_metrics.get_foreground_ari,
    #             'segments_key': 'spatial/level3_segments',
    #             'gt_key': 'segments',
    #             'max_objects': 64,
    #             'filter_less_than': 64
    #         },
    #         'online_agg_func': eval_metrics.filter_results_by_num_objects,
    #         'agg_func': eval_metrics.agg_mean,
    #         'val_length': 2000
    #     }
    }
}
