from collections import OrderedDict
import tensorflow.compat.v1 as tf
from tfutils import optimizer
import psgnets.models as models
from psgnets.models.convrnn.convrnn_model import ConvRNN
from psgnets.models.decoding import QtrDecoder, DeltaImages, DEFAULT_PRED_DIMS
import psgnets.ops as ops
from psgnets.trainval.utils import collect_and_flatten, total_loss
import psgnets.trainval.eval_metrics as eval_metrics
from psgnets.data.tdw_data import TdwSequenceDataProvider
import psgnets.models.losses as losses
from psgnets.models.preprocessing import preproc_hsv, preproc_rgb, delta_images

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
INPUT_NAMES = [IMAGES, DEPTHS, NORMALS, OBJECTS, VALID, DELTAS, PMAT]
DECODERS = ['qtr_level1', 'qtr_level2']
OUTPUT_NAMES = ['sampled_pred_attrs']
TENSOR_NAMES = PRED_DIMS.keys()
SAVE_TENSORS = ['inputs/' + nm for nm in INPUT_NAMES]
SAVE_TENSORS += ['spatial/level' + str(lev) + '_segments' for lev in [1,2]]
for decoder in DECODERS:
    for out_nm in OUTPUT_NAMES:
        for tens_nm in TENSOR_NAMES:
            SAVE_TENSORS.append(decoder + '/' + out_nm + '/' + tens_nm)

config = {
    'save_to_gfs': SAVE_TENSORS,
    'data_params': {
        'func': TdwSequenceDataProvider,
        'delta_time': 2,
        'enqueue_batch_size': 20,
        'buffer_mult': 1000,
        'max_depth': 20.,
        'shuffle_val': True,
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
    'validation_params': {
        'rendered': {
            'targets': {
                'func': eval_metrics.get_decoder_outputs,
                'decoder_names': DECODERS,
                'output_names': OUTPUT_NAMES,
                'tensor_names': TENSOR_NAMES,
                'input_names': [IMAGES, DELTAS, OBJECTS],
                'segment_names': ['level2']
            },
            'online_agg_func': eval_metrics.append_each_val,
            'agg_func': eval_metrics.concatenate_vals,
            'val_length': 32
        }
    }
}
