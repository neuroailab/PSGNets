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
DELTA_RGB = 'delta_rgb'
PMAT = 'projection_matrix'
VALID = 'valid'
INPUT_SIZE = 256
RESIZES = {k:[INPUT_SIZE]*2 for k in [IMAGES, DEPTHS, NORMALS, OBJECTS]}
INP_SEQUENCE_LEN = 4

LEVEL_NAMES = ['level1', 'level2', 'level3']
OUTPUT_NAMES = ['child_nodes', 'parent_nodes', 'parent_segment_ids']
INPUT_NAMES = [IMAGES, OBJECTS, DELTAS]

SAVE_TENSORS = ['inputs/' + nm for nm in INPUT_NAMES] + ['dims/' + lev + '_dims' for lev in LEVEL_NAMES]
for lev in LEVEL_NAMES:
    for nm in OUTPUT_NAMES:
        SAVE_TENSORS.append(lev + '/' + nm)

config = {
    'save_to_gfs': SAVE_TENSORS,
    'data_params': {
        'func': TdwSequenceDataProvider,
        'delta_time': 1,
        'enqueue_batch_size': 20,
        'buffer_mult': 1000,
        # 'buffer_mult': 20,
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
        'levels': {
            'targets': {
                'func': eval_metrics.get_level_outputs,
                'level_names': LEVEL_NAMES,
                'output_names': OUTPUT_NAMES,
                'input_names': INPUT_NAMES,
            },
            'online_agg_func': eval_metrics.append_each_val,
            'agg_func': eval_metrics.concatenate_vals,
            'val_length': 8 * INP_SEQUENCE_LEN
        }
    }
}
