import copy
import h5py
import numpy as np
import json
import os
import tensorflow as tf
from PIL import Image
import io
import time
import sys
import glob
import _pickle as cPickle
from tqdm import trange, tqdm
from multiprocessing import Pool
from joblib import Parallel, delayed
from tdwflextools.postprocess.deprecated import tfrecords_utils as utils
from tdwflextools.postprocess.deprecated.tfrecords_utils import Attribute
from scipy import signal
import argparse

# logging
import logging
logging.basicConfig(level=logging.INFO)

def get_arguments():
    parser = argparse.ArgumentParser(
            description='TDW data generation controller')
    # General settings
    parser.add_argument(
        '-b', '--base_dir',
        default=None, type=str,
        required=True,
        help="Base directory")
    parser.add_argument(
        '-d', '--dataset_name',
        default='dataset0', type=str,
        help='Dataset name')
    parser.add_argument(
        '-o', '--output_dir',
        default='.', type=str,
        help='Output directory')
    parser.add_argument(
        '-g', '--group_size',
        default=10, type=int,
        help='Number of files to read/write at once')
    parser.add_argument(
        '-s', '--batch_size',
        default=256, type=int,
        help='Number of frames to write in one batch')
    parser.add_argument(
        '-p', '--prefix',
        default='trial', type=str,
        help='Prefix string to each output tfrecord name')
    return parser.parse_args()


args = get_arguments()
# Configs
IS_DEBUG = False
BATCH_SIZE = None
KEEP_EXISTING_FILES = False
WITH_IMAGES = True
USE_FLOW = False
POSTFIX = ''
IMAGE_NAMES = [nm + POSTFIX for nm in ['images', 'depths', 'normals', 'objects']]
STATIC_NAMES = ['projection_matrix', 'camera_matrix']

# Paths
DATASET_NAME = args.dataset_name
BASE_DIR = args.base_dir
OUT_DIR = args.output_dir
PREFIX = args.prefix

HDF5_FILE_PATHS = sorted(glob.glob(os.path.join(BASE_DIR, DATASET_NAME + '.hdf5')))
NUM_FILES = len(HDF5_FILE_PATHS)
FILE_GROUP_SIZE = args.group_size
NUM_GROUPS = NUM_FILES // FILE_GROUP_SIZE
NUM_GROUPS += (NUM_FILES % FILE_GROUP_SIZE) > 0
BATCH_SIZE = args.batch_size

NEW_TFRECORD_TRAIN_PATH = os.path.join(OUT_DIR, 'new_tfdata')
NEW_TFRECORD_VAL_PATH = os.path.join(OUT_DIR, 'new_tfvaldata')

# static data
MAX_N_DYNAMIC_OBJECTS = 3
HEIGHT = 256
WIDTH = 256

FALSE_INDICATORS = ['is_acting']

# Attributes to generate
ATTRIBUTES = [
    Attribute('is_moving', (1,), tf.float32),
    Attribute('is_not_teleporting', (1,), tf.float32),
    Attribute('is_acting', (1,), tf.float32),
    Attribute('is_object_in_view', (MAX_N_DYNAMIC_OBJECTS,), tf.int32),
    Attribute('object_ids', (MAX_N_DYNAMIC_OBJECTS,), tf.int32),
    Attribute('reference_ids', (2,), tf.int32),    
    Attribute('camera_matrix', (4,4), tf.float32),
    Attribute('projection_matrix', (4,4), tf.float32),
]


if WITH_IMAGES:
    ATTRIBUTES.extend([
        Attribute(im_nm, (HEIGHT, WIDTH, 3), tf.uint8) for im_nm in IMAGE_NAMES
        ])


if USE_FLOW:
    ATTRIBUTES.extend([
        Attribute('flows' + POSTFIX, (HEIGHT, WIDTH, 3), tf.uint8), 
        ])

if IS_DEBUG:
    ATTRIBUTES = []

ATTRIBUTE_NAMES = [attr.name for attr in ATTRIBUTES]
ATTRIBUTE_SHAPES = dict([(attr.name, attr.shape) for attr in ATTRIBUTES])
ATTRIBUTE_DTYPES = dict([(attr.name, attr.dtype) for attr in ATTRIBUTES])
ATTRIBUTES_TO_HDF5 = {
    'images': '_img',
    'depths': '_depth',
    'normals': '_normals',
    'objects': '_id',
    'camera_matrix': 'camera_matrix',
    'projection_matrix': 'projection_matrix'
}

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def get_camera_matrix(hf):
    return np.reshape(np.array(hf['static']['camera_matrix']), [4,4]).astype(np.float32)

def get_projection_matrix(hf):
    return np.reshape(np.array(hf['static']['projection_matrix']), [4,4]).astype(np.float32)    

def get_subset_indicators(actions, batch_number=1):
    '''returns num_frames x 4 np array
    binary indicators
    '''
    action_types = [act['action_type'] for act in actions]
    is_not_waiting = []
    is_acting = []
    is_not_teleporting = []
    if len(action_types) != BATCH_SIZE:
        raise ValueError('Not enough data. Batch length: %d' % len(action_types))
    for act_type in action_types:
        not_waiting_now = int('WAIT' not in act_type)
        not_tele_now = int('TELEPORT' not in act_type)
        not_acting_now = int('ACT' not in act_type and 'APPLY' not in act_type)
        assert not_waiting_now + not_tele_now + not_acting_now == 2, \
                (not_waiting_now, not_tele_now, not_acting_now, act_type)
        acting_now = 1 - not_acting_now
        is_not_waiting.append(np.array([not_waiting_now]).astype(np.float32))
        is_not_teleporting.append(np.array([not_tele_now]).astype(np.float32))
        is_acting.append(np.array([acting_now]).astype(np.float32))

    # first 10 frames and last frame are not to be used as well as 
    # one frame before and one frame after a teleport
    def get_is_moving(safety_distance=1, do_not_use_first_frames = False):
        is_moving = np.squeeze(copy.deepcopy(np.array(is_not_teleporting)))
        is_teleporting = np.concatenate(([0], np.equal(
            is_moving, 0).view(np.int8), [0]))
        is_teleporting = signal.convolve(is_teleporting, \
                [1] * (safety_distance * 2 + 1), 'same')[1:-1]
        is_moving = 1 - is_teleporting
        is_moving[:safety_distance] = 0
        is_moving[-safety_distance:] = 0
        if do_not_use_first_frames and batch_number == 0:
            is_moving[0:10] = np.array([0]).astype(np.float32)
        is_moving = is_moving[:,np.newaxis]
        is_moving = is_moving.astype(np.float32)
        return is_moving

    is_moving = get_is_moving(1)
    is_moving2 = get_is_moving(2)
    is_moving3 = get_is_moving(3)

    return {'is_not_waiting' : is_not_waiting,
            'is_acting' : is_acting,
            'is_not_teleporting' : is_not_teleporting, 
            'is_moving': is_moving,
            'is_moving2': is_moving2,
            'is_moving3': is_moving3}


# def get_reference_ids(file_num, bn):
#     if WORLD_DATASET_NUMBER is not None:
#         file_num = WORLD_DATASET_NUMBER
#     return [np.array([file_num, bn * BATCH_SIZE + i]).astype(np.int32) \
#             for i in range(BATCH_SIZE)]

def get_batch_data(file_num_and_bn, with_images = True):
    file_num = file_num_and_bn[0]
    bn = file_num_and_bn[1]
    f = FILES[file_num]
    initial_distances = get_initial_particle_distances(f)
    start = time.time()
    if with_images:
        objects = f['objects1'][bn * BATCH_SIZE : (bn + 1) * BATCH_SIZE]
        images = f['images1'][bn * BATCH_SIZE : (bn + 1) * BATCH_SIZE]
        depths = f['depths1'][bn * BATCH_SIZE : (bn + 1) * BATCH_SIZE]
        normals = f['normals1'][bn * BATCH_SIZE : (bn + 1) * BATCH_SIZE]
        if USE_FLOW:
            flows = f['flows1'][bn * BATCH_SIZE : (bn + 1) * BATCH_SIZE]
    else:
        objects = np.zeros((BATCH_SIZE, HEIGHT, WIDTH, 3))

    unpadded_particles = \
            f['particles'][bn * BATCH_SIZE : (bn + 1) * BATCH_SIZE]
    #pad particles with zeros to 10,000 * 7 dim
    particles = np.zeros((BATCH_SIZE, MAX_N_PARTICLES * 7))
    particles_mask = np.zeros((BATCH_SIZE, MAX_N_PARTICLES))
    for p_index, p in enumerate(unpadded_particles):
        particles[p_index,:int(p.shape[0])] = p
        particles_mask[p_index,:int(p.shape[0] / 7)] = 1

    num_particles = np.sum(particles_mask, axis=1, keepdims=True).astype(np.int32)

    particles = particles.astype(np.float32)
    orig_particles = copy.deepcopy(particles)
    particles_mask = particles_mask.astype(np.float32)

    actions_raw = f['actions'][bn * BATCH_SIZE : (bn + 1) * BATCH_SIZE]
    actions_raw = [json.loads(act) for act in actions_raw]
    indicators = get_subset_indicators(actions_raw, bn)
    worldinfos = [json.loads(info) for info in \
            f['worldinfo'][bn * BATCH_SIZE : (bn + 1) * BATCH_SIZE]]
    camera_matrixs = [get_camera_matrix(info) \
            for info in worldinfos]
    projection_matrixs = [get_projection_matrix(info) \
            for info in worldinfos]
    actions, particle_forces = get_actions(actions_raw, \
            worldinfos)
    static_particles, is_settled = get_static_data(worldinfos)
    object_data, is_object_there, is_object_in_view, \
            is_colliding_static, is_colliding_dynamic, gravity, \
            stiffness, original_id_order, segmentation_color, \
            materials, fixed_particles = \
            get_object_data(worldinfos, objects, actions_raw, \
            indicators, camera_matrixs, projection_matrixs)
    max_coordinates, min_coordinates, full_particles, full_particles_agent, \
            screen_poses = create_particle_data(particles, \
            particles_mask, object_data, original_id_order, \
            actions, particle_forces, fixed_particles, \
            camera_matrixs, projection_matrixs)
    actions = np.array(actions)[:,:,0:9]
    kNN, collision, static_collision, self_collision = get_kNN(full_particles, \
            static_particles, initial_distances, MAX_N_KNN, bn)
    agent_data = get_agent_data(worldinfos)
    reference_ids = get_reference_ids(file_num, bn)
    good_examples = get_good_examples(bn)

    inafivm = np.array(indicators['is_moving']) * \
            (1 - np.array(indicators['is_acting'])) * \
            np.array(is_object_in_view)[:,0:1]
    inafivm = list(inafivm)

    to_ret = {
            'num_particles': num_particles,
            'actions' : actions, 
            'object_data': object_data, 
            'agent_data': agent_data, 
            'reference_ids': reference_ids, 
            'is_object_there': is_object_there, 
            'is_object_in_view': is_object_in_view, 
            'particles': orig_particles, 
            'max_coordinates': max_coordinates, 
            'min_coordinates': min_coordinates, 
            'full_particles': full_particles, 
            'full_particles_agent': full_particles_agent, 
            'kNN': kNN, 
            'collision': collision, 
            'good_examples': good_examples,
            'is_not_acting_first_in_view_moving': inafivm,
            'is_colliding_static': is_colliding_static, 
            'is_colliding_dynamic': is_colliding_dynamic, 
            'gravity': gravity, 
            'self_collision': self_collision, 
            'static_collision': static_collision, 
            'is_settled': is_settled, 
            'stiffness': stiffness,
            'materials': materials,
            'segmentation_color': segmentation_color,
            'particle_screen_poses': screen_poses,
            'camera_matrix' : camera_matrixs,
            'projection_matrix': projection_matrixs}

    if with_images:
        to_ret.update({
            'images' + POSTFIX: images,
            'objects' + POSTFIX: objects, 
            'depths' + POSTFIX: depths, 
            'normals' + POSTFIX: normals,
            })
        if USE_FLOW:
            to_ret.update({
            'flows' + POSTFIX: flows,
            })

    to_ret.update(indicators)
    for i in range(BATCH_SIZE):
        for k in ATTRIBUTE_SHAPES:
            if ATTRIBUTE_SHAPES[k] is not None:
                assert to_ret[k][i].shape == ATTRIBUTE_SHAPES[k], \
                        (k, to_ret[k][i].shape, ATTRIBUTE_SHAPES[k])
    return to_ret


# def write_stuff(batch_data, writers):
#     start = time.time()
#     batch_size = len(list(batch_data[list(batch_data)[0]]))
#     for k, writer in writers.items():
#         for i in range(batch_size):
#             datum = tf.train.Example( \
#                     features = tf.train.Features( \
#                     feature = {k: _bytes_feature( \
#                     batch_data[k][i].tostring())}))
#             writer.write(datum.SerializeToString())


def write_in_thread(file_num, batches, write_path, prefix):
    if prefix is None:
        prefix = file_num
    # Open writers 
    output_files = [os.path.join(write_path, attr_name, 
        str(prefix) + '-' + str(batches[0]) + '-' + str(batches[-1]) \
                + '.tfrecords') for attr_name in ATTRIBUTE_NAMES]
    if KEEP_EXISTING_FILES:
        for i, output_file in enumerate(output_files):
            if os.path.isfile(output_file):
                print('Skipping file %s' % output_file)
                return 
    writers = dict((attr_name, tf.python_io.TFRecordWriter(file_name)) \
            for (attr_name, file_name) in zip(ATTRIBUTE_NAMES, output_files))

    for _, batch in enumerate(batches):
        try:
            batch_data_dict = get_batch_data((file_num, batch), \
                with_images = WITH_IMAGES)
        #'''
        except ValueError as e:
            print('Error \'%s\' in batch %d - %d! Skipping batch' \
                    % (e, batches[0], batches[-1]))
            # Close writers
            for writer in writers.values():
                writer.close()
            for output_file in output_files:
                os.remove(output_file)
            return
        #'''
        # Write batch
        write_stuff(batch_data_dict, writers)
    # Close writers
    for writer in writers.values():
        writer.close()
    return 0

def build_writers(filename, write_path, prefix):
    # open writers
    output_files = [
        os.path.join(
            write_path, attr_name,\
            str(prefix) + '-' + filename + '.tfrecords')
        for attr_name in ATTRIBUTE_NAMES]
    
    for i, output_file in enumerate(output_files):
        if os.path.isfile(output_file):
            if KEEP_EXISTING_FILES:
                print('Skipping file %s' % output_file)
                return None
            else:
                os.remove(output_file)
            
    writers = {
        attr_name: tf.io.TFRecordWriter(file_name)
        for (attr_name, file_name) in zip(ATTRIBUTE_NAMES, output_files)
    }
    
    return writers, output_files

def get_static_data(hfile, key, shape):
    datum = np.array(hfile['static'][key])
    datum = np.reshape(datum, shape)
    return datum

def get_image_data(hfile, key, frame, shape):
    im = np.array(hfile['frames'][frame]['images'][key])
    im = Image.open(io.BytesIO(im))
    im = np.array(im).reshape(shape)
    return im

def get_indicator_data(name, is_int, shape):
    dtype = np.int32 if is_int else np.float32
    if name in FALSE_INDICATORS:
        indicator = np.zeros(shape=shape, dtype=dtype)
    else:
        indicator = np.ones(shape=shape, dtype=dtype)

    return indicator

def get_reference_id(group, fnum, frame):
    file_id = group*FILE_GROUP_SIZE + fnum
    reference_id = np.array([file_id, frame], dtype=np.int32)
    return reference_id

def get_hdf5_data(hf, attrs, g, fnum):
    # number of frames and objects in this file
    frames = sorted(list(hf['frames'].keys()))
    num_frames = len(frames)
    num_objects = len(hf['static']['object_ids'])

    attribute_shapes = copy.deepcopy(ATTRIBUTE_SHAPES)
    # Alter shapes for this file
    for attr in ['object_ids', 'is_object_in_view']:
        ATTRIBUTE_SHAPES[attr] = (num_objects,)

    attr_data = {}
    for attr in attrs:
        key = ATTRIBUTES_TO_HDF5.get(attr, attr)
        if key in hf['static'].keys():
            data = get_static_data(
                hf, key, shape=ATTRIBUTE_SHAPES[attr])
            data = np.stack(
                [data]*num_frames, axis=0)
        elif attr in IMAGE_NAMES:
            data = np.stack([
                get_image_data(hf, key, frame, shape=ATTRIBUTE_SHAPES[attr])
                for frame in frames], axis=0)
        elif 'is_' in attr: # make up the flags for now
            data = get_indicator_data(key, is_int=('object' in key), shape=ATTRIBUTE_SHAPES[attr])
            data = np.stack([data]*num_frames, axis=0)
        elif attr == 'reference_ids':
            data = np.stack([
                get_reference_id(g, fnum, fi) for fi in range(num_frames)
            ], axis=0)
        else:
            raise ValueError("No HDF5 data exist for attribute %s" % attr)

        # print(attr, key, data.shape, data.dtype)
        attr_data[attr] = data

    return attr_data

def write_data(writer, attr_name, attr_data, batch_size=BATCH_SIZE):
    start = time.time()
    num_examples = attr_data.shape[0]
    if batch_size is None:
        batch_size = num_examples
    # print("batch size", batch_size)
    for ex in range(min([num_examples, batch_size])):
        try:
            datum = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        attr_name: _bytes_feature(
                            attr_data[ex].tostring()
                        )}))
            writer.write(datum.SerializeToString())
        except:
            writer.close()
            return 1
    
    writer.close()
    end = time.time()
    # print("write time: {:2f}".format(end-start))
    return 0
    
def do_write(all_images = True):
    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)
    if not os.path.exists(NEW_TFRECORD_TRAIN_PATH):
        os.mkdir(NEW_TFRECORD_TRAIN_PATH)
    if not os.path.exists(NEW_TFRECORD_VAL_PATH):
        os.mkdir(NEW_TFRECORD_VAL_PATH)

    for nm in ATTRIBUTE_NAMES:
        write_dir_train = os.path.join(NEW_TFRECORD_TRAIN_PATH, nm)
        write_dir_val = os.path.join(NEW_TFRECORD_VAL_PATH, nm)
        if not os.path.exists(write_dir_train):
            os.mkdir(write_dir_train)
        if not os.path.exists(write_dir_val):
            os.mkdir(write_dir_val)
        meta = {
                nm: {
                    'dtype': tf.string,
                    'shape': [],
                    'rawtype': ATTRIBUTE_DTYPES[nm],
                    'rawshape': list(ATTRIBUTE_SHAPES[nm]),
                    }
                }
        if nm in IMAGE_NAMES:
            meta = {nm: {'dtype': ATTRIBUTE_DTYPES[nm],
                'shape': list(ATTRIBUTE_SHAPES[nm])}}
        with open(os.path.join(write_dir_train, 'meta.pkl'), 'wb') as f:
            cPickle.dump(meta, f, protocol=2)
        with open(os.path.join(write_dir_val, 'meta.pkl'), 'wb') as f:
            cPickle.dump(meta, f, protocol=2)

    my_rng = np.random.RandomState(seed = 0)

    # create file groups
    gs = FILE_GROUP_SIZE
    groups = [
        HDF5_FILE_PATHS[i*gs:(i+1)*gs] for i in range(NUM_GROUPS)
    ]

    # write only in one group at a time
    for g, file_group in enumerate(groups):
        # create batch tasks        
        write_tasks = []

        # hdf5 files in this group
        fnames = [
            fpath.split('/')[-1].split('.hdf5')[0]
            for fpath in file_group]
        hfiles = [h5py.File(f, 'r') for f in file_group]
        logging.info("Reading File Group: %d/%d" % (g+1, NUM_GROUPS))

        for fnum, hf in enumerate(tqdm(hfiles, desc='Written tfrecords')):

            # train or val
            # if my_rng.rand() > 0.1:
            if my_rng.rand() > 0.0:                
                write_path = NEW_TFRECORD_TRAIN_PATH
            else:
                write_path = NEW_TFRECORD_VAL_PATH

            # one tfrecord per attribute per hfile
            writers, output_files = build_writers(fnames[fnum], write_path, prefix=PREFIX)

            # get the data
            try:
                data = get_hdf5_data(hf, attrs=writers.keys(), g=g, fnum=fnum)
            except ValueError as e:
                print('Read Error \'%s\' in file %d of group %d, skipping file'\
                      % (e, fnum, g))
                for writer in writers.values():
                    writer.close()
                for of in output_files:
                    os.remove(of)

            # write the data
            for attr, writer in writers.items():
                try:
                    w = write_data(writer, attr, data[attr])
                    if w:
                        raise ValueError("Error in writing attr %s" % attr)
                except ValueError as e:
                    print('Write Error \'%s\' in file %d of group %d, skipping file'\
                          % (e, fnum, g))
                    for writer in writers.value():
                        writer.close()
                    for of in output_files:
                        os.remove(of)
                    break
                    
            # close infile
            hf.close()

    # rs = []
    # pools = []
    # for write_task in write_tasks:
    #     for wt in tqdm(write_task, desc="Written tfrecords"):
    #         write_in_thread(*wt)


if __name__ == '__main__':

    print("BASE DIR: ", args.base_dir)
    print("FILE PATHS: ", args.dataset_name)
    print("NUM FILES: ", NUM_FILES)
    print("GROUP SIZE: ", args.group_size)
    print("OUT DIR: ", args.output_dir)

    do_write()
    # for f in FILES:
    #     f.close()
