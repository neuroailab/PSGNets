import numpy as np
import json
import tensorflow.compat.v1 as tf
import os
try:
    import cPickle
except:
    import pickle as cPickle
    import pdb
    import sys
    import copy

from .base import DataProvider
from .utils import *
from psgnets.models.preprocessing import delta_images, delta_rgb

NEAR = 0.01
FAR = 160.
IMAGE_SOURCES=['images', 'depths', 'normals', 'objects', 'flows', 'projection_matrix', 'camera_matrix', 'valid', 'segments']

def auto_fill_dataset_path(dataset, dataprefix=None, validate_on_train_data=False):

    dataprefix = dataprefix if dataprefix is not None else '/mnt/fs1/datasets'

    data_path = os.path.join(dataprefix, dataset, 'new_tfdata')
    if validate_on_train_data:
        valdata_path = data_path
    else:
        valdata_path = os.path.join(dataprefix, dataset, 'new_tfvaldata')

    return data_path, valdata_path

def combine_interaction_data(
        data_paths, nums_examples, group_paths
        ):
    data = []
    curr_indx = 0
    for data_path, num_examples, group_path in \
            zip(data_paths, nums_examples, group_paths):
        group_data = {'which_dataset': curr_indx}
        data.append((data_path, num_examples, group_data))
        curr_indx +=1
    return data

class TdwSequenceDataProvider(DataProvider):
    '''
    Sequence data provider, outputs a sequence of data of the requested length.
    This data provider supports data filtering
    This data provider uses new dataset interface in tensorflow
    '''
    DATA_PATH = "/data4/dbear/tdw_datasets"
    def __init__(
            self,
            data_paths,
            sources,
            sequence_len,
            is_training,
            image_sources=IMAGE_SOURCES,
            enqueue_batch_size=4,
            buffer_size=1024,
            resizes={},
            get_delta_images=False,
            motion_filter=False,
            motion_thresh=0.1,
            motion_area_thresh=0.05,
            delta_time=1,
            filter_rule=None,
            special_delta=1,
            depth_kwargs={'new':True, 'normalization':100.1, 'background_depth':FAR},
            max_depth=FAR,
            get_segments=False,
            file_pattern='*.tfrecords',
            shuffle_seed=0,
            shuffle_queue_seed=None,
            shuffle_queue=True,
            shuffle_val=False,
            map_pcall_num=48,
            sub_key_list=[],
            num_particle_filter=None,
            *args,
            **kwargs):
        '''
        resizes: dict with {source: resize_shape} pairs. resize_shapes are [H,W]
        '''
        # list of paths to the datasets
        self.data = data_paths

        # list of keys to each data source. Each key must be name of a directory in [self.data_path]/new_tfdata/
        self.sources = sources
        self.image_sources = image_sources or self.sources

        # the number of frames in each output sequence of data
        self.sequence_len = sequence_len

        # for depth
        self.depth_kwargs = copy.deepcopy(depth_kwargs)
        self.max_depth = max_depth

        # filter on motion
        self.get_delta_images = delta_images
        self.motion_filter = motion_filter
        self.motion_thresh = motion_thresh
        self.motion_area_thresh = motion_area_thresh

        self.enqueue_batch_size = enqueue_batch_size
        self.resizes = resizes
        self.delta_time = delta_time
        self.filter_rule = filter_rule
        self.special_delta = special_delta
        self.get_segments = get_segments
        self.map_pcall_num = map_pcall_num
        self.is_training = is_training
        self.shuffle_queue = shuffle_queue
        self.file_pattern = file_pattern
        self.shuffle_val = shuffle_val
        self.shuffle_seed = shuffle_seed
        self.shuffle_queue_seed = shuffle_queue_seed
        self.buffer_size = buffer_size
        self.all_sources = copy.deepcopy(sources)
        self.sub_key_list = sub_key_list
        self.num_particle_filter = num_particle_filter

        assert self.delta_time >= 1, \
            ('delta time has to be at least 1')
        assert self.sequence_len >= 1, \
            ('sequence length has to be at least 1')
        assert self.enqueue_batch_size >= self.sequence_len * self.delta_time, \
            ('batch size has to be at least equal to sequence length ' + \
             'times delta time')

    @staticmethod
    def get_data_params(batch_size,
                        sequence_len,
                        delta_time=1,
                        resizes={},
                        get_segments=False,
                        buffer_mult=4,
                        num_parallel_calls=48,
                        dataset_names=None,
                        dataprefix='/mnt/fs1/Datasets/',
                        sources=[],
                        n_tr_per_dataset=(74*1024),
                        n_val_per_dataset=(8*1024),
                        enqueue_batch_size=None,
                        train_filter_rule=None,
                        val_filter_rule=None,
                        shuffle_seed=0,
                        shuffle_val=False,
                        validate_on_train_data=False,
                        **kwargs
    ):

        # from new_data import SequenceNewDataProvider
        data_path = []
        valdata_path = []
        for dataset in dataset_names:
            _data_path, _valdata_path = auto_fill_dataset_path(dataset,
                                                               dataprefix=dataprefix,
                                                               validate_on_train_data=validate_on_train_data)
            data_path.append(_data_path)
            valdata_path.append(_valdata_path)

        ns_train_examples = [n_tr_per_dataset] * len(dataset_names)
        ns_val_examples = [n_val_per_dataset] * len(dataset_names)

        enqueue_batch_size = batch_size if enqueue_batch_size is None else enqueue_batch_size

        data_params_base = {
            'main_source_key':'full_particles',
            'enqueue_batch_size':enqueue_batch_size,
            'sources': sources,
            'sequence_len': sequence_len,
            'resizes': resizes,
            'get_segments': get_segments,
            'delta_time': delta_time,
            'special_delta':1,
            'shuffle_seed':shuffle_seed,
            'shuffle_queue_seed':shuffle_seed,
            'shuffle_queue': True,
            'shuffle_val': shuffle_val,
            'filter_rule': None,
            'buffer_size': batch_size * buffer_mult,
            'map_pcall_num': num_parallel_calls,
            'motion_filter': kwargs.get('motion_filter', False),
            'motion_thresh': kwargs.get('motion_thresh', 0.2),
            'motion_thresh': kwargs.get('motion_area_thresh', None)
        }

        train_data_params_base = copy.deepcopy(data_params_base)
        train_data_params_base['filter_rule'] = train_filter_rule

        val_data_params_base = copy.deepcopy(data_params_base)
        val_data_params_base['filter_rule'] = val_filter_rule

        train_data_params = {
            'data_paths': combine_interaction_data(data_path, ns_train_examples, [None] * len(dataset_names)),
            'is_training': True
        }
        train_data_params.update(train_data_params_base)

        val_data_params = {
            'data_paths': combine_interaction_data(valdata_path, ns_val_examples, [None] * len(dataset_names)),
            'is_training': False
        }
        val_data_params.update(val_data_params_base)

        return train_data_params, val_data_params


    # make it each example wise, rather than batch wise
    def apply_filter(self, data):
        sequence_len = tf.constant(self.sequence_len, dtype = tf.int32)
        for f in self.filter.keys:
            data[f] = tf.cast(data[f], tf.bool)
            # Add the batch dimension
            data[f] = tf.expand_dims(data[f], axis=0)
            # combine filters according to specified filter rule
        master_filter = self.filter.eval(data)
        # check if ALL binary labels within sequence are not zero
        master_filter_sum = tf.reduce_sum(tf.cast(master_filter, tf.int32))
        # gather positive examples for each data entry

        return tf.equal(master_filter_sum, sequence_len)

    def apply_motion_filter(self, data):
        # ims = tf.cast(data['images'], tf.float32) / 255.
        # intensities = tf.reduce_mean(ims, axis=-1) # [T,H,W]
        # delta_ims = tf.abs(intensities[1:] - intensities[:-1]) # [T-1,H,W]
        delta_ims = delta_images(data['images'][tf.newaxis], thresh=None)[0]

        if self.motion_area_thresh is not None:
            moving = tf.cast(delta_ims > self.motion_thresh, tf.float32)
            moving_fraction = tf.reduce_mean(moving, axis=[1,2])
            motion_filter = tf.reduce_any(moving_fraction > self.motion_area_thresh)
        else:
            motion_filter = tf.reduce_max(delta_ims) > self.motion_thresh

        return motion_filter

    def enqueue_many_func(self, all_tensors):
        return tf.data.Dataset.zip(
            {key: tf.data.Dataset.from_tensor_slices(value)
                    for key, value in all_tensors.items()})

    def postproc_each(self, str_loaded, source):
        all_metas = self.meta_dict[source]
        for curr_source, curr_meta in all_metas.items():
            if curr_meta['dtype'] == tf.uint8: # images and objects are encoded as such
                curr_meta['rawshape'] = curr_meta['shape']
                curr_meta['rawtype'] = curr_meta['dtype']
                curr_meta['shape'] = []
                curr_meta['dtype'] = tf.string
        keys_to_features = {
            curr_source: tf.FixedLenFeature(
                curr_meta['shape'],
                curr_meta['dtype'],
            )
            for curr_source, curr_meta in all_metas.items()
        }
        parsed = tf.parse_single_example(str_loaded, keys_to_features)

        for each_source, curr_data in parsed.items():
            if curr_data.dtype is tf.string:
                curr_meta = self.meta_dict[source][each_source]
                curr_data = tf.decode_raw(curr_data, curr_meta['rawtype'])
                curr_data = tf.reshape(curr_data, curr_meta['rawshape'])

            if curr_data.dtype==tf.int16:
                curr_data = tf.cast(curr_data, tf.int32)

            # resizing
            resize_shape = self.resizes.get(each_source, None)
            if resize_shape is not None:
                curr_dtype = curr_data.dtype
                # print("resizing: source=", each_source)
                # print("orig data shape", curr_data.shape.as_list(), curr_data.dtype)
                try:
                    curr_data = tf.image.resize_images(curr_data, size=resize_shape, method=1) # nearest neighbor
                except ValueError:
                    print("not resizing %s with shape %s" % (each_source, curr_data.shape))
                # curr_data = tf.cast(
                #     tf.image.resize_bilinear(tf.expand_dims(curr_data, 0), size=resize_shape)[0],
                #     dtype=curr_dtype)
                # print("new data shape", curr_data.shape.as_list(), curr_data.dtype)
            parsed[each_source] = curr_data

        if len(parsed.keys())==1:
            return parsed[parsed.keys()[0]]
        else:
            return parsed

    def parse_standard_tfmeta(self, path_dict):
        meta_dict = {}
        for source in path_dict:
            path = path_dict[source]
            if isinstance(path, str):
                if path.startswith('meta') and path.endswith('.pkl'):
                    mpaths = [path]
                else:
                    assert os.path.isdir(path), path
                    mpaths = filter(
                        lambda x: x.startswith('meta') \
                        and x.endswith('.pkl'),
                        os.listdir(path))
                    mpaths = [os.path.join(path, mp) for mp in mpaths]
            else:
                # in this case, it's a list
                assert isinstance(path, list), "Path should be a list"
                mpaths = path
            d = {}
            for mpath in mpaths:
                d.update(cPickle.load(open(mpath)))
                meta_dict[source] = d
        return meta_dict

    def set_data_shape(self, data):
        shape = data.get_shape().as_list()
        shape[0] = self.enqueue_batch_size
        for s in shape:
            assert s is not None, ("Unknown shape", shape)
            data.set_shape(shape)
        return data

    def create_data_sequence(self, data):
        if self.special_delta==1 and self.delta_time==1:
            data = tf.expand_dims(data, 1)
            data_shape = data.get_shape().as_list()
            data_type = data.dtype
            shift_len = self.enqueue_batch_size - (self.sequence_len - 1)
            shifts = [data[i : i + shift_len] \
                    for i in range(self.sequence_len)]
            return tf.concat(shifts, axis = 1)
        else:
            data = tf.expand_dims(data, 0)
            sequences = [data[:, i : i+self.sequence_len*self.delta_time : \
                              self.delta_time] for i in \
                         range(self.enqueue_batch_size - (self.sequence_len - 1) * \
                               self.delta_time)]
            return tf.concat(sequences, axis = 0)

    def build_one_dataset(self, curr_data):
        # Unpack the data related info, num_examples is not used
        curr_data_path, _, extra_tensors = curr_data

        # Dictionary with keys being source, and values being directories
        self.source_paths = {
            source: os.path.join(curr_data_path, source) \
                for source in self.sources }

        # load filters, add that to source_paths
        if self.filter_rule:
            self.filter = Filter(self.filter_rule)
            for f in self.filter.keys:
                self.source_paths[f] = os.path.join(curr_data_path, f)
                if f not in self.all_sources:
                    self.all_sources.append(f)
        else:
            self.filter = None

        # load metas
        self.meta_dict = self.parse_standard_tfmeta(self.source_paths)

        # Get tfr filenames
        source_lists = {
            source: self.get_tfr_filenames(
                self.source_paths[source])
            for source in self.source_paths}

        # This shuffle needs to be False to keep the order of every attribute
        # the same
        file_datasets = {
            source: tf.data.Dataset.list_files(curr_files, shuffle=False) \
                for source, curr_files in source_lists.items()}

        if self.is_training or self.shuffle_val:
            # Shuffle file names using the same shuffle_seed
            file_datasets = {
                source: curr_dataset.shuffle(
                    buffer_size=len(source_lists.values()[0]),
                    seed=self.shuffle_seed).repeat() \
                    for source,curr_dataset in file_datasets.items()}

        # Create dataset for both
        def _fetch_dataset(filename):
	        buffer_size = 8 * 1024 * 1024
	        dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
	        return dataset

        each_dataset = {
            source: curr_dataset.apply(
                tf.contrib.data.parallel_interleave(
                    _fetch_dataset,
                    cycle_length=1,
                    sloppy=False)) \
                for source,curr_dataset in file_datasets.items()
        }

        # Decode raw first before zip
        each_dataset = {
            source: curr_dataset.map(
                lambda x: self.postproc_each(x, source),
                num_parallel_calls=self.map_pcall_num,
            ) \
                for source, curr_dataset in each_dataset.items()
        }

        # Zip, repeat, batch
        zip_dataset = tf.data.Dataset.zip(each_dataset)
        def _expand_group_keys(value):
            new_value = {}
            for key, curr_value in value.items():
                if key in self.sub_key_list:
                    new_value.update(curr_value)
                else:
                    new_value[key] = curr_value
            return new_value
        zip_dataset = zip_dataset.map(
            _expand_group_keys,
            num_parallel_calls=self.map_pcall_num,
        )
        zip_dataset = zip_dataset.repeat()
        zip_dataset = zip_dataset.batch(self.enqueue_batch_size)

        # Set shape (first dimension to be batchsize)
        zip_dataset = zip_dataset.map(
            lambda x: {
                key: self.set_data_shape(value)
                    for key,value in x.items()},
            num_parallel_calls=self.map_pcall_num)

        # Create sequence for each dataset
        zip_dataset = zip_dataset.map(
            lambda x: {
                key: self.create_data_sequence(value)
                    for key, value in x.items()},
            num_parallel_calls=self.map_pcall_num)

        # Add extra tensors
        def add_extra_tensors(value):
            for extra_key, extra_tensor in extra_tensors.items():
                assert extra_key not in value, "%s already found!" % extra_key
                batch_size = value[value.keys()[0]].get_shape().as_list()[0]
                time = value[value.keys()[0]].get_shape().as_list()[1]
                extra_tensor = tf.constant(extra_tensor, dtype=tf.float32)
                extra_shape = extra_tensor.get_shape().as_list()
                value[extra_key] = tf.tile(
                    tf.reshape(
                        extra_tensor,
                        [1, 1] + extra_shape),
                    [batch_size, time] + [1] * len(extra_shape))
                if extra_key not in self.all_sources:
                    self.all_sources.append(extra_key)
            return value
        zip_dataset = zip_dataset.map(
            add_extra_tensors,
            num_parallel_calls=self.map_pcall_num)

        return zip_dataset

    def get_max_shapes(self, zip_datasets):
        max_shapes = {}

        for each_dataset in zip_datasets:
            curr_shapes = each_dataset.output_shapes
            for source, curr_shape in {k:v for k,v in curr_shapes.items() if k not in self.image_sources}.items():
                curr_shape = curr_shape.as_list()
                if source not in max_shapes:
                    max_shapes[source] = curr_shape
                    assert len(max_shapes[source]) == len(curr_shape), \
                        "Length of shapes should be the same! " \
                        + str(source) + " " + str(curr_shape) \
                        + ", " + str(max_shapes[source])

                max_shapes[source] = list(np.maximum( \
                                                      max_shapes[source], \
                                                      curr_shape))

        return max_shapes

    def pad_tensors(self, zip_datasets):
        max_shapes = self.get_max_shapes(zip_datasets)

        def _pad_up_to_using_0(tensor, max_shape):
            shape = tensor.get_shape().as_list()
            paddings = [[0, m - shape[i]] if m is not None else [0, 0] \
                    for (i, m) in enumerate(max_shape)]
            return tf.pad(
                tensor, paddings, 'CONSTANT', \
                constant_values=0)

        def _pad_to_max_shapes(value):
            for source, max_shape in max_shapes.items():
                mask_key = source + '_mask'
                assert mask_key not in value, "%s mask already found!" % mask_key
                value[mask_key] = _pad_up_to_using_0(
                    tf.ones(tf.shape(value[source]), dtype=tf.bool),
                    max_shape)
                value[mask_key].set_shape(max_shape)
                value[source] = _pad_up_to_using_0(value[source], max_shape)
                value[source].set_shape(max_shape)

                if mask_key not in self.all_sources:
                    self.all_sources.append(mask_key)
            return value

        for idx in range(len(zip_datasets)):
            zip_datasets[idx] = zip_datasets[idx].map(
                _pad_to_max_shapes,
                num_parallel_calls=self.map_pcall_num)
        return zip_datasets

    def concate_datasets(self, zip_datasets):
        zip_dataset = tf.data.Dataset.zip(tuple(zip_datasets))

        def _concate(*value):
            new_value = {}
            all_sources = value[0].keys()
            for source in all_sources:
                new_value[source] = []
                for _each_value in value:
                    new_value[source].append(_each_value[source])
                    new_value[source] = tf.concat(new_value[source], axis=0)
            return new_value
        zip_dataset = zip_dataset.map(
            _concate,
            num_parallel_calls=self.map_pcall_num)
        return zip_dataset

    def build_datasets(self, batch_size):
        # Build dataset for every data path
        zip_datasets = [
            self.build_one_dataset(curr_data)\
                for curr_data in self.data]

        # Pad and concatenate
        zip_datasets = self.pad_tensors(zip_datasets)
        zip_dataset = self.concate_datasets(zip_datasets)

        # "Enqueue_many" it, shuffle it
        zip_dataset = zip_dataset.flat_map(self.enqueue_many_func)
        # Apply filters
        if self.filter:
            zip_dataset = zip_dataset.filter(self.apply_filter)
        if self.motion_filter:
            zip_dataset = zip_dataset.filter(self.apply_motion_filter)
        if (self.is_training or self.shuffle_val) and self.shuffle_queue:
            # Shuffle it
            zip_dataset = zip_dataset.shuffle(
                buffer_size=self.buffer_size,
                seed=self.shuffle_queue_seed
                # seed=None,
                    )
            # Batch it again
        zip_dataset = zip_dataset.batch(batch_size, drop_remainder=True)
        zip_dataset = zip_dataset.prefetch(2)

        return zip_dataset

    def preproc_segment_ids(self, segment_ids):
        assert len(segment_ids.shape.as_list()) == 5
        B,T,H,W,C = segment_ids.shape.as_list()
        assert B <= 128 and segment_ids.dtype == tf.uint8, "max hash value must be < 256**4 / 2 = 2^16"
        # add batch values to make the hashing unique
        b_inds = tf.tile(tf.reshape(tf.range(B, dtype=tf.int32), [B,1,1,1,1]), [1,T,H,W,C])
        segment_ids = tf.concat([b_inds, tf.cast(segment_ids, tf.int32)], axis=-1)
        segment_ids = object_id_hash(segment_ids, dtype_out=tf.int32, val=256)
        _, segment_ids = tf.unique(tf.reshape(segment_ids, [-1]))
        segment_ids = tf.reshape(segment_ids, [B,T,H,W])
        segment_ids = segment_ids - tf.reduce_min(segment_ids, axis=[1,2,3], keepdims=True)
        return segment_ids

    def preproc_batch(self, input_dict):
        input_dict['valid'] = tf.logical_and(
            tf.reduce_sum(input_dict['normals'], axis=-1, keepdims=True) > tf.cast(0, input_dict['normals'].dtype),
            read_depths_image(input_dict['depths'], **self.depth_kwargs) <= self.max_depth)

        # preproc segment ids
        if self.get_segments:
            segment_ids = input_dict['objects']
            input_dict['segments'] = self.preproc_segment_ids(segment_ids)

        if self.get_delta_images:
            input_dict['delta_images'] = delta_images(input_dict['images'], thresh=None)
            input_dict['delta_rgb'] = delta_rgb(input_dict['images'])

        return input_dict

    # entry point for TFUtils
    def input_fn(self, batch_size, train, **kwargs):
        zip_dataset = self.build_datasets(batch_size)
        zip_iter = zip_dataset.make_one_shot_iterator()
        input_dict = zip_iter.get_next()

        self.preproc_batch(input_dict)

        return input_dict

if __name__ == '__main__':

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    # DATAPATH = '/mnt/fs1/datasets/'
    # DATASETS = ['ball_hits_tower_random_experiment', 'ball_hits_primitive_experiment']

    # DATAPATH = '/mnt/fs4/cfan/tdw-agents/data/'
    # DATASETS = ['sphere_static']

    DATAPATH = '/data4/dbear/tdw_datasets/'
    DATASETS = ['playroom_v1']

    SOURCES = ['images', 'depths', 'normals', 'objects', 'projection_matrix', 'camera_matrix', 'reference_ids']
    BATCH_SIZE = 3
    SEQUENCE_LENGTH = 4
    TRAIN = False
    NUM_STEPS = 1
    DT = 1

    tfrecord_dir = 'new_tfdata' if TRAIN else 'new_tfvaldata'
    data_paths = [os.path.join(DATAPATH, d, tfrecord_dir) for d in DATASETS]
    data_paths = combine_interaction_data(
        data_paths,
        nums_examples=[1]*len(DATASETS),
        group_paths=[None]*len(DATASETS),
    )

    # for filtering out frames with teleports and actions
    filter_rule = (
        moving_and_any_inview_and_not_acting_func,
        ['is_moving', 'is_object_in_view', 'is_acting']
    )

    file_pattern = "trial-001*.tfrecords"

    data_provider = TdwSequenceDataProvider(
        data_paths=data_paths,
        sources=SOURCES,
        sequence_len=SEQUENCE_LENGTH,
        get_segments=True,
        filter_rule=None,
        file_pattern=file_pattern,
        is_training=TRAIN,
        buffer_size=1024,
        enqueue_batch_size=20,
        shuffle_queue=True,
        shuffle_val=True,
        shuffle_seed=1,
        delta_time=DT
    )

    for i in range(NUM_STEPS):
        inputs = data_provider.input_fn(BATCH_SIZE, TRAIN)
        print("batch number: %d\n" % i)
        for k,tensor in inputs.items():
            if '_mask' not in k:
                print("tensor name: %s" % k)
                print("tensor dtype: %s" % tensor.dtype)
                print("tensor shape: %s\n" % tensor.shape.as_list())
                print("=================================\n")

    sess = tf.Session()
    outdata = sess.run({k:v for k,v in inputs.items() if '_mask' not in k})
    for k,data in outdata.items():
        print(k, data.shape, data.dtype)

    with open('/home/dbear/new_tdw_tfrdata.pkl', 'wb') as f:
        cPickle.dump(outdata, f, protocol=-1)

    # print(segs.shape)
    # print([[(i,t,np.unique(segs[i][t])) for t in range(SEQUENCE_LENGTH)] for i in range(BATCH_SIZE)])
