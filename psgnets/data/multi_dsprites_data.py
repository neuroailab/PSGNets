# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Multi-dSprites dataset reader."""

import copy
import functools
import tensorflow.compat.v1 as tf
from base import DataProvider

COMPRESSION_TYPE = tf.io.TFRecordOptions.get_compression_type_string('GZIP')
IMAGE_SIZE = [64, 64]
# The maximum number of foreground and background entities in each variant
# of the provided datasets. The values correspond to the number of
# segmentation masks returned per scene.
MAX_NUM_ENTITIES = {
  'binarized': 4,
  'colored_on_grayscale': 6,
  'colored_on_colored': 5
}
BYTE_FEATURES = ['mask', 'image']

def feature_descriptions(max_num_entities, is_grayscale=False):
  """Create a dictionary describing the dataset features.

  Args:
    max_num_entities: int. The maximum number of foreground and background
      entities in each image. This corresponds to the number of segmentation
      masks and generative factors returned per scene.
    is_grayscale: bool. Whether images are grayscale. Otherwise they're assumed
      to be RGB.

  Returns:
    A dictionary which maps feature names to `tf.Example`-compatible shape and
    data type descriptors.
  """

  num_channels = 1 if is_grayscale else 3
  return {
    'image': tf.FixedLenFeature(IMAGE_SIZE+[num_channels], tf.string),
    'mask': tf.FixedLenFeature(IMAGE_SIZE+[max_num_entities, 1], tf.string),
    'x': tf.FixedLenFeature([max_num_entities], tf.float32),
    'y': tf.FixedLenFeature([max_num_entities], tf.float32),
    'shape': tf.FixedLenFeature([max_num_entities], tf.float32),
    'color': tf.FixedLenFeature([max_num_entities, num_channels], tf.float32),
    'visibility': tf.FixedLenFeature([max_num_entities], tf.float32),
    'orientation': tf.FixedLenFeature([max_num_entities], tf.float32),
    'scale': tf.FixedLenFeature([max_num_entities], tf.float32),
  }


def _decode(example_proto, features):
  # Parse the input `tf.Example` proto using a feature description dictionary.
  single_example = tf.parse_single_example(example_proto, features)
  for k in BYTE_FEATURES:
    single_example[k] = tf.squeeze(tf.decode_raw(single_example[k], tf.uint8),
                                   axis=-1)
    # To return masks in the canonical [entities, height, width, channels] format,
    # we need to transpose the tensor axes.
  single_example['mask'] = tf.transpose(single_example['mask'], [2, 0, 1, 3])
  return single_example


def dataset(tfrecords_path, dataset_variant, read_buffer_size=None,
            map_parallel_calls=None):
  """Read, decompress, and parse the TFRecords file.

  Args:
    tfrecords_path: str. Path to the dataset file.
    dataset_variant: str. One of ['binarized', 'colored_on_grayscale',
      'colored_on_colored']. This is used to identify the maximum number of
      entities in each scene. If an incorrect identifier is passed in, the
      TFRecords file will not be read correctly.
    read_buffer_size: int. Number of bytes in the read buffer. See documentation
      for `tf.data.TFRecordDataset.__init__`.
    map_parallel_calls: int. Number of elements decoded asynchronously in
      parallel. See documentation for `tf.data.Dataset.map`.

  Returns:
    An unbatched `tf.data.TFRecordDataset`.
  """
  if dataset_variant not in MAX_NUM_ENTITIES:
    raise ValueError('Invalid `dataset_variant` provided. The supported values'
                     ' are: {}'.format(list(MAX_NUM_ENTITIES.keys())))
  max_num_entities = MAX_NUM_ENTITIES[dataset_variant]
  is_grayscale = dataset_variant == 'binarized'
  raw_dataset = tf.data.TFRecordDataset(
    tfrecords_path, compression_type=COMPRESSION_TYPE,
    buffer_size=read_buffer_size)
  features = feature_descriptions(max_num_entities, is_grayscale)
  partial_decode_fn = functools.partial(_decode, features=features)
  return raw_dataset.map(partial_decode_fn,
                         num_parallel_calls=map_parallel_calls)


class MultiDspritesData(DataProvider):

  DATA_PATH = "/data4/dbear/tdw_datasets"
  def __init__(
      self,
      data_paths="",
      file_pattern="multi_dsprites*.tfrecords",
      sources=BYTE_FEATURES,
      variant='colored_on_colored',
      is_training=True,
      temporal=True,
      sequence_len=1,
      get_objects=True,
      get_valid=True,
      map_parallel_calls=24,
      read_buffer_size=1024,
      q_cap=1024,
      **kwargs
  ):

    self.data = data_paths
    self.variant = variant
    self.sources = sources
    self.temporal = temporal
    self.sequence_len = sequence_len
    self.get_objects = get_objects
    self.get_valid = get_valid

    self.is_training = is_training
    self.file_pattern = file_pattern
    self.map_pcall_num = map_parallel_calls
    self.read_buffer_size = read_buffer_size
    self.q_cap = q_cap

  @staticmethod
  def get_data_params(batch_size, sequence_len=1, dataprefix=DATA_PATH, **kwargs):
    data_params = copy.deepcopy(kwargs)
    data_params['data_paths'] = dataprefix
    data_params['temporal'] = (sequence_len is not None)
    data_params['sequence_len'] = sequence_len
    return copy.deepcopy(data_params), copy.deepcopy(data_params)


  def input_fn(self, batch_size, train, **kwargs):

    self.is_training = train
    tfr_list = self.get_tfr_filenames(self.data)
    tfr_path = tfr_list[0]
    print("tfr path", tfr_path)
    data = dataset(tfr_path, self.variant, self.read_buffer_size, self.map_pcall_num)

    if self.is_training:
      data = data.shuffle(buffer_size=self.q_cap)

    data = data.batch(batch_size)
    iterator = tf.data.make_one_shot_iterator(data)
    inputs = iterator.get_next()

    if self.get_objects:
      masks = inputs['mask']
      segments = tf.cast(tf.argmax(masks, axis=1), tf.int32)
      inputs['objects'] = segments

    inputs = {k:v for k,v in inputs.items() if k in ['image', 'objects']}
    inputs['images'] = inputs.pop('image')

    if self.get_valid:
      inputs['valid'] = tf.ones_like(inputs['images'][...,0:1])

    for k,inp in inputs.items():
      shape = inp.shape.as_list()
      inputs[k] = tf.reshape(inp, [batch_size]+shape[1:])

    if self.temporal:
      inputs = {k:v[:,tf.newaxis] for k,v in inputs.items()}

    if (self.sequence_len > 1) and self.temporal:
      inputs = {
        k: tf.concat([v]*self.sequence_len, axis=1)
        for k,v in inputs.items()
      }
      inputs['delta_rgb'] = tf.cast(tf.zeros_like(inputs['images']), tf.float32)
      inputs['delta_images'] = inputs['delta_rgb'][...,0:1]

    return inputs

if __name__ == '__main__':

  import os
  os.environ['CUDA_VISIBLE_DEVICES'] = ''
  BATCH_SIZE = 8
  TRAIN = True

  train_data, val_data = MultiDspritesData.get_data_params(BATCH_SIZE)
  data_provider = MultiDspritesData(**(train_data if TRAIN else val_data))
  func = data_provider.input_fn
  inputs = func(BATCH_SIZE, TRAIN)
  print("data", inputs)

  sess = tf.Session()
  inputs = sess.run(inputs)
  print(inputs['images'].shape, inputs['objects'].shape)
  import numpy as np
  print(np.unique(inputs['objects'][:,0,...,0]))

  import pickle
  with open('/home/dbear/multi_dsprites_ims.pkl', 'wb') as f:
    pickle.dump(inputs, f)
    f.close()
