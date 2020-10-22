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
"""CLEVR (with masks) dataset reader."""

import copy
import tensorflow.compat.v1 as tf
from base import DataProvider

COMPRESSION_TYPE = tf.io.TFRecordOptions.get_compression_type_string('GZIP')
IMAGE_SIZE = [240, 320]
# The maximum number of foreground and background entities in the provided
# dataset. This corresponds to the number of segmentation masks returned per
# scene.
MAX_NUM_ENTITIES = 11
BYTE_FEATURES = ['mask', 'image', 'color', 'material', 'shape', 'size']

# Create a dictionary mapping feature names to `tf.Example`-compatible
# shape and data type descriptors.
features = {
    'image': tf.FixedLenFeature(IMAGE_SIZE+[3], tf.string),
    'mask': tf.FixedLenFeature([MAX_NUM_ENTITIES]+IMAGE_SIZE+[1], tf.string),
    'x': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
    'y': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
    'z': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
    'pixel_coords': tf.FixedLenFeature([MAX_NUM_ENTITIES, 3], tf.float32),
    'rotation': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
    'size': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.string),
    'material': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.string),
    'shape': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.string),
    'color': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.string),
    'visibility': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
}


def _decode(example_proto):
  # Parse the input `tf.Example` proto using the feature description dict above.
  single_example = tf.parse_single_example(example_proto, features)
  for k in BYTE_FEATURES:
    single_example[k] = tf.squeeze(tf.decode_raw(single_example[k], tf.uint8),
                                   axis=-1)
  return single_example


def dataset(tfrecords_path, read_buffer_size=None, map_parallel_calls=None):
  """Read, decompress, and parse the TFRecords file.

  Args:
    tfrecords_path: str. Path to the dataset file.
    read_buffer_size: int. Number of bytes in the read buffer. See documentation
      for `tf.data.TFRecordDataset.__init__`.
    map_parallel_calls: int. Number of elements decoded asynchronously in
      parallel. See documentation for `tf.data.Dataset.map`.

  Returns:
    An unbatched `tf.data.TFRecordDataset`.
  """
  raw_dataset = tf.data.TFRecordDataset(
      tfrecords_path, compression_type=COMPRESSION_TYPE,
      buffer_size=read_buffer_size)
  raw_dataset = raw_dataset.repeat()
  return raw_dataset.map(_decode, num_parallel_calls=map_parallel_calls)

class ClevrData(DataProvider):

  DATA_PATH = "/data4/dbear/tdw_datasets"
  def __init__(
      self,
      data_paths="",
      file_pattern="clevr*.tfrecords",
      sources=BYTE_FEATURES,
      is_training=True,
      temporal=True,
      get_objects=True,
      get_valid=True,
      map_parallel_calls=8,
      read_buffer_size=1024,
      q_cap=1024,
      **kwargs
  ):

    self.data = data_paths
    self.sources = sources
    self.temporal = temporal
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
    return copy.deepcopy(data_params), copy.deepcopy(data_params)

  def input_fn(self, batch_size, train, **kwargs):
    self.is_training = train
    tfr_list = self.get_tfr_filenames(self.data)
    tfr_path = tfr_list[0]
    print("tfr path", tfr_path)
    data = dataset(tfr_path, self.read_buffer_size, self.map_pcall_num)

    if self.is_training:
      data = data.shuffle(buffer_size=self.q_cap)

    data = data.batch(batch_size)
    iterator = tf.data.make_one_shot_iterator(data)
    inputs = iterator.get_next()

    if self.get_objects:
      masks = inputs['mask']
      segments = tf.cast(tf.argmax(masks, axis=1), tf.int32)
      inputs['objects'] = segments


    inputs = {k:v for k,v in inputs.items() if k in ['image', 'objects', 'mask']}
    inputs['images'] = inputs.pop('image')

    if self.get_valid:
      inputs['valid'] = tf.ones_like(inputs['images'][...,0:1])

    for k,inp in inputs.items():
      shape = inp.shape.as_list()
      inputs[k] = tf.reshape(inp, [batch_size]+shape[1:])

    if self.temporal:
      inputs = {k:v[:,tf.newaxis] for k,v in inputs.items()}

    return inputs

if __name__ == '__main__':

  import os
  os.environ['CUDA_VISIBLE_DEVICES'] = ''
  BATCH_SIZE = 4
  TRAIN = False

  train_data, val_data = ClevrData.get_data_params(BATCH_SIZE)
  data_provider = ClevrData(**(train_data if TRAIN else val_data))
  func = data_provider.input_fn
  inputs = func(BATCH_SIZE, TRAIN)
  print("data", inputs)

  sess = tf.Session()
  inputs = sess.run(inputs)

  mask = inputs['mask'][0,0]
  mask = mask.sum(axis=(1,2,3))
  print(mask)

  # import pickle
  # with open('/home/dbear/clevr_ims.pkl', 'wb') as f:
  #   pickle.dump(inputs, f)
  #   f.close()
