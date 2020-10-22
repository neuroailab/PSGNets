import numpy as np
import tensorflow.compat.v1 as tf
import os

from utils import object_id_hash

class DataProvider(object):

    def __init__(
            self,
            data_paths,
            sequence_len=1,
            file_pattern="*.tfrecords",
            train_len=None,
            val_len=None,
            **kwargs
    ):
        self.data_paths = data_paths
        self.sequence_len = sequence_len
        self.is_training = None
        self.file_pattern = file_pattern
        self.TRAIN_LEN = train_len
        self.VAL_LEN = val_len

    @staticmethod
    def get_data_params(batch_size, sequence_len, **kwargs):
        raise NotImplementedError("You must overwrite the get_data_params staticmethod for the data provider class!")

    def get_tfr_filenames(self, folder_name):
        tfrecord_pattern = os.path.join(folder_name, self.file_pattern)
        datasource = tf.gfile.Glob(tfrecord_pattern)
        datasource.sort()
        return datasource

    def preprocessing(self, image_string):
        raise NotImplementedError("Preprocessing not implemented")

    def preproc_segment_ids(self, segment_ids, invalid_ids=[]):

        assert len(segment_ids.shape.as_list()) == 5
        B,T,H,W,C = segment_ids.shape.as_list()

        if len(invalid_ids):
            seg_ids_orig = object_id_hash(segment_ids, decreasing=False)
            invalid = tf.reshape(tf.stack(invalid_ids), [1,1,1,1,-1])
            segment_ids *= tf.cast(tf.logical_not(
                tf.reduce_any(tf.equal(seg_ids_orig, invalid), axis=-1, keepdims=True)), segment_ids.dtype)

        assert B <= 128 and segment_ids.dtype == tf.uint8, "max hash value must be < 256**4 / 2 = 2^16"
        # add batch values to make the hashing unique
        b_inds = tf.tile(tf.reshape(tf.range(B, dtype=tf.int32), [B,1,1,1,1]), [1,T,H,W,C])
        segment_ids = tf.concat([b_inds, tf.cast(segment_ids, tf.int32)], axis=-1)
        segment_ids = object_id_hash(segment_ids, dtype_out=tf.int32, val=256)
        _, segment_ids = tf.unique(tf.reshape(segment_ids, [-1]))
        segment_ids = tf.reshape(segment_ids, [B,T,H,W])
        segment_ids = segment_ids - tf.reduce_min(segment_ids, axis=[1,2,3], keepdims=True)
        return segment_ids


    def input_fn(self, batch_size, train, **kwargs):
        raise NotImplementedError("You must overwrite the input_fn method for the %s data provider class!" % type(self).__name__)
