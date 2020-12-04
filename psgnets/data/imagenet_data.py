"""
This script contains class method `dataset_func` which will return dataset elements

The data format structure of ImageNet required for `dataset_func` is similar as
data structure generated by following structure:
    https://github.com/tensorflow/models/blob/master/research/inception/inception/data/build_imagenet_data.py

The only difference is that each tfrecords file only contains two attributes:
    images: jpeg format of images
    labels: int64 of 0-999 labels
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

import os, sys, copy
import numpy as np
import pdb

from .base import DataProvider
from .utils import *

## change this to wherever you're storing imagenet tfrecords
IMAGENET_DIR = '/mnt/fs1/Dataset/TFRecord_Imagenet_standard/image_label_full'

def _at_least_x_are_equal(a, b, x):
    """At least `x` of `a` and `b` `Tensors` are equal."""
    match = tf.equal(a, b)
    match = tf.cast(match, tf.int32)
    return tf.greater_equal(tf.reduce_sum(match), x)

def color_normalize(image):
    print("color normalizing")
    image = tf.cast(image, tf.float32) / 255
    imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - imagenet_mean) / imagenet_std
    return image

class ImageNet(DataProvider):
    """
    Class where data provider for ImageNet will be built
    """
    DATA_PATH = IMAGENET_DIR
    TRAIN_LEN = 1281167
    VAL_LEN = 50000

    def __init__(
            self, data_paths, prep_type='resnet',
            crop_size=256, smallest_side=256, resize=None,
            images_key='images', labels_key='labels', temporal=True,
            get_valid=True,
            do_color_normalize=False, q_cap=51200, **kwargs):
        self.data_paths = data_paths

        # Parameters about preprocessing
        self.prep_type = prep_type
        self.crop_size = crop_size
        self.smallest_side = smallest_side
        self.resize = resize
        self.temporal = temporal
        self.images_key = images_key
        self.labels_key = labels_key
        self.do_color_normalize = do_color_normalize
        self.q_cap = q_cap

        # Placeholders to be filled later
        self.file_pattern = None
        self.is_training = None

        # dataset specific
        self.get_valid = get_valid


    @staticmethod
    def get_data_params(batch_size, sequence_len=None, dataprefix=IMAGENET_DIR, **kwargs):
        data_params = copy.deepcopy(kwargs)
        data_params['data_paths'] = dataprefix
        data_params['temporal'] = (sequence_len is not None)
        return copy.deepcopy(data_params), copy.deepcopy(data_params)

    # def get_tfr_filenames(self):
    #     """
    #     Get list of tfrecord filenames
    #     for given folder_name fitting the given file_pattern
    #     """
    #     assert self.file_pattern, "Please specify file pattern!"
    #     tfrecord_pattern = os.path.join(folder_name, self.file_pattern)
    #     datasource = tf.gfile.Glob(tfrecord_pattern)
    #     datasource.sort()
    #     return np.asarray(datasource)

    def get_resize_scale(self, height, width):
        """
        Get the resize scale so that the shortest side is `smallest_side`
        """
        smallest_side = tf.convert_to_tensor(self.smallest_side, dtype=tf.int32)

        height = tf.to_float(height)
        width = tf.to_float(width)
        smallest_side = tf.to_float(smallest_side)

        scale = tf.cond(
                tf.greater(height, width),
                lambda: smallest_side / width,
                lambda: smallest_side / height)
        return scale

    def resize_cast_to_uint8(self, image):
        image = tf.cast(
                tf.image.resize_bilinear(
                    [image],
                    [self.crop_size, self.crop_size])[0],
                dtype=tf.uint8)
        image.set_shape([self.crop_size, self.crop_size, 3])
        return image

    def central_crop_from_jpg(self, image_string):
        """
        Resize the image to make its smallest side to be 256;
        then get the central 224 crop
        """
        shape = tf.image.extract_jpeg_shape(image_string)
        scale = self.get_resize_scale(shape[0], shape[1])
        cp_height = tf.cast(self.crop_size / scale, tf.int32)
        cp_width = tf.cast(self.crop_size / scale, tf.int32)
        cp_begin_x = tf.cast((shape[0] - cp_height) / 2, tf.int32)
        cp_begin_y = tf.cast((shape[1] - cp_width) / 2, tf.int32)
        bbox = tf.stack([
                cp_begin_x, cp_begin_y, \
                cp_height, cp_width])
        crop_image = tf.image.decode_and_crop_jpeg(
                image_string,
                bbox,
                channels=3)
        image = self.resize_cast_to_uint8(crop_image)

        return image

    def resnet_crop_from_jpg(self, image_str):
        """
        Random crop in Inception style, see GoogLeNet paper, also used by ResNet
        """
        shape = tf.image.extract_jpeg_shape(image_str)
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
                shape,
                bounding_boxes=bbox,
                min_object_covered=0.1,
                aspect_ratio_range=(3. / 4, 4. / 3.),
                area_range=(0.08, 1.0),
                max_attempts=100,
                use_image_if_no_bounding_boxes=True)

        # Get the cropped image
        bbox_begin, bbox_size, bbox = sample_distorted_bounding_box
        random_image = tf.image.decode_and_crop_jpeg(
                image_str,
                tf.stack([bbox_begin[0], bbox_begin[1], \
                          bbox_size[0], bbox_size[1]]),
                channels=3)
        bad = _at_least_x_are_equal(shape, tf.shape(random_image), 3)

        # central crop if bad
        min_size = tf.minimum(shape[0], shape[1])
        offset_height = tf.random_uniform(
                shape=[],
                minval=0, maxval=shape[0] - min_size + 1,
                dtype=tf.int32
                )
        offset_width = tf.random_uniform(
                shape=[],
                minval=0, maxval=shape[1] - min_size + 1,
                dtype=tf.int32
                )
        bad_image = tf.image.decode_and_crop_jpeg(
                image_str,
                tf.stack([offset_height, offset_width, \
                          min_size, min_size]),
                channels=3)
        image = tf.cond(
                bad,
                lambda: bad_image,
                lambda: random_image,
                )

        image = self.resize_cast_to_uint8(image)
        return image

    def alexnet_crop_from_jpg(self, image_string):
        """
        Resize the image to make its smallest side to be 256;
        then randomly get a 224 crop
        """
        shape = tf.image.extract_jpeg_shape(image_string)
        scale = self.get_resize_scale(shape[0], shape[1])
        cp_height = tf.cast(self.crop_size / scale, tf.int32)
        cp_width = tf.cast(self.crop_size / scale, tf.int32)

        # Randomly sample begin x and y
        x_range = [0, shape[0] - cp_height + 1]
        y_range = [0, shape[1] - cp_width + 1]
        if self.prep_type == 'alex_center':
            # Original AlexNet preprocessing uses center 256*256 to crop
            min_shape = tf.minimum(shape[0], shape[1])
            x_range = [
                    tf.cast((shape[0] - min_shape) / 2, tf.int32),
                    shape[0] - cp_height + 1 - \
                            tf.cast(
                                (shape[0] - min_shape) / 2,
                                tf.int32),
                    ]
            y_range = [
                    tf.cast((shape[1] - min_shape) / 2, tf.int32),
                    shape[1] - cp_width + 1 - \
                            tf.cast(
                                (shape[1] - min_shape) / 2,
                                tf.int32),
                    ]

        cp_begin_x = tf.random_uniform(
                shape=[],
                minval=x_range[0], maxval=x_range[1],
                dtype=tf.int32
                )
        cp_begin_y = tf.random_uniform(
                shape=[],
                minval=y_range[0], maxval=y_range[1],
                dtype=tf.int32
                )

        bbox = tf.stack([
                cp_begin_x, cp_begin_y, \
                cp_height, cp_width])
        crop_image = tf.image.decode_and_crop_jpeg(
                image_string,
                bbox,
                channels=3)
        image = self.resize_cast_to_uint8(crop_image)

        return image

    # TODO: write something equivalent to preprocess_for_eval frmo TPU
    # Then validate on GPU using that and compare results.
    # Results so far seem to be higher with this current preproc, than on tpu
    # They should be equal to tpu validation
    # If they are the same, we can revert back to this preproc (since it seems to be better)
    # But we'll know that this was the source of the discrepancy.
    def preprocessing(self, image_string):
        """
        Preprocessing for each image
        """
        assert self.is_training is not None, "Must specify is_train"

        def _rand_crop(image_string):
            if self.prep_type == 'resnet':
                image = self.resnet_crop_from_jpg(image_string)
            else:
                image = self.alexnet_crop_from_jpg(image_string)

            return image

        if self.is_training:
            image = _rand_crop(image_string)
            image = tf.image.random_flip_left_right(image)
        else:
            image = self.central_crop_from_jpg(image_string)

        if self.do_color_normalize:
            image = color_normalize(image)

        if self.resize is not None:
            image = tf.image.resize_images(image, [self.resize, self.resize], align_corners=True)
        return image

    ### BEGIN raw image preproc
    def raw_image_preproc(self, image_string):
        image = self.central_crop_from_jpg(image_string)
        image = tf.image.resize_images(image, [self.resize, self.resize], align_corners=True)
        return image
    ### END raw image preproc

    def data_parser(self, value):
        """
        Parse record and preprocessing
        """
        # Load the image and preprocess it
        keys_to_features = {
                'images': tf.FixedLenFeature((), tf.string, ''),
                'labels': tf.FixedLenFeature([], tf.int64, -1)}
        parsed = tf.parse_single_example(value, keys_to_features)
        image_string = parsed['images']
        image_label = parsed['labels']

        # Do the preprocessing
        image = self.preprocessing(image_string)
        ret_dict = {
                self.images_key:image,
                self.labels_key:image_label
                # 'raw_images': self.raw_image_preproc(image_string)}
            }
        return ret_dict

    def build_dataset(self, batch_size, train, **kwargs):
        self.is_training = train
        self.file_pattern = "train*" if train else "validation*"

        # First get tfrecords names
        tfr_list = self.get_tfr_filenames(self.data_paths)

        def _fetch_dataset(filename):
            """
            Useful util function for fetching records
            """
            buffer_size = 32 * 1024 * 1024     # 32 MiB per file
            dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
            return dataset

        # Build list_file dataset from tfrecord files
        if self.is_training:
            dataset = tf.data.Dataset.list_files(tfr_list)
            dataset = dataset.apply(
                    tf.contrib.data.shuffle_and_repeat(
                        len(tfr_list)))
            # Read each file
            dataset = dataset.apply(
                    tf.contrib.data.parallel_interleave(
                       _fetch_dataset, cycle_length=8, sloppy=True))
        else:
            dataset = tf.data.Dataset.list_files(tfr_list, shuffle=False)
            dataset = dataset.repeat()
            # Read each file
            dataset = dataset.apply(
                    tf.contrib.data.parallel_interleave(
                       _fetch_dataset, cycle_length=8, sloppy=False))

        # Shuffle and preprocessing
        if self.is_training:
            dataset = dataset.shuffle(buffer_size=self.q_cap)
        dataset = dataset.prefetch(batch_size * 4)
        dataset = dataset.map(
                self.data_parser,
                num_parallel_calls=48)

        # Batch the dataset and make iteratior
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
        dataset = dataset.prefetch(4)

        return dataset

    def input_fn(
            self, batch_size, train, **kwargs):
        """
        Build the dataset, get the elements
        """
        dataset = self.build_dataset(batch_size, train, **kwargs)
        next_element = dataset.make_one_shot_iterator().get_next()

        if self.temporal:
            next_element = {k:tf.expand_dims(tensor, 1) for k,tensor in next_element.items()}

        if self.get_valid:
            next_element['valid'] = tf.cast(tf.ones_like(next_element[self.images_key][...,0:1]), dtype=tf.float32)
        return next_element

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    BATCH_SIZE = 64
    TRAIN = True
    train_data, val_data = ImageNet.get_data_params(BATCH_SIZE)
    data_provider = ImageNet(**(train_data if TRAIN else val_data))
    func = data_provider.input_fn
    data = func(BATCH_SIZE, TRAIN)
    print("data", data)
