from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import tensorflow as tf


IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
IMAGE_CHANNELS = 3


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized = reader.read(filename_queue)
    features = tf.parse_single_example(serialized=serialized, features=
    {
        "image": tf.FixedLenFeature([], tf.string),
        "label": tf.FixedLenFeature([], tf.int64)
    })
    flatten_image = tf.decode_raw(features['image'], tf.uint8)
    label = features['label']
    flatten_image.set_shape([IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS])
    image = tf.reshape(flatten_image, [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS])
    return image, label


def input_producer(filename, batch_size, min_after_queue):
    filename_queue = tf.train.string_input_producer([filename])
    image, label = read_and_decode(filename_queue=filename_queue)
    batch_images, batch_labels = tf.train.shuffle_batch([image, label],
                                                        batch_size=batch_size,
                                                        min_after_dequeue=min_after_queue,
                                                        capacity=min_after_queue + 3 * batch_size)
    return batch_images, batch_labels

tf.import_graph_def()
tf.train.import_meta_graph