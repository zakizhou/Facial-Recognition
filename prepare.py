from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import tensorflow as tf
import os
from scipy.ndimage import imread
import sys
import re
import cPickle as pickle
import codecs


def convert_to_records(dir):
    """
    convert all images in a whole dir into .tfrecord file
    Args:
    :param dir: a dir contain all image files with specific name
    :return: None
    """
    filenames = os.listdir(dir)
    num_files = len(filenames)
    writer = tf.python_io.TFRecordWriter("records/examples.tfrecords")
    print("start parse images and write to tfreocrds:")
    names = []
    for i, filename in enumerate(filenames):
        sys.stdout.write('\r')
        sys.stdout.write("processing No.%2dth %% %d image" % (i + 1, num_files))
        image = imread("images/" + filename.decode("utf-8"))
        label = re.findall(r'(.*?)\_', filename.decode("utf-8"))[0]
        label_integer = -1
        if label not in names:
            names.append(label)
            label_integer += 1
        example = tf.train.Example(features=tf.train.Features(feature={
            "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tostring()])),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label_integer]))
        }))
        writer.write(example.SerializeToString())
        sys.stdout.flush()
    print("Done")
    name2id = dict(zip(names, range(len(names))))
    id2name = dict([(id, name) for (name, id) in name2id.items()])
    name2id_file = codecs.open("dictionary/name2id.p", mode='wr', encoding="utf-8")
    pickle.dump(name2id, name2id_file)
    id2name_file = codecs.open("dictionary/id2name.p", mode='wr', encoding="utf-8")
    pickle.dump(id2name, id2name_file)


def main():
    convert_to_records('images')


if __name__ == "__main__":
    main()