from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import tensorflow as tf


class Recognizer(object):
    def __init__(self, inputs):
        num_classes = inputs.num_classes
        batch_size = inputs.batch_size
        images = inputs.images
        labels = inputs.labels
        learning_rate = inputs.learning_rate
        with tf.variable_scope("conv_pool_1"):
            kernel = tf.get_variable(name="kernel",
                                     shape=[5, 5, 3, 48],
                                     initializer=tf.truncated_normal_initializer(stddev=0.05),
                                     dtype=tf.float32)
            biases = tf.get_variable(name="biases",
                                     shape=[48],
                                     initializer=tf.constant_initializer(value=0.),
                                     dtype=tf.float32)
            conv = tf.nn.conv2d(input=images,
                                filter=kernel,
                                strides=[1, 1, 1, 1],
                                padding="SAME")
            conv_bias = tf.nn.bias_add(value=conv,
                                       bias=biases,
                                       name="add_biases")
            relu = tf.nn.relu(conv_bias)
            pool = tf.nn.max_pool(value=relu,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding="SAME",
                                  name="pooling")

        with tf.variable_scope("conv_pool_2"):
            kernel = tf.get_variable(name="kernel",
                                     shape=[5, 5, 48, 64],
                                     initializer=tf.truncated_normal_initializer(stddev=0.05),
                                     dtype=tf.float32)
            biases = tf.get_variable(name="biases",
                                     shape=[64],
                                     initializer=tf.constant_initializer(value=0.),
                                     dtype=tf.float32)
            conv = tf.nn.conv2d(input=pool,
                                filter=kernel,
                                strides=[1, 1, 1, 1],
                                padding="SAME")
            conv_bias = tf.nn.bias_add(value=conv,
                                       bias=biases,
                                       name="add_biases")
            relu = tf.nn.relu(conv_bias)
            pool = tf.nn.max_pool(value=relu,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding="SAME",
                                  name="pooling")
        with tf.variable_scope("conv_pool_3"):
            kernel = tf.get_variable(name="kernel",
                                     shape=[5, 5, 64, 72],
                                     initializer=tf.truncated_normal_initializer(stddev=0.05),
                                     dtype=tf.float32)
            biases = tf.get_variable(name="biases",
                                     shape=[72],
                                     initializer=tf.constant_initializer(value=0.),
                                     dtype=tf.float32)
            conv = tf.nn.conv2d(input=pool,
                                filter=kernel,
                                strides=[1, 2, 2, 1],
                                padding="SAME")
            conv_bias = tf.nn.bias_add(value=conv,
                                       bias=biases,
                                       name="add_biases")
            relu = tf.nn.relu(conv_bias)
            pool = tf.nn.max_pool(value=relu,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding="SAME",
                                  name="pooling")
        reshape = tf.reshape(pool, shape=[batch_size, -1])
        dims = reshape.get_shape().as_list()[-1]
        with tf.variable_scope("fully_conn"):
            weights = tf.get_variable(name="weights",
                                      shape=[dims, 256],
                                      initializer=tf.truncated_normal_initializer(stddev=0.05),
                                      dtype=tf.float32)
            biases = tf.get_variable(name="biases",
                                     shape=[256],
                                     initializer=tf.constant_initializer(value=0.),
                                     dtype=tf.float32)
            output = tf.nn.xw_plus_b(x=reshape,
                                     weights=weights,
                                     biases=biases)
            conn = tf.nn.relu(output)
        with tf.variable_scope("output"):
            weights = tf.get_variable(name="weights",
                                      shape=[256, num_classes],
                                      initializer=tf.truncated_normal_initializer(stddev=0.05),
                                      dtype=tf.float32)
            biases = tf.get_variable(name="biases",
                                     shape=[num_classes],
                                     initializer=tf.constant_initializer(value=0.),
                                     dtype=tf.float32)
            logits = tf.nn.xw_plus_b(x=conn,
                                     weights=weights,
                                     biases=biases)
        with tf.name_scope("loss"):
            loss_per_example = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            self.__loss = tf.reduce_mean(loss_per_example, name="loss")

        with tf.name_scope("train"):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.__train_op = optimizer.minimize(self.__loss, name="train_op")

        with tf.name_scope("validate"):
            predict = tf.argmax(logits, dimension=1)
            equal = tf.equal(predict, labels)
            self.__accuracy = tf.reduce_mean(tf.cast(equal, tf.float32), name="validate")

    @property
    def loss(self):
        return self.__loss

    @property
    def train_op(self):
        return self.__train_op

    @property
    def validate(self):
        return self.__accuracy
