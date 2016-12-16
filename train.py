from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import tensorflow as tf
from model import Recognizer
from input import Inputs


def main():
    inputs4train = Inputs(train=True)
    inputs4valid = Inputs(train=False)
    with tf.name_scope("train"):
        with tf.variable_scope("model", reuse=None):
            m = Recognizer(inputs=inputs4train)

    with tf.name_scope("validate"):
        with tf.variable_scope("model", reuse=True):
            mvalid = Recognizer(inputs=inputs4valid)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
        index = 0
        while not coord.should_stop():
            _, loss = sess.run([m.train_op, m.loss])
            index += 1
            print("step: % 4d, loss is %4.4f" % (index, loss))
            if index % 5 == 0:
                valid_loss, valid_accuracy = sess.run([mvalid.loss, mvalid.validate])
                print("loss is %4.4f, validate accuracy is %1.4f" % (valid_loss, valid_accuracy))
    except tf.errors.OutOfRangeError:
        print("Done training! stopping threads")
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()

if __name__ == "__main__":
    main()