import tensorflow as tf
import numpy as np
import unittest


def readfile():
    with open('train.txt', 'r') as f:
        train_data = np.array([int(x) for x in f.read()]).reshape((6000, 2))
    with open('label.txt', 'r') as f:
        train_label = np.array([int(x) for x in f.read()])

    with open('train_test.txt', 'r') as f:
        train_data_test = np.array([int(x) for x in f.read()]).reshape((1000, 2))
    with open('label_test.txt', 'r') as f:
        train_label_test = np.array([int(x) for x in f.read()])
    return train_data, train_label, train_data_test, train_label_test


def main():
    x = tf.placeholder(tf.float32, [None, 2])
    w = tf.Variable(tf.zeros([2, 3]))
    b = tf.Variable(tf.zeros([3]))
    y = tf.matmul(x, w) + b
    y_ = tf.placeholder(tf.int64, [None])  # 一维的,就是它的真实值
    # tf.losses.sparse_softmax_cross_entropy 这种方法的label是真实值
    # tf.nn.softmax_cross_entropy_with_logits 两个方法可以定义损失函数,这种方法的label是由one-hot编码

    # The raw formulation of cross-entropy,
    #
    #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
    #                                        reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.losses.sparse_softmax_cross_entropy on the raw
    # logit outputs of 'y', and then average across the batch.
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(logits=y, labels=y_)
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    train_data, train_label, train_data_test, train_label_test = readfile()
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    for _ in range(1000):
        sess.run(train_step, feed_dict={x: train_data, y_: train_label})

    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: train_data_test,
                                        y_: train_label_test}))
    print(sess.run(w))
    print(sess.run(b))


if __name__ == '__main__':
    main()


class TestDict(unittest.TestCase):

    def test_init(self):
        w = np.array([[-1.00487339, - 1.15891397, 2.16378975],
                      [-0.44634497, 4.75635386, -4.31000757]])
        b = np.array([-0.44633889, 4.75634384, -4.31001806])
        y = np.matmul([6, 1], w) + b
        print(y)
