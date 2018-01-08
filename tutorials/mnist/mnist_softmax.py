# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def main(_):
    # Import data
    # Namespace(data_dir='/tmp/tensorflow/mnist/input_data') 经过main函数执行命令后
    # 从这个目录读取mnist数据集，如果没有，则自动下载tar.gz文件但是不自动解压，直接读取压缩文件
    mnist = input_data.read_data_sets(FLAGS.data_dir)

    # Create the model
    """数学知识点1：矩阵乘法和矩阵的点乘(dot product)
    #  tf.matmul(x, W)是矩阵的乘法 能够相乘的两个矩阵为 (m*s)(s*n)得到(m*n)的矩阵
    # a*b 是矩阵的点乘，条件是矩阵的维度相同
    """
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    # tf.matmul(x, W) (-1,784) (784,10)得到y (-1,10)的矩阵
    b = tf.Variable(tf.zeros([10]))
    # y是预测值 这是训练定义的模型
    y = tf.matmul(x, W) + b

    # Define loss and optimizer
    # y_是从数据集中读取的真实值,不是one-hot的格式
    y_ = tf.placeholder(tf.int64, [None])

    # The raw formulation of cross-entropy,
    #   y和y_进行dot product
    """
    数学知识点2:softmax函数和交叉熵(用交叉熵定义损失函数)
    """
    #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
    #                                 reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.losses.sparse_softmax_cross_entropy on the raw
    # outputs of 'y', and then average across the batch.
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
    """
    数学知识点3:梯度函数
    """
    train_step = tf.train.GradientDescentOptimizer(0.8).minimize(cross_entropy)

    """
    调整不同的优化函数显示不同的准确度(控制learning Rate为0.5)
    ==============================
    | AdamOptimizer      | 0.8679
    ==============================
    | AdagradOptimizer   | 0.9202
    ==============================
    | GradientDescentOptimizer | 0.9152
    ==============================
    调整不同的learning rate(控制梯度函数为GradientDescentOptimizer 0.5=0.9152 )
    ==============================
    | 0.1                | 0.9097
    ==============================
    | 0.2                | 0.9163
    ==============================
    | 0.3                 | 0.9181
    ==============================
    | 0.8                 | 0.9234
    ==============================
    """

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    # Train
    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        print(batch_ys)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # Test trained model
    """
    tensorflow有一组以tf.arg__开头的api，来获得数组某些值的下标。 
    tf.argmax([])得到数组最大值的下标，这个下标就是图片代表的数字
    """
    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                        y_: mnist.test.labels}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    print(FLAGS)
    print(unparsed)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
