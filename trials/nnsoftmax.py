import tensorflow as tf
import numpy as np
import unittest
import numpy


# 将label集合转化为one-hot编码,精妙绝伦的转换(from tensorflow/contrib/learn/python/learn/datasets/mnist.py)
def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors.
       a = np.zeros(shape=[50000], dtype=int)
       a.shape[0]
       Out[7]: 50000
    """
    num_labels = labels_dense.shape[0]
    # 生成一维矩阵
    # index_offset 是每行第0个元素拉平之后的下标 如果有两个label 三个分类
    # [0,1,2] index_offset = [0,3] = arange(2)*3 = [0,1]*3 = [0,3]
    # [3,4,5]
    # 如果第二个label是第二个分类[0,0,1] 3+2 =5 下标为5的那个点是1
    # 这样就将label转化为one-hot编码了
    index_offset = numpy.arange(num_labels) * num_classes
    # 生成 num_labels * num_classes 的矩阵，并填充0
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    # ravel()散开
    # labels_one_hot.flat将二维矩阵拉平，使用类似数组的方式得到取值赋值
    """
    array([[1, 2, 3],
           [4, 5, 6]])
    # flat和ravel的区别
    a.flat  ---返回一个内存位置
    Out[13]: <numpy.flatiter at 0x103050200>
    a.flat[0] # 1
    a.flat[1] # 2
    a.ravel()  ----返回一个拉平的矩阵
    Out[12]: array([1, 2, 3, 4, 5, 6])

    """
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


num_classes = 3;


def readfile():
    with open('train.txt', 'r') as f:
        train_data = np.array([int(x) for x in f.read()]).reshape((6000, 2))
    with open('label.txt', 'r') as f:
        train_label = dense_to_one_hot(np.array([int(x) for x in f.read()]), num_classes)

    with open('train_test.txt', 'r') as f:
        train_data_test = np.array([int(x) for x in f.read()]).reshape((1000, 2))
    with open('label_test.txt', 'r') as f:
        train_label_test = dense_to_one_hot(np.array([int(x) for x in f.read()]), num_classes)
    return train_data, train_label, train_data_test, train_label_test


def main():
    x = tf.placeholder(tf.float32, [None, 2])
    w = tf.Variable(tf.zeros([2, 3]))
    b = tf.Variable(tf.zeros([3]))
    y = tf.matmul(x, w) + b
    y_ = tf.placeholder(tf.float32, [None, 3])  # 一维的,就是它的真实值
    # tf.losses.sparse_softmax_cross_entropy 这种方法的label是真实值
    # tf.nn.softmax_cross_entropy_with_logits 两个方法可以定义损失函数,这种方法的label是由one-hot编码
    # 使用one-hot编码来与真实值比对的模型
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    train_data, train_label, train_data_test, train_label_test = readfile()
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    for _ in range(1000):
        sess.run(train_step, feed_dict={x: train_data, y_: train_label})
    # 真实值和预测值都需要使用argmax 将one-hot编码转化为真实值
    y_pred_cls = tf.argmax(y, axis=1)
    y_true_cls = tf.argmax(y_, axis=1)
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: train_data_test,
                                        y_: train_label_test}))
    print(sess.run(w))
    print(sess.run(b))


"""
值好大啊 但是是复合条件的
1.0
[[ -4475.66650391  -1870.66699219   6346.33203125]
 [ -1000.00128174  11364.         -10364.        ]]
[ -1000.04705811  11363.97070312 -10364.00683594]
"""

if __name__ == '__main__':
    main()


class TestDict(unittest.TestCase):
    def test_init(self):
        w = np.array([[-4475.66650391, - 1870.66699219, 6346.33203125],
                      [-1000.00128174, 11364., - 10364.]])
        b = np.array([-0.44633889, 4.75634384, -4.31001806])
        y = np.matmul([6, 1], w) + b
        print(y)
