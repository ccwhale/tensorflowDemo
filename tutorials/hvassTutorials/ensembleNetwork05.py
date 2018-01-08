"""
  集成卷积神经网络:ensemble neural network
  几个神经网络,取输出的平均值
总结:
  这篇教程创建了有5个卷积神经网络的集成神经网络来给MNIST中的手写的数字数据集做分类。集成是通过取5个单独神经网络预测标签的平均值。
这个结果对分类的准确性有微小的提高，集成的准确性未99.1%,单独神经网络最好的准确性是98.9%
  集成的性能并不总是比单个的神经网络的性能好。
  集成学习使用在这里称作bagging或者Bootstrap Aggregating 引导聚集，这个主要用途是避免过拟合。也可能用在其他方面，对特别的神经网络或者
数据集不是特别必须的
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import os

# Use PrettyTensor to simplify Neural Network construction.
import prettytensor as pt

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST/', one_hot=True)

print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(data.test.labels)))
print("- Validation-set:\t{}".format(len(data.validation.labels)))
"""
Size of:
- Training-set:		55000
- Test-set:		10000
- Validation-set:	5000
"""

data.test.cls = np.argmax(data.test.labels, axis=1)
data.validation.cls = np.argmax(data.validation.labels, axis=1)

# 将训练集和验证集连接起来
combined_images = np.concatenate([data.train.images, data.validation.images], axis=0)  # (60000, 784)
combined_labels = np.concatenate([data.train.labels, data.validation.labels], axis=0)  # (60000, 10)
combined_size = len(combined_images) # 60000
train_size = int(0.8 * combined_size) # 48000

# validation_size = 12000
validation_size = combined_size - train_size

# 将连接好的 combined_images combined_labels随机分成训练集和验证集
def random_training_set():
    # Create a randomized index into the full / combined training-set.
    """https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.random.permutation.html"""
    idx = np.random.permutation(combined_size) # 60000的数组进行shuffle,返回新的数组

    # Split the random index into training- and validation-sets.
    idx_train = idx[0:train_size]
    idx_validation = idx[train_size:]

    # Select the images and labels for the new training-set.
    """
    np.random.permutation(10)
    array([1, 7, 4, 3, 0, 9, 2, 5, 8, 6])
    >>> a = np.random.permutation(a)
    >>> a
    array([[6, 7, 8],
           [0, 1, 2],
           [3, 4, 5]])
          
    a = np.arange(9).reshape(3,3)
    array([[0, 1, 2],
       [3, 4, 5],
       [6, 7, 8]])
    >>> a[[1,2],:]
   array([[3, 4, 5],
         [6, 7, 8]]) 选取了第1和第2行的数组返回
    idx = np.random.permutation(combined_size) 输出0到60000的乱序的array
    x_train = combined_images[idx_train, :] 在60000中选取48000张图片，通过下标选取
    """
    x_train = combined_images[idx_train, :]
    y_train = combined_labels[idx_train, :]

    # Select the images and labels for the new validation-set.
    x_validation = combined_images[idx_validation, :]
    y_validation = combined_labels[idx_validation, :]

    # Return the new training- and validation-sets.
    return x_train, y_train, x_validation, y_validation

# We know that MNIST images are 28 pixels in each dimension.
img_size = 28

# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 1

# Number of classes, one class for each of 10 digits.
num_classes = 10

x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')

x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')

y_true_cls = tf.argmax(y_true, axis=1)

x_pretty = pt.wrap(x_image)

with pt.defaults_scope(activation_fn=tf.nn.relu):
    y_pred, loss = x_pretty.\
        conv2d(kernel=5, depth=16, name='layer_conv1').\
        max_pool(kernel=2, stride=2).\
        conv2d(kernel=5, depth=36, name='layer_conv2').\
        max_pool(kernel=2, stride=2).\
        flatten().\
        fully_connected(size=128, name='layer_fc1').\
        softmax_classifier(num_classes=num_classes, labels=y_true)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

y_pred_cls = tf.argmax(y_pred, axis=1)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver(max_to_keep=100)

save_dir = 'checkpoints/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def get_save_path(net_number):
    return save_dir + 'network' + str(net_number)

session = tf.Session()


def init_variables():
    session.run(tf.initialize_all_variables())


train_batch_size = 64

def random_batch(x_train, y_train):
    # Total number of images in the training-set.
    num_images = len(x_train)

    # Create a random index into the training-set.
    # 48000 64 np.arange(48000)取出64个 返回大小为64的数组
    idx = np.random.choice(num_images,
                           size=train_batch_size,
                           replace=False)

    # Use the random index to select random images and labels.
    # 取出下标为任意选出的图片
    x_batch = x_train[idx, :]  # Images.
    y_batch = y_train[idx, :]  # Labels.

    # Return the batch.
    return x_batch, y_batch


def optimize(num_iterations, x_train, y_train):
    # Start-time used for printing time-usage below.
    start_time = time.time()

    # num_iterations = 10000
    for i in range(num_iterations):

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        # 每次选出了64张 迭代10000次
        """https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.random.choice.html"""
        x_batch, y_true_batch = random_batch(x_train, y_train)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)

        # Print status every 100 iterations and after last iteration.
        # 每经过100次迭代输出那个时刻的精度值
        if i % 100 == 0:
            # Calculate the accuracy on the training-batch.
            acc = session.run(accuracy, feed_dict=feed_dict_train)

            # Status-message for printing.
            msg = "Optimization Iteration: {0:>6}, Training Batch Accuracy: {1:>6.1%}"

            # Print it.
            print(msg.format(i + 1, acc))

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

num_networks = 5

num_iterations = 10000

if False:
    # For each of the neural networks.
    for i in range(num_networks):
        print("Neural network: {0}".format(i))

        # Create a random training-set. Ignore the validation-set.
        # x_train, y_train, x_validation, y_validation
        x_train, y_train, _, _ = random_training_set()

        # Initialize the variables of the TensorFlow graph.
        session.run(tf.global_variables_initializer())

        # Optimize the variables using this training-set.
        # x_train y_train _ _ (x_train, y_train, x_validation, y_validation 48000 12000)
        optimize(num_iterations=num_iterations,
                 x_train=x_train,
                 y_train=y_train)

        # Save the optimized variables to disk.
        saver.save(sess=session, save_path=get_save_path(i))

        # Print newline.
        print()

# Split the data-set in batches of this size to limit RAM usage.
batch_size = 256


def predict_labels(images):
    # Number of images.
    num_images = len(images)

    # Allocate an array for the predicted labels which
    # will be calculated in batches and filled into this array.
    # shape = (-1, 10)
    pred_labels = np.zeros(shape=(num_images, num_classes),
                           dtype=np.float)

    # Now calculate the predicted labels for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_images:
        # The ending index for the next batch is denoted j.
        j = min(i + batch_size, num_images)

        # Create a feed-dict with the images between index i and j.
        # (i到j的图片, 784)
        feed_dict = {x: images[i:j, :]}

        # Calculate the predicted labels using TensorFlow.
        pred_labels[i:j] = session.run(y_pred, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    return pred_labels


def correct_prediction(images, labels, cls_true):
    # Calculate the predicted labels.
    #
    pred_labels = predict_labels(images=images)

    # Calculate the predicted class-number for each image.
    cls_pred = np.argmax(pred_labels, axis=1)

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    return correct


def test_correct():
    return correct_prediction(images=data.test.images,
                              labels=data.test.labels,
                              cls_true=data.test.cls)


def validation_correct():
    return correct_prediction(images=data.validation.images,
                              labels=data.validation.labels,
                              cls_true=data.validation.cls)


def classification_accuracy(correct):
    # When averaging a boolean array, False means 0 and True means 1.
    # So we are calculating: number of True / len(correct) which is
    # the same as the classification accuracy.
    return correct.mean()


def test_accuracy():
    # Get the array of booleans whether the classifications are correct
    # for the test-set.
    correct = test_correct()

    # Calculate the classification accuracy and return it.
    # 求平均数
    return classification_accuracy(correct)


def validation_accuracy():
    # Get the array of booleans whether the classifications are correct
    # for the validation-set.
    correct = validation_correct()

    # Calculate the classification accuracy and return it.
    return classification_accuracy(correct)


def ensemble_predictions():
    # Empty list of predicted labels for each of the neural networks.
    pred_labels = []

    # Classification accuracy on the test-set for each network.
    test_accuracies = []

    # Classification accuracy on the validation-set for each network.
    val_accuracies = []

    # For each neural network in the ensemble.
    # num_networks = 5
    for i in range(num_networks):
        # Reload the variables into the TensorFlow graph.
        saver.restore(sess=session, save_path=get_save_path(i))

        # Calculate the classification accuracy on the test-set.
        test_acc = test_accuracy()

        # Append the classification accuracy to the list.
        test_accuracies.append(test_acc)

        # Calculate the classification accuracy on the validation-set.
        val_acc = validation_accuracy()

        # Append the classification accuracy to the list.
        val_accuracies.append(val_acc)

        # Print status message.
        msg = "Network: {0}, Accuracy on Validation-Set: {1:.4f}, Test-Set: {2:.4f}"
        print(msg.format(i, val_acc, test_acc))

        # Calculate the predicted labels for the images in the test-set.
        # This is already calculated in test_accuracy() above but
        # it is re-calculated here to keep the code a bit simpler.
        pred = predict_labels(images=data.test.images)

        # Append the predicted labels to the list.
        pred_labels.append(pred)

    return np.array(pred_labels), \
           np.array(test_accuracies), \
           np.array(val_accuracies)


pred_labels, test_accuracies, val_accuracies = ensemble_predictions()

print("Mean test-set accuracy: {0:.4f}".format(np.mean(test_accuracies)))
print("Min test-set accuracy:  {0:.4f}".format(np.min(test_accuracies)))
print("Max test-set accuracy:  {0:.4f}".format(np.max(test_accuracies)))

ensemble_pred_labels = np.mean(pred_labels, axis=0)
print(ensemble_pred_labels.shape)
# (10000, 10)
ensemble_cls_pred = np.argmax(ensemble_pred_labels, axis=1)
print(ensemble_cls_pred.shape)
# (10000,)
ensemble_correct = (ensemble_cls_pred == data.test.cls)
ensemble_incorrect = np.logical_not(ensemble_correct)

print(test_accuracies)
# [ 0.9875  0.9881  0.9895  0.9896  0.9894]
best_net = np.argmax(test_accuracies)
print(best_net)
# 3
print(test_accuracies[best_net])
# 0.9896
"""
Network: 0, Accuracy on Validation-Set: 0.9934, Test-Set: 0.9875
Network: 1, Accuracy on Validation-Set: 0.9946, Test-Set: 0.9881
Network: 2, Accuracy on Validation-Set: 0.9938, Test-Set: 0.9895
Network: 3, Accuracy on Validation-Set: 0.9944, Test-Set: 0.9896
Network: 4, Accuracy on Validation-Set: 0.9954, Test-Set: 0.9894
Mean test-set accuracy: 0.9888
Min test-set accuracy:  0.9875
Max test-set accuracy:  0.9896
(10000, 10)
(10000,)
"""


best_net_pred_labels = pred_labels[best_net, :, :]
best_net_cls_pred = np.argmax(best_net_pred_labels, axis=1)
best_net_correct = (best_net_cls_pred == data.test.cls)
best_net_incorrect = np.logical_not(best_net_correct)

np.sum(ensemble_correct)
np.sum(best_net_correct)

ensemble_better = np.logical_and(best_net_incorrect,
                                 ensemble_correct)

ensemble_better.sum()

best_net_better = np.logical_and(best_net_correct,
                                 ensemble_incorrect)

best_net_better.sum()
session.close()
