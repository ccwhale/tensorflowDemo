import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math

# from tf.keras.models import Sequential  # This does not work!
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, Activation
from tutorials.hvassTutorials.helperFunction import plot_images, plot_conv_weights

from tensorflow.examples.tutorials.mnist import input_data

data = input_data.read_data_sets("data/MNIST/", one_hot=True)

data.test.cls = np.array([label.argmax() for label in data.test.labels])

# We know that MNIST images are 28 pixels in each dimension.
img_size = 28

# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size

# Tuple with height and width of images used to reshape arrays.
# This is used for plotting the images.
img_shape = (img_size, img_size)

# Tuple with height, width and depth used to reshape arrays.
# This is used for reshaping in Keras.
img_shape_full = (img_size, img_size, 1)

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 1

# Number of classes, one class for each of 10 digits.
num_classes = 10

# Start construction of the Keras Sequential model.
# keras API有两种模型来建立神经网络。最简单的是Sequential Mdodel,只允许layer层添加到model中
# 构造函数
model = Sequential()

# Add an input layer which is similar to a feed_dict in TensorFlow.
# 添加一个input层，跟tensorflow的feed_dict一样 img_size_flat = 28*28 传的是数字,实际传入的shapes是(784,)
# Note that the input-shape must be a tuple containing the image-size.
model.add(InputLayer(input_shape=(img_size_flat,)))
# 可以设置默认的激活函数
# model.add(Activation('sigmoid'))

# The input is a flattened array with 784 elements,
# but the convolutional layers expect images with shape (28, 28, 1)
# img_shape_full = (img_size, img_size, 1) (28, 28, 1)
model.add(Reshape(img_shape_full))

# First convolutional layer with ReLU-activation and max-pooling.
# 第一层卷积()
model.add(Conv2D(kernel_size=5, strides=1, filters=16, padding='same', activation='relu', name='layer_conv1'))
model.add(MaxPooling2D(pool_size=3, strides=3))

# Second convolutional layer with ReLU-activation and max-pooling.
model.add(Conv2D(kernel_size=5, strides=1, filters=36, padding='same', activation='relu', name='layer_conv2'))
model.add(MaxPooling2D(pool_size=3, strides=3))

# Flatten the 4-rank output of the convolutional layers
# to 2-rank that can be input to a fully-connected / dense layer.
model.add(Flatten())

# First fully-connected / dense layer with ReLU-activation.
# 第一层全连接 relu 输入到128个神经元的
model.add(Dense(12, activation='relu'))

# Last fully-connected / dense layer with softmax-activation
# for use in classification.
# 第二层全连接,softmax函数
model.add(Dense(num_classes, activation='softmax'))

from tensorflow.python.keras.optimizers import Adam

# 优化函数 Adam
optimizer = Adam(lr=1e-3)

"""
模型编译:神经元网络已经定义并且必须通过添加一个损失函数，优化函数，性能度量确定下来。在keras中叫做模型编译(model compilation)
"""
# 对于分类问题例如MNIST有10个分类，我们使用loss-function,叫做categorical_crossentropy
# 我们关注的性能度量是分类的准确性
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 训练函数
# epochs=1, batch_size=128 执行1轮,每次取128个训练数据(784,128)
# Epoch 1/1
#  128/55000 [..............................] - ETA: 299s - loss: 2.3275 - acc: 0.0703
#  256/55000 [..............................] - ETA: 171s - loss: 2.2832 - acc: 0.1133
# acc 0.9799
# acc: 96.65% stride = 1
# acc: 95.97% stride = 3
# epoch=2  acc: 98.35%
# sigmoid sigmoid relu epoch = 1 acc 0.8352


model.fit(x=data.train.images,
          y=data.train.labels,
          epochs=1, batch_size=128)

# 评估
# 上面的模型已经训练好了,用测试数据计算数据的准确性
result = model.evaluate(x=data.test.images,
                        y=data.test.labels)

for name, value in zip(model.metrics_names, result):
    print(name, value)

print("{0}: {1:.2%}".format(model.metrics_names[1], result[1]))

# 存储模型

path_model = 'model.keras'

model.save(path_model)

del model

from tensorflow.python.keras.models import load_model

model1 = load_model(path_model)
images = data.test.images[0:9]
cls_true = data.test.cls[0:9]

y_pred = model1.predict(x=images)
cls_pred = np.argmax(y_pred, axis=1)
plot_images(images=images,
            cls_pred=cls_pred,
            cls_true=cls_true)

model1.summary()
layer_input = model1.layers[0]
layer_conv1 = model1.layers[2]

layer_conv2 = model1.layers[4]

weights_conv1 = layer_conv1.get_weights()[0]
print(weights_conv1.shape)
plot_conv_weights(weights=weights_conv1, input_channel=0)

# 输出卷积层的图片
