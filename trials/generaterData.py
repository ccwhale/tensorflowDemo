import random
import unittest
import numpy as np

# 生成(6000,2)的训练集合
with open('label.txt', 'w') as file:
    with open('train.txt', 'w') as f:
        for x in range(6000):
            a = random.randint(0, 9)
            if a > 5:
                file.write('2')
            else:
                file.write('1')
            f.write(str(a) + '1')

# 生成(1000,2)的训练集合
with open('label_test.txt', 'w') as file:
    with open('train_test.txt', 'w') as f:
        for _ in range(1000):
            a = random.randint(0, 9)
            if a > 5:
                file.write('2')
            else:
                file.write('1')
            f.write(str(a) + '1')

#
# import numpy as np
#
# w = np.array([[-1.00487339, - 1.15891397, 2.16378975],
#               [-0.44634497, 4.75635386, -4.31000757]])
# b = np.array([-0.44633889, 4.75634384, -4.31001806])
# y = np.matmul([6, 1], w) + b
# x = np.matmul([5, 1], w) + b
#
# print(y)
# print(x)

w = np.array([[-4475.66650391, - 1870.66699219, 6346.33203125],
              [-1000.00128174, 11364., - 10364.]])
b = np.array([-0.44633889, 4.75634384, -4.31001806])
y = np.matmul([6, 1], w) + b
print(y)
