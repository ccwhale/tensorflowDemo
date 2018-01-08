## 可视化训练过程

1. plot_first_image_of_training_data.png 可以看到训练集是手写的图片<br>

2. 在简单线性模型中和卷积中训练得出的权重<br>
weights_print_simple_linear.png<br>
weights_first_convolutional_neural_network.png<br>
可以看到上面的权重是比较直观的0,1,2,3，机器只识别这样的0,1,2,3...<br>
经过第一层卷积得到的权重，权重看不懂<br>

3. 经过一层和两层的卷积输出的图片，卷积就是上面看不懂的小图片每个像素上的权重*原图片，然后池化(池化的原因是减少计算量
作为人，就算把图片隔行抽掉一列像素，使整个图片的像素减半，我们还是可以看出这个图片是什么，机器也是这样，池化就是减少原图片的像素)<br>
plot_first_image_after_first_convolution.png 28*28 的图片 池化 有变为 14*14<br>
plot_first_image_after_second_convolution.png  14*14 池化后变为 7*7 <br>

4. 对角线上的是预测成功的，可以看到还是有些预测失败了。<br>
训练之后的性能矩阵.png<br>
