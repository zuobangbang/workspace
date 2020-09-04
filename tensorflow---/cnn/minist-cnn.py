import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 加载数据集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 每个批次的大小
batch_size = 50

# 共有xx个批次
n_batch = mnist.train.num_examples // batch_size


# 初始化权重
def weight_variable(shape):
    # 生成shape结构的变量,方差=0.1
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 初始化偏置
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 卷积层
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 池化层
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 定义变量等
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# 将一维的图片数据转化为2维/张
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 开始写我们的第一层卷积层
# 初始化第一层的权重,第一层的权重应该为5*5*32的,在一通道的图像上采集
# 初始化第一层的偏置,第一层的偏置应与第一层卷积核数一致
# 得到第一层的卷积输出,=用32个卷积核分别对其循环相乘在相加
# 进行第一层的池化,池化采用2*2的卷积核,采用samepadding的方式进行池化
W_conv1 = weight_variable([5, 5, 1, 32])
B_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + B_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 第2层的卷积
W_conv2 = weight_variable([5, 5, 32, 64])
B_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + B_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 使用全连接神经网络进行训练
# 经过第一层卷积池化之后得到14*14*32
# 经过第二层卷积池化之后得到7*7*64,通道数为64

# 定义神经网络,输入为7*7*64,第一层隐藏层采用1024个神经元
# 初始化第一层的权重
W_nn1 = weight_variable([7 * 7 * 64, 1024])
B_nn1 = bias_variable([1024])

# 此时将我们的输出展开为1维,以方便神经网络进行计算
h_x_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

# 进行神经网络计算,h_nn1为第一层隐藏层的输出
h_nn1 = tf.nn.relu(tf.matmul(h_x_flat, W_nn1) + B_nn1)

# 进行dropout处理
keep_props = tf.placeholder(tf.float32)
h_nn1 = tf.nn.dropout(h_nn1, keep_props)

# 第二层神经层
W_nn2 = weight_variable([1024, 10])
B_nn2 = bias_variable([10])
h_nn2 = tf.nn.relu(tf.matmul(h_nn1, W_nn2) + B_nn2)

# 得到神经网络的预测结果h_nn2

# 定义神经网络代价函数
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=h_nn2))

# 定义优化器
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 预测结果和正确结果的比对
correct_prediction = tf.equal(tf.argmax(h_nn2, 1), tf.argmax(y, 1))

# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 进行循环反向传递

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(21):
        for batch in range(n_batch):
            x_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: x_batch, y: y_batch, keep_props: 0.7})

        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_props: 1})
        print('iter {0}, acc={1}'.format(i, acc))
