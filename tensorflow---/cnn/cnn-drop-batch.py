import  tensorflow as tf
from  tensorflow.examples.tutorials.mnist import  input_data

#导入数据
# mnist=input_data.read_data_sets('./data',one_hot=True)


import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
from functools import partial


# 记录训练花费的时间
def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    # timedelta是用于对间隔进行规范化输出，间隔10秒的输出为：00:00:10
    return timedelta(seconds=int(round(time_dif)))


# 准备训练数据集、验证集和测试集，并生成小批量样本
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

X_train = X_train.astype(np.float32).reshape(-1, 28 * 28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28 * 28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]


def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch


height = 28
width = 28
channels = 1
n_inputs = height * width

# 第一个卷积层有16个卷积核
# 卷积核的大小为（3,3）
# 步幅为1
# 通过补零让输入与输出的维度相同
conv1_fmaps = 16
conv1_ksize = 3
conv1_stride = 1
conv1_pad = "SAME"

conv2_fmaps = 32
conv2_ksize = 3
conv2_stride = 1
conv2_pad = "SAME"
# 在池化层丢弃25%的神经元
conv2_dropout_rate = 0.25

pool3_fmaps = conv2_fmaps

n_fc1 = 32
# 在全连接层丢弃50%的神经元
fc1_dropout_rate = 0.5

n_outputs = 10

with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, shape=[None, n_inputs], name="X")
    X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
    y = tf.placeholder(tf.int32, shape=[None], name="y")

training = tf.placeholder_with_default(False, shape=[], name='training')
# 构建一个batch norm层，便于复用。用移动平均求全局的样本均值和方差，动量参数取0.9
my_batch_norm_layer = partial(tf.layers.batch_normalization,
                              training=training, momentum=0.9)

with tf.name_scope("conv"):
    # batch norm之后在激活，所以这里不设定激活函数
    conv1 = tf.layers.conv2d(X_reshaped, filters=conv1_fmaps, kernel_size=conv1_ksize,
                             strides=conv1_stride, padding=conv1_pad,
                             activation=None, name="conv1")
    # 进行batch norm之后，再激活
    batch_norm1 = tf.nn.selu(my_batch_norm_layer(conv1))
    conv2 = tf.layers.conv2d(batch_norm1, filters=conv2_fmaps, kernel_size=conv2_ksize,
                             strides=conv2_stride, padding=conv2_pad,
                             activation=None, name="conv2")
    batch_norm2 = tf.nn.selu(my_batch_norm_layer(conv2))

with tf.name_scope("pool3"):
    pool3 = tf.nn.max_pool(batch_norm2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
    # 把特征图拉平成一个向量
    pool3_flat = tf.reshape(pool3, shape=[-1, pool3_fmaps * 14 * 14])
    # 丢弃25%的神经元
    pool3_flat_drop = tf.layers.dropout(pool3_flat, conv2_dropout_rate, training=training)

with tf.name_scope("fc1"):
    fc1 = tf.layers.dense(pool3_flat_drop, n_fc1, activation=None, name="fc1")
    # 在全连接层进行batch norm，然后激活
    batch_norm4 = tf.nn.selu(my_batch_norm_layer(fc1))
    # 丢弃50%的神经元
    fc1_drop = tf.layers.dropout(batch_norm4, fc1_dropout_rate, training=training)

with tf.name_scope("output"):
    logits = tf.layers.dense(fc1_drop, n_outputs, name="output")
    logits_batch_norm = my_batch_norm_layer(logits)
    Y_proba = tf.nn.softmax(logits_batch_norm, name="Y_proba")

with tf.name_scope("loss_and_train"):
    learning_rate = 0.01
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_batch_norm, labels=y)
    loss = tf.reduce_mean(xentropy)

    optimizer = tf.train.AdamOptimizer(learning_rate)
    # 这是需要额外更新batch norm的参数
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # 模型参数的优化依赖与batch norm参数的更新
    with tf.control_dependencies(extra_update_ops):
        training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

n_epochs = 20
batch_size = 100

with tf.Session() as sess:
    init.run()
    start_time = time.time()

    # 记录总迭代步数，一个batch算一步
    # 记录最好的验证精度
    # 记录上一次验证结果提升时是第几步。
    # 如果迭代2000步后结果还没有提升就中止训练。
    total_batch = 0
    best_acc_val = 0.0
    last_improved = 0
    require_improvement = 2000

    flag = False
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):

            sess.run(training_op, feed_dict={training: True, X: X_batch, y: y_batch})

            # 每次迭代10步就验证一次
            if total_batch % 10 == 0:
                acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
                # 每次输入1000个样本进行评估，然后求平均值
                acc_val = []
                for i in range(len(X_valid) // 1000):
                    acc_val.append(accuracy.eval(
                        feed_dict={X: X_valid[i * 1000:(i + 1) * 1000], y: y_valid[i * 1000:(i + 1) * 1000]}))
                acc_val = np.mean(acc_val)

                # 如果验证精度提升了，就替换为最好的结果，并保存模型
                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    last_improved = total_batch
                    save_path = saver.save(sess, "./my_model_CNN_stop.ckpt")
                    improved_str = 'improved!'
                else:
                    improved_str = ''

                # 记录训练时间，并格式化输出验证结果，如果提升了，会在后面提示：improved！
                time_dif = get_time_dif(start_time)
                msg = 'Epoch:{0:>4}, Iter: {1:>6}, Acc_Train: {2:>7.2%}, Acc_Val: {3:>7.2%}, Time: {4} {5}'
                print(msg.format(epoch, total_batch, acc_batch, acc_val, time_dif, improved_str))

            # 记录总迭代步数
            total_batch += 1

            # 如果2000步以后还没提升，就中止训练。
            if total_batch - last_improved > require_improvement:
                print("Early stopping in  ", total_batch, " step! And the best validation accuracy is ", best_acc_val,
                      '.')
                # 跳出这个轮次的循环
                flag = True
                break
        # 跳出所有训练轮次的循环
        if flag:
            break

with tf.Session() as sess:
    saver.restore(sess, "./my_model_CNN_stop.ckpt")
    # 每次输入1000个样本进行测试，再求平均值
    acc_test = []
    for i in range(len(X_test) // 1000):
        acc_test.append(
            accuracy.eval(feed_dict={X: X_test[i * 1000:(i + 1) * 1000], y: y_test[i * 1000:(i + 1) * 1000]}))
    acc_test = np.mean(acc_test)
    print("\nTest_accuracy:{0:>7.2%}".format(acc_test))