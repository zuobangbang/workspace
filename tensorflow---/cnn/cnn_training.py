from loadingdata import loadingdata
from cnn_classify import CNN,CNN_Config
import tensorflow as tf
from numpy import *


def feed_data(x_batch, y_batch, keep_prob,is_training):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob,
        model.is_training: is_training
    }
    return feed_dict
def evaluate(sess,x,y):
    length=len(x)
    acc_sum=0.0
    loss_sum=0.0
    for x_batch,y_batch in readdata.batch_item(x,y,batch_size=128):
        batch_len=len(x_batch)
        feed_dict=feed_data(x_batch,y_batch,config.dropout_keep_prob,False)
        loss,acc=sess.run([model.loss,model.acc],feed_dict=feed_dict)
        acc_sum+=acc*batch_len
        loss_sum+=loss*batch_len
    return acc_sum/length,loss_sum/length

if __name__=="__main__":
    total_batch=0
    batch_size = 64
    config=CNN_Config()
    model=CNN(config)
    readdata = loadingdata(path='/Users/zuobangbang/Desktop/cnews.train.txt', seq_length=600, word_size=30000,
                           batch_size=64, rate=0.8)
    train_data, train_label, test_data, test_label=readdata.read_data()
    # readdata = loadingdata(path='/Users/zuobangbang/Desktop/cnews.test.txt', seq_length=600, word_size=30000,
    #                        batch_size=64, rate=0.8)
    # test_data, test_label=readdata.read_data()
    print(shape(train_data))
    print(shape(train_label))
    session=tf.Session()
    session.run(tf.global_variables_initializer())
    for i in range(config.num_epochs):
        print('Epoch is ',i)
        result = readdata.batch_item(train_data, train_label, batch_size)
        for train_x, train_y in result:
            feed_dict = feed_data(train_x, train_y, config.dropout_keep_prob,False)
            session.run(model.opti, feed_dict=feed_dict)
            if total_batch % config.print_per_batch == 0:
                loss, acc = session.run([model.loss, model.acc], feed_dict=feed_dict)
                test_acc, test_loss = evaluate(session, test_data, test_label)
                print("total batch is {4},train loss is {0},train acc is {1},test loss is {2},test acc is {3}".format(
                    loss, acc, test_loss, test_acc, total_batch))
            total_batch += 1
