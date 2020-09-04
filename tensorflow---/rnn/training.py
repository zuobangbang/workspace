from loadingdata import loadingdata
from rnn_classify import RNN,rnnconfig
import tensorflow as tf
from numpy import *


def feed_data(x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict
def evaluate(sess,x,y):
    length=len(x)
    acc_sum=0.0
    loss_sum=0.0
    for x_batch,y_batch in readdata.batch_item(x,y,batch_size=128):
        batch_len=len(x_batch)
        feed_dict=feed_data(x_batch,y_batch,rnnconfig.dropout_keep_prob)
        loss,acc=sess.run([model.loss,model.acc],feed_dict=feed_dict)
        acc_sum+=acc*batch_len
        loss_sum+=loss*batch_len
    return acc_sum/length,loss_sum/length


if __name__=="__main__":
    config=rnnconfig()
    model=RNN(config)
    readdata=loadingdata(path='/Users/zuobangbang/Desktop/tb.txt', seq_length=190, word_size=5000, batch_size=64,rate=0.8)
    train_data,train_label,test_data,test_label=readdata.read_data()
    print(shape(train_data))
    print(shape(train_label))
    batch_size=128
    total_batch=0
    stop_inter=1000
    session=tf.Session()
    session.run(tf.global_variables_initializer())
    for i in range(rnnconfig.num_epochs):
        print('Epoch is %d'%i)
        result=readdata.batch_item(train_data,train_label,batch_size)
        for train_x,train_y in result:
            feed_dict=feed_data(train_x,train_y,config.dropout_keep_prob)
            session.run(model.opti,feed_dict=feed_dict)
            if total_batch%rnnconfig.print_per_batch==0:
                loss,acc=session.run([model.loss,model.acc],feed_dict=feed_dict)
                test_acc,test_loss=evaluate(session,test_data,test_label)
                print("total batch is {4},train loss is {0},train acc is {1},test loss is {2},test acc is {3}".format(loss,acc,test_loss,test_acc,total_batch))
            total_batch+=1

        # config=rnnconfig()
        # model=RNN.rnn(config)
        # print(model.loss)