from load_data import LData
from params import poetry_params as pp
from lstm_model import LSTM_network
import tensorflow as tf


def feed_data(x_batch, y_batch, keep_prob):
    feed_dict = {
        lstm_model.input_x: x_batch,
        lstm_model.input_y: y_batch,
        lstm_model.keep_prob: keep_prob
    }
    return feed_dict

if __name__=='__main__':
    lstm_model=LSTM_network()
    session=tf.Session()
    readdata=LData()
    saver=tf.train.Saver()
    session.run(tf.global_variables_initializer())
    total_loss=0.0
    for i in range(pp.num_epochs):
        print("Epoch is ",i,'total_loss  is ',total_loss/(1+pp.num_epochs))
        result=readdata.get_batch()
        # session.run(tf.assign(lstm_model.learning_rate, 0.001 * (pp.learning_rate ** i)))
        t=0
        for train_x,train_y in result:
            t+=1
            feed_dict=feed_data(train_x,train_y,pp.keep_prob)
            opti,output,loss,logits=session.run([lstm_model.opti,lstm_model.output,lstm_model.loss,lstm_model.logits],feed_dict=feed_dict)
            # print('target.shape is :',target.shape,'logits.shape is :',logits.shape)
            # loss=session.run([lstm_model.loss],feed_dict=feed_dict)
            # print(loss)
            total_loss+=loss
            if t%50==0:
                print(t,loss,logits.shape,output.shape)
    saver.save(session,'model/poetry.module', global_step=100)


