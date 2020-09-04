import tensorflow as tf
from hyperparams import Hyperparams
from load_data import *
from seq2seq import *
from model import Seq2Seq

def Feed_data(source,target,x_length,y_length,keep_prob):
    feed_dict={
        cou_model.source:source,
    cou_model.target :target,
    cou_model.source_length:x_length,
    cou_model.target_length:y_length,
        cou_model.keep_prob:keep_prob
    }
    return feed_dict
if __name__=='__main__':
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    total_loss=0.0
    cou_model=Seq2Seq()
    for i in range(Hyperparams.num_epochs):
        result=get_batch_data('x')
        t=0
        for source,target,x_length,y_length in result:
            t+=1
            feed_data=Feed_data(source,target,x_length,y_length,Hyperparams.encode_keep_prob)
            state,logit=sess.run([cou_model.training_final_state,cou_model.logit],feed_dict=feed_data)
            print(state)

