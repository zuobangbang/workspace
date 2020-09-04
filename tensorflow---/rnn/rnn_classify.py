import  tensorflow  as tf
import jieba
from string import punctuation
import tensorflow.contrib.keras as kr
from numpy import *
class rnnconfig(object):
    word_embedding = 64
    seq_length = 190
    word_size = 5000
    batch_size = 64
    n_classes = 2
    hidden_dim = 128
    n_layers = 2
    dropout_keep_prob = 0.8
    learningrate = 0.001
    num_epochs = 2000
    print_per_batch=10

class RNN(object):
    def __init__(self,config):
        self.config=config
        self.word_embedding=self.config.word_embedding
        self.seq_length = self.config.seq_length
        self.word_size =self.config.word_size
        self.batch_size =self.config.batch_size
        self.n_classes=self.config.n_classes
        self.hidden_dim=self.config.hidden_dim
        self.n_layers=self.config.n_layers
        self.dropout_keep_prob=self.config.dropout_keep_prob
        self.learningrate=self.config.learningrate
        self.num_epochs=self.config.num_epochs
        self.input_x = tf.placeholder(tf.int32, shape=[None, self.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, shape=[None, self.n_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.rnn()




    def rnn(self):

        def dropout():
            cell=tf.nn.rnn_cell.BasicRNNCell(self.hidden_dim)
            return tf.nn.rnn_cell.DropoutWrapper(cell,output_keep_prob=self.keep_prob)

        embedding=tf.get_variable('embedding',shape=[self.word_size,self.word_embedding])
        embedding_input=tf.nn.embedding_lookup(embedding, self.input_x)

        cells=[dropout() for _ in range(self.n_layers)]
        rnn_cells=tf.nn.rnn_cell.MultiRNNCell(cells,state_is_tuple=True)
        output,state=tf.nn.dynamic_rnn(rnn_cells,inputs=embedding_input,dtype=tf.float32)
        last=output[:,-1,:]
        # 取最后一个时序输出作为结果进入全联接层
        #rnn第二层的输出接全连接层，进行最后的输出
        fc=tf.layers.dense(last,units=self.hidden_dim,name='fc1',activation=None)
        fc=tf.layers.dropout(fc,self.keep_prob)
        fc=tf.nn.relu(fc)

        self.logit=tf.layers.dense(fc,self.n_classes,name='fc2')
        self.pre_y=tf.argmax(tf.nn.softmax(self.logit), 1)
        cross_enropy=tf.nn.softmax_cross_entropy_with_logits(logits=self.logit, labels=self.input_y)
        self.loss=tf.reduce_mean(cross_enropy)
        self.opti=tf.train.AdagradOptimizer(learning_rate=self.learningrate).minimize(self.loss)
        correct_pre=tf.equal(self.pre_y, tf.argmax(self.input_y, 1))
        self.acc=tf.reduce_mean(tf.cast(correct_pre,tf.float32))




