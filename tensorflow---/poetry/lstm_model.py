import tensorflow as tf
from params import poetry_params as pp

class LSTM_network(object):
    def __init__(self):
        # self.learning_rate=tf.Variable(0.0, trainable=False)
        self.learning_rate=pp.learning_rate
        self.n_layers=pp.n_layers
        self.hidden_dim=pp.hidden_dim
        self.keep_prob=pp.keep_prob
        self.batch_zise=pp.batch_size
        self.word_size=pp.word_size+1
        self.word_embedding=pp.word_embedding
        self.input_x = tf.placeholder(tf.int32, shape=[pp.batch_size,None], name='input_x')
        self.input_y = tf.placeholder(tf.int32, shape=[pp.batch_size,None], name='input_y')
        self.keep_prob=tf.placeholder(tf.float32,name='keep_prob')
        self.lstm()

    # embedding = tf.get_variable('embedding', shape=[self.word_size, self.word_embedding])
    # embedding_input = tf.nn.embedding_lookup(embedding, self.input_x)
    def lstm(self, targets=None):
        embedding = tf.get_variable(name='embedding', shape=[self.word_size, self.word_embedding])
        embedding_input = tf.nn.embedding_lookup(embedding, self.input_x)
        def dropout():
            cell=tf.nn.rnn_cell.LSTMCell(num_units=self.hidden_dim)
            return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
        cells=[dropout() for _ in range(self.n_layers)]

        lstm_cells=tf.nn.rnn_cell.MultiRNNCell(cells,state_is_tuple=True)
        initial_state = lstm_cells.zero_state(self.batch_zise, tf.float32)
        self.softmax_w=tf.get_variable(name='sotfmax_w',dtype=tf.float32,shape=[self.hidden_dim,self.word_size])
        self.softmax_b=tf.get_variable(name='sotfmax_b',dtype=tf.float32,shape=[self.word_size])
        self.outputs,state=tf.nn.dynamic_rnn(lstm_cells,inputs=embedding_input,dtype=tf.float32,initial_state=initial_state)
        self.output=tf.reshape(self.outputs, [-1, self.hidden_dim])
        self.logits= tf.matmul(self.output, self.softmax_w) + self.softmax_b
        self.probs=tf.nn.softmax(self.logits)
        # self.target=tf.reshape(tf.one_hot(self.input_y,self.word_size),[-1,self.word_size])
        self.target = tf.reshape(self.input_y, [-1])
        cross_enropy = tf.contrib.legacy_seq2seq.sequence_loss_by_example([self.logits], [self.target],[tf.ones_like(self.target, dtype=tf.float32)], self.word_size)
        # cross_enropy=tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,labels=self.target)
        self.loss=tf.reduce_mean(cross_enropy)
        self.opti=tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


