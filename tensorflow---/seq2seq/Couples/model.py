import tensorflow as tf
from hyperparams import Hyperparams
from load_data import load_vocab
from seq2seq import *



class Seq2Seq():
    def __init__(self):
        self.source=tf.placeholder(tf.int32,shape=[Hyperparams.batch_size,None],name='source')
        self.target=tf.placeholder(tf.int32,shape=[Hyperparams.batch_size,None],name='target')
        self.source_length=tf.placeholder(tf.int32,shape=[Hyperparams.batch_size,])
        self.target_length=tf.placeholder(tf.int32,shape=[Hyperparams.batch_size,])
        self.encode_layers=Hyperparams.encode_layers
        self.num_units=Hyperparams.num_units
        self.keep_prob=tf.placeholder(tf.float32,name='keep_prob')
        word2idx, idx2word=load_vocab()
        self.vocab_size=len(word2idx)
        self.max_output_sequence_length=Hyperparams.max_output_sequence_length
        self.seq()

    def seq(self):
        # tf.global_variables_initializer()
        char2idx, idx2char = load_vocab()
        embedding=tf.get_variable(name='embedding',dtype=tf.float32,shape=[len(char2idx),Hyperparams.embedding_size])
        self.enb_source=tf.nn.embedding_lookup(embedding,self.source)
        self.enb_target = tf.nn.embedding_lookup(embedding, self.target)
        # encode_output, encoder_state=encode(self.enb_source, self.source_length, self.encode_layers, self.keep_prob, self.num_units)
        encode_output,encoder_state=encoder(self.num_units, self.encode_layers,self.enb_source,self.source_length)
        # decode_cell=attention(encode_output, self.source_length, self.num_units, self.encode_layers, self.keep_prob)
        # attention_decoder(self.enb_target, self.num_units, self.encode_layers, self.source_length,  self.source_length,
        #                   self.vocab_size, self.target_length, self.max_output_sequence_length, encode_output)
        self.training_final_output, self.training_final_state, training_sequence_length=decode( encoder_state,encode_output, self.source_length,
                                                                                     self.num_units, self.encode_layers,
                                                                                     self.keep_prob, self.enb_target, self.target_length,
                                                                                     self.vocab_size,self.max_output_sequence_length)
        self.logit=tf.nn.softmax(self.training_final_output.rnn_output)



