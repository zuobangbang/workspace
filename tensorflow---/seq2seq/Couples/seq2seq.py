import tensorflow as tf
from hyperparams import Hyperparams
#
# TrainingHelper：适用于训练的helper。
#
# InferenceHelper：适用于测试的helper。
#
# GreedyEmbeddingHelper：适用于测试中采用Greedy策略sample的helper。
#
# CustomHelper：用户自定义的helper。
def getLayerDense(layer_size, output_keep_prob, num_units):
    return tf.nn.rnn_cell.MultiRNNCell([
        tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units=num_units), output_keep_prob=output_keep_prob)
        for _ in range(layer_size)],state_is_tuple=True)


def encode(embed_input,enbed_input_length,encode_layers,keep_prob,num_units):
    encode_cell_bw = getLayerDense(encode_layers, keep_prob,num_units)
    encode_cell_fw = getLayerDense(encode_layers,keep_prob,num_units)
    encode_output, bi_encode_state = tf.nn.bidirectional_dynamic_rnn(encode_cell_fw, encode_cell_bw, embed_input,sequence_length=enbed_input_length,dtype=tf.float32)

    encode_output=tf.concat(encode_output,-1)
    encoder_state = []
    for layer_id in range(encode_layers):
        encoder_state.append(bi_encode_state[0][layer_id])
        encoder_state.append(bi_encode_state[1][layer_id])
    encoder_state = tuple(encoder_state)
    return encode_output, encoder_state

def get_lstm_cell(hidden_size):
    lstm_cell = tf.contrib.rnn.LSTMCell(
        num_units=hidden_size,
        initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1, seed=2019)
    )
    return lstm_cell
def encoder(hidden_size, num_layers,encoder_embedding_input,input_sequence_length):
    # encoder_embedding_input = tf.nn.embedding_lookup(params=emb_matrix, ids=input_data)
    encoder_cells = tf.contrib.rnn.MultiRNNCell(
        [get_lstm_cell(hidden_size) for i in range(num_layers)]
    )
    encoder_output, encoder_state= tf.nn.dynamic_rnn(cell=encoder_cells,
                  inputs=encoder_embedding_input,
                  sequence_length=input_sequence_length,
                  dtype=tf.float32
                 )
    return encoder_output, encoder_state
def attention(encode_output, in_seq_len, num_units, layer_size,keep_prob):
    attention_mechanism=tf.contrib.seq2seq.BahdanauAttention(num_units=num_units,memory=encode_output,memory_sequence_length=in_seq_len)
    # cells=getLayerDense(layer_size,keep_prob,num_units)
    cells=tf.contrib.rnn.MultiRNNCell(
        [get_lstm_cell(num_units) for i in range(layer_size)]
    )
    decode_cell=tf.contrib.seq2seq.AttentionWrapper(cell=cells,attention_mechanism=attention_mechanism,attention_layer_size=num_units)
    return decode_cell


def decode(encoder_state,encode_output, in_seq_len, num_units, layer_size,keep_prob,decoder_embedding_input, tar_seq_len,vocab_size,max_output_sequence_length):
    decode_cells=attention(encode_output, in_seq_len, num_units, layer_size,keep_prob)
    #Helper对象
    training_helper = tf.contrib.seq2seq.TrainingHelper(inputs = decoder_embedding_input,sequence_length = tar_seq_len)
    # init_state=decode_cells.zero_state(batch_size=tf.shape(in_seq_len)[0],dtype=tf.float32)
    init_state = decode_cells.zero_state(batch_size=Hyperparams.batch_size, dtype=tf.float32)
    project_layer=tf.layers.Dense(units=vocab_size,use_bias=False,
                                  kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))#权重矩阵初始化，vocab_size是词语的数量
    #Basic Decode
    training_decoder = tf.contrib.seq2seq.BasicDecoder(cell = decode_cells,helper = training_helper,output_layer = project_layer,initial_state = init_state)
    # Dynamic RNN
    training_final_output, training_final_state, training_sequence_length = tf.contrib.seq2seq.dynamic_decode(
        decoder = training_decoder, output_time_major = True,
    impute_finished = True, maximum_iterations = max_output_sequence_length)
    return training_final_output, training_final_state, training_sequence_length

def get_lstm_cell(hidden_size):
    lstm_cell = tf.contrib.rnn.LSTMCell(
        num_units=hidden_size,
        initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1, seed=2019)
    )
    return lstm_cell
def attention_decoder(decoder_embedding_input, hidden_size, num_layers,input_sequence_length,corpus_size,
            vocab_size, output_sequence_length, max_output_sequence_length,  encoder_output):
    # numpy数据切片 output_data[0:corpus_size:1,0:-1:1]，删除output_data最后一列数据
    decoder_cells = tf.contrib.rnn.MultiRNNCell([get_lstm_cell(hidden_size) for i in range(num_layers)])
    # Attention机制
    attention_mechanism = tf.contrib.seq2seq.LuongAttention(
        num_units=hidden_size,
        memory=encoder_output,
        memory_sequence_length=input_sequence_length
    )
    decoder_cells = tf.contrib.seq2seq.AttentionWrapper(
        cell=decoder_cells,
        attention_mechanism=attention_mechanism,
        attention_layer_size=hidden_size
    )
    project_layer = tf.layers.Dense(
    units=vocab_size, # 全连接层神经元个数
    kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1) # 权重矩阵初始化
    )
    with tf.variable_scope('Decoder'):
        # Helper对象
        training_helper = tf.contrib.seq2seq.TrainingHelper(
            inputs=decoder_embedding_input,
            sequence_length=output_sequence_length)
        # Basic Decoder
        training_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=decoder_cells,
            helper=training_helper,
            output_layer=project_layer,
            initial_state=decoder_cells.zero_state(batch_size=corpus_size, dtype=tf.float32)
        )
        # Dynamic RNN
        training_final_output, training_final_state, training_sequence_length = tf.contrib.seq2seq.dynamic_decode(
            decoder=training_decoder,
            maximum_iterations=max_output_sequence_length,
            impute_finished=True)
    # with tf.variable_scope('Decoder', reuse=True):
    #     # Helper对象
    #     start_tokens = tf.tile(input=tf.constant(value=[word2id['_BOS']], dtype=tf.int32),
    #                            multiples=[corpus_size], name='start_tokens')
    #     inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
    #         embedding=emb_matrix,
    #         start_tokens=start_tokens,
    #         end_token=word2id['_EOS'])
    #     # Basic Decoder
    #     inference_decoder = tf.contrib.seq2seq.BasicDecoder(
    #         cell=decoder_cells,
    #         helper=inference_helper,
    #         output_layer=project_layer,
    #         initial_state=decoder_cells.zero_state(batch_size=corpus_size, dtype=tf.float32)
    #     )
    #     # Dynamic RNN
    #     inference_final_output, inference_final_state, inference_sequence_length = tf.contrib.seq2seq.dynamic_decode(
    #         decoder=inference_decoder,
    #         maximum_iterations=max_inference_sequence_length,
    #         impute_finished=True)
    return training_final_output, training_final_state






