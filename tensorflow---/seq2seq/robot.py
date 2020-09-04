import os
import jieba
import json
from gensim.models import Word2Vec
import numpy as np
import tensorflow as tf
corpus_path = 'data.txt'

corpus = []
with open(corpus_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        corpus.extend(lines)
corpus = [sentence.replace('\n', '') for sentence in corpus]
corpus = [sentence.replace('\ufeff', '') for sentence in corpus]
print('语料库读取完成'.center(30, '='))



corpus_cut = [jieba.lcut(sentence) for sentence in corpus]
print('分词完成'.center(30, '='))
from tkinter import _flatten
tem = _flatten(corpus_cut)
_PAD, _BOS, _EOS, _UNK = '_PAD', '_BOS', '_EOS', '_UNK'
all_dict = [_PAD, _BOS, _EOS, _UNK] + list(set(tem))
print('词典构建完成'.center(30, '='))

id2word = {i: j for i, j in enumerate(all_dict)}
word2id = {j: i for i, j in enumerate(all_dict)}
# dict(zip(id2word.values(), id2word.keys()))
print('映射关系构建完成'.center(30, '='))

ids = [[word2id.get(word, word2id[_UNK]) for word in sentence] for sentence in corpus_cut]

fromids = ids[::2]
toids = ids[1::2]
len(fromids) == len(toids)

emb_size = 50
tmp = [list(map(str, id)) for id in ids]
if not os.path.exists('word2vec.model'):
    model = Word2Vec(tmp, size=emb_size, window=10, min_count=1, workers=-1)
    model.save('word2vec.model')
else:
    print('词向量模型已构建，可直接调取'.center(50, '='))

# 用记事本存储
# with open('fromids.txt', 'w', encoding='utf-8') as f:
#     f.writelines([' '.join(map(str, fromid)) for fromid in fromids])
# # 用json存储
# with open('ids.json', 'w') as f:
#     json.dump({'fromids':fromids, 'toids':toids}, fp=f, ensure_ascii=False)
#
#
# with open('ids.json', 'r') as f:
#     tmp = json.load(f)
# fromids = tmp['fromids']
# toids = tmp['toids']
# with open('dic.txt', 'r', encoding='utf-8') as f:
#     all_dict = f.read().split('\n')
# word2id = {j: i for i, j in enumerate(all_dict)}
# id2word = {i: j for i, j in enumerate(all_dict)}
model = Word2Vec.load('word2vec.model')
emb_size = model.layer1_size

vocab_size = len(all_dict)  # 词典大小
corpus_size = len(fromids)  # 对话长度

embedding_matrix = np.zeros((vocab_size, emb_size), dtype=np.float32)
tmp = np.diag([1] * emb_size) # 对于词典中不存在的词语

k = 0
for i in range(vocab_size):
    try:
        embedding_matrix[i] = model.wv[str(i)]
    except:
        embedding_matrix[i] = tmp[k]
        k += 1

from_length = [len(i) for i in fromids]
max_from_length = max(from_length)
source = [i + [word2id['_PAD']] * (max_from_length - len(i)) for i in fromids]
to_length = [len(i) for i in toids]
max_to_length = max(to_length)
target = [i + [word2id['_PAD']] * (max_to_length - len(i)) for i in toids]


num_layers = 2 # 神经元层数
hidden_size = 100 # 隐藏神经元个数
learning_rate = 0.001 # 学习率，0.0001-0.001
max_inference_sequence_length = 50
with tf.variable_scope('tensor', reuse=tf.AUTO_REUSE):
    # 输入
    input_data = tf.placeholder(tf.int32, [corpus_size, None], name='source')
    # 输出
    output_data = tf.placeholder(tf.int32, [corpus_size, None], name='target')
    # 输入句子的长度
    input_sequence_length = tf.placeholder(tf.int32, [corpus_size,], name='source_sequence_length')
    # 输出句子的长度
    output_sequence_length = tf.placeholder(tf.int32, [corpus_size,], name='target_sequence_length')
    # 输出句子的最大长度
    max_output_sequence_length = tf.reduce_max(output_sequence_length)
    # 词向量矩阵
    emb_matrix = tf.constant(embedding_matrix, name='embedding_matrix', dtype=tf.float32)


def get_lstm_cell(hidden_size):
    lstm_cell = tf.contrib.rnn.LSTMCell(
        num_units=hidden_size,
        initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1, seed=2019)
    )
    return lstm_cell
def encoder(hidden_size, num_layers, emb_matrix, input_data):
    encoder_embedding_input = tf.nn.embedding_lookup(params=emb_matrix, ids=input_data)
    encoder_cells = tf.contrib.rnn.MultiRNNCell(
        [get_lstm_cell(hidden_size) for i in range(num_layers)]
    )
    encoder_output, encoder_state= tf.nn.dynamic_rnn(cell=encoder_cells,
                  inputs=encoder_embedding_input,
                  sequence_length=input_sequence_length,
                  dtype=tf.float32
                 )
    return encoder_output, encoder_state


def attention_decoder(output_data, corpus_size, word2id, emb_matrix, hidden_size, num_layers,
            vocab_size, output_sequence_length, max_output_sequence_length, max_inference_sequence_length, encoder_output):
    # numpy数据切片 output_data[0:corpus_size:1,0:-1:1]，删除output_data最后一列数据
    ending = tf.strided_slice(output_data, begin=[0, 0], end=[corpus_size, -1], strides=[1, 1])
    begin_sigmal = tf.fill(dims=[corpus_size, 1], value=word2id['_BOS'])
    decoder_input_data = tf.concat([begin_sigmal, ending], axis=1, name='decoder_input_data')
    decoder_embedding_input = tf.nn.embedding_lookup(params=emb_matrix, ids=decoder_input_data)
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
    with tf.variable_scope('Decoder', reuse=True):
        # Helper对象
        start_tokens = tf.tile(input=tf.constant(value=[word2id['_BOS']], dtype=tf.int32),
                               multiples=[corpus_size], name='start_tokens')
        inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            embedding=emb_matrix,
            start_tokens=start_tokens,
            end_token=word2id['_EOS'])
        # Basic Decoder
        inference_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=decoder_cells,
            helper=inference_helper,
            output_layer=project_layer,
            initial_state=decoder_cells.zero_state(batch_size=corpus_size, dtype=tf.float32)
        )
        # Dynamic RNN
        inference_final_output, inference_final_state, inference_sequence_length = tf.contrib.seq2seq.dynamic_decode(
            decoder=inference_decoder,
            maximum_iterations=max_inference_sequence_length,
            impute_finished=True)
    return training_final_output, training_final_state, inference_final_output, inference_final_state


encoder_output, encoder_state = encoder(hidden_size, num_layers, emb_matrix, input_data)
# training_final_output, training_final_state, inference_final_output, inference_final_state = decoder(
#     output_data, corpus_size, word2id, emb_matrix, hidden_size, num_layers, vocab_size,
#     output_sequence_length, max_output_sequence_length, max_inference_sequence_length, encoder_state)
training_final_output, training_final_state, inference_final_output, inference_final_state = attention_decoder(
    output_data, corpus_size, word2id, emb_matrix, hidden_size, num_layers, vocab_size,
    output_sequence_length, max_output_sequence_length, max_inference_sequence_length, encoder_output)

def seq_loss(training_logits,target, seq_len,max_seq_len):
    # target = target[:, 1:]
    masks = tf.sequence_mask(seq_len, max_seq_len, dtype=tf.float32, name='masks')

    cost =tf.contrib.seq2seq.sequence_loss(training_logits,target,masks)
    return cost
# tf.identity 相当与 copy
#
training_logits = tf.identity(input=training_final_output.rnn_output, name='training_logits')
# inference_logits的值表示在最大句长的预测句子的词id，shape=(batch_size,max_sequence_length)
inference_logits = tf.identity(input=inference_final_output.sample_id, name='inference_logits')
cost=seq_loss(training_logits, target,output_sequence_length,max_output_sequence_length)
# cost=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=training_logits,
#                                                labels=target)
# [2,5] -> [[1,1,0,0,0],[1,1,1,1,1]]
mask = tf.sequence_mask(lengths=output_sequence_length, maxlen=max_output_sequence_length, name='mask', dtype=tf.float32)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

gradients = optimizer.compute_gradients(cost) # 计算损失函数的梯度
clip_gradients = [(tf.clip_by_value(t=grad, clip_value_max=5, clip_value_min=-5),var) for grad, var in gradients if grad is not None]
train_op = optimizer.apply_gradients(clip_gradients)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ckpt_dir = './checkpoint/'
    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint(checkpoint_dir=ckpt_dir)
    if ckpt:
        saver.restore(sess, ckpt)
        print('加载模型完成')
    else:
        print('没有找到训练过的模型')
    for n in range(200):
        _, training_pre, loss,tr,j= sess.run([train_op, training_final_output.sample_id, cost,training_logits,inference_logits],
            feed_dict={
                input_data:source,
                output_data:target,
                input_sequence_length:from_length,
                output_sequence_length:to_length
        })
        if n % 100 == 0:
            print(f'第{n}次训练'.center(50, '='))
            print(j)
            print(f'损失值为{loss}'.center(50, '='))
            print(tr)
            saver.save(sess, ckpt_dir + 'trained_model.ckpt')
            inference_pre = sess.run(
                inference_final_output.sample_id,
                feed_dict={
                    input_data:source,
                    input_sequence_length:from_length
                })
            # for j in range(2):
            #     print('输入：',' '.join([id2word[i] for i in source[j] if i != word2id['_PAD']]))
            #     print('输出：',' '.join([id2word[i] for i in target[j] if i != word2id['_PAD']]))
            #     print('Train预测：',' '.join([id2word[i] for i in training_pre[j] if i != word2id['_PAD']]))
            #     print('Inference预测：',' '.join([id2word[i] for i in inference_pre[j] if i != word2id['_PAD']]))
            print('模型已保存'.center(50, '='))