from hyperparams import Hyperparams
import numpy as np
def load_vocab():
    vocab = [line.split()[0] for line in open(Hyperparams.vocab, 'r') if int(line.split('\t')[1])>=Hyperparams.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word


def load_train_data():
    data_in = [i.replace('\n', '') for i in open(Hyperparams.source_train, 'r')]
    data_out = [i.replace('\n', '') for i in open(Hyperparams.target_train, 'r')]
    return data_in, data_out

def load_test_data():
    data_in = [i.replace('\n','') for i in open(Hyperparams.source_test,'r')]
    data_out=[i.replace('\n','') for i in open(Hyperparams.target_test,'r')]
    return data_in,data_out


def create_data(source_sents, target_sents):
    char2idx, idx2char = load_vocab()

    # Index
    x_list, y_list, Sources, Targets = [], [], [], []
    for i,(source_sent, target_sent) in enumerate(zip(source_sents, target_sents)):
        x =  [char2idx.get(word) for word in ("<S> "+source_sent.strip() + " </S>").split()] # 1: OOV, </S>: End of Text
        y = [char2idx.get(word) for word in ("<S> "+target_sent.strip() + " </S>").split()]
        if None in x or None in y:
            pass
        else:
            if max(len(x), len(y)) <= Hyperparams.maxlen:
                x_list.append(np.array(x))
                y_list.append(np.array(y))
                Sources.append(source_sent)
                Targets.append(target_sent)

    # Pad
    X = np.zeros([len(x_list), Hyperparams.maxlen], np.int32)
    Y = np.zeros([len(y_list), Hyperparams.maxlen], np.int32)
    x_length,y_length=[],[]
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        x_length.append(len(x))
        y_length.append(len(y))
        X[i] = np.lib.pad(x, [0, Hyperparams.maxlen - len(x)], 'constant', constant_values=(0, 0))
        Y[i] = np.lib.pad(y, [0, Hyperparams.maxlen - len(y)], 'constant', constant_values=(0, 0))
    return X, Y, Sources, Targets,x_length,y_length

def get_batch_data(t):
    if t=='train':
        source,target=load_train_data()
    else:
        source, target = load_test_data()
    X, Y, Sources, Targets,x_length,y_length=create_data(source,target)
    n=int(len(source)/Hyperparams.batch_size)
    for i in range(n+1):
        start_id = i * Hyperparams.batch_size
        end_id = min((i + 1) * Hyperparams.batch_size, len(source))
        yield X[start_id:end_id], Y[start_id:end_id],x_length[start_id:end_id],y_length[start_id:end_id]

