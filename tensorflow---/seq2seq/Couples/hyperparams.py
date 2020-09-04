class Hyperparams(object):
    '''Hyperparameters'''
    # data
    source_train = '/Users/zuobangbang/Desktop/data/train/in.txt'
    target_train = '/Users/zuobangbang/Desktop/data/train/out.txt'
    source_test = '/Users/zuobangbang/Desktop/data/test/in.txt'
    target_test = '/Users/zuobangbang/Desktop/data/test/out.txt'
    vocab = '/Users/zuobangbang/Desktop/data/vocab'

    # training
    batch_size = 128
    lr = 0.0001  # learning rate.
    logdir = 'logdir'
    embedding_size=64
    encode_layers=2
    encode_keep_prob=0.8
    num_units=256
    max_output_sequence_length=100

    # model
    maxlen = 30  # max length for a sentence
    min_cnt = 0  # frequency threshold for vocabulary
    hidden_units = 512
    num_blocks = 6  # number of encoder/decoder blocks
    num_epochs = 20
    num_heads = 8
    dropout_rate = 0.1
    sinusoid = False  # If True, use sinusoid position embedding. If false, positional embedding.