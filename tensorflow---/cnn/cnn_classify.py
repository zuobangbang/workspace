import tensorflow as tf


class CNN_Config(object):
    embedding_dim=64
    seq_length=600
    word_size=30000
    dropout_keep_prob=0.3

    conv_nums=1

    num_classes=10
    conv1_num_filters=[256,128]
    conv1_kernel_size=[5,3]
    conv1_stride=[1,1]
    conv1_padding=['same','same']


    pool_feature=conv1_num_filters[conv_nums-1]
    n_hidden=128
    learning_rate=0.001
    batch_size = 64
    num_epochs = 10
    print_per_batch = 100
    save_per_batch = 10

class CNN(object):
    def __init__(self,config):
        self.config=config
        self.embedding_dim = self.config.embedding_dim
        self.seq_length = self.config.seq_length
        self.word_size = self.config.word_size

        self.num_classes = self.config.num_classes
        self.conv_nums=self.config.conv_nums
        self.conv1_num_filters =self.config.conv1_num_filters
        self.conv1_kernel_size = self.config.conv1_kernel_size
        self.conv1_stride = self.config.conv1_stride
        self.conv1_padding =self.config.conv1_padding

        # self.conv2_num_filters = self.config.conv2_num_filters
        # self.conv2_kernel_size = self.config.conv2_kernel_size
        # self.conv2_stride = self.config.conv2_stride
        # self.conv2_padding = self.config.conv2_padding

        self.pool_feature =self.config.pool_feature
        self.n_hidden =self.config.n_hidden
        self.learning_rate = self.config.learning_rate
        self.batch_size = self.config.batch_size
        self.num_epochs = self.config.num_epochs
        self.print_per_batch = self.config.print_per_batch
        self.save_per_batch = self.config.save_per_batch
        self.input_x=tf.placeholder(tf.int32,shape=[None,self.seq_length])
        self.input_y=tf.placeholder(tf.float32,shape=[None,self.num_classes])
        self.keep_prob=tf.placeholder(tf.float32,name='keep_prob')
        self.cnn()

    def cnn(self):
        embedding=tf.get_variable('embedding',shape=[self.word_size,self.embedding_dim])
        embedding_input=tf.nn.embedding_lookup(embedding,self.input_x)
        self.is_training=tf.placeholder(tf.bool)
        # self.pooled_outputs = []
        # for i in range(self.conv_nums):
        #     # b = tf.Variable(tf.constant(0.1, shape=[self.conv1_num_filters[i]]), name="b")
        #     conv1=tf.layers.conv1d(embedding_input,filters=self.conv1_num_filters[i],padding=self.conv1_padding[i],strides=self.conv1_stride[i],kernel_size=self.conv1_kernel_size[i])
        #     # conv=tf.nn.relu(tf.nn.bias_add(conv1,b),name='relu')
        #     conv = tf.nn.relu(conv1, name='relu')
        #     # pooled=tf.nn.max_pool(conv,ksize=[1,self.seq_length+1-self.conv1_kernel_size[i],1,1],strides=[1,1,1,1],padding='VALID',name='pool')
        #     pooled = tf.reduce_max(conv1, reduction_indices=[1], name='pooled')
        #     self.pooled_outputs.append(pooled)
        #     print(self.pooled_outputs)
        i=0
        conv1 = tf.layers.conv1d(embedding_input, filters=self.conv1_num_filters[i], padding=self.conv1_padding[i],
                                 strides=self.conv1_stride[i], kernel_size=self.conv1_kernel_size[i])
        conv = tf.nn.relu(conv1, name='relu')
        # pooled=tf.nn.max_pool(conv,ksize=[1,self.seq_length+1-self.conv1_kernel_size[i],1,1],strides=[1,1,1,1],padding='VALID',name='pool')
        pooled = tf.reduce_max(conv1, reduction_indices=[1], name='pooled')
        fc=tf.layers.dense(pooled,units=self.n_hidden,name='fc',activation=None)
        fc=tf.layers.dropout(fc,self.keep_prob)
        fc=tf.nn.relu(tf.layers.batch_normalization(fc, training=self.is_training))
        self.logit=tf.layers.dense(fc,units=self.num_classes)
        self.pred_y=tf.argmax(tf.nn.softmax(self.logit),1)
        cross_entropy=tf.nn.softmax_cross_entropy_with_logits(logits=self.logit,labels=self.input_y)
        self.loss=tf.reduce_mean(cross_entropy)
        self.opti=tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        corret_pred=tf.equal(tf.argmax(self.input_y,1),self.pred_y)
        self.acc=tf.reduce_mean(tf.cast(corret_pred,tf.float32))




