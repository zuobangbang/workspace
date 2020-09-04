from string import punctuation
import tensorflow.contrib.keras as kr
import jieba
from numpy import *

# path='/Users/zuobangbang/Desktop/tb.txt'
# seq_length = 100
# word_size = 5000
# batch_size=64
class loadingdata(object):
    def __init__(self,path,seq_length,word_size,batch_size,rate):
        self.path=path
        self.seq_length =seq_length
        self.word_size =word_size
        self.batch_size=batch_size
        self.rate=rate

    def read_data(self):
        f=open(self.path,'r')
        data=[]
        label=[]
        for i in f:
            f=i.replace('\n','').strip()
            h=f[-2:]
            data.append(f[:-3])
            if h=='正面':
                label.append(1)
            else:
                label.append(0)
        indices = random.permutation(arange(len(data)))
        data, label=self.fenci(data,label)
        data, label = array(data), array(label)
        data_shuffled = data[indices]
        label_shuffled = label[indices]
        k=int(len(data)*self.rate)
        train_data,test_data=data_shuffled[:k],data_shuffled[k:]
        train_label,test_label=label_shuffled[:k],label_shuffled[k:]

        return train_data,train_label,test_data,test_label

    def fenci(self, data, label):
        word_dict={}
        jiedata=[]
        for i in data:
            w=jieba.cut(i,cut_all=True)
            ww='/'.join(w)
            x=[]
            for j in ww.split('/'):
                if  not j in punctuation:
                    x.append(j)
                    if not word_dict.__contains__(j):
                        word_dict[j]=1
                    else:
                        word_dict[j]+=1
            jiedata.append(x)
        word_dicr_sequen=sorted(word_dict.items(),key=lambda x:x[1],reverse=True)[:self.word_size]
        word_dict={}
        for i in range(len(word_dicr_sequen)):
            word_dict[word_dicr_sequen[i][0]]=i
        word_id=[]
        for i in jiedata:
            s=[]
            for j in i:
                if j in word_dict:
                    s.append(word_dict[j])
            word_id.append(s)
        x_pad = kr.preprocessing.sequence.pad_sequences(word_id, self.seq_length)
        y_pad = kr.utils.to_categorical(label, num_classes=2)
        return x_pad,y_pad

    def batch_item(self,data,label,batch_size):
        # indices = random.permutation(arange(len(data)))
        # data, label = array(data), array(label)
        # data_shuffled = data[indices]
        # label_shuffled = label[indices]
        k = int(len(data) / self.batch_size)
        for i in range(k + 1):
            start_id = i * self.batch_size
            end_id = min((i + 1) * self.batch_size, len(data))
            yield data[start_id:end_id], label[start_id:end_id]

