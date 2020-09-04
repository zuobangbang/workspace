from params import poetry_params as pp
import collections
import numpy as np
# class LoadData():
class LData(object):
    def __init__(self):
        self.path=pp.file_path

    def read_data(self):
        poetrys = []
        file=open(pp.file_path,'r')
        # poetrys=[i.strip().split(':')[1] for i in file if 5<=len(i.strip().split(':')[1])<=95]
        for i in file:
            content=i.strip().split(':')[1].replace(' ','')
            if len(content) < 12 or len(content) > 79:
                continue
            poetrys.append('['+content+']')
        return poetrys

    def poetry_vec(self):
        poetrys=self.read_data()
        poetrys = sorted(poetrys,key=lambda line: len(line))
        print('The total number of the tang dynasty: ', len(poetrys))
        all_word=[]
        for poetry in poetrys:
            all_word+=[word for word in poetry]
        word_dict={}
        for i in all_word:
            if not word_dict.__contains__(i):
                word_dict[i]=1
            else:word_dict[i]+=1
        counter=sorted(word_dict.items() ,key=lambda x:x[1] ,reverse=True)[:pp.word_size]
        words, _ = zip(*counter)
        words=words[:pp.word_size] + (' ',)
        word_id=dict(zip(words,range(len(words))))
        word_vec=[]
        for poetry in poetrys:
            po=[word_id[word] for word in poetry if word in word_id]
            word_vec.append(po)
        return word_vec,word_id

    def get_batch(self):
        poe_vec,word_id=self.poetry_vec()
        k=int(len(poe_vec)/pp.batch_size)
        for i in range(k):
            start_id = i * pp.batch_size
            end_id = min((i + 1) * pp.batch_size, len(poe_vec))
            q=poe_vec[start_id:end_id]
            big=max(map(len,q))
            data=[]
            xdata = np.full((end_id-start_id, big), word_id[' '], np.int32)
            for j in range(len(q)):
                xdata[j,:len(q[j])]=q[j]
            ydata=np.copy(xdata)
            ydata[:,:-1]=xdata[:,1:]
            yield xdata,ydata




q=LData()
next(q.get_batch())
