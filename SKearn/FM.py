from numpy import *
from random import normalvariate

class fm():
    def __init__(self):
        self.max_iter=3000
        self.alpha=0.01
        self.k=3
        self.error=0.0000001

    def initialize(self,n,k):
        v=zeros((n,k))
        for i in range(n):
            for j in range(k):
                v[i,j]=normalvariate(0,0.2)
        return v

    def sigmod(self,nu):
        return 1.0/(exp(-nu)+1)

    def get_error(self,a,b):
        allerror=0
        for i in range(len(a)):
            allerror=allerror-log(self.sigmod(a[i]*b[i]))
        return allerror


    def predict(self,dataset,w0,w,v):
        result=[]
        m=shape(dataset)[0]
        for i in range(m):
            inter_1 = dataset[i] * v
            inter_2 = multiply(dataset[i], dataset[i]) * multiply(v, v)
            interaction = sum(multiply(inter_1, inter_1) - inter_2) / 2
            y = w0 + dataset[x] * w + interaction
            result.append(self.sigmod(y[0,0]))
        return result

    def accuracy(self,a,b):
        n=0
        for i in range(len(a)):
            if a[i]>=0.5:
                if b[i]==1:
                    n+=1
            else:
                if b[i]==0:
                    n+=1
        return n/len(a)

    # def get_data(self):
    #     return dataset, datalabel
    def sga(self,dataset,datalabel):
        m,n=shape(dataset)
        #w为特征向量
        w=zeros((n,1))
        w0=0
        v=self.initialize(n, self.k)
        error=inf
        iter=0
        while(error>=self.error or iter>=self.max_iter):
            for x in range(m):
                inter_1=dataset[x]*v
                inter_2=multiply(dataset[x],dataset[x])*multiply(v,v)
                interaction=sum(multiply(inter_1,inter_1)-inter_2)/2
                y=w0+dataset[x]*w+interaction
                lo=self.sigmod(datalabel[x]*y[0,0])-1
                w0=w0-self.alpha*lo*datalabel[x]
                for i in range(n):
                    if dataset[x,i]!=0:
                        w[i,0]= w[i,0]-self.alpha*lo*datalabel[x]*dataset[x,i]
                        for f in range(self.k):
                            v[i,f]=v[i,f]-self.alpha*lo*datalabel[x]*(dataset[x,i]*dataset[x]*v[:,f]-v[i,f]*dataset[x,i]*dataset[x,i])
            ally=self.predict(dataset,w0,w,v)
            nowerror=self.get_error(ally,datalabel)
            if nowerror<error:
                error=nowerror
        return w0,w,v


    def fm_modef(self):
        dataset,datalabel=self.get_data()
        w0, w, v=self.sga(dataset,datalabel)
        ally = self.predict(dataset, w0, w, v)
        acc = self.accuracy(ally, datalabel)
        print('the accuracy of the fm model is %f',acc)





