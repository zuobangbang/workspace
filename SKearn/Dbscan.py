
from sklearn.datasets.samples_generator import make_circles

import matplotlib.pyplot as plt
from numpy import *
from sklearn.cluster import DBSCAN
import time

class dbscan():
    def __init__(self):
        self.min_sample=5
        self.eps=0.1

    def get_data(self):
        x, y_true = make_circles(n_samples=1000, noise=0.02)  # 这是一个圆环形状的
        plt.scatter(x[:, 0], x[:, 1], c=y_true)
        plt.show()
        return x,y_true

    def dis(self,va,vb):
        s=(va-vb)
        f=sqrt(s*s.T)
        return f[0,0]

    def get_distance(self,dataset):
        m,n=shape(dataset)[0],shape(dataset)[1]
        dataset=mat(dataset)
        dis=mat(zeros((m,m)))
        for i in range(m):
            for j in range(i,m):
                dis[i,j]=self.dis(dataset[i,],dataset[j,])
                dis[j,i]=dis[i,j]
        return dis

    def find_core_point(self,dismatrix):
        core_point=[]
        core_point_dict={}
        m=shape(dismatrix)[0]
        for i in range(m):
            ind=[]
            for j in range(m):
                if dismatrix[i,j]<self.eps:
                    ind.append(j)
            if len(ind)>=self.min_sample:
                core_point.append(i)
                core_point_dict[str(i)]=ind
        core_point_core={}
        for key,value in core_point_dict.items():
            o=[]
            for i in value:
                if i in core_point:
                    o.append(i)
            core_point_core[key]=o
        return core_point,core_point_dict,core_point_core

    def get_same(self,a,b):
        for i in a:
            if i in b:
                return True
        return False


    def join_core_point(self,core_point,core_point_dict,core_point_core):
        labels=array(zeros((1,len(core_point))))
        num=1
        result={}
        result[str(num)]=core_point_core[str(core_point[0])]
        for i in range(1,len(core_point)):
            q=[]
            for key,value in result.items():
                r=self.get_same(core_point_core[str(core_point[i])],value)
                if r:
                    q.append(key)
            if q:
                n=result[q[0]].copy()
                n.extend(core_point_core[str(core_point[i])])
                for i in range(1,len(q)):
                    n.extend(result[q[i]])
                    del result[q[i]]
                result[q[0]]=list(set(n))
            else:
                num=num+1
                result[str(num)]=core_point_core[str(core_point[i])]
        return result

            # a=0
            # if len(labels[0, labels[0, :] == 0]) == 0:
            #     break
            # labels[0,i]=num
            # for j in core_point_core[str(core_point[i])]:
            #     r=core_point.index(j)
            #     if labels[0,r]==0:
            #         labels[0,r]=num
            #     else:
            #         labels[0,labels[0, :] == num]=labels[0,r]
            #         a=1






    def ddbscan(self,data, label):
        # data,label=self.get_data()
        s=time.time()
        m=shape(data)[0]
        dismatrix=self.get_distance(data)
        # types中，1为核心点，0为边界点，-1为噪音点
        types=array(zeros((1,m)))
        number=1
        core_point, core_point_dict,core_point_core=self.find_core_point(dismatrix)
        # print(core_point)
        # print(len(core_point))
        # print(core_point_dict)
        # print(core_point_core)
        if len(core_point):
            core_result=self.join_core_point(core_point,core_point_dict,core_point_core)
            for key,value in core_result.items():
                k=int(key)
                for i in value:
                    types[0,i]=k
                    for j in core_point_dict[str(i)]:
                        types[0, j] = k
            print(types)
        newlabel=types.tolist()[0]
        data=array(data)
        q=list(set(newlabel))
        print(q)
        colors = ['r', 'b', 'g', 'y', 'c', 'm', 'orange']
        for ii in q:
            i=int(ii)
            xy=data[types[0,:]==i,:]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=colors[q.index(ii)], markeredgecolor='w', markersize=5)
            # plt.scatter(data[types[0,:]==i,0],data[types[0,:]==i,1])
            # plt.plot(data[types[0,:]==i,0],data[types[0,:]==i,1], 'o', markerfacecolor=i, markeredgecolor='w', markersize=10)
        plt.title('DBSCAN' )
        print(time.time()-s)
        plt.show()


    def skdbscan(self,data,label):
        # data, label = self.get_data()
        s=time.time()
        data = array(data)
        db = DBSCAN(eps=self.eps, min_samples=self.min_sample, metric='euclidean').fit(data)
        core_samples_mask = zeros_like(db.labels_, dtype=bool)
        print(db.labels_)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        print(db.labels_)
        unique_labels = set(labels)
        colors = ['r', 'b', 'g', 'y', 'c', 'm', 'orange']
        for k, col in zip(unique_labels, colors):
            if k == -1:
                col = 'k'
            class_member_mask = (labels == k)
            xy = data[class_member_mask & core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markeredgecolor='w', markersize=10)

            # xy = data[class_member_mask & ~core_samples_mask]
            # plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markeredgecolor='w', markersize=3)
        print(time.time() - s)
        plt.title('Estimated number of clusters: %d' % n_clusters_)
        plt.show()




if __name__=='__main__':
    dddbscan=dbscan()
    data, label = dddbscan.get_data()
    # dddbscan.ddbscan(data, label)
    dddbscan.skdbscan(data, label)