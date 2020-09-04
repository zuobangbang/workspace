from sklearn.ensemble import RandomForestClassifier
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import csv
import numpy as np
class rf():
    def __init__(self):
        self.path='/Users/zuobangbang/Desktop/样本特征.csv'
        self.dic={
            '居住用地':0,
            '商服用地':1,
            '企业用地':2,
            '科教文化用地':3,
            '绿地与广场用地':4,
            '公服用地':5
        }
        self.per=0.9


    def load_data(self):
        csv_file = csv.reader(open('/Users/zuobangbang/Desktop/样本特征.csv', 'r',encoding='gbk'))
        data=[]
        label=[]
        self.feature=next(csv_file)[5:]
        for i in csv_file:
            try:
                h=[float(j) for j in i[5:]]
                label.append(float(i[4]))
                data.append(h)
            except Exception:
                pass
        l=len(data)
        q=int(l*self.per)
        data=np.array(data)
        label=np.array(label)
        x=[i for i in range(l)]
        random.shuffle(x)
        traindata=data[x[:q]]
        testdata=data[x[q:]]
        trainlabel=label[x[:q]]
        testlabel=label[x[q:]]
        return traindata,trainlabel,testdata,testlabel

    def fea(self,l):
        d={}
        for i in range(len(self.feature)):
            d[self.feature[i]]=l[i]
        return d

    def rf_model(self):
        traindata, trainlabel, testdata, testlabel=self.load_data()
        clf = RandomForestClassifier(n_estimators=660, min_samples_leaf=3, max_features=0.5, n_jobs=-1,oob_score=True)
        clf.fit(traindata, trainlabel)
        # 特征重要性排名
        importances = clf.feature_importances_
        print("各特征重要度分别为：",self.fea(importances))
        # 预测值
        predict_results = clf.predict(testdata)
        print('预测结果为：',predict_results)
        all_result=clf.predict_proba(testdata)
        print('所有结果预测概率分别为：',all_result)
        print(accuracy_score(predict_results, testlabel))






# if __name__=='__main__':
#     e=rf()
#     e.rf_model()




# n=int(5)
# # q=[]
# # for i in range(2):
# #     l=list(map(str,input().split('')))
# #     q.append(l)
# # u=list(map(str,input().split('')))
# # d=list(map(str,input().split('')))
# u='..x.x'
# d='xx...'
#
# def p(s):
#     if s=='.':
#         return 1
#     else:return 0
# def f(x,y,a):
#     if len(x)>2:
#         if a==1:
#             return f(x[:-1],y[:-1],1)*p(x[-1])+f(x[:-1],y[:-1],2)*p(x[-1])
#         elif a==2:
#             return f(x[:-1],y[:-1],1)*p(y[-1])+f(x[:-1],y[:-1],2)*p(y[-1])
#     else:
#         if a==1:
#             return p(x[1])*(p(x[0])+p(y[0]))
#         else:
#             return p(y[1]) * (p(x[0]) + p(y[0]))
#
#
#
# r=f(u,d,2)
# print(r)


n=9
k=1
l=1
r=3

def number(s,n):

    if n>1:
        q = 0
        if r*n>=s>=l*n:
            for i in range(l,r+1):
                try:
                    q+=number(s-i,n-1)
                except Exception:
                    pass
        return q
    else:
        if l<=s<=r:
            return 1


p=[]
a=l*n
b=r*n
h=a
x=h/k
j=0
for i in range(int(a/k)+1,int(b/k)+1):
    j+=number(i*k,n)

print(j)
