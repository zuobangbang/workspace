from numpy import *
import xlrd
import csv

class xt():
    def __init__(self):
        self.path='/Users/zuobangbang/Desktop/data.xlsx'
        self.rec_num=5
        self.file='/Users/zuobangbang/Desktop/result.csv'
        self.nfile = '/Users/zuobangbang/Desktop/result111.csv'

    def read_data(self):
        file=xlrd.open_workbook(self.path)
        sheet=file.sheet_by_name('Sheet1')
        rownums=sheet.nrows
        colnums=sheet.ncols
        shop=[]
        data=zeros((rownums-1,colnums-1))
        for i in range(1,rownums):
            shop.append(int(sheet.cell_value(i,0)))
        name=[]
        for i in range(1,colnums):
            name.append(int(sheet.cell_value(0,i)))
        for i in range(1,rownums):
            for j in range(1,colnums):
                if sheet.cell_value(i,j)!='':
                    data[i-1][j-1]=sheet.cell_value(i,j)
        return mat(data),shop,name,sheet

    def cos(self,va,vb):
        return (va.T*vb)/(sqrt(va.T*va)*sqrt(vb.T*vb))

    def user_similarity(self,data_matrix):
        nums=shape(data_matrix)[1]
        similarity=zeros((nums,nums))
        for i in range(nums):
            for j in range(i,nums):
                if i==j:
                    similarity[i][j]=1
                else:
                    similarity[i][j]=self.cos(data_matrix[:,i],data_matrix[:,j])
                    similarity[j][i]=similarity[i][j]
        return mat(similarity)



    def main_code(self):
        data,shop,name,sheet=self.read_data()
        shopnums,nums = shape(data)[0],shape(data)[1]
        similarity=self.user_similarity(data)
        file=open(self.file,'w')
        x=csv.writer(file)
        x.writerow(['用户id','推荐餐厅'])
        nfile = open(self.nfile, 'w')
        y = csv.writer(nfile)
        c=['用户id']
        c.extend(shop)
        y.writerow(c)
        mea=0.0
        r=1
        for i in range(nums):
            dic={}
            ff={}
            for j in range(shopnums):
                m=(similarity[i]*data[j,:].T)/(sum(array(similarity)[i,array(data)[j,:]!=0]))
                n=m[0,0]
                if data[j,i]==0:
                    dic[shop[j]]=n
                    ff[shop[j]] = n
                else:
                    ff[shop[j]]=data[j,i]
                    mea+=(n-data[j,i])*(n-data[j,i])
                    r+=1
            q=sorted(dic.items(),key=lambda x:x[1],reverse=True)
            w=[]
            w.append(name[i])
            res=[]
            res.append(name[i])
            for k in shop:
                res.append(ff[k])
            y.writerow(res)
            # print('为用户%s推荐%d餐厅：' % (name[i], self.rec_num))
            if len(q):
                for z in q[:self.rec_num]:
                    w.append(z[0])
            else:
                pass
            x.writerow(w)
        print(mea/r)
        file.close()
        nfile.close()





if __name__=='__main__':
    q=xt()
    q.main_code()