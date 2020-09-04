import numpy
from  numpy import *
import xlrd
import math
import pandas
import csv
class svvdd():
    def __init__(self):
        self.path='/Users/zuobangbang/Desktop/data.xlsx'
        self.rec_num=5
        self.itera_num=5000
        self.error=0.0005
        self.matrix_col=3
        self.file = '/Users/zuobangbang/Desktop/result.csv'
        self.learn=0.0005
        self.nfile='/Users/zuobangbang/Desktop/result11.csv'

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
        return mat(data).T,shop,name

    def genera(self,a,b):
        left=zeros((a,self.matrix_col))
        right=zeros((self.matrix_col,b))
        for i in range(a):
            for j in range(self.matrix_col):
                left[i][j]=random.randn()
        for i in range(b):
            for j in range(self.matrix_col):
                right[j][i]=random.randn()
        return mat(left),mat(right)

    def sse(self,left,right,data):
        a,b=shape(data)[0],shape(data)[1]
        errormatrix=zeros((a,b))
        for i in range(a):
            for j in range(b):
                if data[i,j]!=0:
                    errormatrix[i][j]=data[i,j]-left[i,:]*right[:,j]
        return errormatrix

    # def renewal(self,left,right,errormatrix):




    def main_code(self):
        data,shop,name=self.read_data()
        row, col=shape(data)[0],shape(data)[1]
        #初始设置left,right
        left,right=self.genera(row,col)
        #重新读取csv调用left,right
        # left=numpy.loadtxt(open("left_0.csv","rb"),delimiter=",",skiprows=0)
        # right = numpy.loadtxt(open("right_0.csv", "rb"), delimiter=",", skiprows=0)
        errormatrix = zeros((row, col))
        file = open(self.file, 'w')
        x = csv.writer(file)
        nfile = open(self.nfile, 'w')
        y = csv.writer(nfile)
        x.writerow(['用户id', '推荐餐厅'])
        c = ['用户id']
        c.extend(shop)
        y.writerow(c)
        e=0
        for i in range(row):
            for j in range(col):
                if data[i, j] != 0:
                    e+=1
        for num in range(self.itera_num):
            for i in range(row):
                for j in range(col):
                    if data[i, j] != 0:
                        errormatrix[i][j] = data[i, j] - left[i, :] * right[:, j]
                        for h in range(self.matrix_col):
                            # if math.isnan(self.learn*errormatrix[i][j]*right[h,j]) or math.isnan(self.learn*errormatrix[i][j]*left[i,h]):
                            #     right[h, j] = 1
                            #     print(right[h, j])
                            left[i,h]=left[i,h]+self.learn*errormatrix[i][j]*right[h,j]
                            right[h,j]=right[h,j]+self.learn*errormatrix[i][j]*left[i,h]

            # print(left,right)
            err = self.sse(left, right, data)
            print(sum(errormatrix*errormatrix)/e,sum(errormatrix),sum(err))
            if abs(sum(errormatrix*errormatrix)/e) < self.error:
                print(sum(errormatrix),err)
                break
                #96行的222自己设置，每222次保存一次
            if num%222==0:
                numpy.savetxt('left_{}.csv'.format(str(num)), left, delimiter=',')
                numpy.savetxt('right_{}.csv'.format(str(num)), right, delimiter=',')


        for i in range(row):
            dic={}
            ff={}
            for j in range(col):
                if data[i,j]==0:
                    n=left[i,:]*right[:,j]
                    dic[shop[j]]=n[0,0]
                    ff[shop[j]]=dic[shop[j]]
                else:
                    ff[shop[j]] =data[i,j]
            q=sorted(dic.items(),key=lambda x:x[1],reverse=True)
            w = []
            w.append(name[i])
            # print('为用户%s推荐%d餐厅：' % (name[i], self.rec_num))
            if len(q):
                for z in q[:self.rec_num]:
                # print(q[z][0])
                    w.append(z[0])
            res = []
            res.append(name[i])
            for k in shop:
                res.append(ff[k])
            x.writerow(w)
            y.writerow(res)
        file.close()






if __name__=='__main__':
    q=svvdd()
    q.main_code()