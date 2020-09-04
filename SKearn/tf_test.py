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
        self.file = '/Users/zuobangbang/Desktop/result5.csv'
        self.learn=0.005

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
        left,right=self.genera(row,col)
        errormatrix = zeros((row, col))
        file = open(self.file, 'w')
        x = csv.writer(file)
        x.writerow(['用户id', '推荐餐厅'])
        e=0
        minerror=inf
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
            w=sum(errormatrix*errormatrix)/e
            if abs(minerror-w)<0.0001:
                self.learn=0.05
            minerror=w
            print(w,sum(errormatrix),sum(err))
            if abs(w) < self.error:
                print(sum(errormatrix),err)
                break
        for i in range(row):
            dic={}
            for j in range(col):
                if data[i,j]!=0:
                    dic[shop[j]]=left[i,:]*right[:,j]
            q=sorted(dic.items(),key=lambda x:x[1],reverse=True)
            w = []
            w.append(name[i])
            # print('为用户%s推荐%d餐厅：' % (name[i], self.rec_num))
            for z in range(self.rec_num):
                # print(q[z][0])
                w.append(q[z][0])
            x.writerow(w)
        file.close()






if __name__=='__main__':
    q=svvdd()
    q.main_code()