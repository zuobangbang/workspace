import pandas as pd
import xlrd
import xlwt

class cs():
    def __init__(self):
        self.ceshi_path='/Users/zuobangbang/Desktop/ceshi.xlsx'
        self.matrix_path='/Users/zuobangbang/Desktop/dili_matrix.csv'


    def ceshi(self):
        worksheet=xlrd.open_workbook(self.ceshi_path)
        data=worksheet.sheet_by_name('Sheet1')
        shop=[]
        user=[]
        for i in range(1,data.nrows):
            shop.append(int(data.cell_value(i,0)))
        for i in range(1,data.ncols):
            user.append(int(data.cell_value(0,i)))
        mse=0.0
        n=0
        matrix=pd.read_csv(self.matrix_path)
        for i in range(1,data.ncols):
            for j in range(1,data.nrows):
                if not data.cell_value(j,i)=='':
                    n+=1
                    q=list(matrix[i - 1:i][str(shop[j])])[0]-float(data.cell_value(j,i))
                    if q*q >5:
                        print(q*q)
                        pass

                    mse+=q*q
        print(mse/n)

if __name__=='__main__':
    f=cs()
    f.ceshi()