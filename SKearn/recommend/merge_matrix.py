import csv
import pandas as pd

matrix_one='/Users/zuobangbang/Desktop/result11.csv'
matrix_two='/Users/zuobangbang/Desktop/result111.csv'
path='/Users/zuobangbang/Desktop/finall.csv'
a=0.5
df1=pd.read_csv(matrix_one)
df2=pd.read_csv(matrix_two)
for index in df1.columns[1:]:
    df1[index]=df1[index]*a+df2[index]*(1-a)
df1.to_csv(path)