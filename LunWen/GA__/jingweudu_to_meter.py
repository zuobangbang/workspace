#-*- coding:utf-8 -*-
from math import radians, cos, sin, asin, sqrt
import csv

def haversine(lon1, lat1, lon2, lat2):  # 经度1，纬度1，经度2，纬度2 （十进制度数）
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # 地球平均半径，单位为公里
    return c * r * 1000

# w=haversine(lon1, lat1, lon2, lat2)
p=open('stop.txt','r',encoding='utf-8')
point_1=[]
N=[]
# N=[编号，总——出，最大-出，最大-出-进]
for i in p.readlines()[:]:
    w=i.replace('\n','').split(',')
    f=[w[0],w[24],w[25]]
    N.append([w[0],w[43],w[43],w[53]])
    point_1.append(f)


p=open('bike.txt','r',encoding='utf-8')
point_2=[]
for i in p.readlines()[:]:
    w=i.replace('\n','').split(',')
    f=[w[0],w[2],w[3]]
    point_2.append(f)

file=open('meter.csv','w')
nfile=csv.writer(file)
d=[]
for i in point_2:
    d.append(i[0])
nfile.writerow(d)

for i in point_1[1:]:
    d=[i]
    for j in point_2[1:]:
        w = haversine(float(i[2]),float(i[1]), float(j[2]),float(j[1]))
        d.append(w)
    nfile.writerow(d)
print(point_1)
print(point_2)
print(N)

file.close()