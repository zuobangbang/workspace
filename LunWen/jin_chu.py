#-*- coding:utf-8 -*-
import requests
import xlwt
import xlrd
import re
import time
import csv
import pandas as pd

# for i in range(8,22):
#     csv_file = open('lujing3fin.txt', 'r')
#     n=0
#     file=open('/Users/zuobangbang/PycharmProjects/untitled/LunWen/lat_lon_{}.csv'.format(str(i)),'w')
#     nfile=csv.writer(file)
#     for line in csv_file.readlines():
#         r = line.replace('\n', '').replace('[', '').replace(']', '').replace('\'', '').split(',')
#         for j in range(1, len(r), 4):
#             a,b=i,i+24
#             if r[j+2]==' {}'.format(str(a)) or r[j+2]==' {}'.format(str(b)):
#                 d = []
#                 d.append(r[0])
#                 d.append(r[j])
#                 d.append(r[j+1])
#                 nfile.writerow(d)
#                 n+=1
#     print(i,n)
#     file.close()

import csv
import math



def out_of_china(lng, lat):
    """
    判断是否在国内，不在国内不做偏移
    :param lng:
    :param lat:
    :return:
    """
    return not (lng > 73.66 and lng < 135.05 and lat > 3.86 and lat < 53.55)
def _transformlng(lng, lat):
    pi = 3.1415926535897932384626  # π
    ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + \
          0.1 * lng * lat + 0.1 * math.sqrt(math.fabs(lng))
    ret += (20.0 * math.sin(6.0 * lng * pi) + 20.0 *
            math.sin(2.0 * lng * pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lng * pi) + 40.0 *
            math.sin(lng / 3.0 * pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(lng / 12.0 * pi) + 300.0 *
            math.sin(lng / 30.0 * pi)) * 2.0 / 3.0
    return ret
def _transformlat(lng, lat):
    x_pi = 3.14159265358979324 * 3000.0 / 180.0
    pi = 3.1415926535897932384626  # π
    a = 6378245.0  # 长半轴
    ee = 0.00669342162296594323  # 扁率
    ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + \
          0.1 * lng * lat + 0.2 * math.sqrt(math.fabs(lng))
    ret += (20.0 * math.sin(6.0 * lng * pi) + 20.0 *
            math.sin(2.0 * lng * pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lat * pi) + 40.0 *
            math.sin(lat / 3.0 * pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(lat / 12.0 * pi) + 320 *
            math.sin(lat * pi / 30.0)) * 2.0 / 3.0
    return ret
def gcj02_to_wgs84(lng, lat):

    """
    GCJ02(火星坐标系)转GPS84
    :param lng:火星坐标系的经度
    :param lat:火星坐标系纬度
    :return:
    """
    x_pi = 3.14159265358979324 * 3000.0 / 180.0
    pi = 3.1415926535897932384626  # π
    a = 6378245.0  # 长半轴
    ee = 0.00669342162296594323  # 扁率
    if out_of_china(lng, lat):
        return lng, lat
    dlat = _transformlat(lng - 105.0, lat - 35.0)
    dlng = _transformlng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * pi
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * pi)
    dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * pi)
    mglat = lat + dlat
    mglng = lng + dlng
    return [lng * 2 - mglng, lat * 2 - mglat]


import xlwt
# for i in range(1,25):
#     wb = xlwt.Workbook()
#     ws = wb.add_sheet('A Test Sheet')
#     ws.write(0, 0,'ID')
#     ws.write(0, 1, 'lat')
#     ws.write(0, 2, 'lon')
#     csv_file = open('lujing3fin.txt', 'r')
#     n = 1
#     for line in csv_file.readlines():
#         r = line.replace('\n', '').replace('[', '').replace(']', '').replace('\'', '').split(',')
#         for j in range(1, len(r), 4):
#             a, b = i, i + 24
#             if r[j + 2] == ' {}'.format(str(a)) or r[j + 2] == ' {}'.format(str(b)):
#                 ws.write(n, 0, r[0])
#                 # lng, lat = gcj02_to_wgs84(118.780266, 32.062935)
#                 # print(lng, lat)
#                 lng, lat = gcj02_to_wgs84(float(r[j]), float(r[j+1]))
#                 ws.write(n, 1, lng)
#                 ws.write(n, 2, lat)
#                 n += 1
#     wb.save('/Users/zuobangbang/PycharmProjects/untitled/LunWen/lat_lon_{}.xls'.format(str(i)))

n=0
wb = xlwt.Workbook()
ws = wb.add_sheet('A Test Sheet')
ws.write(0, 0, 'lat')
ws.write(0, 1, 'lon')

h=n
n = 1
csv_file = open('lujing3fin.txt', 'r')
jin={15: [], 16: [], 17: [], 18: [], 19: [], 20: [], 21: [], 22: [], 23: [], 24: [], 25: [], 26: [], 27: [], 28: [], 29: [], 30: [], 31: [], 32: [], 33: [], 34: [], 35: [], 36: [], 37: [], 38: [], 39: []}

chu={15: [], 16: [], 17: [], 18: [], 19: [], 20: [], 21: [], 22: [], 23: [], 24: [], 25: [], 26: [], 27: [], 28: [], 29: [], 30: [], 31: [], 32: [], 33: [], 34: [], 35: [], 36: [], 37: [], 38: [], 39: []}

for line in csv_file.readlines():
    r = line.replace('\n', '').replace('[', '').replace(']', '').replace('\'', '').split(',')
    if len(r) > 6:
        for j in range(5, len(r), 4):
            from_lng, from_lat = gcj02_to_wgs84(float(r[j - 4]), float(r[j - 3]))
            f,t=int(r[j-2]),int(r[j+2])
            # if 14<int(r[j-1])<40:
            #     if not chu[f].__contains__(str(from_lng) + str(from_lat)):
            #         chu[f][str(from_lng) + str(from_lat)] = 1
            #     else:
            #         chu[f][str(from_lng) + str(from_lat)] = +1
            # if 14<int(r[j+2])<40:
            #     lng, lat = gcj02_to_wgs84(float(r[j]), float(r[j + 1]))
            #     if not jin[t].__contains__(str(lng) + str(lat)):
            #         jin[t][str(lng) + str(lat)] = 1
            #     else:
            #         jin[t][str(lng) + str(lat)] = +1
            lng, lat = gcj02_to_wgs84(float(r[j]), float(r[j + 1]))
            if 14 < f < 40:
                chu[f].append([r[0],from_lng,from_lat])
            if 14 < t < 40:
                jin[t].append([r[0],lng,lat])

for key,value in jin.items():
    file=open('lat_lon_jin_{}.csv'.format(str(key)),'w')
    nfile=csv.writer(file)
    nfile.writerow(['ID',"lat","lon"])
    # wb = xlwt.Workbook()
    # ws = wb.add_sheet('A Test Sheet')
    # ws.write(0, 1, 'lat')
    # ws.write(0, 2, 'lon')
    # ws.write(0,0, 'ID')
    n=1
    if len(value):
        for kk in value:
            d=[kk[0],kk[2],kk[1]]
            # ws.write(n, 1, kk[2])
            # ws.write(n, 2, kk[1])
            # ws.write(n, 0, kk[0])
            nfile.writerow(d)
            n+=1
    file.close()
    # wb.save('/Users/zuobangbang/PycharmProjects/untitled/LunWen/lat_lon_jin_{}.xls'.format(str(key)))

for key, value in chu.items():
    file = open('lat_lon_chu_{}.csv'.format(str(key)),'w')
    nfile = csv.writer(file)
    nfile.writerow(['ID', "lat", "lon"])
    # wb = xlwt.Workbook()
    # ws = wb.add_sheet('A Test Sheet')
    # ws.write(0, 1, 'lat')
    # ws.write(0, 2, 'lon')
    # ws.write(0,0, 'ID')
    n = 1
    if len(value):
        for kk in value:
            d = [kk[0], kk[2], kk[1]]
            # ws.write(n, 1, kk[2])
            # ws.write(n, 2, kk[1])
            # ws.write(n, 0, kk[0])
            nfile.writerow(d)
            n += 1
    file.close()
    # wb.save('/Users/zuobangbang/PycharmProjects/untitled/LunWen/lat_lon_chu_{}.xls'.format(str(key)))
    # ws.write(n, 0, str(lng))
                # ws.write(n, 1, str(lat))
                # n += 1
# print(jin)
# print(chu)




