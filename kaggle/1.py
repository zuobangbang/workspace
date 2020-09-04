import pandas as pd
import csv
import matplotlib.pyplot as plt
import math
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
from sklearn import preprocessing
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.callbacks import EarlyStopping
# import lightgbm as lgb
# import gc, sys
# from sklearn.metrics import mean_squared_error
# gc.enable()
# from sklearn.ensemble import RandomForestRegressor
# dtypes = {
#     'assists': 'uint8',
#     'boosts': 'uint8',
#     'damageDealt': 'float16',
#     'DBNOs': 'uint8',
#     'headshotKills': 'uint8',
#     'heals': 'uint8',
#     'killPlace': 'uint8',
#     'killPoints': 'uint8',
#     'kills': 'uint8',
#     'killStreaks': 'uint8',
#     'longestKill': 'float16',
#     'maxPlace': 'uint8',
#     'numGroups': 'uint8',
#     'revives': 'uint8',
#     'rideDistance': 'float16',
#     'roadKills': 'uint8',
#     'swimDistance': 'float16',
#     'teamKills': 'uint8',
#     'vehicleDestroys': 'uint8',
#     'walkDistance': 'float16',
#     'weaponsAcquired': 'uint8',
#     'winPoints': 'uint8',
#     'winPlacePerc': 'float16'
# }
#
#
# def feature_engineering(filename, train=False):
#     data = pd.read_csv(filename, dtype=dtypes)
#     data = data[data['maxPlace'] > 1]
#     data['headshotrate'] = data['kills'] / data['headshotKills']
#     data['killStreakrate'] = data['killStreaks'] / data['kills']
#     data['healthitems'] = data['heals'] + data['boosts']
#     data['totalDistance'] = data['rideDistance'] + data["walkDistance"] + data["swimDistance"]
#     data['killPlace_over_maxPlace'] = data['killPlace'] / data['maxPlace']
#     data['headshotKills_over_kills'] = data['headshotKills'] / data['kills']
#     data['distance_over_weapons'] = data['totalDistance'] / data['weaponsAcquired']
#     data['walkDistance_over_heals'] = data['walkDistance'] / data['heals']
#     data['walkDistance_over_kills'] = data['walkDistance'] / data['kills']
#     data['killsPerWalkDistance'] = data['kills'] / data['walkDistance']
#     data["skill"] = data["headshotKills"] + data["roadKills"]
#     data[data == np.Inf] = np.NaN
#     data[data == np.NINF] = np.NaN
#
#     data.fillna(0, inplace=True)
#     feature = list(data.columns)
#     feature.remove('Id')
#     feature.remove('groupId')
#     feature.remove('matchId')
#     feature.remove('matchType')
#     if (train):
#         labels = np.array(data.groupby(['matchId', 'groupId'])['winPlacePerc'].agg('mean'), dtype=np.float64)
#         feature.remove('winPlacePerc')
#     else:
#         labels = data[['Id']]
#
#     print("group_max")
#     agg = data.groupby(['matchId', 'groupId'])[feature].agg('max')
#     agg_rank = agg.groupby('matchId')[feature].rank(pct=True).reset_index()
#     if train:
#         data_out = agg.reset_index()[['matchId', 'groupId']]
#     else:
#         data_out = data[['matchId', 'groupId']]
#     data_out = data_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
#     data_out = data_out.merge(agg_rank, suffixes=["_max", "_max_rank"], how='left', on=['matchId', 'groupId'])
#
#     print("group_mean")
#     agg = data.groupby(['matchId', 'groupId'])[feature].agg('mean')
#     agg_rank = agg.groupby('matchId')[feature].rank(pct=True).reset_index()
#     data_out = data_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
#     data_out = data_out.merge(agg_rank, suffixes=["_mean", "_mean_rank"], how='left', on=['matchId', 'groupId'])
#
#     print("group_min")
#     agg = data.groupby(['matchId', 'groupId'])[feature].agg('min')
#     agg_rank = agg.groupby('matchId')[feature].rank(pct=True).reset_index()
#     data_out = data_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
#     data_out = data_out.merge(agg_rank, suffixes=["_min", "_min_rank"], how='left', on=['matchId', 'groupId'])
#
#     print("match_mean")
#     agg = data.groupby(['matchId'])[feature].agg('mean').reset_index()
#     data_out = data_out.merge(agg, suffixes=["", "_match_mean"], how='left', on=['matchId'])
#
#     print("match_max")
#     agg = data.groupby(['matchId'])[feature].agg('max').reset_index()
#     data_out = data_out.merge(agg, suffixes=["", "_match_max"], how='left', on=['matchId'])
#
#     print("match_size")
#     agg = data.groupby(['matchId']).size().reset_index(name='match_size')
#     data_out = data_out.merge(agg, how='left', on=['matchId'])
#
#     del data, agg, agg_rank
#     gc.collect()
#     data_out.drop(["matchId", "groupId"], axis=1, inplace=True)
#
#     data_out = reduce_size(data_out)
#     X = data_out
#     del data_out, feature
#     gc.collect()
#     return X, labels
#
#
# def reduce_size(merged_data_out):
#     print('      Starting size is %d Mb' % (sys.getsizeof(merged_data_out) / 1024 / 1024))
#     print('      Columns: %d' % (merged_data_out.shape[1]))
#     feats = merged_data_out.columns[merged_data_out.dtypes == 'float64']
#     for feat in feats:
#         merged_data_out[feat] = merged_data_out[feat].astype('float32')
#
#     feats = merged_data_out.columns[merged_data_out.dtypes == 'int16']
#     for feat in feats:
#         mm = np.abs(merged_data_out[feat]).max()
#         if mm < 126:
#             merged_data_out[feat] = merged_data_out[feat].astype('int8')
#
#     feats = merged_data_out.columns[merged_data_out.dtypes == 'int32']
#     for feat in feats:
#         mm = np.abs(merged_data_out[feat]).max()
#         if mm < 126:
#             merged_data_out[feat] = merged_data_out[feat].astype('int8')
#         elif mm < 30000:
#             merged_data_out[feat] = merged_data_out[feat].astype('int16')
#
#     feats = merged_data_out.columns[merged_data_out.dtypes == 'int64']
#     for feat in feats:
#         mm = np.abs(merged_data_out[feat]).max()
#         if mm < 126:
#             merged_data_out[feat] = merged_data_out[feat].astype('int8')
#         elif mm < 30000:
#             merged_data_out[feat] = merged_data_out[feat].astype('int16')
#         elif mm < 2000000000:
#             merged_data_out[feat] = merged_data_out[feat].astype('int32')
#     print('      Ending size is %d Mb' % (sys.getsizeof(merged_data_out) / 1024 / 1024))
#     return merged_data_out
#
#
# params = {
#     'objective': 'regression',
#     'early_stopping_rounds': 200,
#     'n_estimators': 20000,
#     'metric': 'mae',
#     "bagging_seed": 0,
#     'num_leaves': 31,
#     'learning_rate': 0.05,
#     'bagging_fraction': 0.9,
#     "num_threads": 4,
#     "colsample_bytree": 0.7
# }
#
# if __name__ == '__main__':
#     batch_size = 512
#     num_of_features = 0
#     # features = load_csv_data('../input/train_V2.csv')
#     # test = load_csv_data('../input/test_V2.csv')
#     trainpath ='/Users/zuobangbang/Desktop/pubg/train_V2.csv'
#     testpath = '/Users/zuobangbang/Desktop/pubg/test_V2.csv'
#     features, labels = feature_engineering(trainpath, train=True)
#     num_of_features = features.shape[1]
#
#     filepath = "best.h5"
#     split = int(len(labels) * 0.8)
#     lgb_train = lgb.Dataset(features[:split], labels[:split])
#     lgb_val = lgb.Dataset(features[split:], labels[split:])
#     del features, labels
#     gc.collect()
#     gbm = lgb.train(params, lgb_train, verbose_eval=100, valid_sets=[lgb_train, lgb_val], early_stopping_rounds=200)
#     del lgb_train, lgb_val
#     gc.collect()
#
#     features_test, test = feature_engineering(testpath)
#     predict = gbm.predict(features_test, num_iteration=gbm.best_iteration)
#     del features_test
#     gc.collect()
#     predict = predict.reshape(-1)
#     test['winPlacePerc'] = predict
#
#     df_test = pd.read_csv(testpath)
#
#     # Restore some columns
#     test = test.merge(df_test[["Id", "matchId", "groupId", "maxPlace", "numGroups"]], on="Id", how="left")
#
#     # Sort, rank, and assign adjusted ratio
#     df_sub_group = test.groupby(["matchId", "groupId"]).first().reset_index()
#     df_sub_group["rank"] = df_sub_group.groupby(["matchId"])["winPlacePerc"].rank()
#     df_sub_group["adjusted_perc"] = (df_sub_group["rank"] - 1) / (df_sub_group["numGroups"] - 1)
#
#     test = test.merge(df_sub_group[["adjusted_perc", "matchId", "groupId"]], on=["matchId", "groupId"], how="left")
#     test["winPlacePerc"] = test["adjusted_perc"]
#
#     # Deal with edge cases
#     test.loc[test.maxPlace == 0, "winPlacePerc"] = 0
#     test.loc[test.maxPlace == 1, "winPlacePerc"] = 1
#
#     # Align with maxPlace
#     # Credit: https://www.kaggle.com/anycode/simple-nn-baseline-4
#     subset = test.loc[test.maxPlace > 1]
#     gap = 1.0 / (subset.maxPlace.values - 1)
#     new_perc = np.around(subset.winPlacePerc.values / gap) * gap
#     test.loc[test.maxPlace > 1, "winPlacePerc"] = new_perc
#
#     # Edge case
#     test.loc[(test.maxPlace > 1) & (test.numGroups == 1), "winPlacePerc"] = 0
#     assert test["winPlacePerc"].isnull().sum() == 0
#
#     test[["Id", "winPlacePerc"]].to_csv("submission_adjusted.csv", index=False)

import seaborn as sns

a =pd.read_csv('/Users/zuobangbang/Desktop/houseprices/train.csv')
b=pd.read_csv('/Users/zuobangbang/Desktop/houseprices/test.csv')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
# num=a.select_dtypes(exclude='object')
# numcorr=num.corr()
# f,ax=plt.subplots(figsize=(19,1))
# sns.heatmap(numcorr.sort_values(by=['SalePrice'], ascending=False).head(1),annot=True, fmt = ".2f")
# plt.title(" Numerical features correlation with the sale price", weight='bold', fontsize=18)
# plt.show()
# Num=numcorr['SalePrice'].sort_values(ascending=False).head(20).to_frame()
#
# cm = sns.light_palette("cyan", as_cmap=True)
#
# s = Num.style.background_gradient(cmap=cm)
# s
# plt.style.use('seaborn')
# sns.set_style('whitegrid')
#
# plt.subplots(0,0,figsize=(15,3))


a.isnull().mean().sort_values(ascending=False).plot.bar(color='black')
# plt.axhline(y=0.1, color='r', linestyle='-')
# plt.title('Missing values average per column: Train set', fontsize=20, weight='bold' )
# plt.show()

plt.subplots(1,0,figsize=(15,3))
b.isnull().mean().sort_values(ascending=False).plot.bar(color='black')
# plt.axhline(y=0.1, color='r', linestyle='-')
# plt.title('Missing values average per column: Test set ', fontsize=20, weight='bold' )
# plt.show()
na = a.shape[0]
nb = b.shape[0]
y_train = a['SalePrice'].to_frame()
#Combine train and test sets
c1 = pd.concat((a, b), sort=False).reset_index(drop=True)
#Drop the target "SalePrice" and Id columns
c1.drop(['SalePrice'], axis=1, inplace=True)
c1.drop(['Id'], axis=1, inplace=True)
print("Total size is :",c1.shape)
#axis=1   删除包含缺失值的列
c=c1.dropna(thresh=len(c1)*0.9, axis=1)
print('We dropped ',c1.shape[1]-c.shape[1], ' features in the combined set')
allna = (c.isnull().sum() / len(c))
allna = allna.drop(allna[allna == 0].index).sort_values(ascending=False)
# plt.figure(figsize=(12, 8))
# allna.plot.barh(color='purple')
# plt.title('Missing values average per column', fontsize=25, weight='bold' )
# plt.show()
NA=c[['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond','GarageYrBlt','BsmtFinType2','BsmtFinType1','BsmtCond', 'BsmtQual','BsmtExposure', 'MasVnrArea','MasVnrType','Electrical','MSZoning','BsmtFullBath','BsmtHalfBath','Utilities','Functional','Exterior1st','BsmtUnfSF','Exterior2nd','TotalBsmtSF','GarageArea','GarageCars','KitchenQual','BsmtFinSF2','BsmtFinSF1','SaleType']]
NAcat=NA.select_dtypes(include='object')
NAnum=NA.select_dtypes(exclude='object')
print('We have :',NAcat.shape[1],'categorical features with missing values')
print('We have :',NAnum.shape[1],'numerical features with missing values')
c['MasVnrArea']=c.MasVnrArea.fillna(0)
#GarageYrBlt:  Year garage was built, we fill the gaps with the median: 1980
c['GarageYrBlt']=c["GarageYrBlt"].fillna(1980)
NAcat1= NAcat.isnull().sum().to_frame().sort_values(by=[0]).T
cm = sns.light_palette("lime", as_cmap=True)

NAcat1 = NAcat1.style.background_gradient(cmap=cm)
NAcat1
c['Electrical']=c['Electrical'].fillna(method='ffill')
c['SaleType']=c['SaleType'].fillna(method='ffill')
c['KitchenQual']=c['KitchenQual'].fillna(method='ffill')
c['Exterior1st']=c['Exterior1st'].fillna(method='ffill')
c['Exterior2nd']=c['Exterior2nd'].fillna(method='ffill')
c['Functional']=c['Functional'].fillna(method='ffill')
c['Utilities']=c['Utilities'].fillna(method='ffill')
c['MSZoning']=c['MSZoning'].fillna(method='ffill')
#Categorical missing values
NAcols=c.columns
for col in NAcols:
    if c[col].dtype == "object":
        c[col] = c[col].fillna("None")
for col in NAcols:
    if c[col].dtype != "object":
        c[col]= c[col].fillna(0)
c['TotalArea'] = c['TotalBsmtSF'] + c['1stFlrSF'] + c['2ndFlrSF'] + c['GrLivArea'] +c['GarageArea']
c['Bathrooms'] = c['FullBath'] + c['HalfBath']*0.5
c['Year average']= (c['YearRemodAdd']+c['YearBuilt'])/2

cb=pd.get_dummies(c)
print("the shape of the original dataset",c.shape)
print("the shape of the encoded dataset",cb.shape)
print("We have ",cb.shape[1]- c.shape[1], 'new encoded features')
from scipy.stats import skew

numeric_feats = c.dtypes[c.dtypes != "object"].index

skewed_feats = c[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

c[skewed_feats] = np.log1p(c[skewed_feats])

Train = cb[:na]  #na is the number of rows of the original training set
Test = cb[na:]
# fig = plt.figure(figsize=(15,10))
# ax1 = plt.subplot2grid((2,2),(0,0))
# plt.scatter(x=a['GrLivArea'], y=a['SalePrice'], color=('yellowgreen'))
# plt.axvline(x=4600, color='r', linestyle='-')
# plt.title('Ground living Area- Price scatter plot', fontsize=15, weight='bold' )
#
# ax1 = plt.subplot2grid((2,2),(0,1))
# plt.scatter(x=a['TotalBsmtSF'], y=a['SalePrice'], color=('red'))
# plt.axvline(x=5900, color='r', linestyle='-')
# plt.title('Basement Area - Price scatter plot', fontsize=15, weight='bold' )
#
# ax1 = plt.subplot2grid((2,2),(1,0))
# plt.scatter(x=a['1stFlrSF'], y=a['SalePrice'], color=('deepskyblue'))
# plt.axvline(x=4000, color='r', linestyle='-')
# plt.title('First floor Area - Price scatter plot', fontsize=15, weight='bold' )
#
# ax1 = plt.subplot2grid((2,2),(1,1))
# plt.scatter(x=a['MasVnrArea'], y=a['SalePrice'], color=('gold'))
# plt.axvline(x=1500, color='r', linestyle='-')
# plt.title('Masonry veneer Area - Price scatter plot', fontsize=15, weight='bold' )
train=Train[(Train['GrLivArea'] < 4600) & (Train['MasVnrArea'] < 1500)]

print('We removed ',Train.shape[0]- train.shape[0],'outliers')
target=a[['SalePrice']]
target.loc[1298]
target.loc[523]
pos = [1298,523, 297]
target.drop(target.index[pos], inplace=True)
# plt.style.use('seaborn')
# sns.set_style('whitegrid')
# fig = plt.figure(figsize=(15,5))
# #1 rows 2 cols
# #first row, first col
# ax1 = plt.subplot2grid((1,2),(0,0))
# plt.scatter(x=a['GrLivArea'], y=a['SalePrice'], color=('yellowgreen'))
# plt.title('Area-Price plot with outliers',weight='bold', fontsize=18)
# plt.axvline(x=4600, color='r', linestyle='-')
# #first row sec col
# ax1 = plt.subplot2grid((1,2),(0,1))
# plt.scatter(x=train['GrLivArea'], y=target['SalePrice'], color='navy')
# plt.axvline(x=4600, color='r', linestyle='-')
# plt.title('Area-Price plot without outliers',weight='bold', fontsize=18)
target["SalePrice"] = np.log1p(target["SalePrice"])
# plt.style.use('seaborn')
# sns.set_style('whitegrid')
# fig = plt.figure(figsize=(15,5))
# #1 rows 2 cols
# #first row, first col
# ax1 = plt.subplot2grid((1,2),(0,0))
# plt.hist(a.SalePrice, bins=10, color='mediumpurple')
# plt.title('Sale price distribution before normalization',weight='bold', fontsize=18)
#first row sec col
# ax1 = plt.subplot2grid((1,2),(0,1))
# plt.hist(target.SalePrice, bins=10, color='darkcyan')
# plt.title('Sale price distribution after normalization',weight='bold', fontsize=18)
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
import math
import sklearn.model_selection as ms
import sklearn.metrics as sklm
x=train
y=np.array(target)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size = .33, random_state=0)
from xgboost.sklearn import XGBRegressor

xgb= XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=0.5, gamma=0,
             importance_type='gain', learning_rate=0.01, max_delta_step=0,
             max_depth=3, min_child_weight=3, missing=None, n_estimators=4000,
             n_jobs=1, nthread=None, objective='reg:squarederror', random_state=0,
             reg_alpha=0.0001, reg_lambda=0.01, scale_pos_weight=1, seed=None,
             silent=None, subsample=1, verbosity=1)
xgmod=xgb.fit(x_train,y_train)
xg_pred=xgmod.predict(x_test)
print('Root Mean Square Error test = ' + str(math.sqrt(sklm.mean_squared_error(y_test, xg_pred))))

