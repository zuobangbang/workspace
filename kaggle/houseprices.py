from numpy import *
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import seaborn as sns
# from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelEncoder, Imputer, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from pandas.api.types import is_string_dtype, is_numeric_dtype
from sklearn.ensemble import forest
from sklearn.tree import export_graphviz
from IPython.display import display
import multiprocessing
import fastai
from xgboost.sklearn import XGBRegressor




class hp():
    def __init__(self):
        self.trainpath = '/Users/zuobangbang/Desktop/houseprices/train.csv'
        self.testpath='/Users/zuobangbang/Desktop/houseprices/test.csv'
        self.submit='/Users/zuobangbang/Desktop/houseprices/sample_submission.csv'

    def read_data(self):
        traindata=pd.read_csv(self.trainpath)
        testdata=pd.read_csv(self.testpath)
        testsub=pd.read_csv(self.submit)
        testsub.drop(columns=['Id'], inplace=True)
        # display(traindata.info())
        # display(traindata.describe(include='all'))
        # display(traindata)
        # display(traindata.head())
        # traindata.drop(columns=['PoolQC','MiscFeature','Fence'])
        # print(traindata[traindata['SalePrice'].isnull()])
        return traindata,testdata,testsub


    def feature_engineer(self,traindata):

        #处理缺失值，将缺失值大于0.2的特征去除,小于0.2的部分分别进行填充
        missingdata=(traindata.isnull().sum())/(traindata.isnull().count())
        # dfdata = traindata.drop(missingdata[missingdata > 0.2].index, 1)
        exobj=(traindata.select_dtypes(exclude='object').columns)&(missingdata[(missingdata < 0.2) & (missingdata>0)].index)
        inobj=(traindata.select_dtypes(include='object').columns) & (
            missingdata[(missingdata < 0.2) & (missingdata > 0)].index)
        # print(dfdata.select_dtypes(exclude='object').columns)
        # print(missingdata[(missingdata < 0.2) & (missingdata>0)].index)
        traindata[exobj]=traindata[exobj].fillna(traindata[exobj].median())
        traindata[inobj]=traindata[inobj].fillna(method='ffill')
        # print(traindata[missingdata[(missingdata < 0.2) & (missingdata > 0)].index][:100])
        traindata = traindata.drop(missingdata[missingdata > 0.2].index, 1)
        # print(traindata[missingdata[(missingdata < 0.2) & (missingdata>0)].index].fillna(traindata[missingdata[(missingdata < 0.2) & (missingdata>0)].index].median())[:100])
        # display(traindata.isnull().sum())
        # display(traindata.isnull().count())
        # print(traindata.shape)#(1460, 76)
        #将object类型数据转化为哑变量
        inobj=traindata.select_dtypes(include='object').columns
        print(inobj)
        for i in inobj:
            traindata[i]=traindata[i].astype('category').cat.codes
        # traindata['LotFrontage'].fillna(traindata['LotFrontage'].median)
        #检查特征间的相关性
        num = traindata.select_dtypes(exclude='object')
        numcorr = num.corr()
        f,ax=plt.subplots(figsize=(15,10))
        # heatmap=sns.heatmap(numcorr)
        # sns.heatmap(numcorr.sort_values(by=['SalePrice'], ascending=False).head(1), annot=True, fmt=".2f")
        # plt.title(" Numerical features correlation with the sale price", weight='bold', fontsize=18)
        # plt.show()
        # print(numcorr['SalePrice'].sort_values(ascending=False).head(20).to_frame())
        #特征组合
        traindata['TotalArea'] = traindata['TotalBsmtSF'] + traindata['1stFlrSF'] +traindata['2ndFlrSF'] + traindata['GrLivArea'] + traindata['GarageArea']
        traindata['Bathrooms'] = traindata['FullBath'] + traindata['HalfBath'] * 0.5
        traindata['Year average'] = (traindata['YearRemodAdd'] + traindata['YearBuilt']) / 2
        display(traindata[['TotalArea','Bathrooms','Year average','YearRemodAdd','YearBuilt','FullBath','HalfBath']])
        #除去离群点
        exobj = (traindata.select_dtypes(exclude='object').columns)
        traindata = traindata[(traindata['GrLivArea'] < 4600) & (traindata['MasVnrArea'] < 1500)]

        # plt.figure(figsize=(15,10))
        # plt.subplot2grid((2,2),(0,0))
        # sns.distplot(traindata['OverallQual'],bins=10)
        # plt.subplot2grid((2, 2), (0, 1))
        # sns.distplot(traindata['GrLivArea'],bins=10)
        # plt.subplot2grid((2, 2), (1, 0))
        # sns.distplot(traindata['GarageCars'],bins=10)
        # plt.subplot2grid((2, 2), (1, 1))
        # sns.distplot(traindata['GarageArea'],bins=10)
        # plt.show()
        # print(exobj)
        # display(traindata.columns)
        # plt.figure(figsize=(10,15))
        # sns.distplot(traindata['LotFrontage'],bins=10)
        # plt.show()
        # display(traindata.info())
        #
        #
        #
        #
        #
        # display(traindata['HouseStyle_code'].unique())
        # traindata['MSZoning_code']=traindata['MSZoning'].astype('category').cat.codes
        # display(traindata['MSZoning_code'].unique())
        # display(traindata.describe())

        return traindata


    def splitdata(self,dataset,ty):
        if ty==1:
            # print(dataset.shape)
            nums=int(shape(dataset)[0]*0.8)
            prices=log1p(dataset['SalePrice'])
            y_train,y_test=prices[:nums],prices[nums:]
            dataset.drop(columns=['SalePrice'],inplace=True)
            x_train,x_test=dataset[:nums],dataset[nums:]
            # print(dataset.shape)
            return x_train,x_test,y_train,y_test
        else:
            y_train = log1p(dataset['SalePrice'])
            dataset.drop(columns=['SalePrice'], inplace=True)
            x_train=dataset[:]
            return x_train,y_train







    def main_code(self):
        traindata,testdata,testsub=self.read_data()
        testsub=log1p(testsub['SalePrice'])
        traindata=self.feature_engineer(traindata)
        testdata=self.feature_engineer(testdata)
        ty=2
        xgb = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                           colsample_bynode=1, colsample_bytree=0.5, gamma=0,
                           importance_type='gain', learning_rate=0.01, max_delta_step=0,
                           max_depth=3, min_child_weight=3, missing=None, n_estimators=4000,
                           n_jobs=1, nthread=None, objective='reg:squarederror', random_state=0,
                           reg_alpha=0.0001, reg_lambda=0.01, scale_pos_weight=1, seed=None,
                           silent=None, subsample=1, verbosity=1)
        if ty==1:
            x_train, x_test, y_train, y_test=self.splitdata(traindata,1)
            xgbmod = xgb.fit(x_train, y_train)
            xg_pred = xgbmod.predict(x_test)
            print(mean_squared_error(xg_pred, y_test))
        else:
            x_train, y_train=self.splitdata(traindata,2)
            xgbmod = xgb.fit(x_train, y_train)
            xg_pred = xgbmod.predict(testdata)
            print(mean_squared_error(testsub,xg_pred))
        #
        # prection=clip(a=m1.predict(testdata,a_min=0.0,a_max=1.0))
        # pre_df=pd.DataFrame({'Id':test['Id'],'winPlacePerc':prection})
        # pre_df.to_csv('submission.csv')





if __name__=='__main__':
    q=hp()
    q.main_code()




