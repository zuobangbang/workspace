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


# Based on the "playersJoined" feature we can create (or change) a lot of others to normalize their values.
# For example i will create the "killsNorm" and "damageDealtNorm" features.
# When there are 100 players in the game it might be easier to find and kill someone, than when there are 90 players.
# So i will normalize the kills in a way that a kill in 100 players will score 1 (as it is) and in 90 players it will score (100-90)/100 + 1 = 1.1.
# This is just an assumption. You can use different scales.
# Id - 用户ID
# matchDuration - 比赛时长
# matchType - 比赛类型
# rankPoints - Elo排名
# winPlacePerc - 百分比排名
# groupId - 匹配的队伍ID
# matchId - 该场比赛ID   get
# assists -被队友杀死的敌人玩家数量
# boosts - 使用能量道具数量    get
# damageDealt - 造成的全部伤害    get
# DBNOs - 被击倒的敌方玩家数量
# headshotKills - 爆头数   get
# heals - 治疗次数   get
# killPlace - 本场比赛杀敌排名    get
# killPoints - Elo杀敌排名
# kills - 杀敌数   get
# killStreaks - 连续杀敌数
# longestKill - 玩家和敌人被杀死时的最大距离    get
# maxPlace - 本局最差排名
# numGroups - 游戏中匹配的队伍数量
# revives - 复活队友次数
# rideDistance - 驾驶总距离    get
# roadKills - 驾车杀敌数
# swimDistance - 游泳总距离   get
# teamKills - 杀死队友次数
# vehicleDestroys - 被毁车辆数
# walkDistance - 步行距离    get
# weaponsAcquired - 拾起武器数     get
# winPoints - 胜率Elo排名
# winPlacePerc - 百分比排名


class pubgg():
    def __init__(self):
        self.trainpath = '/Users/zuobangbang/Desktop/pubg/train_V2.csv'
        self.testpath='/Users/zuobangbang/Desktop/pubg/test_V2.csv'
        self.submit='/Users/zuobangbang/Desktop/pubg/sample_submission_V2.csv'

    def read_data(self):
        traindata=pd.read_csv(self.trainpath)
        testdata=pd.read_csv(self.testpath)
        testsub=pd.read_csv(self.submit)
        testsub.drop(columns=['Id'],inplace=True)
        #显示训练样本各个特征空缺值数量
        # print(traindata.isnull().sum())
        # print(testdata.isnull().sum())
        # display(testdata.shape)
        # display(traindata.head())
        # display(traindata.tail())
        # print(traindata.describe())
        # print(traindata.info())
        # print(traindata['winPlacePerc'].isnull())
        print(traindata[traindata['winPlacePerc'].isnull()])
        traindata.drop(2744604,inplace=True)
        print(traindata[traindata['winPlacePerc'].isnull()])

        traindata.groupby('matchId')['matchId'].count()
        return traindata,testdata,testsub


    def feature_engineer(self,traindata):
        traindata['playersJoined'] = traindata.groupby('matchId')['matchId'].transform('count')
        # plt.figure(figsize=(15, 10))
        # sns.countplot(traindata[traindata['playersJoined'] >= 75]['playersJoined'])
        # plt.title('playersJoined')
        # plt.show()
        traindata['headkillnorm']=traindata['headshotKills']/traindata['kills']
        traindata['headkillnorm'].fillna(0,inplace=True)
        traindata['killnorm']=traindata['kills']*((100-traindata['playersJoined'])/100+1)
        traindata['killPlacenorm']=traindata['killPlace']*((100-traindata['playersJoined'])/100+1)
        traindata['maxPlacenorm']=traindata['maxPlace']*((100-traindata['playersJoined'])/100+1)
        traindata['damageDealtnorm']=traindata['damageDealt']*((100-traindata['playersJoined'])/100+1)
        traindata['matchDurationnorm']=traindata['matchDuration']*((100-traindata['playersJoined'])/100+1)
        traindata['boostsandheals']=traindata['boosts']+traindata['heals']
        traindata['totaldistance']=traindata['rideDistance']+traindata['swimDistance']+traindata['walkDistance']
        traindata['killwithoutmoving']=(traindata['kills']>0) & (traindata['totaldistance']==0)
        # display(traindata[traindata['killwithoutmoving']==True][['boostsandheals','boosts','heals','totaldistance','kills']].tail())
        label=[['boostsandheals',60],['kills',30],['totaldistance',20000],['rideDistance',25000],['weaponsAcquired',50],['walkDistance',15000],['swimDistance',2000],['longestKill',800],['rideDistance',20000]]
        # for i in label:
        #     traindata=self.dropdata(traindata,i[0],i[1])
        display(traindata['matchType'].unique())
        traindata['Idcat']=traindata['Id'].astype('category').cat.codes
        traindata['matchIdcat']=traindata['matchId'].astype('category').cat.codes
        traindata['groupIdcat'] = traindata['groupId'].astype('category').cat.codes
        traindata['matchTypecat']=traindata['matchType'].astype('category').cat.codes
        # display(traindata[['matchIdcat','groupIdcat','matchTypecat']][:100])
        return traindata


    def dropdata(self,data,labels,nums):
        print(labels)
        # plt.figure(figsize=(20,15))
        # sns.distplot(data[labels],bins=10)
        # plt.title(labels)
        # plt.show()
        # nums=input()
        display(data[data[labels]>=nums].shape)
        data.drop(data[data[labels]>=nums].index,inplace=True)
        return data

    def get_sample(self,data):
        sample=data.shape[0]
        dfdata=data.sample(sample)
        label = dfdata['winPlacePerc']
        dfdata.drop(columns = ['winPlacePerc','matchId','groupId','Id','matchType'],inplace=True)
        # dfdata.drop(columns=['winPlacePerc'], inplace=True)
        return dfdata,label




    def main_code(self):
        traindata,testdata,testsub=self.read_data()
        traindata=self.feature_engineer(traindata)
        #选取训练样本，测试样本建模
        testdata=self.feature_engineer(testdata)
        x_train,y_train=self.get_sample(traindata)
        testdata.drop(columns = ['matchId','groupId','Id','matchType'],inplace=True)

        m1 = RandomForestRegressor(n_estimators=70, min_samples_leaf=3, max_features=0.5, n_jobs=-1)
        # fi=rf_feat_importance(m1, x_train)
        # plot1 = fi[:20].plot('cols', 'imp', figsize=(14, 6), legend=False, kind='barh')
        # plot1
        # to_keep = fi[fi.imp > 0.005].cols
        # print('Significant features: ', len(to_keep))
        m1.fit(x_train, y_train)
        scoretrain = mean_squared_error(y_train, m1.predict(x_train))
        scoretest = mean_squared_error(testsub, m1.predict(testdata))
        #从训练样本选取部分数据建模
        # data,y=self.get_sample(traindata)
        # display(y)
        # x=int(0.8*shape(data)[0])
        # x_train,x_test=data[:x],data[x:]
        # y_train,y_test=y[:x],y[x:]
        # print('shape of traindata is ',x_train.shape,
        #       'shape of trainlabels is',y_train.shape,
        #       'shape of test data is',x_test.shape)
        # m1=RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features='sqrt',n_jobs=-1)
        # m1.fit(x_train,y_train)
        # scoretrain=mean_squared_error(y_train,m1.predict(x_train))
        # scoretest=mean_squared_error(y_test,m1.predict(x_test))
        print(scoretrain)
        print(scoretest)
        prection=clip(a=m1.predict(testdata,a_min=0.0,a_max=1.0))
        pre_df=pd.DataFrame({'Id':test['Id'],'winPlacePerc':prection})
        pre_df.to_csv('submission.csv')





if __name__=='__main__':
    q=pubgg()
    q.main_code()




