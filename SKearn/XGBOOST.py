import xgboost as xgb
import pandas as pd
import time
import numpy as np

# 第一种
# now = time.time()
#
# dataset = pd.read_csv("../input/train.csv")  # 注意自己数据路径
#
# train = dataset.iloc[:, 1:].values
# labels = dataset.iloc[:, :1].values
#
# tests = pd.read_csv("../input/test.csv")  # 注意自己数据路径
# # test_id = range(len(tests))
# test = tests.iloc[:, :].values
#
# params = {
#     'booster': 'gbtree',
#     # 这里手写数字是0-9，是一个多类的问题，因此采用了multisoft多分类器，
#     'objective': 'multi:softmax',
#     'num_class': 10,  # 类数，与 multisoftmax 并用
#     'gamma': 0.05,  # 在树的叶子节点下一个分区的最小损失，越大算法模型越保守 。[0:]
#     'max_depth': 12,  # 构建树的深度 [1:]
#     # 'lambda':450,  # L2 正则项权重
#     'subsample': 0.4,  # 采样训练数据，设置为0.5，随机选择一般的数据实例 (0:1]
#     'colsample_bytree': 0.7,  # 构建树树时的采样比率 (0:1]
#     # 'min_child_weight':12, # 节点的最少特征数
#     'silent': 1,
#     'eta': 0.005,  # 如同学习率
#     'seed': 710,
#     'nthread': 4,  # cpu 线程数,根据自己U的个数适当调整
# }
#
# plst = list(params.items())
#
# # Using 10000 rows for early stopping.
# offset = 35000  # 训练集中数据50000，划分35000用作训练，15000用作验证
#
# num_rounds = 500  # 迭代你次数
# xgtest = xgb.DMatrix(test)
#
# # 划分训练集与验证集
# xgtrain = xgb.DMatrix(train[:offset, :], label=labels[:offset])
# xgval = xgb.DMatrix(train[offset:, :], label=labels[offset:])
#
# # return 训练和验证的错误率
# watchlist = [(xgtrain, 'train'), (xgval, 'val')]
#
# # training model
# # early_stopping_rounds 当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练
# model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=100)
# # model.save_model('./model/xgb.model') # 用于存储训练出的模型
# preds = model.predict(xgtest, ntree_limit=model.best_iteration)
#
# # 将预测结果写入文件，方式有很多，自己顺手能实现即可
# np.savetxt('submission_xgb_MultiSoftmax.csv', np.c_[range(1, len(test) + 1), preds],
#            delimiter=',', header='ImageId,Label', comments='', fmt='%d')

#第二种
import xgboost
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 载入数据集
# dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")
# # split data into X and y
# X = dataset[:, 0:8]
# Y = dataset[:, 8]
#
# # 把数据集拆分成训练集和测试集
# seed = 7
# test_size = 0.33
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
#
# # 拟合XGBoost模型
# model = XGBClassifier()
# model.fit(X_train, y_train)
#
# # 对测试集做预测
# y_pred = model.predict(X_test)
# predictions = [round(value) for value in y_pred]
#
# # 评估预测结果
# accuracy = accuracy_score(y_test, predictions)
# print("Accuracy: %.2f%%" % (accuracy * 100.0))
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

plt.rcParams.update({'figure.autolayout': True})

df = pd.DataFrame({'x': [-2.1, -0.9, 0, 1, 2, 2.5, 3, 4],
                   'y': [-10, 0, -5, 10, 20, 10, 30, 40]})
X_train = df.drop('y', axis=1)
Y_train = df['y']
X_pred = [-4, -3, -2, -1, 0, 0.4, 0.6, 1, 1.4, 1.6, 2, 3, 4, 5, 6, 7, 8]


def process_list(list_in):
    result = map(lambda x: "%8.2f" % round(float(x), 2), list_in)
    return list(result)


def customObj3(real, predict):
    grad = predict - real
    hess = np.power(np.abs(grad), 0.1)
    # print 'predict', process_list(predict.tolist()), type(predict)
    # print ' real  ', process_list(real.tolist()), type(real)
    # print ' grad  ', process_list(grad.tolist()), type(grad)
    # print ' hess  ', process_list(hess.tolist()), type(hess), '\n'
    return grad, hess


def customObj1(real, predict):
    grad = predict - real
    hess = np.power(np.abs(grad), 0.5)

    return grad, hess


for n_estimators in range(5, 600, 5):
    booster_str = "gblinear"
    model = xgb.XGBRegressor(objective=customObj1,
                             booster=booster_str,
                             n_estimators=n_estimators)
    model2 = xgb.XGBRegressor(objective="reg:linear",
                              booster=booster_str,
                              n_estimators=n_estimators)
    model3 = xgb.XGBRegressor(objective=customObj3,
                              booster=booster_str,
                              n_estimators=n_estimators)
    model.fit(X=X_train, y=Y_train)
    model2.fit(X=X_train, y=Y_train)
    model3.fit(X=X_train, y=Y_train)

    y_pred = model.predict(data=pd.DataFrame({'x': X_pred}))
    y_pred2 = model2.predict(data=pd.DataFrame({'x': X_pred}))
    y_pred3 = model3.predict(data=pd.DataFrame({'x': X_pred}))

    plt.figure(figsize=(6, 5))
    plt.axes().set(title='n_estimators=' + str(n_estimators))

    plt.plot(df['x'], df['y'], marker='o', linestyle=":", label="Real Y")
    plt.plot(X_pred, y_pred, label="predict - real; |grad|**0.5")
    plt.plot(X_pred, y_pred3, label="predict - real; |grad|**0.1")
    plt.plot(X_pred, y_pred2, label="reg:linear")

    plt.xlim(-4.5, 8.5)
    plt.ylim(-25, 55)

    plt.legend()
    plt.show()
    # plt.savefig("output/n_estimators_" + str(n_estimators) + ".jpg")
    # plt.close()
    # print(n_estimators)
from sklearn.cluster import DBSCAN