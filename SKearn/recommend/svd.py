# coding=UTF-8
from sqlalchemy import create_engine
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import pandas as pd
import numpy as np
from keras.utils import to_categorical
import jieba
# import multiprocessing
import time
import warnings
warnings.filterwarnings("ignore")


def step1():
    # 初始化数据库连接，使用pymysql模块
    engine = create_engine('mysql+pymysql://root:123456@localhost:3306/user')

    # 查询语句，选出employee表中的所有数据
    sql1 = '''select * from user_shop;'''
    sql2 = '''select * from shop_review;'''
    #sql3 = '''select * from dzdp_shop_detail;'''
    sql4 = '''SELECT * FROM dzdp_shop_count;'''
    sql5 = '''SELECT * FROM dzdp_user_count;'''
    # read_sql_query的两个参数: sql语句， 数据库连接
    user_shop = pd.read_sql_query(sql1, engine)
    user_shop = user_shop[['userId', 'shopId', 'taste', 'environment', 'cost', 'star']]
    user_shop = user_shop.astype('str')
    # 商店评论
    reviews = pd.read_sql_query(sql2, engine)
    # 提取使用的shopId
    shopid = pd.read_sql_query(sql4, engine)
    # 提取使用的userId
    userid = pd.read_sql_query(sql5, engine)

    def str2score(df):
        score_dict = {'': None, '一般': 2, '好': 3,
                      '差': 1, '很好': 4, '非常好': 5}
        star_dict = {'0': None, '10': 1, '20': 2,
                     '30': 3, '40': 4, '50': 5}
        target_labels = ['taste', 'environment', 'cost']
        for label in target_labels:
            df[label] = [score_dict[i] for i in df[label]]
        df['star'] = [star_dict[i] for i in df['star']]
        a = 0.5
        score = a * ((df['taste'] + df['environment'] + df['cost'])/3) + \
                ((1 - a) * df['star'])
        df['score'] = score
        return df[['userId', 'shopId', 'score']]

    df = pd.DataFrame(index=shopid['shopId'], columns=userid['userId'])
    user_shop = str2score(user_shop)
    score_mat = user_shop.pivot_table(index='shopId', columns='userId')
    score_mat.columns = [i[1] for i in score_mat.columns]
    user_lst = score_mat.columns
    shop_lst = score_mat.index
    score_mat.to_csv(root_path + 'score_mat.csv')


    message_all = reviews.reviewBody
    messages_cut = [' '.join(jieba.cut(message,cut_all=False)) for message in tqdm(message_all)]
    #  从文件导入停用词表
    stpwrdpath = root_path + 'stop_words.txt'
    stpwrd_dic = open(stpwrdpath, 'r',encoding='gbk')
    stpwrd_content = stpwrd_dic.read()
    jieba.load_userdict=("/Users/wei/Desktop/model2/add_words.txt")

    #  将停用词表转换为list
    stpwrdlst = stpwrd_content.splitlines()
    stpwrd_dic.close()

    #  计算词频
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    corpus = messages_cut

    #vectorizer = CountVectorizer()
    Vectorizer = CountVectorizer(stop_words=stpwrdlst)
    transformer = TfidfTransformer()
    #cntTf = cntVector.fit_transform(corpus)
    tfidf = transformer.fit_transform(Vectorizer.fit_transform(corpus))


    #  (652936, 103655)
    #  获取词频
    vobs = Vectorizer.vocabulary_
    vobs_order = Vectorizer.get_feature_names()
    Vectorizer_words = tfidf.sum(axis=0)
    Vectorizer_words = Vectorizer_words.tolist()[0]
    words_fred = list(zip(vobs_order, Vectorizer_words))
    words_Vectorizer_df = pd.DataFrame(words_fred, columns=['词语', '频率'])
    words_Vectorizer_df.to_csv(root_path + 'words_cnt_df.csv', index=False, encoding='gbk')


    #  LDA主题聚类
    lda = LatentDirichletAllocation(n_components=5, random_state=0, verbose=1, n_jobs=-1)
    docres = lda.fit_transform(tfidf)
    components = lda.components_
    sort_ = np.argsort(-components, axis=1)

    # 计算主题词汇
    topic_words = []
    for line in sort_[:,:40]:
        topic_words.append([vobs_order[i] for i in line])
    topic_words = pd.DataFrame(topic_words, index=['主题一', '主题二', '主题三',
                                                   '主题四', '主题五'])
    topic_words.to_excel(root_path + 'topic_words.xls')

    #  计算餐厅分类
    doc_cat = docres.argmax(axis=1)
    onehot_doc = to_categorical(doc_cat)
    df_lad = pd.DataFrame(data=onehot_doc)

    # 构造餐厅特征表
    df_new = pd.concat([reviews[['userId', 'shopId', 'cateName']], df_lad], axis=1)
    lda_score = df_new.groupby(by='shopId').sum()
    lda_score['shopId'] = lda_score.index
    lda_score = lda_score[lda_score['shopId'].isin(shop_lst)]
    shop_vec = lda_score.iloc[:, 1:6].values
    shop_vec = [i/i.sum()for i in shop_vec]
    shop_vec = np.array(shop_vec)
    lda_score.iloc[:, 1:6] = shop_vec


    def cosSim(inA,inB):#inA,inB是列向量
        inA = np.expand_dims(inA, 1)
        inB = np.expand_dims(inB, 1)
        num = float(np.matmul(inA.T, inB))
        denom = np.linalg.norm(inA)*np.linalg.norm(inB)
        return 0.5+0.5*(num/denom)


    # 餐厅聚类
    kmodel = KMeans(n_clusters=18, n_jobs=-1)
    kmodel.fit(lda_score.iloc[:, 1:6].values)
    cats = kmodel.predict(lda_score.iloc[:, 1:6].values)
    lda_score['cluster_cate'] = cats
    lda_score.to_excel(root_path + 'lda_score.xls')

def step2(start=0):
    score_mat = pd.read_csv(root_path + 'score_mat.csv', index_col=0)
    lda_score = pd.read_excel(root_path + 'lda_score.xls')

    shopid2cats = dict(zip(lda_score['shopId'], lda_score['cluster_cate']))


    # 给定餐厅id计算虚拟分数
    def padding_score(df_shopscore, shop_):
        sim_lst = []
        cate_name = shopid2cats[int(shop_)]
        df_shopscore = df_shopscore[df_shopscore['cate'] == cate_name]
        for i, row in df_shopscore.iterrows():
            shopid_ = int(row['shopid'])
            shop1_vec = lda_score[lda_score['shopId'] == shopid_].iloc[:, 2:8]
            shop2_vec = lda_score[lda_score['shopId'] == int(shop_)].iloc[:, 2:8]
            similarity = cosine_similarity(shop1_vec, shop2_vec)
            similarity = similarity[0][0]
            sim_lst.append(similarity)
        df_shopscore['sim'] = sim_lst
        df_shopscore['pred_score'] = df_shopscore['sim'] * df_shopscore['score']
        pred_socre = df_shopscore['pred_score'].mean()
        return pred_socre

    def loop(thread_data):
        idx = start
        for userid_, row in thread_data.iterrows():

            print('开始填补第{}个'.format(idx))
            shopid_list = row[row.notnull()]
            cats_lst = []
            for i in shopid_list.index:
                try:
                    cats_lst.append(shopid2cats[int(i)])
                except KeyError:
                    cats_lst.append(99)

            df_shopscore = pd.DataFrame()
            df_shopscore['score'] = shopid_list.values
            df_shopscore['cate'] = cats_lst
            df_shopscore['shopid'] = shopid_list.index
            shop_need2pad = set(all_shopid) - set(shopid_list.index)

            for shop_ in tqdm(shop_need2pad):
                try:
                    pred_socre = padding_score(df_shopscore, shop_)
                    score_mat.loc[shop_, userid_] = pred_socre
                except KeyError:
                    pass
            print('填补{}成功!----------------'.format(userid_))
            idx += 1
            if idx % 10 == 0 :

                score_mat.to_csv(root_path + 'score_mat.csv')
                print('每隔10个用户保存一次, 保存成功!')
    #  2.计算用户消费过的餐厅聚类并且返回该类别
    all_shopid = score_mat.index
    thread_data = score_mat.transpose().iloc[start:, :]
    loop(thread_data)
    score_mat.to_csv(root_path + 'score_mat.csv')
    print('填补完成----------------------------------')

if __name__ == "__main__":
    root_path = '/Users/wei/Desktop/model2/'
    # step1()
    step2(start=0)