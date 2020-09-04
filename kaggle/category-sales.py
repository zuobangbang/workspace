import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
from numpy import *
import seaborn as sns







def load_data():
    item_categories=pd.read_csv( '/Users/zuobangbang/Desktop/future-sales/item_categories.csv')
    print('item_categories:',item_categories.head())
    items=pd.read_csv('/Users/zuobangbang/Desktop/future-sales/items.csv')
    sales_train=pd.read_csv('/Users/zuobangbang/Desktop/future-sales/sales_train.csv')
    shops = pd.read_csv('/Users/zuobangbang/Desktop/future-sales/shops.csv')
    test =pd.read_csv( '/Users/zuobangbang/Desktop/future-sales/test.csv')
    print('test:',test.head())
    # sub=pd.read_csv( '/Users/zuobangbang/Desktop/future-sales/sample_submission.csv')
    #剔除离群点
    fig=plt.figure(figsize=(15,10))
    # sns.distplot(sales_train['item_price'],bins=10)
    # sns.boxplot(sales_train['item_price'])
    # plt.show()
    # print(sales_train.shape)
    sales_train=sales_train[(sales_train['item_cnt_day']<1000) & (sales_train['item_price']<100000)]
    # print(sales_train.shape)
    #检查是否有空值
    # print(sales_train.isnull().sum())
    #检查价格是否有负值，有的话用平均数代替
    # print(sales_train[sales_train['item_price']<=0])
    va=sales_train[(sales_train['shop_id']==32) & (sales_train['item_id']==2973) & (sales_train['date_block_num']==4)&(sales_train['item_price']>0)]['item_price'].median()
    sales_train.loc[sales_train['item_price']<=0,'item_price'] =va
    # print(sales_train.loc[484683])
    shops.loc[shops['shop_name']=='Сергиев Посад ТЦ "7Я"','shop_name']='СергиевПосад ТЦ "7Я"'
    shops['city']=shops['shop_name'].str.split(' ').map(lambda x:x[0])
    shops.loc[shops['city'] == '!Якутск', 'city'] = 'Якутск'
    shops['city_code']=shops['city'].astype('category').cat.codes
    item_categories['type']=item_categories['item_category_name'].str.split('-').map(lambda x:x[0])
    item_categories['type_code']=item_categories['type'].astype('category').cat.codes
    item_categories['subtype'] = item_categories['item_category_name'].str.split('-').map(lambda x: x[1].strip() if len(x)>1 else x[0].strip())
    item_categories['subtype_code'] = item_categories['subtype'].astype('category').cat.codes
    newitems=pd.merge(items,item_categories,on='item_category_id')
    print(newitems.head())
    # len(list(set(sales_train['item_id'])))-len(list(set(newitems['item_id']).intersection(set(sales_train['item_id']))))=0
    train=pd.merge(sales_train,newitems,on='item_id')
    train=pd.merge(train,shops,on='shop_id')
    print(train.shape)
    # print(item_categories)
    print('items:',items.head())
    print('sales_train:',sales_train.head())
    # print(sales_train['item_cnt_day'].unique())
    print('shops:',shops.head())
    # print(test.head())
    print(sales_train['date_block_num'].unique())
    sales_train['revenue'] = sales_train['item_price'] * sales_train['item_cnt_day']
    matrix = []
    # cols = ['date_block_num', 'shop_id', 'item_id']
    # for i in range(34):
    #     sales = sales_train[sales_train.date_block_num == i]
    #     matrix.append(array(list(product([i], sales.shop_id.unique(), sales.item_id.unique())), dtype='int16'))
    #     print(matrix)
    #
    # matrix = pd.DataFrame(np.vstack(matrix), columns=cols)
    # matrix['date_block_num'] = matrix['date_block_num'].astype(np.int8)
    # matrix['shop_id'] = matrix['shop_id'].astype(np.int8)
    # matrix['item_id'] = matrix['item_id'].astype(np.int16)
    # matrix.sort_values(cols, inplace=True)
    test['date_block_num']=34
    s=0
    for i in sales_train['shop_id'].unique():
        s+=len(sales_train[sales_train['shop_id']==i]['item_id'].unique())
    print(s)
    s = 0
    for i in test['shop_id'].unique():
        s += len(test[test['shop_id'] == i]['item_id'].unique())
    print(s)
    print(sales_train['shop_id'].unique(),len(sales_train['shop_id'].unique()))
    print(sales_train['item_id'].unique(), len(sales_train['item_id'].unique()))
    print(test['shop_id'].unique(), len(test['shop_id'].unique()))
    print(test['item_id'].unique(), len(test['item_id'].unique()))
    print(sales_train[(sales_train['item_id']==9782) & (sales_train['shop_id']==45)])
    print(test.tail(22))


if __name__=='__main__':
    load_data()
