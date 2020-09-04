import requests
from bs4 import BeautifulSoup
import os
# 搜索相似图片抓取
class pic():
    def __init__(self):
        # 文件目录
        self.path='/Users/zuobangbang/Desktop/image3'
        self.alder=['cat','butterfly','apple','helmet']

    def get_pic(self,url,pic_path):
        html=requests.get(url).content
        with open(pic_path,'wb') as f:
            f.write(html)

    def spider(self):
        files = os.listdir(path=self.path)
        print(files)
        # files = files[::-1]
        for file in files:
            w = os.listdir(self.path + '/' + file)
            # 如果某一类超过1800张就不抓了
            if not len(w) > 1800:
                start=len(w)
                # 把file修改为要抓取的文件名，如果需要换成中文搜索，就把pict换成中文的url编码格式
                if file=='frenchhorn':
                    pict='frenchhorn'
                else:
                    continue
                nums=start+1
                headers = {
                    'Host': 'www.veer.com',
                    'Connection': 'keep-alive',
                    'Cache-Control': 'max-age=0',
                    'Upgrade-Insecure-Requests': '1',
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.87 Safari/537.36',
                    # 'Sec-Fetch-Mode': 'navigate',
                    # 'Sec-Fetch-User': '?1',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3',
                    'Sec-Fetch-Site': 'none',
                    # 'Referer': 'https://www.veer.com/query/photo/?phrase=%E9%A6%99%E8%95%89&page=4&perpage=100&key=B2NCDBG',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Accept-Language': 'zh-CN,zh;q=0.9',
                    'Cookie': 'bd_vid=8186901724063600985; bd_vid.sig=JVkP5e0Lt2HPcFA21I32zaQxOWE; logidUrl=https://www.veer.com/photo/?utm_source=baidu&utm_medium%20=cpc&utm_campaign=%E9%80%9A%E7%94%A8%E8%AF%8D-%E7%BD%91%E7%AB%99&utm_content=%E9%80%9A%E7%94%A8%E8%AF%8D-%E7%BD%91%E7%AB%99&utm_term=%E6%9C%80%E5%A4%A7%E7%9A%84%E5%9B%BE%E7%89%87%E7%BD%91&chid=901&b_scene_zt=1&bd_vid=8186901724063600985; logidUrl.sig=5q8K6SaRehAbQ030csKGoVpcTic; koa.sid=b4Yh7wI7CW7XvF8Lczi6uNOWtvL2tqGk; koa.sid.sig=ESn4wuc_vydG4HeMv8MAMxLnw4M; _ga=GA1.2.218858338.1575025518; _gid=GA1.2.652787100.1575025518; sajssdk_2015_cross_new_user=1; Hm_lvt_f2de34005a64c75f44715498bcb136f9=1575025521; sensorsdata2015jssdkcross=%7B%22distinct_id%22%3A%2216eb6d62c52a29-0c59cd90469a4f-1c3e6754-1296000-16eb6d62c536bc%22%2C%22%24device_id%22%3A%2216eb6d62c52a29-0c59cd90469a4f-1c3e6754-1296000-16eb6d62c536bc%22%2C%22props%22%3A%7B%22%24latest_referrer%22%3A%22%22%2C%22%24latest_traffic_source_type%22%3A%22%E7%9B%B4%E6%8E%A5%E6%B5%81%E9%87%8F%22%2C%22%24latest_search_keyword%22%3A%22%E6%9C%AA%E5%8F%96%E5%88%B0%E5%80%BC_%E7%9B%B4%E6%8E%A5%E6%89%93%E5%BC%80%22%2C%22%24latest_utm_source%22%3A%22baidu%22%2C%22%24latest_utm_campaign%22%3A%22%E9%80%9A%E7%94%A8%E8%AF%8D-%E7%BD%91%E7%AB%99%22%2C%22%24latest_utm_content%22%3A%22%E9%80%9A%E7%94%A8%E8%AF%8D-%E7%BD%91%E7%AB%99%22%2C%22%24latest_utm_term%22%3A%22%E6%9C%80%E5%A4%A7%E7%9A%84%E5%9B%BE%E7%89%87%E7%BD%91%22%7D%7D; Hm_lpvt_f2de34005a64c75f44715498bcb136f9=1575032100; _fp_=eyJpcCI6IjExMi4yLjI1Mi40MyIsImZwIjoiNzk2NDhkNjJmMTY4YzMxMzMyY2FhNzMyMDQ5NTVhZWMiLCJocyI6IiQyYSQwOCQ0TVdLdDZOY1RCUnFDdjJTeW5JTXN1bEpWUXJ5UDlkc0pqY2xxQVp5aWhiZ2lCOFJxRzVYSyJ9'

                }
                pic_path=self.path+'/'+file+'/'+file+'_'
                if nums<100:
                    x=int(start / 200) + 1
                else:
                    x=int(start / 200) + 2
                for page in range(x, 13):
                    print('start :', pict,'and the page is ',page)
                    # perpage默认改为200，page={1}，如果要修改pharse的话就改成pharse={0}
                    url='https://www.veer.com/query/photo/?phrase=%E5%9C%86%E5%8F%B7&page={1}&perpage=200&similarId=51117002&key=7PNM0Q'.format(pict,page)
                    html=requests.get(url,headers=headers)
                    soup=BeautifulSoup(html.text,'lxml')
                    for i in soup.find_all('a',class_='search_result_asset_link'):
                        if nums%50==0:
                            print('the number of',file,'is ',nums)
                        pic_url=i.find('img').get('src')
                        if nums<100:
                            picpath=pic_path+'0'+str(nums)+'.jpg'
                        else:
                            picpath = pic_path + str(nums) + '.jpg'
                        self.get_pic(pic_url,picpath)
                        nums+=1
                    # 如果抓取的总数大于1800，那这一类就抓取完毕
                    if len(w)>1800:
                        break







if __name__=="__main__":
    q=pic()
    q.spider()



# car
# ['apple', 'helmet', 'cat', 'butterfly', 'piano', 'car', 'pig', 'hammer', 'snail','turtle',
# baseball，ladybird(部分)
# 'sealion', 'mushroom', 'dog', 'lemon', 'babycrib', 'bear', 'ladybird', 'penguin', 'baseball', 'boat',
# 'camel', 'tiger', /'guitar', 'starfish','motorbike', 'chook', 'coffeecup','mouse', 'lizard', 'sofa',
# frenchhorn,aeroplane
# 'bowl', 'bird', /'aeroplane', 'frenchhorn', 'train', /'monkey', 'rabbit','banana', 'axe','viola',
# mobulidae
# 'billiardtable',/ 'frog', 'mobulidae', 'pepper',/ 'horse', 'pineapple', 'goldenfish', 'cow', 'deer', 'snake']
# 蝠鲼
# ['cat', 'butterfly', 'apple', 'helmet','snake]

if __name__=="__main__":
    q=pic()
    q.spider()