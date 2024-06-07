import math, random, json, os
from tqdm import tqdm
import pandas as pd
# 定义一个基于用户协同过滤的推荐系统类
class UserCFRec:
    # 类初始化方法
    def __init__(self, datafile):
        # 获取电影标题
        self.movie_title = None

        self.datafile = datafile  # 数据文件路径
        self.data = self.loadData()  # 加载数据

        # 分割数据为训练集和测试集
        self.trainData, self.testData = self.splitData(3, 47)
        # 计算用户相似度
        self.users_sim = self.UserSimilarityBest()

    # 加载数据方法
    def loadData(self):
        data = []
        # 从文件中读取数据，每行格式为userid::itemid::record::timestamp
        for line in open(self.datafile+'/ratings.dat'):
            userid, itemid, record, _ = line.split("::")
            if userid.isdigit():
                # 将数据添加到列表中，每个元素为一个元组(userid, itemid, record)
                data.append((userid, itemid, float(record)))
        movies = pd.read_csv(self.datafile + '/' + 'movies.dat', header=None, sep='::', engine='python',
                             encoding='ISO-8859-1')
        movies.columns = ["movieId", "title", "genres"]
        self.movie_title = movies
        return data

    # 数据分割方法，用于创建训练集和测试集
    def splitData(self, k, seed, M=8):
        train, test = {}, {}
        random.seed(seed)
        # 遍历数据，随机分配到训练集或测试集
        for user, item, record in self.data:
            vt = random.randint(0,M)
            if vt == k:
                test.setdefault(user, {})
                test[user][item] = record
            else:
                train.setdefault(user, {})
                train[user][item] = record

        return train, test

    # 计算用户相似度的方法
    def UserSimilarityBest(self):
        # print("开始计算用户之间的相似度 ...")
        # 如果已经计算过用户相似度，直接从文件加载
        if os.path.exists(self.datafile+"/user_sim.json"):
            # print("用户相似度从文件加载 ...")
            userSim = json.load(open(self.datafile+'/user_sim.json', "r"))
        else:
            # print('开始计算用户相似度')
            # 构建物品-用户倒排表
            item_users = dict()
            for u, items in self.trainData.items():
                for i in items.keys():
                    item_users.setdefault(i, set()) # 创建的是字符串为key，集合为value的字典
                    if self.trainData[u][i] > 0:
                        item_users[i].add(u)
            # 计算用户间共同评分的物品数
            count = dict()
            user_item_count = dict()
            for i, users in item_users.items():
                for u in users:
                    user_item_count.setdefault(u, 0)
                    user_item_count[u] += 1
                    count.setdefault(u, {})
                    for v in users:
                        count[u].setdefault(v, 0)
                        if u == v:
                            continue
                        # 更新u和v的共同评分物品数，使用惩罚项避免热门物品对相似度的影响
                        count[u][v] += 1 / math.log(1 + len(users))
            # 构建相似度矩阵
            # print('正在构建相似度矩阵')
            userSim = dict()
            for u, related_users in tqdm(count.items()):
                userSim.setdefault(u, {})
                for v, cuv in related_users.items():
                    if u == v:
                        continue
                    userSim[u].setdefault(v, 0.0)
                    # 计算u和v的相似度
                    userSim[u][v] = cuv / math.sqrt(user_item_count[u] * user_item_count[v])
            # 将计算结果保存到文件
            json.dump(userSim, open('./ml-1m/user_sim.json', 'w'))
        return userSim

    # 推荐方法
    def recommend(self, user, nitems=10, k=8 ):
        result = dict()
        user = str(int(user))
        # 获取用户已评分的物品
        have_score_items = self.trainData.get(user, {})
        similar_people = sorted(self.users_sim[user].items(), key=lambda x: x[1], reverse=True)[0:k]
        # 遍历与用户相似的前k个用户
        for v, wuv in similar_people:
            # 遍历这些用户评分的物品
            for i, rvi in self.trainData[v].items():
                if i in have_score_items:
                    continue
                result.setdefault(i, 0)
                # 计算推荐分数
                result[i] += wuv * rvi
        # 返回前nitems个推荐结果
        movies = dict(sorted(result.items(), key=lambda x: x[1], reverse=True)[0:nitems]).keys()
        movies_list = list(map(int, movies))
        titles = self.movie_title[self.movie_title['movieId'].isin(movies_list)]['title']
        similar_people = list(map(lambda x: x[0], similar_people))
        return list(titles), similar_people
    # 计算RMSE
    # def RMSE(self):

    # 计算准确率的方法
    def precision(self, k=8, nitems=10):
        hit = 0
        precision = 0
        # 遍历训练集中的用户
        for user in tqdm(self.trainData.keys()):
            tu = self.testData.get(user, {})
            # 获取推荐列表
            rank = self.recommend(user, k=k, nitems=nitems)
            # 计算命中率
            for item, rate in rank.items():
                if item in tu:
                    hit += 1
            precision += nitems
        # 返回准确率
        return hit / (precision * 1.0) # 0.19594370860927152
if __name__ == '__main__':

    # 创建推荐系统实例
    path_25m = './ml-25m/ratings.csv'
    path_1m = '../data/ml-1m'
    rec = UserCFRec(path_1m)
    # rec = UserCFRec('./ml-1m/ratings.dat')
    # 计算准确率
    # pre = rec.precision(10,10)
    # print('precision', pre)
    print(rec.recommend('11'))