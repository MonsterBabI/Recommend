import sys
import time
import numpy as np
import pandas as pd
import random
from gensim.models import Word2Vec
from gensim.models.word2vec import Word2Vec
from tqdm import tqdm
import pickle
import os

def cosine_similarity(list1, list2):
    # 将列表转换为NumPy数组
    vec1 = np.array(list1)
    vec2 = np.array(list2)
    # 计算两个向量的点积
    dot_product = np.dot(vec1, vec2)
    # 计算两个向量的模长
    vec1_norm = np.linalg.norm(vec1)
    vec2_norm = np.linalg.norm(vec2)
    # 计算余弦相似度
    similarity = dot_product / (vec1_norm * vec2_norm)
    return similarity

class Item_base_rec:
    def __init__(self, datafile):
        self.test_users = None
        self.train_users = None
        self.test_data = None
        self.train_data = None
        self.model = None  # 如果训练后则可得到这个模型
        self.movie = None
        self.data_path = datafile  # 训练数据的路径
        self.load_data()  # 实例化后自动获得对应数据

    # 导入数据
    def load_data(self):
        if '25m' in self.data_path:
            # 导入数据集
            ratings = pd.read_csv(self.data_path+'/'+'ratings.csv')
            movies = pd.read_csv(self.data_path+'/'+'movies.csv')
            # 切片，测试使用
            # ratings = ratings.iloc[:int(len(ratings)*0.0001)]
            # 划分数据集
            self.train_data = ratings.sample(frac=0.8, random_state=1)
            self.test_data = ratings.drop(self.train_data.index)
            self.train_users = self.train_data['userId'].unique().tolist()
            self.test_users = self.test_data['userId'].unique().tolist()
            # 训练集全部投入使用
            self.train_data = ratings
            self.train_users = self.train_data['userId'].unique().tolist()
        elif '1m' in self.data_path:
            ratings = pd.read_csv(self.data_path+'/'+'ratings.dat', header=None, sep='::', engine='python')
            movies = pd.read_csv(self.data_path+'/'+'movies.dat', header=None, sep='::', engine='python', encoding='ISO-8859-1')
            ratings.columns = ["userId", "movieId", "rating", "timestamp"]
            movies.columns = ["movieId", "title", "genres"]
            self.train_data = ratings.sample(frac=0.8, random_state=1)
            self.test_data = ratings.drop(self.train_data.index)
            self.train_users = self.train_data['userId'].unique().tolist()
            self.test_users = self.test_data['userId'].unique().tolist()
            self.movie = movies


    # 用于训练,在放入数据后得到训练出来的模型
    def train(self, path=None):
        # 若有模型文件路径即可跳过训练
        if path is None:
            movies_history = []
            # movies_history里面是训练集中用户的电影评分历史
            for i in tqdm(self.train_users):
                # astype，将元素变成str类型，将其作为模型的输入
                temp = self.train_data[self.train_data['userId'] == i]['movieId'].astype(str).tolist()
                movies_history.append(temp)
            print('进行模型的训练')
            model = Word2Vec(window=10, min_count=1, sg=1, hs=0, negative=10, alpha=0.03, min_alpha=0.0007, seed=14)
            # movies_history作为输入
            model.build_vocab(movies_history, progress_per=200)
            model.train(movies_history, total_examples=model.corpus_count, epochs=10, report_delay=1)
            # model.save('./model')
            self.model = model
            now_time = str(time.time())
            model.save('../model/model'+now_time[5:8])
        else:
            self.model = Word2Vec.load(path)
        return self.model
        # word_vec = model.wv

    def predict_score(self, userId, movieId):
        word_vec = self.model.wv
        # 向量对应的id
        word_vec_index = word_vec.index_to_key
        # movies_list 获得对应用户id所评分过的电影
        movies_list = list(self.train_data[self.train_data['userId'] == int(userId)]['movieId'])
        vectors = word_vec[word_vec.key_to_index]
        average_vector = np.mean(vectors, axis=0)
        # record[1]对应movieId
        if movieId in word_vec:  # 判断是否存在在word_vec的训练数据中，v为目标评分电影的向量
            v = word_vec[movieId]
        else:
            v = np.mean(vectors, axis=0)
        movies_vec = []  # movies_vec 获取用户所评分过的电影的电影向量
        for i in movies_list:
            if str(i) in word_vec:
                movies_vec.append(word_vec[str(i)])
            else:
                movies_vec.append(average_vector)
        # movies_sim 得到与目标电影相似的相似度
        movies_sim = [cosine_similarity(v, i) for i in movies_vec]
        movies_sim = np.array(movies_sim)
        # 获取该用户评估电影的评分列表
        rates_list = np.array(self.train_data[self.train_data['userId'] == int(userId)]['rating'])
        score = np.dot(movies_sim, rates_list) / sum(movies_sim)
        return score

    # 给用户推荐
    def recommend(self, user_Id, n):
        if int(user_Id) in self.train_users:
            movieIds = np.array(self.train_data['movieId'])
            movieIds = np.unique(movieIds)
            score_list = np.array([self.predict_score(user_Id, str(i)) for i in movieIds])
            max_index = score_list.argpartition(-n)[-n:]
            rec_movie = self.movie['title'].iloc[max_index]
            return rec_movie
        else:
            print('该用户不存在')
            return None

    # 评价指标
    def rmse(self):
        scorelist = []
        for _, record in self.test_data.iterrows():
            userId = int(record[0])
            movieId = int(record[1])
            scorelist.append(self.predict_score(userId, movieId))
        print('开始计算RMSE')
        score_new = []
        for i in scorelist:
            if i == 0:
                print('?')
                score_new.append(3)
            else:
                score_new.append(i)
        a = np.array(score_new)
        b = np.array(list(self.test_data['rating']))
        result = np.mean((a - b) ** 2)
        return result ** 0.5

    def inference(self):
        pass

if __name__ == '__main__':
    path = '../data/ml-1m'
    item = Item_base_rec(path)
    start = time.time()
    item.train('../model/model363')
    # print(item.predict_userId('2', '33'))
    print(item.recommend('2',6))
    # print(item.rmse())
    end = time.time()
    print(f"用时:{((end-start)/60)} mins") #