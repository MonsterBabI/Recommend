import numpy
import numpy as np
import pandas as pd
import random
from gensim.models import Word2Vec
from gensim.similarities import SparseMatrixSimilarity
from gensim import matutils
from tqdm import tqdm
import pickle
import os
# 导入数据集
ratings = pd.read_csv('./ml-25m/ratings.csv')
movies = pd.read_csv('./ml-25m/movies.csv')
# 切片，测试使用
ratings = ratings.iloc[:int(len(ratings)*0.1), ]

# 划分数据集
train_data = ratings.sample(frac=0.8, random_state=1)
test_data = ratings.drop(train_data.index)
train_users = train_data['userId'].unique().tolist()
test_users = test_data['userId'].unique().tolist()

def train(train_users):
    movies_history = []
    # movies_history里面是训练集中用户的电影评分历史
    for i in tqdm(train_users):
        # astype，将元素变成str类型，将其作为模型的输入
        temp = train_data[train_data['userId']==i]['movieId'].astype(str).tolist()
        movies_history.append(temp)
    print('进行模型的训练')
    model = Word2Vec(window=10, min_count=1, sg=1, hs=0, negative=10, alpha=0.03, min_alpha=0.0007, seed=14)
    # movies_history作为输入
    model.build_vocab(movies_history,progress_per=200)
    model.train(movies_history, total_examples=model.corpus_count, epochs=10, report_delay=1)
    return model


# 计算RMSE
def predict_score(model):
    score_list = []
    print()
    print('进行评分的预测')
    word_vec = model.wv
    # 获取word_vec中所有的向量
    vectors = word_vec[word_vec.key_to_index]
    vectors_file_name = 'vector_2.5m.npy'
    np.save(vectors_file_name, vectors)
    # 计算平均向量
    average_vector = np.mean(vectors, axis=0)
    for _, record in tqdm(test_data.iterrows()):
        movies_list = list(train_data[train_data['userId'] == int(record[0])]['movieId'])
        if str(int(record[1])) in word_vec:
            v = word_vec[str(int(record[1]))]
        else:
            v = average_vector
        # movies_vec = [word_vec.get(str(i), np.zeros(41)) for i in movies_list]
        movies_vec = []
        for i in movies_list:
            if str(i) in word_vec:
                movies_vec.append(word_vec[str(i)])
            else:
                movies_vec.append(average_vector)
        movies_sim = [cosine_similarity(v, i) for i in movies_vec]
        movies_sim = np.array(movies_sim)
        rates_list = np.array(train_data[train_data['userId']==int(record[0])]['rating'])
        score = np.dot(movies_sim, rates_list)/sum(movies_sim)
        score_list.append(score)
    return score_list
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


def read_score(path):
    with open(path, 'rb') as f:
        predict_score_list = pickle.load(f)
    return predict_score_list

def RMSE(score_List):
    print('开始计算RMSE')
    result = 0
    score_new = []
    for i in score_List:
        if i == 0:
            score_new.append(3)
        else:
            score_new.append(i)
    a = np.array(score_new)
    b = np.array(list(test_data['rating']))
    result = np.mean((a-b)**2)
    # for j in tqdm(range(len(score_new))):
    #     result += (score_new[j]-list(test_data['rating'])[j])**2
    # result = (result/len(score_new))**0.5
    return result**0.5


def recommend_movies(path):
    products = train_data[['userId', 'movieId']]
    products.drop_duplicates(inplace=True, subset='movieId', keep='last')
    products = products.groupby('movieId')[
        'userId'].apply(list).to_dict()
    vector = np.load(path)

    print()
    pass


# path = 'vector_2.5m.npy'
# recommend_movies(path)
s_list = pickle.load(open('predict_score_2.5m.pkl', 'rb'))
print(RMSE(s_list))