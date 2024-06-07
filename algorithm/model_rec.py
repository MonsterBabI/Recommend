import time
import torch.cuda
from tqdm import tqdm
# from model_base.preprocess import load_data, data_preprocess
# from model_base.dataset import MovieLensDataset
# from model_base.model import UserFeatureNetwork, MovieFeatureNetwork, RecommenderNetwork
from .model_base.preprocess import load_data, data_preprocess
from .model_base.dataset import MovieLensDataset
from .model_base.model import UserFeatureNetwork, MovieFeatureNetwork, RecommenderNetwork
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
# UserId_count = 6041
# movie_id_count = 3953

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    print(device)
    path = '../data/ml-1m'
    # dataset：user_feature, movie_feature, rating
    # user_feature：（4，）数组
    # movie_feature：（2，）列表
    # rating：int
    data = load_data(path)
    users_process, movies_process, ratings_process = data_preprocess(*data)
    # 切片，测试使用
    # ratings_process = ratings_process.iloc[:int(0.001*len(ratings_process)),:]
    dataset = MovieLensDataset(users_process, movies_process, ratings_process)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    # 初始化模型
    user_network = UserFeatureNetwork(user_id_count=len(users_process)+1, gender_count=2, age_count=7, job_count=21, embed_dim=128).to(device)
    movie_network = MovieFeatureNetwork(movie_id_count=max(movies_process['movieId'])+1, movie_genre_count=len(movies_process['genre'][0]), embed_dim=128).to(device)
    model = RecommenderNetwork(user_network, movie_network).to(device)
    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=1e-7)
    # criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    # 训练循环
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        loss = 0
        for user_feature, movieId, genre, rating in tqdm(dataloader):
            user_ids, user_gender, user_age, user_job = user_feature.t()
            user_ids, user_gender, user_age, user_job = user_ids.to(device), user_gender.to(device), user_age.to(device), user_job.to(device)
            movieId, genre = movieId.to(device), genre.to(device)

            # 前向传播
            outputs = model(user_ids, user_gender, user_age, user_job, movieId, genre)

            # outputs = outputs.diag()
            # loss = criterion(outputs.float(), rating.float())
            n = len(outputs)
            rating = (rating / 5.0 / 128).to(device)
            loss_u = sum([criterion(outputs[i], rating) for i in range(n)]) / n
            loss_m = sum([criterion(j, rating) for j in outputs.t()]) / n
            loss = (loss_m + loss_u)/2

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        time.sleep(0.1)
        print(evaluation(user_network, movie_network, users_process, movies_process, device))
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    torch.save(model.state_dict(), '../model/model_based_model/model.pt')
    torch.save(user_network.state_dict(), '../model/model_based_model/user_network.pt')
    torch.save(movie_network.state_dict(), '../model/model_based_model/movie_network.pt')

def evaluation(unet, mnet, users_process, movies_process, device):
    unet.eval(), mnet.eval()
    # 得到用户特征
    # user = users_process[users_process['userId'] == userId].values
    user = users_process.values
    user = torch.tensor(user).to(device).T
    user_feature = unet(*user)  # 1*200

    # 电影的矩阵
    movieId = np.array(movies_process['movieId'])
    genre = np.array(list(movies_process['genre']))

    movieId = torch.tensor(movieId).to(device)
    genre = torch.tensor(genre).to(device).float()
    # movie[0], movie[1] = ,
    movie_feature = mnet(movieId, genre)
    match_degree = torch.mm(user_feature, movie_feature.t())
    max_index = torch.argsort(match_degree, descending=True, dim=1)[:, :3].cpu()
    max_index = torch.sum(max_index, dim=1).numpy()
    evla = np.unique(max_index)

    return len(evla), len(evla)/len(match_degree)
    # movie_title = data[1].iloc[max_index]['title']



# def recommend(data_path, path, userId=2, n=3):
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     data = load_data(data_path)
#     users_process, movies_process, ratings_process = data_preprocess(*data)
#     # dataset = MovieLensDataset(users_process, movies_process, ratings_process)
#     # dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
#
#     user_network = UserFeatureNetwork(user_id_count=6041, gender_count=2, age_count=7, job_count=21, embed_dim=128).to(device)
#     movie_network = MovieFeatureNetwork(movie_id_count=3953, movie_genre_count=18, embed_dim=128).to(device)
#     model = RecommenderNetwork(user_network, movie_network).to(device)
#     model.load_state_dict(torch.load(path, map_location=device))
#     unet = model.user_feature_net
#     mnet = model.movie_feature_net
#     # user_network.load_state_dict(torch.load('model_base/user_network.pt', map_location=device))
#     # movie_network.load_state_dict(torch.load('model_base/movie_network.pt', map_location=device))
#     # user_network.eval()
#     # movie_network.eval()
#
#     # 得到用户特征
#     user = users_process[users_process['userId'] == userId].values
#     user = torch.tensor(user).to(device).T
#     user_feature = unet(*user) # 1*200
#
#     # 电影的矩阵
#     movieId = np.array(movies_process['movieId'])
#     genre = np.array(list(movies_process['genre']))
#
#     movieId = torch.tensor(movieId).to(device)
#     genre = torch.tensor(genre).to(device).float()
#     # movie[0], movie[1] = ,
#     movie_feature = mnet(movieId, genre)
#     max_index = torch.argsort(torch.mm(user_feature, movie_feature.t())[0], descending=True)[:n].cpu()
#
#     movie_title = data[1].iloc[max_index]['title']
#     return movie_title.values

class model_based_rec:
    def __init__(self, data_path, mpath, upath):
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = 'cpu'
        self.data = load_data(data_path)
        self.users_process, self.movies_process, self.ratings_process = data_preprocess(*self.data)
        self.user_network = UserFeatureNetwork(user_id_count=6041, gender_count=2, age_count=7, job_count=21,
                                          embed_dim=128).to(self.device)
        self.movie_network = MovieFeatureNetwork(movie_id_count=3953, movie_genre_count=18, embed_dim=128).to(self.device)
        self.user_network.load_state_dict(torch.load(upath, map_location=self.device))
        self.movie_network.load_state_dict(torch.load(mpath, map_location=self.device))
        self.user_network.eval()
        self.movie_network.eval()
        # 电影的矩阵
        movieId = np.array(self.movies_process['movieId'])
        genre = np.array(list(self.movies_process['genre']))

        movieId = torch.tensor(movieId).to(self.device)
        genre = torch.tensor(genre).to(self.device).float()
        # movie[0], movie[1] = ,
        self.movie_feature = self.movie_network(movieId, genre)
    def recommend(self, userId, n):
        # 得到用户特征
        user = self.users_process[self.users_process['userId'] == userId].values
        user = torch.tensor(user).to(self.device).reshape(4, 1)
        user_feature = self.user_network(*user)  # 1*200


        max_index = torch.argsort(torch.mm(user_feature, self.movie_feature.t())[0], descending=True)[:n].cpu()

        movie_title = list(self.data[1].iloc[max_index]['title'])
        return movie_title

if __name__ == '__main__':
    # train()
    for i in range(10, 15):
        print(recommend('../data/ml-1m','../model/model_based_model/model.pt', userId=i))
