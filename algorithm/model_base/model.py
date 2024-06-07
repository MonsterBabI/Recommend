import torch
import torch.nn as nn
import torch.nn.functional as F


class UserFeatureNetwork(nn.Module):
    def __init__(self, user_id_count, gender_count, age_count, job_count, embed_dim, dropout=0.5):
        super(UserFeatureNetwork, self).__init__()

        # 定义嵌入层
        self.user_id_embed = nn.Embedding(user_id_count, embed_dim)
        self.gender_embed = nn.Embedding(gender_count, embed_dim // 2)
        self.age_embed = nn.Embedding(age_count, embed_dim // 2)
        self.job_embed = nn.Embedding(job_count, embed_dim // 2)

        # 定义全连接层
        self.user_id_fc = nn.Linear(embed_dim, embed_dim)
        self.gender_fc = nn.Linear(embed_dim // 2, embed_dim)
        self.age_fc = nn.Linear(embed_dim // 2, embed_dim)
        self.job_fc = nn.Linear(embed_dim // 2, embed_dim)

        # 定义dropout层
        self.dropout = nn.Dropout(dropout)

        # # 定义attention
        # self.attention = Attention(embed_dim*4, embed_dim*4)

        # 最后的全连接层
        self.user_combine_fc = nn.Linear(embed_dim * 4, 200)

    def forward(self, user_id, user_gender, user_age, user_job):
        # 嵌入层
        user_id_embed = self.user_id_embed(user_id)
        gender_embed = self.gender_embed(user_gender)
        age_embed = self.age_embed(user_age)
        job_embed = self.job_embed(user_job)

        # 全连接层并应用dropout
        # user_id_fc = F.relu(self.user_id_fc(user_id_embed))
        # gender_fc = F.relu(self.gender_fc(gender_embed))
        # age_fc = F.relu(self.age_fc(age_embed))
        # job_fc = F.relu(self.job_fc(job_embed))
        user_id_fc = self.dropout(F.relu(self.user_id_fc(user_id_embed)))
        gender_fc = self.dropout(F.relu(self.gender_fc(gender_embed)))
        age_fc = self.dropout(F.relu(self.age_fc(age_embed)))
        job_fc = self.dropout(F.relu(self.job_fc(job_embed)))

        # 拼接所有特征
        user_combine_feature = torch.cat((user_id_fc, gender_fc, age_fc, job_fc), dim=1)

        # 注意力
        # user_combine_feature = self.attention(user_combine_feature)
        # 最后的全连接层
        user_feature = F.relu(self.user_combine_fc(user_combine_feature))
        return user_feature


class MovieFeatureNetwork(nn.Module):
    def __init__(self, movie_id_count, movie_genre_count, embed_dim, dropout=0.5):
        super(MovieFeatureNetwork, self).__init__()
        self.movie_id_embed = nn.Embedding(movie_id_count, embed_dim//2)
        # self.movie_genre_embed = nn.Embedding(movie_genre_count, embed_dim//2)
        self.movie_id_fc = nn.Linear(embed_dim//2, embed_dim)
        self.movie_genre_fc = nn.Linear(movie_genre_count, embed_dim)
        self.dropout = nn.Dropout(dropout)
        # self.attention = Attention(2*embed_dim, 2*embed_dim)
        self.movie_combine_fc = nn.Linear(2*embed_dim, 200)


    def forward(self, movie_id, movie_genre):
        movie_id_embed = self.movie_id_embed(movie_id)
        # movie_genre_embed = self.movie_genre_embed(movie_genre)
        # movie_id_fc = self.dropout(F.relu(self.movie_id_fc(movie_id_embed)))
        # movie_genre_fc = self.dropout(F.relu(self.movie_genre_fc((movie_genre).float())))
        movie_id_fc = F.relu(self.movie_id_fc(movie_id_embed))
        movie_genre_fc = F.relu(self.movie_genre_fc((movie_genre).float()))
        movie_combine_feature = torch.cat((movie_id_fc, movie_genre_fc), dim=1)
        # movie_combine_feature = self.attention(movie_combine_feature)
        # movie_combine_feature = movie_combine_feature.transpose(1,2)
        movie_feature = F.relu(self.movie_combine_fc(movie_combine_feature))
        return movie_feature


class RecommenderNetwork(nn.Module):
    def __init__(self, user_feature_net, movie_feature_net):
        super(RecommenderNetwork, self).__init__()
        self.user_feature_net = user_feature_net
        self.movie_feature_net = movie_feature_net
        self.temperature = 0.07

    def forward(self, user_id, user_gender, user_age, user_job, movie_id, movie_genre):
        user_feature = self.user_feature_net(user_id, user_gender, user_age, user_job)
        movie_feature = self.movie_feature_net(movie_id, movie_genre)

        # 归一化
        #
        user_feature = F.softmax(user_feature, dim=1) / (self.temperature)
        movie_feature = F.softmax(movie_feature, dim=1) / (self.temperature)
        similar = torch.mm(user_feature, movie_feature.t())
        similar = F.softmax(similar, dim=1)
        # return torch.mm(user_feature, movie_feature.t())
        return similar

def test_user():
    # 假设一些输入参数
    USER_ID_COUNT = 1000
    GENDER_COUNT = 2
    AGE_COUNT = 10
    JOB_COUNT = 20
    EMBED_DIM = 32
    DROPOUT_KEEP_PROB = 0.5
    # 初始化网络
    model = UserFeatureNetwork(USER_ID_COUNT, GENDER_COUNT, AGE_COUNT, JOB_COUNT, EMBED_DIM, DROPOUT_KEEP_PROB)

    # 生成一些虚拟数据来进行测试
    batch_size = 5  # 批量大小
    user_ids = torch.randint(0, USER_ID_COUNT, (batch_size,))
    user_genders = torch.randint(0, GENDER_COUNT, (batch_size,))
    user_ages = torch.randint(0, AGE_COUNT, (batch_size,))
    user_jobs = torch.randint(0, JOB_COUNT, (batch_size,))

    # 将模型切换到评估模式
    model.eval()

    # 进行前向传播并打印结果
    with torch.no_grad():
        user_features = model(user_ids, user_genders, user_ages, user_jobs)
        print("输出用户特征向量的形状:", user_features.shape)
        print("输出用户特征向量:", user_features)

def test_movie():
    movieID = 1000
    movieG = 18
    embed = 32
    model = MovieFeatureNetwork(movieID, movieG, embed)

    batch_size = 5  # 批量大小
    movie_id = torch.randint(0, movieID, (batch_size,))
    movie_gen = torch.randint(0, movieG, (batch_size,18)).float()
    model.eval()
    with torch.no_grad():
        movie_feature = model(movie_id, movie_gen)
        print(movie_feature.shape)
        print(movie_feature)

if __name__ == '__main__':
    user_network = UserFeatureNetwork(user_id_count=6041, gender_count=2, age_count=7, job_count=21, embed_dim=128)
    movie_network = MovieFeatureNetwork(movie_id_count=3953, movie_genre_count=18, embed_dim=128)
    model = RecommenderNetwork(user_network, movie_network)
    print(model)
    # test_movie()
    # import dataset
    #
    # path = '../../data/ml-1m'
    # user, movie, _ = dataset.data_preprocess(*dataset.load_data(path))
    # print()
    # movieId = movie['movieId']
    # genre = movie['genre']




