from torch.utils.data import Dataset, DataLoader
import torch

class MovieLensDataset(Dataset):
    def __init__(self, user_data, movie_data, ratings):
        self.user_data = user_data
        self.movie_data = movie_data
        self.ratings = ratings
        self.data_dict = {}

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        user_id = self.ratings.iloc[idx]['userId']
        movie_id = self.ratings.iloc[idx]['movieId']
        rating = self.ratings.iloc[idx]['rating']

        user_feature = self.user_data[self.user_data['userId'] == user_id].values.squeeze() # 这里是一个(4,)数组
        movie_feature = self.movie_data[self.movie_data['movieId'] == movie_id].values.squeeze() # 这里是一个(2,)列表

        movieId, genre = movie_feature
        return user_feature, movieId, genre, rating

if __name__ == '__main__':
    from preprocess import data_preprocess, load_data
    def test():
        path = '../../data/ml-1m'
        dataset = MovieLensDataset(*data_preprocess(*load_data(path)))
        print()

    if __name__ == '__main__':
        test()
