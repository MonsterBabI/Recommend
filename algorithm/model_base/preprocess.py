import pandas as pd
import numpy as np

def load_data(path):
    ratings_header = ('userId', 'movieId', 'rating', 'timestamp')
    ratings = pd.read_csv(path+'/'+'ratings.dat',
                          header=None,
                          names=ratings_header,
                          sep='::',
                          engine='python')
    movies_header = ('movieId', 'title', 'genre')
    movies = pd.read_csv(path + '/' + 'movies.dat',
                          header=None,
                          names=movies_header,
                          sep='::',
                          engine='python',
                          encoding='ISO-8859-1')
    # UserID::Gender::Age::Occupation::Zip - code
    users_header = ('userId', 'gender', 'age', 'occupation', 'zipcode')
    users = pd.read_csv(path + '/' + 'users.dat',
                         header=None,
                         names=users_header,
                         sep='::',
                         engine='python')
    return users, movies, ratings


def genres_multi_hot(genre_int_map):
    """
    电影类型使用multi-hot编码
    :param genre_int_map:genre到数字的映射字典
    :return:
    """

    def helper(genres):
        genre_int_list = [genre_int_map[genre] for genre in genres.split('|')]
        multi_hot = np.zeros(len(genre_int_map))
        multi_hot[genre_int_list] = 1
        return multi_hot

    return helper

def data_preprocess(users, movies, ratings):
    # F->0， M->1
    users_process = users.copy()
    users_process.replace({'gender': {'F': 0, 'M': 1},
                   'age': {1:0,18:1,25:2,35:3,45:4,50:5,56:6}},
                  inplace=True)
    users_process = users_process.drop('zipcode', axis=1)
    ratings_process = ratings.copy()
    ratings_process = ratings_process.drop('timestamp', axis=1)
    movies_process = movies.copy()
    genre_list = []
    for val in movies_process['genre']:
        for i in str(val).split('|'):
            if i not in genre_list:
                genre_list.append(i)
    genre_int_map = {val: ii for ii, val in enumerate(genre_list)}
    movies_process['genre'] = movies_process['genre'].map(genres_multi_hot(genre_int_map))
    movies_process = movies_process.drop('title', axis=1)
    return users_process, movies_process, ratings_process


def test():
    path = '../../data/ml-1m'
    data_preprocess(*load_data(path))

if __name__ == '__main__':
    test()
