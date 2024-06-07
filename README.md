# 期末报告
* 作者：Zerone
* 日期：2024年6月7日

## 目录
* 基于模型的推荐算法
    * 数据集的预处理
    * 模型架构设计
* GUI界面设计


## 数据集的预处理
* 数据集采用ml-1m
    * users.dat：用于构建用户特征矩阵
    * movies.dat：用于构建电影特征矩阵
    * ratings.dat：主要用于训练模型
* users.dat的处理
users.dat数据格式为：UserID::Gender::Age::Occupation::Zip-code
对Gender,Age都是有限的类别，故对其进行编码。Zip-code不使用，Occupation保持原数据
Gender:F用0表示，M用1表示
Age:共有7个年龄段，使用0-6七个数字来表示

* movies.dat的处理
movies.dat数据格式为：MovieID::Title::Genres
movieID和Title不处理，Genres包括该电影的类别，故采用Multi-hot编码处理
首先遍历所有电影，获得所有不重复的电影类别，这些数据以字符串的形式放在genre_list中，然后创建索引和类别一一对应的类别字典genre_int_map
```python
genre_list = []
    for val in movies_process['genre']:
        for i in str(val).split('|'):
            if i not in genre_list:
                genre_list.append(i)
genre_int_map = {val: ii for ii, val in enumerate(genre_list)}
```

使用map方法对于电影类别的每个元素进行编码处理
```python
movies_process['genre'] = movies_process['genre'].map(genres_multi_hot(genre_int_map))
```
genres_multi_hot将每个类别（字符串）以‘|’进行分割，获得对应的类别列表，再使用之前得到的类别字典一一映射到int型
```python
    def genres_multi_hot(genre_int_map):
        def helper(genres):
            genre_int_list = [genre_int_map[genre] for genre in genres.split('|')]
            multi_hot = np.zeros(len(genre_int_map))
            multi_hot[genre_int_list] = 1
            return multi_hot
    return helper
```

### 数据处理展示  
![h:200 w:800](https://github.com/Zerone-yang/Recommend/blob/main/img/image.png)  
![h:220 w:800](https://github.com/Zerone-yang/Recommend/blob/main/img/image-1.png)


### 自定义数据集
我的模型采用pytorch来构建，在训练时需要继承Dataset类重构数据集
继承Dataset需要重写其__init__，__ len__， __getitem__方法
```python
class MovieLensDataset(Dataset):
    def __init__(self, user_data, movie_data, ratings):
        self.user_data = user_data,self.movie_data = movie_data
        self.ratings = ratings, self.data_dict = {}
    def __len__(self):
        return len(self.ratings)
    def __getitem__(self, idx):
        user_id = self.ratings.iloc[idx]['userId']
        movie_id = self.ratings.iloc[idx]['movieId']
        rating = self.ratings.iloc[idx]['rating']
        user_feature = self.user_data[self.user_data['userId'] == user_id].values.squeeze() 
        movie_feature = self.movie_data[self.movie_data['movieId'] == movie_id].values.squeeze() 
        movieId, genre = movie_feature
        return user_feature, movieId, genre, rating
```

## 模型架构设计

![bg right h:600 w:500](https://github.com/Zerone-yang/Recommend/blob/main/img/%E6%A8%A1%E5%9E%8B%E6%A1%86%E6%9E%B6.png)

* 整体设计
UserNetworkFeature：根据用户数据提取用户特征矩阵，尺寸n x embed
MovieNetworkFeature：根据电影数据提取电影特征矩阵，尺寸n x embed
RecommenderNetwork：将用户特征和电影特征点积获得匹配度矩阵,n x n

* 用户特征矩阵
user_data的每个特征都是一个int类型，即（batch，1）当经过Embedding层后获得（batch,emb），然后将每个特征分别经过一个全连接层，再将四个特征拼接（cat）在一起，再经过一个全连接层，获得最终的用户特征矩阵（batch,emb）

* 电影特征矩阵
movie_data共两个特征，id为int类型，即（batch,1）genre为Multi-hot编码向量，为（batch, 18）
id经过Embeding层后接入一个全连接层，genre则直接接入全连接层，然后同样将两者拼接，经过全连接层后获得电影特征矩阵。
 
* 匹配度矩阵
U_feature: batch x emb
M_feature: batch x emb
将两者在行方向进行Softmax后分别乘以一个温度常数，之后将两者点积，获得匹配度矩阵
$$
match=
$$
$$
softmax(U_f \cdot M_f^T)
$$
 
* 模型的训练
主要使用rating.dat进行模型的训练。ratings.dat包括用户id，电影id，以及用户给出的评分。模型训练的思路：遍历每一个评分记录，根据用户id和电影id分别得到对应的user_data和movie_data，将两者放入模型中，获得匹配度矩阵。将匹配度矩阵进行一定的放大后与ratings进行比较，比较的原则是使对角元素最大的同时，让其他元素最小。根据这个想法可以使用对比学习的方法将对角元素作为正样本，其他元素作为负样本进行训练。预测评分可以看做1-5的分类任务，损失函数使用交叉熵函数
```python
n = len(outputs)
loss_u = sum([criterion(outputs[i], rating) for i in range(n)]) / n
loss_m = sum([criterion(j, rating) for j in outputs.t()]) / n
loss = (loss_m + loss_u)/2
```
 
* 训练结果
训练进行了多次，发现模型容易过拟合，并且总是倾向于输出为同一值，再进一步调小学习率后，模型训练五次得到最终结果。对于评分与rating的平均损失结果为0.2344
* 问题与改进
1. 该模型的对于电影和用户的特征提取能力太差，可以使用一些更复杂的模型结构比如Transformer或者Resnet来代替全连接
2. 由于一次训练至少要20分钟，故没有使用数据集规模更大的ml-25m，导致模型很容易过拟合
3. 对于匹配度矩阵的设计还太简陋，应该寻找相关论文使用其他方法来代替。
 
## GUI界面设计
![bg right h:600 w:600](https://github.com/Zerone-yang/Recommend/blob/main/img/image-3.png)
该界面使用tkinter设计，共有三个功能
1. 根据用户登录的id和所设定的推荐数量，推荐用户可能感兴趣的电影
2. 用户可以自主选择推荐电影的模型算法
3. 在给用户推荐电影的同时推荐相似用户
 
# 整体项目的问题和总结
1. 界面设计简陋，所选择的tkinter库很多操作不支持
2. 基于项目的模型一次推荐的时间太长
3. 推荐系统的功能太少
4. 项目代码太臃肿，需要进一步优化
