from tkinter import *
from tkinter import ttk
from algorithm.item_based import Item_base_rec
from algorithm.user_base import UserCFRec
from algorithm.model_rec import model_based_rec
import os

class RecommendInterface:
    def __init__(self, init_window, data_path, item_path=None, model_path=None):
        print('初始化')
        self.init_window = init_window
        self.data_path = data_path
        self.irec = Item_base_rec(data_path)
        self.urec = UserCFRec(data_path)
        self.item_path = item_path + '/' + os.listdir(item_path)[0]
        self.irec.train(self.item_path)
        self.movie_network_path = model_path + '/' + 'movie_network.pt'
        self.user_network_path = model_path + '/' + 'user_network.pt'
        self.model_based_rec = model_based_rec(data_path, self.movie_network_path, self.user_network_path)
        print("初始化完成")

    # 设计界面
    def set_init_window(self):
        self.init_window.title('电影推荐系统_v1')
        self.init_window.geometry('1024x640+50-100')

        # 设置样式
        style = ttk.Style()
        style.configure("TLabel", font=("Helvetica", 12))
        style.configure("TButton", font=("Helvetica", 12))
        style.configure("TCombobox", font=("Helvetica", 12))

        # 用户信息框架
        user_frame = ttk.Frame(self.init_window)
        user_frame.grid(row=0, column=0, padx=20, pady=20, sticky=W)

        # 用户ID标签和输入框
        self.user_id_label = ttk.Label(user_frame, text="用户ID:")
        self.user_id_label.grid(row=0, column=0, padx=5, pady=5, sticky=W)
        self.user_id_entry = ttk.Entry(user_frame)
        self.user_id_entry.grid(row=0, column=1, padx=5, pady=5, sticky=W)

        # 推荐数量标签和输入框
        self.num_recommend_label = ttk.Label(user_frame, text="推荐数量:")
        self.num_recommend_label.grid(row=1, column=0, padx=5, pady=5, sticky=W)
        self.num_recommend_entry = ttk.Entry(user_frame)
        self.num_recommend_entry.grid(row=1, column=1, padx=5, pady=5, sticky=W)

        # 模型选择
        self.model_label = ttk.Label(user_frame, text="选择模型:")
        self.model_label.grid(row=2, column=0, padx=5, pady=5, sticky=W)
        self.model_var = StringVar(value="User-based")
        self.model_combobox = ttk.Combobox(user_frame, textvariable=self.model_var)
        self.model_combobox['values'] = ('User-based', 'Model-based', 'Item-based')
        self.model_combobox.grid(row=2, column=1, padx=5, pady=5, sticky=W)

        # 推荐按钮
        self.recommend_button = ttk.Button(user_frame, text='推荐', command=self.recommend_movie)
        self.recommend_button.grid(row=3, column=0, columnspan=2, padx=5, pady=15, sticky=W)

        # 推荐电影列表
        self.recommend_movie_text = Text(self.init_window, height=20, width=80)
        self.recommend_movie_text.grid(row=1, column=0, padx=20, pady=20, sticky=NSEW)

        # 让文本框自动换行
        self.recommend_movie_text.config(wrap=WORD)

        # 推荐认识的人框架
        self.similar_people_frame = ttk.Frame(self.init_window)
        self.similar_people_frame.grid(row=0, column=1, padx=20, pady=20, sticky=NE)

        # 推荐认识的人的按钮和文本框
        # self.similar_people_button = ttk.Button(self.similar_people_frame, text="推荐认识的人", command=self.recommend_movie)
        # self.similar_people_button.grid(row=0, column=0, padx=5, pady=5, sticky=W)

        self.similar_people_text = Text(self.similar_people_frame, height=20, width=40)
        self.similar_people_text.grid(row=1, column=0, padx=5, pady=5, sticky=NSEW)

        # 窗口网格配置
        self.init_window.grid_rowconfigure(1, weight=1)
        self.init_window.grid_columnconfigure(0, weight=1)

    # 下面这个函数用来接入算法
    def recommend_movie(self):
        self.recommend_movie_text.delete(1.0, END)
        self.similar_people_text.delete(1.0, END)
        user_id = self.user_id_entry.get().strip()
        num_recommend = self.num_recommend_entry.get().strip()

        if not user_id.isdigit():
            self.recommend_movie_text.insert(1.0, "请输入有效的用户ID（数字）")
            return

        if not num_recommend.isdigit():
            self.recommend_movie_text.insert(1.0, "请输入有效的推荐数量（数字）")
            return

        user_id = int(user_id)
        num_recommend = int(num_recommend)
        selected_model = self.model_var.get()
        movies_similar_people = self.urec.recommend(user_id)
        if selected_model == "Item-based":
            movies = self.irec.recommend(user_id, num_recommend)
        elif selected_model == 'User-based':
            movies = movies_similar_people[0]
        else:
            movies = self.model_based_rec.recommend(userId=user_id, n=num_recommend)
        self.recommend_movie_text.insert(1.0, "推荐的电影: \n")
        for movie in movies:
            self.recommend_movie_text.insert(END, f"- {movie}\n")
        self.similar_people_text.insert(1.0, "推荐认识的人: \n")
        similar_people = movies_similar_people[1]
        for person in similar_people:
            self.similar_people_text.insert(END, f"- {person}\n")

def debug_gui():
    init_window = Tk()
    path = '../data/ml-1m/'
    item_path = '../model/item_based_model'
    model_path = '../model/model_based_model'
    test_class = RecommendInterface(init_window, path, item_path, model_path)
    test_class.set_init_window()
    init_window.mainloop()

if __name__ == '__main__':
    debug_gui()
