# -*- coding: utf-8 -*-
# @Time    : 2022/5/18
# @Author  : LiYao
# @FileName: randomForest_model.py
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from utils.data_process import data_pre_process
from sklearn.model_selection import RandomizedSearchCV
import argparse


class RandomForest(object):
    def __init__(self, load_model_path):
        self.train_data = None
        self.test_data = None
        self.file_path = './data/merge_labeled_data.csv'
        self.mode_path = './model_save/rf.pickle'
        self.load_model_path = load_model_path

    # 读取已标记的数据集文件
    def __load_data_file(self):
        self.data_src = pd.read_csv(self.file_path)

    # 数据预处理
    def __data_pre_process(self):
        # 将数据与标签分离，并且转为np.array形式
        self.labels = np.array(self.data_src['label'])
        self.data = self.data_src.drop('label', axis=1)
        self.data = np.array(self.data)
        # 划分数据集
        self.train_x, self.test_x, self.train_y, self.test_y \
            = train_test_split(self.data, self.labels, test_size=0.25,
                               random_state=0)

    # 模型训练及保存主函数
    def __random_forest_main_execute_function(self):
        # 训练模型并给出预测评分
        self.rfc = RandomForestClassifier(n_estimators=100, max_depth=100, random_state=1)
        self.rfc = self.rfc.fit(self.train_x, self.train_y)
        score_train = self.rfc.score(self.train_x, self.train_y)
        score_test = self.rfc.score(self.test_x, self.test_y)
        print("初始随机森林训练集精确度:", format(score_train))
        print("初始随机森林测试集精确度:", format(score_test))
        print('-----------------')
        # 调参(备用)
        # self.__grid_search_function()
        self.__save_model()

    # 网格搜索寻找最优超参数
    def __grid_search_function(self):
        # 设置随机搜索网格参数
        n_estimators = np.arange(10, 200, step=5)
        max_features = ["auto", "sqrt", "log2"]
        max_depth = list(np.arange(4, 100, step=2)) + [None]
        min_samples_split = np.arange(2, 10, step=2)
        min_samples_leaf = [1, 2, 4]
        bootstrap = [True, False]

        param_grid = {
            "n_estimators": n_estimators,
            "max_features": max_features,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "bootstrap": bootstrap,
        }

        # 随机搜索并进行10折交叉验证
        random_cv = RandomizedSearchCV(
            self.rfc, param_grid, n_iter=1000, cv=10, scoring="r2", n_jobs=-1
        )
        self.rfc = random_cv.fit(self.train_x, self.train_y)
        print("最佳参数:\n")
        print(random_cv.best_params_)

    # 使用模型进行数据预测
    def __data_predict(self):
        # 使用随机森林带的预测方法进行预测
        _clf = self.__load_model()
        score_r = _clf.score(self.test_x, self.test_y)
        print("模型准确率", format(score_r))
        # 以下方法也可以计算准确率
        '''
        predictions = _clf.predict(self.test_x)
        # 计算绝对误差
        errors = abs(predictions - self.test_y)
        # 如果error是1，则预测错误
        error_num = 0
        for error in errors:
            if error == 1:
                error_num += 1
        accuracy = 1 - error_num / len(predictions)
        print('模型准确率:', round(accuracy, 2))
        '''


    # 保存模型
    def __save_model(self):
        with open(self.mode_path, 'wb') as f:
            pickle.dump(self.rfc, f)

    # 加载模型
    def __load_model(self):
        with open(self.load_model_path, 'rb') as f:
            _clf = pickle.load(f)
        return _clf

    # 训练执行函数
    def train_main(self):
        self.__load_data_file()
        self.__data_pre_process()
        self.__random_forest_main_execute_function()

    # 预测执行函数
    def predict_main(self):
        self.__load_data_file()
        self.__data_pre_process()
        self.__data_predict()


# 20220518 ly test
if __name__ == '__main__':
    '''
    # 测试
    # 命令行参数
    parser = argparse.ArgumentParser(description='输入参数')
    parser.add_argument('--benign_path', dest='benign', type=str, help='良性pcap路径')
    parser.add_argument('--mal_path', dest='malware', type=str, help='恶意pcap路径')
    parser.add_argument('--model_path', dest='model', type=str, help='模型路径')
    # 0为训练，1为预测
    parser.add_argument('--execute_type', dest='execute_type', type=int, help='训练模型/预测')
    args = parser.parse_args()

    benign_path = args.benign
    mal_path = args.malware
    model_path = args.model
    execute_type = args.execute_type
    '''

    # test file path
    benign_path = './data/benign'
    mal_path = './data/malicious'

    execute_type = 1
    model_path = './model_save/rf.pickle'

    # 模型测试
    if execute_type == 0:
        # dp = data_pre_process(benign_path, mal_path)
        # dp.benign_data()
        # dp.malicious_data()
        # dp.merge_data()

        model_path = None
        rf_object = RandomForest(model_path)
        rf_object.train_main()
    elif execute_type == 1:
        rf_object = RandomForest(model_path)
        rf_object.predict_main()
