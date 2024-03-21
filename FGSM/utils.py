import random
import numpy as np
import pandas as pd
import itertools
from keras import Sequential
from keras.layers import Dropout, Dense, LSTM
from matplotlib import pyplot as plt
from tensorflow import keras


# 获取测试数据
def getData(path):
    # 读取数据
    data = pd.read_csv(path)
    data = np.array(data)

    # 打乱顺序
    index = [i for i in range(len(data))]
    random.shuffle(index)
    data = data[index]

    # 特征与标签分离
    feature = data[:, 1:]
    label = data[:, :1]

    feature = np.reshape(feature, (feature.shape[0], 1, feature.shape[1]))

    return feature, label


# 返回LSTM网络模型
def getLSTM():
    # 定义LSTM网络
    model = Sequential()
    model.add(LSTM(32, activation='relu', input_dim=14, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64, activation='relu', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    # print(model.summary())
    return model


# LossHistory类，保存loss和acc
class LossHistory(keras.callbacks.Callback):

    def __init__(self, path):
        self.val_acc = None
        self.val_loss = None
        self.accuracy = None
        self.losses = None
        self.path = path

    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('accuracy'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_accuracy'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('accuracy'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_accuracy'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.savefig(self.path)
        plt.show()


# 画图，模型检测效果可视化
class Plot:
    def __init__(self, name):
        self.path = name

    # 画混淆矩阵
    def plot_confusion_matrix(self, cm, classes):
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('confusion matrix')
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j], horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(self.path + 'confusion_matrix.png', bbox_inches='tight')
        plt.show()

    # 画准确率、精确率、回归率、f1
    def plot_valuation(self, num_list):
        rects = plt.bar(range(len(num_list)), num_list)
        plt.ylim(ymax=110, ymin=0)
        plt.xticks([0, 1, 2, 3], ['accuracy', 'precision', 'recall', 'f1'])
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2, height, str(round(height, 2)) + '%', ha='center', va='bottom')
        # ax = plt.gca()
        # ax.spines['top'].set_color('none')
        # ax.spines['right'].set_color('none')
        plt.savefig(self.path + 'valuation.png', bbox_inches='tight')
        plt.show()

    # 画ROC
    def plot_roc(self, fpr, tpr, roc_auc):
        plt.figure()
        lw = 2
        plt.figure()
        # 假正率为横坐标，真正率为纵坐标做曲线
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.savefig(self.path + 'ROC.png', bbox_inches='tight')
        plt.show()
