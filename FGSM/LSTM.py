from __future__ import print_function
from utils import LossHistory, getLSTM, getData
import numpy as np

batch_size = 16
epochs = 50
split = 0.9
dataPath = '../data/merge_labeled_data.csv'
modelPath = 'model/LSTM.hdf5'
pngPath = 'pic/accLoss.png'

# 创建一个实例history
history = LossHistory(pngPath)

# 获取训练集
feature, label = getData(dataPath)

# 划分训练集和验证集
train_split = int(len(feature) * split)
train_feature = feature[:train_split]
train_label = label[:train_split]
val_feature = feature[train_split:]
val_label = label[train_split:]

# 调整数据形状以匹配模型输入
train_feature = train_feature.reshape(train_feature.shape[0], 1, train_feature.shape[1])
val_feature = val_feature.reshape(val_feature.shape[0], 1, val_feature.shape[1])

# 获取LSTM网络模型
model = getLSTM()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_feature, train_label, batch_size=batch_size, epochs=epochs, shuffle=True,
          validation_data=(val_feature, val_label), callbacks=[history])

# 训练好的模型的检测效果评估
loss, accuracy = model.evaluate(val_feature, val_label)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))

# 绘制acc-loss曲线
history.loss_plot('epoch')

# 保存训练模型
model.save(modelPath)
