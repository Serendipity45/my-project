import tensorflow as tf
import numpy as np
from pandas import DataFrame
from utils import getData, getLSTM

dataPath = '../data/merge_labeled_data.csv'
saveAdvPath = './data/merge_adv_data.csv'
modelPath = './model/LSTM.hdf5'

# FGSM对抗攻击,产生扰动
def create_adversarial_pattern(input_feature, input_label):
    # 获得网络模型
    model = getLSTM()
    model.load_weights(modelPath)

    with tf.GradientTape() as tape:
        input_feature = tf.convert_to_tensor(input_feature, tf.float32)
        tape.watch(input_feature)
        prediction = model(input_feature)
        loss = tf.keras.losses.binary_crossentropy(input_label, prediction)

    gradient = tape.gradient(loss, input_feature)
    signed_grad = (tf.sign(gradient)).numpy().reshape(input_feature.shape)

    return signed_grad

# 生成对抗样本
def FGSM():
    feature, label = getData(dataPath)

    eps = 0.1
    adv_pattern = create_adversarial_pattern(feature, label)
    attack = feature + adv_pattern * eps
    attack = np.clip(attack, 0., 1.)

    adv_attack = attack.reshape((len(feature), feature.shape[1]))

    adv_attack = np.concatenate((label, adv_attack), axis=1)
    data = DataFrame(adv_attack)

    header = ['label', 'diff_ip_num', 'port_entropy', 'all_pkt_num', 'ack_num', 'ack_rate', 'syn_num', 'syn_rate',
              'psh_num', 'psh_rate', 'all_pkt_bytes', 'avg_pkt_bytes', 'one_quarter', 'two_quarter', 'three_quarter']

    data.to_csv(saveAdvPath, header=header, index=False)
    print('成功')

if __name__ == '__main__':
    FGSM()
