# -*- coding: utf-8 -*-
# @Time    : 2022/5/16 16:30
# @Author  : liujun
# @First Modified: 2022/5/18
# @Modified :LiYao
# @FileName: data_process.py

from sklearn import preprocessing
from sklearn.utils import shuffle

from feature_statistic.trafic_feature import feature_statistic
from features.load_pcap import feature_extractor
import numpy as np
import pandas as pd


class data_pre_process(object):

    def __init__(self, benign_path, mal_path, time_threshold=1):
        self.data = None
        self.benign_pcap_path = benign_path
        self.malicious_pcap_path = mal_path
        self.benign_data_path = '../data/benign_data.csv'
        self.malicious_data_path = '../data/malicious_data.csv'
        self.time_threshold = time_threshold

    def update_benign_pcap_path(self, _benign_pcap_path):
        self.benign_pcap_path = _benign_pcap_path

    def update_malicious_pcap_path(self, _malicious_pcap_path):
        self.malicious_pcap_path = _malicious_pcap_path

    def update_benign_data_path(self, _benign_data_path):
        self.benign_data_path = _benign_data_path

    def update_malicious_data_path(self, _malicious_data_path):
        self.malicious_data_path = _malicious_data_path

    def update_time_threshold(self, _time_threshold):
        self.time_threshold = _time_threshold

    def benign_data(self):
        fs_obj = feature_statistic()
        src_pkt_array = feature_extractor(self.benign_pcap_path, self.time_threshold)
        for data_block in src_pkt_array:
            fs_obj.recv_pkt_array(np.array(data_block))
            fs_obj.main()
        fs_obj.write_to_csv(self.benign_data_path)
        return fs_obj

    def malicious_data(self):
        _fs_obj = feature_statistic()
        _src_pkt_array = feature_extractor(self.malicious_pcap_path, self.time_threshold)
        for _data_block in _src_pkt_array:
            _fs_obj.recv_pkt_array(np.array(_data_block))
            _fs_obj.main()
        _fs_obj.write_to_csv(self.malicious_data_path)
        return _fs_obj

    def merge_data(self):
        benign_data = pd.read_csv(self.benign_data_path)
        malicious_data = pd.read_csv(self.malicious_data_path)

        self.data = benign_data.append(malicious_data, ignore_index=True)
        # 数据归一化
        Scaler_data = preprocessing.MinMaxScaler()
        self.data = Scaler_data.fit_transform(self.data)

        # 输出合并后的csv文件
        diff_ip_list = self.data[:, 0]
        port_information_entropy_list = self.data[:, 1]
        all_pkt_num_list = self.data[:, 2]
        ack_num_list = self.data[:, 3]
        ack_rate_list = self.data[:, 4]
        syn_num_list = self.data[:, 5]
        syn_rate_list = self.data[:, 6]
        psh_num_list = self.data[:, 7]
        psh_rate_list = self.data[:, 8]
        all_pkt_bytes_list = self.data[:, 9]
        avg_pkt_bytes_list = self.data[:, 10]
        avg_bytes_percentile_one_list = self.data[:, 11]
        avg_bytes_percentile_two_list = self.data[:, 12]
        avg_bytes_percentile_three_list = self.data[:, 13]

        label_list = []
        for i in range(0, len(benign_data)):  # 给数据打标签
            label_list.append(0)
        for j in range(0, len(malicious_data)):
            label_list.append(1)

        _data_frame = pd.DataFrame({
            'label': label_list,
            'diff_ip_num': diff_ip_list,
            'port_entropy': port_information_entropy_list,
            'all_pkt_num': all_pkt_num_list,
            'ack_num': ack_num_list,
            'ack_rate': ack_rate_list,
            'syn_num': syn_num_list,
            'syn_rate': syn_rate_list,
            'psh_num': psh_num_list,
            'psh_rate': psh_rate_list,
            'all_pkt_bytes': all_pkt_bytes_list,
            'avg_pkt_bytes': avg_pkt_bytes_list,
            'one_quarter': avg_bytes_percentile_one_list,
            'two_quarter': avg_bytes_percentile_two_list,
            'three_quarter': avg_bytes_percentile_three_list})
        _data_frame = shuffle(_data_frame)
        _data_frame.to_csv('../data/merge_labeled_data.csv', index=False, sep=',', encoding='utf-8')

    # def main(self,):


if __name__ == '__main__':
    
    # 修改此处的路径参数为实际文件路径
    benign_path = '../data/benign_pcap_folder'
    malicious_path = '../data/malicious_pcap_folder'
    dp = data_pre_process(benign_path, malicious_path, 1)
    
    
    dp.benign_data()
    dp.malicious_data()
    dp.merge_data()
    pass
