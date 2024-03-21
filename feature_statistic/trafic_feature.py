import numpy as np
import pandas as pd
import math
from features.load_pcap import feature_extractor

'''
Time:20220512
Descripition:一定时间内数据包的特征统计
'''


class feature_statistic(object):
    """
    @:param:diff_ip_num--外连主机数
    @:param:port_entropy--端口信息熵
    @:param:all_pkt_num--包总数
    @:param:ack_num--ack为1的包数量
    @:param:ack_rate--ack为1的包比例
    @:param:syn_num--syn为1的包数量
    @:param:syn_rate--syn为1的包比例
    @:param:psh_num--psh为1的包数量
    @:param:psh_rate--psh为1的包比例
    @:param:all_pkt_bytes--所有包总字节数
    @:param:avg_pkt_bytes--平均字节数
    @:param:one_quarter--包总字节数的25%分位
    @:param:two_quarter--包总字节数的50%分位
    @:param:three_quarter-- 包总字节数的75%分位
    """

    def __init__(self):
        # 某含有时间间隔t内数据包特征的二维数组
        self.pkt_array = None
        # 以下为要统计的特征初始化
        self.diff_ip_num = -1
        self.port_entropy = -1
        self.all_pkt_num = -1
        self.ack_num = -1
        self.ack_rate = -1
        self.syn_num = -1
        self.syn_rate = -1
        self.psh_num = -1
        self.psh_rate = -1
        self.all_pkt_bytes = -1
        self.avg_pkt_bytes = -1
        self.one_quarter = -1
        self.two_quarter = -1
        self.three_quarter = -1
        # 特征对应的列表
        self.diff_ip_list = []
        self.port_information_entropy_list = []
        self.all_pkt_num_list = []
        self.ack_num_list = []
        self.ack_rate_list = []
        self.syn_num_list = []
        self.syn_rate_list = []
        self.psh_num_list = []
        self.psh_rate_list = []
        self.all_pkt_bytes_list = []
        self.avg_pkt_bytes_list = []
        self.avg_bytes_percentile_one_list = []
        self.avg_bytes_percentile_two_list = []
        self.avg_bytes_percentile_three_list = []

    # srcIP和dstIP相异的数量（外联主机数，不区分源IP与目的IP）
    def __diff_ip_statistic(self):
        src_ip_array = self.pkt_array[:, 0]
        dst_ip_array = self.pkt_array[:, 1]

        src_ip_set = set(src_ip_array)
        dst_ip_set = set(dst_ip_array)

        diff_ip_set = src_ip_set.union(dst_ip_set)
        self.diff_ip_num = len(diff_ip_set)

    # 端口信息熵（不区分源端口与目的端口）
    def __port_information_entropy_statistic(self):
        port_dict = {}
        probability = []
        src_port_array = self.pkt_array[:, 2]
        dst_port_array = self.pkt_array[:, 3]

        src_port_set = set(src_port_array)
        dst_port_set = set(dst_port_array)
        all_port_set = src_port_set.union(dst_port_set)
        array_len = len(src_port_array)
        all_port_num = array_len * 2

        for element in iter(all_port_set):
            port_dict.update({element: 0})

        for i in range(array_len):
            key_src = src_port_array[i]
            key_dst = dst_port_array[i]
            port_dict[key_src] += 1
            port_dict[key_dst] += 1

        # 计算每个端口的概率
        for value in port_dict.values():
            probability.append(value / all_port_num)
        # 计算信息熵
        self.__information_entropy(probability)

    # 信息熵计算函数
    def __information_entropy(self, probability):
        entropy = 0
        for p in probability:
            entropy += math.log(p, 2)
        self.port_entropy = -entropy

    # 总包数统计
    def __all_pkt_num_statistic(self):
        self.all_pkt_num = len(self.pkt_array)

    # ack个数以及置1的包所占比率
    def __ack_num_statistic(self):
        ack_array = self.pkt_array[:, 4]
        ack_num = 0
        for i in ack_array:
            if i == '1':
                ack_num += 1
            else:
                pass
        self.ack_num = ack_num
        self.ack_rate = ack_num / self.all_pkt_num

    # syn个数以及置1的包所占的比率
    def __syn_num_statistic(self):
        syn_array = self.pkt_array[:, 5]
        syn_num = 0
        for i in syn_array:
            if i == '1':
                syn_num += 1
            else:
                pass
        self.syn_num = syn_num
        self.syn_rate = syn_num / self.all_pkt_num

    # psh个数以及置1的包所占的比率
    def __psh_num_statistic(self):
        psh_array = self.pkt_array[:, 6]
        psh_num = 0
        for i in psh_array:
            if i == '1':
                psh_num += 1
            else:
                pass
        self.psh_num = psh_num
        self.psh_rate = psh_num / self.all_pkt_num

    # 窗口时间内总字节数
    def __all_pkt_bytes_statistic(self):
        all_pkt_bytes = 0
        pkt_bytes_array = self.pkt_array[:, 7]
        for pkt_bytes in pkt_bytes_array:
            all_pkt_bytes += int(pkt_bytes)
        self.all_pkt_bytes = all_pkt_bytes

    # 窗口时间内的平均字节数
    def __avg_pkt_bytes_statistic(self):
        self.avg_pkt_bytes = self.all_pkt_bytes / self.all_pkt_num

    # 平均字节数的25%、50%以及75%分位数
    def __avg_bytes_percentile_statistic(self):
        pkt_bytes_array = self.pkt_array[:, 7]
        new_pkt_bytes_array = []
        # 字符转数字
        for i in pkt_bytes_array:
            new_pkt_bytes_array.append(int(i))
        self.one_quarter = np.percentile(new_pkt_bytes_array, 25)
        self.two_quarter = np.percentile(new_pkt_bytes_array, 50)
        self.three_quarter = np.percentile(new_pkt_bytes_array, 75)

    # 接收到的上游数据
    def recv_pkt_array(self, pkt_array_src):
        self.pkt_array = pkt_array_src

    # 每个时间块的数加入列表
    def __write_to_list(self):
        self.diff_ip_list.append(self.diff_ip_num)
        self.port_information_entropy_list.append(self.port_entropy)
        self.all_pkt_num_list.append(self.all_pkt_num)
        self.ack_num_list.append(self.ack_num)
        self.ack_rate_list.append(self.ack_rate)
        self.syn_num_list.append(self.syn_num)
        self.syn_rate_list.append(self.syn_rate)
        self.psh_num_list.append(self.psh_num)
        self.psh_rate_list.append(self.psh_rate)
        self.all_pkt_bytes_list.append(self.all_pkt_bytes)
        self.avg_pkt_bytes_list.append(self.avg_pkt_bytes)
        self.avg_bytes_percentile_one_list.append(self.one_quarter)
        self.avg_bytes_percentile_two_list.append(self.two_quarter)
        self.avg_bytes_percentile_three_list.append(self.three_quarter)

    # 写入数据文件
    def write_to_csv(self, _output_file_path):
        _data_frame = pd.DataFrame({
            'diff_ip_num': self.diff_ip_list,
            'port_entropy': self.port_information_entropy_list,
            'all_pkt_num': self.all_pkt_num_list,
            'ack_num': self.ack_num_list,
            'ack_rate': self.ack_rate_list,
            'syn_num': self.syn_num_list,
            'syn_rate': self.syn_rate_list,
            'psh_num': self.psh_num_list,
            'psh_rate': self.psh_rate_list,
            'all_pkt_bytes': self.all_pkt_bytes_list,
            'avg_pkt_bytes': self.avg_pkt_bytes_list,
            'one_quarter': self.avg_bytes_percentile_one_list,
            'two_quarter': self.avg_bytes_percentile_two_list,
            'three_quarter': self.avg_bytes_percentile_three_list})
        _data_frame.to_csv(_output_file_path, index=False, sep=',', encoding='utf-8')

    # 特征统计处理主函数
    def main(self):
        self.__diff_ip_statistic()
        self.__port_information_entropy_statistic()
        self.__all_pkt_num_statistic()
        self.__ack_num_statistic()
        self.__syn_num_statistic()
        self.__psh_num_statistic()
        self.__all_pkt_bytes_statistic()
        self.__avg_pkt_bytes_statistic()
        self.__avg_bytes_percentile_statistic()
        self.__write_to_list()


if __name__ == '__main__':
    # 统计特征
    fs_obj = feature_statistic()
    # 创建输出文件（.csv）
    data_frame = pd.DataFrame({
        'diff_ip_num': [],
        'port_entropy': [],
        'all_pkt_num': [],
        'ack_num': [],
        'ack_rate': [],
        'syn_num': [],
        'syn_rate': [],
        'psh_num': [],
        'psh_rate': [],
        'all_pkt_bytes': [],
        'avg_pkt_bytes': [],
        'one_quarter': [],
        'two_quarter': [],
        'three_quarter': []})
    # 调用时间窗口函数接口 时间为秒级别
    src_pkt_array = feature_extractor('../data/malicious_data', 5)
    for data_block in src_pkt_array:
        fs_obj.recv_pkt_array(np.array(data_block))
        fs_obj.main()
    # 输出数据文件
    fs_obj.write_to_csv('../data/output.csv')
