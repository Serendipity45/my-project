# -*- coding: utf-8 -*-
# @Time    : 2022/5/12 10:06
# @Author  : liujun
# @FileName: load_pcap.py
import socket
from itertools import groupby

import dpkt

from features.Utils.fileUtils import get_file_path


def translate_ip(ip):
    """
    transfer IP(IPv6) address in bin format to string format.
    :param ip:
    :return:
    """
    try:
        return socket.inet_ntop(socket.AF_INET, ip)
    except ValueError:
        return socket.inet_ntop(socket.AF_INET6, ip)


def get_flow_id(pkt_buf):
    """
    get flow id from a raw dpkt packet
    :param pkt_buf: dpkt unpack buf
    :return: return flow id in string, or return None on failure
    """
    eth = dpkt.ethernet.Ethernet(pkt_buf)
    ip = eth.data
    if not isinstance(ip, dpkt.ip.IP) and not isinstance(ip, dpkt.ip6.IP6):
        return None
    tcp_udp = ip.data
    if not isinstance(tcp_udp, dpkt.tcp.TCP) and not isinstance(tcp_udp, dpkt.udp.UDP):
        return None

    src_ip = translate_ip(ip.src)
    dst_ip = translate_ip(ip.dst)
    src_port = tcp_udp.sport
    dst_port = tcp_udp.dport
    protocol = str(ip.p)

    return str(src_ip) + "-" + str(dst_ip) + "-" + str(src_port) + "-" + str(dst_port) + "-" + protocol


def parsing_packet(timestamp, pkt):
    """
    :param timestamp: timestamp
    :param pkt: dpkt packet
    :return: packet feature :list(timestamp, srcIP. dstIP, srcPort, dstPort, Ack, Syn, Psh, bytes)
    """
    feature = []
    eth = dpkt.ethernet.Ethernet(pkt)
    ip = eth.data

    if not isinstance(ip, dpkt.ip.IP) and not isinstance(ip, dpkt.ip6.IP6):
        return None
    trans = ip.data
    if not isinstance(trans, dpkt.tcp.TCP) and not isinstance(trans, dpkt.udp.UDP):
        return None
    src_ip = translate_ip(ip.src)
    dst_ip = translate_ip(ip.dst)
    src_port = trans.sport
    dst_port = trans.dport
    pkt_len = len(pkt)
    if isinstance(ip.data, dpkt.tcp.TCP):
        tcp_ack = 1 if trans.flags & dpkt.tcp.TH_ACK != 0 else 0
        tcp_syn = 1 if trans.flags & dpkt.tcp.TH_SYN != 0 else 0
        tcp_psh = 1 if trans.flags & dpkt.tcp.TH_PUSH != 0 else 0
    else:
        tcp_ack = 0
        tcp_syn = 0
        tcp_psh = 0

    feature.append(timestamp)
    feature.append(src_ip)
    feature.append(dst_ip)
    feature.append(src_port)
    feature.append(dst_port)
    feature.append(tcp_ack)
    feature.append(tcp_syn)
    feature.append(tcp_psh)
    feature.append(pkt_len)
    return feature


def feature_extractor(pcap_file_list, time_limit):
    """
    :param time_limit: time threshold
####AA    param pcap_file_list: only pcap file
    :return: feature[[]]
    """
    feature = []
    opened_pcap_files = [dpkt.pcap.Reader(open(file, "rb")) for file in get_file_path(pcap_file_list, '.pcap')]
    for pcap_file in opened_pcap_files:
        for ts, buf in pcap_file:
            timestamp = int(ts / time_limit)
            if parsing_packet(timestamp, buf) is not None:
                feature.append(parsing_packet(timestamp, buf))
    lstg = groupby(feature, lambda x: x[0])
    result = []
    for key, group in lstg:
        temp_list = list(map(lambda x: x[1:], list(group)))
        result.append(temp_list)

    return result


if __name__ == '__main__':
    print(feature_extractor('data', 1))
