import numpy as np
import matplotlib.pyplot as plt
import copy as copy
import networkx as nx
from scipy.optimize import nnls
import csv
import sys
import os
import shutil
import argparse
from datetime import datetime
import random


def ensure_directory(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"创建目录: {directory}")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='网络模拟器 - 生成链路和路径时延数据')
    parser.add_argument('--experiment_dir', type=str, default=None,
                        help='实验结果保存目录，如果提供，将同时将输出保存到该目录')
    # 添加是否随机生成故障链路的参数 (store_true类型的参数，默认为False，提供时为True)
    parser.add_argument('--random_faults', action='store_true',
                        help='是否随机选择故障链路位置')
    # 添加随机种子参数，用于控制随机性
    parser.add_argument('--random_seed', type=int, default=None,
                        help='随机数生成器种子，用于复现实验')
    # 添加故障链路比例的参数
    parser.add_argument('--external_fault_ratio', type=float, default=0.15,
                        help='外部流量致拥链路占总链路比例，默认0.15')
    parser.add_argument('--bandwidth_fault_ratio', type=float, default=0.1,
                        help='带宽减小致拥链路占总链路比例，默认0.1')
    # 添加直接指定故障链路的参数
    parser.add_argument('--external_fault_links', type=str, default=None,
                        help='外部流量致拥链路ID列表，逗号分隔，例如"2,5,8,18,20"')
    parser.add_argument('--bandwidth_fault_links', type=str, default=None,
                        help='带宽减小致拥链路ID列表，逗号分隔，例如"10,17,24"')

    args = parser.parse_args()

    # 如果提供了随机种子，则固定随机数生成
    if args.random_seed is not None:
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)

    # 如果提供了实验目录，确保目录存在
    experiment_dir = args.experiment_dir
    if experiment_dir is None:
        # 如果没有提供实验目录，使用默认的时间戳目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = f"./experiment_results/{timestamp}"

    if experiment_dir:
        ensure_directory(experiment_dir)
        # 创建子目录
        ensure_directory(f"{experiment_dir}/output")
        ensure_directory(f"{experiment_dir}/path_variance")
        print(f"实验结果将同时保存到: {experiment_dir}")

    # 确保原始输出目录存在
    ensure_directory('./output')
    ensure_directory('./path_variance')

    file_path = './output/route_matrix.txt'
    output_path = './out/'

    # 读取文件
    try:
        R = np.loadtxt(file_path, delimiter='\t')
        print("Loaded route matrix")
        # print(R)
    except FileNotFoundError as e:
        print(f"File read error:{e}")
        # 如果文件不存在，则生成新的路由矩阵
        print("生成新的路由矩阵...")
        # 这里可以添加路由矩阵生成代码，或从其他位置复制
        R = np.zeros((10, 36))  # 示例矩阵，根据需要调整
        for i in range(10):
            for j in range(36):
                if np.random.rand() < 0.3:  # 30%的概率为1
                    R[i, j] = 1

        # 保存新生成的路由矩阵
        np.savetxt(file_path, R, delimiter='\t')
        print(f"新的路由矩阵已保存到 {file_path}")

    # 由路由矩阵构建出网络拓扑
    num_paths = R.shape[0]  # 获取路径数
    num_links = R.shape[1]  # 获取链路数
    # 创建链路和目标节点列表，链路和路径均从 0 开始编号
    node_id, dest_marked = 0, [[i, 0] for i in range(num_paths)]  # 初始化节点ID计数器和目标标记列表
    edge_list, edge_marked = [], [[i, 0] for i in range(num_links)]  # 初始化边列表和边标记列表
    # dest_marked[0] = [1,3] 代表第一条路径的目标节点是节点3

    for i in range(num_paths):
        link_src = 0  # 每个新路径开始时，源节点重置为0

        for j in range(num_links):
            if not edge_marked[j][1] and R[i, j]:
                # 如果当前链路未被标记且路径-链路矩阵中对应项为真
                node_id += 1  # 新增节点
                link_dest = node_id  # 将新增节点作为新增边的目的节点

                edge_list.append((link_src, link_dest, {'label': j}))  # 添加边及其标签
                edge_marked[j][1] = len(edge_list)  # 标记边信息存放的位置

                link_src = link_dest  # 更新源节点

            elif edge_marked[j][1] and R[i, j]:
                # 如果当前链路已被标记且路径-链路矩阵中对应项为真
                link_src = edge_list[edge_marked[j][1] - 1][1]  # 更新源节点

        dest_marked[i][1] = link_src  # 记录路径的目的节点

    assert all(edge_marked)  # 确保所有边都被标记了
    assert all(dest_marked)  # 确保所有路径的目的节点都被标记了

    # 以链路列表信息，创建一个有向树图
    T = nx.DiGraph()
    T.add_edges_from(edge_list)
    assert nx.is_tree(T);  # 检查是否为树图

    # =============================
    # 确定故障链路
    # =============================
    # 检查是否直接指定了故障链路
    if args.external_fault_links is not None and args.bandwidth_fault_links is not None:
        # 使用指定的故障链路
        external_fault_links = [int(idx) for idx in args.external_fault_links.split(',') if idx.strip()]
        bandwidth_fault_links = [int(idx) for idx in args.bandwidth_fault_links.split(',') if idx.strip()]
        print("使用指定的故障链路")
        print(f"外部流量致拥链路: {external_fault_links}")
        print(f"带宽减小致拥链路: {bandwidth_fault_links}")
    elif args.random_faults:
        # 随机选择故障链路
        # 计算每种类型的链路数量
        num_external_fault = max(1, int(num_links * args.external_fault_ratio))
        num_bandwidth_fault = max(1, int(num_links * args.bandwidth_fault_ratio))

        # 确保故障链路数量不超过总链路数
        if num_external_fault + num_bandwidth_fault > num_links * 0.5:
            # 限制故障链路总数不超过链路总数的50%
            total_faults = int(num_links * 0.5)
            # 按原来的比例分配
            ratio_sum = args.external_fault_ratio + args.bandwidth_fault_ratio
            num_external_fault = max(1, int(total_faults * args.external_fault_ratio / ratio_sum))
            num_bandwidth_fault = max(1, total_faults - num_external_fault)

        print(f"随机选择 {num_external_fault} 条外部流量致拥链路和 {num_bandwidth_fault} 条带宽减小致拥链路")

        # 随机选择链路作为故障链路
        all_links = list(range(num_links))
        random.shuffle(all_links)

        external_fault_links = all_links[:num_external_fault]
        bandwidth_fault_links = all_links[num_external_fault:num_external_fault + num_bandwidth_fault]

        print(f"外部流量致拥链路: {external_fault_links}")
        print(f"带宽减小致拥链路: {bandwidth_fault_links}")
    else:
        # 使用默认的固定故障链路
        external_fault_links = [2, 5, 8, 18, 20]
        bandwidth_fault_links = [10, 17, 24]
        print("使用预定义的故障链路")
        print(f"外部流量致拥链路: {external_fault_links}")
        print(f"带宽减小致拥链路: {bandwidth_fault_links}")

    # =============================
    # 生成网络中每条边的延迟
    # =============================
    # =============================
    # 生成网络中每条边的延迟
    # =============================
    link_delays_true = {}
    num_samples = 1000

    # 对数正态分布参数 - 均值和标准差
    default_mu = 3.0  # 正常链路的对数均值参数
    default_sigma = 0.2  # 正常链路的对数标准差参数

    external_mu = 3.2  # 外部流量致拥链路的对数均值参数
    external_sigma = 0.4  # 外部流量致拥链路的对数标准差参数

    bandwidth_mu = 3.4  # 带宽减小致拥链路的对数均值参数
    bandwidth_sigma = 0.6  # 带宽减小致拥链路的对数标准差参数

    # 记录真实的链路类型，用于后续评估
    link_types = {}

    for edge in T.edges():
        label = T.edges[edge]['label']
        mu = default_mu
        sigma = default_sigma
        link_type = 0  # 默认正常链路

        if label in external_fault_links:  # 外部致拥
            mu = external_mu
            sigma = external_sigma
            link_type = 1
        if label in bandwidth_fault_links:  # 带宽减小
            mu = bandwidth_mu
            sigma = bandwidth_sigma
            link_type = 2

        # 记录链路类型
        link_types[label] = link_type

        # 使用 'label' 属性作为键，生成符合对数正态分布的时延数据
        link_delays_true[label] = np.random.lognormal(mean=mu, sigma=sigma, size=num_samples)

    # 计算链路的真实方差
    link_variances_true = {}
    for link_id, delays in link_delays_true.items():
        link_variances_true[link_id] = np.var(delays)

    # 创建输出目录
    output_dir = './path_variance'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 保存链路延迟方差到txt文件
    link_variance_file_path = os.path.join(output_dir, 'link_variance.txt')
    with open(link_variance_file_path, 'w') as file:
        file.write("Link ID, Variance\n")  # 写入表头
        # 根据link_id排序后写入文件
        for link_id, variance in sorted(link_variances_true.items()):
            file.write(f"{link_id}, {variance:.6f}\n")

    # 保存链路类型信息到文件
    link_types_file_path = os.path.join(output_dir, 'link_types.txt')
    with open(link_types_file_path, 'w') as file:
        file.write("Link ID, Type\n")  # 写入表头
        # 根据link_id排序后写入文件
        for link_id, link_type in sorted(link_types.items()):
            file.write(f"{link_id}, {link_type}\n")

    # 保存链路标签信息到单独文件
    link_labels_file_path = os.path.join(output_dir, 'link_labels.txt')
    # 创建一个包含所有链路标签的数组
    link_labels = np.zeros(num_links)
    for link_id, link_type in link_types.items():
        link_labels[link_id] = link_type
    # 保存为单列数据
    np.savetxt(link_labels_file_path, link_labels, fmt='%d')

    # =============================
    # 计算路径观测时延
    # =============================
    nodes = list(T.nodes)
    paths = []
    for path_id, dst in dest_marked:
        if nx.has_path(T, 0, dst):
            path = nx.shortest_path(T, 0, dst)
            paths.append((dst, path))

    path_delays = []
    path_variances = []  # 存储每条路径的延迟方差
    # 遍历每条路径，计算路径的延迟
    for dst, path in paths:
        path_delay_samples = np.zeros(num_samples)  # 初始化路径延迟数组

        # 对路径上的每一条边计算其延迟贡献
        for i in range(len(path) - 1):
            edge_label = T.edges[(path[i], path[i + 1])]['label']
            if edge_label in link_delays_true:
                path_delay_samples += link_delays_true[edge_label]

        path_variance = np.var(path_delay_samples)
        path_variances.append(path_variance)
        # 将路径的目的节点ID和路径延迟样本添加到列表中
        path_delays.append(path_delay_samples)

    # =============================
    # 计算路径时延的协方差
    # =============================
    path_variances = np.array(path_variances)
    sigma_sample = np.cov(path_delays)

    # 保存路径延迟方差到txt文件
    variance_file_path = os.path.join(output_dir, 'path_variance.txt')
    with open(variance_file_path, 'w') as file:
        file.write("Path ID, Variance\n")  # 写入表头
        for idx, variance in enumerate(path_variances):
            file.write(f"{idx}, {variance:.6f}\n")  # 写入每条路径的ID和对应的方差值

    # 保存路径延迟的协方差矩阵到另一个txt文件
    covariance_file_path = os.path.join(output_dir, 'path_covariances.txt')
    with open(covariance_file_path, 'w') as file:
        # 写入表头，方便理解数据结构
        file.write("Covariance Matrix of Path Delays\n")
        # 写入每个元素，保留六位小数
        for row in sigma_sample:
            line = ', '.join([f"{x:.6f}" for x in row])
            file.write(line + '\n')

    # 如果提供了实验目录，将文件复制到该目录
    if experiment_dir:
        # 复制路由矩阵
        ensure_directory(f"{experiment_dir}/output")
        shutil.copy2(file_path, f"{experiment_dir}/output/route_matrix.txt")

        # 复制路径方差文件
        ensure_directory(f"{experiment_dir}/path_variance")
        shutil.copy2(link_variance_file_path, f"{experiment_dir}/path_variance/link_variance.txt")
        shutil.copy2(link_types_file_path, f"{experiment_dir}/path_variance/link_types.txt")
        shutil.copy2(link_labels_file_path, f"{experiment_dir}/path_variance/link_labels.txt")
        shutil.copy2(variance_file_path, f"{experiment_dir}/path_variance/path_variance.txt")
        shutil.copy2(covariance_file_path, f"{experiment_dir}/path_variance/path_covariances.txt")

        # 额外保存故障链路信息供后续实验参考
        fault_links_file_path = f"{experiment_dir}/path_variance/fault_links.txt"
        with open(fault_links_file_path, 'w') as file:
            file.write("# 实验故障链路配置\n")
            file.write(f"外部流量致拥链路: {external_fault_links}\n")
            file.write(f"带宽减小致拥链路: {bandwidth_fault_links}\n")

        print(f"所有数据文件也已复制到: {experiment_dir}")

    print("模拟器运行完成，已生成所有必要数据文件")

    # 返回一些统计信息
    print(f"路径数量: {num_paths}")
    print(f"链路数量: {num_links}")
    print(f"链路类型统计:")
    link_type_counts = {0: 0, 1: 0, 2: 0}  # 正常、外部流量拥塞、带宽减小拥塞
    for link_type in link_types.values():
        link_type_counts[link_type] += 1
    print(f"  正常链路: {link_type_counts[0]}条")
    print(f"  外部流量致拥链路: {link_type_counts[1]}条")
    print(f"  带宽减小致拥链路: {link_type_counts[2]}条")


if __name__ == "__main__":
    main()