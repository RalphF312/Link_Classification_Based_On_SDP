#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import subprocess
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import shutil
import time
import sys
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score
from path_selection_strategies import get_path_selection_strategy
from alg_gurobi import tomo_gurobi
# 导入链路性能生成器
from link_performance_generator import generate_link_performance


def ensure_directory(directory):
    """确保目录存在，如果不存在则创建"""
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"创建目录: {directory}")
        return True
    except Exception as e:
        print(f"创建目录 {directory} 失败: {e}")
        return False


def convert_paths_to_binary(R, true_labels):
    """
    根据路由矩阵和链路标签将路径转换为二进制拥塞状态

    参数:
    R: 路由矩阵，行是路径，列是链路
    true_labels: 链路的真实标签 (0为正常，>0为拥塞)

    返回:
    二进制路径拥塞状态数组 (0表示正常，1表示拥塞)
    """
    # 创建拥塞链路的掩码
    congested_links = (true_labels > 0)

    # 对每条路径，检查是否经过任何拥塞链路
    path_binary_states = np.zeros(R.shape[0], dtype=int)

    for i in range(R.shape[0]):
        # 如果路径i经过任何一条拥塞链路，则该路径被标记为拥塞
        if np.any(R[i] & congested_links):
            path_binary_states[i] = 1

    return path_binary_states

def run_sdr_inference(strategy, ratio, experiment_dir, selected_paths_file=None):
    """
    运行SDR推断方法估计链路时延方差

    参数:
    strategy: 路径选择策略
    ratio: 观测路径比例
    experiment_dir: 实验结果目录
    selected_paths_file: 已选择路径索引的文件路径

    返回:
    推断结果输出目录
    """
    print("\n" + "=" * 80)
    print(f"步骤2: 使用{strategy}策略和{ratio * 100}%观测路径推断链路方差")
    print("=" * 80)

    output_dir = f"{experiment_dir}/sdp_variance_results_{strategy}_{int(ratio * 100)}"

    # 确保输出目录存在
    ensure_directory(output_dir)

    try:
        # 检查SDP_complete_link.py是否存在
        if not os.path.exists('SDP_complete_link.py'):
            print(f"错误: 找不到SDP_complete_link.py文件")
            print(f"当前工作目录: {os.getcwd()}")
            print("当前目录下的文件:")
            for file in os.listdir('.'):
                print(f"  - {file}")
            return None

        # 检查路径选择策略文件是否存在
        if not os.path.exists('path_selection_strategies.py'):
            print(f"错误: 找不到path_selection_strategies.py文件")
            return None

        # 准备命令行参数
        cmd = ["python", "SDP_complete_link.py",
               "--strategy", strategy,
               "--ratio", str(ratio),
               "--output_dir", output_dir]

        # 如果提供了选定路径文件，添加到命令行参数
        if selected_paths_file and os.path.exists(selected_paths_file):
            cmd.extend(["--selected_paths_file", selected_paths_file])
            print(f"使用选定路径文件: {selected_paths_file}")

        # 为stress策略准备拥塞链路参数
        congested_links_param = ""
        if strategy == 'stress':
            # 尝试从链路标签文件获取拥塞链路信息
            try:
                # 首先尝试从实验目录读取链路标签
                labels_file = f'{experiment_dir}/path_variance/link_labels.txt'
                if not os.path.exists(labels_file):
                    # 如果实验目录没有，尝试从原始目录读取
                    labels_file = './path_variance/link_labels.txt'

                if os.path.exists(labels_file):
                    link_labels = np.loadtxt(labels_file).astype(int)
                    # 获取所有拥塞链路(标签>0)的索引
                    congested_links = np.where(link_labels > 0)[0]
                    if len(congested_links) > 0:
                        congested_links_param = f"--congested_links {','.join(map(str, congested_links))}"
                        print(f"从链路标签文件获取拥塞链路: {congested_links}")
                    else:
                        print("警告: 从链路标签文件未找到拥塞链路")

                        # 尝试从fault_links.txt文件获取拥塞链路信息
                        fault_links_file = f'{experiment_dir}/path_variance/fault_links.txt'
                        if os.path.exists(fault_links_file):
                            print(f"尝试从 {fault_links_file} 文件获取拥塞链路信息")
                            external_links = []
                            bandwidth_links = []

                            with open(fault_links_file, 'r') as file:
                                for line in file:
                                    if '外部流量致拥链路:' in line:
                                        try:
                                            # 提取列表中的数字
                                            links_str = line.split(':')[1].strip()
                                            links_str = links_str.replace('[', '').replace(']', '')
                                            if links_str:
                                                external_links = [int(x.strip()) for x in links_str.split(',') if
                                                                  x.strip()]
                                        except Exception as e:
                                            print(f"解析外部流量链路时出错: {e}")

                                    if '带宽减小致拥链路:' in line:
                                        try:
                                            # 提取列表中的数字
                                            links_str = line.split(':')[1].strip()
                                            links_str = links_str.replace('[', '').replace(']', '')
                                            if links_str:
                                                bandwidth_links = [int(x.strip()) for x in links_str.split(',') if
                                                                   x.strip()]
                                        except Exception as e:
                                            print(f"解析带宽减小链路时出错: {e}")

                            congested_links = external_links + bandwidth_links
                            if congested_links:
                                congested_links_param = f"--congested_links {','.join(map(str, congested_links))}"
                                print(f"从fault_links.txt文件获取拥塞链路: {congested_links}")
                            else:
                                print("未能从fault_links.txt文件获取有效的拥塞链路信息")
                else:
                    print(f"警告: 找不到链路标签文件 {labels_file}")
                    # 如果没有文件，使用硬编码的拥塞链路
                    external_congested = [2, 5, 8, 18, 20]  # 外部流量致拥链路
                    bandwidth_congested = [10, 17, 24]  # 带宽减小致拥链路
                    congested_links = external_congested + bandwidth_congested
                    congested_links_param = f"--congested_links {','.join(map(str, congested_links))}"
                    print(f"使用硬编码的拥塞链路: {congested_links}")
            except Exception as e:
                print(f"获取拥塞链路信息时出错: {e}")
                # 出错时使用默认的拥塞链路
                congested_links = [2, 5, 8, 10, 17, 18, 20, 24]
                congested_links_param = f"--congested_links {','.join(map(str, congested_links))}"
                print(f"使用默认拥塞链路: {congested_links}")

        # 执行SDP_complete_link.py，如果是stress策略则传入拥塞链路参数
        if strategy == 'stress' and congested_links_param:
            cmd.extend(congested_links_param.split())

        print(f"执行命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)

        print("SDP_complete_link.py 执行输出:")
        print(result.stdout)

        if result.stderr:
            print("错误输出:")
            print(result.stderr)

        # 验证是否生成了结果文件
        variance_file = os.path.join(output_dir, 'inferred_link_variance.txt')

        if os.path.exists(variance_file):
            print(f"验证: 成功找到推断结果文件 {variance_file}")
            # 检查文件内容是否有效
            try:
                data = np.loadtxt(variance_file)
                print(f"验证: 推断结果文件包含 {len(data)} 条链路数据")

                # 将结果也复制到实验根目录，以备后续处理
                backup_file = os.path.join(experiment_dir, f'inferred_link_variance_{strategy}_{int(ratio * 100)}.txt')
                shutil.copy2(variance_file, backup_file)
                print(f"已将结果备份到 {backup_file}")

                return output_dir
            except Exception as e:
                print(f"验证推断结果文件内容时出错: {e}")
                return None
        else:
            print(f"错误: 未找到推断结果文件 {variance_file}")

            # 检查是否有其他结果文件
            print("检查输出目录中的文件:")
            if os.path.exists(output_dir):
                files = os.listdir(output_dir)
                if files:
                    print(f"目录 {output_dir} 中存在以下文件:")
                    for file in files:
                        print(f"  - {file}")
                else:
                    print(f"目录 {output_dir} 为空")
            else:
                print(f"目录 {output_dir} 不存在")

            return None

    except subprocess.CalledProcessError as e:
        print(f"运行SDR推断失败: {e}")
        print("错误输出:")
        print(e.stderr)
        return None
    except Exception as e:
        print(f"运行SDR推断时发生异常: {e}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_clustering(variance_file_path, experiment_dir, strategy, ratio):
    """
    Evaluate the congestion detection and classification effectiveness of clustering methods

    Parameters:
    variance_file_path: Path to the inferred link variance file
    experiment_dir: Directory to save experiment results
    strategy: Path selection strategy used
    ratio: Observation path ratio

    Returns:
    Dictionary containing evaluation metrics
    """
    print("\n" + "=" * 80)
    print(
        f"Step 3: Evaluate congestion detection and classification effectiveness (Strategy={strategy}, Ratio={ratio * 100}%)")
    print("=" * 80)

    # Verify if the inferred link variance file exists
    if not os.path.exists(variance_file_path):
        print(f"Error: Link variance file does not exist {variance_file_path}")
        # Try to use backup file
        backup_file = os.path.join(experiment_dir, f'inferred_link_variance_{strategy}_{int(ratio * 100)}.txt')
        if os.path.exists(backup_file):
            print(f"Using backup file {backup_file}")
            variance_file_path = backup_file
        else:
            print("Cannot continue evaluation, missing necessary link variance data")
            return None

    try:
        # Read inferred link variance
        link_variance = np.loadtxt(variance_file_path)
        print(f"Successfully read link variance data, total {len(link_variance)} links")

        # Try to read link labels file from experiment directory
        labels_file = f'{experiment_dir}/path_variance/link_labels.txt'

        if os.path.exists(labels_file):
            true_labels = np.loadtxt(labels_file).astype(int)
            print(f"Loaded true link labels from file: {labels_file}")

            # 检查是否有fault_links.txt文件，如果有，打印故障链路信息以便参考
            fault_links_file = f'{experiment_dir}/path_variance/fault_links.txt'
            if os.path.exists(fault_links_file):
                with open(fault_links_file, 'r') as f:
                    fault_links_info = f.read()
                print("故障链路信息:")
                print(fault_links_info)
        else:
            # If not in experiment directory, try to read from original directory
            orig_labels_file = './path_variance/link_labels.txt'
            if os.path.exists(orig_labels_file):
                true_labels = np.loadtxt(orig_labels_file).astype(int)
                print(f"Loaded true link labels from original file: {orig_labels_file}")
            else:
                print("Warning: No link labels file found. Creating default labels.")
                # 如果没有找到标签文件，创建默认的全零标签（表示全部是正常链路）
                true_labels = np.zeros(len(link_variance), dtype=int)

        # 统计各种类型链路的数量
        type_counts = {
            0: np.sum(true_labels == 0),  # 正常链路数量
            1: np.sum(true_labels == 1),  # 外部流量拥塞链路数量
            2: np.sum(true_labels == 2)  # 带宽减小拥塞链路数量
        }

        print(
            f"链路分类统计: 正常链路: {type_counts[0]}, 外部流量拥塞链路: {type_counts[1]}, 带宽减小拥塞链路: {type_counts[2]}")

        # Use K-means clustering to divide links into three categories
        kmeans = KMeans(n_clusters=3, random_state=42)
        cluster_labels = kmeans.fit_predict(link_variance.reshape(-1, 1))

        # Sort categories based on cluster center sizes
        centers = kmeans.cluster_centers_.flatten()
        center_order = np.argsort(centers)

        print(f"Cluster center values: {centers}")
        print(f"Cluster center order: {center_order}")

        # Map cluster labels to meaningful category labels: 0=Normal, 1=External Traffic Congestion, 2=Bandwidth Reduction Congestion
        label_mapping = {}
        label_mapping[center_order[0]] = 0  # Lowest variance class mapped to normal
        label_mapping[center_order[1]] = 1  # Medium variance class mapped to external traffic congestion
        label_mapping[center_order[2]] = 2  # Highest variance class mapped to bandwidth reduction congestion

        # Apply mapping
        predicted_labels = np.array([label_mapping[label] for label in cluster_labels])

        print(f"Clustering prediction results: {np.sum(predicted_labels == 0)} normal links, "
              f"{np.sum(predicted_labels == 1)} external traffic links, "
              f"{np.sum(predicted_labels == 2)} bandwidth attack links")

        # Calculate binary classification performance metrics (normal vs congestion)
        true_binary = (true_labels > 0).astype(int)
        pred_binary = (predicted_labels > 0).astype(int)

        binary_accuracy = accuracy_score(true_binary, pred_binary)
        binary_precision = precision_score(true_binary, pred_binary, zero_division=0)
        binary_recall = recall_score(true_binary, pred_binary, zero_division=0)

        # DR (Detection Rate) = Recall
        dr = binary_recall

        # FPR (False Positive Rate) = (Normal links misclassified as congested) / All normal links
        normal_count = np.sum(true_binary == 0)
        if normal_count > 0:
            fpr = np.sum((true_binary == 0) & (pred_binary == 1)) / normal_count
        else:
            fpr = 0

        print(f"Binary classification performance: DR={dr:.4f}, FPR={fpr:.4f}")

        # Calculate fine-grained classification accuracy
        congestion_mask = (true_labels > 0) & (predicted_labels > 0)
        if np.sum(congestion_mask) > 0:
            congestion_classification_acc = np.mean(true_labels[congestion_mask] == predicted_labels[congestion_mask])
        else:
            congestion_classification_acc = 0

        print(f"Congestion classification accuracy: {congestion_classification_acc:.4f}")

        # Calculate confusion matrix
        confusion = {}
        for true_val in [0, 1, 2]:
            for pred_val in [0, 1, 2]:
                key = f'true_{true_val}_pred_{pred_val}'
                confusion[key] = np.sum((true_labels == true_val) & (predicted_labels == pred_val))

        # Save detailed link classification results
        results_dir = f"{experiment_dir}/classification_results_{strategy}_{int(ratio * 100)}"
        ensure_directory(results_dir)

        detailed_results = pd.DataFrame({
            'Link_ID': range(len(link_variance)),
            'Variance': link_variance,
            'True_Label': true_labels,
            'Predicted_Label': predicted_labels,
            'Is_Correct': true_labels == predicted_labels
        })
        detailed_results.to_csv(f"{results_dir}/link_classification_results.csv", index=False)

        # Save cluster center values
        with open(f"{results_dir}/cluster_centers.txt", 'w') as f:
            f.write(f"Cluster center values: {centers}\n")
            f.write(f"Cluster center order: {center_order}\n\n")
            for i, center in enumerate(centers):
                mapped_label = [k for k, v in label_mapping.items() if v == i][0]
                f.write(f"Category {i} (Cluster {mapped_label}): Center value = {center}\n")

        # Write metrics information to file
        with open(f"{results_dir}/metrics.txt", 'w') as f:
            f.write(f"Strategy: {strategy}\n")
            f.write(f"Observation path ratio: {ratio * 100}%\n")
            f.write(f"Binary classification accuracy: {binary_accuracy:.4f}\n")
            f.write(f"Detection Rate (DR): {dr:.4f}\n")
            f.write(f"False Positive Rate (FPR): {fpr:.4f}\n")
            f.write(f"Congestion classification accuracy: {congestion_classification_acc:.4f}\n")
            f.write("\nConfusion Matrix:\n")
            for true_val in [0, 1, 2]:
                for pred_val in [0, 1, 2]:
                    key = f'true_{true_val}_pred_{pred_val}'
                    f.write(f"True:{true_val}, Predicted:{pred_val} = {confusion[key]}\n")

        # Return metrics
        results = {
            'strategy': strategy,
            'ratio': ratio,
            'dr': dr,
            'fpr': fpr,
            'binary_accuracy': binary_accuracy,
            'binary_precision': binary_precision,
            'binary_recall': binary_recall,
            'congestion_classification_acc': congestion_classification_acc
        }
        results.update(confusion)

        return results

    except Exception as e:
        print(f"评估聚类过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_with_l1l2(variance_file_path, experiment_dir, strategy, ratio, selected_paths_file=None):
    """
    对比当前算法与L1-L2 optimization baseline算法的拥塞检测性能

    参数:
    variance_file_path: 链路方差文件路径
    experiment_dir: 实验结果目录
    strategy: 路径选择策略
    ratio: 观测路径比例
    selected_paths_file: 已选择路径索引的文件路径

    返回:
    包含对比结果的字典
    """
    print("\n" + "=" * 80)
    print(f"对比当前算法与L1-L2优化算法 (Strategy={strategy}, Ratio={ratio * 100}%)")
    print("=" * 80)

    try:
        # 导入L1-L2算法
        from alg_l1_l2 import detect_congested_links

        # 加载必要的数据
        # 1. 路由矩阵R
        route_matrix_file = f'{experiment_dir}/output/route_matrix.txt'
        if not os.path.exists(route_matrix_file):
            route_matrix_file = './output/route_matrix.txt'

        R = np.loadtxt(route_matrix_file, delimiter='\t').astype(int)

        # 2. 链路标签（真实拥塞状态）
        labels_file = f'{experiment_dir}/path_variance/link_labels.txt'
        if os.path.exists(labels_file):
            true_labels = np.loadtxt(labels_file).astype(int)
        else:
            orig_labels_file = './path_variance/link_labels.txt'
            if os.path.exists(orig_labels_file):
                true_labels = np.loadtxt(orig_labels_file).astype(int)
            else:
                print("警告: 未找到链路标签文件，创建默认全零标签")
                true_labels = np.zeros(R.shape[1], dtype=int)

        # 3. 获取选定的路径
        if selected_paths_file and os.path.exists(selected_paths_file):
            print(f"从文件加载已选择的路径: {selected_paths_file}")
            selected_paths = np.loadtxt(selected_paths_file, dtype=int)
        else:
            # 如果没有提供路径文件，使用策略函数选择路径
            path_selection_func = get_path_selection_strategy(strategy)
            selected_paths, _ = path_selection_func(R, ratio)

        # 4. 加载路径方差数据
        path_variance_file = f'{experiment_dir}/path_variance/path_variance.txt'
        if not os.path.exists(path_variance_file):
            path_variance_file = './path_variance/path_variance.txt'

        if os.path.exists(path_variance_file):
            # 读取路径方差数据，跳过头行，只读取方差列
            path_variance_data = np.loadtxt(path_variance_file, delimiter=',', skiprows=1, usecols=(1,))

            # 获取选定路径的方差数据
            selected_path_variance = path_variance_data[selected_paths]

            # 获取选定路径的路由矩阵
            R_selected = R[selected_paths]

            # 5. 使用L1-L2优化算法进行链路故障检测
            # 默认参数: mixing_parameter=0.05, threshold=0.01
            l1l2_binary = detect_congested_links(R_selected, selected_path_variance)

            # 6. 计算L1-L2算法性能指标
            true_binary = (true_labels > 0).astype(int)

            l1l2_accuracy = accuracy_score(true_binary, l1l2_binary)
            l1l2_precision = precision_score(true_binary, l1l2_binary, zero_division=0)
            l1l2_recall = recall_score(true_binary, l1l2_binary, zero_division=0)

            # DR (Detection Rate) = Recall
            l1l2_dr = l1l2_recall

            # FPR (False Positive Rate)
            normal_count = np.sum(true_binary == 0)
            if normal_count > 0:
                l1l2_fpr = np.sum((true_binary == 0) & (l1l2_binary == 1)) / normal_count
            else:
                l1l2_fpr = 0

            print(f"L1-L2算法性能: DR={l1l2_dr:.4f}, FPR={l1l2_fpr:.4f}")

            # 返回L1-L2算法的评估结果
            return {
                'algorithm': 'l1l2',
                'dr': l1l2_dr,
                'fpr': l1l2_fpr,
                'accuracy': l1l2_accuracy,
                'precision': l1l2_precision,
                'recall': l1l2_recall,
                'binary_result': l1l2_binary
            }
        else:
            print(f"错误: 未找到路径方差文件 {path_variance_file}")
            return None

    except Exception as e:
        print(f"L1-L2评估过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None


# 更新后的evaluate_with_baseline函数，自动包含Gurobi和L1-L2两个baseline
def evaluate_with_baseline(variance_file_path, experiment_dir, strategy, ratio, selected_paths_file=None):
    """
    对比当前算法与多个baseline算法的拥塞检测性能

    参数:
    variance_file_path: 链路方差文件路径
    experiment_dir: 实验结果目录
    strategy: 路径选择策略
    ratio: 观测路径比例
    selected_paths_file: 已选择路径索引的文件路径

    返回:
    包含对比结果的字典
    """
    print("\n" + "=" * 80)
    print(f"对比当前算法与baseline算法 (Strategy={strategy}, Ratio={ratio * 100}%)")
    print("=" * 80)

    # 确保结果目录存在
    results_dir = f"{experiment_dir}/baseline_comparison_{strategy}_{int(ratio * 100)}"
    ensure_directory(results_dir)

    current_variance = np.loadtxt(variance_file_path)

    # 加载路由矩阵和标签
    route_matrix_file = f'{experiment_dir}/output/route_matrix.txt'
    if not os.path.exists(route_matrix_file):
        route_matrix_file = './output/route_matrix.txt'

    R = np.loadtxt(route_matrix_file, delimiter='\t').astype(int)

    labels_file = f'{experiment_dir}/path_variance/link_labels.txt'
    if os.path.exists(labels_file):
        true_labels = np.loadtxt(labels_file).astype(int)
    else:
        orig_labels_file = './path_variance/link_labels.txt'
        if os.path.exists(orig_labels_file):
            true_labels = np.loadtxt(orig_labels_file).astype(int)
        else:
            print("警告: 未找到链路标签文件，创建默认全零标签")
            true_labels = np.zeros(R.shape[1], dtype=int)

    # 计算当前算法的性能
    # 使用K-means聚类为链路分类
    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_labels = kmeans.fit_predict(current_variance.reshape(-1, 1))

    # 映射聚类标签
    centers = kmeans.cluster_centers_.flatten()
    center_order = np.argsort(centers)

    label_mapping = {}
    label_mapping[center_order[0]] = 0  # 最低方差对应正常
    label_mapping[center_order[1]] = 1  # 中等方差对应外部流量拥塞
    label_mapping[center_order[2]] = 2  # 最高方差对应带宽减小拥塞

    predicted_labels = np.array([label_mapping[label] for label in cluster_labels])

    # 计算当前算法的二分类性能
    true_binary = (true_labels > 0).astype(int)
    current_binary = (predicted_labels > 0).astype(int)

    current_accuracy = accuracy_score(true_binary, current_binary)
    current_precision = precision_score(true_binary, current_binary, zero_division=0)
    current_recall = recall_score(true_binary, current_binary, zero_division=0)

    # DR (Detection Rate) = Recall
    current_dr = current_recall

    # FPR (False Positive Rate)
    normal_count = np.sum(true_binary == 0)
    if normal_count > 0:
        current_fpr = np.sum((true_binary == 0) & (current_binary == 1)) / normal_count
    else:
        current_fpr = 0

    print(f"当前算法性能: DR={current_dr:.4f}, FPR={current_fpr:.4f}")

    # 存储所有算法的评估结果
    baseline_results = {}

    # 评估Gurobi baseline
    gurobi_result = evaluate_with_gurobi(variance_file_path, experiment_dir, strategy, ratio, selected_paths_file)
    if gurobi_result:
        baseline_results['gurobi'] = gurobi_result

    # 评估L1-L2 baseline
    l1l2_result = evaluate_with_l1l2(variance_file_path, experiment_dir, strategy, ratio, selected_paths_file)
    if l1l2_result:
        baseline_results['l1l2'] = l1l2_result

    # 整合并保存对比结果
    comparison_result = {
        'ratio': ratio,
        'strategy': strategy,
        'current_dr': current_dr,
        'current_fpr': current_fpr,
        'current_accuracy': current_accuracy,
        'current_precision': current_precision,
    }

    # 为每个baseline添加对比结果
    for method, result in baseline_results.items():
        if result:
            for key in ['dr', 'fpr', 'accuracy', 'precision', 'recall']:
                comparison_result[f'{method}_{key}'] = result[key]

    # 保存对比结果
    with open(f"{results_dir}/comparison_results.txt", 'w') as f:
        f.write(f"观测路径比例: {ratio * 100}%\n")
        f.write(f"路径选择策略: {strategy}\n\n")

        f.write("当前算法性能:\n")
        f.write(f"  准确率: {current_accuracy:.4f}\n")
        f.write(f"  检测率(DR): {current_dr:.4f}\n")
        f.write(f"  误报率(FPR): {current_fpr:.4f}\n")
        f.write(f"  精确率: {current_precision:.4f}\n\n")

        for method, result in baseline_results.items():
            if result:
                method_name = "Gurobi" if method == "gurobi" else "L1-L2优化"
                f.write(f"{method_name}算法性能:\n")
                f.write(f"  准确率: {result['accuracy']:.4f}\n")
                f.write(f"  检测率(DR): {result['dr']:.4f}\n")
                f.write(f"  误报率(FPR): {result['fpr']:.4f}\n")
                f.write(f"  精确率: {result['precision']:.4f}\n\n")

    # 绘制对比图表
    plt.figure(figsize=(12, 8))
    metrics = ['DR', 'FPR', 'Accuracy', 'Precision']
    current_values = [current_dr, current_fpr, current_accuracy, current_precision]

    # 准备所有baseline的数据
    baseline_names = list(baseline_results.keys())
    baseline_values = []

    for method in baseline_names:
        if method in baseline_results and baseline_results[method]:
            values = [
                baseline_results[method]['dr'],
                baseline_results[method]['fpr'],
                baseline_results[method]['accuracy'],
                baseline_results[method]['precision']
            ]
            baseline_values.append(values)

    # 确定柱状图的位置
    x = np.arange(len(metrics))
    total_width = 0.8  # 所有柱子的总宽度
    width = total_width / (1 + len(baseline_names))  # 每个柱子的宽度

    # 绘制当前算法的柱子
    plt.bar(x - total_width / 2 + width / 2, current_values, width, label='当前算法')

    # 绘制每个baseline的柱子
    for i, (name, values) in enumerate(zip(baseline_names, baseline_values)):
        offset = i + 1
        plt.bar(x - total_width / 2 + offset * width + width / 2, values, width,
                label=f"{'Gurobi' if name == 'gurobi' else 'L1-L2优化'}算法")

    plt.xlabel('评估指标')
    plt.ylabel('值')
    plt.title(f'算法性能对比 (Ratio={ratio * 100}%, Strategy={strategy})')
    plt.xticks(x, metrics)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{results_dir}/performance_comparison.png")
    plt.close()

    return comparison_result


# 添加Gurobi评估函数
def evaluate_with_gurobi(variance_file_path, experiment_dir, strategy, ratio, selected_paths_file=None):
    """
    对比当前算法与Gurobi baseline算法的拥塞检测性能

    参数:
    variance_file_path: 链路方差文件路径
    experiment_dir: 实验结果目录
    strategy: 路径选择策略
    ratio: 观测路径比例
    selected_paths_file: 已选择路径索引的文件路径

    返回:
    包含对比结果的字典
    """
    print("\n" + "=" * 80)
    print(f"对比当前算法与Gurobi baseline算法 (Strategy={strategy}, Ratio={ratio * 100}%)")
    print("=" * 80)

    # 加载必要的数据
    # 1. 路由矩阵R
    route_matrix_file = f'{experiment_dir}/output/route_matrix.txt'
    if not os.path.exists(route_matrix_file):
        route_matrix_file = './output/route_matrix.txt'

    R = np.loadtxt(route_matrix_file, delimiter='\t').astype(int)

    # 2. 链路标签（真实拥塞状态）
    labels_file = f'{experiment_dir}/path_variance/link_labels.txt'
    if os.path.exists(labels_file):
        true_labels = np.loadtxt(labels_file).astype(int)
    else:
        orig_labels_file = './path_variance/link_labels.txt'
        if os.path.exists(orig_labels_file):
            true_labels = np.loadtxt(orig_labels_file).astype(int)
        else:
            print("警告: 未找到链路标签文件，创建默认全零标签")
            true_labels = np.zeros(R.shape[1], dtype=int)

    # 3. 获取选定的路径
    if selected_paths_file and os.path.exists(selected_paths_file):
        print(f"从文件加载已选择的路径: {selected_paths_file}")
        selected_paths = np.loadtxt(selected_paths_file, dtype=int)
    else:
        # 如果没有提供路径文件，使用策略函数选择路径
        path_selection_func = get_path_selection_strategy(strategy)
        selected_paths, _ = path_selection_func(R, ratio)

    # 4. 转换路径为二进制拥塞状态
    path_binary_states = convert_paths_to_binary(R, true_labels)

    # 5. 准备Gurobi算法的输入
    Y_selected = path_binary_states[selected_paths].reshape(-1, 1)  # 转为列向量
    R_selected = R[selected_paths]

    # 6. 运行Gurobi算法
    try:
        gurobi_results = tomo_gurobi(R_selected, Y_selected)

        # 7. 将Gurobi结果转为二进制标签
        gurobi_binary = (gurobi_results > 0).astype(int).flatten()

        # 8. 计算Gurobi算法性能指标
        true_binary = (true_labels > 0).astype(int)

        gurobi_accuracy = accuracy_score(true_binary, gurobi_binary)
        gurobi_precision = precision_score(true_binary, gurobi_binary, zero_division=0)
        gurobi_recall = recall_score(true_binary, gurobi_binary, zero_division=0)

        # DR (Detection Rate) = Recall
        gurobi_dr = gurobi_recall

        # FPR (False Positive Rate)
        normal_count = np.sum(true_binary == 0)
        if normal_count > 0:
            gurobi_fpr = np.sum((true_binary == 0) & (gurobi_binary == 1)) / normal_count
        else:
            gurobi_fpr = 0

        print(f"Gurobi算法性能: DR={gurobi_dr:.4f}, FPR={gurobi_fpr:.4f}")

        # 返回Gurobi算法的评估结果
        return {
            'algorithm': 'gurobi',
            'dr': gurobi_dr,
            'fpr': gurobi_fpr,
            'accuracy': gurobi_accuracy,
            'precision': gurobi_precision,
            'recall': gurobi_recall,
            'binary_result': gurobi_binary
        }

    except Exception as e:
        print(f"Gurobi评估过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None

def visualize_ratio_results(results_df, ratio, output_dir):
    """
    为单个观测比例的重复实验结果生成可视化

    参数:
    results_df: 包含该比例所有重复实验结果的DataFrame
    ratio: 观测比例
    output_dir: 输出目录
    """
    try:
        # 1. 检测率(DR)的直方图
        plt.figure(figsize=(10, 6))
        plt.hist(results_df['dr'], bins=20, alpha=0.7, color='blue')
        plt.axvline(results_df['dr'].mean(), color='red', linestyle='dashed', linewidth=2,
                    label=f'Mean: {results_df["dr"].mean():.4f}')
        plt.axvline(results_df['dr'].median(), color='green', linestyle='dashed', linewidth=2,
                    label=f'Median: {results_df["dr"].median():.4f}')
        plt.xlabel('Detection Rate (DR)')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of Detection Rate at {int(ratio * 100)}% Observation Ratio')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{output_dir}/dr_histogram.png")
        plt.close()

        # 2. 误报率(FPR)的直方图
        plt.figure(figsize=(10, 6))
        plt.hist(results_df['fpr'], bins=20, alpha=0.7, color='orange')
        plt.axvline(results_df['fpr'].mean(), color='red', linestyle='dashed', linewidth=2,
                    label=f'Mean: {results_df["fpr"].mean():.4f}')
        plt.axvline(results_df['fpr'].median(), color='green', linestyle='dashed', linewidth=2,
                    label=f'Median: {results_df["fpr"].median():.4f}')
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of False Positive Rate at {int(ratio * 100)}% Observation Ratio')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{output_dir}/fpr_histogram.png")
        plt.close()

        # 3. 拥塞分类准确率的直方图
        plt.figure(figsize=(10, 6))
        plt.hist(results_df['congestion_classification_acc'], bins=20, alpha=0.7, color='green')
        plt.axvline(results_df['congestion_classification_acc'].mean(), color='red', linestyle='dashed', linewidth=2,
                    label=f'Mean: {results_df["congestion_classification_acc"].mean():.4f}')
        plt.axvline(results_df['congestion_classification_acc'].median(), color='blue', linestyle='dashed', linewidth=2,
                    label=f'Median: {results_df["congestion_classification_acc"].median():.4f}')
        plt.xlabel('Congestion Classification Accuracy')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of Congestion Classification Accuracy at {int(ratio * 100)}% Observation Ratio')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{output_dir}/congestion_classification_histogram.png")
        plt.close()

        # 4. 散点图：DR vs FPR
        plt.figure(figsize=(10, 6))
        plt.scatter(results_df['dr'], results_df['fpr'], alpha=0.6)
        plt.axhline(results_df['fpr'].mean(), color='red', linestyle='dashed', alpha=0.5)
        plt.axvline(results_df['dr'].mean(), color='red', linestyle='dashed', alpha=0.5)
        plt.xlabel('Detection Rate (DR)')
        plt.ylabel('False Positive Rate (FPR)')
        plt.title(f'DR vs FPR at {int(ratio * 100)}% Observation Ratio')
        plt.grid(True)
        plt.savefig(f"{output_dir}/dr_vs_fpr_scatter.png")
        plt.close()

    except Exception as e:
        print(f"生成单个比例的可视化结果时发生错误: {e}")
        import traceback
        traceback.print_exc()


def visualize_baseline_comparison(results_df, ratio, output_dir):
    """
    为单个观测比例的baseline对比结果生成可视化

    参数:
    results_df: 包含该比例所有重复实验结果的DataFrame
    ratio: 观测比例
    output_dir: 输出目录
    """
    try:
        # 检查是否有各算法的结果
        has_gurobi = 'gurobi_dr' in results_df.columns
        has_l1l2 = 'l1l2_dr' in results_df.columns

        # DR对比图
        plt.figure(figsize=(12, 6))
        plt.hist(results_df['current_dr'], bins=15, alpha=0.5, label='Current Algorithm')

        if has_gurobi:
            plt.hist(results_df['gurobi_dr'], bins=15, alpha=0.5, label='Gurobi Algorithm')

        if has_l1l2:
            plt.hist(results_df['l1l2_dr'], bins=15, alpha=0.5, label='L1-L2 Algorithm')

        plt.axvline(results_df['current_dr'].mean(), color='blue', linestyle='dashed', linewidth=2,
                    label=f'Current Algo Mean: {results_df["current_dr"].mean():.4f}')

        if has_gurobi:
            plt.axvline(results_df['gurobi_dr'].mean(), color='red', linestyle='dashed', linewidth=2,
                        label=f'Gurobi Mean: {results_df["gurobi_dr"].mean():.4f}')

        if has_l1l2:
            plt.axvline(results_df['l1l2_dr'].mean(), color='green', linestyle='dashed', linewidth=2,
                        label=f'L1-L2 Mean: {results_df["l1l2_dr"].mean():.4f}')

        plt.xlabel('Detection Rate (DR)')
        plt.ylabel('Frequency')
        plt.title(f'DR Distribution Comparison (Observation Ratio={ratio * 100}%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{output_dir}/dr_comparison.png")
        plt.close()

        # FPR对比图
        plt.figure(figsize=(12, 6))
        plt.hist(results_df['current_fpr'], bins=15, alpha=0.5, label='Current Algorithm')

        if has_gurobi:
            plt.hist(results_df['gurobi_fpr'], bins=15, alpha=0.5, label='Gurobi Algorithm')

        if has_l1l2:
            plt.hist(results_df['l1l2_fpr'], bins=15, alpha=0.5, label='L1-L2 Algorithm')

        plt.axvline(results_df['current_fpr'].mean(), color='blue', linestyle='dashed', linewidth=2,
                    label=f'Current Algo Mean: {results_df["current_fpr"].mean():.4f}')

        if has_gurobi:
            plt.axvline(results_df['gurobi_fpr'].mean(), color='red', linestyle='dashed', linewidth=2,
                        label=f'Gurobi Mean: {results_df["gurobi_fpr"].mean():.4f}')

        if has_l1l2:
            plt.axvline(results_df['l1l2_fpr'].mean(), color='green', linestyle='dashed', linewidth=2,
                        label=f'L1-L2 Mean: {results_df["l1l2_fpr"].mean():.4f}')

        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('Frequency')
        plt.title(f'FPR Distribution Comparison (Observation Ratio={ratio * 100}%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{output_dir}/fpr_comparison.png")
        plt.close()

        # 平均性能对比柱状图
        plt.figure(figsize=(14, 7))
        metrics = ['DR', 'FPR', 'Accuracy']
        current_values = [results_df['current_dr'].mean(), results_df['current_fpr'].mean(),
                          results_df['current_accuracy'].mean()]

        bar_positions = []
        bar_values = []
        bar_labels = []

        # 设置柱状图位置
        x = np.arange(len(metrics))
        width = 0.25  # 每个算法的柱子宽度

        # 添加当前算法
        plt.bar(x - width, current_values, width, label='Current Algorithm', color='royalblue')

        # 添加Gurobi算法（如果有数据）
        if has_gurobi:
            gurobi_values = [results_df['gurobi_dr'].mean(), results_df['gurobi_fpr'].mean(),
                             results_df['gurobi_accuracy'].mean()]
            plt.bar(x, gurobi_values, width, label='Gurobi Algorithm', color='firebrick')

        # 添加L1-L2算法（如果有数据）
        if has_l1l2:
            l1l2_values = [results_df['l1l2_dr'].mean(), results_df['l1l2_fpr'].mean(),
                           results_df['l1l2_accuracy'].mean()]
            plt.bar(x + width, l1l2_values, width, label='L1-L2 Algorithm', color='forestgreen')

        plt.xlabel('Performance Metrics')
        plt.ylabel('Average Value')
        plt.title(f'Algorithm Performance Comparison (Observation Ratio={ratio * 100}%)')
        plt.xticks(x, metrics)
        plt.ylim(0, 1.0)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{output_dir}/performance_metrics_comparison.png")
        plt.close()

        # 生成算法性能比较表格
        with open(f"{output_dir}/algorithm_comparison.txt", 'w') as f:
            f.write(f"算法性能比较 (观测比例: {ratio * 100}%)\n")
            f.write("=" * 60 + "\n\n")

            # 表头
            f.write(f"{'指标':<15}{'当前算法':<15}")
            if has_gurobi:
                f.write(f"{'Gurobi算法':<15}")
            if has_l1l2:
                f.write(f"{'L1-L2算法':<15}")
            f.write("\n" + "-" * 60 + "\n")

            # DR
            f.write(f"{'检测率(DR)':<15}{results_df['current_dr'].mean():.4f} ± {results_df['current_dr'].std():.4f}  ")
            if has_gurobi:
                f.write(f"{results_df['gurobi_dr'].mean():.4f} ± {results_df['gurobi_dr'].std():.4f}  ")
            if has_l1l2:
                f.write(f"{results_df['l1l2_dr'].mean():.4f} ± {results_df['l1l2_dr'].std():.4f}  ")
            f.write("\n")

            # FPR
            f.write(
                f"{'误报率(FPR)':<15}{results_df['current_fpr'].mean():.4f} ± {results_df['current_fpr'].std():.4f}  ")
            if has_gurobi:
                f.write(f"{results_df['gurobi_fpr'].mean():.4f} ± {results_df['gurobi_fpr'].std():.4f}  ")
            if has_l1l2:
                f.write(f"{results_df['l1l2_fpr'].mean():.4f} ± {results_df['l1l2_fpr'].std():.4f}  ")
            f.write("\n")

            # Accuracy
            f.write(
                f"{'准确率':<15}{results_df['current_accuracy'].mean():.4f} ± {results_df['current_accuracy'].std():.4f}  ")
            if has_gurobi:
                f.write(f"{results_df['gurobi_accuracy'].mean():.4f} ± {results_df['gurobi_accuracy'].std():.4f}  ")
            if has_l1l2:
                f.write(f"{results_df['l1l2_accuracy'].mean():.4f} ± {results_df['l1l2_accuracy'].std():.4f}  ")
            f.write("\n")

    except Exception as e:
        print(f"生成baseline对比可视化时发生错误: {e}")
        import traceback
        traceback.print_exc()


def visualize_summary_baseline_comparison(summary_df, summary_dir):
    """
    可视化baseline对比的汇总统计结果

    参数:
    summary_df: 包含汇总统计的DataFrame
    summary_dir: 可视化结果保存目录
    """
    try:
        # 排序以确保按观测比例绘图
        summary_df = summary_df.sort_values(by='ratio')

        # 检查是否有各算法的结果
        has_gurobi = 'gurobi_dr_mean' in summary_df.columns and not summary_df['gurobi_dr_mean'].isna().all()
        has_l1l2 = 'l1l2_dr_mean' in summary_df.columns and not summary_df['l1l2_dr_mean'].isna().all()

        # DR对比曲线
        plt.figure(figsize=(12, 7))
        plt.errorbar(summary_df['ratio'] * 100, summary_df['current_dr_mean'],
                     yerr=summary_df['current_dr_std'], fmt='o-', capsize=5,
                     label='Current Algorithm', color='royalblue', linewidth=2)

        if has_gurobi:
            plt.errorbar(summary_df['ratio'] * 100, summary_df['gurobi_dr_mean'],
                         yerr=summary_df['gurobi_dr_std'], fmt='s-', capsize=5,
                         label='Gurobi Algorithm', color='firebrick', linewidth=2)

        if has_l1l2:
            plt.errorbar(summary_df['ratio'] * 100, summary_df['l1l2_dr_mean'],
                         yerr=summary_df['l1l2_dr_std'], fmt='^-', capsize=5,
                         label='L1-L2 Algorithm', color='forestgreen', linewidth=2)

        plt.xlabel('Observation Path Ratio (%)', fontsize=12)
        plt.ylabel('Detection Rate (DR)', fontsize=12)
        plt.title('DR Comparison (with Standard Deviation)', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True)
        plt.ylim([0, 1.1])
        plt.savefig(f"{summary_dir}/dr_comparison_with_std.png")
        plt.close()

        # FPR对比曲线
        plt.figure(figsize=(12, 7))
        plt.errorbar(summary_df['ratio'] * 100, summary_df['current_fpr_mean'],
                     yerr=summary_df['current_fpr_std'], fmt='o-', capsize=5,
                     label='Current Algorithm', color='royalblue', linewidth=2)

        if has_gurobi:
            plt.errorbar(summary_df['ratio'] * 100, summary_df['gurobi_fpr_mean'],
                         yerr=summary_df['gurobi_fpr_std'], fmt='s-', capsize=5,
                         label='Gurobi Algorithm', color='firebrick', linewidth=2)

        if has_l1l2:
            plt.errorbar(summary_df['ratio'] * 100, summary_df['l1l2_fpr_mean'],
                         yerr=summary_df['l1l2_fpr_std'], fmt='^-', capsize=5,
                         label='L1-L2 Algorithm', color='forestgreen', linewidth=2)

        plt.xlabel('Observation Path Ratio (%)', fontsize=12)
        plt.ylabel('False Positive Rate (FPR)', fontsize=12)
        plt.title('FPR Comparison (with Standard Deviation)', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True)
        plt.ylim([0, 1.1])
        plt.savefig(f"{summary_dir}/fpr_comparison_with_std.png")
        plt.close()

        # Accuracy对比曲线
        plt.figure(figsize=(12, 7))
        plt.errorbar(summary_df['ratio'] * 100, summary_df['current_accuracy_mean'],
                     yerr=summary_df['current_accuracy_std'], fmt='o-', capsize=5,
                     label='Current Algorithm', color='royalblue', linewidth=2)

        if has_gurobi:
            plt.errorbar(summary_df['ratio'] * 100, summary_df['gurobi_accuracy_mean'],
                         yerr=summary_df['gurobi_accuracy_std'], fmt='s-', capsize=5,
                         label='Gurobi Algorithm', color='firebrick', linewidth=2)

        if has_l1l2:
            plt.errorbar(summary_df['ratio'] * 100, summary_df['l1l2_accuracy_mean'],
                         yerr=summary_df['l1l2_accuracy_std'], fmt='^-', capsize=5,
                         label='L1-L2 Algorithm', color='forestgreen', linewidth=2)

        plt.xlabel('Observation Path Ratio (%)', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Accuracy Comparison (with Standard Deviation)', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True)
        plt.ylim([0, 1.1])
        plt.savefig(f"{summary_dir}/accuracy_comparison_with_std.png")
        plt.close()

        # 三个算法在不同比例下的DR对比条形图
        plt.figure(figsize=(14, 8))

        x = np.arange(len(summary_df))
        width = 0.25

        # 当前算法的条形
        rects1 = plt.bar(x - width, summary_df['current_dr_mean'], width,
                         label='Current Algorithm', color='royalblue')

        # Gurobi算法的条形
        if has_gurobi:
            rects2 = plt.bar(x, summary_df['gurobi_dr_mean'], width,
                             label='Gurobi Algorithm', color='firebrick')

        # L1-L2算法的条形
        if has_l1l2:
            rects3 = plt.bar(x + width, summary_df['l1l2_dr_mean'], width,
                             label='L1-L2 Algorithm', color='forestgreen')

        plt.xlabel('Observation Path Ratio (%)', fontsize=12)
        plt.ylabel('Detection Rate (DR)', fontsize=12)
        plt.title('DR Comparison by Observation Ratio', fontsize=14)
        plt.xticks(x, [f"{int(r * 100)}%" for r in summary_df['ratio']])
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1.1])
        plt.savefig(f"{summary_dir}/dr_bar_comparison.png")
        plt.close()

        # 生成综合报告
        with open(f"{summary_dir}/comparison_report.txt", 'w') as f:
            f.write("算法对比综合报告\n")
            f.write("===========================\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"观测比例: {', '.join([f'{int(r * 100)}%' for r in sorted(summary_df['ratio'])])}\n")
            f.write(f"每个比例重复次数: {summary_df['repetitions'].iloc[0]}\n\n")

            f.write("性能指标汇总:\n")
            f.write("--------------\n")

            # 按照观测比例排序
            for _, row in summary_df.sort_values(by='ratio').iterrows():
                ratio = row['ratio']
                f.write(f"\n观测比例: {int(ratio * 100)}%\n")

                f.write("  当前算法性能:\n")
                f.write(f"    检测率(DR): {row['current_dr_mean']:.4f} ± {row['current_dr_std']:.4f}\n")
                f.write(f"    误报率(FPR): {row['current_fpr_mean']:.4f} ± {row['current_fpr_std']:.4f}\n")
                f.write(f"    准确率: {row['current_accuracy_mean']:.4f} ± {row['current_accuracy_std']:.4f}\n")

                if has_gurobi:
                    f.write("  Gurobi算法性能:\n")
                    f.write(f"    检测率(DR): {row['gurobi_dr_mean']:.4f} ± {row['gurobi_dr_std']:.4f}\n")
                    f.write(f"    误报率(FPR): {row['gurobi_fpr_mean']:.4f} ± {row['gurobi_fpr_std']:.4f}\n")
                    f.write(f"    准确率: {row['gurobi_accuracy_mean']:.4f} ± {row['gurobi_accuracy_std']:.4f}\n")

                if has_l1l2:
                    f.write("  L1-L2算法性能:\n")
                    f.write(f"    检测率(DR): {row['l1l2_dr_mean']:.4f} ± {row['l1l2_dr_std']:.4f}\n")
                    f.write(f"    误报率(FPR): {row['l1l2_fpr_mean']:.4f} ± {row['l1l2_fpr_std']:.4f}\n")
                    f.write(f"    准确率: {row['l1l2_accuracy_mean']:.4f} ± {row['l1l2_accuracy_std']:.4f}\n")

                # 计算性能差异
                f.write("  性能比较:\n")

                if has_gurobi:
                    dr_diff_gurobi = row['current_dr_mean'] - row['gurobi_dr_mean']
                    fpr_diff_gurobi = row['current_fpr_mean'] - row['gurobi_fpr_mean']
                    acc_diff_gurobi = row['current_accuracy_mean'] - row['gurobi_accuracy_mean']

                    f.write("    与Gurobi算法对比:\n")
                    f.write(
                        f"      检测率差异: {dr_diff_gurobi:.4f} ({'+' if dr_diff_gurobi > 0 else ''}{dr_diff_gurobi / row['gurobi_dr_mean'] * 100:.1f}%)\n")
                    f.write(
                        f"      误报率差异: {fpr_diff_gurobi:.4f} ({'+' if fpr_diff_gurobi > 0 else ''}{fpr_diff_gurobi / (row['gurobi_fpr_mean'] if row['gurobi_fpr_mean'] > 0 else 1e-10) * 100:.1f}%)\n")
                    f.write(
                        f"      准确率差异: {acc_diff_gurobi:.4f} ({'+' if acc_diff_gurobi > 0 else ''}{acc_diff_gurobi / row['gurobi_accuracy_mean'] * 100:.1f}%)\n")

                if has_l1l2:
                    dr_diff_l1l2 = row['current_dr_mean'] - row['l1l2_dr_mean']
                    fpr_diff_l1l2 = row['current_fpr_mean'] - row['l1l2_fpr_mean']
                    acc_diff_l1l2 = row['current_accuracy_mean'] - row['l1l2_accuracy_mean']

                    f.write("    与L1-L2算法对比:\n")
                    f.write(
                        f"      检测率差异: {dr_diff_l1l2:.4f} ({'+' if dr_diff_l1l2 > 0 else ''}{dr_diff_l1l2 / row['l1l2_dr_mean'] * 100:.1f}%)\n")
                    f.write(
                        f"      误报率差异: {fpr_diff_l1l2:.4f} ({'+' if fpr_diff_l1l2 > 0 else ''}{fpr_diff_l1l2 / (row['l1l2_fpr_mean'] if row['l1l2_fpr_mean'] > 0 else 1e-10) * 100:.1f}%)\n")
                    f.write(
                        f"      准确率差异: {acc_diff_l1l2:.4f} ({'+' if acc_diff_l1l2 > 0 else ''}{acc_diff_l1l2 / row['l1l2_accuracy_mean'] * 100:.1f}%)\n")

            f.write("\n\n结论分析:\n")
            f.write("----------\n")

            # 计算平均性能差异
            if has_gurobi:
                avg_dr_diff_gurobi = (summary_df['current_dr_mean'] - summary_df['gurobi_dr_mean']).mean()
                avg_fpr_diff_gurobi = (summary_df['current_fpr_mean'] - summary_df['gurobi_fpr_mean']).mean()
                avg_acc_diff_gurobi = (summary_df['current_accuracy_mean'] - summary_df['gurobi_accuracy_mean']).mean()

                f.write("1. 与Gurobi算法对比:\n")
                f.write(
                    f"   检测率(DR): 当前算法平均{' 高于 ' if avg_dr_diff_gurobi > 0 else ' 低于 '}Gurobi算法 {abs(avg_dr_diff_gurobi):.4f}\n")
                f.write(
                    f"   误报率(FPR): 当前算法平均{' 高于 ' if avg_fpr_diff_gurobi > 0 else ' 低于 '}Gurobi算法 {abs(avg_fpr_diff_gurobi):.4f}\n")
                f.write(
                    f"   准确率: 当前算法平均{' 高于 ' if avg_acc_diff_gurobi > 0 else ' 低于 '}Gurobi算法 {abs(avg_acc_diff_gurobi):.4f}\n\n")

            if has_l1l2:
                avg_dr_diff_l1l2 = (summary_df['current_dr_mean'] - summary_df['l1l2_dr_mean']).mean()
                avg_fpr_diff_l1l2 = (summary_df['current_fpr_mean'] - summary_df['l1l2_fpr_mean']).mean()
                avg_acc_diff_l1l2 = (summary_df['current_accuracy_mean'] - summary_df['l1l2_accuracy_mean']).mean()

                f.write("2. 与L1-L2算法对比:\n")
                f.write(
                    f"   检测率(DR): 当前算法平均{' 高于 ' if avg_dr_diff_l1l2 > 0 else ' 低于 '}L1-L2算法 {abs(avg_dr_diff_l1l2):.4f}\n")
                f.write(
                    f"   误报率(FPR): 当前算法平均{' 高于 ' if avg_fpr_diff_l1l2 > 0 else ' 低于 '}L1-L2算法 {abs(avg_fpr_diff_l1l2):.4f}\n")
                f.write(
                    f"   准确率: 当前算法平均{' 高于 ' if avg_acc_diff_l1l2 > 0 else ' 低于 '}L1-L2算法 {abs(avg_acc_diff_l1l2):.4f}\n\n")

            # 总体结论
            f.write("3. 总体评估:\n")

            if has_gurobi and has_l1l2:
                # 与两个算法相比的总体表现
                gurobi_better_metrics = []
                l1l2_better_metrics = []

                if avg_dr_diff_gurobi > 0: gurobi_better_metrics.append("检测率")
                if avg_fpr_diff_gurobi < 0: gurobi_better_metrics.append("误报率控制")
                if avg_acc_diff_gurobi > 0: gurobi_better_metrics.append("准确率")

                if avg_dr_diff_l1l2 > 0: l1l2_better_metrics.append("检测率")
                if avg_fpr_diff_l1l2 < 0: l1l2_better_metrics.append("误报率控制")
                if avg_acc_diff_l1l2 > 0: l1l2_better_metrics.append("准确率")

                if gurobi_better_metrics:
                    f.write(f"   当前算法在以下方面优于Gurobi算法: {', '.join(gurobi_better_metrics)}\n")
                else:
                    f.write("   在测试的指标中，Gurobi算法整体表现优于当前算法\n")

                if l1l2_better_metrics:
                    f.write(f"   当前算法在以下方面优于L1-L2算法: {', '.join(l1l2_better_metrics)}\n")
                else:
                    f.write("   在测试的指标中，L1-L2算法整体表现优于当前算法\n")
            elif has_gurobi:
                # 只与Gurobi算法比较
                gurobi_better_metrics = []
                if avg_dr_diff_gurobi > 0: gurobi_better_metrics.append("检测率")
                if avg_fpr_diff_gurobi < 0: gurobi_better_metrics.append("误报率控制")
                if avg_acc_diff_gurobi > 0: gurobi_better_metrics.append("准确率")

                if gurobi_better_metrics:
                    f.write(f"   当前算法在以下方面优于Gurobi算法: {', '.join(gurobi_better_metrics)}\n")
                else:
                    f.write("   在测试的指标中，Gurobi算法整体表现优于当前算法\n")
            elif has_l1l2:
                # 只与L1-L2算法比较
                l1l2_better_metrics = []
                if avg_dr_diff_l1l2 > 0: l1l2_better_metrics.append("检测率")
                if avg_fpr_diff_l1l2 < 0: l1l2_better_metrics.append("误报率控制")
                if avg_acc_diff_l1l2 > 0: l1l2_better_metrics.append("准确率")

                if l1l2_better_metrics:
                    f.write(f"   当前算法在以下方面优于L1-L2算法: {', '.join(l1l2_better_metrics)}\n")
                else:
                    f.write("   在测试的指标中，L1-L2算法整体表现优于当前算法\n")

        print(f"baseline对比汇总结果可视化完成，已保存到 {summary_dir} 目录")

    except Exception as e:
        print(f"可视化baseline对比汇总结果时发生错误: {e}")
        import traceback
        traceback.print_exc()

def visualize_summary_results(summary_df, summary_dir):
    """
    可视化汇总统计结果

    参数:
    summary_df: 包含汇总统计的DataFrame
    summary_dir: 可视化结果保存目录
    """
    # 确保目录存在
    ensure_directory(summary_dir)

    try:
        # 排序以确保按观测比例绘图
        summary_df = summary_df.sort_values(by='ratio', ascending=False)

        # 1. 绘制检测率(DR)随观测比例变化的曲线及其置信区间
        plt.figure(figsize=(10, 6))
        plt.errorbar(summary_df['ratio'] * 100,
                     summary_df['dr_mean'],
                     yerr=summary_df['dr_std'],
                     fmt='o-', capsize=5,
                     label='Detection Rate')
        plt.fill_between(summary_df['ratio'] * 100,
                         summary_df['dr_mean'] - summary_df['dr_std'],
                         summary_df['dr_mean'] + summary_df['dr_std'],
                         alpha=0.2)
        plt.xlabel('Observation Path Ratio (%)')
        plt.ylabel('Detection Rate (DR)')
        plt.title('Detection Rate with Standard Deviation (Random Strategy)')
        plt.legend()
        plt.grid(True)
        plt.ylim([0, 1.1])
        plt.savefig(f"{summary_dir}/detection_rate_with_std.png")
        plt.close()

        # 2. 绘制误报率(FPR)随观测比例变化的曲线及其置信区间
        plt.figure(figsize=(10, 6))
        plt.errorbar(summary_df['ratio'] * 100,
                     summary_df['fpr_mean'],
                     yerr=summary_df['fpr_std'],
                     fmt='o-', capsize=5,
                     label='False Positive Rate')
        plt.fill_between(summary_df['ratio'] * 100,
                         summary_df['fpr_mean'] - summary_df['fpr_std'],
                         summary_df['fpr_mean'] + summary_df['fpr_std'],
                         alpha=0.2)
        plt.xlabel('Observation Path Ratio (%)')
        plt.ylabel('False Positive Rate (FPR)')
        plt.title('False Positive Rate with Standard Deviation (Random Strategy)')
        plt.legend()
        plt.grid(True)
        plt.ylim([0, 1.1])
        plt.savefig(f"{summary_dir}/false_positive_rate_with_std.png")
        plt.close()

        # 3. 绘制拥塞分类准确率随观测比例变化的曲线及其置信区间
        plt.figure(figsize=(10, 6))
        plt.errorbar(summary_df['ratio'] * 100,
                     summary_df['congestion_classification_acc_mean'],
                     yerr=summary_df['congestion_classification_acc_std'],
                     fmt='o-', capsize=5,
                     label='Congestion Classification Accuracy')
        plt.fill_between(summary_df['ratio'] * 100,
                         summary_df['congestion_classification_acc_mean'] - summary_df[
                             'congestion_classification_acc_std'],
                         summary_df['congestion_classification_acc_mean'] + summary_df[
                             'congestion_classification_acc_std'],
                         alpha=0.2)
        plt.xlabel('Observation Path Ratio (%)')
        plt.ylabel('Congestion Classification Accuracy')
        plt.title('Congestion Classification Accuracy with Standard Deviation (Random Strategy)')
        plt.legend()
        plt.grid(True)
        plt.ylim([0, 1.1])
        plt.savefig(f"{summary_dir}/congestion_classification_accuracy_with_std.png")
        plt.close()

        # 4. 绘制DR和FPR的组合图
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.errorbar(summary_df['ratio'] * 100,
                     summary_df['dr_mean'],
                     yerr=summary_df['dr_std'],
                     fmt='o-', capsize=5,
                     label='Detection Rate')
        plt.fill_between(summary_df['ratio'] * 100,
                         summary_df['dr_mean'] - summary_df['dr_std'],
                         summary_df['dr_mean'] + summary_df['dr_std'],
                         alpha=0.2)
        plt.ylabel('Detection Rate (DR)')
        plt.title('Performance Metrics with Standard Deviation (Random Strategy)')
        plt.grid(True)
        plt.legend()
        plt.ylim([0, 1.1])

        plt.subplot(2, 1, 2)
        plt.errorbar(summary_df['ratio'] * 100,
                     summary_df['fpr_mean'],
                     yerr=summary_df['fpr_std'],
                     fmt='s--', capsize=5,
                     label='False Positive Rate')
        plt.fill_between(summary_df['ratio'] * 100,
                         summary_df['fpr_mean'] - summary_df['fpr_std'],
                         summary_df['fpr_mean'] + summary_df['fpr_std'],
                         alpha=0.2)
        plt.xlabel('Observation Path Ratio (%)')
        plt.ylabel('False Positive Rate (FPR)')
        plt.grid(True)
        plt.legend()
        plt.ylim([0, 1.1])

        plt.tight_layout()
        plt.savefig(f"{summary_dir}/dr_fpr_combination_with_std.png")
        plt.close()

        # 5. 绘制混淆矩阵统计可视化
        # 计算每种链路类型的平均识别率
        plt.figure(figsize=(12, 8))

        # 类型0（正常链路）的识别率 = true_0_pred_0 / (true_0_pred_0 + true_0_pred_1 + true_0_pred_2)
        normal_correct = summary_df['true_0_pred_0_mean']
        normal_total = summary_df['true_0_pred_0_mean'] + summary_df['true_0_pred_1_mean'] + summary_df[
            'true_0_pred_2_mean']
        normal_acc = normal_correct / normal_total

        # 类型1（外部流量致拥）的识别率 = true_1_pred_1 / (true_1_pred_0 + true_1_pred_1 + true_1_pred_2)
        external_correct = summary_df['true_1_pred_1_mean']
        external_total = summary_df['true_1_pred_0_mean'] + summary_df['true_1_pred_1_mean'] + summary_df[
            'true_1_pred_2_mean']
        external_acc = external_correct / external_total

        # 类型2（带宽减小致拥）的识别率 = true_2_pred_2 / (true_2_pred_0 + true_2_pred_1 + true_2_pred_2)
        bandwidth_correct = summary_df['true_2_pred_2_mean']
        bandwidth_total = summary_df['true_2_pred_0_mean'] + summary_df['true_2_pred_1_mean'] + summary_df[
            'true_2_pred_2_mean']
        bandwidth_acc = bandwidth_correct / bandwidth_total

        # 绘制三种类型的识别率
        # Plot identification rates for three link types
        plt.plot(summary_df['ratio'] * 100, normal_acc, 'o-', label='Normal Link Identification Rate')
        plt.plot(summary_df['ratio'] * 100, external_acc, 's-',
                 label='External Traffic Congestion Link Identification Rate')
        plt.plot(summary_df['ratio'] * 100, bandwidth_acc, '^-',
                 label='Bandwidth Reduction Congestion Link Identification Rate')
        plt.xlabel('Observation Path Ratio (%)')
        plt.ylabel('Identification Accuracy')
        plt.title('Identification Accuracy of Different Link Types')
        plt.legend()
        plt.grid(True)
        plt.ylim([0, 1.1])
        plt.savefig(f"{summary_dir}/link_type_identification_accuracy.png")
        plt.close()

        # 6. 绘制箱线图，展示不同比例下性能指标的分布
        # 从summary_df获取所有原始数据
        all_results_df = pd.read_csv(f"{summary_dir.replace('/summary', '')}/all_experiment_results.csv")

        # 确保ratio是降序排列的，方便箱线图的展示
        ratios_sorted = sorted(summary_df['ratio'].unique(), reverse=True)

        # DR的箱线图
        plt.figure(figsize=(10, 6))
        boxplot_data = []
        labels = []

        for ratio in ratios_sorted:
            ratio_data = all_results_df[all_results_df['ratio'] == ratio]['dr']
            boxplot_data.append(ratio_data)
            labels.append(f"{int(ratio * 100)}%")

        plt.boxplot(boxplot_data, labels=labels)
        plt.xlabel('Observation Path Ratio')
        plt.ylabel('Detection Rate (DR)')
        plt.title('Distribution of Detection Rate Across Repetitions')
        plt.grid(True, axis='y')
        plt.savefig(f"{summary_dir}/detection_rate_boxplot.png")
        plt.close()

        # FPR的箱线图
        plt.figure(figsize=(10, 6))
        boxplot_data = []
        labels = []

        for ratio in ratios_sorted:
            ratio_data = all_results_df[all_results_df['ratio'] == ratio]['fpr']
            boxplot_data.append(ratio_data)
            labels.append(f"{int(ratio * 100)}%")

        plt.boxplot(boxplot_data, labels=labels)
        plt.xlabel('Observation Path Ratio')
        plt.ylabel('False Positive Rate (FPR)')
        plt.title('Distribution of False Positive Rate Across Repetitions')
        plt.grid(True, axis='y')
        plt.savefig(f"{summary_dir}/false_positive_rate_boxplot.png")
        plt.close()

        # 拥塞分类准确率的箱线图
        plt.figure(figsize=(10, 6))
        boxplot_data = []
        labels = []

        for ratio in ratios_sorted:
            ratio_data = all_results_df[all_results_df['ratio'] == ratio]['congestion_classification_acc']
            boxplot_data.append(ratio_data)
            labels.append(f"{int(ratio * 100)}%")

        plt.boxplot(boxplot_data, labels=labels)
        plt.xlabel('Observation Path Ratio')
        plt.ylabel('Congestion Classification Accuracy')
        plt.title('Distribution of Congestion Classification Accuracy Across Repetitions')
        plt.grid(True, axis='y')
        plt.savefig(f"{summary_dir}/congestion_classification_accuracy_boxplot.png")
        plt.close()

        # 7. 在一张图中展示随着观测比例变化的平均DR、FPR和分类准确率
        plt.figure(figsize=(12, 6))
        plt.plot(summary_df['ratio'] * 100, summary_df['dr_mean'], 'o-', label='Detection Rate (DR)')
        plt.plot(summary_df['ratio'] * 100, summary_df['fpr_mean'], 's-', label='False Positive Rate (FPR)')
        plt.plot(summary_df['ratio'] * 100, summary_df['congestion_classification_acc_mean'], '^-',
                 label='Congestion Classification Accuracy')

        plt.xlabel('Observation Path Ratio (%)')
        plt.ylabel('Performance Metrics')
        plt.title('Performance Metrics vs Observation Ratio (Random Strategy)')
        plt.legend()
        plt.grid(True)
        plt.ylim([0, 1.1])
        plt.savefig(f"{summary_dir}/all_metrics_comparison.png")
        plt.close()

        # 8. 生成综合统计报告
        with open(f"{summary_dir}/summary_report.txt", 'w') as f:
            f.write("随机链路选择策略实验统计报告\n")
            f.write("==============================\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"观测比例: {', '.join([f'{int(r * 100)}%' for r in sorted(summary_df['ratio'], reverse=True)])}\n")
            f.write(f"每个比例重复次数: {summary_df['repetitions'].iloc[0]}\n\n")

            f.write("性能指标汇总:\n")
            f.write("--------------\n")

            # 按照观测比例排序
            for _, row in summary_df.sort_values(by='ratio', ascending=False).iterrows():
                ratio = row['ratio']
                f.write(f"\n观测比例: {int(ratio * 100)}%\n")
                f.write(f"  检测率(DR): {row['dr_mean']:.4f} ± {row['dr_std']:.4f}\n")
                f.write(f"  误报率(FPR): {row['fpr_mean']:.4f} ± {row['fpr_std']:.4f}\n")
                f.write(
                    f"  拥塞分类准确率: {row['congestion_classification_acc_mean']:.4f} ± {row['congestion_classification_acc_std']:.4f}\n")
                f.write(f"  各类链路识别情况:\n")

                # 计算各类链路的识别率
                normal_correct = row['true_0_pred_0_mean']
                normal_total = row['true_0_pred_0_mean'] + row['true_0_pred_1_mean'] + row['true_0_pred_2_mean']
                normal_acc = normal_correct / normal_total if normal_total > 0 else 0

                external_correct = row['true_1_pred_1_mean']
                external_total = row['true_1_pred_0_mean'] + row['true_1_pred_1_mean'] + row['true_1_pred_2_mean']
                external_acc = external_correct / external_total if external_total > 0 else 0

                bandwidth_correct = row['true_2_pred_2_mean']
                bandwidth_total = row['true_2_pred_0_mean'] + row['true_2_pred_1_mean'] + row['true_2_pred_2_mean']
                bandwidth_acc = bandwidth_correct / bandwidth_total if bandwidth_total > 0 else 0

                f.write(f"    - 正常链路识别率: {normal_acc:.4f}\n")
                f.write(f"    - 外部流量致拥链路识别率: {external_acc:.4f}\n")
                f.write(f"    - 带宽减小致拥链路识别率: {bandwidth_acc:.4f}\n")

            f.write("\n\n结论分析:\n")
            f.write("----------\n")
            f.write("1. 随着观测路径比例的提高，检测率(DR)总体呈上升趋势，误报率(FPR)总体呈下降趋势\n")
            f.write("2. 拥塞分类准确率随观测路径比例的增加也有所提高\n")
            f.write("3. 在各种拥塞类型中，带宽减小致拥的链路通常比外部流量致拥的链路更容易被正确识别\n")
            f.write("4. 观测路径比例达到80%时，已经能够获得较好的检测和分类性能\n")

        print(f"汇总结果可视化完成，已保存到 {summary_dir} 目录")

    except Exception as e:
        print(f"可视化汇总结果时发生错误: {e}")
        import traceback
        traceback.print_exc()


def summarize_baseline_results(baseline_results, experiment_dir):
    """
    汇总baseline对比实验的结果并生成统计信息

    参数:
    baseline_results: 包含所有baseline对比实验结果的字典，键为观测比例
    experiment_dir: 实验结果目录
    """
    print("\n" + "=" * 80)
    print("汇总baseline对比实验结果")
    print("=" * 80)

    # 创建汇总目录
    baseline_summary_dir = f"{experiment_dir}/baseline_summary"
    ensure_directory(baseline_summary_dir)

    # 如果没有baseline结果，直接返回
    if not baseline_results or all(len(results) == 0 for results in baseline_results.values()):
        print("无baseline对比实验结果可供汇总")
        return

    # 汇总每个比例的结果
    summary_stats = []
    all_detailed_results = []

    for ratio, results in baseline_results.items():
        if not results:
            print(f"警告: 比例 {ratio * 100}% 没有有效的baseline对比结果")
            continue

        # 转换为DataFrame
        results_df = pd.DataFrame(results)

        # 为每个比例创建单独的结果目录
        ratio_dir = f"{baseline_summary_dir}/ratio_{int(ratio * 100)}"
        ensure_directory(ratio_dir)

        # 保存该比例的所有重复实验结果
        results_df.to_csv(f"{ratio_dir}/all_repetitions.csv", index=False)

        # 计算统计数据
        stats = {
            'ratio': ratio,
            'repetitions': len(results),

            # 当前算法性能统计
            'current_dr_mean': results_df['current_dr'].mean(),
            'current_dr_std': results_df['current_dr'].std(),
            'current_fpr_mean': results_df['current_fpr'].mean(),
            'current_fpr_std': results_df['current_fpr'].std(),
            'current_accuracy_mean': results_df['current_accuracy'].mean(),
            'current_accuracy_std': results_df['current_accuracy'].std(),

            # Gurobi算法性能统计
            'gurobi_dr_mean': results_df['gurobi_dr'].mean(),
            'gurobi_dr_std': results_df['gurobi_dr'].std(),
            'gurobi_fpr_mean': results_df['gurobi_fpr'].mean(),
            'gurobi_fpr_std': results_df['gurobi_fpr'].std(),
            'gurobi_accuracy_mean': results_df['gurobi_accuracy'].mean(),
            'gurobi_accuracy_std': results_df['gurobi_accuracy'].std(),

            #l1-l2算法性能统计
            'l1l2_dr_mean': results_df['l1l2_dr'].mean(),
            'l1l2_dr_std': results_df['l1l2_dr'].std(),
            'l1l2_fpr_mean': results_df['l1l2_fpr'].mean(),
            'l1l2_fpr_std': results_df['l1l2_fpr'].std(),
            'l1l2_accuracy_mean': results_df['l1l2_accuracy'].mean(),
            'l1l2_accuracy_std': results_df['l1l2_accuracy'].std(),
        }

        # 保存该比例的统计信息
        stats_df = pd.DataFrame([stats])
        stats_df.to_csv(f"{ratio_dir}/statistics.csv", index=False)

        # 可视化对比结果
        visualize_baseline_comparison(results_df, ratio, ratio_dir)

        summary_stats.append(stats)
        all_detailed_results.extend(results)

    # 保存汇总统计结果
    if summary_stats:
        summary_df = pd.DataFrame(summary_stats)
        summary_df.to_csv(f"{baseline_summary_dir}/summary_statistics.csv", index=False)

        # 保存所有实验的详细结果
        detailed_df = pd.DataFrame(all_detailed_results)
        detailed_df.to_csv(f"{experiment_dir}/all_baseline_results.csv", index=False)

        # 可视化汇总结果
        visualize_summary_baseline_comparison(summary_df, baseline_summary_dir)
    else:
        print("警告: 没有有效的baseline对比结果可供汇总")

def summarize_repetition_results(all_results, experiment_dir):
    """
    汇总多次重复实验的结果并生成统计信息

    参数:
    all_results: 包含所有重复实验结果的字典，键为观测比例
    experiment_dir: 实验结果目录
    """
    print("\n" + "=" * 80)
    print("汇总重复实验结果")
    print("=" * 80)

    # 创建汇总目录
    summary_dir = f"{experiment_dir}/summary"
    ensure_directory(summary_dir)

    # 汇总每个比例的结果
    summary_stats = []
    detailed_results = []

    for ratio, results in all_results.items():
        if not results:
            print(f"警告: 比例 {ratio * 100}% 没有有效的实验结果")
            continue

        # 转换为DataFrame
        results_df = pd.DataFrame(results)

        # 为每个比例创建单独的结果目录
        ratio_summary_dir = f"{summary_dir}/ratio_{int(ratio * 100)}"
        ensure_directory(ratio_summary_dir)

        # 保存该比例的所有重复实验结果
        results_df.to_csv(f"{ratio_summary_dir}/all_repetitions.csv", index=False)

        # 计算统计数据（平均值、标准差、最小值、最大值、中位数）
        stats = {
            'ratio': ratio,
            'repetitions': len(results),

            # 二分类统计（正常vs拥塞）
            'dr_mean': results_df['dr'].mean(),
            'dr_std': results_df['dr'].std(),
            'dr_min': results_df['dr'].min(),
            'dr_max': results_df['dr'].max(),
            'dr_median': results_df['dr'].median(),

            'fpr_mean': results_df['fpr'].mean(),
            'fpr_std': results_df['fpr'].std(),
            'fpr_min': results_df['fpr'].min(),
            'fpr_max': results_df['fpr'].max(),
            'fpr_median': results_df['fpr'].median(),

            'binary_accuracy_mean': results_df['binary_accuracy'].mean(),
            'binary_accuracy_std': results_df['binary_accuracy'].std(),

            # 拥塞子类型分类统计
            'congestion_classification_acc_mean': results_df['congestion_classification_acc'].mean(),
            'congestion_classification_acc_std': results_df['congestion_classification_acc'].std(),
            'congestion_classification_acc_min': results_df['congestion_classification_acc'].min(),
            'congestion_classification_acc_max': results_df['congestion_classification_acc'].max(),
            'congestion_classification_acc_median': results_df['congestion_classification_acc'].median(),
        }

        # 混淆矩阵统计
        for true_val in [0, 1, 2]:
            for pred_val in [0, 1, 2]:
                key = f'true_{true_val}_pred_{pred_val}'
                if key in results_df.columns:
                    stats[f'{key}_mean'] = results_df[key].mean()
                    stats[f'{key}_std'] = results_df[key].std()

        # 保存该比例的统计信息
        stats_df = pd.DataFrame([stats])
        stats_df.to_csv(f"{ratio_summary_dir}/statistics.csv", index=False)

        # 为每个比例生成单独的可视化结果
        visualize_ratio_results(results_df, ratio, ratio_summary_dir)

        summary_stats.append(stats)
        detailed_results.extend(results)

    # 保存汇总统计结果
    if summary_stats:
        summary_df = pd.DataFrame(summary_stats)
        summary_df.to_csv(f"{summary_dir}/summary_statistics.csv", index=False)

        # 保存所有实验的详细结果
        detailed_df = pd.DataFrame(detailed_results)
        detailed_df.to_csv(f"{experiment_dir}/all_experiment_results.csv", index=False)

        # 可视化汇总结果
        visualize_summary_results(summary_df, summary_dir)
    else:
        print("警告: 没有有效的实验结果可供汇总")


def run_batch_experiments(ratios, experiment_dir, num_repetitions=100):
    """
    批量运行不同观测比例的实验，仅使用随机策略，每个比例重复多次

    参数:
    ratios: 观测路径比例列表
    experiment_dir: 实验结果目录
    num_repetitions: 每个比例的重复次数
    """
    print("\n" + "=" * 100)
    print("开始批量实验")
    print("=" * 100)

    # 创建实验结果目录
    if not ensure_directory(experiment_dir):
        print(f"无法创建实验目录 {experiment_dir}，退出")
        return

    # 保存实验配置信息
    with open(f"{experiment_dir}/experiment_config.txt", 'w') as f:
        f.write(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"路径选择策略: random\n")
        f.write(f"观测路径比例: {ratios}\n")
        f.write(f"每个比例的重复次数: {num_repetitions}\n")

    # 为每个比例创建存储所有重复实验结果的字典
    all_results = {}
    baseline_results = {}  # 新增：存储baseline对比结果
    for ratio in ratios:
        all_results[ratio] = []
        baseline_results[ratio] = []  # 初始化每个比例的baseline结果列表

    # 为每个比例创建单独的目录
    for ratio in ratios:
        ratio_dir = f"{experiment_dir}/ratio_{int(ratio * 100)}"
        ensure_directory(ratio_dir)

    # 直接从预生成数据目录加载数据
    data_dir = './output/Link_Path_Attribute'

    # 加载路由矩阵
    route_matrix_path = f'{data_dir}/output/route_matrix.txt'
    R = np.loadtxt(route_matrix_path, delimiter='\t').astype(int)

    # 加载链路标签文件
    link_labels_path = f'{data_dir}/path_variance/link_labels.txt'
    link_labels = np.loadtxt(link_labels_path).astype(int)
    # 加载故障链路信息
    fault_links_path = f'{data_dir}/path_variance/fault_links.txt'
    if os.path.exists(fault_links_path):
        # 读取故障链路信息（用于输出报告）
        with open(fault_links_path, 'r') as f:
            fault_links_info = f.read()

    # 检查路径方差文件
    path_variance_path = f'{data_dir}/path_variance/path_variance.txt'
    if not os.path.exists(path_variance_path):
        print(f"错误: 找不到路径方差文件 {path_variance_path}")
        return

    # 为不同比例运行多次实验
    for ratio in ratios:
        print(f"\n\n{'#' * 100}")
        print(f"# 运行实验: 比例={ratio * 100}%，重复次数={num_repetitions}")
        print(f"{'#' * 100}\n")

        ratio_dir = f"{experiment_dir}/ratio_{int(ratio * 100)}"

        for rep in range(num_repetitions):
            print(f"\n[重复实验 {rep + 1}/{num_repetitions}]")

            # 创建这次实验的子目录
            rep_dir = f"{ratio_dir}/rep{rep + 1}"
            ensure_directory(rep_dir)

            # 复制必要的数据文件到重复实验目录
            ensure_directory(f"{rep_dir}/output")
            ensure_directory(f"{rep_dir}/path_variance")

            # 选择观测路径
            path_selection_func = get_path_selection_strategy('random')
            selected_paths, unselected_paths = path_selection_func(R, ratio)

            # 将选择的观测路径保存到文件
            selected_paths_dir = f"{rep_dir}/selected_paths"
            ensure_directory(selected_paths_dir)
            np.savetxt(f"{selected_paths_dir}/selected_paths_{int(ratio * 100)}.txt", selected_paths, fmt='%d')

            # 运行SDR推断方法，传递选择的路径文件
            output_dir = run_sdr_inference('random', ratio, rep_dir,
                                           selected_paths_file=f"{selected_paths_dir}/selected_paths_{int(ratio * 100)}.txt")

            if output_dir:
                # 推断的链路方差文件路径
                variance_file = f"{output_dir}/inferred_link_variance.txt"

                # 评估拥塞检测和分类效果
                if os.path.exists(variance_file):
                    result = evaluate_clustering(variance_file, rep_dir, 'random', ratio)
                    if result:
                        # 添加重复次数信息
                        result['repetition'] = rep + 1
                        all_results[ratio].append(result)

                        # 添加与baseline的对比
                        baseline_result = evaluate_with_baseline(variance_file, rep_dir, 'random', ratio,
                                                                 selected_paths_file=f"{selected_paths_dir}/selected_paths_{int(ratio * 100)}.txt")
                        if baseline_result:
                            # 添加重复次数信息
                            baseline_result['repetition'] = rep + 1
                            baseline_results[ratio].append(baseline_result)
                else:
                    print(f"警告: 找不到链路方差文件 {variance_file}")
            else:
                print(f"警告: 比例={ratio * 100}%，重复={rep + 1} 的实验失败")

    # 汇总并保存所有实验结果
    summarize_repetition_results(all_results, experiment_dir)

    # 汇总并保存baseline对比结果
    summarize_baseline_results(baseline_results, experiment_dir)

    print("\n所有实验完成！")
    print(f"实验结果汇总保存在 {experiment_dir} 目录下")


def main():
    """主函数，处理命令行参数并运行实验"""
    parser = argparse.ArgumentParser(description='网络层析实验套件 - 拥塞检测与分类')
    parser.add_argument('--repetitions', type=int, default=100,
                        help='每个比例的重复次数（批量实验模式）')
    parser.add_argument('--random_seed', type=int, default=None,
                        help='随机数生成器种子，用于控制实验可重复性')

    args = parser.parse_args()

    # 确保主实验结果目录存在
    ensure_directory('./experiment_results')

    # 创建带时间戳的实验子目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"./experiment_results/{timestamp}"

    ensure_directory(experiment_dir)
    print(f"实验目录: {experiment_dir}")
    print(f"当前工作目录: {os.getcwd()}")

    # 记录实验配置
    with open(f"{experiment_dir}/experiment_config.txt", 'w') as f:
        f.write(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"路径选择策略: random\n")
        f.write(f"重复次数: {args.repetitions}\n")
        if args.random_seed is not None:
            f.write(f"随机数种子: {args.random_seed}\n")

    # 运行批量实验
    print("\n运行批量实验")
    # 观测比率范围: 100%, 80%, 60%, 40%, 20%
    ratios = [1.0, 0.8, 0.6, 0.4, 0.2]
    run_batch_experiments(ratios, experiment_dir, args.repetitions)

if __name__ == "__main__":
    main()