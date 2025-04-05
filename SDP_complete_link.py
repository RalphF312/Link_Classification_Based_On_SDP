import numpy as np
import cvxpy as cp
import os
import pandas as pd
import argparse
import time
import sys
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix


def load_route_matrix(file_path='./route_matrix.txt'):
    """加载路由矩阵"""
    R = np.loadtxt(file_path, delimiter=None).astype(int)
    assert np.all(np.logical_or(R == 0, R == 1)), "路由矩阵必须为0-1矩阵"
    return R


def load_path_variance_data(file_path='./simulation_results/path_delay_variances.csv'):
    """加载路径方差数据"""
    try:
        # 尝试加载CSV格式数据
        path_variance_df = pd.read_csv(file_path)
        return path_variance_df
    except Exception as e:
        print(f"加载路径方差数据出错: {e}")
        return None


def load_link_types_data(file_path='./simulation_results/link_types.csv'):
    """加载链路类型数据（真实值）用于评估"""
    try:
        # 尝试加载CSV格式数据
        link_types_df = pd.read_csv(file_path)
        return link_types_df
    except Exception as e:
        print(f"加载链路类型数据出错: {e}")
        return None


def alg_SDP_var(Y_obs, A_rm):
    """
    使用半正定规划(SDP)推断链路方差

    参数:
    Y_obs: 观测路径的方差
    A_rm: 路由矩阵

    返回:
    链路方差估计值
    """
    num_path_obs, num_link = A_rm.shape

    # 定义半正定矩阵变量X
    X = cp.Variable((num_link, num_link), symmetric=True)

    # 添加半正定约束
    constraints = [X >> 0]

    # 添加非对角元素为0的约束（隐含了链路延迟相互独立的假设）
    Y0, A0 = [0], [np.ones((num_link, num_link)) - np.eye(num_link)]
    constraints += [cp.trace(A0[i] @ X) == Y0[i] for i in range(len(A0))]

    # 添加所有对角元素为非负的约束
    constraints += [X[i, i] >= 0 for i in range(num_link)]

    # 添加观测路径方差约束
    A = [np.diag(A_rm[i]) for i in range(num_path_obs)]
    constraints += [cp.trace(A[i] @ X) == Y_obs[i] for i in range(num_path_obs)]

    # 修改优化目标，添加L1正则化
    lambda_reg = 0.01  # 正则化参数
    prob = cp.Problem(cp.Minimize(cp.trace(X) + lambda_reg * cp.sum(cp.abs(cp.diag(X)))), constraints)

    try:
        # 求解优化问题
        prob.solve(solver=cp.SCS)

        if prob.status == cp.OPTIMAL or prob.status == cp.OPTIMAL_INACCURATE:
            # 提取对角线元素作为链路方差估计
            X_estimated = np.diag(X.value)
            return X_estimated
        else:
            print(f"求解器状态: {prob.status}")
            return None
    except Exception as e:
        print(f"求解过程中出错: {e}")
        return None


def classify_links_by_clustering(link_variances, output_dir=None):
    """
    使用两次聚类方法根据链路方差将链路分类

    参数:
    link_variances: 链路方差数组
    output_dir: 输出目录，用于保存聚类结果的可视化图表

    返回:
    link_types: 链路类型数组，0-正常，1-普通拥塞，2-攻击拥塞
    link_congestion: 链路拥塞状态数组，0-正常，1-拥塞
    """
    # 对数据进行预处理，取对数以减小方差的数值差异
    X = np.log(link_variances + 1).reshape(-1, 1)  # 加1避免取对数时出现问题

    # 标准化数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 第一次聚类：分成正常(0)和拥塞(1,2)两类
    kmeans1 = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels1 = kmeans1.fit_predict(X_scaled)

    # 确定哪个簇是正常链路（应该是方差较小的那个簇）
    cluster_centers_orig = np.exp(scaler.inverse_transform(kmeans1.cluster_centers_)) - 1
    normal_cluster = np.argmin(cluster_centers_orig.flatten())
    congested_cluster = 1 - normal_cluster

    # 创建初始分类：0表示正常，1表示拥塞
    link_congestion = np.zeros_like(labels1)
    link_congestion[labels1 == congested_cluster] = 1

    # 第二次聚类：只对拥塞链路进行聚类，分成普通拥塞(1)和攻击拥塞(2)
    congested_indices = np.where(link_congestion == 1)[0]

    # 如果拥塞链路不足以分成两类，则全部标记为普通拥塞
    if len(congested_indices) <= 2:
        link_types = link_congestion.copy()
        print("拥塞链路数量不足，无法进行第二次聚类。所有拥塞链路标记为普通拥塞(1)。")
    else:
        # 只对拥塞链路进行第二次聚类
        X_congested = X_scaled[congested_indices]

        kmeans2 = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels2 = kmeans2.fit_predict(X_congested)

        # 确定哪个簇是攻击拥塞（应该是方差更大的那个簇）
        cluster_centers2_orig = np.exp(scaler.inverse_transform(kmeans2.cluster_centers_)) - 1
        attack_cluster = np.argmax(cluster_centers2_orig.flatten())
        normal_congested_cluster = 1 - attack_cluster

        # 创建最终分类：0表示正常，1表示普通拥塞，2表示攻击拥塞
        link_types = link_congestion.copy()

        # 将普通拥塞标记为1，攻击拥塞标记为2
        for i, idx in enumerate(congested_indices):
            if labels2[i] == normal_congested_cluster:
                link_types[idx] = 1  # 普通拥塞
            else:
                link_types[idx] = 2  # 攻击拥塞

    # 保存聚类结果的可视化图表
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        # 创建聚类结果可视化图表
        plt.figure(figsize=(10, 6))

        # 绘制散点图，根据类型着色
        plt.scatter(np.arange(len(link_variances))[link_types == 0],
                    link_variances[link_types == 0],
                    c='green', label='正常链路', alpha=0.6)
        plt.scatter(np.arange(len(link_variances))[link_types == 1],
                    link_variances[link_types == 1],
                    c='orange', label='普通拥塞链路', alpha=0.6)
        plt.scatter(np.arange(len(link_variances))[link_types == 2],
                    link_variances[link_types == 2],
                    c='red', label='攻击拥塞链路', alpha=0.6)

        plt.yscale('log')  # 使用对数尺度更好地显示方差差异
        plt.xlabel('链路索引')
        plt.ylabel('链路方差 (对数尺度)')
        plt.title('链路分类结果')
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.3)

        # 保存图表
        plt.tight_layout()
        plt.savefig(output_path / 'link_clustering_results.png')
        plt.close()

        # 保存聚类阈值信息
        with open(output_path / 'clustering_thresholds.txt', 'w') as f:
            f.write("聚类阈值信息\n")
            f.write("=================\n\n")

            # 第一次聚类信息
            cluster_centers1 = np.exp(scaler.inverse_transform(kmeans1.cluster_centers_)) - 1
            f.write(f"第一次聚类 (正常 vs 拥塞):\n")
            f.write(f"  - 簇0中心: {cluster_centers1[0][0]:.2f}\n")
            f.write(f"  - 簇1中心: {cluster_centers1[1][0]:.2f}\n")
            f.write(f"  - 正常链路簇: {normal_cluster}\n")
            f.write(f"  - 拥塞链路簇: {congested_cluster}\n\n")

            # 如果进行了第二次聚类
            if len(congested_indices) > 2:
                cluster_centers2 = np.exp(scaler.inverse_transform(kmeans2.cluster_centers_)) - 1
                f.write(f"第二次聚类 (普通拥塞 vs 攻击拥塞):\n")
                f.write(f"  - 簇0中心: {cluster_centers2[0][0]:.2f}\n")
                f.write(f"  - 簇1中心: {cluster_centers2[1][0]:.2f}\n")
                f.write(f"  - 普通拥塞簇: {normal_congested_cluster}\n")
                f.write(f"  - 攻击拥塞簇: {attack_cluster}\n")
            else:
                f.write("第二次聚类未进行（拥塞链路数量不足）\n")

    # 输出聚类结果统计
    num_normal = np.sum(link_types == 0)
    num_congested = np.sum(link_types == 1)
    num_attacked = np.sum(link_types == 2)

    print(f"聚类分类结果统计:")
    print(f"  - 正常链路(0): {num_normal} ({num_normal / len(link_types) * 100:.1f}%)")
    print(f"  - 普通拥塞链路(1): {num_congested} ({num_congested / len(link_types) * 100:.1f}%)")
    print(f"  - 攻击拥塞链路(2): {num_attacked} ({num_attacked / len(link_types) * 100:.1f}%)")
    print(
        f"  - 拥塞链路总数: {num_congested + num_attacked} ({(num_congested + num_attacked) / len(link_types) * 100:.1f}%)")

    return link_types, link_congestion


def classify_links_by_threshold(link_variances, threshold):
    """
    使用阈值方法对链路进行分类

    参数:
    link_variances: 链路方差数组
    threshold: 方差阈值，大于该阈值的链路被标记为拥塞

    返回:
    link_types: 链路类型数组，0-正常，1-拥塞
    """
    # 创建初始分类：0表示正常，1表示拥塞
    link_congestion = np.zeros(len(link_variances), dtype=int)
    link_congestion[link_variances > threshold] = 1

    # 可选：如果需要区分普通拥塞和攻击拥塞，可以再设置一个高阈值
    high_threshold = threshold * 5  # 只是一个示例，实际需要根据数据调整
    link_types = link_congestion.copy()
    link_types[link_variances > high_threshold] = 2

    # 输出阈值分类结果统计
    num_normal = np.sum(link_types == 0)
    num_congested = np.sum(link_types == 1)
    num_attacked = np.sum(link_types == 2)

    print(f"阈值分类结果统计 (阈值={threshold}):")
    print(f"  - 正常链路(0): {num_normal} ({num_normal / len(link_types) * 100:.1f}%)")
    print(f"  - 普通拥塞链路(1): {num_congested} ({num_congested / len(link_types) * 100:.1f}%)")
    print(f"  - 攻击拥塞链路(2): {num_attacked} ({num_attacked / len(link_types) * 100:.1f}%)")
    print(
        f"  - 拥塞链路总数: {num_congested + num_attacked} ({(num_congested + num_attacked) / len(link_types) * 100:.1f}%)")

    return link_types, link_congestion


def evaluate_classification(true_link_types, pred_link_types, pred_link_congestion=None):
    """
    评估链路分类的准确性

    参数:
    true_link_types: 真实链路类型数组
    pred_link_types: 预测链路类型数组
    pred_link_congestion: 预测链路拥塞状态数组（可选）

    返回:
    metrics: 性能指标字典
    """
    # 如果没有提供拥塞状态，将类型大于0的设为拥塞
    if pred_link_congestion is None:
        pred_link_congestion = np.where(pred_link_types > 0, 1, 0)

    # 转换真实类型为二分类（拥塞或非拥塞）
    true_link_congestion = np.where(true_link_types > 0, 1, 0)

    # 计算二分类准确率和其他指标
    binary_accuracy = accuracy_score(true_link_congestion, pred_link_congestion)
    binary_precision = precision_score(true_link_congestion, pred_link_congestion, zero_division=0)
    binary_recall = recall_score(true_link_congestion, pred_link_congestion, zero_division=0)
    binary_f1 = f1_score(true_link_congestion, pred_link_congestion, zero_division=0)

    # 计算多分类准确率
    multi_accuracy = accuracy_score(true_link_types, pred_link_types)

    # 计算混淆矩阵统计
    cm = confusion_matrix(true_link_congestion, pred_link_congestion)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # 如果没有某个类别的样本，混淆矩阵可能不是2x2的
        tp = np.sum((true_link_congestion == 1) & (pred_link_congestion == 1))
        tn = np.sum((true_link_congestion == 0) & (pred_link_congestion == 0))
        fp = np.sum((true_link_congestion == 0) & (pred_link_congestion == 1))
        fn = np.sum((true_link_congestion == 1) & (pred_link_congestion == 0))

    metrics = {
        "binary_accuracy": binary_accuracy,
        "binary_precision": binary_precision,
        "binary_recall": binary_recall,
        "binary_f1": binary_f1,
        "multi_accuracy": multi_accuracy,
        "confusion_matrix": {
            "TP": int(tp),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn)
        }
    }

    return metrics


def alg_SDP_diagnosis(route_matrix_path, path_variance_path, link_types_path=None, output_dir='./sdp_results'):
    """
    执行SDP算法进行链路诊断的主函数（兼容原始接口）

    参数:
    route_matrix_path: 路由矩阵文件路径
    path_variance_path: 路径方差数据文件路径
    link_types_path: 链路类型数据文件路径（用于评估）
    output_dir: 输出目录

    返回:
    link_types: 链路类型数组
    link_congestion: 链路拥塞状态数组
    """
    print("注意: 您正在使用原始的alg_SDP_diagnosis接口，建议改用SDPNetworkDiagnosisOrchestrator类。")

    # 加载路由矩阵
    print(f"正在加载路由矩阵: {route_matrix_path}")
    R = load_route_matrix(route_matrix_path)
    print(f"路由矩阵形状: {R.shape}")
    print(f"路径数量: {R.shape[0]}")
    print(f"链路数量: {R.shape[1]}")

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # 加载路径方差数据
    print(f"正在加载路径方差数据: {path_variance_path}")
    path_variance_df = load_path_variance_data(path_variance_path)
    if path_variance_df is None:
        print("无法加载路径方差数据，诊断失败")
        return None, None

    # 获取第一行数据（假设每一行代表一次实验）
    if len(path_variance_df) > 0:
        path_variance = path_variance_df.iloc[0].values
    else:
        print("路径方差数据为空，诊断失败")
        return None, None

    # 使用SDP推断链路方差
    print("开始进行链路方差推断...")
    link_variances = alg_SDP_var(path_variance, R)

    if link_variances is not None:
        print("链路方差推断完成！")

        # 使用聚类方法对链路进行分类
        print("使用聚类方法对链路进行分类...")
        link_types, link_congestion = classify_links_by_clustering(link_variances, output_dir)

        # 如果提供了真实链路类型数据，评估分类性能
        metrics = None
        if link_types_path:
            print(f"正在加载链路类型数据用于评估: {link_types_path}")
            link_types_df = load_link_types_data(link_types_path)
            if link_types_df is not None:
                true_link_types = link_types_df.iloc[0].values
                print("评估分类性能...")
                metrics = evaluate_classification(true_link_types, link_types, link_congestion)

                print(f"性能评估:")
                print(f"  - 二分类准确率: {metrics['binary_accuracy']:.4f}")
                print(f"  - 多分类准确率: {metrics['multi_accuracy']:.4f}")
                print(f"  - 精确率: {metrics['binary_precision']:.4f}")
                print(f"  - 召回率: {metrics['binary_recall']:.4f}")
                print(f"  - F1分数: {metrics['binary_f1']:.4f}")

        # 保存结果到CSV文件
        results_df = pd.DataFrame({
            'link_id': range(len(link_variances)),
            'variance': link_variances,
            'link_type': link_types,
            'link_congestion': link_congestion
        })
        results_df.to_csv(output_path / 'sdp_diagnosis_results.csv', index=False)

        # 如果有评估指标，保存到文件
        if metrics:
            metrics_df = pd.DataFrame({
                'metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Multi-class Accuracy'],
                'value': [
                    metrics['binary_accuracy'],
                    metrics['binary_precision'],
                    metrics['binary_recall'],
                    metrics['binary_f1'],
                    metrics['multi_accuracy']
                ]
            })
            metrics_df.to_csv(output_path / 'sdp_performance_metrics.csv', index=False)

        print(f"SDP诊断结果已保存到目录: {output_dir}")
        return link_types, link_congestion
    else:
        print("链路方差推断失败")
        return None, None

class SDPNetworkDiagnosisOrchestrator:
    """
    SDP网络诊断调度器，用于协调已生成的链路数据和SDP算法的执行。
    该类假设链路性能数据已经由link_performance_generator.py生成。
    """

    def __init__(self, route_matrix_file, simulation_results_dir="./simulation_results",
                 output_dir="./diagnosis_results"):
        """
        初始化诊断调度器。

        参数:
            route_matrix_file: 路由矩阵文件的路径
            simulation_results_dir: 链路性能生成器输出的目录
            output_dir: 诊断结果的输出目录
        """
        self.route_matrix_file = route_matrix_file
        self.route_matrix = np.loadtxt(route_matrix_file, dtype=int)

        # 设置输入输出目录
        self.sim_results_dir = Path(simulation_results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.results = {
            "precision": [],
            "recall": [],
            "f1_score": [],
            "accuracy": [],
            "time_taken": [],
            "link_variances": []
        }

        # 验证模拟结果目录是否存在
        if not self.sim_results_dir.exists():
            print(f"警告：模拟结果目录 {simulation_results_dir} 不存在。")
            print("请先运行 link_performance_generator.py 生成链路性能数据。")

    def check_simulation_data_exists(self):
        """
        检查必要的模拟数据文件是否存在。

        返回:
            bool: 如果所有必要文件都存在，则返回True
        """
        required_files = [
            "link_types.csv",
            "path_delay_variances.csv"
        ]

        for file in required_files:
            if not (self.sim_results_dir / file).exists():
                print(f"错误：找不到所需的模拟数据文件 {file}")
                print(f"请确保 link_performance_generator.py 已经运行并在 {self.sim_results_dir} 中生成了所需文件。")
                return False

        return True

    def load_simulation_data(self):
        """
        加载模拟数据用于SDP算法分析。

        返回:
            simulation_data: 包含所有所需数据的字典
        """
        # 检查数据文件存在
        if not self.check_simulation_data_exists():
            raise FileNotFoundError(f"模拟数据文件缺失。请先运行链路性能生成器。")

        # 加载链路类型数据（真实值）
        link_types_df = load_link_types_data(self.sim_results_dir / "link_types.csv")

        # 加载路径延迟方差数据
        path_delay_variances_df = load_path_variance_data(self.sim_results_dir / "path_delay_variances.csv")

        # 整理数据格式
        num_links = len(link_types_df.columns)
        num_paths = len(path_delay_variances_df.columns)
        num_repeats = len(link_types_df)

        print(f"成功加载模拟数据：")
        print(f"  - 链路数量: {num_links}")
        print(f"  - 路径数量: {num_paths}")
        print(f"  - 实验次数: {num_repeats}")

        return {
            "link_types_df": link_types_df,
            "path_delay_variances_df": path_delay_variances_df,
            "num_links": num_links,
            "num_paths": num_paths,
            "num_repeats": num_repeats
        }

    def run_sdp_diagnosis(self, classification_method="clustering", threshold=None):
        """
        使用SDP算法进行网络诊断。

        参数:
            classification_method: 分类方法，可选 "clustering" 或 "threshold"
            threshold: 如果使用threshold方法，指定链路方差的阈值

        返回:
            avg_results: 包含平均性能指标的字典
        """
        print(f"开始运行SDP诊断算法...")

        try:
            # 加载模拟数据
            sim_data = self.load_simulation_data()
        except FileNotFoundError as e:
            print(f"错误：{e}")
            return None

        link_types_df = sim_data["link_types_df"]
        path_delay_variances_df = sim_data["path_delay_variances_df"]
        num_repeats = sim_data["num_repeats"]

        # 创建结果目录
        results_dir = self.output_dir / "sdp_results"
        results_dir.mkdir(exist_ok=True)

        # 创建分类方法对应的目录
        method_dir = results_dir / f"method_{classification_method}"
        if classification_method == "threshold" and threshold is not None:
            method_dir = results_dir / f"method_{classification_method}_threshold_{threshold}"
        method_dir.mkdir(exist_ok=True)

        # 预准备数据结构存储结果
        all_true_link_types = []
        all_pred_link_types = []
        all_link_variances = []

        # 对每次实验进行SDP诊断
        for repeat_idx in range(num_repeats):
            # 获取这次实验的真实链路状态
            true_link_types = np.array([
                link_types_df.iloc[repeat_idx, i]
                for i in range(len(link_types_df.columns))
            ])

            # 获取观察到的路径延迟方差
            path_variances = np.array([
                path_delay_variances_df.iloc[repeat_idx, i]
                for i in range(len(path_delay_variances_df.columns))
            ])

            # 运行SDP算法
            start_time = time.time()
            link_variances = alg_SDP_var(path_variances, self.route_matrix)

            if link_variances is None:
                print(f"实验 {repeat_idx + 1} SDP求解失败，跳过")
                continue

            # 分类链路
            if classification_method == "clustering":
                pred_link_types, pred_link_congestion = classify_links_by_clustering(
                    link_variances, output_dir=method_dir if repeat_idx == 0 else None)
            elif classification_method == "threshold":
                if threshold is None:
                    print("使用阈值分类方法时必须提供阈值参数")
                    return None
                pred_link_types, pred_link_congestion = classify_links_by_threshold(link_variances, threshold)
            else:
                print(f"不支持的分类方法: {classification_method}")
                return None

            end_time = time.time()

            # 存储结果
            all_true_link_types.append(true_link_types)
            all_pred_link_types.append(pred_link_types)
            all_link_variances.append(link_variances)

            # 计算性能指标
            metrics = evaluate_classification(true_link_types, pred_link_types, pred_link_congestion)

            precision = metrics["binary_precision"]
            recall = metrics["binary_recall"]
            f1 = metrics["binary_f1"]
            accuracy = metrics["binary_accuracy"]
            time_taken = end_time - start_time

            # 存储性能指标
            self.results["precision"].append(precision)
            self.results["recall"].append(recall)
            self.results["f1_score"].append(f1)
            self.results["accuracy"].append(accuracy)
            self.results["time_taken"].append(time_taken)
            self.results["link_variances"].append(link_variances)

            # 每10次实验打印一次进度
            if (repeat_idx + 1) % 10 == 0 or repeat_idx == 0:
                print(f"完成 {repeat_idx + 1}/{num_repeats} 次诊断")
                print(f"  - 精确率: {precision:.4f}")
                print(f"  - 召回率: {recall:.4f}")
                print(f"  - F1分数: {f1:.4f}")
                print(f"  - 准确率: {accuracy:.4f}")

        # 保存所有实验的链路方差
        np.save(method_dir / "link_variances.npy", np.array(all_link_variances))

        # 保存真实和预测的链路类型
        all_true_link_types = np.array(all_true_link_types)
        all_pred_link_types = np.array(all_pred_link_types)
        np.save(method_dir / "true_link_types.npy", all_true_link_types)
        np.save(method_dir / "pred_link_types.npy", all_pred_link_types)

        # 保存性能指标
        results_df = pd.DataFrame({
            "precision": self.results["precision"],
            "recall": self.results["recall"],
            "f1_score": self.results["f1_score"],
            "accuracy": self.results["accuracy"],
            "time_taken": self.results["time_taken"]
        })

        results_df.to_csv(method_dir / "performance_metrics.csv", index=False)

        # 计算平均性能指标
        avg_results = {
            "avg_precision": np.mean(self.results["precision"]),
            "avg_recall": np.mean(self.results["recall"]),
            "avg_f1_score": np.mean(self.results["f1_score"]),
            "avg_accuracy": np.mean(self.results["accuracy"]),
            "avg_time_taken": np.mean(self.results["time_taken"])
        }

        # 计算混淆矩阵
        all_true_congestion = np.where(all_true_link_types > 0, 1, 0).flatten()
        all_pred_congestion = np.where(all_pred_link_types > 0, 1, 0).flatten()
        cm = confusion_matrix(all_true_congestion, all_pred_congestion)

        # 保存摘要结果
        with open(method_dir / "summary_results.txt", "w") as f:
            f.write("# SDP算法诊断结果摘要\n\n")
            f.write(f"总实验次数: {num_repeats}\n")
            f.write(f"链路数量: {sim_data['num_links']}\n")
            f.write(f"路径数量: {sim_data['num_paths']}\n")
            f.write(f"分类方法: {classification_method}")
            if classification_method == "threshold":
                f.write(f", 阈值: {threshold}")
            f.write("\n\n")

            f.write("## 平均性能指标\n\n")
            for metric, value in avg_results.items():
                f.write(f"{metric}: {value:.4f}\n")

            f.write("\n## 混淆矩阵\n\n")
            f.write("```\n")
            if cm.shape == (2, 2):
                f.write(f"TN: {cm[0, 0]}, FP: {cm[0, 1]}\n")
                f.write(f"FN: {cm[1, 0]}, TP: {cm[1, 1]}\n")
            else:
                f.write(f"混淆矩阵: {cm}\n")
            f.write("```\n")

        print(f"SDP诊断完成，结果保存在 {method_dir}")

        return avg_results

    def analyze_threshold_impact(self, thresholds):
        """
        分析不同阈值对SDP算法分类性能的影响。

        参数:
            thresholds: 要测试的阈值列表
        """
        print("分析不同链路方差阈值对诊断性能的影响...")

        threshold_results = {
            "threshold": [],
            "precision": [],
            "recall": [],
            "f1_score": [],
            "accuracy": []
        }

        for threshold in thresholds:
            print(f"测试阈值: {threshold}")
            # 重置结果
            self.results = {
                "precision": [],
                "recall": [],
                "f1_score": [],
                "accuracy": [],
                "time_taken": [],
                "link_variances": []
            }

            # 运行SDP诊断
            avg_results = self.run_sdp_diagnosis(classification_method="threshold", threshold=threshold)

            if avg_results is None:
                print("由于数据加载问题，无法完成阈值分析。")
                return None

            # 存储结果
            threshold_results["threshold"].append(threshold)
            threshold_results["precision"].append(avg_results["avg_precision"])
            threshold_results["recall"].append(avg_results["avg_recall"])
            threshold_results["f1_score"].append(avg_results["avg_f1_score"])
            threshold_results["accuracy"].append(avg_results["avg_accuracy"])

        # 保存阈值分析结果
        threshold_df = pd.DataFrame(threshold_results)
        threshold_df.to_csv(self.output_dir / "sdp_threshold_analysis.csv", index=False)

        # 创建阈值分析的可视化图表
        plt.figure(figsize=(12, 8))

        # 绘制精确率、召回率、F1分数随阈值变化的曲线
        plt.subplot(2, 1, 1)
        plt.plot(threshold_results["threshold"], threshold_results["precision"], 'b-', label='精确率')
        plt.plot(threshold_results["threshold"], threshold_results["recall"], 'g-', label='召回率')
        plt.plot(threshold_results["threshold"], threshold_results["f1_score"], 'r-', label='F1分数')
        plt.xlabel('阈值')
        plt.ylabel('指标值')
        plt.title('SDP算法性能指标随阈值变化')
        plt.legend()
        plt.grid(True)

        # 绘制准确率随阈值变化的曲线
        plt.subplot(2, 1, 2)
        plt.plot(threshold_results["threshold"], threshold_results["accuracy"], 'k-', label='准确率')
        plt.xlabel('阈值')
        plt.ylabel('准确率')
        plt.title('SDP算法准确率随阈值变化')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(self.output_dir / "sdp_threshold_analysis.png")
        plt.close()

        print(f"阈值分析完成，结果保存在 {self.output_dir}/sdp_threshold_analysis.csv")
        print(f"阈值分析图表保存在 {self.output_dir}/sdp_threshold_analysis.png")

        # 找出最佳F1分数的阈值
        best_f1_idx = np.argmax(threshold_results["f1_score"])
        best_threshold = threshold_results["threshold"][best_f1_idx]
        best_precision = threshold_results["precision"][best_f1_idx]
        best_recall = threshold_results["recall"][best_f1_idx]
        best_f1 = threshold_results["f1_score"][best_f1_idx]
        best_accuracy = threshold_results["accuracy"][best_f1_idx]

        print(f"\n最佳阈值 (基于F1分数):")
        print(f"  - 阈值: {best_threshold}")
        print(f"  - 精确率: {best_precision:.4f}")
        print(f"  - 召回率: {best_recall:.4f}")
        print(f"  - F1分数: {best_f1:.4f}")
        print(f"  - 准确率: {best_accuracy:.4f}")

        return threshold_df


def main():
    # 路由矩阵文件路径
    route_matrix_path = "./route_matrix.txt"

    # 初始化SDP调度器
    orchestrator = SDPNetworkDiagnosisOrchestrator(route_matrix_path)

    # 确认模拟数据存在
    if not orchestrator.check_simulation_data_exists():
        print("\n请先运行 link_performance_generator.py 生成链路性能数据。")
        print("示例命令: python link_performance_generator.py")
        return

    # 使用聚类方法进行SDP诊断
    print("\n使用聚类方法进行链路分类...")
    results_clustering = orchestrator.run_sdp_diagnosis(classification_method="clustering")

    if results_clustering:
        print("\n聚类方法诊断结果摘要:")
        for metric, value in results_clustering.items():
            print(f"  {metric}: {value:.4f}")

    # 询问用户是否要测试不同阈值
    answer = input("\n是否要分析不同阈值对诊断性能的影响？(y/n): ")

    if answer.lower() == 'y':
        # 尝试使用不同的阈值进行诊断
        print("\n测试不同阈值对诊断性能的影响...")
        default_thresholds = [100, 500, 1000, 2000, 5000, 10000]
        threshold_input = input(f"请输入要测试的阈值（以逗号分隔，直接回车使用默认值 {default_thresholds}）: ")

        if threshold_input.strip():
            try:
                thresholds = [float(t.strip()) for t in threshold_input.split(',')]
            except ValueError:
                print("输入格式错误，使用默认阈值。")
                thresholds = default_thresholds
        else:
            thresholds = default_thresholds

        threshold_results = orchestrator.analyze_threshold_impact(thresholds)

        if threshold_results is not None:
            # 打印总结
            print("\n诊断结果摘要:")
            print(f"使用聚类方法:")
            for metric, value in results_clustering.items():
                print(f"  {metric}: {value:.4f}")

            print("\n最佳阈值结果:")
            best_idx = threshold_results["f1_score"].argmax()
            best_threshold = threshold_results.iloc[best_idx]
            print(f"  最佳阈值: {best_threshold['threshold']}")
            print(f"  Precision: {best_threshold['precision']:.4f}")
            print(f"  Recall: {best_threshold['recall']:.4f}")
            print(f"  F1 Score: {best_threshold['f1_score']:.4f}")
            print(f"  Accuracy: {best_threshold['accuracy']:.4f}")

    print("\n所有结果已保存到 ./diagnosis_results/sdp_results 目录")

if __name__ == "__main__":
    main()