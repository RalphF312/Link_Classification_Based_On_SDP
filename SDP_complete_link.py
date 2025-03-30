import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import os
import pandas as pd
import argparse
from path_selection_strategies import get_path_selection_strategy


def load_route_matrix(file_path='./output/route_matrix.txt'):
    """加载路由矩阵"""
    R = np.loadtxt(file_path, delimiter='\t').astype(int)
    assert np.all(np.logical_or(R == 0, R == 1)), "路由矩阵必须为0-1矩阵"
    return R


def load_variance_data():
    """加载路径方差和协方差数据"""
    # 读取路径方差数据
    path_variance = np.loadtxt('./path_variance/path_variance.txt', delimiter=',', skiprows=1, usecols=(1,))
    # 读取链路真实方差数据（用于比较）
    link_variance = np.loadtxt('./path_variance/link_variance.txt', delimiter=',', usecols=(1,), skiprows=1)
    # 读取路径协方差矩阵
    covariance_file_path = './path_variance/path_covariances.txt'
    with open(covariance_file_path, 'r') as file:
        # 跳过表头行
        next(file)
        # 读取剩余内容并转换为二维数组
        lines = file.readlines()
        rows = [list(map(float, line.strip().split(', '))) for line in lines]
        path_covariance = np.array(rows)

    return path_covariance, path_variance, link_variance


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


def save_inferred_variance(link_variances_inferred, output_dir):
    """
    仅保存推断的链路方差结果到文件，方便后续拥塞定位处理
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 将推断的链路方差保存到CSV文件
    results_df = pd.DataFrame({
        'Link_ID': range(0, len(link_variances_inferred)),
        'Inferred_Variance': link_variances_inferred
    })

    # 保存为CSV格式
    results_df.to_csv(os.path.join(output_dir, 'inferred_link_variance.csv'), index=False)

    # 同时保存为简单的文本文件，只包含方差值
    np.savetxt(os.path.join(output_dir, 'inferred_link_variance.txt'), link_variances_inferred, fmt='%.6f')

    print(f"推断的链路方差已保存到 {output_dir} 目录下")


def visualize_variance_comparison(link_variances_true, link_variances_inferred, output_dir):
    """
    Visualize direct comparison between original link variances and inferred link variances
    """
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Scatter plot comparison of true link variances vs inferred variances
    plt.figure(figsize=(10, 6))
    plt.scatter(link_variances_true, link_variances_inferred, alpha=0.6)
    max_val = max(np.max(link_variances_true), np.max(link_variances_inferred))
    plt.plot([0, max_val], [0, max_val], 'r--')
    plt.xlabel('True Link Variance')
    plt.ylabel('Inferred Link Variance')
    plt.title('Link Variance Comparison: True vs Inferred')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'link_variance_comparison.png'))

    # Bar chart comparing variances for each link
    plt.figure(figsize=(12, 6))
    x = np.arange(len(link_variances_true))
    width = 0.35
    plt.bar(x - width / 2, link_variances_true, width, label='True Variance')
    plt.bar(x + width / 2, link_variances_inferred, width, label='Inferred Variance')
    plt.xlabel('Link ID')
    plt.ylabel('Variance')
    plt.title('Variance Comparison by Link')
    plt.xticks(x, [str(i) for i in range(len(link_variances_true))], rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'link_variance_bar_comparison.png'))

    # Heat map of variance differences
    plt.figure(figsize=(10, 6))
    relative_diff = np.abs(link_variances_inferred - link_variances_true) / (link_variances_true + 1e-10)
    plt.scatter(range(len(link_variances_true)), relative_diff, c=relative_diff, cmap='viridis', alpha=0.7)
    plt.colorbar(label='Relative Difference')
    plt.xlabel('Link ID')
    plt.ylabel('Relative Difference')
    plt.title('Relative Difference in Link Variance Inference')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'link_variance_relative_diff.png'))


def save_variance_results(link_variances_true, link_variances_inferred, output_dir):
    """
    保存原始和推断的链路方差结果到文件
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 将链路方差比较结果保存到CSV文件
    results_df = pd.DataFrame({
        'Link_ID': range(0, len(link_variances_true)),
        'True_Variance': link_variances_true,
        'Inferred_Variance': link_variances_inferred,
        'Absolute_Difference': np.abs(link_variances_inferred - link_variances_true),
        'Relative_Difference': np.abs(link_variances_inferred - link_variances_true) / (link_variances_true + 1e-10)
    })

    results_df.to_csv(os.path.join(output_dir, 'link_variance_comparison.csv'), index=False)

    # 输出简要统计信息到文本文件
    with open(os.path.join(output_dir, 'variance_statistics.txt'), 'w') as f:
        f.write("链路方差推断统计信息\n")
        f.write("=================\n\n")
        f.write(f"链路总数: {len(link_variances_true)}\n")
        f.write(f"平均真实方差: {np.mean(link_variances_true):.6f}\n")
        f.write(f"平均推断方差: {np.mean(link_variances_inferred):.6f}\n")
        f.write(f"平均绝对差异: {np.mean(np.abs(link_variances_inferred - link_variances_true)):.6f}\n")
        mean_rel_diff = np.mean(np.abs(link_variances_inferred - link_variances_true) / (link_variances_true + 1e-10))
        f.write(f"平均相对差异: {mean_rel_diff:.6f}\n")


# 这个函数是SDP_complete_link.py中的main函数的修改版本
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='网络层析推断链路方差')
    parser.add_argument('--strategy', type=str, default='importance',
                        choices=['importance', 'random', 'coverage', 'clustering', 'stress', 'mixed'],
                        help='路径选择策略：importance（基于重要性）, random（随机选择）, coverage（链路覆盖）, '
                             'clustering（聚类划分）, stress（压力测试）, mixed（混合策略）')
    parser.add_argument('--ratio', type=float, default=0.25,
                        help='要选择的路径比例，默认为0.25（25%）')
    parser.add_argument('--output_dir', type=str, default='./output/sdp_variance_results',
                        help='输出目录路径')
    parser.add_argument('--congested_links', type=str, default=None,
                        help='拥塞链路索引列表，逗号分隔，例如"0,5,10"（仅用于stress策略）')
    # 新增参数
    parser.add_argument('--selected_paths_file', type=str, default=None,
                        help='包含已选择路径索引的文件路径')

    args = parser.parse_args()

    # 加载路由矩阵和方差数据
    R = load_route_matrix()
    path_covariance, path_variance, link_variance_true = load_variance_data()

    print(f"路由矩阵形状: {R.shape}")
    print(f"路径数量: {R.shape[0]}")
    print(f"链路数量: {R.shape[1]}")
    print(f"使用路径选择策略: {args.strategy}")
    print(f"选择路径比例: {args.ratio}")

    # 如果提供了选定路径文件，直接从文件加载
    if args.selected_paths_file and os.path.exists(args.selected_paths_file):
        print(f"从文件加载已选择的路径: {args.selected_paths_file}")
        selected_paths = np.loadtxt(args.selected_paths_file, dtype=int)
        print(f"选择的路径数量: {len(selected_paths)}")
    else:
        # 原有的路径选择逻辑
        # 获取路径选择策略
        path_selection_func = get_path_selection_strategy(args.strategy)

        # 处理拥塞链路参数（如果有）
        congested_links = None
        if args.congested_links and args.strategy == 'stress':
            congested_links = [int(idx) for idx in args.congested_links.split(',')]
            print(f"指定拥塞链路: {congested_links}")

        # 确保对stress策略有默认的拥塞链路
        if args.strategy == 'stress' and congested_links is None:
            # 使用一些默认的拥塞链路（根据已知情况）
            congested_links = [2, 5, 8, 10, 17, 18, 20, 24]
            print(f"未提供拥塞链路参数，使用默认拥塞链路: {congested_links}")

        # 根据选定的策略分割路径
        if args.strategy == 'stress':
            selected_paths, _ = path_selection_func(R, args.ratio, congested_links)
        elif args.strategy == 'mixed':
            # 混合策略使用默认权重
            selected_paths, _ = path_selection_func(R, args.ratio)
        else:
            selected_paths, _ = path_selection_func(R, args.ratio)

    print(f"选择的路径数量: {len(selected_paths)}")

    # 计算链路覆盖情况
    R_selected = R[selected_paths]
    covered_links = np.any(R_selected > 0, axis=0)
    num_covered_links = np.sum(covered_links)
    print(f"覆盖的链路数量: {num_covered_links}/{R.shape[1]} ({num_covered_links / R.shape[1] * 100:.1f}%)")

    # 用选定的路径进行SDP链路方差推断
    Y_selected = path_variance[selected_paths]

    print("开始进行链路方差推断...")
    link_variance_inferred = alg_SDP_var(Y_selected, R_selected)

    if link_variance_inferred is not None:
        print("链路方差推断完成！")

        # 设置输出目录
        output_dir = args.output_dir

        # 可视化真实和推断的链路方差对比
        print("生成可视化图表...")
        visualize_variance_comparison(link_variance_true, link_variance_inferred, output_dir)

        # 保存结果到文件
        print("保存结果到文件...")
        save_variance_results(link_variance_true, link_variance_inferred, output_dir)

        # 保存推断的链路方差到文件
        save_inferred_variance(link_variance_inferred, output_dir)

        print(f"所有结果已保存到目录: {output_dir}")
    else:
        print("链路方差推断失败")

if __name__ == "__main__":
    main()