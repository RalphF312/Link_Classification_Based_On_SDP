import numpy as np
from sklearn.cluster import KMeans


def split_paths_random(R, ratio):
    """
    完全随机选择路径

    参数:
    R: 路由矩阵
    ratio: 选择的路径比例

    返回:
    选择的路径索引和未选择的路径索引
    """
    num_paths = R.shape[0]
    num_selected = int(num_paths * ratio)

    # 随机打乱路径索引
    all_paths = np.arange(num_paths)
    np.random.shuffle(all_paths)

    # 选择前num_selected个路径
    selected_paths = all_paths[:num_selected]
    unselected_paths = all_paths[num_selected:]

    return selected_paths, unselected_paths


def split_paths_by_link_coverage(R, ratio):
    """
    基于链路覆盖率选择路径，确保尽可能多的链路被至少一条选定路径覆盖

    参数:
    R: 路由矩阵
    ratio: 选择的路径比例

    返回:
    选择的路径索引和未选择的路径索引
    """
    num_paths = R.shape[0]
    num_links = R.shape[1]
    num_selected = int(num_paths * ratio)

    # 贪心算法选择路径
    selected_paths = []
    covered_links = np.zeros(num_links, dtype=bool)

    # 首先选择一条覆盖最多链路的路径
    path_coverage = np.sum(R, axis=1)
    first_path = np.argmax(path_coverage)
    selected_paths.append(first_path)
    covered_links = np.logical_or(covered_links, R[first_path] > 0)

    # 继续选择能覆盖最多未覆盖链路的路径
    for _ in range(1, num_selected):
        if np.all(covered_links):  # 如果所有链路都已覆盖
            # 使用其他策略选择剩余路径
            remaining_count = num_selected - len(selected_paths)
            remaining_paths = [i for i in range(num_paths) if i not in selected_paths]

            # 按照路径覆盖链路数量排序选择剩余路径
            remaining_coverage = [np.sum(R[i]) for i in remaining_paths]
            sorted_indices = np.argsort(-np.array(remaining_coverage))
            next_paths = [remaining_paths[i] for i in sorted_indices[:remaining_count]]
            selected_paths.extend(next_paths)
            break

        # 计算每条路径能够新覆盖的链路数量
        new_coverage = np.zeros(num_paths)
        for i in range(num_paths):
            if i in selected_paths:
                continue
            # 计算该路径能新覆盖的链路数
            new_links = np.logical_and(R[i] > 0, ~covered_links)
            new_coverage[i] = np.sum(new_links)

        # 选择能新覆盖最多链路的路径
        next_path = np.argmax(new_coverage)
        if new_coverage[next_path] == 0:  # 如果没有新链路可覆盖
            # 选择覆盖已选链路最多的路径（增加冗余）
            redundant_coverage = np.zeros(num_paths)
            for i in range(num_paths):
                if i in selected_paths:
                    continue
                redundant_coverage[i] = np.sum(np.logical_and(R[i] > 0, covered_links))
            next_path = np.argmax(redundant_coverage)

        selected_paths.append(next_path)
        covered_links = np.logical_or(covered_links, R[next_path] > 0)

    # 确保选择了正确数量的路径
    if len(selected_paths) < num_selected:
        remaining_paths = [i for i in range(num_paths) if i not in selected_paths]
        np.random.shuffle(remaining_paths)
        selected_paths.extend(remaining_paths[:num_selected - len(selected_paths)])

    # 确定未选择的路径
    unselected_paths = [i for i in range(num_paths) if i not in selected_paths]

    return np.array(selected_paths), np.array(unselected_paths)


def split_paths_by_graph_partitioning(R, ratio):
    """
    使用K-means聚类将路径分组，然后从每个组中选择代表性路径

    参数:
    R: 路由矩阵
    ratio: 选择的路径比例

    返回:
    选择的路径索引和未选择的路径索引
    """
    num_paths = R.shape[0]
    num_selected = int(num_paths * ratio)

    # 使用K-means将路径聚类
    # 选择比最终需要选择的路径数量稍多的簇
    n_clusters = min(num_selected + 5, num_paths - 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(R)

    # 从每个簇中选择最接近中心的路径
    selected_paths = []
    for cluster_id in range(n_clusters):
        cluster_paths = np.where(cluster_labels == cluster_id)[0]
        if len(cluster_paths) == 0:
            continue

        # 计算到簇中心的距离
        cluster_center = kmeans.cluster_centers_[cluster_id]
        distances = np.array([np.linalg.norm(R[path] - cluster_center) for path in cluster_paths])

        # 选择最接近中心的路径
        closest_path_idx = cluster_paths[np.argmin(distances)]
        selected_paths.append(closest_path_idx)

        # 如果簇足够大，可以选择多个路径
        if len(cluster_paths) > 3 and len(selected_paths) < num_selected:
            # 选择第二接近的路径
            sorted_idx = np.argsort(distances)
            if len(sorted_idx) > 1:  # 确保簇中有至少两条路径
                second_closest_path_idx = cluster_paths[sorted_idx[1]]
                selected_paths.append(second_closest_path_idx)

    # 如果选择的路径不足，从未选择的路径中随机选择
    if len(selected_paths) < num_selected:
        remaining_paths = [i for i in range(num_paths) if i not in selected_paths]
        np.random.shuffle(remaining_paths)
        selected_paths.extend(remaining_paths[:num_selected - len(selected_paths)])

    # 如果选择的路径过多，只保留前num_selected个
    if len(selected_paths) > num_selected:
        selected_paths = selected_paths[:num_selected]

    # 确定未选择的路径
    unselected_paths = [i for i in range(num_paths) if i not in selected_paths]

    return np.array(selected_paths), np.array(unselected_paths)


def split_paths_stress_test(R, ratio, congested_links=None):
    """
    设计一个压力测试方案，针对特定的拥塞链路，选择尽量少覆盖它们的路径

    参数:
    R: 路由矩阵
    ratio: 选择的路径比例
    congested_links: 拥塞链路索引列表，如果为None则随机选择

    返回:
    选择的路径索引和未选择的路径索引
    """
    num_paths = R.shape[0]
    num_links = R.shape[1]
    num_selected = int(num_paths * ratio)

    # 如果没有提供拥塞链路，则随机选择约20%的链路作为拥塞链路
    if congested_links is None:
        num_congested = max(1, int(num_links * 0.2))
        congested_links = np.random.choice(num_links, num_congested, replace=False)

    # 计算每条路径覆盖拥塞链路的数量
    congestion_coverage = np.zeros(num_paths)
    for i in range(num_paths):
        congestion_coverage[i] = sum(R[i, link] for link in congested_links)

    # 按照覆盖拥塞链路数量的升序排序（优先选择不覆盖拥塞链路的路径）
    sorted_paths = np.argsort(congestion_coverage)

    # 检查是否能够覆盖所有链路
    selected_paths = sorted_paths[:num_selected]
    R_selected = R[selected_paths]
    covered_links = np.any(R_selected > 0, axis=0)

    # 如果有未覆盖的链路，需要调整选择
    if not np.all(covered_links):
        uncovered_links = np.where(~covered_links)[0]

        # 从未选择的路径中找出能覆盖未覆盖链路的路径
        remaining_paths = sorted_paths[num_selected:]
        paths_to_add = []

        for link in uncovered_links:
            # 找到覆盖该链路且覆盖拥塞链路最少的路径
            best_path = -1
            min_congestion = float('inf')

            for path in remaining_paths:
                if path in paths_to_add or path in selected_paths:
                    continue

                if R[path, link] > 0:
                    if congestion_coverage[path] < min_congestion:
                        min_congestion = congestion_coverage[path]
                        best_path = path

            if best_path != -1:
                paths_to_add.append(best_path)

        # 更新选择的路径
        if paths_to_add:
            # 移除等量的已选路径（优先移除覆盖拥塞链路多的路径）
            selected_paths = list(selected_paths)
            to_remove = sorted_paths[:len(paths_to_add)]
            for path in to_remove:
                if path in selected_paths:
                    selected_paths.remove(path)

            # 添加新路径
            selected_paths.extend(paths_to_add)

    # 确保选择了正确数量的路径
    if len(selected_paths) > num_selected:
        selected_paths = selected_paths[:num_selected]
    elif len(selected_paths) < num_selected:
        remaining_paths = [i for i in range(num_paths) if i not in selected_paths]
        np.random.shuffle(remaining_paths)
        selected_paths.extend(remaining_paths[:num_selected - len(selected_paths)])

    # 确定未选择的路径
    unselected_paths = [i for i in range(num_paths) if i not in selected_paths]

    return np.array(selected_paths), np.array(unselected_paths)


def select_paths_mixed_strategy(R, ratio, strategies=['importance', 'coverage', 'clustering', 'random'], weights=None):
    """
    混合策略选择路径，结合多种方法的优势

    参数:
    R: 路由矩阵
    ratio: 选择的路径总比例
    strategies: 使用的策略列表
    weights: 各策略的权重，如果为None则平均分配

    返回:
    选择的路径索引和未选择的路径索引
    """
    num_paths = R.shape[0]
    num_selected = int(num_paths * ratio)

    if weights is None:
        weights = [1 / len(strategies)] * len(strategies)

    # 确保权重和为1
    weights = np.array(weights) / sum(weights)

    # 计算每种策略选择的路径数量
    strategy_counts = np.floor(weights * num_selected).astype(int)
    # 确保总数为num_selected
    remaining = num_selected - sum(strategy_counts)
    for i in range(int(remaining)):
        strategy_counts[i] += 1

    # 应用每种策略选择路径
    selected_paths = []
    available_paths = set(range(num_paths))

    for i, strategy in enumerate(strategies):
        if strategy_counts[i] == 0:
            continue

        # 计算该策略需要选择的路径数
        count = min(strategy_counts[i], len(available_paths))
        if count == 0:
            continue

        # 从可用路径中选择子集
        R_available = np.array([R[j] for j in available_paths])
        path_indices = np.array(list(available_paths))

        # 应用对应的策略
        sub_ratio = count / len(available_paths)

        if strategy == 'importance':
            # 使用原始的重要性计算方法
            from functools import partial
            compute_path_importance_func = globals().get('compute_path_importance',
                                                         lambda R: np.sum(1.0 / (np.sum(R, axis=0) + 1e-10) * R,
                                                                          axis=1))
            sub_paths, _ = split_paths_by_importance(R_available, sub_ratio)
        elif strategy == 'coverage':
            sub_paths, _ = split_paths_by_link_coverage(R_available, sub_ratio)
        elif strategy == 'clustering':
            sub_paths, _ = split_paths_by_graph_partitioning(R_available, sub_ratio)
        elif strategy == 'random':
            sub_paths, _ = split_paths_random(R_available, sub_ratio)
        else:
            # 默认使用随机选择
            sub_paths, _ = split_paths_random(R_available, sub_ratio)

        # 转换回原始路径索引
        selected_sub_paths = path_indices[sub_paths]
        selected_paths.extend(selected_sub_paths)

        # 更新可用路径集合
        available_paths -= set(selected_sub_paths)

    # 如果选择的路径不足，从剩余路径中随机选择
    if len(selected_paths) < num_selected and available_paths:
        remaining_paths = list(available_paths)
        np.random.shuffle(remaining_paths)
        selected_paths.extend(remaining_paths[:num_selected - len(selected_paths)])

    # 确定未选择的路径
    unselected_paths = [i for i in range(num_paths) if i not in selected_paths]

    return np.array(selected_paths), np.array(unselected_paths)


# 原始的基于路径重要性的选择方法（保留以便兼容性）
def compute_path_importance(R):
    """
    计算每条路径的重要性得分
    """
    # 计算每条链路被多少条路径覆盖
    link_coverage = np.sum(R, axis=0)

    # 计算路径的重要性得分：覆盖稀有链路的路径更重要
    path_scores = np.zeros(R.shape[0])
    for i in range(R.shape[0]):
        score = 0
        for j in range(R.shape[1]):
            if R[i, j] == 1:
                score += 1.0 / link_coverage[j]
        path_scores[i] = score

    return path_scores


def split_paths_by_importance(R, ratio):
    """
    根据路径重要性将路径分组

    参数:
    R: 路由矩阵
    ratio: 选择的路径比例

    返回:
    选择的路径索引和未选择的路径索引
    """
    path_scores = compute_path_importance(R)

    # 按重要性得分对路径进行排序
    sorted_paths = np.argsort(-path_scores)  # 降序排列

    # 确定选择路径数量
    num_selected_paths = int(R.shape[0] * ratio)
    selected_paths = sorted_paths[:num_selected_paths]
    unselected_paths = sorted_paths[num_selected_paths:]

    return selected_paths, unselected_paths


def get_path_selection_strategy(strategy_name):
    """
    根据策略名称返回对应的路径选择函数

    参数:
    strategy_name: 策略名称，可选值: 'importance', 'random', 'coverage', 'clustering', 'stress', 'mixed'

    返回:
    路径选择函数
    """
    strategies = {
        'importance': split_paths_by_importance,
        'random': split_paths_random,
        'coverage': split_paths_by_link_coverage,
        'clustering': split_paths_by_graph_partitioning,
        'stress': split_paths_stress_test,
        'mixed': select_paths_mixed_strategy
    }

    return strategies.get(strategy_name, split_paths_by_importance)