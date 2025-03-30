import numpy as np
from scipy import optimize


class LinkQualityClassifier:
    """
    基于压缩感知的链路质量分类器实现
    使用 ℓ1-ℓ2 优化方法，为网络时延方差估计定制
    """

    def __init__(self, routing_matrix, mixing_parameter=0.05):
        """
        初始化链路质量分类器

        参数:
        -----------
        routing_matrix : numpy.ndarray
            二进制路由矩阵 A，其中 a_ij = 1 表示链路 j 在路径 i 上，否则为 0
        mixing_parameter : float
            混合参数 λ，控制稀疏性和精度之间的权衡
        """
        self.A = routing_matrix
        self.lambda_param = mixing_parameter

    def estimate_link_loss_rates(self, path_measurements):
        """
        使用 ℓ1-ℓ2 优化估计链路方差

        参数:
        -----------
        path_measurements : numpy.ndarray
            端到端路径方差测量向量

        返回:
        --------
        link_variances : numpy.ndarray
            估计的链路方差
        """
        # 直接求解 ℓ1-ℓ2 优化问题
        # 不需要像丢包率那样进行对数变换，因为方差是可加的
        link_variances = self._solve_l1_l2_optimization(path_measurements)

        # 确保所有方差都是非负的
        link_variances = np.maximum(link_variances, 0)

        return link_variances

    def classify_links(self, link_variances, threshold=0.01):
        """
        将链路分类为正常或拥塞状态

        参数:
        -----------
        link_variances : numpy.ndarray
            估计的链路方差
        threshold : float
            阈值，低于该值的链路方差被视为正常

        返回:
        --------
        classification : dict
            包含 'higher_quality' 和 'lower_quality' 键的字典，对应链路索引列表
        binary_classification : numpy.ndarray
            二进制数组，1 表示拥塞链路，0 表示正常链路
        """
        # 应用阈值识别正常链路
        binary_classification = np.zeros_like(link_variances, dtype=int)

        # 找出方差高于阈值的链路，标记为拥塞
        congested_indices = np.where(link_variances > threshold)[0]
        binary_classification[congested_indices] = 1

        higher_quality = np.where(binary_classification == 0)[0]
        lower_quality = np.where(binary_classification == 1)[0]

        return {
            'higher_quality': higher_quality,
            'lower_quality': lower_quality
        }, binary_classification

    def _solve_l1_l2_optimization(self, y):
        """
        使用 ISTA 算法求解 ℓ1-ℓ2 优化问题

        参数:
        -----------
        y : numpy.ndarray
            路径方差向量

        返回:
        --------
        x : numpy.ndarray
            优化问题的解（链路方差估计）
        """

        # 定义目标函数: (1/2)||y - Ax||_2^2 + λ||x||_1
        def objective(x):
            l2_term = 0.5 * np.sum((y - self.A @ x) ** 2)
            l1_term = self.lambda_param * np.sum(np.abs(x))
            return l2_term + l1_term

        # 迭代收缩阈值算法 (ISTA)
        n_links = self.A.shape[1]
        x = np.zeros(n_links)

        # 计算梯度步的 Lipschitz 常数
        L = np.linalg.norm(self.A.T @ self.A, 2)
        if L == 0:  # 处理矩阵全为零的情况
            L = 1

        max_iter = 1000
        tol = 1e-6

        for _ in range(max_iter):
            # 计算数据保真项的梯度
            gradient = self.A.T @ (self.A @ x - y)

            # 梯度下降步骤
            z = x - (1 / L) * gradient

            # 软阈值算子（L1 范数的近端映射）
            x_new = np.sign(z) * np.maximum(np.abs(z) - self.lambda_param / L, 0)

            # 检查收敛性
            if np.linalg.norm(x_new - x) < tol:
                break

            x = x_new

        return x

    def set_mixing_parameter(self, lambda_value):
        """
        更新混合参数 λ

        参数:
        -----------
        lambda_value : float
            混合参数的新值
        """
        self.lambda_param = lambda_value


def detect_congested_links(routing_matrix, path_variances, mixing_parameter=0.05, threshold=0.01):
    """
    使用 L1-L2 优化从路径方差数据检测拥塞链路

    参数:
    -----------
    routing_matrix : numpy.ndarray
        二进制路由矩阵，其中条目 (i,j) 为 1 表示链路 j 在路径 i 上
    path_variances : numpy.ndarray
        路径方差测量向量
    mixing_parameter : float
        L1-L2 优化的混合参数（默认: 0.05）
    threshold : float
        将链路分类为拥塞的阈值（默认: 0.01）

    返回:
    --------
    binary_classification : numpy.ndarray
        二进制数组，1 表示拥塞链路，0 表示正常链路
    """
    # 创建分类器
    classifier = LinkQualityClassifier(routing_matrix, mixing_parameter)

    # 估计链路方差
    link_variances = classifier.estimate_link_loss_rates(path_variances)

    # 分类链路
    _, binary_classification = classifier.classify_links(link_variances, threshold)

    return binary_classification