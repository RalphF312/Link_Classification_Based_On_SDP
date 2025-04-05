import numpy as np
import pandas as pd
import os
import random
from pathlib import Path


class RouteDataGenerator:
    """
    网络链路数据生成器，用于模拟不同类型的路径及其延迟特性。
    """

    def __init__(self, route_matrix_file, congestion_ratio=0.15, attack_ratio=0.10, attack_probability=1.0):
        """
        参数:
            route_matrix_file: 路由矩阵文件的路径
            congestion_ratio: 正常拥塞链路的比例 (f1)
            attack_ratio: 可能被攻击的链路比例 (f2)
            attack_probability: 当链路被选为待攻击链路时，实际攻击的概率
        """
        self.route_matrix = self.load_route_matrix(route_matrix_file)
        self.num_paths = self.route_matrix.shape[0]
        self.num_links = self.route_matrix.shape[1]

        # 拥塞和攻击参数
        self.congestion_ratio = congestion_ratio  # f1
        self.attack_ratio = attack_ratio  # f2
        self.attack_probability = attack_probability

        # 创建输出文件夹
        self.output_dir = Path("./simulation_results")
        self.output_dir.mkdir(exist_ok=True)

        # 存储所有实验的结果
        self.all_link_delay_variances = []
        self.all_link_types = []
        self.all_path_delay_variances = []
        self.all_path_types = []

    def load_route_matrix(self, matrix_file):
        return np.loadtxt(matrix_file, dtype=int)

    def generate_delay_distributions(self):
        """
        为每种链路类型生成延迟分布参数。

        返回:
            dict: 每种链路类型的延迟分布参数
        """
        # 类型 0: 正常链路 (泊松分布，参数为30)
        # 类型 1: 外部流量致拥链路 (泊松分布，参数为100)
        # 类型 2: 带宽被攻击减小链路 (泊松分布，参数为200)

        distributions = {
            0: {"type": "poisson", "lam": 30},
            1: {"type": "poisson", "lam": 100},
            2: {"type": "poisson", "lam": 200}
        }

        return distributions

    def generate_link_delays(self, link_type, distribution_params, num_samples=1000):
        """
        根据链路类型生成延迟样本。

        参数:
            link_type: 链路类型 (0, 1, 或 2)
            distribution_params: 分布参数
            num_samples: 要生成的延迟样本数量

        返回:
            numpy.ndarray: 延迟样本数组
        """
        if link_type == 0:  # Normal Link
            params = distribution_params[0]
            delays = np.random.poisson(lam=params["lam"], size=num_samples)
        elif link_type == 1:  # External traffic link
            params = distribution_params[1]
            delays = np.random.poisson(lam=params["lam"], size=num_samples)
        elif link_type == 2:  # Attack-reduced link
            params = distribution_params[2]
            delays = np.random.poisson(lam=params["lam"], size=num_samples)
        else:
            raise ValueError(f"Invalid link type: {link_type}")

        return delays

    def calculate_link_delays_variance(self, link_delays):
        """
        计算每个链路的延迟方差。

        参数:
            link_delays: 将链路索引映射到延迟数组的字典

        返回:
            dict: 将链路索引映射到延迟方差的字典
        """
        link_variances = {}
        for link_idx, delays in link_delays.items():
            link_variances[link_idx] = np.var(delays)
        return link_variances

    def calculate_path_delays(self, link_delays):
        """
        根据链路延迟和路径矩阵计算路径延迟。

        参数:
            link_delays: 将链路索引映射到延迟数组的字典

        返回:
            dict: 将路径索引映射到延迟数组的字典
        """
        path_delays = {}

        for path_idx in range(self.num_paths):
            # 获取此路径中的链路
            links_in_path = np.where(self.route_matrix[path_idx] == 1)[0]

            # 计算每个时间样本的延迟
            path_delay = np.zeros(1000)
            for link_idx in links_in_path:
                path_delay += link_delays[link_idx]

            path_delays[path_idx] = path_delay

        return path_delays

    def calculate_metrics(self, path_delays, path_types):
        """
        计算每条路径的性能指标。

        参数:
            path_delays: 将路径索引映射到延迟数组的字典
            path_types: 将路径索引映射到路径类型的字典

        返回:
            pandas.DataFrame: 路径指标的数据框
        """
        metrics = []

        for path_idx, delays in path_delays.items():
            path_type = path_types[path_idx]

            # 计算指标
            delay_variance = np.var(delays)

            metrics.append({
                "path_id": path_idx,
                "path_type": path_type,
                "delay_variance": delay_variance
            })

        return pd.DataFrame(metrics)

    def run_simulation(self, repeat_id):
        """
        运行单次模拟迭代。

        参数:
            repeat_id: 重复运行的ID

        返回:
            dict: 模拟结果
        """
        # 生成分布参数
        distribution_params = self.generate_delay_distributions()

        # 分配链路类型 - 默认情况下，所有链路都是类型0（正常）
        link_types = {link_idx: 0 for link_idx in range(self.num_links)}

        # 随机选择一些链路为类型1（普通拥塞）
        num_type1 = max(1, int(self.num_links * self.congestion_ratio))
        type1_links = random.sample(range(self.num_links), num_type1)
        for link_idx in type1_links:
            link_types[link_idx] = 1

        # 随机选择一些链路为潜在攻击目标
        num_type2_candidates = max(1, int(self.num_links * self.attack_ratio))
        type2_candidates = random.sample(range(self.num_links), num_type2_candidates)

        # 根据攻击概率决定哪些候选链路会被实际攻击
        for link_idx in type2_candidates:
            # 根据攻击概率决定是否攻击此链路
            if random.random() < self.attack_probability:
                link_types[link_idx] = 2  # 如果链路已经是类型1，会被覆盖为类型2

        # 生成链路时延
        link_delays = {}
        for link_idx, link_type in link_types.items():
            link_delays[link_idx] = self.generate_link_delays(link_type, distribution_params)

        # 计算链路延迟的方差
        link_variances = self.calculate_link_delays_variance(link_delays)

        # 计算路径延迟
        path_delays = self.calculate_path_delays(link_delays)

        # 计算路径类型
        path_types = {}
        for path_idx in range(self.num_paths):
            # 如果路径包含任何1型2型链路，路径类型为1，拥塞
            # 否则，它是类型0路径，即正常路径
            path_links = np.where(self.route_matrix[path_idx] == 1)[0]

            if any(link_types[link] in [1, 2] for link in path_links):
                path_types[path_idx] = 1
            else:
                path_types[path_idx] = 0

        # 计算路径延迟的方差
        path_variances = {}
        for path_idx, delays in path_delays.items():
            path_variances[path_idx] = np.var(delays)

        # 存储该次实验的结果
        self.all_link_delay_variances.append(link_variances)
        self.all_link_types.append(link_types)
        self.all_path_delay_variances.append(path_variances)
        self.all_path_types.append(path_types)

        return {
            "link_types": link_types,
            "link_delay_variances": link_variances,
            "path_types": path_types,
            "path_delay_variances": path_variances
        }

    def calculate_prior_congestion_probability(self):
        """
        计算每个链路的先验拥塞概率，基于前20次模拟结果。

        返回:
            dict: 将链路索引映射到先验拥塞概率的字典
        """
        prior_prob = {}

        # 确保我们至少有20次模拟结果，否则使用所有可用结果
        num_samples = min(20, len(self.all_link_types))

        for link_idx in range(self.num_links):
            # 计算链路为拥塞（类型1或2）的次数
            congestion_count = sum(1 for i in range(num_samples)
                                   if self.all_link_types[i][link_idx] in [1, 2])
            # 计算拥塞概率
            prior_prob[link_idx] = congestion_count / num_samples

        return prior_prob

    def export_results(self, num_repeats):
        """
        将所有模拟结果导出到CSV文件。

        参数:
            num_repeats: 完成的模拟重复次数
        """
        # 1. 导出链路时延方差数据
        link_variance_data = {}
        for link_idx in range(self.num_links):
            link_variance_data[f"link_{link_idx}"] = [self.all_link_delay_variances[i][link_idx]
                                                      for i in range(num_repeats)]

        link_variance_df = pd.DataFrame(link_variance_data)
        link_variance_df.to_csv(self.output_dir / "link_delay_variances.csv", index=False)

        # 2. 导出链路类型数据
        link_type_data = {}
        for link_idx in range(self.num_links):
            link_type_data[f"link_{link_idx}"] = [self.all_link_types[i][link_idx]
                                                  for i in range(num_repeats)]

        link_type_df = pd.DataFrame(link_type_data)
        link_type_df.to_csv(self.output_dir / "link_types.csv", index=False)

        # 3. 导出路径时延方差数据
        path_variance_data = {}
        for path_idx in range(self.num_paths):
            path_variance_data[f"path_{path_idx}"] = [self.all_path_delay_variances[i][path_idx]
                                                      for i in range(num_repeats)]

        path_variance_df = pd.DataFrame(path_variance_data)
        path_variance_df.to_csv(self.output_dir / "path_delay_variances.csv", index=False)

        # 4. 导出路径类型数据
        path_type_data = {}
        for path_idx in range(self.num_paths):
            path_type_data[f"path_{path_idx}"] = [self.all_path_types[i][path_idx]
                                                  for i in range(num_repeats)]

        path_type_df = pd.DataFrame(path_type_data)
        path_type_df.to_csv(self.output_dir / "path_types.csv", index=False)

        # 5. 计算并导出链路先验拥塞概率
        prior_prob = self.calculate_prior_congestion_probability()
        prior_prob_df = pd.DataFrame({
            "link_id": list(prior_prob.keys()),
            "prior_congestion_probability": list(prior_prob.values())
        })
        prior_prob_df.to_csv(self.output_dir / "link_prior_congestion_probability.csv", index=False)

        # 6. 导出实验配置到txt文件
        config_file_path = self.output_dir / "experiment_config.txt"
        with open(config_file_path, 'w') as f:
            f.write(f"# 实验配置参数\n")
            f.write(f"congestion_ratio = {self.congestion_ratio}  # 正常拥塞链路占比 (f1)\n")
            f.write(f"attack_ratio = {self.attack_ratio}  # 待攻击链路占比 (f2)\n")
            f.write(f"attack_probability = {self.attack_probability}  # 攻击概率\n")
            f.write(f"num_links = {self.num_links}  # 链路总数\n")
            f.write(f"num_paths = {self.num_paths}  # 路径总数\n")
            f.write(f"num_repeats = {num_repeats}  # 实验重复次数\n")

    def run_multiple_simulations(self, num_repeats=100):
        """
        运行多次模拟重复。

        参数:
            num_repeats: 模拟重复的次数
        """
        for repeat_id in range(1, num_repeats + 1):
            print(f"Running simulation repeat {repeat_id}/{num_repeats}")
            self.run_simulation(repeat_id)

        # 所有模拟完成后导出结果
        self.export_results(num_repeats)


def main():
    route_matrix_path = "./route_matrix.txt"

    # 实验参数
    congestion_ratio = 0.15  # f1: 正常拥塞链路占比
    attack_ratio = 0.10  # f2: 待攻击链路占比
    attack_probability = 1.0  # 攻击概率 (100%)

    # 生成链路数据
    generator = RouteDataGenerator(
        route_matrix_path,
        congestion_ratio=congestion_ratio,
        attack_ratio=attack_ratio,
        attack_probability=attack_probability
    )
    generator.run_multiple_simulations(num_repeats=100)

    print("Simulation completed successfully!")


if __name__ == "__main__":
    main()