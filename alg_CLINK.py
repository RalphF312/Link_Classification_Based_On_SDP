'''
[INFOCOM'07] The Boolean Solution to the Congested IP Link Location Problem Theory and Practice
'''
import copy
import numpy as np
import pandas as pd
import os
from pathlib import Path
import time
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def alg_clink(y: np.ndarray, A_rm: np.ndarray, x_pc: np.ndarray):
    """
    clink算法的调用接口
    :param y: 观测的路径拥塞状态，列向量；如为矩阵，横纬度为时间
    :param A_rm: routing matrix,矩阵，纵维度为路径，横维度为链路
    :param x_pc: ’probability of congestion' 列向量
    :return: x_identified 为返回的已识别的链路拥塞状态；列向量，如为矩阵，横维度为时间，纵维度为链路推测状态
    """

    if np.ndim(y) <= 1:  # 强制转换为列向量
        y = y.reshape((-1, 1))

    m, n = A_rm.shape
    _, num_time = y.shape

    x_identified = np.zeros((n, num_time))

    for i in range(num_time):
        paths_state_obv = y[:, i]
        links_state_infered = clink_a_groub(paths_state_obv, A_rm, x_pc)
        x_identified[:, i] = links_state_infered

    return np.int8(x_identified)


def clink_a_groub(y, A_rm, x_pc):
    """
    clink 算法测试一组数据
    :param y: np.ndarray  观测路径的拥塞状态，普通向量
    :param A_rm: np.ndarray  路由矩阵
    :param x_pc: 链路先验拥塞概率，np.ndarray  普通向量
    :return:links_state_inferred: np.ndarray 普通
    """
    tree_vector = rm_to_vector(A_rm)
    num_links = len(tree_vector)
    links_congest_pro = list(x_pc.flatten())

    route_matrix = A_rm

    paths_state_obv = copy.deepcopy(y)  # 观测路径状态数组

    links_state_inferred = diagnose(paths_state_obv, route_matrix, num_links, links_congest_pro)
    return links_state_inferred


def diagnose(paths_state_obv, route_matrix: np.ndarray, num_links, links_cong_pro: list):
    """
    clink算法核心部分
    :param paths_state_obv: array(m,)  路径的观测状态
    :param route_matrix:  array(m,n)   路由矩阵
    :param num_links: int  链路数量
    :param links_cong_pro: list 链路的拥塞概率
    :return:
    """
    paths_cong_obv, paths_no_cong_obv = cal_cong_path_info(paths_state_obv)
    # print('链路的拥塞概率:', links_cong_pro)
    congested_path = (paths_cong_obv - 1).tolist()
    un_congested_path = (paths_no_cong_obv - 1).tolist()
    # print("congested_path",congested_path)
    # print("un_congested_path",un_congested_path)

    # 生成正常链路和不确定链路
    good_link, uncertain_link = get_link_state_class(un_congested_path, route_matrix, num_links)
    # print('位于不拥塞路径中的链路:', good_link)
    # print('不确定拥塞状态的链路:', uncertain_link)

    # 获取经过一条链路的所有路径domain
    domain_dict = {}
    for i in uncertain_link:
        domain_dict[i] = [j for j in get_paths(i + 1, route_matrix) if j in congested_path]
    # print("domain_dict")
    # print(domain_dict)

    links_state_inferred = np.zeros(num_links)
    links_cong_inferred = []
    # 计算所有的链路
    temp_state = [1e8 for _ in range(len(uncertain_link))]
    # print('temp_state:', temp_state)

    if not temp_state and len(congested_path):  # 如果存在无解的情形
        links_state_inferred = links_state_inferred + np.nan

    while temp_state and len(congested_path) > 0:
        # 找到最小的值对应的链路
        for index, i in enumerate(uncertain_link):
            # print(self._congestion_prob_links)
            # 方法1 公式(log((1-p)/p))|domain(x)|
            a = np.log((1 - links_cong_pro[i]) / (links_cong_pro[i]))
            b = len(domain_dict[i])
            if b == 0:
                temp_state[index] = 1e8
            else:
                temp_state[index] = a / b

            # 方法2 公式log((1-p)/p/|domain(x)|)
            # b=len(domain_dict[i])
            # if b==0:
            #     temp_state[index]=1e8
            # else:
            #     a=np.log((1 - self._congestion_prob_links[i]) / (self._congestion_prob_links[i])/b)
            #     temp_state[index]=a

        # print(temp_state)
        index = temp_state.index(min(temp_state))
        links_state_inferred[uncertain_link[index]] = 1
        links_cong_inferred.append(uncertain_link[index] + 1)
        # print("推断的链路",uncertain_link[index]+1)
        for item in domain_dict[uncertain_link[index]]:
            if item in congested_path:
                # print('congested_path', congested_path)
                # print('item:', item)
                congested_path.remove(item)
        domain_dict.pop(uncertain_link[index])
        uncertain_link.remove(uncertain_link[index])
        temp_state.remove(temp_state[index])

        for k, v in domain_dict.items():
            temp = []
            for i in v:
                if i in congested_path:
                    temp.append(i)
            domain_dict[k] = copy.deepcopy(temp)

        # print("domain_dict")
        # print(domain_dict)
        # print("uncertain_link",uncertain_link)
        # print("congest_path",congested_path)
    return links_state_inferred
    # print("真实的链路拥塞",self._links_congested)
    # print('推测的链路拥塞:', self.link_state_inferred)


def get_paths(link: int, route_matrix):
    """
    获取经过指定链路的所有路径。

    在路由矩阵中，第 0 列代表链路 1，第 1 列代表链路 2。依次类推。
    第 0 行代表路径 1，第 1 行代表路径 2。依次类推。
    :param link: 链路的编号
    :return:
    """
    assert link > 0
    paths, = np.where(route_matrix[:, link - 1] > 0)
    return paths.tolist()


def get_link_state_class(un_congested_path: list, route_matrix, num_links):
    """
    根据非拥塞路径，返回正常链路列表，和拥塞链路列表
    :param un_congested_path:list
    :return:good_link:list ,uncertain_link:list   存储链路下标
    """
    # 所有经过了不拥塞路径的链路
    good_link = []

    for i in un_congested_path:
        for index, item in enumerate(route_matrix[i]):
            if int(item) == 1 and index not in good_link:
                good_link.append(index)

    all_links = [i for i in range(num_links)]
    # 排除那些肯定不拥塞的链路
    uncertain_link = []
    for item in all_links:
        if item not in good_link:
            uncertain_link.append(item)
    return good_link, uncertain_link


def cal_cong_path_info(paths_state_obv):
    """
    根据路径的观测信息，计算拥塞路径和非拥塞路径
    :param paths_state_obv:
    :return:
    """
    paths_cong = []
    paths_no_cong = []
    for index in range(len(paths_state_obv)):
        if int(paths_state_obv[index]) == 1:
            # if int(self.path_states[index]) == 1:
            paths_cong.append(index + 1)
        else:
            paths_no_cong.append(index + 1)
    return np.array(paths_cong), np.array(paths_no_cong)


def rm_to_vector(A_rm: np.ndarray):
    """
    将路由矩阵转换为树向量
    :param A_rm:
    :return:
    """

    tree_vector = [0] * (A_rm.shape[1])

    for i in range(A_rm.shape[0]):
        path = A_rm[i]
        pre_node = 0
        for j in range(path.shape[0]):
            if path[j] == 1:
                tree_vector[j] = pre_node
                pre_node = j + 1

    return tree_vector

class NetworkDiagnosisOrchestrator:
    """
    网络诊断调度器，用于协调已生成的链路数据和CLINK算法的执行。
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
            "time_taken": []
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
            "path_delay_variances.csv",
            "path_types.csv",
            "link_prior_congestion_probability.csv"
        ]

        for file in required_files:
            if not (self.sim_results_dir / file).exists():
                print(f"错误：找不到所需的模拟数据文件 {file}")
                print(f"请确保 link_performance_generator.py 已经运行并在 {self.sim_results_dir} 中生成了所需文件。")
                return False

        return True

    def load_simulation_data(self):
        """
        加载模拟数据用于CLINK算法分析。

        返回:
            simulation_data: 包含所有所需数据的字典
        """
        # 检查数据文件存在
        if not self.check_simulation_data_exists():
            raise FileNotFoundError(f"模拟数据文件缺失。请先运行链路性能生成器。")

        # 加载链路类型数据（真实值）
        link_types_df = pd.read_csv(self.sim_results_dir / "link_types.csv")

        # 加载路径延迟方差数据
        path_delay_variances_df = pd.read_csv(self.sim_results_dir / "path_delay_variances.csv")

        # 加载路径类型数据
        path_types_df = pd.read_csv(self.sim_results_dir / "path_types.csv")

        # 加载链路先验拥塞概率
        link_prior_prob_df = pd.read_csv(self.sim_results_dir / "link_prior_congestion_probability.csv")

        # 整理数据格式
        num_links = len(link_types_df.columns)
        num_paths = len(path_types_df.columns)
        num_repeats = len(link_types_df)

        # 构建先验拥塞概率向量
        x_pc = link_prior_prob_df['prior_congestion_probability'].values

        print(f"成功加载模拟数据：")
        print(f"  - 链路数量: {num_links}")
        print(f"  - 路径数量: {num_paths}")
        print(f"  - 实验次数: {num_repeats}")

        return {
            "link_types_df": link_types_df,
            "path_types_df": path_types_df,
            "path_delay_variances_df": path_delay_variances_df,
            "x_pc": x_pc,
            "num_links": num_links,
            "num_paths": num_paths,
            "num_repeats": num_repeats
        }

    def run_clink_diagnosis(self, threshold=None):
        """
        使用CLINK算法进行网络诊断。

        参数:
            threshold: 路径延迟方差的阈值，超过此阈值认为路径拥塞。如果为None，则使用路径类型作为输入。

        返回:
            avg_results: 包含平均性能指标的字典
        """
        print(f"开始运行CLINK诊断算法...")

        try:
            # 加载模拟数据
            sim_data = self.load_simulation_data()
        except FileNotFoundError as e:
            print(f"错误：{e}")
            return None

        link_types_df = sim_data["link_types_df"]
        path_types_df = sim_data["path_types_df"]
        path_delay_variances_df = sim_data["path_delay_variances_df"]
        x_pc = sim_data["x_pc"]
        num_repeats = sim_data["num_repeats"]

        # 创建结果目录
        results_dir = self.output_dir / "clink_results"
        results_dir.mkdir(exist_ok=True)

        # 预准备数据结构存储结果
        all_y_true = []
        all_y_pred = []

        # 对每次实验进行CLINK诊断
        for repeat_idx in range(num_repeats):
            # 获取这次实验的真实链路状态（将类型1和类型2视为拥塞链路）
            true_link_states = np.array([
                1 if link_types_df.iloc[repeat_idx, i] in [1, 2] else 0
                for i in range(len(link_types_df.columns))
            ])

            # 获取观察到的路径状态
            if threshold is None:
                # 如果没有提供阈值，直接使用路径类型作为输入
                observed_path_states = np.array([
                    path_types_df.iloc[repeat_idx, i]
                    for i in range(len(path_types_df.columns))
                ])
                diagnosis_method = "路径类型"
            else:
                # 如果提供了阈值，使用路径延迟方差与阈值比较来确定路径拥塞状态
                observed_path_states = np.array([
                    1 if path_delay_variances_df.iloc[repeat_idx, i] > threshold else 0
                    for i in range(len(path_delay_variances_df.columns))
                ])
                diagnosis_method = f"延迟方差阈值 {threshold}"

            # 运行CLINK算法
            start_time = time.time()
            inferred_link_states = alg_clink(observed_path_states, self.route_matrix, x_pc)
            end_time = time.time()

            # 提取第一列作为结果（因为alg_clink返回的是二维数组）
            inferred_link_states = inferred_link_states[:, 0]

            # 存储结果
            all_y_true.append(true_link_states)
            all_y_pred.append(inferred_link_states)

            # 计算性能指标
            precision = precision_score(true_link_states, inferred_link_states, zero_division=1)
            recall = recall_score(true_link_states, inferred_link_states, zero_division=1)
            f1 = f1_score(true_link_states, inferred_link_states, zero_division=1)
            accuracy = np.mean(true_link_states == inferred_link_states)
            time_taken = end_time - start_time

            # 存储性能指标
            self.results["precision"].append(precision)
            self.results["recall"].append(recall)
            self.results["f1_score"].append(f1)
            self.results["accuracy"].append(accuracy)
            self.results["time_taken"].append(time_taken)

            # 每10次实验打印一次进度
            if (repeat_idx + 1) % 10 == 0:
                print(f"完成 {repeat_idx + 1}/{num_repeats} 次诊断")

        # 将所有预测结果保存到文件
        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)

        # 保存预测结果
        np.save(results_dir / "true_link_states.npy", all_y_true)
        np.save(results_dir / "inferred_link_states.npy", all_y_pred)

        # 保存性能指标
        results_df = pd.DataFrame(self.results)

        # 创建结果文件名，根据使用的方法区分
        if threshold is None:
            results_filename = "clink_performance_path_types.csv"
            summary_filename = "summary_results_path_types.txt"
        else:
            results_filename = f"clink_performance_threshold_{threshold}.csv"
            summary_filename = f"summary_results_threshold_{threshold}.txt"

        results_df.to_csv(results_dir / results_filename, index=False)

        # 计算并保存平均性能指标
        avg_results = {
            "avg_precision": np.mean(self.results["precision"]),
            "avg_recall": np.mean(self.results["recall"]),
            "avg_f1_score": np.mean(self.results["f1_score"]),
            "avg_accuracy": np.mean(self.results["accuracy"]),
            "avg_time_taken": np.mean(self.results["time_taken"])
        }

        # 计算混淆矩阵
        all_true_flat = all_y_true.flatten()
        all_pred_flat = all_y_pred.flatten()
        cm = confusion_matrix(all_true_flat, all_pred_flat)

        # 保存摘要结果
        with open(results_dir / summary_filename, "w") as f:
            f.write("# CLINK算法诊断结果摘要\n\n")
            f.write(f"总实验次数: {num_repeats}\n")
            f.write(f"链路数量: {sim_data['num_links']}\n")
            f.write(f"路径数量: {sim_data['num_paths']}\n")
            f.write(f"诊断方法: {diagnosis_method}\n\n")

            f.write("## 平均性能指标\n\n")
            for metric, value in avg_results.items():
                f.write(f"{metric}: {value:.4f}\n")

            f.write("\n## 混淆矩阵\n\n")
            f.write("```\n")
            f.write(f"TN: {cm[0, 0]}, FP: {cm[0, 1]}\n")
            f.write(f"FN: {cm[1, 0]}, TP: {cm[1, 1]}\n")
            f.write("```\n")

        print(f"CLINK诊断完成，结果保存在 {results_dir}")

        return avg_results

    def analyze_threshold_impact(self, thresholds):
        """
        分析不同阈值对CLINK算法性能的影响。

        参数:
            thresholds: 要测试的阈值列表
        """
        print("分析不同路径延迟方差阈值对诊断性能的影响...")

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
                "time_taken": []
            }

            # 运行CLINK诊断
            avg_results = self.run_clink_diagnosis(threshold=threshold)

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
        threshold_df.to_csv(self.output_dir / "threshold_analysis.csv", index=False)

        print(f"阈值分析完成，结果保存在 {self.output_dir}/threshold_analysis.csv")

        return threshold_df

def main():
    # 路由矩阵文件路径
    route_matrix_path = "./route_matrix.txt"

    # 初始化调度器
    orchestrator = NetworkDiagnosisOrchestrator(route_matrix_path)

    # 确认模拟数据存在
    if not orchestrator.check_simulation_data_exists():
        print("\n请先运行 link_performance_generator.py 生成链路性能数据。")
        print("示例命令: python link_performance_generator.py")
        return

    # 使用CLINK算法进行诊断（使用路径类型作为输入）
    print("\n使用路径类型作为输入进行诊断...")
    results_with_path_types = orchestrator.run_clink_diagnosis(threshold=None)

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
            print(f"使用路径类型作为输入:")
            for metric, value in results_with_path_types.items():
                print(f"  {metric}: {value:.4f}")

            print("\n最佳阈值结果:")
            best_idx = threshold_results["f1_score"].argmax()
            best_threshold = threshold_results.iloc[best_idx]
            print(f"  最佳阈值: {best_threshold['threshold']}")
            print(f"  Precision: {best_threshold['precision']:.4f}")
            print(f"  Recall: {best_threshold['recall']:.4f}")
            print(f"  F1 Score: {best_threshold['f1_score']:.4f}")
            print(f"  Accuracy: {best_threshold['accuracy']:.4f}")

    print("\n所有结果已保存到 ./diagnosis_results 目录")


if __name__ == "__main__":
    main()