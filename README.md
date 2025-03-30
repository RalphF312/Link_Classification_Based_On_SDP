# 基于SDP的网络拥塞链路诊断方案
该项目实现了基于网络层析的链路拥塞检测与用色类别分类算法，并提供了一套完整的实验框架。
## 项目概述
网络层析（Network Tomography）是一种通过端到端测量来推断网络内部状态的技术。本项目专注于链路拥塞检测与分类问题，通过部分观测路径的端到端测量，推断全网链路的拥塞状态和拥塞原因。
主要特点：
+ 实现了基于半正定规划（SDP）的链路时延方差估计
+ 使用K-means聚类进行链路拥塞分类
+ 提供完整的实验比较框架，包括与Gurobi和L1-L2优化Baseline算法的对比
+ 支持批量实验和结果统计分析
## 项目结构
+ `main.py`：主程序入口，包含实验控制、结果评估和可视化功能
+ `simulator.py`：网络模拟器，生成链路和路径时延数据
+ `SDP_complete_link.py`：实现基于半正定规划的链路方差推断算法
+ `alg_gurobi.py`：实现基于Gurobi优化的链路拥塞检测算法（作为Baseline）
+ `alg_l1_l2.py`：实现基于L1-L2优化的链路拥塞检测算法（作为Baseline）
+ `path_selection_strategies.py`：实现不同的路径选择策略
## 安装依赖
本项目需要以下库：
```bash
pip install numpy matplotlib pandas networkx scipy cvxpy gurobipy scikit-learn
```
## 使用方法
```python
python main.py --repetitions 100
```
默认的观测比例为：100%、80%、60%、40%、20%，使用随机路径选择策略。
## 实验结果
所有实验结果默认保存在`./experiment_results/`目录下，每次实验会创建一个带时间戳的子目录，包含：
+ 链路方差推断结果
+ 拥塞检测和分类评估指标
+ 链路分类结果可视化
+ 性能指标统计和图表
+ Baseline算法对比结果
对于批量实验，还会生成汇总统计和可视化：
+ 各观测比例的检测率(DR)、误报率(FPR)和分类准确率对比
+ 不同链路类型识别准确率对比
+ 性能指标随观测比例变化的趋势图