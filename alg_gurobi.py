#! python3
# -*- coding: utf-8 -*-
# @Author  : 杜承泽
# @Email   : Monickar@foxmail.com

import gurobipy as gp
from gurobipy import GRB
import numpy as np


def tomo_gurobi(A, Y):
    '''
    Tomo_Gurobi 算法的调用接口
    :param Y: 观测的路径拥塞状态，列向量；如为矩阵，横纬度为时间
    :param A: routing matrix,矩阵，纵维度为路径，横维度为链路
    :return: x_identified 为返回的已识别的链路拥塞状态；列向量，如为矩阵，横维度为时间，纵维度为链路推测状态
    '''
    m, n = A.shape
    t = Y.shape[1]

    # 创建模型
    model = gp.Model()
    model.setParam('OutputFlag', 0)

    # 创建决策变量X，每个元素都是二进制的
    X = model.addMVar((n, t), vtype=GRB.BINARY, name="X")

    # 将非故障路径上的链路定义为好
    for i in range(t):
        zero_index = np.where(Y[:, i] == 0)[0]
        for j in zero_index:
            one_index = np.where(A[j, :] == 1)[0]
            for k in one_index:
                # 加约束： 将X中第k行第i列的元素设为0
                model.addConstr(X[k, i] == 0)

    # 创建约束条件 A * X >= Y
    model.addConstr(A @ X >= Y)

    # 定义目标函数，最小化X中每一列中1的个数（即最小化故障链路的个数）
    objective = gp.quicksum(X[k, j] for k in range(n) for j in range(t))
    model.setObjective(objective, GRB.MINIMIZE)

    # 求解优化问题
    model.optimize()

    # 打印结果
    if model.status == GRB.OPTIMAL:
        X_ = np.array([[abs(X[k, j].x) for k in range(n)] for j in range(t)])
    else:
        print("No optimal solution found.")

    model.dispose()
    return X_.T

def test_tomo():
    y = np.array([[0, 0, 0],
                  [0, 0, 1],
                  [1, 0, 0],
                  [0, 1, 1],
                  [1, 0, 0],
                  [0, 0, 1],
                  [1, 1, 0],
                  [1, 1, 1]], dtype=np.int8).T

    A_rm = np.array([
        [1, 0, 1, 1, 0],
        [1, 0, 1, 0, 1],
        [0, 1, 1, 0, 1]], dtype=np.int8)

    links_state_inferred = tomo_gurobi(A_rm, y)
    print(links_state_inferred)


if __name__ == '__main__':
    test_tomo()