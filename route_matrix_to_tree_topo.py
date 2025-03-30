import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pydot
from networkx.drawing.nx_pydot import graphviz_layout
from copy import copy
import pandas as pd
import os
import igraph as ig

# 读取GML文件
input_gml_file = 'Chinanet.gml'
G = nx.read_gml(input_gml_file)
pos = {}
missing_nodes = [node for node in G.nodes() if node not in pos]
print("Edges in G:\n")

# 根据实际情况补充缺失的节点的位置信息
# 示例：假设给缺失的节点赋予默认值或特定值
default_pos = (140.0, 30.0)  # 或者任何合适的默认经纬度
for node in missing_nodes:
    pos[node] = default_pos

for node, attributes in G._node.items():
    if 'Longitude' in attributes and 'Latitude' in attributes:
        pos[node] = (attributes['Longitude'], attributes['Latitude'])


# 绘制图形
plt.figure(figsize=(12, 8))  # 调整图像大小以便更好地查看
nx.draw(G, pos, with_labels=True, node_size=50, node_color='skyblue', font_size=8)
plt.title('Graph Visualization of Chinanet')
plt.show()
# print(G)

def convert_to_directed_tree(G, root):
    """
    """
    visited = set([root])
    queue = [root]
    parent = {root:None}
    temp_G = nx.DiGraph()

    while queue:
        current = queue.pop(0)
        for neighbor in G.neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                parent[neighbor] = current

    for node, par in parent.items():
        if par is not None:
            temp_G.add_edge(par, node)
    return temp_G

T = convert_to_directed_tree(G, 'Beijing')
# 初始化入度和出度字典
in_degree = {node: 0 for node in T.nodes()}
out_degree = {node: 0 for node in T.nodes()}

for u, v in T.edges():
    out_degree[u] += 1
    in_degree[v] += 1

# for node in T.nodes():
#     print(f"Node: {node} In-degree = {in_degree[node]}, Out-degree = {out_degree[node]}")

def remove_degree_one_nodes(T):
    # 创建一个要删除的节点列表
    nodes_to_remove = []

    # 遍历所有节点以找出入度和出度都为1的节点
    for node in T.nodes():
        if T.in_degree(node) == 1 and T.out_degree(node) == 1:
            nodes_to_remove.append(node)

    # print(f"{nodes_to_remove} nodes to remove.")
    # 删除这些节点
    T.remove_nodes_from(nodes_to_remove)

    return T

def add_new_root(T, new_root_name, old_root):
    """
    在树 T 中添加一个新的根节点，并将旧的根节点作为新根节点的子节点。

    参数:
    - T: 树状拓扑（有向图）
    - new_root_name: 新根节点的名字
    - old_root: 旧的根节点名字

    返回:
    - 更新后的树 T
    """
    # 创建一个新的有向图来避免直接修改原始图
    new_T = T.copy()

    # 添加新的根节点，并从新根指向旧根
    new_T.add_node(new_root_name)
    new_T.add_edge(new_root_name, old_root)

    return new_T

def remove_zero_indegree(T):
    temp_T = T
    nodes_to_remove = []
    for node in temp_T.nodes():
        if temp_T.in_degree(node) == 0 and node != 'Beijing':
            nodes_to_remove.append(node)
    temp_T.remove_nodes_from(nodes_to_remove)
    return temp_T

T = remove_degree_one_nodes(T)
T = remove_zero_indegree(T)
T = add_new_root(T, "root", "Beijing")

pos = nx.drawing.nx_agraph.graphviz_layout(T, prog='dot', args='-Gnodesep=2 -Grankdir=TB')
# 制图形
plt.figure(figsize=(12, 8))  # 调整图像大小以便更好地查看
nx.draw(T, pos, with_labels=True, node_size=50, node_color='skyblue', font_size=8)
plt.title('Graph Visualization of Chinanet')
plt.show()

# print(T.nodes)

pos=nx.drawing.nx_agraph.graphviz_layout(T, prog='dot')
nx.draw(T, pos, with_labels=False)
plt.show()

# for edge in T.edges():
#     print(edge)
in_degree['root'] = 0
out_degree['root'] = 1

# 定义链路集合
link_list = {}
num_links = 0
def cal_path(G, node, path, paths, link_list):
    global num_links
    path.append(node)
    if out_degree[node] == 0:
        paths.append(path.copy())
    else:
        for neighbor in G.neighbors(node):
            link_list[(node, neighbor)] = num_links
            num_links += 1
            cal_path(G, neighbor, path, paths, link_list)
    path.pop()

path_list = []
cal_path(T, 'root', [], path_list, link_list)

# 初始化路径-链路关系矩阵
num_paths = len(path_list)
num_links = len(link_list)
R = np.zeros((num_paths, num_links), dtype=int)

# 填充路径-链路矩阵
for path_idx, path in enumerate(path_list):
    for i in range(len(path) - 1):
        link = (path[i], path[i + 1])
        if link in link_list:
            link_idx = link_list[link]
            R[path_idx][link_idx] = 1

df_R = pd.DataFrame(R, index=[f"Path_{i}" for i in range(1, len(path_list) + 1)], columns=[f"Link_{i}" for i in range(1, len(link_list) + 1)])

def write_matrix_to_file(matrix, filepath):
    """
    将矩阵写入到指定路径的文件中。

    参数:
    - matrix: 要写入的矩阵（numpy数组）
    - filepath: 输出文件的完整路径
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.savetxt(filepath, matrix, fmt='%d', delimiter='\t')
        print(f"Matrix has been written to {filepath}")
    except Exception as e:
        print(f"Error writing matrix to file: {e}")


output_dir = './output/'  # 可根据需要修改为你想要保存文件的目录
matrix_filename = os.path.join(output_dir, 'route_matrix.txt')
paths_filename = os.path.join(output_dir, 'paths_with_matrix_rows.txt')

# 写入路由矩阵到文件
write_matrix_to_file(R, matrix_filename)

col_sums = np.sum(R, axis=0)

for link in link_list:
    print(f"Link: {link}, no:{link_list[link]}")

def draw_tree_from_routing_matrix(A_rm: np.ndarray):
    """
    根据路由矩阵绘制树型拓扑图
    注意：要求 A_rm 各行路径上的链路（从左往右先后出现）的顺序，与其从 “根节点 --> 目的节点” 方向出现的顺序保持一致

    A_rm: 2维的路由矩阵，行为路径，列为链路
    """

    num_paths, num_links = A_rm.shape  # 路径数，链路数

    # 创建链路和目标节点列表，链路和路径均从 0 开始编号
    node_id, dest_marked = 0, [[i, 0] for i in range(num_paths)]
    edge_list, edge_marked = [], [[i, 0] for i in range(num_links)]
    for i in range(num_paths):
        link_src = 0

        for j in range(num_links):
            if not edge_marked[j][1] and A_rm[i, j]:
                node_id += 1  # 新增节点
                link_dest = copy(node_id)  # 将新增节点作为新增边的目的节点

                edge_list.append((link_src, link_dest, {'label': j}))  # 添加边
                edge_marked[j][1] = len(edge_list)  # 标记边信息存放的位置

                link_src = copy(link_dest)  # 更新源节点

            elif edge_marked[j][1] and A_rm[i, j]:
                link_src = edge_list[edge_marked[j][1] - 1][1]  # 更新源节点

        dest_marked[i][1] = copy(link_src)  # 记录路径的目的节点

    assert all(edge_marked)  # 检查是否所有边都已成功标记
    assert all(dest_marked)  # 检查是否所有路径的目的节点都已成功标记

    # 以链路列表信息，创建一个有向树图
    T = nx.DiGraph()
    T.add_edges_from(edge_list)

    assert nx.is_tree(T);  # 检查是否为树图

    # 基于 Reingold-Tilford 树布局算法，计算树中各节点的位置
    tree_layout = ig.Graph.from_networkx(T).layout_reingold_tilford(root=[0])  # 以节点 0 为根节点

    pos, ind = {}, {i for edge in edge_list for i in edge[:2]}  # 根据链路顺序，获得节点的顺序 存放于 ind 中
    for i, e in zip(range(len(tree_layout)), ind):
        pos[e] = (tree_layout.coords[i][0], tree_layout.coords[i][1] * -1)  # 为了符合作图习惯，反转 tree_layout 中的 y 轴坐标；
        # 不然树拓扑是朝上画，即根节点在最下方

    plt.figure(figsize=(12, 8))
    # 绘制树状图
    nx.draw(T, pos,
            node_size=10, node_color="black",
            edge_color="black",
            linewidths=0.5, width=0.5,
            arrowsize=5,
            arrowstyle="->",
            arrows=True,
            style="dashed")

    # 标注链路编号
    edge_labels = {(u, v): f'$\ell{l["label"]}$' for j, (u, v, l) in enumerate(edge_list)}
    nx.draw_networkx_edge_labels(T, pos, edge_labels=edge_labels, font_size=8, font_color='blue')

    # 标注路径编号
    for i in range(num_paths):
        plt.text(pos[dest_marked[i][1]][0], pos[dest_marked[i][1]][1] - 0.15, \
                 f'$p{i}$', fontsize=8, color='red')

    plt.show()
draw_tree_from_routing_matrix(R)