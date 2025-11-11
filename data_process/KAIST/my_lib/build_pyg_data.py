import os
import scipy.io as sio
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

def build_local_temporal_graph(csv_path: str, save_dir: str, num_edges: int = 10):
    """
    基于时间顺序构建局部时序图。
    每一行是一个节点，上下相邻样本构成边。

    参数:
        csv_path (str): 输入 CSV 文件路径。
        save_dir (str): 图结构文件的保存文件夹。
        num_edges (int): 每个节点的边数（上下平均分配）。
                         例如 10 表示上5下5；若边界不足则单边补齐。
    返回:
        (nodes_csv, edges_csv, graph_pt): 保存的文件路径元组。
    """
    os.makedirs(save_dir, exist_ok=True)

    # 读取数据
    df = pd.read_csv(csv_path)
    num_nodes = len(df)
    half = num_edges // 2

    # === 构建边列表 ===
    edges = []
    for i in range(num_nodes):
        # 上方节点索引
        start_up = max(0, i - half)
        # 下方节点索引
        end_down = min(num_nodes, i + half + 1)

        up_neighbors = list(range(start_up, i))
        down_neighbors = list(range(i + 1, end_down))

        # 若两边不够数量，补另一边
        total_needed = num_edges
        current = len(up_neighbors) + len(down_neighbors)
        if current < total_needed:
            remaining = total_needed - current
            # 优先补下边
            if i + half + 1 >= num_nodes:  # 下方不够
                extra_up = list(range(max(0, start_up - remaining), start_up))
                up_neighbors = extra_up + up_neighbors
            elif i - half < 0:  # 上方不够
                extra_down = list(range(end_down, min(num_nodes, end_down + remaining)))
                down_neighbors += extra_down

        # 添加边（双向）
        for j in up_neighbors + down_neighbors:
            edges.append((i, j))
            edges.append((j, i))

    # === 保存节点与边 ===
    nodes_path = os.path.join(save_dir, "nodes.csv")
    edges_path = os.path.join(save_dir, "edges.csv")
    graph_path = os.path.join(save_dir, "graph.pt")

    df.to_csv(nodes_path, index=False)
    edge_df = pd.DataFrame(edges, columns=["source", "target"])
    edge_df.to_csv(edges_path, index=False)

    # === 转换为PyG图结构 ===
    edge_index = torch.tensor(edges, dtype=torch.long).T
    x = torch.tensor(df.values, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)

    torch.save(data, graph_path)

    print(f"✅ 图结构构建完成，共 {num_nodes} 个节点，{len(edges)//2} 条无向边")
    print(f"📁 nodes.csv: {nodes_path}")
    print(f"📁 edges.csv: {edges_path}")
    print(f"📁 graph.pt : {graph_path}")

    return nodes_path, edges_path, graph_path

def build_similarity_knn_graph(csv_path: str, save_dir: str, num_edges: int = 10):
    """
    基于样本间余弦相似度 + KNN 建图。
    忽略首列(序号)与末列(标签)，输出结构与 build_local_temporal_graph 一致。

    参数:
        csv_path (str): 输入CSV文件路径。
        save_dir (str): 图结构文件的保存文件夹。
        num_edges (int): 每个节点连接的邻点数(KNN数量)。
    返回:
        (nodes_csv, edges_csv, graph_pt): 保存的文件路径元组。
    """

    # ===================== 1️⃣ 读取数据 =====================
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ 找不到输入文件: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"📊 已读取数据: {df.shape}")

    # 🚫 忽略首列和末列
    if df.shape[1] <= 2:
        raise ValueError("❌ 数据列数过少，无法同时忽略首列和末列。")
    df = df.iloc[:, 1:-1]

    # 仅保留数值列
    df = df.select_dtypes(include=["float", "int"])
    features = df.values.astype(np.float32)
    num_nodes = features.shape[0]
    print(f"🧩 使用特征列数: {features.shape[1]} | 忽略首尾列后: {list(df.columns)[:5]} ...")

    # ===================== 2️⃣ 计算余弦相似度矩阵 =====================
    print("⚙️ 正在计算余弦相似度矩阵...")
    sim_matrix = cosine_similarity(features)
    np.fill_diagonal(sim_matrix, -np.inf)  # 排除自身

    # ===================== 3️⃣ 构建 KNN 边 =====================
    print(f"🔍 正在为每个节点选取 {num_edges} 个最相似邻居...")
    edges = []
    for i in range(num_nodes):
        topk_idx = np.argpartition(sim_matrix[i], -num_edges)[-num_edges:]
        for j in topk_idx:
            edges.append([i, j])
            edges.append([j, i])  # 无向边

    edges = np.array(edges)
    edge_index = torch.tensor(edges.T, dtype=torch.long)
    x = torch.tensor(features, dtype=torch.float)

    # ===================== 4️⃣ 构建 PyG Data 对象 =====================
    data = Data(x=x, edge_index=edge_index)

    # ===================== 5️⃣ 保存文件 =====================
    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    nodes_csv = os.path.join(save_dir, f"{base_name}_nodes.csv")
    edges_csv = os.path.join(save_dir, f"{base_name}_edges.csv")
    graph_pt = os.path.join(save_dir, f"{base_name}_graph.pt")

    # 保存节点与边文件
    pd.DataFrame(features).to_csv(nodes_csv, index=False)
    pd.DataFrame(edges, columns=["source", "target"]).to_csv(edges_csv, index=False)
    torch.save(data, graph_pt)

    print(f"✅ 图构建完成，共 {num_nodes} 个节点，{len(edges)//2} 条无向边。")
    print(f"📁 节点文件: {nodes_csv}")
    print(f"📁 边文件:   {edges_csv}")
    print(f"📁 图文件:   {graph_pt}")

    return nodes_csv, edges_csv, graph_pt