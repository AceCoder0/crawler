import json
from sklearn.cluster import DBSCAN
import numpy as np
from apted import APTED, Config
from lxml import html
from itertools import combinations
from tqdm import tqdm
from multiprocessing import Pool
from joblib import Parallel, delayed


class AptedNode:
    """自定义节点类，用于APTED库"""
    def __init__(self, name, children=None):
        self.name = name
        self.children = children or []

    def __repr__(self):
        return f"AptedNode(name={self.name}, children={self.children})"

def html_to_tree(html_str):
    """将HTML字符串转换为APTED库所需的树结构"""
    def traverse(node):
        # 创建一个新的节点，标签名为节点名
        tree_node = AptedNode(node.tag)
        for child in node:
            # 递归地将子节点添加到当前节点中
            tree_node.children.append(traverse(child))
        return tree_node

    # 解析HTML字符串为DOM树
    dom_tree = html.fromstring(html_str)
    # 将DOM树转换为APTED库所需的树结构
    return traverse(dom_tree)

def calculate_normalized_ted(html_a, html_b):
    """计算两个HTML字符串之间的归一化TED"""
    tree_a = html_to_tree(html_a)
    tree_b = html_to_tree(html_b)
    
    # 使用APTED计算编辑距离
    apted = APTED(tree_a, tree_b, Config())
    distance = apted.compute_edit_distance()
    
    # 计算两个DOM树的节点数量
    len_a = len(list(html.fromstring(html_a).iter()))
    len_b = len(list(html.fromstring(html_b).iter()))
    
    # 归一化编辑距离
    normalized_distance = distance / max(len_a, len_b) if max(len_a, len_b) > 0 else 0.0
    return normalized_distance

def compute_distance_matrix(html_list):
    """计算HTML列表的距离矩阵，并显示进度条"""
    n = len(html_list)
    distance_matrix = np.zeros((n, n))
    
    # 使用itertools.combinations生成所有唯一的(i, j)对
    pairs = list(combinations(range(n), 2))
    
    # 使用tqdm显示进度条
    for i, j in tqdm(pairs, desc="Calculating distances"):
            distance = calculate_normalized_ted(html_list[i], html_list[j])
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

def parallel_compute_distance_matrix(html_list):
    """并行计算HTML列表的距离矩阵"""

    from concurrent.futures import ProcessPoolExecutor, as_completed

    n = len(html_list)
    distance_matrix = np.zeros((n, n))

    # 使用itertools.combinations生成所有唯一的(i, j)对
    pairs = list(combinations(range(n), 2))

    # 使用ProcessPoolExecutor并行计算
    with ProcessPoolExecutor() as executor:
        # 提交任务
        futures = {executor.submit(calculate_normalized_ted, html_list[i], html_list[j]): (i, j) for i, j in pairs}
        
        # 使用tqdm显示进度
        for future in tqdm(as_completed(futures), total=len(pairs), desc="Calculating distances"):
            i, j = futures[future]
            distance = future.result()
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
    return distance_matrix



# Define compute_distance at the top level
def compute_distance(html_list, i, j):
    distance = calculate_normalized_ted(html_list[i], html_list[j])
    return i, j, distance

def parallel_compute_distance_matrix_jb(html_list):
    """并行计算HTML列表的距离矩阵"""
    n = len(html_list)
    distance_matrix = np.zeros((n, n))
    
    # Generate all unique pairs (i, j)
    pairs = list(combinations(range(n), 2))
    
    # Use joblib to compute in parallel
    results = Parallel(n_jobs=-1, backend='multiprocessing', verbose=0)(
        delayed(compute_distance)(html_list, i, j) for i, j in tqdm(pairs, desc="Calculating distances")
    )
    
    # Fill the distance matrix
    for i, j, distance in results:
        distance_matrix[i, j] = distance
        distance_matrix[j, i] = distance
    
    return distance_matrix

def cluster_htmls(html_data, eps=0.5, min_samples=2):
    """对HTML列表进行聚类"""
    # 提取HTML内容
    html_list = [item['html_content'] for item in html_data]
    
    # 计算距离矩阵
    # distance_matrix = compute_distance_matrix(html_list)
    distance_matrix = parallel_compute_distance_matrix(html_list)
    
    # 使用DBSCAN进行聚类
    dbscan = DBSCAN(metric='precomputed', eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(distance_matrix)
    
    # 将聚类结果添加到原始数据中
    for i, item in enumerate(html_data):
        item['cluster_label'] = int(labels[i])
        item.pop('html_content')
    
    # 返回聚类数量（排除噪声点）
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return n_clusters, html_data

def save_clustering_results(results, output_file):
    """将聚类结果保存到JSON文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

def load_html_content(html_data):
    """加载HTML内容"""
    for item in html_data:
        with open(item['html_path'], 'r', encoding='utf-8') as f:
                item['html_content'] = f.read()

# 示例用法
if __name__ == "__main__":
    # 输入数据：包含URL和文件路径的字典列表
    import json
    html_data = json.load(open('test/https--www-zaobao-com-/table.json'))

    # 加载HTML内容
    load_html_content(html_data)

    # 对HTML进行聚类
    n_clusters, clustered_data = cluster_htmls(html_data, eps=0.015, min_samples=2)

    # 输出聚类数量
    print(f"Number of clusters: {n_clusters}")

    # 将聚类结果保存到JSON文件
    save_clustering_results(clustered_data, "clustering_results.json")