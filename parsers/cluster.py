import json
from sklearn.cluster import DBSCAN
import numpy as np
from lxml import html
from itertools import combinations
from tqdm import tqdm
from multiprocessing import Pool
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter
from lxml import html
from bs4 import BeautifulSoup


def dom_to_feature_vector(html_str):
    """将 DOM 树转换为特征向量"""
    dom_tree = html.fromstring(html_str)
    tags = [node.tag for node in dom_tree.iter() if isinstance(node.tag, str)]
    tag_counter = Counter(tags)
    # 示例标签（可以根据需要扩展）
    feature_vector = [tag_counter.get(tag, 0) for tag in ['div', 'p', 'a', 'span', 'img', 'h1', 'h2', 'h3', 'ul', 'li']]
    return feature_vector

def ss_distance(html_str1, html_str2):
    from html_similarity.style_similarity import style_similarity
    from html_similarity.structural_similarity import structural_similarity


    def similarity(document_1, document_2, k=0.5):
        return k * structural_similarity(document_1, document_2) + (1 - k) * style_similarity(document_1, document_2)
    return 1-similarity(html_str1, html_str2)

def remove_script_style(html_str):
    """
    去除HTML字符串中的<script>和<style>标签及其内容

    参数:
    html_str (str): 输入的HTML字符串

    返回:
    str: 处理后的HTML字符串
    """
    soup = BeautifulSoup(html_str, 'html.parser')
    # 找到所有的<script>和<style>标签并删除
    for tag in soup.find_all(['script', 'style']):
        tag.decompose()
    # 返回处理后的HTML字符串
    return str(soup)

def dom_to_feature_vectorv2(html_str, max_depth=5):
    """将 DOM 树转换为特征向量，考虑标签的深度信息"""
    dom_tree = html.fromstring(html_str)
    
    # 定义常见的标签列表
    common_tags = ['html', 'head', 'title', 'body', 'div', 'p', 'a', 'span', 'img',
                   'h1', 'h2', 'h3', 'ul', 'ol', 'li', 'table', 'tr', 'td',
                   'header', 'footer', 'nav', 'section', 'article']
    
    # 统计每个标签在不同深度的出现次数
    tag_depth_counter = defaultdict(Counter)
    
    # 遍历 DOM 树，并记录标签及其深度
    def traverse(node, depth):
        if depth > max_depth:
            return
        if isinstance(node.tag, str):
            tag_depth_counter[node.tag][depth] += 1
        for child in node:
            traverse(child, depth + 1)
    
    traverse(dom_tree, 1)  # 从深度1开始
    
    # 构建特征向量
    feature_vector = []
    for tag in common_tags:
        for depth in range(1, max_depth + 1):
            feature_vector.append(tag_depth_counter[tag].get(depth, 0))
    
    return feature_vector

def calculate_feature_distance(html_a, html_b):
    """基于特征向量计算两个 HTML 之间的距离"""
    vec_a = dom_to_feature_vectorv2(html_a)
    vec_b = dom_to_feature_vectorv2(html_b)
    # 使用余弦相似度
    similarity = cosine_similarity([vec_a], [vec_b])[0][0]
    # 将相似度转换为距离（距离 = 1 - 相似度）
    distance = 1 - similarity
    return distance

def compute_distance_matrix(html_list, dist_func=calculate_feature_distance):
    """计算HTML列表的距离矩阵，并显示进度条"""
    n = len(html_list)
    distance_matrix = np.zeros((n, n))
    
    # 使用itertools.combinations生成所有唯一的(i, j)对
    pairs = list(combinations(range(n), 2))
    print(f"{n=}, {len(pairs)=}")
    # 使用tqdm显示进度条
    for i, j in tqdm(pairs, desc="Calculating distances"):
            distance = calculate_feature_distance(html_list[i], html_list[j])
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
    
    return distance_matrix


def cluster_htmls(html_data, eps=0.5, min_samples=2, eplion=1e-5):
    """对HTML列表进行聚类"""
    # 提取HTML内容
    html_list = [item['html_content'] for item in html_data]
    html_list = [remove_script_style(i) for i in html_list]
    # 计算距离矩阵
    # distance_matrix = parallel_compute_distance_matrix(html_list)
    distance_matrix = compute_distance_matrix(html_list, dist_func=ss_distance)+eplion
    
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
    html_data = json.load(open('test/https--www-zaobao-com-/table.json'))[:500]

    # 加载HTML内容
    load_html_content(html_data)

    # 对HTML进行聚类
    n_clusters, clustered_data = cluster_htmls(html_data, eps=0.015, min_samples=2)

    # 输出聚类数量
    print(f"Number of clusters: {n_clusters}")

    # 将聚类结果保存到JSON文件
    save_clustering_results(clustered_data, "clustering_results.json")