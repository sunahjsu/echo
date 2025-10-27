#!/usr/bin/env python3
"""
评估检索性能 - DVC流水线阶段3
"""
import numpy as np
import json
from scipy.spatial.distance import euclidean
import time
from tqdm import tqdm
import yaml
import os
import matplotlib.pyplot as plt

def load_params():
    """加载DVC参数"""
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f)

def evaluate_retrieval():
    """评估检索性能"""
    params = load_params()
    retrieval_params = params['retrieval']
    evaluation_params = params['evaluation']
    
    print("🔍 评估检索性能...")
    
    # 创建输出目录
    os.makedirs('results', exist_ok=True)
    
    # 加载数据
    original_images = np.load('data/processed/images.npy').reshape(
        len(np.load('data/processed/images.npy')), -1
    )
    pca_images = np.load('data/processed/images_pca.npy')
    labels = np.load('data/processed/labels.npy')
    
    # 随机选择查询图像
    np.random.seed(evaluation_params['random_state'])
    query_indices = np.random.choice(
        len(original_images), 
        evaluation_params['n_queries'], 
        replace=False
    )
    
    def retrieve_similar(query_idx, images, top_k):
        """检索相似图像"""
        query_vec = images[query_idx]
        query_label = labels[query_idx]
        
        start_time = time.time()
        distances = []
        
        for i, vec in enumerate(images):
            if i != query_idx:
                dist = euclidean(query_vec, vec)
                distances.append((i, dist))
        
        distances.sort(key=lambda x: x[1])
        top_matches = distances[:top_k]
        
        # 计算准确率
        correct_matches = sum(1 for idx, _ in top_matches if labels[idx] == query_label)
        accuracy = correct_matches / top_k
        
        return accuracy, time.time() - start_time
    
    # 评估两种方法
    original_accuracies = []
    original_times = []
    pca_accuracies = []
    pca_times = []
    
    for query_idx in tqdm(query_indices, desc="评估查询"):
        # 原始向量检索
        acc_orig, time_orig = retrieve_similar(
            query_idx, original_images, retrieval_params['top_k']
        )
        original_accuracies.append(acc_orig)
        original_times.append(time_orig)
        
        # PCA检索
        acc_pca, time_pca = retrieve_similar(
            query_idx, pca_images, retrieval_params['top_k']
        )
        pca_accuracies.append(acc_pca)
        pca_times.append(time_pca)
    
    # 计算指标
    metrics = {
        'original_accuracy': float(np.mean(original_accuracies)),
        'original_time': float(np.mean(original_times)),
        'pca_accuracy': float(np.mean(pca_accuracies)),
        'pca_time': float(np.mean(pca_times)),
        'speedup_ratio': float(np.mean(original_times) / np.mean(pca_times)),
        'accuracy_difference': float(np.mean(pca_accuracies) - np.mean(original_accuracies)),
        'n_queries': evaluation_params['n_queries']
    }
    
    # 保存指标
    with open('results/evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # 创建可视化
    create_plots(metrics, original_accuracies, pca_accuracies, original_times, pca_times)
    
    print("✅ 评估完成")
    print(f"   原始准确率: {metrics['original_accuracy']:.3f}")
    print(f"   PCA准确率: {metrics['pca_accuracy']:.3f}")
    print(f"   速度提升: {metrics['speedup_ratio']:.2f}x")

def create_plots(metrics, orig_acc, pca_acc, orig_time, pca_time):
    """创建评估图表"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # 准确率比较
    methods = ['Original', 'PCA']
    accuracies = [metrics['original_accuracy'], metrics['pca_accuracy']]
    ax1.bar(methods, accuracies, color=['skyblue', 'lightcoral'])
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Retrieval Accuracy Comparison')
    ax1.set_ylim(0, 1)
    
    # 时间比较
    times = [metrics['original_time'], metrics['pca_time']]
    ax2.bar(methods, times, color=['skyblue', 'lightcoral'])
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('Retrieval Time Comparison')
    
    # 准确率分布
    ax3.boxplot([orig_acc, pca_acc], labels=methods)
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Accuracy Distribution')
    
    # 速度提升
    ax4.bar(['Speedup'], [metrics['speedup_ratio']], color='lightgreen')
    ax4.set_ylabel('Ratio')
    ax4.set_title(f'Speedup Ratio: {metrics["speedup_ratio"]:.2f}x')
    
    plt.tight_layout()
    plt.savefig('results/accuracy_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    evaluate_retrieval()