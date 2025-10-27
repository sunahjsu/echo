#!/usr/bin/env python3
"""
训练PCA模型 - DVC流水线阶段2
"""
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import yaml
import os
import json

def load_params():
    """加载DVC参数"""
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f)

def train_pca():
    """训练PCA模型"""
    params = load_params()
    pca_params = params['pca']
    
    print("🎯 训练PCA模型...")
    
    # 创建输出目录
    os.makedirs('models', exist_ok=True)
    
    # 加载数据
    images = np.load('data/processed/images.npy')
    n_samples = images.shape[0]
    flattened_images = images.reshape(n_samples, -1)
    
    # 标准化
    scaler = StandardScaler()
    scaled_images = scaler.fit_transform(flattened_images)
    
    # 训练PCA
    pca = PCA(n_components=pca_params['n_components'], 
              random_state=pca_params['random_state'])
    images_pca = pca.fit_transform(scaled_images)
    
    # 保存模型和数据
    with open('models/pca_model.pkl', 'wb') as f:
        pickle.dump(pca, f)
    
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    np.save('data/processed/images_pca.npy', images_pca)
    
    # 保存PCA指标
    pca_metrics = {
        'explained_variance_ratio': float(pca.explained_variance_ratio_.sum()),
        'original_dimension': scaled_images.shape[1],
        'reduced_dimension': pca_params['n_components'],
        'n_samples': n_samples
    }
    
    with open('models/pca_metrics.json', 'w') as f:
        json.dump(pca_metrics, f, indent=2)
    
    print("✅ PCA训练完成")
    print(f"   累计解释方差: {pca_metrics['explained_variance_ratio']:.4f}")
    print(f"   维度缩减: {pca_metrics['original_dimension']} -> {pca_metrics['reduced_dimension']}")

if __name__ == "__main__":
    train_pca()