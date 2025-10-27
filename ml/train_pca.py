#!/usr/bin/env python3
"""
训练PCA模型 - DVC流水线阶段2（集成MLflow）
"""
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import yaml
import os
import json
import mlflow  # 导入MLflow
from mlflow.models.signature import infer_signature  # 用于模型签名



import os  # 需导入os模块处理路径

def load_params():
    """加载ml目录下的params.yaml"""
    # 获取当前脚本（train_pca.py）所在的目录路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 拼接得到params.yaml的完整路径
    params_path = os.path.join(script_dir, 'params.yaml')
    with open(params_path, 'r') as f:
        return yaml.safe_load(f)

def train_pca():
    """训练PCA模型（集成MLflow跟踪）"""
    params = load_params()
    pca_params = params['pca']
    
    print("🎯 训练PCA模型并使用MLflow跟踪...")
    
    # 创建输出目录
    os.makedirs('models', exist_ok=True)
    
    # 加载数据
    images = np.load('data/processed/images.npy')
    n_samples = images.shape[0]
    flattened_images = images.reshape(n_samples, -1)
    
    # 标准化
    scaler = StandardScaler()
    scaled_images = scaler.fit_transform(flattened_images)
    
    # 启动MLflow运行
    with mlflow.start_run(run_name="pca_training"):
        # 记录参数
        mlflow.log_param("n_components", pca_params['n_components'])
        mlflow.log_param("random_state", pca_params['random_state'])
        mlflow.log_param("original_dimension", scaled_images.shape[1])
        mlflow.log_param("n_samples", n_samples)
        
        # 训练PCA
        pca = PCA(n_components=pca_params['n_components'], 
                  random_state=pca_params['random_state'])
        images_pca = pca.fit_transform(scaled_images)
        
        # 记录指标
        explained_variance = float(pca.explained_variance_ratio_.sum())
        mlflow.log_metric("explained_variance_ratio", explained_variance)
        mlflow.log_metric("dimension_reduction", 
                         scaled_images.shape[1] - pca_params['n_components'])
        
        # 保存模型和数据到本地
        with open('models/pca_model.pkl', 'wb') as f:
            pickle.dump(pca, f)
        
        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        
        np.save('data/processed/images_pca.npy', images_pca)
        
        # 保存PCA指标到本地
        pca_metrics = {
            'explained_variance_ratio': explained_variance,
            'original_dimension': scaled_images.shape[1],
            'reduced_dimension': pca_params['n_components'],
            'n_samples': n_samples
        }
        
        with open('models/pca_metrics.json', 'w') as f:
            json.dump(pca_metrics, f, indent=2)
        
        # 记录模型到MLflow（包含签名）
        signature = infer_signature(scaled_images, images_pca)
        mlflow.sklearn.log_model(
            sk_model=pca,
            artifact_path="pca_model",
            signature=signature,
            registered_model_name="pca-image-reducer"  # 可选：注册模型
        )
        
        # 记录scaler到MLflow
        mlflow.sklearn.log_model(
            sk_model=scaler,
            artifact_path="scaler",
            signature=infer_signature(flattened_images, scaled_images)
        )
        
        # 记录输出文件作为 artifact
        mlflow.log_artifact("models/pca_metrics.json")
        mlflow.log_artifact("data/processed/images_pca.npy")
    
    print("✅ PCA训练完成（已记录到MLflow）")
    print(f"   累计解释方差: {explained_variance:.4f}")
    print(f"   维度缩减: {scaled_images.shape[1]} -> {pca_params['n_components']}")

if __name__ == "__main__":
    # 可选：设置MLflow跟踪URI（默认使用本地文件系统）
    # mlflow.set_tracking_uri("http://mlflow-server:5000")  # 如有远程服务器
    train_pca()