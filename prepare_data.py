#!/usr/bin/env python3
"""
数据准备脚本 - DVC流水线阶段1
"""
import os
import numpy as np
import cv2
import pickle
from tqdm import tqdm
import yaml

def load_params():
    """加载DVC参数"""
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f)

def prepare_data():
    """准备数据"""
    params = load_params()
    data_params = params['data']
    preprocess_params = params['preprocess']
    
    print("🔄 准备数据...")
    
    # 创建输出目录
    os.makedirs('data/processed', exist_ok=True)
    
    # 加载图像数据
    images = []
    labels = []
    image_paths = []
    
    data_path = data_params['path']
    for category in os.listdir(data_path):
        category_path = os.path.join(data_path, category)
        if os.path.isdir(category_path):
            for img_file in tqdm(os.listdir(category_path), desc=f"处理 {category}"):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(category_path, img_file)
                    
                    # 读取图像
                    if preprocess_params['grayscale']:
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    else:
                        img = cv2.imread(img_path)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    if img is not None:
                        # 调整尺寸
                        img = cv2.resize(img, tuple(preprocess_params['image_size']))
                        
                        # 归一化
                        if preprocess_params['normalize']:
                            img = img.astype(np.float32) / 255.0
                        
                        images.append(img)
                        labels.append(category)
                        image_paths.append(img_path)
    
    # 转换为numpy数组
    images = np.array(images)
    labels = np.array(labels)
    
    # 保存处理后的数据
    np.save('data/processed/images.npy', images)
    np.save('data/processed/labels.npy', labels)
    
    with open('data/processed/image_paths.pkl', 'wb') as f:
        pickle.dump(image_paths, f)
    
    print(f"✅ 数据准备完成: {len(images)} 张图像")
    print(f"   图像形状: {images.shape}")
    print(f"   类别数量: {len(np.unique(labels))}")

if __name__ == "__main__":
    prepare_data()