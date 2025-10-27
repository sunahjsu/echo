import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from typing import List, Tuple, Dict
import time

class PCASVMImageRetrieval:
    def __init__(self, dataset_path: str, n_components: int = 50):
        self.dataset_path = dataset_path
        self.n_components = n_components
        self.images = []
        self.labels = []
        self.filenames = []
        self.images_pca = None  # PCA降维后的特征
        self.pca = None
        self.scaler = StandardScaler()
        self.svm_model = None
        
    def load_dataset(self):
        """加载数据集并提取特征（确保数据非空）"""
        print("正在加载数据集...")
        self.images = []
        self.labels = []
        self.filenames = []
        
        if not os.path.exists(self.dataset_path):
            print(f"警告: 数据集目录 {self.dataset_path} 不存在，创建空数据集")
            return self
            
        for class_name in os.listdir(self.dataset_path):
            class_path = os.path.join(self.dataset_path, class_name)
            if os.path.isdir(class_path):
                for img_file in os.listdir(class_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(class_path, img_file)
                        img = cv2.imread(img_path)
                        if img is not None:
                            img = cv2.resize(img, (128, 128))
                            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            flattened = gray_img.flatten()
                            self.images.append(flattened)
                            self.labels.append(class_name)
                            self.filenames.append(img_path)
        
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        print(f"数据集加载完成，共 {len(self.images)} 张图像")
        return self
    
    def apply_pca(self):
        """应用PCA降维（增加数据校验和异常处理）"""
        print("应用PCA降维...")
        if len(self.images) == 0:
            print("警告：无图像数据，跳过PCA降维")
            self.images_pca = None
            return self
        
        # 标准化数据
        images_scaled = self.scaler.fit_transform(self.images)
        
        # 检查数据方差是否为0（避免除以零错误）
        data_variance = np.var(images_scaled)
        if data_variance < 1e-9:  # 方差接近0，数据无差异
            print("警告：数据方差接近0，无法进行PCA，使用原始特征")
            self.images_pca = images_scaled
            self.pca = None
            return self
        
        # 调整n_components（避免超过特征数或样本数）
        max_possible_components = min(self.n_components, self.images.shape[1], len(self.images))
        if max_possible_components != self.n_components:
            print(f"调整PCA组件数：{self.n_components} -> {max_possible_components}（受数据限制）")
            self.n_components = max_possible_components
        
        # 应用PCA
        self.pca = PCA(n_components=self.n_components)
        self.images_pca = self.pca.fit_transform(images_scaled)
        
        # 处理可能的nan解释方差比
        explained_variance = self.pca.explained_variance_ratio_.sum()
        if np.isnan(explained_variance):
            explained_variance = 0.0
        print(f"降维完成: {self.images.shape[1]} -> {self.n_components} 维")
        print(f"解释方差比: {explained_variance:.4f}")
        return self
    
    def train_svm(self):
        """训练SVM分类器"""
        if self.images_pca is None or len(self.images_pca) == 0:
            print("警告：无PCA特征数据，跳过SVM训练")
            return self
            
        print("训练SVM分类器...")
        self.svm_model = SVC(kernel='rbf', probability=True, random_state=42)
        self.svm_model.fit(self.images_pca, self.labels)
        print("SVM训练完成")
        return self
    
    def find_query_index(self, query_image_path: str) -> int:
        """查找查询图像在数据集中的索引"""
        query_abs_path = os.path.abspath(query_image_path)
        for i, filename in enumerate(self.filenames):
            if os.path.abspath(filename) == query_abs_path:
                return int(i)
        return -1
    
    def retrieve_similar(self, query_idx: int, use_pca: bool = True) -> Dict:
        """检索相似图像（动态调整返回数量）"""
        start_time = time.time()
        
        # 校验查询索引有效性
        if query_idx < 0 or query_idx >= len(self.images):
            raise ValueError(f"无效的查询索引：{query_idx}（总图像数：{len(self.images)}）")
        
        # 选择特征（PCA或原始特征）
        if use_pca and self.images_pca is not None:
            features = self.images_pca
            query_feature = self.images_pca[query_idx]
            method = "PCA检索"
        else:
            images_scaled = self.scaler.transform(self.images)
            features = images_scaled
            query_feature = images_scaled[query_idx]
            method = "原始特征检索"
        
        # 计算欧氏距离
        distances = np.linalg.norm(features - query_feature, axis=1)
        
        # 排序并排除自身
        sorted_indices = np.argsort(distances)
        top_indices = [idx for idx in sorted_indices if idx != query_idx]
        
        # 取前10个（若不足10个则返回全部）
        top_k = min(10, len(top_indices))
        top_indices = top_indices[:top_k]
        top_distances = distances[top_indices]
        
        # 计算准确率
        query_label = self.labels[query_idx]
        correct_count = np.sum(self.labels[top_indices] == query_label)
        accuracy = correct_count / len(top_indices) if len(top_indices) > 0 else 0.0
        
        end_time = time.time()
        return {
            "method": method,
            "query_index": int(query_idx),
            "results": [(int(idx), float(dist)) for idx, dist in zip(top_indices, top_distances)],
            "accuracy": float(accuracy),
            "retrieval_time": float(end_time - start_time),
            "feature_dim": int(features.shape[1])
        }
    
    def show_results(self, result: Dict):
        """可视化检索结果"""
        query_idx = result["query_index"]
        if query_idx < 0 or query_idx >= len(self.filenames):
            print("无效的查询索引，无法显示结果")
            return
            
        plt.figure(figsize=(15, 5))
        plt.suptitle(f"{result['method']} - 准确率: {result['accuracy']:.2f} - 时间: {result['retrieval_time']:.3f}s")
        
        # 显示查询图像
        plt.subplot(1, len(result['results'])+1, 1)
        img = cv2.imread(self.filenames[query_idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else np.zeros((128,128,3))
        plt.imshow(img)
        plt.title("查询图像")
        plt.axis('off')
        
        # 显示检索结果
        for i, (idx, dist) in enumerate(result["results"]):
            plt.subplot(1, len(result['results'])+1, i + 2)
            img = cv2.imread(self.filenames[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else np.zeros((128,128,3))
            plt.imshow(img)
            
            same_class = "同类" if self.labels[idx] == self.labels[query_idx] else "不同类"
            plt.title(f"结果{i+1}\n{same_class}\n距离: {dist:.2f}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()

def run_experiment(dataset_path: str, query_image_path: str):
    """运行完整实验并记录到MLflow"""
    mlflow.set_experiment("PCA_Image_Retrieval")
    
    with mlflow.start_run():
        retrieval = PCASVMImageRetrieval(dataset_path, n_components=50)
        retrieval.load_dataset().apply_pca().train_svm()
        
        mlflow.log_param("n_components", retrieval.n_components)
        mlflow.log_param("dataset_size", len(retrieval.images))
        mlflow.log_param("original_dim", retrieval.images.shape[1] if len(retrieval.images) > 0 else 0)
        
        query_idx = retrieval.find_query_index(query_image_path)
        if query_idx == -1:
            print("错误：找不到查询图像")
            return
        
        pca_result = retrieval.retrieve_similar(query_idx, use_pca=True)
        orig_result = retrieval.retrieve_similar(query_idx, use_pca=False)
        
        print(f"PCA检索 - 准确率: {pca_result['accuracy']:.2f}, 时间: {pca_result['retrieval_time']:.3f}s")
        print(f"原始特征检索 - 准确率: {orig_result['accuracy']:.2f}, 时间: {orig_result['retrieval_time']:.3f}s")
        
        mlflow.log_metrics({
            "pca_accuracy": pca_result["accuracy"],
            "pca_retrieval_time": pca_result["retrieval_time"],
            "original_accuracy": orig_result["accuracy"],
            "original_retrieval_time": orig_result["retrieval_time"],
            "speedup_ratio": orig_result["retrieval_time"] / pca_result["retrieval_time"] if pca_result["retrieval_time"] > 0 else 0
        })
        
        if retrieval.pca:
            mlflow.sklearn.log_model(retrieval.pca, "pca_model")
        if retrieval.svm_model:
            mlflow.sklearn.log_model(retrieval.svm_model, "svm_model")
        
        retrieval.show_results(pca_result)
        retrieval.show_results(orig_result)
        
        return pca_result, orig_result