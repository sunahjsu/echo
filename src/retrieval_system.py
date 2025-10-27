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
        self.pca = None
        self.scaler = StandardScaler()
        self.svm_model = None
        
    def load_dataset(self):
        """加载数据集并提取特征"""
        print("正在加载数据集...")
        self.images = []
        self.labels = []
        self.filenames = []
        
        for class_name in os.listdir(self.dataset_path):
            class_path = os.path.join(self.dataset_path, class_name)
            if os.path.isdir(class_path):
                for img_file in os.listdir(class_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(class_path, img_file)
                        # 读取图像并转换为灰度图
                        img = cv2.imread(img_path)
                        if img is not None:
                            # 调整图像大小以保持一致性
                            img = cv2.resize(img, (128, 128))
                            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            # 展平为向量
                            flattened = gray_img.flatten()
                            self.images.append(flattened)
                            self.labels.append(class_name)
                            self.filenames.append(img_path)
        
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        print(f"数据集加载完成，共 {len(self.images)} 张图像")
        return self
    
    def apply_pca(self):
        """应用PCA降维"""
        print("应用PCA降维...")
        # 标准化数据
        images_scaled = self.scaler.fit_transform(self.images)
        
        # 应用PCA
        self.pca = PCA(n_components=self.n_components)
        self.images_pca = self.pca.fit_transform(images_scaled)
        print(f"降维完成: {self.images.shape[1]} -> {self.n_components} 维")
        print(f"解释方差比: {self.pca.explained_variance_ratio_.sum():.4f}")
        return self
    
    def train_svm(self):
        """训练SVM分类器"""
        print("训练SVM分类器...")
        self.svm_model = SVC(kernel='rbf', probability=True, random_state=42)
        self.svm_model.fit(self.images_pca, self.labels)
        print("SVM训练完成")
        return self
    
    def find_query_index(self, query_image_path: str) -> int:
        """查找查询图像在数据集中的索引"""
        # 标准化路径比较
        query_abs_path = os.path.abspath(query_image_path)
        for i, filename in enumerate(self.filenames):
            if os.path.abspath(filename) == query_abs_path:
                return int(i)  # 确保返回Python int
        return -1
    
    def retrieve_similar(self, query_idx: int, use_pca: bool = True) -> Dict:
        """检索相似图像"""
        start_time = time.time()
        
        if use_pca:
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
        
        # 获取前10个最相似的图像（排除自己）
        sorted_indices = np.argsort(distances)
        # 跳过第一个（自己）
        top_indices = sorted_indices[1:11]
        top_distances = distances[top_indices]
        
        # 计算准确率
        query_label = self.labels[query_idx]
        correct_count = np.sum(self.labels[top_indices] == query_label)
        accuracy = correct_count / 10.0
        
        end_time = time.time()
        retrieval_time = end_time - start_time
        
        # 确保使用Python原生类型以便JSON序列化
        return {
            "method": method,
            "query_index": int(query_idx),  # 转换为Python int
            "results": [(int(idx), float(dist)) for idx, dist in zip(top_indices, top_distances)],  # 转换为Python类型
            "accuracy": float(accuracy),  # 转换为Python float
            "retrieval_time": float(retrieval_time),  # 转换为Python float
            "feature_dim": int(features.shape[1])  # 转换为Python int
        }
    
    def show_results(self, result: Dict):
        """可视化显示检索结果"""
        query_idx = result["query_index"]
        
        plt.figure(figsize=(15, 5))
        plt.suptitle(f"{result['method']} - 准确率: {result['accuracy']:.2f} - 时间: {result['retrieval_time']:.3f}s")
        
        # 显示查询图像
        plt.subplot(1, 11, 1)
        img = cv2.imread(self.filenames[query_idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.title("查询图像")
        plt.axis('off')
        
        # 显示检索结果
        for i, (idx, dist) in enumerate(result["results"]):
            plt.subplot(1, 11, i + 2)
            img = cv2.imread(self.filenames[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
        # 初始化系统
        retrieval = PCASVMImageRetrieval(dataset_path, n_components=50)
        
        # 加载数据并处理
        retrieval.load_dataset().apply_pca().train_svm()
        
        # 记录参数
        mlflow.log_param("n_components", 50)
        mlflow.log_param("dataset_size", len(retrieval.images))
        mlflow.log_param("original_dim", retrieval.images.shape[1])
        
        # 查找查询图像
        query_idx = retrieval.find_query_index(query_image_path)
        
        if query_idx == -1:
            print("错误：找不到查询图像")
            return
        
        # PCA检索
        pca_result = retrieval.retrieve_similar(query_idx, use_pca=True)
        print(f"PCA检索 - 准确率: {pca_result['accuracy']:.2f}, 时间: {pca_result['retrieval_time']:.3f}s")
        
        # 原始特征检索
        orig_result = retrieval.retrieve_similar(query_idx, use_pca=False)
        print(f"原始特征检索 - 准确率: {orig_result['accuracy']:.2f}, 时间: {orig_result['retrieval_time']:.3f}s")
        
        # 记录指标
        mlflow.log_metrics({
            "pca_accuracy": float(pca_result["accuracy"]),
            "pca_retrieval_time": float(pca_result["retrieval_time"]),
            "original_accuracy": float(orig_result["accuracy"]),
            "original_retrieval_time": float(orig_result["retrieval_time"]),
            "speedup_ratio": float(orig_result["retrieval_time"] / pca_result["retrieval_time"])
        })
        
        # 记录模型
        mlflow.sklearn.log_model(retrieval.pca, "pca_model")
        mlflow.sklearn.log_model(retrieval.svm_model, "svm_model")
        
        # 显示结果
        retrieval.show_results(pca_result)
        retrieval.show_results(orig_result)
        
        return pca_result, orig_result