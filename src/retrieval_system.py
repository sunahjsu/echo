import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import mlflow
import mlflow.sklearn
import dagshub
from typing import List, Dict
import time

class PCASVMImageRetrieval:
    def __init__(self, dataset_path: str, n_components: int = 50):
        self.dataset_path = dataset_path
        self.n_components = n_components
        self.images = []  # 原始图像特征（展平后）
        self.labels = []  # 图像类别标签
        self.filenames = []  # 图像文件路径
        self.images_pca = None  # PCA降维后的特征
        self.pca = None  # PCA模型
        self.scaler = StandardScaler()  # 标准化器
        self.svm_model = None  # SVM分类模型
        
    def load_dataset(self):
        """加载数据集并提取特征"""
        print("正在加载数据集...")
        self.images = []
        self.labels = []
        self.filenames = []
        
        # 检查数据集路径是否存在
        if not os.path.exists(self.dataset_path):
            raise ValueError(f"数据集路径不存在: {self.dataset_path}")
        
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
                            self.images.append(gray_img.flatten())
                            self.labels.append(class_name)
                            self.filenames.append(img_path)
        
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        print(f"数据集加载完成，共 {len(self.images)} 张图像")
        
        # 空数据集检查
        if len(self.images) == 0:
            raise ValueError("数据集为空，请检查路径或图像文件格式")
        return self
    
    # 以下方法（apply_pca、train_svm等）保持不变...
    def apply_pca(self):
        """应用PCA降维"""
        print("应用PCA降维...")
        images_scaled = self.scaler.fit_transform(self.images)
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
        query_abs_path = os.path.abspath(query_image_path)
        for i, filename in enumerate(self.filenames):
            if os.path.abspath(filename) == query_abs_path:
                return int(i)
        return -1
    
    def retrieve_similar(self, query_idx: int, use_pca: bool = True) -> Dict:
        """检索相似图像并返回结果"""
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
        sorted_indices = np.argsort(distances)
        top_indices = sorted_indices[1:11]  # 排除自身，取前10
        top_distances = distances[top_indices]
        
        # 计算准确率
        query_label = self.labels[query_idx]
        correct_count = np.sum(self.labels[top_indices] == query_label)
        accuracy = correct_count / 10.0
        
        return {
            "method": method,
            "query_index": int(query_idx),
            "results": [(int(idx), float(dist)) for idx, dist in zip(top_indices, top_distances)],
            "accuracy": float(accuracy),
            "retrieval_time": float(time.time() - start_time),
            "feature_dim": int(features.shape[1])
        }
    
    def show_results(self, result: Dict):
        """可视化检索结果"""
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
    """运行完整实验并记录到MLflow和DagsHub"""
    # 初始化DagsHub（与app.py保持一致）
    dagshub.init(
        repo_owner="sunahjsu",
        repo_name="echo",
        mlflow=True
    )
    
    # 配置MLflow实验（与app.py保持一致）
    mlflow.set_experiment("PCA_Image_Retrieval")
    
    with mlflow.start_run():
        # 初始化检索系统
        retrieval = PCASVMImageRetrieval(dataset_path, n_components=50)
        retrieval.load_dataset().apply_pca().train_svm()
        
        # 记录实验参数
        mlflow.log_params({
            'n_components': 50,
            'dataset_size': len(retrieval.images),
            'original_dim': retrieval.images.shape[1],
            'dataset_path': dataset_path,
            'query_image': query_image_path
        })
        mlflow.set_tag("model_type", "PCA+SVM")
        mlflow.set_tag("task", "image_retrieval")
        mlflow.set_tag("source", "script")  # 标记实验来源（与app区分）
        
        # 查找查询图像
        query_idx = retrieval.find_query_index(query_image_path)
        if query_idx == -1:
            print("错误：找不到查询图像")
            return
        
        # 执行检索并记录指标
        pca_result = retrieval.retrieve_similar(query_idx, use_pca=True)
        orig_result = retrieval.retrieve_similar(query_idx, use_pca=False)
        
        mlflow.log_metrics({
            'pca_accuracy': pca_result["accuracy"],
            'pca_retrieval_time': pca_result["retrieval_time"],
            'original_accuracy': orig_result["accuracy"],
            'original_retrieval_time': orig_result["retrieval_time"],
            'speedup_ratio': orig_result["retrieval_time"] / pca_result["retrieval_time"]
        })
        
        # 记录模型（与app.py保持一致）
        mlflow.sklearn.log_model(
            sk_model=retrieval.pca,
            artifact_path="pca_model",
            registered_model_name="pca-image-retrieval"
        )
        mlflow.sklearn.log_model(
            sk_model=retrieval.svm_model,
            artifact_path="svm_model",
            registered_model_name="svm-image-classifier"
        )
        
        # 可视化结果
        retrieval.show_results(pca_result)
        retrieval.show_results(orig_result)
        
        return pca_result, orig_result

if __name__ == "__main__":
    # 路径适配：根据脚本执行位置动态调整（避免路径错误）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    DATASET_PATH = os.path.join(script_dir, "../Corel-1000")  # 从src目录指向数据集
    QUERY_IMAGE = os.path.join(DATASET_PATH, "beach/101.jpg")  # 确保图像存在
    
    # 运行实验
    run_experiment(DATASET_PATH, QUERY_IMAGE)