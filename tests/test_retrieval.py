import unittest
import os
import numpy as np
from src.retrieval_system import PCASVMImageRetrieval

class TestRetrievalSystem(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """所有测试前执行一次，创建测试数据目录"""
        cls.test_data_path = "tests/test_data"
        os.makedirs(cls.test_data_path, exist_ok=True)
        
        # 创建少量测试图像文件（模拟数据集）
        for i in range(5):
            img_path = os.path.join(cls.test_data_path, f"test_{i}.png")
            with open(img_path, 'w') as f:
                f.write("mock image data")

    @classmethod
    def tearDownClass(cls):
        """所有测试后执行一次，清理测试数据"""
        for file in os.listdir(cls.test_data_path):
            file_path = os.path.join(cls.test_data_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        os.rmdir(cls.test_data_path)

    def setUp(self):
        """每个测试前初始化检索系统"""
        self.retrieval = PCASVMImageRetrieval(
            self.test_data_path, 
            n_components=10
        )

    def test_initialization(self):
        """测试初始化参数正确性"""
        self.assertEqual(self.retrieval.dataset_path, self.test_data_path)
        self.assertEqual(self.retrieval.n_components, 10)
        self.assertIsNone(self.retrieval.images)
        self.assertIsNone(self.retrieval.labels)
        self.assertIsNone(self.retrieval.pca)

    def test_load_dataset_success(self):
        """测试成功加载数据集"""
        loader = self.retrieval.load_dataset()
        self.assertIsInstance(loader, PCASVMImageRetrieval)  # 检查链式调用返回值
        self.assertIsNotNone(self.retrieval.images)
        self.assertIsNotNone(self.retrieval.labels)
        self.assertEqual(len(self.retrieval.images), 5)
        self.assertEqual(len(self.retrieval.labels), 5)
        self.assertIsInstance(self.retrieval.images, np.ndarray)
        self.assertIsInstance(self.retrieval.labels, np.ndarray)

    def test_load_dataset_invalid_path(self):
        """测试加载不存在的数据集路径"""
        invalid_retrieval = PCASVMImageRetrieval("invalid_path", n_components=10)
        with self.assertRaises(FileNotFoundError):
            invalid_retrieval.load_dataset()

    def test_apply_pca_before_load(self):
        """测试未加载数据集时调用PCA的异常"""
        with self.assertRaises(RuntimeError) as ctx:
            self.retrieval.apply_pca()
        self.assertIn("Dataset not loaded", str(ctx.exception))

    def test_pca_transform_shape(self):
        """测试PCA降维后的形状正确性"""
        self.retrieval.load_dataset().apply_pca()
        self.assertIsNotNone(self.retrieval.images_pca)
        self.assertEqual(self.retrieval.images_pca.shape[0], 5)  # 样本数
        self.assertEqual(self.retrieval.images_pca.shape[1], 10)  # 降维后的维度
        self.assertIsInstance(self.retrieval.images_pca, np.ndarray)

    def test_pca_explained_variance(self):
        """测试PCA解释方差比例合理性"""
        self.retrieval.load_dataset().apply_pca()
        self.assertIsNotNone(self.retrieval.pca)
        explained = np.sum(self.retrieval.pca.explained_variance_ratio_)
        self.assertGreater(explained, 0.0)
        self.assertLessEqual(explained, 1.0)

    def test_retrieve_before_pca(self):
        """测试未执行PCA时使用PCA检索的异常"""
        self.retrieval.load_dataset()
        with self.assertRaises(RuntimeError) as ctx:
            self.retrieval.retrieve_similar(0, use_pca=True)
        self.assertIn("PCA not applied", str(ctx.exception))

    def test_retrieve_invalid_index(self):
        """测试使用无效索引检索的异常"""
        self.retrieval.load_dataset().apply_pca()
        with self.assertRaises(IndexError):
            self.retrieval.retrieve_similar(100, use_pca=True)  # 索引超出范围

    def test_retrieve_similar_with_pca(self):
        """测试使用PCA的相似图像检索功能"""
        self.retrieval.load_dataset().apply_pca()
        result = self.retrieval.retrieve_similar(0, use_pca=True)
        
        # 验证结果结构
        self.assertIsInstance(result, dict)
        self.assertIn('accuracy', result)
        self.assertIn('results', result)
        self.assertIn('query_index', result)
        self.assertEqual(result['query_index'], 0)
        
        # 验证结果内容
        self.assertIsInstance(result['results'], list)
        self.assertEqual(len(result['results']), 10)  # 假设返回10个结果
        self.assertIsInstance(result['accuracy'], float)
        self.assertGreaterEqual(result['accuracy'], 0.0)
        self.assertLessEqual(result['accuracy'], 1.0)

    def test_retrieve_similar_without_pca(self):
        """测试不使用PCA的相似图像检索功能"""
        self.retrieval.load_dataset()
        result = self.retrieval.retrieve_similar(0, use_pca=False)
        
        self.assertIsInstance(result, dict)
        self.assertIn('accuracy', result)
        self.assertIn('results', result)
        self.assertEqual(len(result['results']), 10)

if __name__ == '__main__':
    unittest.main(verbosity=2)