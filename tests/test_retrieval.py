import unittest
import os
import cv2
import numpy as np
from src.retrieval_system import PCASVMImageRetrieval

class TestRetrievalSystem(unittest.TestCase):
    
    def setUp(self):
        """测试前准备：创建测试数据集"""
        self.dataset_path = "tests/test_data"
        os.makedirs(self.dataset_path, exist_ok=True)
        
        # 创建2个类别，每个类别2张图像（空白图像模拟）
        for class_id in ['class1', 'class2']:
            class_dir = os.path.join(self.dataset_path, class_id)
            os.makedirs(class_dir, exist_ok=True)
            for i in range(2):
                img = np.zeros((128, 128, 3), dtype=np.uint8)  # 空白图像
                img_path = os.path.join(class_dir, f"img_{i}.jpg")
                cv2.imwrite(img_path, img)
        
        self.retrieval = PCASVMImageRetrieval(self.dataset_path, n_components=10)
    
    def tearDown(self):
        """测试后清理（可选，保留测试数据便于调试）"""
        pass
    
    def test_load_dataset(self):
        """测试数据集加载"""
        self.retrieval.load_dataset()
        self.assertEqual(len(self.retrieval.images), 4)  # 2类×2张
        self.assertEqual(len(self.retrieval.images), len(self.retrieval.labels))
        self.assertTrue(all(isinstance(lab, str) for lab in self.retrieval.labels))
    
    def test_pca_transform(self):
        """测试PCA降维"""
        self.retrieval.load_dataset().apply_pca()
        self.assertEqual(self.retrieval.images_pca.shape[1], 10)  # 匹配n_components
        self.assertEqual(self.retrieval.images_pca.shape[0], 4)  # 样本数
        
    def test_retrieve_similar(self):
        """测试相似图像检索"""
        self.retrieval.load_dataset().apply_pca()
        result = self.retrieval.retrieve_similar(0, use_pca=True)
        
        self.assertIn('accuracy', result)
        self.assertIn('results', result)
        self.assertEqual(len(result['results']), 10)  # 前10个结果
        self.assertIsInstance(result['accuracy'], float)
        self.assertGreaterEqual(result['accuracy'], 0.0)
        self.assertLessEqual(result['accuracy'], 1.0)
    
    def test_empty_dataset(self):
        """测试空数据集处理"""
        empty_path = "tests/empty_data"
        os.makedirs(empty_path, exist_ok=True)
        retrieval = PCASVMImageRetrieval(empty_path)
        with self.assertRaises(ValueError):
            retrieval.load_dataset()

if __name__ == '__main__':
    unittest.main()