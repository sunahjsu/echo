import unittest
import os
import numpy as np
from src.retrieval_system import PCASVMImageRetrieval

class TestRetrievalSystem(unittest.TestCase):
    
    def setUp(self):
        """测试前准备"""
        self.dataset_path = "tests/test_data"
        self.retrieval = PCASVMImageRetrieval(self.dataset_path, n_components=10)
    
    def test_load_dataset(self):
        """测试数据集加载"""
        self.retrieval.load_dataset()
        self.assertGreater(len(self.retrieval.images), 0)
        self.assertEqual(len(self.retrieval.images), len(self.retrieval.labels))
    
    def test_pca_transform(self):
        """测试PCA降维"""
        self.retrieval.load_dataset().apply_pca()
        self.assertEqual(self.retrieval.images_pca.shape[1], 10)
    
    def test_retrieve_similar(self):
        """测试相似图像检索"""
        self.retrieval.load_dataset().apply_pca()
        result = self.retrieval.retrieve_similar(0, use_pca=True)
        
        self.assertIn('accuracy', result)
        self.assertIn('results', result)
        self.assertEqual(len(result['results']), 10)
        self.assertGreaterEqual(result['accuracy'], 0.0)
        self.assertLessEqual(result['accuracy'], 1.0)

if __name__ == '__main__':
    unittest.main()