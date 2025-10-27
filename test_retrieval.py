import unittest
import os
import numpy as np
from src.retrieval_system import PCASVMImageRetrieval

class TestRetrievalSystem(unittest.TestCase):
    
    def setUp(self):
        """测试前准备"""
        # 修改为使用样本数据目录
        self.dataset_path = "sample_data"
        # 确保测试目录存在
        os.makedirs(self.dataset_path, exist_ok=True)
        self.retrieval = PCASVMImageRetrieval(self.dataset_path, n_components=10)
    
    def test_load_dataset(self):
        """测试数据集加载"""
        # 先创建一些测试图片
        self._create_test_images()
        self.retrieval.load_dataset()
        self.assertGreater(len(self.retrieval.images), 0)
        self.assertEqual(len(self.retrieval.images), len(self.retrieval.labels))
    
    def test_pca_transform(self):
        """测试PCA降维"""
        self._create_test_images()
        self.retrieval.load_dataset().apply_pca()
        self.assertEqual(self.retrieval.images_pca.shape[1], 10)
    
    def test_retrieve_similar(self):
        """测试相似图像检索"""
        self._create_test_images()
        self.retrieval.load_dataset().apply_pca()
        result = self.retrieval.retrieve_similar(0, use_pca=True)
        
        self.assertIn('accuracy', result)
        self.assertIn('results', result)
        self.assertEqual(len(result['results']), 10)
        self.assertGreaterEqual(result['accuracy'], 0.0)
        self.assertLessEqual(result['accuracy'], 1.0)
    
    def _create_test_images(self):
        """创建临时测试图片"""
        # 创建两个测试类别
        for class_name in ['class1', 'class2']:
            class_path = os.path.join(self.dataset_path, class_name)
            os.makedirs(class_path, exist_ok=True)
            # 每个类别创建5张图片
            for i in range(5):
                img_path = os.path.join(class_path, f"img_{i}.png")
                # 创建一个简单的128x128图像并保存
                if not os.path.exists(img_path):
                    import cv2
                    img = np.zeros((128, 128, 3), dtype=np.uint8)
                    cv2.imwrite(img_path, img)
    
    def tearDown(self):
        """测试后清理临时文件"""
        import shutil
        if os.path.exists(self.dataset_path):
            shutil.rmtree(self.dataset_path)

if __name__ == '__main__':
    unittest.main()