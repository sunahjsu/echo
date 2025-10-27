import unittest
import os
import numpy as np
from src.retrieval_system import PCASVMImageRetrieval

class TestRetrievalSystem(unittest.TestCase):
    
    def setUp(self):
        """测试前准备：创建带差异的测试数据集"""
        self.dataset_path = "sample_data"
        os.makedirs(self.dataset_path, exist_ok=True)
        self.retrieval = PCASVMImageRetrieval(self.dataset_path, n_components=10)
    
    def test_load_dataset(self):
        """测试数据集加载"""
        self._create_test_images()  # 生成15张带差异的图片
        self.retrieval.load_dataset()
        self.assertEqual(len(self.retrieval.images), 15)  # 验证总数量
        self.assertEqual(len(self.retrieval.images), len(self.retrieval.labels))
    
    def test_pca_transform(self):
        """测试PCA降维（确保无nan）"""
        self._create_test_images()
        self.retrieval.load_dataset().apply_pca()
        self.assertIsNotNone(self.retrieval.images_pca)
        self.assertEqual(self.retrieval.images_pca.shape[1], 10)
        # 检查是否有nan值
        self.assertFalse(np.isnan(self.retrieval.images_pca).any())
    
    def test_retrieve_similar(self):
        """测试相似图像检索（动态调整预期数量）"""
        self._create_test_images()
        self.retrieval.load_dataset().apply_pca()
        
        # 总图片数为15，减去查询自身，最多返回14个结果，取前10个
        result = self.retrieval.retrieve_similar(0, use_pca=True)
        
        self.assertIn('accuracy', result)
        self.assertIn('results', result)
        self.assertEqual(len(result['results']), 10)  # 现在可满足10个结果
        self.assertGreaterEqual(result['accuracy'], 0.0)
        self.assertLessEqual(result['accuracy'], 1.0)
        # 检查返回结果中是否包含自身（应排除）
        result_indices = [idx for idx, _ in result['results']]
        self.assertNotIn(0, result_indices)
    
    def _create_test_images(self):
        """创建带差异的测试图片（3类×5张=15张）"""
        for class_id, class_name in enumerate(['class1', 'class2', 'class3']):
            class_path = os.path.join(self.dataset_path, class_name)
            os.makedirs(class_path, exist_ok=True)
            
            for i in range(5):  # 每个类别5张图
                img_path = os.path.join(class_path, f"img_{i}.png")
                if not os.path.exists(img_path):
                    import cv2
                    # 创建128x128的图像，不同类别添加不同特征（避免全黑）
                    img = np.zeros((128, 128, 3), dtype=np.uint8)
                    
                    # 为不同类别添加独特色块（确保数据方差非0）
                    if class_id == 0:  # class1: 红色块+随机噪点
                        img[20:50, 20:50] = [255, 0, 0]  # 红色
                        img = self._add_noise(img, intensity=10)
                    elif class_id == 1:  # class2: 绿色块+水平线条
                        img[20:50, 20:50] = [0, 255, 0]  # 绿色
                        img[70:80, :] = [100, 100, 100]  # 灰色水平线
                    else:  # class3: 蓝色块+垂直线条
                        img[20:50, 20:50] = [0, 0, 255]  # 蓝色
                        img[:, 70:80] = [100, 100, 100]  # 灰色垂直线
                    
                    cv2.imwrite(img_path, img)
    
    def _add_noise(self, img, intensity=5):
        """为图像添加随机噪点，增强数据差异"""
        noise = np.random.randint(-intensity, intensity+1, img.shape, dtype=np.int8)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return img
    
    def tearDown(self):
        """测试后清理临时文件"""
        import shutil
        if os.path.exists(self.dataset_path):
            shutil.rmtree(self.dataset_path)

if __name__ == '__main__':
    unittest.main()