import pytest
import json
import os
from app import app  # 从你的应用文件导入Flask实例

# 测试客户端 fixture，所有测试用例可复用
@pytest.fixture
def client():
    # 启用测试模式，禁用异常捕获，便于调试
    app.config['TESTING'] = True
    app.config['DEBUG'] = False  # 测试时关闭DEBUG模式，避免意外行为
    
    # 创建测试客户端
    with app.test_client() as client:
        yield client


def test_index_route(client):
    """测试首页访问（根路径）"""
    response = client.get('/')
    # 验证状态码为200（成功）
    assert response.status_code == 200
    # 验证返回内容包含预期信息（根据你的首页实际内容调整）
    assert "图像检索系统" in response.data  # 假设首页包含该文本


def test_health_check(client):
    """测试健康检查接口（/health）"""
    response = client.get('/health')
    assert response.status_code == 200
    
    # 解析JSON响应
    data = json.loads(response.data)
    # 验证响应结构
    assert 'status' in data
    assert 'dataset_loaded' in data
    # 验证状态值合法性
    assert data['status'] in ['healthy', 'initializing', 'error']


def test_model_info(client):
    """测试模型信息接口（/model-info）"""
    response = client.get('/model-info')
    
    # 分两种情况：系统初始化完成/未完成
    if response.status_code == 503:
        # 未初始化时返回"服务不可用"
        data = json.loads(response.data)
        assert 'error' in data
        assert '未初始化' in data['error']
    else:
        # 初始化完成时返回模型信息
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'pca_components' in data
        assert 'dataset_size' in data
        assert 'classes' in data
        assert isinstance(data['pca_components'], int)
        assert data['dataset_size'] > 0  # 数据集非空


def test_list_images(client):
    """测试图像列表接口（/images）"""
    response = client.get('/images')
    
    if response.status_code == 503:
        # 未初始化时跳过详细检查
        pytest.skip("系统未初始化，跳过图像列表测试")
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'images' in data
    assert isinstance(data['images'], list)
    # 验证列表非空（假设数据集已正确加载）
    assert len(data['images']) > 0
    
    # 验证单个图像信息的结构
    sample_image = data['images'][0]
    assert 'path' in sample_image
    assert 'label' in sample_image
    assert os.path.exists(sample_image['path'])  # 验证路径存在


def test_retrieve_similar_success(client):
    """测试相似图像检索接口（/retrieve）的正常情况"""
    # 先获取一个有效的图像路径（依赖/images接口）
    images_response = client.get('/images')
    if images_response.status_code != 200:
        pytest.skip("无法获取图像列表，跳过检索测试")
    
    images_data = json.loads(images_response.data)
    if not images_data['images']:
        pytest.skip("图像列表为空，跳过检索测试")
    
    # 用第一个图像作为查询条件
    test_image_path = images_data['images'][0]['path']
    response = client.post(
        '/retrieve',
        json={'query_image': test_image_path}  # 匹配接口预期的JSON参数
    )
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['success'] is True
    assert 'pca_result' in data
    assert 'original_result' in data
    # 验证返回的相似图像列表非空
    assert len(data['pca_result']) > 0
    assert len(data['original_result']) > 0


def test_retrieve_similar_invalid_path(client):
    """测试检索接口（/retrieve）的异常情况：无效图像路径"""
    response = client.post(
        '/retrieve',
        json={'query_image': 'invalid/path/to/image.jpg'}  # 不存在的路径
    )
    
    assert response.status_code == 400  # 预期返回"Bad Request"
    data = json.loads(response.data)
    assert data['success'] is False
    assert 'error' in data
    assert '不存在' in data['error']  # 假设错误信息包含该文本


def test_retrieve_missing_param(client):
    """测试检索接口（/retrieve）的异常情况：缺少参数"""
    response = client.post(
        '/retrieve',
        json={}  # 未提供query_image参数
    )
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data['success'] is False
    assert '缺少参数' in data['error']  # 假设错误信息包含该文本