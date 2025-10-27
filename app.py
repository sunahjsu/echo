from flask import Flask, render_template, request, jsonify
import os
import sys
import numpy as np
import json
import dagshub  # 新增：导入DagsHub
# 在 app.py 开头添加
import mlflow
mlflow.set_tracking_uri("./mlruns")
# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from retrieval_system import PCASVMImageRetrieval
import mlflow

# 自定义JSON编码器处理NumPy类型
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

app = Flask(__name__)
app.json_encoder = NumpyEncoder

# 关键修改1：初始化DagsHub（与retrieval_system.py保持一致）
dagshub.init(
    repo_owner="sunahjsu",
    repo_name="echo",
    mlflow=True
)

# 关键修改2：统一MLflow实验名称（与retrieval_system.py保持一致）
mlflow.set_experiment("PCA_Image_Retrieval")
# 移除本地MLflow服务地址，使用DagsHub自动配置的跟踪URI
# mlflow.set_tracking_uri("http://127.0.0.1:5001")  # 注释掉此行

# 检查MLflow连接（修改为检查DagsHub配置的URI）
try:
    mlflow.search_experiments(max_results=1)
    print("MLflow（DagsHub）连接成功")
except Exception as e:
    print(f"MLflow（DagsHub）连接警告: {e}（请确保已执行dagshub login）")

# 全局变量
retrieval_system = None
DATASET_PATH = os.getenv('DATASET_PATH', './Corel-1000')
first_request = True  # 标记首次请求

@app.before_request
def initialize_system():
    """初始化检索系统（仅首次请求执行）"""
    global retrieval_system, first_request
    if first_request:
        try:
            retrieval_system = PCASVMImageRetrieval(DATASET_PATH, n_components=50)
            retrieval_system.load_dataset().apply_pca().train_svm()
            print("系统初始化完成")
        except Exception as e:
            print(f"系统初始化失败: {e}")
        first_request = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/retrieve', methods=['POST'])
def retrieve_similar():
    """检索相似图像API"""
    if retrieval_system is None:
        return jsonify({'error': '系统未初始化完成'}), 503
    
    try:
        data = request.json
        query_image = data.get('query_image')
        
        if not query_image:
            return jsonify({'error': '未提供查询图像'}), 400
        
        query_idx = retrieval_system.find_query_index(query_image)
        if query_idx == -1:
            return jsonify({'error': '查询图像不在数据集中'}), 404
        
        # 执行检索
        pca_result = retrieval_system.retrieve_similar(query_idx, use_pca=True)
        orig_result = retrieval_system.retrieve_similar(query_idx, use_pca=False)
        
        # 记录到MLflow（会自动同步到DagsHub）
        with mlflow.start_run():
            mlflow.log_params({
                'n_components': 50,
                'query_image': query_image,
                'dataset_path': DATASET_PATH  # 补充记录数据集路径
            })
            mlflow.log_metrics({
                'pca_accuracy': float(pca_result['accuracy']),
                'original_accuracy': float(orig_result['accuracy']),
                'pca_retrieval_time': float(pca_result['retrieval_time']),  # 补充时间指标
                'original_retrieval_time': float(orig_result['retrieval_time'])
            })
            # 关键：记录模型（与retrieval_system.py保持一致）
            mlflow.sklearn.log_model(
                sk_model=retrieval_system.pca,
                artifact_path="pca_model",
                registered_model_name="pca-image-retrieval"
            )
            mlflow.sklearn.log_model(
                sk_model=retrieval_system.svm_model,
                artifact_path="svm_model",
                registered_model_name="svm-image-classifier"
            )
        
        return jsonify({
            'pca_result': pca_result,
            'original_result': orig_result,
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/images')
def list_images():
    """获取数据集图像列表"""
    if retrieval_system is None:
        return jsonify({'error': '系统未初始化完成'}), 503
    
    try:
        images = []
        for i, filename in enumerate(retrieval_system.filenames):
            images.append({
                'path': filename,
                'label': str(retrieval_system.labels[i]),
                'id': int(i)
            })
        return jsonify({'images': images})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if not os.path.exists('templates'):
        os.makedirs('templates')
    app.run(host='0.0.0.0', port=5000, debug=True)